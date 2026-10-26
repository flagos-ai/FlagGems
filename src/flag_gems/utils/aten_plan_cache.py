# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in structural routing and launch-plan cache for generic ATen ops.

The cache is disabled by default. Call :func:`enable` explicitly, or set
``FLAGGEMS_ATEN_PLAN_CACHE=1`` before importing :mod:`flag_gems`, to install
the reversible wrappers.

Cached values may own compiled Triton kernels and immutable scalar metadata.
They never own tensors, data pointers, accelerator streams/events, or graphs.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cache as functools_cache
from typing import Any

import torch
import triton

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.shape_utils import (
    heuristics_for_num_warps,
    heuristics_for_tile_size,
    stride_order,
)
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SIZE = 128
_DEFAULT_EXCLUDE = ("*arange_func*",)
_RUNTIME_VENDOR = runtime_device.vendor_name
_RUNTIME_DEVICE_NAME = runtime_device.name
_CACHE_SIZE = _DEFAULT_CACHE_SIZE
_ENABLED = False
_INCLUDE = None
_EXCLUDE = _DEFAULT_EXCLUDE
_POLICY_EPOCH = 0
_ALL_CACHES = weakref.WeakSet()
_INSTALLED = False
_ORIGINAL_LIBENTRY_RUN = None
_ORIGINAL_POINTWISE_CALL = None
_ORIGINAL_POINTWISE_INSTANTIATE = None
_MISSING = object()
_UNSET = object()
_ORIGINAL_LIBENTRY_BUILD_PLAN = _MISSING
_ORIGINAL_LIBENTRY_RUN_PLAN = _MISSING
_ORIGINAL_POINTWISE_CACHE_INFO = _MISSING
_ORIGINAL_POINTWISE_CLEAR_CACHE = _MISSING


def _split_patterns(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    return tuple(pattern.strip() for pattern in value if pattern.strip())


def _operator_name(operator):
    return ".".join(str(part) for part in operator if part is not None)


def _operator_enabled(operator):
    name = _operator_name(operator)
    if _INCLUDE is not None and not any(
        fnmatchcase(name, pattern) for pattern in _INCLUDE
    ):
        return False
    return not any(fnmatchcase(name, pattern) for pattern in _EXCLUDE)


@functools_cache
def _device_capability(vendor, device_name, current_device):
    del vendor, device_name
    if current_device == "cpu":
        return (0, 0)
    try:
        return tuple(torch_device_fn.get_device_capability(current_device) or (0, 0))
    except (AttributeError, RuntimeError, TypeError):
        return (0, 0)


@functools_cache
def _device_context(vendor, device_name, device_type, device_index):
    """Build an immutable context once for each process-local device.

    Tensor signatures already distinguish device type and index. Reusing this
    tuple avoids repeated vendor lookup, capability lookup, and tuple building
    on every warm routing-plan hit while preserving per-device cache isolation.
    """

    capability_device = "cpu" if device_type == "cpu" else device_index
    return (
        vendor,
        device_name,
        device_type,
        device_index,
        _device_capability(vendor, device_name, capability_device),
    )


@functools_cache
def _tensor_device_context(value_device):
    """Resolve a ``torch.device`` only on the first use of that device."""

    device_type = getattr(value_device, "type", None)
    if device_type is None:
        device_type = str(value_device)
    device_index = getattr(value_device, "index", None)
    if device_type == "cpu":
        device_index = "cpu"
    if device_index is None:
        return _current_device_context()
    return _device_context(
        _RUNTIME_VENDOR,
        _RUNTIME_DEVICE_NAME,
        device_type,
        device_index,
    )


def _current_device_context():
    """Return a hashable vendor/device/architecture identity."""

    try:
        current_device = torch_device_fn.current_device()
    except (AttributeError, RuntimeError, TypeError):
        current_device = "unknown"
    vendor = _RUNTIME_VENDOR
    device_name = _RUNTIME_DEVICE_NAME
    device_type = "cpu" if current_device == "cpu" else device_name
    return _device_context(
        vendor,
        device_name,
        device_type,
        current_device,
    )


def _argument_device_context(args, kwargs=None):
    values = args if not kwargs else (*args, *kwargs.values())
    for value in values:
        value_device = getattr(value, "device", None)
        if value_device is None:
            continue
        return _tensor_device_context(value_device)
    return _current_device_context()


@dataclass(frozen=True)
class CacheInfo:
    last_hits: int
    lru_hits: int
    misses: int
    evictions: int
    bypasses: int
    size: int
    max_size: int

    @property
    def hits(self) -> int:
        return self.last_hits + self.lru_hits

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class _TwoLevelCache:
    """Device-separated monomorphic last-value cache backed by a bounded LRU."""

    def __init__(self, operator):
        self.operator = operator
        self.last_signature = {}
        self.last_plan = {}
        self.lru = {}
        self.last_hits = 0
        self.lru_hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypasses = 0
        self.pid = os.getpid()
        self.policy_epoch = -1
        self.policy_enabled = False
        _ALL_CACHES.add(self)

    def lookup(self, signature, device_context=None):
        if self.pid != os.getpid():
            self.clear(reset_stats=False)
            self.pid = os.getpid()
        if self.policy_epoch != _POLICY_EPOCH:
            self.policy_enabled = _operator_enabled(self.operator)
            self.policy_epoch = _POLICY_EPOCH
        if not _ENABLED or _CACHE_SIZE == 0 or not self.policy_enabled:
            self.bypasses += 1
            return None, None
        if device_context is None:
            device_context = _current_device_context()
        full_signature = (
            self.operator,
            device_context,
            signature,
        )
        if self.last_signature.get(device_context) == full_signature:
            self.last_hits += 1
            return full_signature, self.last_plan[device_context]
        device_lru = self.lru.setdefault(device_context, OrderedDict())
        plan = device_lru.get(full_signature)
        if plan is not None:
            device_lru.move_to_end(full_signature)
            self.lru_hits += 1
            self.last_signature[device_context] = full_signature
            self.last_plan[device_context] = plan
            return full_signature, plan
        self.misses += 1
        return full_signature, None

    def insert(self, signature, plan):
        if signature is None or plan is None:
            return
        device_context = signature[1]
        device_lru = self.lru.setdefault(device_context, OrderedDict())
        device_lru[signature] = plan
        device_lru.move_to_end(signature)
        while len(device_lru) > _CACHE_SIZE:
            device_lru.popitem(last=False)
            self.evictions += 1
        self.last_signature[device_context] = signature
        self.last_plan[device_context] = plan

    def clear(self, reset_stats=True):
        self.last_signature.clear()
        self.last_plan.clear()
        self.lru.clear()
        if reset_stats:
            self.reset_stats()

    def reset_stats(self):
        self.last_hits = 0
        self.lru_hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypasses = 0

    def info(self):
        return CacheInfo(
            last_hits=self.last_hits,
            lru_hits=self.lru_hits,
            misses=self.misses,
            evictions=self.evictions,
            bypasses=self.bypasses,
            size=sum(len(cache) for cache in self.lru.values()),
            max_size=_CACHE_SIZE,
        )


def _canonical_scalar(value):
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    if isinstance(value, torch.device):
        return ("device", value.type, value.index)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_canonical_scalar(v) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (_canonical_scalar(key), _canonical_scalar(item))
                    for key, item in value.items()
                )
            ),
        )
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _tensor_signature(value, alias_index=None):
    device = getattr(value, "device", None)
    shape = tuple(getattr(value, "shape", ()))
    stride_fn = getattr(value, "stride", None)
    strides = tuple(stride_fn()) if callable(stride_fn) else ()
    alignment = None
    try:
        alignment = value.data_ptr() % 16 == 0
    except (AttributeError, RuntimeError, TypeError):
        alignment = None
    return (
        "tensor",
        type(value).__name__,
        getattr(device, "type", str(device)),
        getattr(device, "index", None),
        str(getattr(value, "dtype", None)),
        shape,
        strides,
        alignment,
        alias_index,
    )


def _argument_signatures(values):
    seen = {}
    result = []
    for index, value in enumerate(values):
        if hasattr(value, "data_ptr"):
            object_id = id(value)
            alias_index = seen.get(object_id)
            seen.setdefault(object_id, index)
            result.append(_tensor_signature(value, alias_index))
        else:
            result.append(_canonical_scalar(value))
    return tuple(result)


def _grid_signature(grid):
    if not callable(grid):
        return ("fixed", tuple(grid) if isinstance(grid, (tuple, list)) else grid)
    code = getattr(grid, "__code__", None)
    if code is None:
        # Callable objects can carry mutable state that is absent from the key.
        return None
    closure = []
    for cell in getattr(grid, "__closure__", ()) or ():
        value = cell.cell_contents
        if hasattr(value, "data_ptr"):
            return None
        closure.append(_canonical_scalar(value))
    globals_signature = []
    namespace = getattr(grid, "__globals__", {})
    for name in code.co_names:
        if name not in namespace:
            continue
        value = namespace[name]
        if hasattr(value, "data_ptr"):
            return None
        if inspect.ismodule(value) or inspect.isroutine(value):
            globals_signature.append(
                (
                    name,
                    "callable",
                    getattr(value, "__module__", None),
                    getattr(value, "__qualname__", getattr(value, "__name__", None)),
                )
            )
        else:
            globals_signature.append((name, _canonical_scalar(value)))
    return (
        "callable",
        getattr(grid, "__module__", None),
        code.co_filename,
        code.co_firstlineno,
        tuple(closure),
        _canonical_scalar(getattr(grid, "__defaults__", None)),
        _canonical_scalar(getattr(grid, "__kwdefaults__", None)),
        tuple(globals_signature),
    )


@dataclass(frozen=True)
class LibEntryLaunchPlan:
    kernel: Any
    grid: tuple[int, int, int]
    argument_sources: tuple[tuple[str, Any], ...]
    device_context: tuple[Any, ...]
    tuning_epoch: int
    constexprs: tuple[tuple[str, Any], ...]
    num_warps: int | None
    num_stages: int | None
    num_ctas: int | None


def _launch_constant(value):
    if hasattr(value, "data_ptr"):
        raise TypeError("launch plans must not retain tensor-like arguments")
    if isinstance(value, (tuple, list)):
        return tuple(_launch_constant(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (_launch_constant(key), _launch_constant(item))
                for key, item in value.items()
            )
        )
    if value is None or isinstance(
        value, (bool, int, float, complex, str, torch.dtype, torch.device)
    ):
        return value
    raise TypeError("launch plans must not retain opaque argument objects")


def _try_build_libentry_plan(self, *args, **kwargs):
    try:
        return _build_libentry_plan(self, *args, **kwargs)
    except Exception:
        logger.debug(
            "FlagGems could not build a LibEntry launch plan for %s",
            self.jit_function.__name__,
            exc_info=True,
        )
        return None


def _libentry_lookup_compiled(self, args, kwargs):
    libentry_module = importlib.import_module("flag_gems.utils.libentry")

    spec_args = []
    dns_args = []
    const_args = []
    kernel_args = OrderedDict()
    param_names = list(self.signature.parameters.keys())
    for index, arg in enumerate(args):
        hashable = arg
        if arg.__class__.__name__ == "TensorDescriptor":
            hashable = (
                "TensorDescriptor",
                tuple(arg.shape) if hasattr(arg, "shape") else None,
                tuple(arg.strides) if hasattr(arg, "strides") else None,
                tuple(arg.block_shape) if hasattr(arg, "block_shape") else None,
                arg.padding if hasattr(arg, "padding") else None,
            )
        if index in self.specialize_indices:
            kernel_args[param_names[index]] = arg
            spec_args.append(hashable)
        elif index in self.do_not_specialize_indices:
            kernel_args[param_names[index]] = arg
            dns_args.append(hashable)
        else:
            if (
                libentry_module.major_version == 3
                and 3 <= libentry_module.minor_version <= 6
            ):
                kernel_args[param_names[index]] = arg
            const_args.append(hashable)
    for parameter in self.jit_function.params[len(args) :]:
        if parameter.name in kwargs:
            value = kwargs[parameter.name]
        elif parameter.default is inspect._empty:
            continue
        else:
            value = parameter.default
        if parameter.is_constexpr:
            const_args.append(value)
            if (
                libentry_module.major_version == 3
                and 3 <= libentry_module.minor_version <= 6
            ):
                kernel_args[parameter.name] = value
        elif parameter.do_not_specialize:
            dns_args.append(value)
            kernel_args[parameter.name] = value
        else:
            spec_args.append(value)
            kernel_args[parameter.name] = value

    if self._has_flagtune_tuner:
        dtypes = libentry_module._infer_tensor_dtypes(kernel_args.values())
        const_args.append(("flagtune_dtypes",) + tuple(str(value) for value in dtypes))
    entry_key = self.key(spec_args, dns_args, const_args)
    current_device = torch_device_fn.current_device()
    cache = (
        self._cpu_cache
        if current_device == "cpu"
        else self.kernel_cache[current_device]
    )
    return cache.get(entry_key), kernel_args, param_names


def _build_libentry_plan(self, *args, **kwargs):
    libentry_module = importlib.import_module("flag_gems.utils.libentry")

    grid = kwargs["grid"]
    entry, kernel_args, param_names = _libentry_lookup_compiled(self, args, kwargs)
    if entry is None:
        return None
    kernel, constexprs, tune_values, heuristic_values, pre_hooks = entry
    if pre_hooks:
        return None
    if callable(grid):
        metadata = {**dict(zip(self.arg_names, args)), **kwargs, **constexprs}
        grid = grid(metadata)
    grid = tuple(grid) + (1, 1)
    grid = tuple(grid[:3])

    positional = {param_names[index]: ("arg", index) for index in range(len(args))}
    keyword = {}
    for name in kernel_args:
        if name not in kwargs:
            continue
        value = kernel_args[name]
        keyword[name] = (
            ("kwarg", name)
            if hasattr(value, "data_ptr")
            else ("const", _launch_constant(value))
        )
    if libentry_module.major_version == 3 and 3 <= libentry_module.minor_version <= 6:
        launch_names = tuple(self.signature.parameters.keys())
    else:
        launch_names = tuple(kernel_args.keys())
    sources = []
    for name in launch_names:
        if name in positional:
            sources.append(positional[name])
        elif name in keyword:
            sources.append(keyword[name])
        elif name in tune_values:
            sources.append(("const", _launch_constant(tune_values[name])))
        elif name in heuristic_values:
            sources.append(("const", _launch_constant(heuristic_values[name])))
        elif name in constexprs:
            sources.append(("const", _launch_constant(constexprs[name])))
        else:
            return None
    return LibEntryLaunchPlan(
        kernel=kernel,
        grid=grid,
        argument_sources=tuple(sources),
        device_context=_argument_device_context(args, kwargs),
        tuning_epoch=getattr(self, "_aten_plan_tuning_epoch", 0),
        constexprs=tuple(
            sorted(
                (name, _launch_constant(value)) for name, value in constexprs.items()
            )
        ),
        num_warps=constexprs.get("num_warps"),
        num_stages=constexprs.get("num_stages"),
        num_ctas=constexprs.get("num_ctas"),
    )


class _StaleLaunchPlan(RuntimeError):
    pass


def _refresh_flagtune(self):
    if not self._has_flagtune_tuner:
        return
    previous_epoch = getattr(self, "_aten_plan_tuning_epoch", 0)
    if self._apply_flagtune():
        if getattr(self, "_aten_plan_tuning_epoch", 0) == previous_epoch:
            self._aten_plan_tuning_epoch = previous_epoch + 1
        cache = getattr(self, "_aten_launch_plan_cache", None)
        if cache is not None:
            cache.clear(reset_stats=False)


def _run_libentry_plan(self, plan, *args, _aten_plan_device_context=None, **kwargs):
    _refresh_flagtune(self)
    device_context = (
        _argument_device_context(args, kwargs)
        if _aten_plan_device_context is None
        else _aten_plan_device_context
    )
    if plan.device_context != device_context:
        raise _StaleLaunchPlan("launch-plan device context changed")
    if plan.tuning_epoch != getattr(self, "_aten_plan_tuning_epoch", 0):
        raise _StaleLaunchPlan("FlagTune changed the selected launch configuration")
    launch_args = []
    for source, payload in plan.argument_sources:
        if source == "arg":
            launch_args.append(args[payload])
        elif source == "kwarg":
            launch_args.append(kwargs[payload])
        else:
            launch_args.append(payload)
    plan.kernel[plan.grid](*launch_args)
    return plan.kernel, dict(plan.constexprs)


def _libentry_signature(self, args, kwargs):
    grid_signature = _grid_signature(kwargs["grid"])
    if grid_signature is None:
        return None
    keyword_signature = tuple(
        sorted(
            (name, _canonical_scalar(value))
            for name, value in kwargs.items()
            if name != "grid" and not hasattr(value, "data_ptr")
        )
    )
    tensor_keywords = tuple(
        sorted(
            (name, _tensor_signature(value))
            for name, value in kwargs.items()
            if name != "grid" and hasattr(value, "data_ptr")
        )
    )
    return (
        _argument_signatures(args),
        keyword_signature,
        tensor_keywords,
        grid_signature,
    )


def _cached_libentry_run(self, *args, **kwargs):
    if not _ENABLED or _CACHE_SIZE == 0:
        return _ORIGINAL_LIBENTRY_RUN(self, *args, **kwargs)
    cache = getattr(self, "_aten_launch_plan_cache", None)
    if cache is None:
        cache = _TwoLevelCache(
            (
                "libentry",
                self.jit_function.__module__,
                self.jit_function.__name__,
                self.jit_function.cache_key,
            )
        )
        self._aten_launch_plan_cache = cache
    signature = _libentry_signature(self, args, kwargs)
    if signature is None:
        cache.bypasses += 1
        return _ORIGINAL_LIBENTRY_RUN(self, *args, **kwargs)
    device_context = _argument_device_context(args, kwargs)
    full_signature, plan = cache.lookup(signature, device_context=device_context)
    if plan is not None:
        try:
            return _run_libentry_plan(
                self,
                plan,
                *args,
                _aten_plan_device_context=device_context,
                **kwargs,
            )
        except _StaleLaunchPlan:
            plan = None
    logger.debug(
        "FlagGems LibEntry launch-plan cache miss: %s", self.jit_function.__name__
    )
    result = _ORIGINAL_LIBENTRY_RUN(self, *args, **kwargs)
    plan = _try_build_libentry_plan(self, *args, **kwargs)
    cache.insert(full_signature, plan)
    return result


@dataclass(frozen=True)
class _AllocationSource:
    kind: str
    index: int


@dataclass(frozen=True)
class PointwisePlan:
    complex_path: bool
    fast_path: bool
    task_shape: tuple[int, ...]
    input_strides: tuple[tuple[int, ...], ...]
    output_strides: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[torch.dtype, ...]
    allocation_dtypes: tuple[torch.dtype, ...]
    allocation_sources: tuple[_AllocationSource | None, ...]
    constant_tail: tuple[Any, ...]
    libentry: Any | None
    launch_plan: LibEntryLaunchPlan | None
    grid: tuple[int, int, int]
    tile_sizes: tuple[int, ...]
    num_warps: int
    num_stages: int | None
    path: str


def _ensure_pointwise_state(function):
    cache = getattr(function, "_aten_routing_plan_cache", None)
    if cache is None:
        cache = _TwoLevelCache(
            (
                "pointwise_dynamic",
                function._scalar_fn.__module__,
                function._scalar_fn.__name__,
                function._scalar_fn_cache_key,
            )
        )
        function._aten_routing_plan_cache = cache
        function._aten_plan_kernels = {}
    return cache


def _pointwise_cache_info(function):
    return _ensure_pointwise_state(function).info()


def _pointwise_clear_plan_cache(function, reset_stats=True):
    return _ensure_pointwise_state(function).clear(reset_stats)


def _pointwise_signature(function, args, kwargs):
    schema = function.fx
    outputs = tuple(
        kwargs.get(f"out{index}") if kwargs.get(f"out{index}") is not None else None
        for index in range(schema.num_output_tensors())
    )
    aliases = tuple(
        next(
            (
                index
                for index, value in enumerate(args)
                if isinstance(value, torch.Tensor) and output is value
            ),
            None,
        )
        for output in outputs
    )
    if all(output is None for output in outputs):
        semantics = "functional"
    elif any(alias is not None for alias in aliases):
        semantics = "inplace"
    else:
        semantics = "out"
    tensor_shapes = [
        tuple(value.shape)
        for index, value in enumerate(args)
        if schema.is_tensor(index) and isinstance(value, torch.Tensor)
    ]
    max_ndim = max((len(shape) for shape in tensor_shapes), default=0)
    broadcast_pattern = tuple(
        (True,) * (max_ndim - len(shape)) + tuple(size == 1 for size in shape)
        for shape in tensor_shapes
    )
    return (
        semantics,
        aliases,
        _argument_signatures(args),
        tuple(None if value is None else _tensor_signature(value) for value in outputs),
        broadcast_pattern,
        bool(function._should_use_complex_path(args)),
        (
            function.config.max_tile_size,
            tuple(function.config.max_grid_size),
            function.config.max_num_warps_per_cta,
            bool(function.config.prefer_1d_tile),
            bool(function.config.prefer_block_pointer),
            bool(function.config.balance_grid),
        ),
    )


def _cached_pointwise_instantiate(self, ndim):
    overload = _ORIGINAL_POINTWISE_INSTANTIATE(self, ndim)
    _ensure_pointwise_state(self)
    key = (
        ndim,
        self.config.max_tile_size,
        tuple(self.config.max_grid_size),
        self.config.max_num_warps_per_cta,
        self.config.prefer_1d_tile,
        self.config.prefer_block_pointer,
        self.config.balance_grid,
    )
    if key not in self._aten_plan_kernels:
        kernel_name, _, _ = self._compute_kernel_names(ndim)
        self._aten_plan_kernels[key] = overload.__globals__[kernel_name]
    return overload


def _pointwise_geometry(function, prepared_args, outputs, ndim, task_shape):
    pointwise_module = importlib.import_module("flag_gems.utils.pointwise_dynamic")

    schema = function.fx
    if ndim == 0:
        return (), {"num_warps": 1}, (1, 1, 1), (), 1
    max_tile_size = function.config.max_tile_size
    if pointwise_module._tensor_inputs_all_complex(schema):
        max_tile_size //= 2
    major, _ = _current_device_context()[-1]
    hopper_fill = function._scalar_fn.__name__.find("fill_scalar") != -1 and major >= 9
    if hopper_fill:
        tile_sizes = (1024,) if function.config.prefer_1d_tile else (64,)
    elif function.config.prefer_1d_tile:
        tile_sizes = heuristics_for_tile_size(max_tile_size, math.prod(task_shape))
    else:
        tile_sizes = heuristics_for_tile_size(max_tile_size, *task_shape)
    tile_size = math.prod(tile_sizes)
    if function.config.prefer_1d_tile:
        num_tiles = triton.cdiv(math.prod(task_shape), tile_size)
    else:
        num_tiles = math.prod(
            triton.cdiv(size, tile) for size, tile in zip(task_shape, tile_sizes)
        )
    if function.config.balance_grid:
        num_ctas, tiles_per_cta = pointwise_module._balanced_grid_partition(
            num_tiles, function.config.max_grid_size[0]
        )
    else:
        num_ctas = (
            num_tiles
            if hopper_fill
            else min(function.config.max_grid_size[0], num_tiles)
        )
        tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    block_pointer = (
        not function.config.prefer_1d_tile and function.config.prefer_block_pointer
    )
    constant_tail = []
    tensor_inputs = tuple(
        prepared_args[index]
        for index in range(schema.num_inputs())
        if schema.is_tensor(index)
    )
    for value in tensor_inputs + outputs:
        strides = tuple(value.stride())
        constant_tail.extend(strides)
        if block_pointer:
            constant_tail.extend(tuple(stride_order(strides)) if ndim >= 2 else (0,))
    constant_tail.extend(task_shape)
    constant_tail.append(math.prod(task_shape))
    kernel_kwargs = {
        "tiles_per_cta": tiles_per_cta,
        "one_tile_per_cta": tiles_per_cta == 1,
        "num_warps": num_warps,
    }
    if function.config.prefer_1d_tile:
        kernel_kwargs["tile_size"] = tile_size
    else:
        kernel_kwargs.update(
            {f"tile_size{index}": value for index, value in enumerate(tile_sizes)}
        )
    return (
        tuple(constant_tail),
        kernel_kwargs,
        (num_ctas, 1, 1),
        tuple(tile_sizes),
        num_warps,
    )


def _build_pointwise_plan(
    function, raw_args, raw_kwargs, prepared_args, prepared_kwargs, ndim
):
    schema = function.fx
    outputs = tuple(
        prepared_kwargs[f"out{index}"] for index in range(schema.num_output_tensors())
    )
    task_shape = tuple(outputs[0].shape)
    input_strides = tuple(
        tuple(prepared_args[index].stride())
        for index in range(schema.num_inputs())
        if schema.is_tensor(index)
    )
    output_strides = tuple(tuple(output.stride()) for output in outputs)
    output_dtypes = tuple(output.dtype for output in outputs)
    missing_outputs = tuple(
        index
        for index in range(schema.num_output_tensors())
        if f"out{index}" not in raw_kwargs
    )
    source_values = []
    source_descriptors = []
    for output_index in range(schema.num_output_tensors()):
        key = f"out{output_index}"
        if key in raw_kwargs:
            source_values.append(raw_kwargs[key])
            source_descriptors.append(_AllocationSource("out", output_index))
    for arg_index, value in enumerate(raw_args):
        if schema.is_tensor(arg_index):
            source_values.append(value)
            source_descriptors.append(_AllocationSource("arg", arg_index))
    fast_path = function.use_fast_path(source_values)
    if fast_path:
        allocation_source = source_descriptors[0]
    else:
        allocation_source = next(
            (
                descriptor
                for value, descriptor in zip(source_values, source_descriptors)
                if tuple(value.shape) == task_shape
            ),
            None,
        )
    num_tasks = math.prod(task_shape)
    if num_tasks:
        constant_tail, kernel_kwargs, grid, tile_sizes, num_warps = _pointwise_geometry(
            function, prepared_args, outputs, ndim, task_shape
        )
    else:
        constant_tail, kernel_kwargs = (), {}
        grid, tile_sizes, num_warps = (0, 0, 0), (), 0
    libentry = None
    launch_plan = None
    if num_tasks:
        key = (
            ndim,
            function.config.max_tile_size,
            tuple(function.config.max_grid_size),
            function.config.max_num_warps_per_cta,
            function.config.prefer_1d_tile,
            function.config.prefer_block_pointer,
            function.config.balance_grid,
        )
        libentry = function._aten_plan_kernels[key]
        kernel_args = tuple(prepared_args) + outputs + constant_tail
        first_tensor = next(
            value
            for value in tuple(prepared_args) + outputs
            if hasattr(value, "device")
        )
        with torch_device_fn.device(first_tensor.device):
            launch_plan = _build_libentry_plan(
                libentry, *kernel_args, grid=grid, **kernel_kwargs
            )
    return PointwisePlan(
        complex_path=False,
        fast_path=fast_path,
        task_shape=task_shape,
        input_strides=input_strides,
        output_strides=output_strides,
        output_dtypes=output_dtypes,
        allocation_dtypes=tuple(output_dtypes[index] for index in missing_outputs),
        allocation_sources=tuple(allocation_source for _ in missing_outputs),
        constant_tail=constant_tail,
        libentry=libentry,
        launch_plan=launch_plan,
        grid=grid,
        tile_sizes=tile_sizes,
        num_warps=num_warps,
        num_stages=launch_plan.num_stages if launch_plan is not None else None,
        path="fast" if fast_path else "broadcast",
    )


def _try_build_pointwise_plan(
    function, raw_args, raw_kwargs, prepared_args, prepared_kwargs, ndim
):
    try:
        return _build_pointwise_plan(
            function,
            raw_args,
            raw_kwargs,
            prepared_args,
            prepared_kwargs,
            ndim,
        )
    except Exception:
        logger.debug(
            "FlagGems could not build a pointwise routing plan for %s",
            function._scalar_fn.__name__,
            exc_info=True,
        )
        return None


def _resolve_source(source, args, kwargs):
    return args[source.index] if source.kind == "arg" else kwargs[f"out{source.index}"]


def _run_pointwise_plan(function, plan, args, kwargs):
    if plan.complex_path:
        return _ORIGINAL_POINTWISE_CALL(function, *args, **kwargs)
    schema = function.fx
    raw_kwargs = {
        f"out{index}": kwargs[f"out{index}"]
        for index in range(schema.num_output_tensors())
        if kwargs.get(f"out{index}") is not None
    }
    first_tensor = next(
        value
        for value in tuple(raw_kwargs.values()) + tuple(args)
        if isinstance(value, torch.Tensor)
    )
    outputs = []
    allocation_index = 0
    for output_index in range(schema.num_output_tensors()):
        key = f"out{output_index}"
        if key in raw_kwargs:
            output = raw_kwargs[key]
        else:
            source = plan.allocation_sources[allocation_index]
            dtype = plan.allocation_dtypes[allocation_index]
            allocation_index += 1
            output = (
                torch.empty(plan.task_shape, dtype=dtype, device=first_tensor.device)
                if source is None
                else torch.empty_like(
                    _resolve_source(source, args, raw_kwargs), dtype=dtype
                )
            )
        outputs.append(output)
    prepared_args = []
    stride_index = 0
    for arg_index, value in enumerate(args):
        if schema.is_tensor(arg_index):
            prepared_args.append(
                StridedBuffer(value, plan.task_shape, plan.input_strides[stride_index])
            )
            stride_index += 1
        else:
            prepared_args.append(value)
    output_buffers = tuple(
        StridedBuffer(output, plan.task_shape, plan.output_strides[index])
        for index, output in enumerate(outputs)
    )
    if math.prod(plan.task_shape):
        if plan.launch_plan is None:
            prepared_kwargs = {
                f"out{index}": output for index, output in enumerate(output_buffers)
            }
            result = function.instantiate(len(plan.task_shape))(
                *prepared_args, **prepared_kwargs
            )
            return function._unwrap(result)
        kernel_args = tuple(prepared_args) + output_buffers + plan.constant_tail
        with torch_device_fn.device(first_tensor.device):
            _run_libentry_plan(
                plan.libentry,
                plan.launch_plan,
                *kernel_args,
                _aten_plan_device_context=plan.launch_plan.device_context,
            )
    return outputs[0] if schema.num_output_tensors() == 1 else tuple(outputs)


def _cached_pointwise_call(self, *args, **kwargs):
    if not _ENABLED or _CACHE_SIZE == 0:
        return _ORIGINAL_POINTWISE_CALL(self, *args, **kwargs)
    cache = _ensure_pointwise_state(self)
    signature = _pointwise_signature(self, args, kwargs)
    if signature[5]:
        cache.bypasses += 1
        return _ORIGINAL_POINTWISE_CALL(self, *args, **kwargs)
    full_signature, plan = cache.lookup(
        signature, device_context=_argument_device_context(args, kwargs)
    )
    if full_signature is None:
        return _ORIGINAL_POINTWISE_CALL(self, *args, **kwargs)
    if plan is not None:
        try:
            return _run_pointwise_plan(self, plan, args, kwargs)
        except _StaleLaunchPlan:
            cache.clear(reset_stats=False)
            return _ORIGINAL_POINTWISE_CALL(self, *args, **kwargs)
    logger.debug(
        "FlagGems pointwise routing-plan cache miss: %s", self._scalar_fn.__name__
    )
    raw_kwargs = {
        f"out{index}": kwargs[f"out{index}"]
        for index in range(self.fx.num_output_tensors())
        if kwargs.get(f"out{index}") is not None
    }
    ndim, prepared_args, prepared_kwargs = self.prepare_args(*args, **kwargs)
    result = self.instantiate(ndim)(*prepared_args, **prepared_kwargs)
    plan = _try_build_pointwise_plan(
        self, args, raw_kwargs, prepared_args, prepared_kwargs, ndim
    )
    cache.insert(full_signature, plan)
    return self._unwrap(result)


def cache_stats():
    caches = tuple(_ALL_CACHES)
    last_hits = sum(cache.last_hits for cache in caches)
    lru_hits = sum(cache.lru_hits for cache in caches)
    misses = sum(cache.misses for cache in caches)
    per_operator = {}
    for cache in caches:
        info = cache.info()
        per_operator[_operator_name(cache.operator)] = {
            "last_hits": info.last_hits,
            "lru_hits": info.lru_hits,
            "hits": info.hits,
            "misses": info.misses,
            "evictions": info.evictions,
            "bypasses": info.bypasses,
            "size": info.size,
            "hit_rate": info.hit_rate,
        }
    return {
        "enabled": _ENABLED,
        "installed": _INSTALLED,
        "last_hits": last_hits,
        "lru_hits": lru_hits,
        "hits": last_hits + lru_hits,
        "misses": misses,
        "evictions": sum(cache.evictions for cache in caches),
        "bypasses": sum(cache.bypasses for cache in caches),
        "size": sum(cache.info().size for cache in caches),
        "hit_rate": (
            (last_hits + lru_hits) / (last_hits + lru_hits + misses)
            if last_hits + lru_hits + misses
            else 0.0
        ),
        "per_operator": per_operator,
    }


def clear_caches(reset_stats=True):
    for cache in tuple(_ALL_CACHES):
        cache.clear(reset_stats=reset_stats)


def reset_stats():
    for cache in tuple(_ALL_CACHES):
        cache.reset_stats()


def _install_hooks() -> bool:
    """Install reversible wrappers on compatible generic FlagGems classes.

    ``LibEntry`` is patched for every vendor because its ABI is shared.  The
    richer pointwise routing cache is installed only on the generic
    implementation; vendor-private forks retain their original call path.
    """

    global _INSTALLED
    global _ORIGINAL_LIBENTRY_RUN
    global _ORIGINAL_LIBENTRY_BUILD_PLAN
    global _ORIGINAL_LIBENTRY_RUN_PLAN
    global _ORIGINAL_POINTWISE_CALL
    global _ORIGINAL_POINTWISE_CACHE_INFO
    global _ORIGINAL_POINTWISE_CLEAR_CACHE
    global _ORIGINAL_POINTWISE_INSTANTIATE
    if _INSTALLED:
        return True
    try:
        from flag_gems.utils.libentry import LibEntry
        from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction
    except (ImportError, AttributeError):
        return False
    required_libentry = ("run", "key", "_apply_flagtune")
    required_pointwise = (
        "prepare_args",
        "instantiate",
        "_compute_kernel_names",
        "_unwrap",
    )
    if not all(hasattr(LibEntry, name) for name in required_libentry):
        return False
    if not all(hasattr(PointwiseDynamicFunction, name) for name in required_pointwise):
        return False
    _ORIGINAL_LIBENTRY_RUN = LibEntry.run
    _ORIGINAL_LIBENTRY_BUILD_PLAN = getattr(LibEntry, "build_launch_plan", _MISSING)
    _ORIGINAL_LIBENTRY_RUN_PLAN = getattr(LibEntry, "run_with_launch_plan", _MISSING)
    _ORIGINAL_POINTWISE_CALL = PointwiseDynamicFunction.__call__
    _ORIGINAL_POINTWISE_CACHE_INFO = getattr(
        PointwiseDynamicFunction, "cache_info", _MISSING
    )
    _ORIGINAL_POINTWISE_CLEAR_CACHE = getattr(
        PointwiseDynamicFunction, "clear_plan_cache", _MISSING
    )
    _ORIGINAL_POINTWISE_INSTANTIATE = PointwiseDynamicFunction.instantiate
    LibEntry.build_launch_plan = _build_libentry_plan
    LibEntry.run_with_launch_plan = _run_libentry_plan
    LibEntry.run = _cached_libentry_run
    PointwiseDynamicFunction.instantiate = _cached_pointwise_instantiate
    PointwiseDynamicFunction.__call__ = _cached_pointwise_call
    PointwiseDynamicFunction.cache_info = _pointwise_cache_info
    PointwiseDynamicFunction.clear_plan_cache = _pointwise_clear_plan_cache
    _INSTALLED = True
    logger.info(
        "Installed FlagGems ATen routing/launch-plan cache (vendor=%s, max_size=%d)",
        runtime_device.vendor_name,
        _CACHE_SIZE,
    )
    return True


def configure(*, max_size=None, include=_UNSET, exclude=_UNSET):
    """Update the runtime cache policy and invalidate existing plans.

    Args:
        max_size: Maximum LRU entries per operator and device.
        include: Optional glob pattern or iterable of patterns. If provided,
            only matching fully-qualified kernel names are cached.
        exclude: Glob pattern or iterable of patterns to bypass. ``arange`` is
            excluded by default because its launch setup is already cheaper
            than the cache lookup for the measured small metadata workloads.
    """

    global _CACHE_SIZE
    global _EXCLUDE
    global _INCLUDE
    global _POLICY_EPOCH
    if max_size is not None:
        max_size = int(max_size)
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        _CACHE_SIZE = max_size
    if include is not _UNSET:
        _INCLUDE = _split_patterns(include)
    if exclude is not _UNSET:
        _EXCLUDE = _split_patterns(exclude) or ()
    _POLICY_EPOCH += 1
    clear_caches(reset_stats=False)


def enable(*, max_size=None, include=_UNSET, exclude=_UNSET) -> bool:
    """Enable the cache explicitly and return whether hooks were installed."""

    global _ENABLED
    if max_size is None:
        max_size = os.getenv("FLAGGEMS_ATEN_PLAN_CACHE_SIZE", _CACHE_SIZE)
    if include is _UNSET and "FLAGGEMS_ATEN_PLAN_CACHE_INCLUDE" in os.environ:
        include = os.environ["FLAGGEMS_ATEN_PLAN_CACHE_INCLUDE"]
    if exclude is _UNSET and "FLAGGEMS_ATEN_PLAN_CACHE_EXCLUDE" in os.environ:
        exclude = os.environ["FLAGGEMS_ATEN_PLAN_CACHE_EXCLUDE"]
    configure(max_size=max_size, include=include, exclude=exclude)
    if not _install_hooks():
        return False
    _ENABLED = True
    logger.info(
        "Enabled FlagGems ATen routing/launch-plan cache (vendor=%s, max_size=%d)",
        runtime_device.vendor_name,
        _CACHE_SIZE,
    )
    return True


def disable(*, clear=True):
    """Disable lookups without removing wrappers, enabling cheap A/B tests."""

    global _ENABLED
    _ENABLED = False
    if clear:
        clear_caches()


def uninstall():
    """Restore the original classes and release all cached launch plans."""

    global _INSTALLED
    if not _INSTALLED:
        disable()
        return
    from flag_gems.utils.libentry import LibEntry
    from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction

    disable()
    if LibEntry.run is _cached_libentry_run:
        LibEntry.run = _ORIGINAL_LIBENTRY_RUN
    if PointwiseDynamicFunction.__call__ is _cached_pointwise_call:
        PointwiseDynamicFunction.__call__ = _ORIGINAL_POINTWISE_CALL
    if PointwiseDynamicFunction.instantiate is _cached_pointwise_instantiate:
        PointwiseDynamicFunction.instantiate = _ORIGINAL_POINTWISE_INSTANTIATE
    for owner, name, expected, original in (
        (
            LibEntry,
            "build_launch_plan",
            _build_libentry_plan,
            _ORIGINAL_LIBENTRY_BUILD_PLAN,
        ),
        (
            LibEntry,
            "run_with_launch_plan",
            _run_libentry_plan,
            _ORIGINAL_LIBENTRY_RUN_PLAN,
        ),
        (
            PointwiseDynamicFunction,
            "cache_info",
            _pointwise_cache_info,
            _ORIGINAL_POINTWISE_CACHE_INFO,
        ),
        (
            PointwiseDynamicFunction,
            "clear_plan_cache",
            _pointwise_clear_plan_cache,
            _ORIGINAL_POINTWISE_CLEAR_CACHE,
        ),
    ):
        if getattr(owner, name, None) is expected:
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
    _INSTALLED = False


def install(**kwargs) -> bool:
    """Backward-compatible alias for :func:`enable`."""

    if _INSTALLED and _ENABLED and not kwargs:
        return True
    return enable(**kwargs)


__all__ = [
    "CacheInfo",
    "LibEntryLaunchPlan",
    "PointwisePlan",
    "cache_stats",
    "clear_caches",
    "configure",
    "disable",
    "enable",
    "install",
    "reset_stats",
    "uninstall",
]
