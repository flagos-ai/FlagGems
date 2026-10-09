# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import hashlib
import importlib.util
import itertools
import logging
import math
import numbers
import operator
import struct
import threading

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import write_atomic

logger = logging.getLogger(__name__)
_group_metadata = getattr(torch._C, "_group_tensors_by_device_and_dtype", None)


@triton.jit
def _adamw_scalar(value, DOUBLE: tl.constexpr):
    # Python floats normally enter Triton as fp32. Double parameters instead
    # receive the exact IEEE bits as runtime integers, never constexpr values.
    if DOUBLE:
        return value.to(tl.int64).to(tl.float64, bitcast=True)
    else:
        return value.to(tl.float32)


@triton.jit
def _adamw_correction(beta, log_beta, step, DOUBLE: tl.constexpr):
    if DOUBLE:
        return 1.0 - tl_extra_shim.pow(beta, step)
    else:
        # Evaluate 1 - exp(step * log(beta)) without cancellation near beta == 1.
        # The small-argument polynomial avoids requiring a vendor expm1 symbol.
        x = step * log_beta
        small = -x * (
            1.0
            + x
            * (
                0.5
                + x
                * (
                    1.0 / 6.0
                    + x
                    * (
                        1.0 / 24.0
                        + x
                        * (
                            1.0 / 120.0
                            + x * (1.0 / 720.0 + x * (1.0 / 5040.0 + x / 40320.0))
                        )
                    )
                )
            )
        )
        correction = tl.where(tl.abs(x) < 0.5, small, 1.0 - tl.exp(x))
        # beta == 0 has log_beta == -inf; pow(0, 0) is nevertheless 1.
        return tl.where(step == 0.0, 0.0, correction)


@triton.jit
def _adamw_update(
    block_id,
    Param,
    Grad,
    ExpAvg,
    ExpAvgSq,
    MaxExpAvgSq,
    Step,
    Lr,
    GradScale,
    FoundInf,
    n,
    lr,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    log_beta1,
    log_beta2,
    weight_decay,
    eps,
    DOUBLE: tl.constexpr,
    TENSOR_LR: tl.constexpr,
    AMSGRAD: tl.constexpr,
    MAXIMIZE: tl.constexpr,
    HAS_GRAD_SCALE: tl.constexpr,
    HAS_FOUND_INF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HCU_HALF_UPDATE: tl.constexpr = False,
):
    # Entry kernels check found_inf before entering this shared update body.
    ACC: tl.constexpr = tl.float64 if DOUBLE else tl.float32
    if TENSOR_LR:
        lr_value = tl.load(Lr).to(ACC)
    else:
        lr_value = _adamw_scalar(lr, DOUBLE)
    b1 = _adamw_scalar(beta1, DOUBLE)
    b2 = _adamw_scalar(beta2, DOUBLE)
    c1 = _adamw_scalar(one_minus_beta1, DOUBLE)
    c2 = _adamw_scalar(one_minus_beta2, DOUBLE)
    log_b1 = _adamw_scalar(log_beta1, DOUBLE)
    log_b2 = _adamw_scalar(log_beta2, DOUBLE)
    decay = _adamw_scalar(weight_decay, DOUBLE)
    epsilon = _adamw_scalar(eps, DOUBLE)
    step = tl.load(Step).to(ACC)
    if HCU_HALF_UPDATE:
        # The first optimizer step has exact bias corrections 1 - beta.
        # Keep a device branch so later steps retain the general formula.
        if step == 1.0:
            bc1 = c1
            bc2 = c2
        else:
            bc1 = _adamw_correction(b1, log_b1, step, DOUBLE)
            bc2 = _adamw_correction(b2, log_b2, step, DOUBLE)
    else:
        bc1 = _adamw_correction(b1, log_b1, step, DOUBLE)
        bc2 = _adamw_correction(b2, log_b2, step, DOUBLE)
    if HCU_HALF_UPDATE:
        inv_sqrt_bc2 = 1.0 / tl.sqrt(bc2)

    offsets = block_id.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    p = tl.load(Param + offsets, mask, other=0).to(ACC)
    g = tl.load(Grad + offsets, mask, other=0).to(ACC)
    m = tl.load(ExpAvg + offsets, mask, other=0).to(ACC)
    v = tl.load(ExpAvgSq + offsets, mask, other=0).to(ACC)
    if HAS_GRAD_SCALE:
        g = g / tl.load(GradScale).to(ACC)
        tl.store(Grad + offsets, g, mask)
    if MAXIMIZE:
        g = -g

    p = p * (1.0 - lr_value * decay)
    if HCU_HALF_UPDATE:
        m = tl.fma(b1, m, c1 * g)
        v = tl.fma(b2, v, (c2 * g) * g)
    else:
        m = b1 * m + c1 * g
        v = b2 * v + c2 * g * g
    variance = v
    if AMSGRAD:
        maximum = tl.load(MaxExpAvgSq + offsets, mask, other=0).to(ACC)
        # std::max keeps its first operand if either comparison is unordered.
        variance = tl.where(maximum < v, v, maximum)
        tl.store(MaxExpAvgSq + offsets, variance, mask)
    if DOUBLE:
        denom = tl_extra_shim.sqrt(variance) / tl_extra_shim.sqrt(bc2) + epsilon
    elif HCU_HALF_UPDATE:
        denom = tl.sqrt(variance) * inv_sqrt_bc2 + epsilon
    else:
        denom = tl.sqrt(variance) / tl.sqrt(bc2) + epsilon
    p = p - (lr_value / bc1) * m / denom
    tl.store(Param + offsets, p, mask)
    tl.store(ExpAvg + offsets, m, mask)
    tl.store(ExpAvgSq + offsets, v, mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "lr",
        "beta1",
        "beta2",
        "one_minus_beta1",
        "one_minus_beta2",
        "log_beta1",
        "log_beta2",
        "weight_decay",
        "eps",
    ]
)
def _fused_adamw_kernel(
    Param,
    Grad,
    ExpAvg,
    ExpAvgSq,
    MaxExpAvgSq,
    Step,
    Lr,
    GradScale,
    FoundInf,
    n,
    lr,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    log_beta1,
    log_beta2,
    weight_decay,
    eps,
    DOUBLE: tl.constexpr,
    TENSOR_LR: tl.constexpr,
    AMSGRAD: tl.constexpr,
    MAXIMIZE: tl.constexpr,
    HAS_GRAD_SCALE: tl.constexpr,
    HAS_FOUND_INF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if HAS_FOUND_INF:
        if tl.load(FoundInf) == 1.0:
            return
    _adamw_update(
        tle.program_id(0),
        Param,
        Grad,
        ExpAvg,
        ExpAvgSq,
        MaxExpAvgSq,
        Step,
        Lr,
        GradScale,
        FoundInf,
        n,
        lr,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        log_beta1,
        log_beta2,
        weight_decay,
        eps,
        DOUBLE,
        TENSOR_LR,
        AMSGRAD,
        MAXIMIZE,
        HAS_GRAD_SCALE,
        HAS_FOUND_INF,
        BLOCK_SIZE,
    )


@functools.lru_cache(maxsize=1)
def _adamw_hcu_runtime():
    """Check the pinned public launch ABI without importing it on other vendors."""
    if triton.__version__.split(".")[:2] != ["3", "6"]:
        return None
    try:
        import inspect

        from triton import knobs
        from triton.compiler import CompiledKernel
        from triton.runtime import driver
        from triton.runtime._distributed import DistributedRtContext

        if (
            "do_not_specialize_on_alignment"
            not in inspect.signature(triton.jit).parameters
            or not hasattr(CompiledKernel, "__getitem__")
            or not hasattr(knobs.runtime, "debug")
            or not hasattr(knobs.compilation, "instrumentation_mode")
        ):
            return None
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    return driver, knobs, DistributedRtContext


@functools.lru_cache(maxsize=1)
def _adamw_hcu_c_launch_abi():
    """Fail closed unless the complete installed C-launch closure is pinned."""
    try:
        import inspect
        from pathlib import Path
        from types import BuiltinFunctionType

        from triton import knobs
        from triton.backends.hcu.compiler_hcu import HIPBackend
        from triton.backends.hcu.driver import HIPLauncher
        from triton.compiler import CompiledKernel
        from triton.compiler.compiler import make_backend

        for cls, expected in (
            (
                CompiledKernel,
                "46e29ef2b751cac5e1fc99337d13ebde0e540e9907e28a0146b2250c081f81d9",
            ),
            (
                HIPLauncher,
                "e06e5aad9ba2963ec3b4e8cae7a2c4c473657db37ceecaee99e8092c46736753",
            ),
            (
                HIPBackend,
                "3d6fe9044081a587938b6f2350d0173e6c4a0daf477038e9d4fa68b0acaccadb",
            ),
            (
                knobs.HookChain,
                "233f2698dcaf166056b8794df1caf55a0a438025f086eeb6ec1c064038b86793",
            ),
        ):
            path = inspect.getsourcefile(cls)
            if (
                path is None
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
            ):
                return None
        # Source-file identity also rejects already substituted Python wrappers.
        for cls, name in (
            (CompiledKernel, "__getitem__"),
            (HIPLauncher, "__call__"),
            (knobs.HookChain, "__call__"),
        ):
            if inspect.getsourcefile(getattr(cls, name)) != inspect.getsourcefile(cls):
                return None
        return (
            CompiledKernel,
            HIPLauncher,
            HIPBackend,
            BuiltinFunctionType,
            make_backend,
            CompiledKernel.__getitem__,
            HIPLauncher.__call__,
            knobs.HookChain,
            knobs.HookChain.__call__,
        )
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return None


def _adamw_hcu_hook_is_empty(hook, abi):
    """Recognize only the pinned no-op HookChain, without changing global hooks."""
    if hook is None:
        return True
    try:
        if type(hook) is not abi[7] or type(hook).__call__ is not abi[8]:
            return False
        calls = hook.calls
        return type(calls) is list and len(calls) == 0 and type(hook.reversed) is bool
    except (AttributeError, TypeError):
        return False


class _AdamwHcuLauncher:
    """Cache compiled functions, with fresh runtime addresses on every launch."""

    def __init__(self, count, branching, hcu_runtime):
        self.count = count
        self.branching = branching
        self.hcu_runtime = hcu_runtime
        self.kernels = {}
        self.c_launch_abi = _adamw_hcu_c_launch_abi()
        self.c_launchers = {}
        self.lock = threading.Lock()

    def _prepare_c_launch(self, kernel, device, wide, double):
        """Inspect an already initialized kernel; never submit a trial launch."""
        if self.c_launch_abi is None:
            return None
        compiled_type, launcher_type, backend_type, builtin_type, make_backend = (
            self.c_launch_abi[:5]
        )
        try:
            if type(kernel) is not compiled_type or kernel.module is None:
                return None
            metadata = kernel.metadata
            if type(make_backend(metadata.target)) is not backend_type:
                return None
            runner = kernel.run
            if (
                type(runner) is not launcher_type
                or type(runner.launch) is not builtin_type
                or runner.launch.__name__ != "launch"
                or runner.launch.__module__ != "__triton_launcher"
                or type(kernel.function) is not int
                or kernel.function <= 0
                or runner.profile_scratch_size != 0
                or metadata.profile_scratch_size != 0
                or runner.launch_cooperative_grid is not False
                or metadata.launch_cooperative_grid is not False
                or metadata.debug
                or metadata.instrumentation_mode
                or metadata.use_nvshmem
                or metadata.tensordesc_meta not in (None, [])
                or getattr(metadata, "cluster_dims", (1, 1, 1)) != (1, 1, 1)
            ):
                return None
            packed = kernel.packed_metadata
            if (
                type(packed) is not tuple
                or len(packed) != 3
                or any(type(value) is not int for value in packed)
                or packed != (metadata.num_warps, metadata.num_ctas, metadata.shared)
                or packed[0] <= 0
                or packed[1] != 1
                or packed[2] < 0
            ):
                return None
            signature = kernel.src.signature
            expected = (
                ("u64",) * (6 * self.count + 3)
                + (("i64" if wide else "i32"),) * (2 * self.count)
                + (("i64" if double else "fp32"),) * 9
                + ("constexpr",) * 10
            )
            if (
                tuple(signature) != tuple(kernel.src.fn.arg_names)
                or tuple(signature.values()) != expected
            ):
                return None
            return (
                runner,
                runner.launch,
                device,
                kernel.module,
                kernel.function,
                metadata,
                packed,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            metadata_begin = 6 * self.count + 3
            wide = (
                max(
                    args[metadata_begin : metadata_begin + 2 * self.count],
                    default=0,
                )
                >= 2**31
            )
            fn = _adamw_flat_kernel(self.count, self.branching, kwargs["DOUBLE"], wide)
            driver, knobs, distributed_context = self.hcu_runtime
            # Preserve modes that depend on JIT-run behavior beyond the public
            # compiled runner. Both paths execute the same Triton update.
            if (
                distributed_context().is_lite_mode
                or fn.pre_run_hooks
                or knobs.runtime.debug
                or knobs.compilation.instrumentation_mode
            ):
                return fn[grid](*args, **kwargs)
            device = driver.active.get_current_device()
            key = (device, wide, tuple(sorted(kwargs.items())))
            # Arity/control flow belong to this instance. The key includes all
            # constexprs, dtypes and launch options. Runtime argument types are
            # explicit and disable value/alignment specialization; no tensors,
            # addresses, dimensions or scalar values are retained here.
            kernel = self.kernels.get(key)
            if kernel is None:
                with self.lock:
                    kernel = self.kernels.get(key)
                    if kernel is None:
                        kernel = fn[grid](*args, **kwargs)
                        self.c_launchers[key] = self._prepare_c_launch(
                            kernel, device, wide, kwargs["DOUBLE"]
                        )
                        self.kernels[key] = kernel
                        return kernel
            flags = (
                "DOUBLE",
                "TENSOR_LR",
                "AMSGRAD",
                "MAXIMIZE",
                "HAS_GRAD_SCALE",
                "HAS_FOUND_INF",
                "BLOCK_SIZE",
                "PARAM_TYPE",
                "STATE_TYPE",
                "ALIGNED_MAIN",
            )
            # HCU 3.6 takes the entire signature, including constexpr objects.
            # The public runner resolves current stream and invokes launch hooks.
            grid3 = tuple(grid) + (1,) * (3 - len(grid))
            direct = self.c_launchers.get(key)
            if (
                direct is not None
                and self.c_launch_abi is not None
                and direct[2] == device
                and kernel.run is direct[0]
                and kernel.module is direct[3]
                and kernel.function == direct[4]
                and kernel.metadata is direct[5]
                and kernel.packed_metadata is direct[6]
                and _adamw_hcu_hook_is_empty(
                    knobs.runtime.launch_enter_hook, self.c_launch_abi
                )
                and _adamw_hcu_hook_is_empty(
                    knobs.runtime.launch_exit_hook, self.c_launch_abi
                )
                and getattr(direct[0], "profile_scratch_size", None) == 0
                and getattr(kernel.metadata, "profile_scratch_size", None) == 0
                and getattr(direct[0], "launch_cooperative_grid", None) is False
                and getattr(kernel.metadata, "launch_cooperative_grid", None) is False
                and getattr(direct[0], "launch", None) is direct[1]
                and type(kernel).__getitem__ is self.c_launch_abi[5]
                and type(direct[0]).__call__ is self.c_launch_abi[6]
                and len(grid3) == 3
                and grid3[1:] == (1, 1)
                and type(grid3[0]) is int
                and 0 <= grid3[0] < 2**31
            ):
                # The pinned C entry retains scalar packing and HIP error checks.
                # Cold JIT already initialized handles/load hooks. Every call reads
                # the current stream; no runtime arguments or addresses are cached.
                stream = driver.active.get_current_stream(device)
                direct[1](
                    False,
                    *grid3,
                    stream,
                    kernel.function,
                    None,
                    kernel.packed_metadata,
                    None,
                    None,
                    None,
                    *args,
                    *(kwargs[name] for name in flags),
                )
            else:
                kernel[grid3](*args, *(kwargs[name] for name in flags))
            return kernel

        return launch


@functools.lru_cache(maxsize=256)
def _adamw_flat_kernel(count, branching, double=None, wide=False):
    """Cache kernels by arity and control flow, never by tensors or addresses."""
    integer_addresses = runtime.device.vendor_name == "hygon"
    if integer_addresses and double is None:
        hcu_runtime = _adamw_hcu_runtime()
        if hcu_runtime is not None:
            return _AdamwHcuLauncher(count, branching, hcu_runtime)
    pointer_groups = ("Param", "Grad", "ExpAvg", "ExpAvgSq", "MaxExpAvgSq", "Step")
    scalar_names = (
        "lr",
        "beta1",
        "beta2",
        "one_minus_beta1",
        "one_minus_beta2",
        "log_beta1",
        "log_beta2",
        "weight_decay",
        "eps",
    )
    flags = (
        "DOUBLE",
        "TENSOR_LR",
        "AMSGRAD",
        "MAXIMIZE",
        "HAS_GRAD_SCALE",
        "HAS_FOUND_INF",
        "BLOCK_SIZE",
    )
    pointers = [f"{group}{i}" for group in pointer_groups for i in range(count)]
    sizes = [f"N{i}" for i in range(count)]
    starts = [f"Start{i}" for i in range(count)]
    scalar_pointers = ["Lr", "GradScale", "FoundInf"]
    arguments = pointers + scalar_pointers + sizes + starts + list(scalar_names)
    signature = [
        (
            f"{name}: tl.uint64"
            if integer_addresses and name in pointers + scalar_pointers
            else name
        )
        for name in arguments
    ] + [f"{name}: tl.constexpr" for name in flags]
    if double is not None:
        for name in sizes + starts:
            index_type = "tl.int64" if wide else "tl.int32"
            signature[arguments.index(name)] = f"{name}: {index_type}"
        for name in scalar_names:
            scalar_type = "tl.int64" if double else "tl.float32"
            signature[arguments.index(name)] = f"{name}: {scalar_type}"
    do_not_specialize = sizes + starts + list(scalar_names)
    if integer_addresses:
        signature += [
            "PARAM_TYPE: tl.constexpr",
            "STATE_TYPE: tl.constexpr",
            "ALIGNED_MAIN: tl.constexpr",
        ]
        do_not_specialize += pointers + scalar_pointers
    decorator = f"@triton.jit(do_not_specialize={do_not_specialize!r})"
    if double is not None:
        decorator = (
            f"@triton.jit(do_not_specialize={do_not_specialize!r}, "
            f"do_not_specialize_on_alignment={do_not_specialize!r})"
        )
    lines = [
        "import triton",
        "import triton.language as tl",
        "from flag_gems.ops._fused_adamw_ import _adamw_update",
        "from flag_gems.utils import triton_lang_extension as tle",
        "",
        decorator,
        "def _flat_adamw(\n    " + ",\n    ".join(signature) + "\n):",
    ]
    if integer_addresses:
        # HCU's tensor specialization repeatedly inspects storage metadata.
        # Fresh scalar addresses bypass that binder work without caching any
        # tensor identity or assuming alignment of storage-offset views.
        for group in pointer_groups:
            pointee = (
                "PARAM_TYPE"
                if group in ("Param", "Grad")
                else "tl.float32" if group == "Step" else "STATE_TYPE"
            )
            for i in range(count):
                lines.append(
                    f"    {group}{i} = tl.cast({group}{i}, tl.pointer_type({pointee}))"
                )
        for name in scalar_pointers:
            lines.append(f"    {name} = tl.cast({name}, tl.pointer_type(tl.float32))")
    lines += [
        "    if HAS_FOUND_INF:",
        "        if tl.load(FoundInf) == 1.0:",
        "            return",
        "    pid = tle.program_id(0).to(tl.int64)",
    ]
    tail = list(scalar_names) + list(flags)
    if integer_addresses:
        tail.append("PARAM_TYPE == tl.float16")
    if branching:
        # Ascend cannot lower pointer-select/phi operations. Each disjoint
        # branch loads and stores its own tensors without merging pointers.
        for i in range(count):
            condition = f"pid >= Start{i}"
            if i + 1 < count:
                condition = f"({condition}) & (pid < Start{i + 1})"
            lines.append(f"    if {condition}:")
            args = [f"pid - Start{i}"] + [f"{group}{i}" for group in pointer_groups]
            args += scalar_pointers + [f"N{i}"] + tail
            lines.append("        _adamw_update(" + ", ".join(args) + ")")
    else:
        # Old CoreX launchers also reject tuples constructed inside a JIT
        # function. Emit scalar names directly throughout the kernel body.
        selected_names = ("p", "g", "m", "v", "mx", "step")
        for name, group in zip(selected_names, pointer_groups):
            lines.append(f"    {name} = {group}0")
        lines += [
            "    n = tl.cast(N0, tl.int64)",
            "    start = tl.cast(Start0, tl.int64)",
        ]
        for i in range(1, count):
            lines.append(f"    selected = pid >= Start{i}")
            for name, group in zip(selected_names, pointer_groups):
                lines.append(f"    {name} = tl.where(selected, {group}{i}, {name})")
            lines.append(f"    n = tl.where(selected, tl.cast(N{i}, tl.int64), n)")
            lines.append(
                f"    start = tl.where(selected, tl.cast(Start{i}, tl.int64), start)"
            )
        if integer_addresses:
            # Attach hints to the cast/selection results, rather than input
            # arguments whose annotations this HCU frontend discards.
            lines.append("    if ALIGNED_MAIN:")
            for name in selected_names[:5]:
                lines.append(f"        {name} = tl.multiple_of({name}, 16)")
            lines.append("        n = tl.multiple_of(n, 8)")
        args = ["pid - start"] + list(selected_names) + scalar_pointers + ["n"] + tail
        lines.append("    _adamw_update(" + ", ".join(args) + ")")
    source = "\n".join(lines) + "\n"
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    module_name = f"_fused_adamw_flat_{count}_{digest}"
    path = code_cache_dir() / f"{module_name}.py"
    write_atomic(path, source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._flat_adamw


def _adamw_dense(tensor):
    if tensor.is_contiguous() or tensor.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _adamw_device_scalar(tensor, name, device):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.numel() != 1
        or tensor.dtype != torch.float32
        or tensor.device != device
    ):
        raise RuntimeError(
            f"_fused_adamw_: {name} must be a one-element float32 tensor "
            f"on {device}"
        )


def _fused_adamw_run(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale,
    found_inf,
):
    groups = (
        ("params", params),
        ("grads", grads),
        ("exp_avgs", exp_avgs),
        ("exp_avg_sqs", exp_avg_sqs),
        ("max_exp_avg_sqs", max_exp_avg_sqs),
        ("state_steps", state_steps),
    )
    for name, tensors in groups:
        if not isinstance(tensors, (list, tuple)):
            raise TypeError(f"_fused_adamw_: {name} must be a list or tuple")
    count = len(params)
    for name, tensors in groups[1:]:
        expected = 0 if name == "max_exp_avg_sqs" and not amsgrad else count
        if len(tensors) != expected:
            raise RuntimeError(
                f"_fused_adamw_: {name} must contain {expected} tensors, "
                f"got {len(tensors)}"
            )

    # Only CPU tensor learning rates can be materialized on the host. Device
    # learning rates remain pointers so changing them works during graph replay.
    if isinstance(lr, torch.Tensor) and lr.device.type == "cpu":
        if lr.layout != torch.strided or lr.numel() != 1 or lr.dtype != torch.float32:
            raise RuntimeError("_fused_adamw_: lr must be a one-element float32 tensor")
        lr = lr.item()
    for name, value in (
        ("beta1", beta1),
        ("beta2", beta2),
        ("weight_decay", weight_decay),
        ("eps", eps),
    ):
        if (
            type(value) not in (float, int) and not isinstance(value, numbers.Real)
        ) or not math.isfinite(value):
            raise ValueError(f"_fused_adamw_: {name} must be finite")
        if value < 0 or (name in ("beta1", "beta2") and value >= 1):
            raise ValueError(f"_fused_adamw_: invalid {name}: {value}")
    tensor_lr = isinstance(lr, torch.Tensor)
    if not tensor_lr and (
        (type(lr) not in (float, int) and not isinstance(lr, numbers.Real))
        or not math.isfinite(lr)
        or lr < 0
    ):
        raise ValueError("_fused_adamw_: lr must be finite and nonnegative")
    if not count:
        return None

    if not isinstance(params[0], torch.Tensor):
        raise TypeError("_fused_adamw_: params must contain tensors")
    device, dtype = params[0].device, params[0].dtype
    vendor = runtime.device.vendor_name
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError("_fused_adamw_: unsupported parameter dtype")
    if dtype == torch.float64 and (
        vendor in ("ascend", "iluvatar", "mthreads") or not runtime.device.support_fp64
    ):
        raise RuntimeError(f"_fused_adamw_: float64 is unsupported on {vendor}")
    if not isinstance(exp_avgs[0], torch.Tensor):
        raise TypeError("_fused_adamw_: exp_avgs must contain tensors")
    state_dtype = exp_avgs[0].dtype
    if state_dtype != dtype and not (
        dtype == torch.float32 and state_dtype == torch.bfloat16
    ):
        raise RuntimeError("_fused_adamw_: unsupported moment dtype")

    # The one-list C++ grouping primitive checks every entry's actual dtype
    # and device. Requiring exactly one expected bucket keeps this stricter
    # than its multi-list optimizer mode, which permits mixed step dtypes.
    # This is host metadata only, with no tensor computation or device sync.
    bulk_metadata = count >= 16 and _group_metadata is not None
    if bulk_metadata:
        parameter_tensors = list(params)
        parameter_tensors.extend(grads)
        if state_dtype == dtype:
            for group in (exp_avgs, exp_avg_sqs, max_exp_avg_sqs):
                parameter_tensors.extend(group)
            batches = ((parameter_tensors, dtype),)
        else:
            state_tensors = list(exp_avgs)
            state_tensors.extend(exp_avg_sqs)
            state_tensors.extend(max_exp_avg_sqs)
            batches = ((parameter_tensors, dtype), (state_tensors, state_dtype))
        for tensors, expected_type in batches:
            try:
                buckets = _group_metadata([tensors], False)
            except (TypeError, RuntimeError) as error:
                # Keep malformed TensorLists readable without slowing the
                # valid path with a second Python pass over all operands.
                if any(not isinstance(t, torch.Tensor) for t in tensors):
                    raise TypeError(
                        "_fused_adamw_: operand lists must contain tensors"
                    ) from error
                raise
            if len(buckets) != 1 or (device, expected_type) not in buckets:
                raise RuntimeError(
                    "_fused_adamw_: incompatible operand dtype or device"
                )

    # HCU's many-tensor path uses C iterators for ordinary contiguous tensors.
    # All fields are read on every call, after the C++ dtype/device validation.
    # Subclasses and uncommon layouts retain the original diagnostic path.
    contiguous_metadata = False
    if vendor == "hygon" and bulk_metadata:
        tensor_groups = tuple(tensors for _, tensors in groups)
        ordinary_types = {torch.Tensor, torch.nn.Parameter}
        if (
            set(map(type, itertools.chain.from_iterable(tensor_groups)))
            <= ordinary_types
        ):
            if state_dtype == dtype:
                # The C++ grouping input already contains every main operand.
                # Check layouts once, then compare shapes without allocating a
                # second composite descriptor tuple for each Tensor.
                shape = operator.attrgetter("shape")
                param_shapes = tuple(map(shape, params))
                contiguous_metadata = (
                    set(map(operator.attrgetter("layout"), parameter_tensors))
                    == {torch.strided}
                    and all(map(torch.Tensor.is_contiguous, parameter_tensors))
                    and all(
                        all(map(operator.eq, map(shape, tensors), param_shapes))
                        for _, tensors in groups[1:5]
                        if tensors
                    )
                    and set(
                        map(
                            operator.attrgetter("dtype", "device", "layout"),
                            state_steps,
                        )
                    )
                    == {(torch.float32, device, torch.strided)}
                    and set(map(torch.Tensor.numel, state_steps)) == {1}
                )
            else:
                descriptor = operator.attrgetter("layout", "shape")
                param_descriptors = tuple(map(descriptor, params))
                contiguous_metadata = (
                    set(map(operator.attrgetter("layout"), params)) == {torch.strided}
                    and all(map(torch.Tensor.is_contiguous, params))
                    and all(
                        all(
                            map(
                                operator.eq, map(descriptor, tensors), param_descriptors
                            )
                        )
                        and all(map(torch.Tensor.is_contiguous, tensors))
                        for _, tensors in groups[1:5]
                        if tensors
                    )
                    and set(
                        map(
                            operator.attrgetter("dtype", "device", "layout"),
                            state_steps,
                        )
                    )
                    == {(torch.float32, device, torch.strided)}
                    and set(map(torch.Tensor.numel, state_steps)) == {1}
                )
    if contiguous_metadata:
        all_numels = tuple(map(torch.Tensor.numel, params))
        active = [index for index, n in enumerate(all_numels) if n]
        numels = [n for n in all_numels if n]
    else:
        # All metadata is local to this call: never reuse tensor identities or
        # addresses across calls. Validate parameters once, then matching effective
        # strides also prove that the corresponding operands are dense.
        metadata = []
        active = []
        numels = []
        for index, param in enumerate(params):
            if not bulk_metadata:
                if not isinstance(param, torch.Tensor):
                    raise TypeError("_fused_adamw_: params must contain tensors")
                if param.device != device or param.dtype != dtype:
                    raise RuntimeError(
                        "_fused_adamw_: inconsistent parameter dtype or device"
                    )
            if param.layout != torch.strided:
                raise RuntimeError("_fused_adamw_: parameters must be strided")
            contiguous = param.is_contiguous()
            if not contiguous and not _adamw_dense(param):
                raise RuntimeError(
                    "_fused_adamw_: parameters must be non-overlapping and dense"
                )
            shape = param.shape
            strides = None if contiguous else param.stride()
            metadata.append((shape, contiguous, strides))
            n = param.numel()
            if n:
                active.append(index)
                numels.append(n)
        for name, tensors in groups[1:5]:
            expected_dtype = dtype if name == "grads" else state_dtype
            for index, tensor in enumerate(tensors):
                if not bulk_metadata:
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"_fused_adamw_: {name} must contain tensors")
                    if tensor.device != device or tensor.dtype != expected_dtype:
                        raise RuntimeError(
                            f"_fused_adamw_: {name}[{index}] has an incompatible dtype or device"
                        )
                if tensor.layout != torch.strided:
                    raise RuntimeError(
                        f"_fused_adamw_: {name}[{index}] must be strided"
                    )
                shape, contiguous, strides = metadata[index]
                if tensor.shape != shape:
                    raise RuntimeError(
                        f"_fused_adamw_: {name}[{index}] has incompatible sizes"
                    )
                if contiguous and tensor.is_contiguous():
                    continue
                if strides is None:
                    strides = params[index].stride()
                if any(
                    size != 1 and left != right
                    for size, left, right in zip(shape, strides, tensor.stride())
                ):
                    raise RuntimeError(
                        f"_fused_adamw_: {name}[{index}] must match parameter strides"
                    )
        for index, step in enumerate(state_steps):
            # Construct indexed diagnostics only on the error path.
            if (
                not isinstance(step, torch.Tensor)
                or step.dtype != torch.float32
                or step.device != device
                or step.layout != torch.strided
                or step.numel() != 1
            ):
                _adamw_device_scalar(step, f"state_steps[{index}]", device)
    for name, tensor in (("grad_scale", grad_scale), ("found_inf", found_inf)):
        if tensor is not None:
            _adamw_device_scalar(tensor, name, device)
    if tensor_lr:
        _adamw_device_scalar(lr, "lr", device)

    double = dtype == torch.float64
    scalars = (
        0.0 if tensor_lr else float(lr),
        float(beta1),
        float(beta2),
        1.0 - beta1,
        1.0 - beta2,
        math.log(beta1) if beta1 else -math.inf,
        math.log(beta2) if beta2 else -math.inf,
        float(weight_decay),
        float(eps),
    )
    if double:
        scalars = tuple(struct.unpack("q", struct.pack("d", x))[0] for x in scalars)
    # Keep a direct path for a single tensor; bounded flat batches amortize
    # launches without host-to-device metadata transfers or captured addresses.
    # Ascend uses bounded independent branches to avoid pointer selects.
    # MUSA cannot allocate the argument payload of a 64-tensor batch.
    batch_size = 8 if vendor == "ascend" else 32 if vendor == "mthreads" else 64
    # Keep the flat argument payload below the legacy 4 KiB kernel ABI limit.
    # Int64 metadata needs a smaller batch than the ordinary int32 sizes/starts.
    if batch_size > 1 and (
        any(n >= 2**31 for n in numels)
        or sum((n + 127) // 128 for n in numels) >= 2**31
    ):
        batch_size = min(batch_size, 32)
    block_cap = (
        8192
        if vendor == "ascend"
        else 512 if vendor == "nvidia" and not double else 1024
    )
    integer_addresses = vendor == "hygon"
    if integer_addresses:
        tl_dtypes = {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32,
            torch.float64: tl.float64,
        }
        pointer_types = {
            "PARAM_TYPE": tl_dtypes[dtype],
            "STATE_TYPE": tl_dtypes[state_dtype],
        }
        if dtype == torch.float16:
            pointer_types["num_warps"] = 2
    if contiguous_metadata:
        # Fresh pointers are local to this invocation; TensorLists own storage.
        # Reuse only this call's second-moment pointers for the unused AMS input.
        address_groups = tuple(
            tuple(map(torch.Tensor.data_ptr, tensors))
            for tensors in (params, grads, exp_avgs, exp_avg_sqs)
        )
        address_groups += (
            (
                tuple(map(torch.Tensor.data_ptr, max_exp_avg_sqs))
                if amsgrad
                else address_groups[3]
            ),
            tuple(map(torch.Tensor.data_ptr, state_steps)),
        )
        if len(active) != count:
            address_groups = tuple(
                tuple(map(addresses.__getitem__, active))
                for addresses in address_groups
            )
    with runtime.torch_device_fn.device(device):
        for first in range(0, len(active), batch_size):
            indices = active[first : first + batch_size]
            sizes = tuple(numels[first : first + batch_size])
            block_size = min(block_cap, max(128, 1 << (max(sizes) - 1).bit_length()))
            starts = []
            blocks = 0
            for n in sizes:
                starts.append(blocks)
                blocks += (n + block_size - 1) // block_size
            operands = (
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs if amsgrad else exp_avg_sqs,
                state_steps,
            )
            if len(indices) == 1 and not integer_addresses:
                kernel = _fused_adamw_kernel
                tensors = tuple(group[indices[0]] for group in operands)
                metadata = (sizes[0],)
            else:
                kernel = _adamw_flat_kernel(len(indices), vendor == "ascend")
                if contiguous_metadata:
                    tensors = tuple(
                        itertools.chain.from_iterable(
                            addresses[first : first + batch_size]
                            for addresses in address_groups
                        )
                    )
                else:
                    tensors = tuple(group[i] for group in operands for i in indices)
                metadata = sizes + tuple(starts)
            if integer_addresses:
                # The original TensorLists remain live for the whole call.
                # Read every address afresh, including device LR and AMP state.
                if not contiguous_metadata:
                    tensors = tuple(tensor.data_ptr() for tensor in tensors)
                # Reject the common small tails before checking addresses.
                # Both classes use fresh pointers and the same update math.
                aligned_main = (
                    all(n % 8 == 0 for n in sizes)
                    and (
                        contiguous_metadata
                        or all(
                            type(group[i]) in (torch.Tensor, torch.nn.Parameter)
                            for group in operands
                            for i in indices
                        )
                    )
                    and all(
                        address % 16 == 0 for address in tensors[: 5 * len(indices)]
                    )
                )
                scalar_pointers = tuple(
                    tensor.data_ptr() if tensor is not None else 0
                    for tensor in (lr if tensor_lr else None, grad_scale, found_inf)
                )
                kernel[(blocks,)](
                    *tensors,
                    *scalar_pointers,
                    *metadata,
                    *scalars,
                    DOUBLE=double,
                    TENSOR_LR=tensor_lr,
                    AMSGRAD=amsgrad,
                    MAXIMIZE=maximize,
                    HAS_GRAD_SCALE=grad_scale is not None,
                    HAS_FOUND_INF=found_inf is not None,
                    BLOCK_SIZE=block_size,
                    enable_fp_fusion=False,
                    ALIGNED_MAIN=aligned_main,
                    **pointer_types,
                )
            else:
                kernel[(blocks,)](
                    *tensors,
                    lr if tensor_lr else None,
                    grad_scale,
                    found_inf,
                    *metadata,
                    *scalars,
                    DOUBLE=double,
                    TENSOR_LR=tensor_lr,
                    AMSGRAD=amsgrad,
                    MAXIMIZE=maximize,
                    HAS_GRAD_SCALE=grad_scale is not None,
                    HAS_FOUND_INF=found_inf is not None,
                    BLOCK_SIZE=block_size,
                    enable_fp_fusion=False,
                )
    return None


def _fused_adamw_(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    """Update AdamW parameters and moments in place; steps are read-only."""
    logger.debug("GEMS _FUSED_ADAMW_")
    return _fused_adamw_run(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _fused_adamw__tensor_lr(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    """AdamW tensor-learning-rate overload, with device-side scalar reads."""
    logger.debug("GEMS _FUSED_ADAMW__TENSOR_LR")
    if not isinstance(lr, torch.Tensor):
        raise TypeError("_fused_adamw_.tensor_lr requires a Tensor lr")
    return _fused_adamw_run(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
