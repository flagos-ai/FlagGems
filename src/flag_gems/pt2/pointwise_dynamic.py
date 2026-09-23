# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Transparent PT2 adapters for generated ``pointwise_dynamic`` kernels.

The Python control plane and tensor execution plane are deliberately split:

* a :class:`PointwiseFamilySpec` binds one existing FlagGems scalar/kernel
  generator to structural input metadata;
* :func:`materialize_pointwise_plan` runs rank code generation and imports the
  generated module before Dynamo starts;
* eager execution keeps the generated wrapper/``LibEntry`` path;
* compiled execution launches the exact same generated ``JITFunction`` through
  ``torch.library.triton_op`` and ``torch.library.wrap_triton``.

Plans contain no Tensor, data pointer, output allocation, token count, grid, or
shape value. Rank, dtype, layout family, and guarded GELU approximation select
an immutable plan; concrete shapes and launch parameters remain symbolic or
runtime-specialized after the Dynamo boundary.

This adapter intentionally supports the common inference subset used by the
vLLM activation backend: one output, tensor-only inputs, equal-dtype elementwise
operands, and optional scalar-tensor broadcasts. Unsupported pointwise schemas
are rejected while materializing instead of silently dropping eager promotion,
mutation, autotune, or wrapper semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import triton

from flag_gems.fused.gelu_and_mul import gelu_none_and_mul_kernel as _gelu_none_source
from flag_gems.fused.gelu_and_mul import gelu_tanh_and_mul_kernel as _gelu_tanh_source
from flag_gems.fused.silu_and_mul import silu_and_mul_kernel as _silu_source
from flag_gems.fused.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp_kernel as _silu_clamp_source,
)
from flag_gems.pt2.manifest import CompileKind, CompileOpSpec, register_compile_spec
from flag_gems.utils.codegen_config_utils import get_heuristics_for_num_warps_fn
from flag_gems.utils.pointwise_dynamic import (
    PointwiseDynamicFunction,
    PointwiseKernelMaterialization,
)

_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)
_SUPPORTED_RANKS = (1, 2, 3)
_SUPPORTED_LAYOUTS = ("contiguous_c", "split_last_dim_c")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class PointwiseFamilySpec:
    """Graph-independent description of one existing generated kernel family."""

    name: str
    source_pointwise: PointwiseDynamicFunction
    num_inputs: int
    scalar_input_indices: tuple[int, ...] = ()

    @property
    def primary_input_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i in range(self.num_inputs) if i not in self.scalar_input_indices
        )


@dataclass(frozen=True)
class PointwisePlan:
    """Tensor-free specialization plan for one generated kernel family."""

    token: int
    op_name: str
    ndim: int
    kernel_ndim: int
    dtype: torch.dtype
    layout_class: str
    num_inputs: int
    scalar_input_indices: tuple[int, ...]
    materialization: PointwiseKernelMaterialization
    num_warps_policy: Callable[[int], int]
    tensor_stride_order: tuple[int, ...]
    scalar_stride_order: tuple[int, ...]

    @property
    def jit_function(self):
        return self.materialization.jit_function

    @property
    def primary_input_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i in range(self.num_inputs) if i not in self.scalar_input_indices
        )


_FAMILIES: dict[str, PointwiseFamilySpec] = {}
_PLANS_BY_KEY: dict[tuple[str, int, torch.dtype, str], PointwisePlan] = {}
_PLANS_BY_TOKEN: dict[int, PointwisePlan] = {}
_NEXT_PLAN_TOKEN = 0


def register_pointwise_family(spec: PointwiseFamilySpec) -> PointwiseFamilySpec:
    """Register structural facts without generating or launching a kernel."""

    if torch.compiler.is_compiling():
        raise RuntimeError("pointwise family registration is forbidden inside Dynamo")
    if not isinstance(spec.source_pointwise, PointwiseDynamicFunction):
        raise TypeError(f"{spec.name!r} is not a PointwiseDynamicFunction")
    if spec.num_inputs < 1:
        raise ValueError("pointwise family must have at least one tensor input")

    scalar_indices = tuple(sorted(set(spec.scalar_input_indices)))
    if scalar_indices != spec.scalar_input_indices:
        raise ValueError("scalar_input_indices must be sorted and unique")
    if any(i < 0 or i >= spec.num_inputs for i in scalar_indices):
        raise ValueError(f"invalid scalar input index for {spec.name!r}")
    if len(scalar_indices) == spec.num_inputs:
        raise ValueError("pointwise family requires a non-scalar primary tensor")

    schema = spec.source_pointwise.fx
    if (
        schema.num_input_tensors() != spec.num_inputs
        or schema.num_non_tensor_args() != 0
        or schema.num_output_tensors() != 1
    ):
        raise RuntimeError(
            f"Unsupported PT2 pointwise schema for {spec.name!r}: {schema}. "
            "This adapter requires tensor-only inputs and exactly one output."
        )

    previous = _FAMILIES.get(spec.name)
    if previous is not None and previous != spec:
        raise RuntimeError(f"Conflicting pointwise family {spec.name!r}")
    _FAMILIES[spec.name] = spec
    return spec


SILU_AND_MUL_FAMILY = register_pointwise_family(
    PointwiseFamilySpec("silu_and_mul", _silu_source, num_inputs=2)
)
GELU_NONE_AND_MUL_FAMILY = register_pointwise_family(
    PointwiseFamilySpec("gelu_and_mul.none", _gelu_none_source, num_inputs=2)
)
GELU_TANH_AND_MUL_FAMILY = register_pointwise_family(
    PointwiseFamilySpec("gelu_and_mul.tanh", _gelu_tanh_source, num_inputs=2)
)
SILU_AND_MUL_WITH_CLAMP_FAMILY = register_pointwise_family(
    PointwiseFamilySpec(
        "silu_and_mul_with_clamp",
        _silu_clamp_source,
        num_inputs=3,
        scalar_input_indices=(2,),
    )
)

ACTIVATION_POINTWISE_FAMILIES = (
    SILU_AND_MUL_FAMILY.name,
    GELU_NONE_AND_MUL_FAMILY.name,
    GELU_TANH_AND_MUL_FAMILY.name,
    SILU_AND_MUL_WITH_CLAMP_FAMILY.name,
)


def _family(name: str) -> PointwiseFamilySpec:
    try:
        return _FAMILIES[name]
    except KeyError as exc:
        raise KeyError(f"unknown pointwise family: {name!r}") from exc


def materialize_pointwise_plan(
    op_name: str,
    *,
    ndim: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    layout_class: str = "split_last_dim_c",
) -> PointwisePlan:
    """Materialize structural codegen state outside a Dynamo graph."""

    global _NEXT_PLAN_TOKEN

    if torch.compiler.is_compiling():
        raise RuntimeError("pointwise plan materialization is forbidden inside Dynamo")
    if ndim not in _SUPPORTED_RANKS:
        raise ValueError(f"supported ranks are {_SUPPORTED_RANKS}, got rank {ndim}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported pointwise dtype: {dtype}")
    if layout_class not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported pointwise layout class: {layout_class!r}")

    family = _family(op_name)
    key = (family.name, ndim, dtype, layout_class)
    cached = _PLANS_BY_KEY.get(key)
    if cached is not None:
        return cached

    # PointwiseDynamic.prepare_args collapses equal-shape C-contiguous tensor
    # operands to one physical task dimension.  Families with a scalar-tensor
    # broadcast cannot take that fast path because not all tensor shapes match.
    # Preserve that exact eager choice instead of merely using the input rank.
    kernel_ndim = (
        1
        if layout_class == "contiguous_c" and not family.scalar_input_indices
        else ndim
    )
    generated = family.source_pointwise.materialize(kernel_ndim)
    if generated.runtime_chain != ("LibEntry", "JITFunction"):
        raise RuntimeError(
            "Transparent pointwise PT2 only supports the generated "
            "LibEntry(JITFunction) chain; refusing to drop tuner/heuristic "
            f"semantics from {generated.runtime_chain!r}"
        )

    plan = PointwisePlan(
        token=_NEXT_PLAN_TOKEN,
        op_name=family.name,
        ndim=ndim,
        kernel_ndim=kernel_ndim,
        dtype=dtype,
        layout_class=layout_class,
        num_inputs=family.num_inputs,
        scalar_input_indices=family.scalar_input_indices,
        materialization=generated,
        num_warps_policy=get_heuristics_for_num_warps_fn(),
        tensor_stride_order=tuple(reversed(range(kernel_ndim))),
        scalar_stride_order=tuple(range(kernel_ndim)),
    )
    _NEXT_PLAN_TOKEN += 1
    _PLANS_BY_KEY[key] = plan
    _PLANS_BY_TOKEN[plan.token] = plan
    return plan


def materialize_pointwise_family_plans(
    op_names: Iterable[str],
    *,
    ranks: Iterable[int] = (2,),
    dtypes: Iterable[torch.dtype] = _SUPPORTED_DTYPES,
    layout_classes: Iterable[str] = _SUPPORTED_LAYOUTS,
) -> tuple[PointwisePlan, ...]:
    """Freeze a Cartesian product of structural plans before Dynamo capture."""

    plans = []
    for op_name in op_names:
        for ndim in ranks:
            for dtype in dtypes:
                for layout_class in layout_classes:
                    plans.append(
                        materialize_pointwise_plan(
                            op_name,
                            ndim=ndim,
                            dtype=dtype,
                            layout_class=layout_class,
                        )
                    )
    return tuple(plans)


def materialize_silu_and_mul_plan(**kwargs) -> PointwisePlan:
    return materialize_pointwise_plan(SILU_AND_MUL_FAMILY.name, **kwargs)


def _gelu_family(approximate: str) -> PointwiseFamilySpec:
    if approximate == "none":
        return GELU_NONE_AND_MUL_FAMILY
    if approximate == "tanh":
        return GELU_TANH_AND_MUL_FAMILY
    raise ValueError(f"Invalid approximate value: {approximate}")


def materialize_gelu_and_mul_plan(
    *, approximate: str = "none", **kwargs
) -> PointwisePlan:
    return materialize_pointwise_plan(_gelu_family(approximate).name, **kwargs)


def materialize_silu_and_mul_with_clamp_plan(**kwargs) -> PointwisePlan:
    return materialize_pointwise_plan(SILU_AND_MUL_WITH_CLAMP_FAMILY.name, **kwargs)


def materialized_pointwise_plans(
    op_name: str | None = None,
) -> tuple[PointwisePlan, ...]:
    """Return a stable diagnostic snapshot without exposing mutable caches."""

    plans = tuple(_PLANS_BY_TOKEN[token] for token in sorted(_PLANS_BY_TOKEN))
    if op_name is None:
        return plans
    return tuple(plan for plan in plans if plan.op_name == op_name)


def _layout_class(family: PointwiseFamilySpec, inputs: tuple[torch.Tensor, ...]) -> str:
    primary = tuple(inputs[i] for i in family.primary_input_indices)
    if all(tensor.is_contiguous() for tensor in primary):
        return "contiguous_c"
    return "split_last_dim_c"


def _resolve_plan_token(op_name: str, inputs: tuple[torch.Tensor, ...]) -> int:
    """Resolve already-materialized metadata; never codegen on a cache miss."""

    family = _family(op_name)
    if len(inputs) != family.num_inputs:
        raise RuntimeError(
            f"{family.name!r} requires {family.num_inputs} inputs, got {len(inputs)}"
        )
    reference = inputs[family.primary_input_indices[0]]
    for tensor in inputs:
        if tensor.dtype != reference.dtype:
            raise RuntimeError(f"{family.name!r} requires one input dtype")
        if tensor.device != reference.device:
            raise RuntimeError(f"{family.name!r} requires one input device")

    key = (
        family.name,
        reference.ndim,
        reference.dtype,
        _layout_class(family, inputs),
    )
    plan = _PLANS_BY_KEY.get(key)
    if plan is None:
        raise RuntimeError(
            "No materialized pointwise plan for "
            f"op={key[0]!r}, rank={key[1]}, dtype={key[2]}, layout={key[3]!r}. "
            "Materialize it outside Dynamo before compiling this specialization."
        )
    return plan.token


def _check_contract(plan: PointwisePlan, inputs: tuple[torch.Tensor, ...]) -> None:
    torch._check(len(inputs) == plan.num_inputs)
    reference = inputs[plan.primary_input_indices[0]]
    torch._check(reference.ndim == plan.ndim)

    for index, tensor in enumerate(inputs):
        torch._check(tensor.dtype == plan.dtype)
        torch._check(tensor.device == reference.device)
        if index in plan.scalar_input_indices:
            torch._check(tensor.numel() == 1)
            continue
        torch._check(tensor.ndim == plan.ndim)
        for axis in range(plan.ndim):
            torch._check(tensor.shape[axis] == reference.shape[axis])
        if plan.layout_class == "contiguous_c":
            torch._check(tensor.is_contiguous())
        else:
            torch._check(not tensor.is_contiguous())
            torch._check(tensor.stride(plan.ndim - 1) == 1)
            for axis in range(plan.ndim - 1):
                torch._check(tensor.stride(axis) >= tensor.stride(axis + 1))


def _task_shape(plan: PointwisePlan, out: torch.Tensor):
    if plan.kernel_ndim == plan.ndim:
        return out.shape
    torch._check(plan.kernel_ndim == 1)
    return (out.numel(),)


def _runtime_strides(plan: PointwisePlan, tensor: torch.Tensor, *, scalar: bool):
    if scalar:
        return (0,) * plan.kernel_ndim
    if plan.kernel_ndim == plan.ndim:
        return tensor.stride()
    torch._check(plan.kernel_ndim == 1)
    return (1,)


def _partition(plan: PointwisePlan, out: torch.Tensor):
    """PT2-safe transcription of the generated wrapper's launch policy."""

    shape = _task_shape(plan, out)
    num_tasks = out.numel()
    if num_tasks == 0:
        return None
    if plan.materialization.prefer_block_pointer:
        # Eager prepare_args selects a non-block-pointer ABI for larger tensors.
        # That ABI must be materialized outside Dynamo as a different plan.
        torch._check(num_tasks <= 2_147_483_647)

    if plan.materialization.prefer_1d_tile:
        tile_size = min(
            plan.materialization.max_tile_size,
            triton.next_power_of_2(num_tasks),
        )
        tile_sizes = (tile_size,)
        num_tiles = triton.cdiv(num_tasks, tile_size)
    else:
        remaining = plan.materialization.max_tile_size
        reversed_tiles = []
        for axis in reversed(range(plan.kernel_ndim)):
            tile_size = min(remaining, triton.next_power_of_2(shape[axis]))
            reversed_tiles.append(tile_size)
            remaining = max(1, remaining // tile_size)
        tile_sizes = tuple(reversed(reversed_tiles))
        num_tiles = 1
        for size, tile_size in zip(shape, tile_sizes):
            num_tiles *= triton.cdiv(size, tile_size)

    num_ctas = min(plan.materialization.max_grid_size[0], num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    tile_volume = 1
    for tile_size in tile_sizes:
        tile_volume *= tile_size
    num_warps = plan.num_warps_policy(tile_volume)
    return (
        (num_ctas, 1, 1),
        num_tasks,
        tiles_per_cta,
        tile_sizes,
        tiles_per_cta == 1,
        num_warps,
    )


def _launch_plan(
    plan: PointwisePlan,
    inputs: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    launch,
) -> None:
    grid, num_tasks, tiles_per_cta, tiles, one_tile, num_warps = launch
    wrapped = torch.library.wrap_triton(plan.jit_function)
    args = [*inputs, out]
    for index, tensor in enumerate(inputs):
        is_scalar = index in plan.scalar_input_indices
        args.extend(_runtime_strides(plan, tensor, scalar=is_scalar))
        if is_scalar:
            stride_order = plan.scalar_stride_order
        else:
            stride_order = plan.tensor_stride_order
        if plan.materialization.prefer_block_pointer:
            args.extend(stride_order)

    args.extend(_runtime_strides(plan, out, scalar=False))
    if plan.materialization.prefer_block_pointer:
        args.extend(plan.tensor_stride_order)
    args.extend(_task_shape(plan, out))
    args.append(num_tasks)

    kwargs = {
        "tiles_per_cta": tiles_per_cta,
        "one_tile_per_cta": one_tile,
        "num_warps": num_warps,
    }
    if plan.materialization.prefer_1d_tile:
        kwargs["tile_size"] = tiles[0]
    else:
        for axis, tile_size in enumerate(tiles):
            kwargs[f"tile_size{axis}"] = tile_size
    wrapped[grid](*args, **kwargs)


def _execute_plan(inputs: tuple[torch.Tensor, ...], plan_token: int) -> torch.Tensor:
    plan = _PLANS_BY_TOKEN[plan_token]
    _check_contract(plan, inputs)
    reference = inputs[plan.primary_input_indices[0]]
    out = torch.empty_like(reference)
    launch = _partition(plan, out)
    if launch is not None:
        _launch_plan(plan, inputs, out, launch)
    return out


if _HAS_TRITON_OP:

    @torch.library.triton_op("flag_gems_pt2::silu_and_mul_pointwise", mutates_args={})
    def _silu_and_mul_pointwise_op(
        gate: torch.Tensor, up: torch.Tensor, plan_token: int
    ) -> torch.Tensor:
        return _execute_plan((gate, up), plan_token)

    @torch.library.triton_op("flag_gems_pt2::gelu_and_mul_pointwise", mutates_args={})
    def _gelu_and_mul_pointwise_op(
        gate: torch.Tensor, up: torch.Tensor, plan_token: int
    ) -> torch.Tensor:
        return _execute_plan((gate, up), plan_token)

    @torch.library.triton_op(
        "flag_gems_pt2::silu_and_mul_with_clamp_pointwise", mutates_args={}
    )
    def _silu_and_mul_with_clamp_pointwise_op(
        gate: torch.Tensor,
        up: torch.Tensor,
        limit: torch.Tensor,
        plan_token: int,
    ) -> torch.Tensor:
        return _execute_plan((gate, up, limit), plan_token)

else:
    _silu_and_mul_pointwise_op = None
    _gelu_and_mul_pointwise_op = None
    _silu_and_mul_with_clamp_pointwise_op = None


_POINTWISE_REQUIRES = (
    "PointwiseDynamicFunction.materialize",
    "torch.library.triton_op",
    "torch.library.wrap_triton",
)

SILU_AND_MUL_POINTWISE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::silu_and_mul_pointwise",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.silu_and_mul.silu_and_mul_kernel"
            ".materialize(ndim).jit_function"
        ),
        dynamic_dims=("n_tokens",),
        requires=_POINTWISE_REQUIRES,
    )
)

GELU_AND_MUL_POINTWISE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::gelu_and_mul_pointwise",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "guarded approximate selects flag_gems.fused.gelu_and_mul."
            "gelu_{none,tanh}_and_mul_kernel.materialize(ndim).jit_function"
        ),
        dynamic_dims=("n_tokens",),
        requires=_POINTWISE_REQUIRES,
    )
)

SILU_AND_MUL_WITH_CLAMP_POINTWISE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::silu_and_mul_with_clamp_pointwise",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.silu_and_mul_with_clamp."
            "silu_and_mul_with_clamp_kernel.materialize(ndim).jit_function"
        ),
        dynamic_dims=("n_tokens",),
        requires=_POINTWISE_REQUIRES,
    )
)


def _missing_triton_op() -> RuntimeError:
    return RuntimeError(
        "This Torch build lacks triton_op/wrap_triton; the transparent "
        "pointwise PT2 contract is unavailable"
    )


def silu_and_mul_pointwise(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Run the original generated SiLU-and-multiply kernel."""

    if torch.compiler.is_compiling():
        if _silu_and_mul_pointwise_op is None:
            raise _missing_triton_op()
        plan_token = _resolve_plan_token(SILU_AND_MUL_FAMILY.name, (gate, up))
        return _silu_and_mul_pointwise_op(gate, up, plan_token)
    return SILU_AND_MUL_FAMILY.source_pointwise(gate, up)


def gelu_and_mul_pointwise(
    gate: torch.Tensor, up: torch.Tensor, approximate: str = "none"
) -> torch.Tensor:
    """Run the guarded original GELU-none or GELU-tanh generated kernel."""

    family = _gelu_family(approximate)
    if torch.compiler.is_compiling():
        if _gelu_and_mul_pointwise_op is None:
            raise _missing_triton_op()
        plan_token = _resolve_plan_token(family.name, (gate, up))
        return _gelu_and_mul_pointwise_op(gate, up, plan_token)
    return family.source_pointwise(gate, up)


def silu_and_mul_with_clamp_pointwise(
    gate: torch.Tensor, up: torch.Tensor, limit: torch.Tensor
) -> torch.Tensor:
    """Run the original generated clamped SiLU-and-multiply kernel."""

    inputs = (gate, up, limit)
    if torch.compiler.is_compiling():
        if _silu_and_mul_with_clamp_pointwise_op is None:
            raise _missing_triton_op()
        plan_token = _resolve_plan_token(SILU_AND_MUL_WITH_CLAMP_FAMILY.name, inputs)
        return _silu_and_mul_with_clamp_pointwise_op(gate, up, limit, plan_token)
    return SILU_AND_MUL_WITH_CLAMP_FAMILY.source_pointwise(*inputs)


__all__ = [
    "ACTIVATION_POINTWISE_FAMILIES",
    "GELU_AND_MUL_POINTWISE_SPEC",
    "GELU_NONE_AND_MUL_FAMILY",
    "GELU_TANH_AND_MUL_FAMILY",
    "PointwiseFamilySpec",
    "PointwisePlan",
    "SILU_AND_MUL_FAMILY",
    "SILU_AND_MUL_POINTWISE_SPEC",
    "SILU_AND_MUL_WITH_CLAMP_FAMILY",
    "SILU_AND_MUL_WITH_CLAMP_POINTWISE_SPEC",
    "gelu_and_mul_pointwise",
    "materialize_gelu_and_mul_plan",
    "materialize_pointwise_family_plans",
    "materialize_pointwise_plan",
    "materialize_silu_and_mul_plan",
    "materialize_silu_and_mul_with_clamp_plan",
    "materialized_pointwise_plans",
    "register_pointwise_family",
    "silu_and_mul_pointwise",
    "silu_and_mul_with_clamp_pointwise",
]
