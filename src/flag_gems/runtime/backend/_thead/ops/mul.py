# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mul import (
    _contiguous_configs,
    _dtype_name,
    _mul,
    _mul_,
    _mul_contiguous_tile,
    _prune_contiguous_configs,
    mul_broadcast_get_configs,
    mul_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=_contiguous_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_contiguous_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.heuristics(
    {"EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0}
)
@triton.jit
def mul_contiguous_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
):
    # The host launcher already restricts this path to n_elements < 2**31.
    if dtype == "float16" or dtype == "float32" or dtype == "bfloat16":
        _mul_contiguous_tile(
            x_ptr,
            y_ptr,
            output_ptr,
            n_elements,
            tl.program_id(0),
            BLOCK_SIZE,
            EVEN_SIZE,
        )
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x & y if IS_BOOL else x * y
        tl.store(output_ptr + offsets, out, mask=mask)


def _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        if (
            n_elements < 2**31
            and not torch._C._is_alias_of(output, a_t)
            and not torch._C._is_alias_of(output, b_t)
        ):
            mul_contiguous_kernel[grid](
                a_t,
                b_t,
                output,
                n_elements,
                dtype=_dtype_name(dtype),
                IS_BOOL=dtype is torch.bool,
            )
            return output
        mul_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
        )
    return output


def _launch_2d_broadcast(
    a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        kernel = mul_broadcast_2d_kernel
        launch = {}
        if any(torch._C._is_alias_of(output, x) for x in (a_t, b_t)):
            kernel = kernel.fn.fn
            launch = dict(BLOCK_SIZE=1024, num_warps=4, num_stages=3)
        kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            out_shape[1],
            a_s0=a_stride[0],
            a_s1=a_stride[1],
            b_s0=b_stride[0],
            b_s1=b_stride[1],
            out_s0=out_stride[0],
            out_s1=out_stride[1],
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
            **launch,
        )
    return output


def _scalar_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in (
            (128, 1),
            (256, 1),
            (256, 2),
            (512, 1),
            (512, 2),
            (1024, 1),
            (2048, 1),
            (2048, 2),
        )
    ] + [triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3)]


def _broadcast_configs():
    return _scalar_configs() + mul_broadcast_get_configs()[:2]


def _canonical_stage_configs(configs):
    """Keep stage 1 for equivalent straight-line scalar/broadcast kernels."""

    def key(config):
        return tuple(
            (name, repr(value))
            for name, value in sorted(config.all_kwargs().items())
            if name != "num_stages"
        )

    eligible = [
        config
        for config in configs
        if config.pre_hook is None and getattr(config, "ir_override", None) is None
    ]
    stage_one = {key(config) for config in eligible if config.num_stages == 1}
    return [
        config
        for config in configs
        if config.pre_hook is not None
        or getattr(config, "ir_override", None) is not None
        or not 1 < config.num_stages <= 8
        or key(config) not in stage_one
    ]


def _prune_mul_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    default_signatures = {str(config) for config in _scalar_configs()}
    # Cost Model scores each verified launch geometry once.
    # Expanded/Train still collect the complete declared config space.
    if any(str(config) not in default_signatures for config in configs):
        return (
            _canonical_stage_configs(configs)
            if args.get("dtype") in ("float32", "bfloat16")
            and runtime.resolve_tuning_mode("mul", supports_cost_model=True)
            is runtime.TuningMode.COST_MODEL
            else configs
        )
    if args.get("dtype") == "bfloat16":
        return configs

    if args.get("dtype") != "float32":
        return [
            c
            for c in configs
            if c.num_warps == 4 and c.num_stages == 3 and c.kwargs["BLOCK_SIZE"] == 1024
        ]
    preferred = (
        {(128, 1)}
        if args.get("n_elements", 0) <= 2048
        else (
            {(512, 1), (512, 2), (1024, 1)}
            if args.get("n_elements", 0) >= 65536
            else None
        )
    )
    return [
        c
        for c in configs
        if c.num_stages == 1
        and c.num_warps in (1, 2, 4, 8)
        and (preferred is None or (c.kwargs["BLOCK_SIZE"], c.num_warps) in preferred)
    ]


@libentry()
@libtuner(
    configs=_scalar_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_mul_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="scalar",
    policy="flagtune",
)
@triton.heuristics(
    {
        "EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0,
        "SINGLE_BLOCK": lambda args: args["n_elements"] <= args["BLOCK_SIZE"],
    }
)
@triton.jit
def mul_scalar_kernel(
    x_ptr,
    output_ptr,
    scalar,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    HALF: tl.constexpr = False,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    if not SINGLE_BLOCK:
        offsets += tl.program_id(0) * BLOCK_SIZE
    # Exact tiles need no per-element bounds checks. All launchers use ceildiv
    # grids, so only the final partial tile requires a mask.
    if EVEN_SIZE:
        x = tl.load(x_ptr + offsets)
        value = x * 0.5 if HALF else (x & scalar if IS_BOOL else x * scalar)
        tl.store(output_ptr + offsets, value)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        value = x * 0.5 if HALF else (x & scalar if IS_BOOL else x * scalar)
        tl.store(output_ptr + offsets, value, mask=mask)


def _launch_scalar(tensor, scalar, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        kernel = mul_scalar_kernel
        launch = {}
        if any(torch._C._is_alias_of(output, x) for x in (tensor,)):
            kernel = kernel.fn.fn
            launch = dict(BLOCK_SIZE=1024, num_warps=4, num_stages=3)
        kernel[grid](
            tensor,
            output,
            scalar,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
            HALF=dtype == torch.bfloat16 and scalar == 0.5 and n_elements >= 1048576,
            **launch,
        )
    return output


def _prune_broadcast_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    default_signatures = {str(config) for config in _broadcast_configs()}
    # Cost Model scores each verified launch geometry once.
    # Expanded/Train still collect the complete declared config space.
    if any(str(config) not in default_signatures for config in configs):
        return (
            _canonical_stage_configs(configs)
            if args.get("dtype") in ("float32", "bfloat16")
            and runtime.resolve_tuning_mode("mul", supports_cost_model=True)
            is runtime.TuningMode.COST_MODEL
            else configs
        )

    # Keep BF16's existing pool; the extra stage-3 configs serve other dtypes.
    if args.get("dtype") == "bfloat16":
        signatures = {str(config) for config in _scalar_configs()}
        return [c for c in configs if str(c) in signatures]

    if args.get("dtype") != "float32":
        return [c for c in configs if c.num_warps == 4 and c.num_stages == 3]

    preferred = (
        {(1024, 1), (2048, 1), (2048, 2)}
        if args.get("n_elements", 0) >= 4194304
        else {(512, 1), (512, 2), (1024, 1)}
    )
    return [
        c
        for c in configs
        if c.num_stages == 1 and (c.kwargs["BLOCK_SIZE"], c.num_warps) in preferred
    ]


@libentry()
@libtuner(
    configs=_broadcast_configs(),
    key=["n_elements", "n_cols", "dtype"],
    strategy=["align32", "default", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_broadcast_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul_broadcast_2d",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="broadcast_2d",
    policy="flagtune",
)
@triton.heuristics(
    {
        "EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0,
        "TILED": lambda args: args["n_elements"] >= 16777216,
    }
)
@triton.jit
def mul_broadcast_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    n_cols: tl.constexpr,
    a_s0: tl.constexpr,
    a_s1: tl.constexpr,
    b_s0: tl.constexpr,
    b_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
    TILED: tl.constexpr,
):
    if (
        dtype == "float32"
        and n_cols == 32
        and TILED
        and a_s0 == 1
        and a_s1 == 0
        and b_s0 == 0
        and b_s1 == 1
        and out_s0 == n_cols
        and out_s1 == 1
    ):
        rows = tl.program_id(0) * (BLOCK_SIZE // 32) + tl.arange(0, BLOCK_SIZE // 32)
        cols = tl.arange(0, 32)
        mask = rows < n_elements // 32
        scale = tl.load(a_ptr + rows, mask=mask, other=0)
        vector = tl.load(b_ptr + cols)
        tl.store(
            out_ptr + rows[:, None] * 32 + cols[None, :],
            scale[:, None] * vector[None, :],
            mask[:, None],
        )
        return

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = offsets // n_cols
    col = offsets % n_cols
    if EVEN_SIZE:
        a = tl.load(a_ptr + row * a_s0 + col * a_s1)
        b = tl.load(b_ptr + row * b_s0 + col * b_s1)
        value = a & b if IS_BOOL else a * b
        tl.store(out_ptr + row * out_s0 + col * out_s1, value)
    else:
        mask = offsets < n_elements
        a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask=mask, other=0)
        b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask=mask, other=0)
        value = a & b if IS_BOOL else a * b
        tl.store(out_ptr + row * out_s0 + col * out_s1, value, mask=mask)


# Reuse common validation and fallbacks with this backend's launch policies.
_REAL_LAUNCHERS = (
    _launch_contiguous_tensor_tensor,
    _launch_scalar,
    _launch_2d_broadcast,
)


def mul(A, B, *, out=None):
    return _mul(A, B, out=out, launchers=_REAL_LAUNCHERS)


def mul_(A, B):
    return _mul_(A, B, launchers=_REAL_LAUNCHERS)
