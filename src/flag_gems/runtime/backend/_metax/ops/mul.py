# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX mul: scalar kernels and broadcast tiles."""

import copy

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


@triton.jit
def _broadcast_tiled(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    N: tl.constexpr,
    TILE_COLS: tl.constexpr,
    OUTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Layout validation and launch parameters belong to the caller's backend.
    BLOCK_N: tl.constexpr = min(BLOCK_SIZE, TILE_COLS)
    BLOCK_M: tl.constexpr = max(1, BLOCK_SIZE // TILE_COLS)
    COL_TILES: tl.constexpr = tl.cdiv(N, BLOCK_N)
    pid = tl.program_id(0)
    rows = (pid // COL_TILES) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = (pid % COL_TILES) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_elements // N
    col_mask = cols < N
    a = tl.load(a_ptr + rows, row_mask, other=0)
    if OUTER:
        b = tl.load(b_ptr + cols, col_mask, other=0)
        value = a[:, None] * b[None, :]
    else:
        b = tl.load(
            b_ptr + rows[:, None] * N + cols[None, :],
            row_mask[:, None] & col_mask[None, :],
            other=0,
        )
        value = a[:, None] * b
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        value,
        row_mask[:, None] & col_mask[None, :],
    )


def scalar_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in ((128, 1), (1024, 1), (2048, 1), (2048, 2), (2048, 4))
    ] + [triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3)]


def broadcast_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in (
            (128, 1),
            (256, 2),
            (512, 2),
            (512, 4),
            (1024, 1),
            (1024, 2),
            (2048, 1),
            (2048, 2),
            (4096, 1),
            (8192, 4),
        )
    ]
    configs.extend(
        triton.Config({"BLOCK_SIZE": block}, num_warps=4, num_stages=3)
        for block in (256, 512, 1024)
    )
    return configs


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


def prune_configs(configs, named_args, **kwargs):
    """Apply measured preferences only to the declared default pool."""
    args = {**named_args, **kwargs}
    defaults = broadcast_configs() if "n_cols" in args else scalar_configs()
    default_signatures = {str(config) for config in defaults}
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

    if args.get("dtype") not in ("float32", "bfloat16"):
        blocks = (256, 512, 1024) if "n_cols" in args else (1024,)
        return [
            c
            for c in configs
            if c.num_warps == 4
            and c.num_stages == 3
            and c.kwargs["BLOCK_SIZE"] in blocks
        ]

    candidates = [
        copy.deepcopy(config)
        for config in configs
        if config.num_warps in (1, 2, 4, 8)
        and (
            config.num_stages == 1
            or (
                config.num_stages == 3
                and config.num_warps == 4
                and config.kwargs["BLOCK_SIZE"] in (256, 512, 1024)
            )
        )
    ]
    cols = args.get("METAX_N_COLS", 0)
    if cols:
        if args["b_s0"] == 0:
            preferred = (
                {(1024, 1, 1), (2048, 2, 1)}
                if args.get("n_elements", 0) <= 1048576
                else {(2048, 1, 1), (8192, 4, 1)}
            )
        else:
            preferred = {(1024, 2, 1), (2048, 1, 1), (2048, 2, 1)}
        candidates = [
            config
            for config in candidates
            if (
                config.kwargs["BLOCK_SIZE"],
                config.num_warps,
                config.num_stages,
            )
            in preferred
        ]
    elif "n_cols" not in args and args.get("dtype") == "float32":
        elements = args.get("n_elements", 0)
        if elements <= 2048:
            preferred = {(128, 1, 1)}
        elif elements <= 131072:
            preferred = {(1024, 1, 1), (2048, 2, 1)}
        elif elements < 1048576:
            preferred = {(2048, 2, 1), (2048, 4, 1)}
        else:
            preferred = {(2048, 1, 1)}
        if preferred:
            candidates = [
                c
                for c in candidates
                if (c.kwargs["BLOCK_SIZE"], c.num_warps, c.num_stages) in preferred
            ]
    elif "n_cols" in args:
        # Small and generic-stride broadcasts retain the original candidates.
        candidates = [
            config
            for config in candidates
            if config.num_stages == 3 and config.num_warps == 4
        ]
    return candidates


@libentry()
@libtuner(
    configs=scalar_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": prune_configs},
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
        "PACKED": lambda args: args["dtype"] == "float32"
        and args["n_elements"] >= 1048576,
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
    PACKED: tl.constexpr,
):
    if PACKED:
        offsets = (
            tl.program_id(0) * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE // 4)[:, None] * 4
            + tl.arange(0, 4)[None, :]
        )
    else:
        offsets = tl.arange(0, BLOCK_SIZE)
        if not SINGLE_BLOCK:
            offsets += tl.program_id(0) * BLOCK_SIZE
    # Exact tiles need no per-element bounds checks. All launchers use ceildiv
    # grids, so only the final partial tile requires a mask.
    if EVEN_SIZE:
        x = tl.load(x_ptr + offsets)
        value = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, value)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        value = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, value, mask=mask)


@libentry()
@libtuner(
    configs=broadcast_configs(),
    key=[
        "n_elements",
        "n_cols",
        "dtype",
        "a_s0",
        "a_s1",
        "b_s0",
        "b_s1",
        "out_s0",
        "out_s1",
        "METAX_N_COLS",
    ],
    strategy=[
        "align32",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
    ],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": prune_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul_broadcast_2d_metax",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="broadcast_2d",
    policy="flagtune",
)
@triton.jit
def mul_broadcast_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    n_cols,
    a_s0: tl.constexpr,
    a_s1: tl.constexpr,
    b_s0: tl.constexpr,
    b_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    METAX_N_COLS: tl.constexpr = 0,
    METAX_TILE_COLS: tl.constexpr = 0,
):
    if METAX_N_COLS:
        _broadcast_tiled(
            a_ptr,
            b_ptr,
            out_ptr,
            n_elements,
            METAX_N_COLS,
            METAX_TILE_COLS,
            b_s0 == 0,
            BLOCK_SIZE,
        )
        return
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = offsets // n_cols
    col = offsets - row * n_cols
    a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask=mask)
    b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask=mask)
    out = a & b if IS_BOOL else a * b
    tl.store(out_ptr + row * out_s0 + col * out_s1, out, mask=mask)


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
            **launch,
        )
    return output


def _launch_2d_broadcast(
    a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    metax_n_cols = 0
    metax_tile_cols = 0
    if dtype in (torch.float32, torch.bfloat16) and output.is_contiguous():
        rows, cols = out_shape
        # Reversed row and outer broadcasts share the same dense tile kernel.
        if b_stride == (1, 0) and a_stride in ((cols, 1), (0, 1)):
            a_t, b_t = (b_t, a_t)
            a_stride, b_stride = (b_stride, a_stride)
        if (
            n_elements < 2**31
            and a_stride == (1, 0)
            and (b_stride in ((cols, 1), (0, 1)))
            and (
                (b_stride[0] == 0 and 1 < cols <= 64)
                or (b_stride[0] != 0 and 32 < rows < 1024 and 1 < cols <= 4096)
            )
        ):
            # These C550 thresholds regress for some shapes on other vendors.
            metax_n_cols = cols
            metax_tile_cols = triton.next_power_of_2(cols)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if metax_n_cols:
        grid = lambda meta: (
            triton.cdiv(out_shape[0], max(1, meta["BLOCK_SIZE"] // metax_tile_cols))
            * triton.cdiv(metax_n_cols, min(meta["BLOCK_SIZE"], metax_tile_cols)),
        )
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
            METAX_N_COLS=metax_n_cols,
            METAX_TILE_COLS=metax_tile_cols,
            **launch,
        )
    return output


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
