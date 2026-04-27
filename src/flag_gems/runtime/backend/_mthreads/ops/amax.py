import builtins
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.amax import amax as base_amax
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

AMAX_REDUCTION_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
]


def _prune_reduction_configs(configs, nargs, **meta):
    n = meta.get("N", nargs["N"])
    if n <= 128:
        max_block_n = 128
    elif n <= 2048:
        max_block_n = 256
    elif n <= 8192:
        max_block_n = 512
    else:
        max_block_n = 1024
    return [cfg for cfg in configs if cfg.kwargs["BLOCK_N"] <= max_block_n]


def _flatten_dim(shape, dim):
    dim = dim % len(shape)
    n = shape[dim]
    inner = math.prod(shape[dim + 1 :]) if dim + 1 < len(shape) else 1
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    return dim, n, inner, outer


@libentry()
@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    vals = tl.load(inp + offset, mask=mask, other=min_value, cache_modifier=".cg")
    tl.store(mid + pid, tl.max(vals))


@libentry()
@triton.jit
def amax_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    vals = tl.load(mid + offset, mask=mask, other=min_value)
    tl.store(out, tl.max(vals))


@libentry()
@triton.jit
def amax_kernel_small(
    inp,
    out_value,
    M,
    N,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    row_mask = row < M
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    stride_outer = tl.full((), STRIDE_OUTER, tl.int64)
    stride_reduce = tl.full((), STRIDE_REDUCE, tl.int64)
    offsets = row.to(tl.int64) * stride_outer + cols.to(tl.int64) * stride_reduce

    dtype = inp.type.element_ty
    acc_type = tl.float32 if (dtype is tl.float16 or dtype is tl.bfloat16) else dtype
    min_value = get_dtype_min(dtype)
    vals = tl.load(inp + offsets, mask=row_mask & col_mask, other=min_value).to(
        acc_type
    )
    row_max = tl.max(vals, axis=0)
    tl.store(out_value + row, row_max, mask=row_mask)


@libentry()
@triton.autotune(
    configs=AMAX_REDUCTION_CONFIGS,
    key=["M", "N"],
    warmup=8,
    rep=40,
    prune_configs_by={"early_config_prune": _prune_reduction_configs},
)
@triton.jit
def amax_kernel(
    inp,
    out_value,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = rows.to(tl.int64)
    row_mask = rows < M

    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    dtype = inp.type.element_ty
    acc_type = tl.float32 if (dtype is tl.float16 or dtype is tl.bfloat16) else dtype
    min_value = get_dtype_min(dtype)
    max_values = tl.full([BLOCK_M], dtype=acc_type, value=min_value)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        n_offset = n_offset.to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value, cache_modifier=".cg")
        inp_vals = inp_vals.to(acc_type)
        local_max = tl.max(inp_vals, axis=1)
        max_values = tl.maximum(max_values, local_max)

    out_value_ptrs = out_value + rows
    tl.store(out_value_ptrs, max_values, mask=row_mask)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS AMAX")

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        # Global reduction
        if not inp.is_contiguous():
            inp = inp.contiguous()
        if inp.numel() == 0:
            return base_amax(inp, dim=dim, keepdim=keepdim)

        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        block_size = builtins.min(block_size * 4, 4096, triton.next_power_of_2(M))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        num_warps_block = builtins.min(8, builtins.max(1, block_size // 128))
        num_warps_mid = builtins.min(8, builtins.max(1, block_mid // 128))

        with torch_device_fn.device(inp.device):
            amax_kernel_1[(mid_size, 1, 1)](
                inp, mid, M, block_size, num_warps=num_warps_block, num_stages=2
            )
            amax_kernel_2[(1, 1, 1)](
                mid, out, mid_size, block_mid, num_warps=num_warps_mid, num_stages=2
            )
        return out
    else:
        # Dimension-specific reduction
        if isinstance(dim, int):
            dim = [dim]

        # For multi-dim reduction, use base implementation
        if len(dim) > 1:
            return base_amax(inp, dim=dim, keepdim=keepdim)

        dim_val = dim[0]
        assert dim_val >= -inp.ndim and dim_val < inp.ndim, "Invalid dim"
        dim_val = dim_val % inp.ndim

        if not inp.is_contiguous():
            return base_amax(inp, dim=dim, keepdim=keepdim)

        shape = list(inp.shape)
        dim_val, N, inner, outer = _flatten_dim(shape, dim_val)
        M = outer * inner
        stride = inp.stride()
        stride_reduce = stride[dim_val]
        stride_outer = stride_reduce * N

        out_value = torch.empty((M,), dtype=inp.dtype, device=inp.device)

        if inner == 1 and N <= 128:
            block_n = builtins.min(triton.next_power_of_2(N), 128)
            grid = (triton.cdiv(M, 1),)
            with torch_device_fn.device(inp.device):
                amax_kernel_small[grid](
                    inp,
                    out_value,
                    M,
                    N,
                    stride_outer,
                    stride_reduce,
                    block_n,
                    num_warps=1,
                    num_stages=1,
                )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
            with torch_device_fn.device(inp.device):
                amax_kernel[grid](
                    inp,
                    out_value,
                    M,
                    N,
                    builtins.max(inner, 1),
                    stride_outer,
                    stride_reduce,
                )

        out_shape = shape.copy()
        out_shape[dim_val] = 1
        out_value = out_value.view(out_shape)
        if not keepdim:
            out_value = torch.squeeze(out_value, dim_val)

        return out_value


__all__ = ["amax"]
