import builtins
import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.max import max as base_max
from flag_gems.ops.max import max_dim as base_max_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

MaxOut = namedtuple("max", ["values", "indices"])

MAX_REDUCTION_CONFIGS = [
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
def max_kernel_1(
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
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    vals = tl.load(mid + offset, mask=mask, other=min_value)
    tl.store(out, tl.max(vals))


@libentry()
@triton.jit
def max_kernel_small(
    inp,
    out_value,
    out_index,
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
    row_max, row_argmax = tl.max(
        vals,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    tl.store(out_value + row, row_max, mask=row_mask)
    tl.store(out_index + row, row_argmax.to(tl.int32), mask=row_mask)


@libentry()
@triton.autotune(
    configs=MAX_REDUCTION_CONFIGS,
    key=["M", "N"],
    warmup=8,
    rep=40,
    prune_configs_by={"early_config_prune": _prune_reduction_configs},
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
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
    argmax_values = tl.full([BLOCK_M], dtype=tl.int32, value=0)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        n_offset = n_offset.to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value, cache_modifier=".cg")
        inp_vals = inp_vals.to(acc_type)
        local_max, local_argmax = tl.max(
            inp_vals,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        local_argmax = local_argmax.to(tl.int32)
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(
            update, (start_n + local_argmax).to(tl.int32), argmax_values
        )

    out_value_ptrs = out_value + rows
    out_index_ptrs = out_index + rows
    tl.store(out_value_ptrs, max_values, mask=row_mask)
    tl.store(out_index_ptrs, argmax_values, mask=row_mask)


def max(inp):
    logger.debug("GEMS_MTHREADS MAX")
    if not inp.is_contiguous():
        inp = inp.contiguous()
    if inp.numel() == 0:
        return base_max(inp)

    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = builtins.min(block_size * 4, 4096, triton.next_power_of_2(M))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    num_warps_block = builtins.min(8, builtins.max(1, block_size // 128))
    num_warps_mid = builtins.min(8, builtins.max(1, block_mid // 128))

    with torch_device_fn.device(inp.device):
        max_kernel_1[(mid_size, 1, 1)](
            inp, mid, M, block_size, num_warps=num_warps_block, num_stages=2
        )
        max_kernel_2[(1, 1, 1)](
            mid, out, mid_size, block_mid, num_warps=num_warps_mid, num_stages=2
        )
    return out


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS MAX DIM")
    assert dim is not None, "dim must be specified"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    if not inp.is_contiguous():
        return base_max_dim(inp, dim=dim, keepdim=keepdim)

    shape = list(inp.shape)
    dim, N, inner, outer = _flatten_dim(shape, dim)
    M = outer * inner
    stride = inp.stride()
    stride_reduce = stride[dim]
    stride_outer = stride_reduce * N

    out_value = torch.empty((M,), dtype=inp.dtype, device=inp.device)
    out_index = torch.empty((M,), dtype=torch.int32, device=inp.device)

    if inner == 1 and N <= 128:
        block_n = builtins.min(triton.next_power_of_2(N), 128)
        grid = (triton.cdiv(M, 1),)
        with torch_device_fn.device(inp.device):
            max_kernel_small[grid](
                inp,
                out_value,
                out_index,
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
            max_kernel[grid](
                inp,
                out_value,
                out_index,
                M,
                N,
                builtins.max(inner, 1),
                stride_outer,
                stride_reduce,
            )

    out_shape = shape.copy()
    out_shape[dim] = 1
    out_value = out_value.view(out_shape)
    out_index = out_index.view(out_shape).to(torch.int64)
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    return MaxOut(values=out_value, indices=out_index)


__all__ = ["max", "max_dim"]
