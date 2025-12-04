import builtins
import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops import min_dim as base_min_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

MinOut = namedtuple("min", ["values", "indices"])

# Expanded coverage favors smaller column tiles and more warps for tall shapes.
NAIVE_REDUCTION_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=16, num_stages=1),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=16, num_stages=2),
]


def _prune_reduction_configs(configs, nargs, **meta):
    n = meta.get("N", None)
    if n is None:
        n = nargs["N"]
    max_block_n = 64 if n <= 128 else 256
    return [cfg for cfg in configs if cfg.kwargs["BLOCK_N"] <= max_block_n]


def _flatten_dim(shape, dim):
    dim = dim % len(shape)
    n = shape[dim]
    inner = math.prod(shape[dim + 1 :]) if dim + 1 < len(shape) else 1
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    return dim, n, inner, outer


@libentry()
@triton.jit
def min_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


@libentry()
@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


@libentry()
@triton.autotune(
    configs=NAIVE_REDUCTION_CONFIGS,
    key=["M", "N"],
    warmup=8,
    rep=40,
    prune_configs_by={"early_config_prune": _prune_reduction_configs},
)
@triton.jit
def min_kernel(
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
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int32, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        n_offset = n_offset.to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value, cache_modifier=".cg")
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        local_argmin = local_argmin.to(tl.int32)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(
            update, (start_n + local_argmin).to(tl.int32), argmin_values
        )

    out_value_ptrs = out_value + rows
    out_index_ptrs = out_index + rows
    tl.store(out_value_ptrs, min_values, mask=row_mask)
    tl.store(out_index_ptrs, argmin_values, mask=row_mask)


def min(inp):
    logger.debug("GEMS_MTHREADS MIN")
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = builtins.min(block_size * 4, 4096, triton.next_power_of_2(M))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    num_warps_block = builtins.min(8, max(1, block_size // 128))
    num_warps_mid = builtins.min(8, max(1, block_mid // 128))

    with torch_device_fn.device(inp.device):
        min_kernel_1[(mid_size, 1, 1)](
            inp, mid, M, block_size, num_warps=num_warps_block, num_stages=2
        )
        min_kernel_2[(1, 1, 1)](
            mid, out, mid_size, block_mid, num_warps=num_warps_mid, num_stages=2
        )
    return out


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS MIN DIM")
    assert dim is not None, "dim must be specified"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    if not inp.is_contiguous():
        # Fall back to the generic implementation (handles arbitrary strides).
        return base_min_dim(inp, dim=dim, keepdim=keepdim)

    shape = list(inp.shape)
    dim, N, inner, outer = _flatten_dim(shape, dim)
    M = outer * inner
    stride = inp.stride()
    stride_reduce = stride[dim]
    stride_outer = stride_reduce * N

    out_value = torch.empty((M,), dtype=inp.dtype, device=inp.device)
    out_index = torch.empty((M,), dtype=torch.int32, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        min_kernel[grid](
            inp,
            out_value,
            out_index,
            M,
            N,
            max(inner, 1),
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
    return MinOut(values=out_value, indices=out_index)


__all__ = ["min", "min_dim"]
