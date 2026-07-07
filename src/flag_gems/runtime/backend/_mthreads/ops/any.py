import logging
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


def _flatten_dim(shape: Sequence[int], dim: int):
    dim = dim % len(shape)
    n = shape[dim]
    inner = math.prod(shape[dim + 1 :]) if dim + 1 < len(shape) else 1
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    return dim, n, inner, outer


# Favor smaller tiles to keep occupancy high on MUSA; wide tiles trigger register
# pressure and hurt latency for large reductions.
def _select_reduction_config(m_rows: int, n_cols: int):
    block_n = min(256, max(64, 1 << int(math.ceil(math.log2(n_cols)))))
    max_block_m = 1 << int(math.floor(math.log2(max(1, m_rows))))
    block_m = min(32, max_block_m)
    num_warps = 8 if block_n >= 256 else 4
    return block_m, block_n, num_warps


@libentry()
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    row_mask = rows < M
    row_offsets = rows * N

    acc = tl.zeros((BLOCK_M,), dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        active = acc == 0
        mask = row_mask[:, None] & col_mask[None, :] & active[:, None]
        vals = tl.load(inp + row_offsets[:, None] + cols[None, :], mask=mask, other=0.0)
        block_any = tl.max(vals != 0, axis=1).to(tl.int1)
        acc = acc | block_any
    tl.store(out + rows, acc, mask=row_mask)


@triton.jit
def any_kernel_dim_strided(
    inp,
    out,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = rows.to(tl.int64)
    row_mask = rows < M

    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    acc = tl.zeros((BLOCK_M,), dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        cols = cols.to(tl.int64)
        col_mask = cols < N
        active = acc == 0
        mask = row_mask[:, None] & col_mask[None, :] & active[:, None]
        vals = tl.load(
            base_ptr[:, None] + cols[None, :] * STRIDE_REDUCE, mask=mask, other=0.0
        )
        block_any = tl.max(vals != 0, axis=1).to(tl.int1)
        acc = acc | block_any
    tl.store(out + rows, acc, mask=row_mask)


@libentry()
@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    any_val = tl.max(inp_val != 0, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, any_val)


@libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
    any_val = tl.max(mid_val, axis=0)
    tl.store(out, any_val)


def any(inp):
    logger.debug("GEMS_MTHREADS ANY")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    block_size = min(block_size * 2, 4096, triton.next_power_of_2(n_elements))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    num_warps_block = min(8, max(1, block_size // 128))
    num_warps_mid = min(8, max(1, block_mid // 128))

    with torch_device_fn.device(inp.device):
        any_kernel_1[(mid_size, 1)](
            inp,
            mid,
            n_elements,
            mid_size,
            block_size,
            num_warps=num_warps_block,
            num_stages=2,
        )
        any_kernel_2[(1, 1)](
            mid,
            out,
            mid_size,
            block_mid,
            num_warps=num_warps_mid,
            num_stages=2,
        )

    return out


def triton_any_dim_strided(
    inp: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    dim = dim % inp.ndim
    shape = list(inp.shape)
    dim, n, inner, outer = _flatten_dim(shape, dim)
    m = outer * inner

    stride = inp.stride()
    stride_reduce = stride[dim]
    stride_outer = stride_reduce * n

    out_flat = torch.empty((m,), dtype=torch.bool, device=inp.device)
    block_m, block_n, num_warps = _select_reduction_config(m, n)
    grid = (triton.cdiv(m, block_m),)
    with torch_device_fn.device(inp.device):
        any_kernel_dim_strided[grid](
            inp,
            out_flat,
            m,
            n,
            inner,
            stride_outer,
            stride_reduce,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

    shape[dim] = 1
    out = out_flat.view(shape)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ANY DIM")
    # shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        return out
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    return triton_any_dim_strided(inp, dim, keepdim=keepdim)


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ANY DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)

    dims = [d % inp.ndim for d in dim]
    dims = sorted(set(dims))
    out = inp
    for d in dims:
        out = triton_any_dim_strided(out, d, keepdim=True)
    if not keepdim:
        for d in reversed(dims):
            out = out.squeeze(dim=d)
    return out


__all__ = ["any", "any_dim", "any_dims"]
