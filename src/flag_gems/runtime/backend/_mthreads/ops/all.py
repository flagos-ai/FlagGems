import logging
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

NAIVE_REDUCTION_CONFIGS = [
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8),
]


@triton.jit
def reduce_all(a, b):
    return a and b


@triton.autotune(configs=NAIVE_REDUCTION_CONFIGS, key=["M", "N"])
@triton.jit
def all_kernel_dim_strided(
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
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = rows.to(tl.int64)
    row_mask = rows < M

    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    acc = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        cols = cols.to(tl.int64)
        col_mask = cols < N
        mask = row_mask[:, None] and col_mask[None, :]
        vals = tl.load(
            base_ptr[:, None] + cols[None, :] * STRIDE_REDUCE, mask, other=1.0
        )
        acc = acc and (vals != 0)
    all_val = tl.reduce(acc, axis=1, combine_fn=reduce_all)
    tl.store(out + rows, all_val, mask=row_mask)


def _flatten_dim(shape: Sequence[int], dim: int):
    dim = dim % len(shape)
    n = shape[dim]
    inner = math.prod(shape[dim + 1 :]) if dim + 1 < len(shape) else 1
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    return dim, n, inner, outer


def triton_all_dim_strided(
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
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    all_kernel_dim_strided[grid](
        inp,
        out_flat,
        m,
        n,
        inner,
        stride_outer,
        stride_reduce,
    )

    shape[dim] = 1
    out = out_flat.view(shape)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)


@libentry()
@triton.jit
def all_kernel_1(
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
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
    mid_ptr = mid + pid
    tl.store(mid_ptr, all_val)


@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


def all(inp):
    logger.debug("GEMS_MTHREADS ALL")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    block_size = min(block_size * 2, 4096, triton.next_power_of_2(n_elements))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
        all_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ALL DIM")
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        return out

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    with torch_device_fn.device(inp.device):
        return triton_all_dim_strided(inp, dim=dim, keepdim=keepdim)


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ALL DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


__all__ = ["all", "all_dim", "all_dims"]
