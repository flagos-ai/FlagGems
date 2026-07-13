import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)


def _promote_for_nansum(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        return inp.to(torch.float32)
    if inp.dtype in (torch.float16, torch.bfloat16):
        return inp.to(torch.float32)
    return inp


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["numel"],
)
@triton.jit
def _nan_to_zero_kernel(
    inp_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise NaN to zero conversion for small tensor or dim == 1."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    val = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    val = tl.where(val != val, 0.0, val)
    tl.store(out_ptr + offsets, val, mask=mask)


@libentry()
@triton.jit
def _nansum_parallel_reduce(
    inp_ptr,
    out_ptr,
    data_size,
    BLOCK_SIZE: tl.constexpr,
    FP64: tl.constexpr,
):
    """Stage 1 parallel reduce: each block reduces BLOCK_SIZE contiguous elements."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < data_size

    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x != x, 0.0, x)

    block_sum = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, block_sum)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["data_size"],
)
@triton.jit
def _nansum_final_reduce(
    inp_ptr,
    out_ptr,
    data_size,
    BLOCK_SIZE: tl.constexpr,
    FP64: tl.constexpr,
):
    """Stage 2 final reduce: single block reads entire input, outputs a scalar."""
    if FP64:
        total = tl.zeros((), dtype=tl.float64)
    else:
        total = tl.zeros((), dtype=tl.float32)

    for start in range(0, data_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < data_size
        val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
        val = tl.where(val != val, 0.0, val)
        total += tl.sum(val, axis=0)

    tl.store(out_ptr, total)


def _nansum_global(inp, *, dtype=None):
    logger.debug("GEMS NANSUM (GLOBAL)")

    if dtype is None:
        dtype = inp.dtype

    if inp.numel() == 0:
        return torch.tensor(0, dtype=dtype, device=inp.device)

    work = _promote_for_nansum(inp.contiguous())
    M = work.numel()
    fp64 = work.dtype == torch.float64

    if M <= 16384:
        out = torch.zeros([], dtype=work.dtype, device=work.device)
        with torch_device_fn.device(work.device):
            _nansum_final_reduce[(1,)](work, out, M, FP64=fp64)
        return out.to(dtype)

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    mid = torch.empty(mid_size, dtype=work.dtype, device=work.device)

    with torch_device_fn.device(work.device):
        _nansum_parallel_reduce[(mid_size,)](work, mid, M, block_size, FP64=fp64)
        out = torch.zeros([], dtype=work.dtype, device=work.device)
        _nansum_final_reduce[(1,)](mid, out, mid_size, FP64=fp64)

    return out.to(dtype)


def nansum(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS NANSUM")
    if dim is None:
        return _nansum_global(inp, dtype=dtype)
    return nansum_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _nansum_dim_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP64: tl.constexpr,
):
    """Reduce along the last dimension using a 2D tile (BLOCK_M x BLOCK_N)."""
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptr = inp_ptr + pid * N
    out_ptr = out_ptr + pid
    row_mask = pid < M

    if FP64:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float64)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        val = tl.load(inp_ptr + cols, mask, other=0.0)
        val = tl.where(val != val, 0.0, val)
        acc += val

    result = tl.sum(acc, axis=1)[:, None]
    tl.store(out_ptr, result, row_mask)


def nansum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS NANSUM_DIM")

    if dtype is None:
        dtype = inp.dtype

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if dim is None:
            out_shape = [1] * inp.ndim if keepdim else []
        else:
            dims = [dim] if isinstance(dim, int) else list(dim)
            dims = [d if d >= 0 else d + inp.ndim for d in dims]
            if keepdim:
                for d in dims:
                    out_shape[d] = 1
            else:
                for d in sorted(set(dims), reverse=True):
                    out_shape.pop(d)
        return torch.zeros(out_shape, dtype=dtype, device=inp.device)

    if dim is None:
        dims = list(range(inp.ndim))
    elif isinstance(dim, int):
        dims = [dim if dim >= 0 else dim + inp.ndim]
    else:
        dims = [d if d >= 0 else d + inp.ndim for d in dim]

    dims = sorted(set(dims), reverse=True)

    if len(dims) == inp.ndim:
        result = _nansum_global(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    work = _promote_for_nansum(inp)
    shape = list(work.shape)
    fp64 = work.dtype == torch.float64

    work = dim_compress(work, dims)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1
    M = work.numel() // N

    if N <= 1:
        out = torch.empty_like(work)
        numel = work.numel()
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(work.device):
            _nan_to_zero_kernel[grid](work, out, numel)
        out = out.reshape(shape)
        if not keepdim:
            for d in sorted(dims, reverse=True):
                out = out.squeeze(dim=d)
        return out.to(dtype)

    out_flat = torch.empty(M, dtype=work.dtype, device=work.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(work.device):
        _nansum_dim_kernel[grid](work, out_flat, M, N, FP64=fp64)

    out = out_flat.reshape(shape)
    if not keepdim:
        for d in sorted(dims, reverse=True):
            out = out.squeeze(dim=d)

    return out.to(dtype)
