import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(inp_val, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)


def mean(inp, *, dtype=None):
    logging.debug("GEMS MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
    mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4)
        for m in [1, 2, 4, 8]
    ],
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logging.debug("GEMS MEAN DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        dim = list(range(x.ndim))

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    order = [i for i in range(x.ndim) if i not in dim] + dim
    x = x.permute(order).contiguous()
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    mean = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    mean_dim_kernel[grid](x, mean, M, N)
    if not keepdim:
        mean = mean.squeeze(dim)
    return mean
