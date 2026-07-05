import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from ..utils.shape_utils import dim_compress


def count_nonzero_dim1_block_m(M):
    if M >= 1024:
        return 16
    if M >= 256:
        return 8
    return 4


def count_nonzero_dim1_block_n(N):
    if N >= 512:
        return 128
    if N >= 256:
        return 64
    return 32


def count_nonzero_dim0_block_m(M):
    if M >= 1024:
        return 64
    if M >= 256:
        return 32
    return 16


def count_nonzero_dim0_block_n(N):
    if N >= 512:
        return 64
    if N >= 128:
        return 32
    return 16


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, mid_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < numel

    x = tl.load(x_ptr + offset, mask=mask, other=0)
    if x.dtype.is_floating():
        x = tl.abs(x)
    is_nonzero = (x != 0).to(tl.float32)

    local_count = tl.sum(is_nonzero[None, :], axis=1)
    store_offset = tl.arange(0, 1)
    tl.store(mid_ptr + pid + store_offset, local_count, mask=store_offset < 1)


@libentry()
@triton.jit
def count_nonzero_kernel_2(mid_ptr, out_ptr, mid_size, BLOCK_MID: tl.constexpr):
    total = tl.zeros([1], dtype=tl.int32)

    for i in range(0, mid_size, BLOCK_MID):
        offset = i + tl.arange(0, BLOCK_MID)
        mask = offset < mid_size
        mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)
        chunk_sum = tl.sum(mid_val[None, :], axis=1)
        total = total + chunk_sum.to(tl.int32)

    store_offset = tl.arange(0, 1)
    tl.store(out_ptr + store_offset, total, mask=store_offset < 1)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero_dim"), key=["M", "N"])
@triton.jit
def count_nonzero_dim_kernel(
    X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    counts = tl.zeros([BLOCK_M, 1], dtype=tl.int32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask=mask, other=0.0)
        if x.dtype.is_floating():
            x = tl.abs(x)
        is_nonzero = (x != 0).to(tl.float32)
        counts += tl.sum(is_nonzero, axis=1).to(tl.int32)[:, None]

    tl.store(Out, counts, row_mask)


@libentry()
@triton.jit
def count_nonzero_dim1_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid_m < M

    counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(x_ptr + pid_m * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        if x.dtype.is_floating():
            x = tl.abs(x)
        counts += (x != 0).to(tl.float32)

    row_counts = tl.sum(counts, axis=1)[:, None]
    tl.store(out_ptr + pid_m, row_counts.to(tl.int32), row_mask)


@libentry()
@triton.jit
def count_nonzero_dim0_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tle.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    col_mask = pid_n < N

    counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, M, BLOCK_M):
        rows = off + tl.arange(0, BLOCK_M)[:, None]
        row_mask = rows < M
        mask = row_mask and col_mask
        x = tl.load(x_ptr + rows * stride_xm + pid_n * stride_xn, mask=mask, other=0.0)
        if x.dtype.is_floating():
            x = tl.abs(x)
        counts += (x != 0).to(tl.float32)

    col_counts = tl.sum(counts, axis=0)[None, :]
    tl.store(out_ptr + pid_n, col_counts.to(tl.int32), col_mask)


def count_nonzero(x, dim=None):
    logging.debug("GEMS_SOPHGO COUNT_NONZERO")
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        shape = list(x.shape)
        dim = dim % x.ndim
        N = shape[dim]
        out_shape = list(shape)
        del out_shape[dim]

        if N == 0 or x.numel() == 0:
            return torch.zeros(out_shape, dtype=torch.int32, device=x.device)

        if x.ndim == 2 and dim in (0, 1):
            x = x.contiguous()
            M, cols = x.shape
            out = torch.empty(out_shape, dtype=torch.int32, device=x.device)

            with torch_device_fn.device(x.device):
                if dim == 1:
                    block_m = count_nonzero_dim1_block_m(M)
                    block_n = count_nonzero_dim1_block_n(cols)
                    grid = (triton.cdiv(M, block_m),)
                    count_nonzero_dim1_2d_kernel[grid](
                        x,
                        out,
                        M,
                        cols,
                        x.stride(0),
                        x.stride(1),
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                    )
                else:
                    block_m = count_nonzero_dim0_block_m(M)
                    block_n = count_nonzero_dim0_block_n(cols)
                    grid = (triton.cdiv(cols, block_n),)
                    count_nonzero_dim0_2d_kernel[grid](
                        x,
                        out,
                        M,
                        cols,
                        x.stride(0),
                        x.stride(1),
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                    )
            return out

        x = dim_compress(x, dim)
        M = x.numel() // N
        out = torch.zeros(out_shape, dtype=torch.int32, device=x.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            count_nonzero_dim_kernel[grid](x.flatten(), out, M, N)
        return out

    x = x.contiguous().flatten()
    numel = x.numel()

    if numel == 0:
        return torch.tensor(0, dtype=torch.int32, device=x.device)

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(numel)))
    mid_size = triton.cdiv(numel, block_size)

    max_safe_mid = max(1, (2**24 - 1) // block_size)
    raw_mid = min(mid_size, max_safe_mid)
    block_mid = 1 << (raw_mid.bit_length() - 1)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
    out = torch.empty([], dtype=torch.int32, device=x.device)

    with torch_device_fn.device(x.device):
        count_nonzero_kernel_1[(mid_size, 1, 1)](x, mid, numel, block_size)
        count_nonzero_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

    return out
