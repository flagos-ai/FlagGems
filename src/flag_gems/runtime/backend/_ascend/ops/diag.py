import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@triton.jit
def diag_1d_to_2d_kernel(
    data_ptr, output_ptr, N, M, stride, diagonal: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    idx = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if diagonal >= 0:
        row_idx = idx
        col_idx = row_idx + diagonal
    else:
        col_idx = idx
        row_idx = col_idx + tl.abs(diagonal)

    mask = (row_idx < M) & (col_idx < M)

    diag_value = tl.load(data_ptr + idx * stride, mask=idx < N, other=0)

    out_offset = row_idx * M + col_idx
    tl.store(output_ptr + out_offset, diag_value, mask=mask)


@triton.jit
def diag_2d_to_1d_kernel(
    data_ptr,
    output_ptr,
    N,
    M,
    stride0,
    stride1,
    diagonal: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if diagonal >= 0:
        row_idx = idx
        col_idx = row_idx + diagonal
    else:
        col_idx = idx
        row_idx = col_idx + tl.abs(diagonal)
    mask = (row_idx < N) & (col_idx < M)

    diag_value = tl.load(
        data_ptr + row_idx * stride0 + col_idx * stride1, mask=mask, other=0
    )
    tl.store(output_ptr + idx, diag_value, mask=mask)


def diag_1d_to_2d(x, diagonal=0):
    N = x.shape[0]
    M = N + abs(diagonal)
    output = torch.zeros((M, M), dtype=x.dtype, device=x.device)

    stride = x.stride(0)
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        diag_1d_to_2d_kernel[grid](
            x, output, N, M, stride, diagonal, BLOCK_SIZE=BLOCK_SIZE
        )
    return output


def diag_2d_to_1d(x, diagonal=0):
    N, M = x.shape
    if diagonal >= 0:
        diag_len = min(N, M - diagonal)
    else:
        diag_len = min(N + diagonal, M)
    if diag_len <= 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)
    output = torch.empty(diag_len, dtype=x.dtype, device=x.device)
    stride0 = x.stride(0)
    stride1 = x.stride(1)
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(diag_len, BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        diag_2d_to_1d_kernel[grid](
            x, output, N, M, stride0, stride1, diagonal, BLOCK_SIZE=BLOCK_SIZE
        )
    return output


def diag(x, diagonal=0):
    logger.debug("GEMS_ASCEND DIAG")
    if x.dim() == 1:
        return diag_1d_to_2d(x, diagonal)
    elif x.dim() == 2:
        return diag_2d_to_1d(x, diagonal)
    else:
        raise ValueError("Input must be a 1D or 2D tensor.")
