import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def diag_1d_to_2d_kernel(
    data_ptr, output_ptr, N, M, stride, diagonal: tl.constexpr
):
    off = tl.arange(0, 1)
    pid = tle.program_id(0)
    i = pid + off

    if diagonal >= 0:
        r = i
        c = r + diagonal
    else:
        c = i
        r = c - diagonal

    mask = (r < M) & (c < M) & (i < N)
    val = tl.load(data_ptr + i * stride, mask=mask)
    tl.store(output_ptr + r * M + c, val, mask=mask)


@triton.jit
def diag_2d_to_1d_kernel(
    data_ptr,
    output_ptr,
    N,
    M,
    stride0,
    stride1,
    diagonal: tl.constexpr,
):
    off = tl.arange(0, 1)
    pid = tle.program_id(0)
    i = pid + off

    if diagonal >= 0:
        r = i
        c = r + diagonal
    else:
        c = i
        r = c - diagonal

    mask = (r < N) & (c < M)
    val = tl.load(data_ptr + r * stride0 + c * stride1, mask=mask)
    tl.store(output_ptr + i, val, mask=mask)


def diag_1d_to_2d(x, diagonal=0):
    N = x.shape[0]
    M = N + abs(diagonal)
    output = torch.zeros((M, M), dtype=x.dtype, device=x.device)

    stride = x.stride(0)

    grid = lambda meta: (N,)

    with torch_device_fn.device(x.device):
        diag_1d_to_2d_kernel[grid](x, output, N, M, stride, diagonal)
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

    grid = lambda meta: (diag_len,)

    with torch_device_fn.device(x.device):
        diag_2d_to_1d_kernel[grid](x, output, N, M, stride0, stride1, diagonal)
    return output


def diag(x, diagonal=0):
    logger.debug("GEMS DIAG")
    if x.dim() == 1:
        return diag_1d_to_2d(x, diagonal)
    elif x.dim() == 2:
        return diag_2d_to_1d(x, diagonal)
    else:
        raise ValueError("Input must be a 1D or 2D tensor.")