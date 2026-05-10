import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def softmax_kernel(
    out_ptr,
    inp_ptr,
    inp_row_stride,
    out_row_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # One program handles one row of the input tensor
    row_idx = tl.program_id(axis=0)
    row_inp = inp_ptr + row_idx * inp_row_stride
    row_out = out_ptr + row_idx * out_row_stride

    # Column offsets - BLOCK_SIZE always >= N so full row fits
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row - pad out-of-bounds with -inf so they dont affect max
    x = tl.load(row_inp + cols, mask=mask, other=-float("inf"))

    # Subtract row max for numerical stability (prevents exp overflow)
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Compute exp of shifted values
    x_exp = tl.exp(x)

    # Compute normalizing constant
    x_sum = tl.sum(x_exp, axis=0)

    # Normalize to get softmax probabilities
    out = x_exp / x_sum

    # Write result back to output row (coalesced store)
    tl.store(row_out + cols, out, mask=mask)


@libentry()
@triton.jit
def softmax_backward_kernel(
    dx_ptr,
    dy_ptr,
    y_ptr,
    inp_row_stride,
    out_row_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # One program handles one row
    row_idx = tl.program_id(axis=0)
    dy_row = dy_ptr + row_idx * inp_row_stride
    y_row = y_ptr + row_idx * inp_row_stride
    dx_row = dx_ptr + row_idx * out_row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load softmax output and upstream gradient
    y = tl.load(y_row + cols, mask=mask, other=0.0)
    dy = tl.load(dy_row + cols, mask=mask, other=0.0)

    # Softmax backward: dx = y * (dy - sum(dy * y))
    dy_sum = tl.sum(dy * y, axis=0)
    dx = y * (dy - dy_sum)

    tl.store(dx_row + cols, dx, mask=mask)


def _forward(x: torch.Tensor, dim: int) -> torch.Tensor:
    orig_dtype = x.dtype
    orig_shape = x.shape
    x = x.contiguous().float()
    if dim not in (-1, len(orig_shape) - 1):
        x = x.transpose(dim, -1).contiguous()
    x2d = x.reshape(-1, x.shape[-1])
    num_rows, N = x2d.shape
    out = torch.empty_like(x2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8
    softmax_kernel[(num_rows,)](
        out,
        x2d,
        x2d.stride(0),
        out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    out = out.view(x.shape)
    if dim not in (-1, len(orig_shape) - 1):
        out = out.transpose(dim, -1).contiguous()
    return out.to(orig_dtype)


def _backward(dy: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    orig_dtype = dy.dtype
    orig_shape = dy.shape
    dy = dy.contiguous().float()
    y = y.contiguous().float()
    if dim not in (-1, len(orig_shape) - 1):
        dy = dy.transpose(dim, -1).contiguous()
        y = y.transpose(dim, -1).contiguous()
    dy2d = dy.reshape(-1, dy.shape[-1])
    y2d = y.reshape(-1, y.shape[-1])
    num_rows, N = dy2d.shape
    dx = torch.empty_like(dy2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 2 if BLOCK_SIZE <= 256 else 4 if BLOCK_SIZE <= 1024 else 8
    softmax_backward_kernel[(num_rows,)](
        dx,
        dy2d,
        y2d,
        dy2d.stride(0),
        dx.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dx = dx.view(dy.shape)
    if dim not in (-1, len(orig_shape) - 1):
        dx = dx.transpose(dim, -1).contiguous()
    return dx.to(orig_dtype)


def softmax(
    x: torch.Tensor, dim: int = -1, half_to_float: bool = False
) -> torch.Tensor:
    logger.debug("GEMS SOFTMAX")
    return _forward(x, dim)


def softmax_out(
    x: torch.Tensor,
    dim: int = -1,
    half_to_float: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS SOFTMAX OUT")
    result = _forward(x, dim)
    if out is not None:
        out.copy_(result)
        return out
    return result


def softmax_backward(dy: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logger.debug("GEMS SOFTMAX BACKWARD")
    return _backward(dy, y, dim)


def softmax_backward_out(
    dy: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS SOFTMAX BACKWARD OUT")
    result = _backward(dy, y, dim)
    if out is not None:
        out.copy_(result)
        return out
    return result
