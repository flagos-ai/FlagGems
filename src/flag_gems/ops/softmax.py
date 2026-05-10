import logging

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


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logger.debug("GEMS SOFTMAX")

    orig_dtype = x.dtype
    orig_shape = x.shape

    # Cast to fp32 for precision - fixes fp16 and bf16 numerical errors
    x = x.contiguous().float()

    # Move softmax dim to last axis for coalesced memory access
    if dim not in (-1, len(orig_shape) - 1):
        x = x.transpose(dim, -1).contiguous()

    # Flatten to 2D: (num_rows, N)
    x2d = x.reshape(-1, x.shape[-1])
    num_rows, N = x2d.shape

    # Allocate output buffer in fp32
    out = torch.empty_like(x2d)

    # BLOCK_SIZE must cover full row - use next power of 2 >= N
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Select num_warps based on block size for optimal occupancy
    if BLOCK_SIZE <= 256:
        num_warps = 2
    elif BLOCK_SIZE <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    # Launch one program per row
    softmax_kernel[(num_rows,)](
        out,
        x2d,
        x2d.stride(0),
        out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Restore original shape
    out = out.view(x.shape)
    if dim not in (-1, len(orig_shape) - 1):
        out = out.transpose(dim, -1).contiguous()

    # Cast back to original dtype
    return out.to(orig_dtype)
