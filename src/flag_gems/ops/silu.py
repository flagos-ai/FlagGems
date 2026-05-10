import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid of x for SiLU activation
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))

    # SiLU is x multiplied by sigmoid of x
    out = x * sigmoid_x

    # Store result back to output buffer
    tl.store(out_ptr + offsets, out, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS SILU")
    orig_dtype = x.dtype
    x = x.contiguous().float()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 4096))
    if BLOCK_SIZE <= 256:
        num_warps = 2
    elif BLOCK_SIZE <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out.to(orig_dtype)
