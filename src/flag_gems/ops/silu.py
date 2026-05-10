import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


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


@libentry()
def silu(x: torch.Tensor) -> torch.Tensor:
    """
    FlagGems optimized SiLU activation - x * sigmoid(x).
    Supports fp16, bf16, fp32 via internal fp32 accumulation.
    Fuses multiply and sigmoid into single Triton kernel pass.
    """
    logger.debug("GEMS SILU")

    orig_dtype = x.dtype

    # Cast to fp32 for precision - fixes fp16 and bf16 numerical errors
    x = x.contiguous().float()
    out = torch.empty_like(x)
    n_elements = x.numel()

    # BLOCK_SIZE covers elements - use next power of 2
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 4096))

    # Select num_warps based on block size for optimal occupancy
    if BLOCK_SIZE <= 256:
        num_warps = 2
    elif BLOCK_SIZE <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    # Launch one program per block of elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    # Cast back to original dtype
    return out.to(orig_dtype)
