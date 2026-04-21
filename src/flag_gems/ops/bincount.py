import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def bincount_kernel(
    inp_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for bincount without weights."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input values (indices)
    indices = tl.load(inp_ptr + offsets, mask=mask, other=0)

    # Atomic add 1 to the output at each index
    # Use int64 for the atomic add
    ones = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)
    tl.atomic_add(out_ptr + indices, ones, mask=mask, sem="relaxed")


@libentry()
@triton.jit
def bincount_weights_kernel(
    inp_ptr,
    weights_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for bincount with weights."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input values (indices) and weights
    indices = tl.load(inp_ptr + offsets, mask=mask, other=0)
    weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0)

    # Atomic add weights to the output at each index
    tl.atomic_add(out_ptr + indices, weights, mask=mask, sem="relaxed")


def bincount(inp, weights=None, minlength=0):
    """
    Count the frequency of each value in an array of non-negative ints.

    Args:
        inp: 1-d int tensor of non-negative integers
        weights: optional weights tensor of same size as inp
        minlength: optional minimum number of bins

    Returns:
        Tensor of shape (max(inp) + 1,) or (minlength,) if minlength is larger
    """
    logger.debug("GEMS BINCOUNT")

    # Input validation
    assert inp.ndim == 1, "bincount only supports 1-d tensors"
    assert inp.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ), "bincount only supports integer tensors"

    N = inp.numel()

    # Handle empty input
    if N == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=inp.device)
        return torch.zeros(minlength, dtype=torch.int64, device=inp.device)

    # Compute output size
    max_val = int(inp.max().item())
    output_size = max(max_val + 1, minlength)

    # Ensure input is contiguous
    inp = inp.contiguous()

    if weights is not None:
        assert weights.shape == inp.shape, "weights must have same shape as input"
        weights = weights.contiguous()

        # Output dtype matches weights dtype
        # For atomic_add compatibility, convert to float32 if float16/bfloat16
        weights_dtype = weights.dtype
        if weights_dtype in (torch.float16, torch.bfloat16):
            weights = weights.to(torch.float32)
            out = torch.zeros(output_size, dtype=torch.float32, device=inp.device)
        else:
            out = torch.zeros(output_size, dtype=weights.dtype, device=inp.device)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        bincount_weights_kernel[grid](
            inp,
            weights,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Convert back if needed
        if weights_dtype in (torch.float16, torch.bfloat16):
            out = out.to(weights_dtype)

        return out
    else:
        # No weights: count occurrences
        out = torch.zeros(output_size, dtype=torch.int64, device=inp.device)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        bincount_kernel[grid](
            inp,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out
