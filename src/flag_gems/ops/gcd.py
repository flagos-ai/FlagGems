import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def gcd_kernel(
    X,
    Y,
    OUT,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask, other=0).to(tl.int64)
    y = tl.load(Y + offsets, mask=mask, other=0).to(tl.int64)

    # Take absolute values
    a = tl.abs(x)
    b = tl.abs(y)

    # Handle zero cases
    orig_a = a
    orig_b = b

    # Binary GCD (Stein's algorithm) - optimized version
    # Step 1: Find common power of 2 (reduced iterations)
    shift = tl.zeros_like(a).to(tl.int64)
    for i in range(5):  # Reduced from 6 to 5
        both_even = ((a & 1) == 0) & ((b & 1) == 0) & (a != 0) & (b != 0)
        a = tl.where(both_even, a >> 1, a)
        b = tl.where(both_even, b >> 1, b)
        shift = tl.where(both_even, shift + 1, shift)

    # Step 2: Make a odd (reduced iterations)
    for i in range(5):  # Reduced from 6 to 5
        a_even = ((a & 1) == 0) & (a != 0)
        a = tl.where(a_even, a >> 1, a)

    # Step 3: Main loop (balanced iterations for accuracy and performance)
    for i in range(13):  # Increased from 12 to 16 for better accuracy
        # Make b odd (reduced iterations)
        for j in range(5):  # Reduced from 6 to 5
            b_even = ((b & 1) == 0) & (b != 0)
            b = tl.where(b_even, b >> 1, b)

        # If b is zero, we're done
        b_is_zero = b == 0

        # Swap if a > b (only when b is not zero)
        need_swap = (a > b) & (~b_is_zero)
        temp = a
        a = tl.where(need_swap, b, a)
        b = tl.where(need_swap, temp, b)

        # b = b - a (only when b is not zero)
        b = tl.where(~b_is_zero, b - a, b)

    # Step 4: Restore common power of 2
    result = a << shift

    # Handle original zero cases
    orig_is_a_zero = orig_a == 0
    orig_is_b_zero = orig_b == 0
    result = tl.where(orig_is_a_zero & orig_is_b_zero, 0, result)
    result = tl.where(orig_is_a_zero & (~orig_is_b_zero), orig_b, result)
    result = tl.where((~orig_is_a_zero) & orig_is_b_zero, orig_a, result)

    tl.store(OUT + offsets, result, mask=mask)


def gcd(input, other):
    """
    Computes the element-wise greatest common divisor (GCD) of input and other.

    Note: Currently only int64 is supported for performance reasons.
    int32 and smaller types will be automatically promoted to int64.

    Args:
        input: the first input tensor
        other: the second input tensor

    Returns:
        A new tensor with the GCD of input and other
    """
    logger.debug("GEMS GCD")

    # Broadcast inputs to the same shape
    input, other = torch.broadcast_tensors(input, other)

    # GCD requires integer types
    assert input.dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ], f"gcd expects integer dtype, got {input.dtype}"

    # Promote to int64 for better performance
    # (int32 and smaller types have poor performance with Binary GCD on GPU)
    original_dtype = input.dtype
    if input.dtype != torch.int64:
        input = input.to(torch.int64)
        other = other.to(torch.int64)

    out = torch.empty_like(input)
    n_elements = input.numel()

    if n_elements == 0:
        return out.to(original_dtype)

    # Use larger block size for better occupancy
    BLOCK = 2048
    grid = (triton.cdiv(n_elements, BLOCK),)
    gcd_kernel[grid](
        input.reshape(-1),
        other.reshape(-1),
        out.reshape(-1),
        n_elements,
        BLOCK=BLOCK,
        num_warps=8,  # Increase warps for better parallelism
    )

    # Convert back to original dtype
    return out.to(original_dtype)
