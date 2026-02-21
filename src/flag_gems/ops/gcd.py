"""
GCD operator implementation for FlagGems.

This module provides the greatest common divisor (GCD) operation using native Triton kernels.
Implements optimized Euclidean algorithm with Lamé's theorem for iteration estimation.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def gcd_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data with cache hint
    a = tl.load(a_ptr + offsets, mask=mask, other=0, cache_modifier=".cg")
    b = tl.load(b_ptr + offsets, mask=mask, other=0, cache_modifier=".cg")

    # GCD is always computed on absolute values
    a = tl.abs(a)
    b = tl.abs(b)

    # Find maximum value in block to estimate required iterations
    # Based on Lamé's theorem: iterations ≤ 5 × number of digits
    min_ab = tl.where(a < b, a, b)
    max_val = tl.max(min_ab)

    # Estimate iterations based on magnitude ranges
    if max_val < 10:
        max_iter = 5
    elif max_val < 100:
        max_iter = 10
    elif max_val < 1000:
        max_iter = 15
    elif max_val < 10000:
        max_iter = 20
    elif max_val < 100000:
        max_iter = 25
    elif max_val < 1000000:
        max_iter = 30
    elif max_val < 10000000:
        max_iter = 35
    else:
        max_iter = 45

    # Standard Euclidean algorithm with loop unrolling (5 iterations per group)
    num_unrolled_groups = max_iter // 5
    remaining_iters = max_iter % 5

    # Unrolled loop groups
    for _ in range(num_unrolled_groups):
        # Iteration 1
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 2
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 3
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 4
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 5
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

    # Remaining iterations
    for _ in range(remaining_iters):
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

    # Result is in 'a'
    result = a

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def gcd(input, other):
    """
    Compute the greatest common divisor (GCD) of two tensors element-wise.

    This function provides the GCD operation for FlagGems, optimized
    for GPU execution using native Triton kernels with loop unrolling
    and Lamé's theorem for iteration estimation.

    Args:
        input (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        Tensor: Tensor containing GCD values

    Examples:
        >>> import torch
        >>> from flag_gems import gcd
        >>> x = torch.tensor([12, 18, 24], device='cuda')
        >>> y = torch.tensor([8, 12, 36], device='cuda')
        >>> z = gcd(x, y)
        >>> print(z)
        tensor([4, 6, 12], device='cuda')
    """
    logger.debug("GEMS GCD")

    # Handle empty tensor
    if input.numel() == 0:
        return torch.empty_like(input)

    # Ensure inputs have the same shape
    if input.shape != other.shape:
        raise ValueError(
            f"Shape mismatch: input.shape={input.shape}, other.shape={other.shape}"
        )

    # Create output tensor
    out = torch.empty_like(input)

    # Flatten tensors for 1D processing
    input_flat = input.contiguous().view(-1)
    other_flat = other.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)

    # Get total number of elements
    n_elements = input_flat.numel()

    # Choose optimal block size based on input size
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 16384:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024

    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)

    # Launch kernel
    gcd_kernel[grid](
        input_flat,
        other_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
