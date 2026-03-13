"""
Log10 operator implementation using Triton.

This module implements the log10 (base-10 logarithm) operation using Triton
for GPU acceleration. The implementation follows PyTorch's log10 API and
supports float32 and float16 data types.

The implementation uses autotune for optimal block size selection and
mathematical optimization (multiplication instead of division) for better
performance.

Reference: https://pytorch.org/docs/stable/generated/torch.log10.html
"""

import torch
import triton
import triton.language as tl
from typing import Optional

# 1/ln(10) constant for optimized computation
# Using multiplication instead of division is faster on GPUs
LOG10_RECIPROCAL = 0.43429448190325182765


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _log10_kernel(
    X_ptr,
    Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing base-10 logarithm.

    This kernel uses autotune to select optimal block size for different
    tensor sizes and GPU architectures. It uses multiplication instead of
    division for better performance.

    Args:
        X_ptr: Pointer to input tensor data
        Y_ptr: Pointer to output tensor data
        N: Total number of elements
        BLOCK_SIZE: Number of elements processed per block (auto-tuned)
    """
    # 1/ln(10) constant for optimized computation
    # Using multiplication instead of division is faster on GPUs
    LOG10_RECIPROCAL: tl.constexpr = 0.43429448190325182765
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Masking: prevent out-of-bounds memory access
    mask = offsets < N
    
    # Load data
    x = tl.load(X_ptr + offsets, mask=mask)
    
    # Compute log10(x) = ln(x) * (1/ln(10))
    # Using multiplication instead of division is faster on GPUs
    output = tl.log(x) * LOG10_RECIPROCAL
    
    # Store result
    tl.store(Y_ptr + offsets, output, mask=mask)


def log10(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the base-10 logarithm of the input tensor element-wise.

    This function implements log10 using Triton for GPU acceleration.
    The output matches PyTorch's torch.log10 implementation.

    The implementation uses autotune to automatically select the optimal
    block size for the given tensor size and GPU architecture, and uses
    mathematical optimizations (multiplication instead of division) for
    better performance.

    Args:
        input: Input tensor. Must be a CUDA tensor with dtype float32 or float16.
        out: Optional output tensor. If provided, results are written into it.

    Returns:
        A tensor with the same shape as input, containing the base-10 logarithm
        of each element.

    Raises:
        AssertionError: If input is not a CUDA tensor or has unsupported dtype.

    Examples:
        >>> import torch
        >>> x = torch.tensor([1.0, 10.0, 100.0], device='cuda')
        >>> y = log10(x)
        >>> print(y)
        tensor([0.0000, 1.0000, 2.0000], device='cuda:0')
    """
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert input.dtype in (torch.float32, torch.float16), \
        f"Unsupported dtype: {input.dtype}. Supported: float32, float16"

    # Ensure input is contiguous for optimal memory access
    # Non-contiguous tensors (e.g., from slicing) can cause incorrect results
    if not input.is_contiguous():
        input = input.contiguous()

    N = input.numel()
    if N == 0:
        # Handle empty tensor
        if out is None:
            return torch.empty_like(input)
        return out

    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.dtype == input.dtype, "Output dtype must match input dtype"
        assert out.is_cuda, "Output must be a CUDA tensor"
        # Ensure output is contiguous
        if not out.is_contiguous():
            out = out.contiguous()

    # Handle float16: Triton's tl.log() doesn't support float16 directly
    # We need to cast to float32, compute, then cast back
    if input.dtype == torch.float16:
        # Cast input to float32 for computation
        input_fp32 = input.to(torch.float32)
        if out is None:
            out_fp32 = torch.empty_like(input_fp32)
        else:
            out_fp32 = out.to(torch.float32)
        
        # Grid function for autotune
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
        
        # Compute with float32
        _log10_kernel[grid](input_fp32, out_fp32, N)
        
        # Cast result back to float16
        if out is None:
            out = out_fp32.to(torch.float16)
        else:
            out.copy_(out_fp32.to(torch.float16))
    else:
        # Grid function for autotune
        # Autotune will automatically select the best BLOCK_SIZE from configs
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
        
        _log10_kernel[grid](input, out, N)

    return out

