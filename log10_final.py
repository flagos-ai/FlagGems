"""
log10 operator implementation for FlagGems.

This module provides a Triton-based implementation of the element-wise
base-10 logarithm function, with full autograd support.
"""

import torch
import triton
import triton.language as tl
from flaggems import kernel_meta
from flaggems.utils import shape_utils

__all__ = ['log10']


@triton.jit
def _log10_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for element-wise log10.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Number of elements per program
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute log10(x) = log(x) / log(10)
    log_x = tl.log(x)
    log_10 = tl.log(10.0)
    result = log_x / log_10
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


class _Log10(torch.autograd.Function):
    """Autograd function for log10."""
    
    @staticmethod
    def forward(ctx, input):
        """Forward pass of log10.
        
        Args:
            input: Input tensor of any shape
            
        Returns:
            Tensor with log10 applied element-wise
        """
        ctx.save_for_backward(input)
        
        # Handle scalar input
        if input.ndim == 0:
            input = input.unsqueeze(0)
            was_scalar = True
        else:
            was_scalar = False
        
        # Allocate output
        output = torch.empty_like(input)
        n_elements = input.numel()
        
        # Configure kernel launch
        grid = kernel_meta.grid(n_elements)
        BLOCK_SIZE = kernel_meta.next_power_of_2(n_elements)
        BLOCK_SIZE = min(BLOCK_SIZE, 65536)  # Limit block size
        
        _log10_kernel[grid](
            input,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        if was_scalar:
            output = output.squeeze(0)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of log10.
        
        Args:
            grad_output: Gradient from upstream
            
        Returns:
            Gradient with respect to input
        """
        input, = ctx.saved_tensors
        
        if not ctx.needs_input_grad[0]:
            return None
        
        # d(log10(x))/dx = 1 / (x * ln(10))
        grad_input = grad_output / (input * torch.log(torch.tensor(10.0)))
        
        return grad_input


def log10(input):
    """Element-wise base-10 logarithm.
    
    Args:
        input (Tensor): Input tensor
        
    Returns:
        Tensor: log10 of input
        
    Examples:
        >>> x = torch.tensor([1.0, 10.0, 100.0])
        >>> log10(x)
        tensor([0., 1., 2.])
        
    Note:
        - Supports float16 and float32 data types
        - Handles arbitrary tensor shapes
        - Returns NaN for negative inputs
        - Returns -inf for zero
    """
    return _Log10.apply(input)
