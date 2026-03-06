"""
Triton implementation of log10 operator for FlagGems.
"""

import torch
import triton
import triton.language as tl
from flaggems import kernel_meta
from flaggems.utils import shape_utils


@triton.jit
def log10_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for element-wise log10 operation.
    
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
    # Using natural log and division for numerical stability
    log_x = tl.log(x)
    log_10 = tl.log(10.0)
    result = log_x / log_10
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


class Log10(torch.autograd.Function):
    """Autograd Function for log10 operator.
    
    Supports both forward and backward passes.
    """
    
    @staticmethod
    def forward(ctx, input):
        """Forward pass: compute log10(input)
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor with log10 applied element-wise
        """
        # Save for backward
        ctx.save_for_backward(input)
        
        # Handle 0-dim tensors
        if input.ndim == 0:
            input = input.unsqueeze(0)
            was_scalar = True
        else:
            was_scalar = False
        
        # Allocate output
        output = torch.empty_like(input)
        
        # Get grid dimensions
        n_elements = input.numel()
        grid = kernel_meta.grid(n_elements)
        
        # Choose block size based on input size
        BLOCK_SIZE = kernel_meta.next_power_of_2(n_elements)
        if BLOCK_SIZE > 65536:
            BLOCK_SIZE = 65536
        
        # Launch kernel
        log10_kernel[grid](
            input,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Return to original shape
        if was_scalar:
            output = output.squeeze(0)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient = grad_output / (input * ln(10))
        
        Args:
            grad_output: Gradient from upstream
            
        Returns:
            Gradient with respect to input
        """
        input, = ctx.saved_tensors
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            # d(log10(x))/dx = 1 / (x * ln(10))
            grad_input = grad_output / (input * torch.log(torch.tensor(10.0)))
        
        return grad_input


def log10(input):
    """Element-wise base-10 logarithm.
    
    Args:
        input: Input tensor
        
    Returns:
        Tensor with log10 applied element-wise
        
    Examples:
        >>> x = torch.tensor([1.0, 10.0, 100.0])
        >>> log10(x)
        tensor([0., 1., 2.])
    """
    return Log10.apply(input)
