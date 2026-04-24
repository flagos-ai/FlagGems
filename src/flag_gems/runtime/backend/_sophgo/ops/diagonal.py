import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


# def diagonal(input, offset=0, dim1=0, dim2=1):
#     """
#     TPU-specific diagonal implementation - delegates to main implementation
#     """
#     from flag_gems.ops.diagonal import diagonal as _diagonal
#     logging.debug("GEMS DIAGONAL (TPU backend)")
#     return _diagonal(input, offset=offset, dim1=dim1, dim2=dim2)


@triton.jit
def diagonal_kernel(
    input_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    input_shape_0,
    input_shape_1,
    offset,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for extracting diagonal elements from a 2D tensor
    """
    # Get current thread's index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate diagonal element positions in the input matrix
    # For standard diagonal extraction (dim1=0, dim2=1):
    # offset=0: (0,0), (1,1), (2,2), ...
    # offset>0: (0,offset), (1,1+offset), (2,2+offset), ...
    # offset<0: (-offset,0), (1-offset,1), (2-offset,2), ...

    row_indices = offsets
    col_indices = offsets + offset

    # For negative offset, adjust starting position
    if offset < 0:
        row_indices = offsets + (-offset)  # Start from row (-offset)
        col_indices = offsets  # Start from column 0

    # Check if indices are within valid range
    valid_mask = (
        mask
        & (row_indices >= 0)
        & (row_indices < input_shape_0)
        & (col_indices >= 0)
        & (col_indices < input_shape_1)
    )

    # Calculate input data memory addresses
    # Ensure correct memory access pattern here
    input_indices = row_indices * input_stride_0 + col_indices * input_stride_1
    input_ptrs = input_ptr + input_indices

    # Read data
    data = tl.load(input_ptrs, mask=valid_mask, other=0.0)

    # Write output
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, data, mask=mask)


def diagonal(input, offset=0, dim1=0, dim2=1):
    """
    Extract diagonal elements from a tensor

    Args:
        input: Input tensor (must be at least 2D)
        offset: Offset of the diagonal from the main diagonal
        dim1: First dimension for diagonal extraction
        dim2: Second dimension for diagonal extraction

    Returns:
        1D tensor containing the diagonal elements
    """
    logging.debug("GEMS DIAGONAL")

    # Check input dimensions
    if input.dim() < 2:
        raise ValueError(
            f"diagonal requires tensor with at least 2 dimensions, got {input.dim()}"
        )

    # Get input tensor shape
    shape = input.shape

    # Ensure dim1 and dim2 are within valid range
    dim1 = dim1 % input.dim()
    dim2 = dim2 % input.dim()

    if dim1 == dim2:
        raise ValueError("dim1 and dim2 cannot be the same")

    # Calculate diagonal length
    if offset >= 0:
        diagonal_size = max(0, min(shape[dim1], shape[dim2] - offset))
    else:
        diagonal_size = max(0, min(shape[dim1] + offset, shape[dim2]))

    if diagonal_size == 0:
        # Empty diagonal
        return torch.empty(0, dtype=input.dtype, device=input.device)

    # Create output tensor
    output = torch.empty(diagonal_size, dtype=input.dtype, device=input.device)

    # Ensure input tensor is contiguous (for 2D case)
    if input.dim() == 2:
        input_2d = input.contiguous()

        # Launch kernel
        grid = lambda meta: (triton.cdiv(diagonal_size, meta["BLOCK_SIZE"]),)
        diagonal_kernel[grid](
            input_2d,
            output,
            input_2d.stride(0),  # row stride
            input_2d.stride(1),  # column stride
            input_2d.shape[0],  # number of rows
            input_2d.shape[1],  # number of columns
            offset,
            diagonal_size,
            BLOCK_SIZE=256,
        )
    else:
        # For higher-dimensional tensors, convert to 2D for processing
        # This is a simplified implementation; actual usage may require more complex handling
        # Fall back to torch implementation here to ensure correctness
        return torch.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)

    return output


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logging.debug("GEMS diagonal backward")
    grad_input = torch.zeros(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )
    diag = torch.diagonal(grad_input, offset, dim1, dim2)
    copy_func.instantiate(grad_output.ndim)(grad_output, out0=diag)
    return grad_input
