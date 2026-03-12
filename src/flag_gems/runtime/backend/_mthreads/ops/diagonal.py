import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@libentry()
@triton.jit
def diagonal_backward_kernel_2d(
    grad_output_ptr,
    grad_input_ptr,
    diag_size,
    stride_dim1,
    stride_dim2,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for 2D diagonal backward.
    grad_output is 1D (the diagonal values), grad_input is 2D.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < diag_size

    # Load gradient values from grad_output (1D contiguous)
    grad_vals = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)

    # Compute destination indices
    if offset >= 0:
        row_idx = offsets
        col_idx = offsets + offset
    else:
        row_idx = offsets - offset
        col_idx = offsets

    # Calculate destination offset in grad_input
    dst_offsets = row_idx * stride_dim1 + col_idx * stride_dim2

    # Store to grad_input
    tl.store(grad_input_ptr + dst_offsets, grad_vals, mask=mask)


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    """
    Optimized diagonal_backward for mthreads backend using Triton kernels.

    This computes the backward pass of torch.diagonal, which creates a zero tensor
    and copies grad_output into the diagonal positions.
    """
    logger.debug("GEMS_MTHREADS DIAGONAL BACKWARD")

    input_sizes = list(input_sizes)
    ndim = len(input_sizes)

    # Normalize dimensions
    if dim1 < 0:
        dim1 = ndim + dim1
    if dim2 < 0:
        dim2 = ndim + dim2

    # Ensure dim1 < dim2 for consistent handling
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset

    # Calculate diagonal size
    size_dim1 = input_sizes[dim1]
    size_dim2 = input_sizes[dim2]

    if offset >= 0:
        diag_size = min(size_dim1, max(0, size_dim2 - offset))
    else:
        diag_size = min(max(0, size_dim1 + offset), size_dim2)

    # Create output tensor filled with zeros
    grad_input = torch.zeros(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )

    if diag_size <= 0:
        return grad_input

    if grad_output.numel() == 0:
        return grad_input

    # Get strides of grad_input
    grad_input_strides = grad_input.stride()
    stride_dim1 = grad_input_strides[dim1]
    stride_dim2 = grad_input_strides[dim2]

    # Choose BLOCK_SIZE based on problem size
    total_elements = grad_output.numel()
    if total_elements <= 256:
        BLOCK_SIZE = 256
    elif total_elements <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048

    if ndim == 2:
        # Simple 2D case - use optimized 2D kernel
        grid = (triton.cdiv(diag_size, BLOCK_SIZE),)

        with torch_device_fn.device(grad_input.device):
            diagonal_backward_kernel_2d[grid](
                grad_output,
                grad_input,
                diag_size,
                stride_dim1,
                stride_dim2,
                offset,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        # N-D case - use view-based approach for reliability
        diag_view = torch.diagonal(grad_input, offset, dim1, dim2)
        diag_view.copy_(grad_output)

    return grad_input
