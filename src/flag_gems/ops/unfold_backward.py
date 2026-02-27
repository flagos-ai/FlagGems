import logging
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _unfold_backward_kernel(
    grad_in_ptr,
    grad_out_ptr,
    numel,
    # grad_in shape (padded to 8)
    gin_s0,
    gin_s1,
    gin_s2,
    gin_s3,
    gin_s4,
    gin_s5,
    gin_s6,
    gin_s7,
    # grad_out strides (padded to 8)
    gout_st0,
    gout_st1,
    gout_st2,
    gout_st3,
    gout_st4,
    gout_st5,
    gout_st6,
    # dim position in the padded representation (adjusted for padding)
    dim_padded,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for unfold_backward.

    grad_in has shape: [S0, ..., S_{dim-1}, num_windows, S_{dim+1}, ..., S_{N-1}, size]
    grad_out has shape: [S0, ..., S_{dim-1}, L, S_{dim+1}, ..., S_{N-1}]

    For each element grad_in[i0, ..., i_{dim}, ..., i_{N-1}, s]:
    - Maps to grad_out[i0, ..., i_{dim} * step + s, ..., i_{N-1}]
    - i.e., same indices except dim-th index becomes i_{dim} * step + s
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load gradient value
    grad_val = tl.load(grad_in_ptr + offsets, mask=mask, other=0.0)

    # Extract indices from flat offset (C-order, rightmost varies fastest)
    remaining = offsets

    # Extract from rightmost to leftmost
    # i7 is the 'size' dimension (always last in grad_in)
    i7 = remaining % gin_s7
    remaining = remaining // gin_s7

    i6 = remaining % gin_s6
    remaining = remaining // gin_s6

    i5 = remaining % gin_s5
    remaining = remaining // gin_s5

    i4 = remaining % gin_s4
    remaining = remaining // gin_s4

    i3 = remaining % gin_s3
    remaining = remaining // gin_s3

    i2 = remaining % gin_s2
    remaining = remaining // gin_s2

    i1 = remaining % gin_s1
    remaining = remaining // gin_s1

    i0 = remaining % gin_s0

    # i7 is the position within window (the 'size' dimension, always last)
    # dim_padded is the position of the num_windows dimension in the padded representation

    # Compute output offset
    # For output:
    # - Dimensions 0 to dim_padded-1: same index as grad_in (i0 to i_{dim_padded-1})
    # - Dimension dim_padded: index = grad_in[dim_padded] * step + grad_in[last]
    # - Dimensions dim_padded+1 to N-2: same index as grad_in (shifted by 1 since we remove 'size' dim)
    # - The 'size' dimension (last) is consumed

    # Compute the modified index for dim_padded dimension
    # After padding, positions map to i0, i1, ..., i7
    # dim_padded tells us which of these is the num_windows dimension
    # i7 is always the 'size' dimension

    # For each position, compute the index used for output stride calculation
    # If position == dim_padded: use index * step + i7
    # Otherwise: use the original index

    # For positions > dim_padded (except i7): use the index from position-1?
    # No wait, the output has one fewer dimension (no 'size' dim)
    # So output dimension d maps to:
    # - grad_in dimension d for d <= dim_padded
    # - grad_in dimension d for d > dim_padded (but we skip dimension 7 which is 'size')

    # Actually, the output strides are already computed for the output shape
    # Output dimension d (0 to N-2) corresponds to:
    # - grad_in dimension d for d < 7 (the last output dim uses grad_in dim 6)

    # Let me think again:
    # grad_in: 8 dimensions (padded), positions 0-7, where position 7 is 'size'
    # grad_out: 7 dimensions (padded), positions 0-6

    # For output, the strides are for positions 0-6
    # The index for output position d should be:
    # - If d < dim_padded: use grad_in index at position d (i.e., i_d)
    # - If d == dim_padded: use grad_in index at position d * step + i7
    # - If d > dim_padded: use grad_in index at position d (since output dims > dim_padded
    #   correspond to grad_in dims d+1, but wait, we only skip the last dim 'size')

    # Hmm, this is confusing. Let me think more carefully.

    # grad_in has N dimensions (before padding). After padding to 8, we have 8 - N leading 1s.
    # grad_out has N-1 dimensions (before padding). After padding to 7, we have 7 - (N-1) = 8 - N leading 1s.

    # For the actual dimensions (ignoring padding):
    # grad_in dims: [d0, d1, ..., d_{dim}, ..., d_{N-2}, size]
    # grad_out dims: [d0, d1, ..., d_{dim}, ..., d_{N-2}]

    # The indices are:
    # grad_in: [i0, i1, ..., i_{dim} (window_idx), ..., i_{N-2}, i_{N-1} (pos_in_window)]
    # grad_out: [i0, i1, ..., i_{dim} * step + i_{N-1}, ..., i_{N-2}]

    # So for output:
    # - Index at position d (for d < dim): i_d
    # - Index at position dim: i_{dim} * step + i_{N-1}
    # - Index at position d (for d > dim): i_d

    # After padding to 8 dimensions (grad_in) and 7 dimensions (output strides):
    # The padding adds 1s at the front, so:
    # - grad_in dim 0 corresponds to padded position 8 - N
    # - grad_out dim 0 corresponds to padded position 8 - N (same offset)

    # dim_padded is the padded position of the 'dim' dimension
    # The 'size' dimension is always at position 7 (i7)

    # For output offset, we need:
    # offset = sum over d in [0, N-2) of (output_index[d] * gout_stride[d])

    # Using padded positions (0 to 6 for output strides):
    # For position p in output (padded):
    # - If p < dim_padded: use i_p as index
    # - If p == dim_padded: use i_p * step + i7 as index
    # - If p > dim_padded and p < 7: use i_p as index

    # Compute output offset
    idx0 = i0
    idx1 = i1
    idx2 = i2
    idx3 = i3
    idx4 = i4
    idx5 = i5
    idx6 = i6

    # Apply the transformation for dim_padded position
    idx0 = tl.where(dim_padded == 0, i0 * step + i7, idx0)
    idx1 = tl.where(dim_padded == 1, i1 * step + i7, idx1)
    idx2 = tl.where(dim_padded == 2, i2 * step + i7, idx2)
    idx3 = tl.where(dim_padded == 3, i3 * step + i7, idx3)
    idx4 = tl.where(dim_padded == 4, i4 * step + i7, idx4)
    idx5 = tl.where(dim_padded == 5, i5 * step + i7, idx5)
    idx6 = tl.where(dim_padded == 6, i6 * step + i7, idx6)

    out_offset = (
        idx0 * gout_st0
        + idx1 * gout_st1
        + idx2 * gout_st2
        + idx3 * gout_st3
        + idx4 * gout_st4
        + idx5 * gout_st5
        + idx6 * gout_st6
    )

    # Atomic add to output (handles overlapping indices)
    tl.atomic_add(grad_out_ptr + out_offset, grad_val, mask=mask, sem="relaxed")


def unfold_backward(grad_in, input_sizes: List[int], dim: int, size: int, step: int):
    """
    Backward pass for unfold operation.

    Args:
        grad_in: Gradient tensor from the forward unfold operation.
                 Shape: [S0, ..., S_{dim-1}, num_windows, S_{dim+1}, ..., S_{N-1}, size]
        input_sizes: Original input tensor shape [S0, ..., S_{dim}, ..., S_{N-1}]
        dim: Dimension along which unfold was performed
        size: Size of the sliding window
        step: Step between windows

    Returns:
        Gradient tensor of shape input_sizes
    """
    logger.debug("GEMS UNFOLD_BACKWARD")

    # Handle negative dim
    ndim_out = len(input_sizes)
    if dim < 0:
        dim = ndim_out + dim

    # Create output tensor (gradient w.r.t. input)
    grad_out = torch.zeros(input_sizes, dtype=grad_in.dtype, device=grad_in.device)

    if grad_in.numel() == 0:
        return grad_out

    # Make grad_in contiguous for easier indexing
    grad_in = grad_in.contiguous()

    # Handle dtype conversion for float16/bfloat16 (atomic_add may not support them)
    dtype_convert = False
    orig_dtype = grad_in.dtype
    if grad_in.dtype == torch.float16 or grad_in.dtype == torch.bfloat16:
        grad_in = grad_in.to(torch.float32)
        grad_out_compute = torch.zeros(
            input_sizes, dtype=torch.float32, device=grad_in.device
        )
        dtype_convert = True
    else:
        grad_out_compute = grad_out

    ndim_in = grad_in.ndim  # N dimensions (including 'size' dim at the end)

    # Pad grad_in shape to 8 dimensions (add 1s at the front)
    gin_shape = list(grad_in.shape)
    padding_count = 8 - ndim_in
    gin_shape_padded = [1] * padding_count + gin_shape

    # Adjust dim to padded position
    dim_padded = padding_count + dim

    # Pad grad_out strides to 7 dimensions (output has ndim_in - 1 dimensions)
    gout_strides = list(grad_out_compute.stride())
    out_padding_count = 7 - (ndim_in - 1)
    gout_strides_padded = [0] * out_padding_count + gout_strides

    numel = grad_in.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    _unfold_backward_kernel[grid](
        grad_in,
        grad_out_compute,
        numel,
        gin_shape_padded[0],
        gin_shape_padded[1],
        gin_shape_padded[2],
        gin_shape_padded[3],
        gin_shape_padded[4],
        gin_shape_padded[5],
        gin_shape_padded[6],
        gin_shape_padded[7],
        gout_strides_padded[0],
        gout_strides_padded[1],
        gout_strides_padded[2],
        gout_strides_padded[3],
        gout_strides_padded[4],
        gout_strides_padded[5],
        gout_strides_padded[6],
        dim_padded,
        step,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if dtype_convert:
        return grad_out_compute.to(orig_dtype)
    return grad_out_compute
