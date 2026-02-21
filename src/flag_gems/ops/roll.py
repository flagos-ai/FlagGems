"""
Roll operator implementation for FlagGems.

This module provides the roll operation using native Triton kernels.
Rolls the tensor elements along a given dimension by specified shifts.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# ==================== Basic roll Kernel (1D) ====================


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("roll_1d"), key=["n_elements"])
@triton.jit(do_not_specialize=["shift"])
def roll_1d_kernel(input_ptr, output_ptr, n_elements, shift, BLOCK_SIZE: tl.constexpr):
    """
    Native Triton kernel for 1D roll computation.

    Rolls elements along a 1D tensor.
    Example: [1, 2, 3, 4, 5], shift=2 -> [4, 5, 1, 2, 3]

    Key concept: Rolling right by shift positions means output[i] = input[(i - shift) % n]
    Or equivalently: output[(i + shift) % n] = input[i]

    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements in the tensor
        shift: Number of positions to roll
        BLOCK_SIZE: Block size for processing (compile-time constant)
    """
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Only process positions where offsets < n_elements
    mask = offsets < n_elements

    # Calculate source index after rolling
    # Use addition instead of subtraction to avoid negative modulo issues
    # output[i] = input[(i - shift + n) % n]
    # Equivalent to: output[i] = input[(i + (n - shift)) % n]
    src_offsets = (offsets + n_elements - shift) % n_elements

    # Load data from source position
    data = tl.load(input_ptr + src_offsets, mask=mask, other=0.0)

    # Store to output at target position
    tl.store(output_ptr + offsets, data, mask=mask)


def roll_1d(input, shift):
    """
    Triton implementation of 1D roll with autotune for optimal BLOCK_SIZE.

    Args:
        input: 1D tensor
        shift: Number of positions to roll

    Returns:
        Tensor: Rolled tensor with same shape as input
    """
    if input.dim() != 1:
        raise ValueError(f"roll_1d expects 1D tensor, got {input.dim()}D")

    N = input.numel()

    # Handle empty tensor
    if N == 0:
        return input.clone()

    out = torch.empty_like(input)

    # Handle negative shift
    shift = shift % N

    # Autotune will automatically select the best BLOCK_SIZE
    grid = lambda cfg: (triton.cdiv(N, cfg["BLOCK_SIZE"]),)

    roll_1d_kernel[grid](
        input,
        out,
        N,
        shift,
    )
    return out


# ==================== 2D Roll Kernel ====================


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("roll_2d"), key=["n_rows", "n_cols"])
@triton.jit(do_not_specialize=["shift", "target_dim"])
def roll_2d_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    shift,
    target_dim,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
    USE_BITWISE: tl.constexpr,
):
    """
    Optimized Triton kernel for 2D tensor roll using 1D grid.

    Uses a 1D grid where each program processes BLOCK_SIZE elements
    treating the tensor as a flattened 1D array. This avoids complex
    for loops and provides a consistent pattern with 3D/4D/5D kernels.

    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_rows: Number of rows in the matrix
        n_cols: Number of columns in the matrix
        shift: Number of positions to roll
        target_dim: 0 for rolling along rows, 1 for rolling along columns
        stride_row: Stride for row dimension
        stride_col: Stride for column dimension
        BLOCK_SIZE: Block size for processing (compile-time constant)
    """
    # Get flat index and convert to 2D coordinates
    flat_idx = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    flat_mask = flat_idx < (n_rows * n_cols)

    # Convert flat index to 2D coordinates
    row_idx = flat_idx // n_cols
    col_idx = flat_idx % n_cols

    # Calculate source indices for both dimensions, then select based on target_dim
    # For roll along rows (dim 0)
    if USE_BITWISE:
        src_row_idx_dim0 = (row_idx + n_rows - shift) & (n_rows - 1)
        src_col_idx_dim1 = (col_idx + n_cols - shift) & (n_cols - 1)
    else:
        src_row_idx_dim0 = (row_idx + n_rows - shift) % n_rows
        src_col_idx_dim1 = (col_idx + n_cols - shift) % n_cols

    # Calculate flat source indices for both cases
    src_flat_idx_dim0 = (
        src_row_idx_dim0 * stride_row + col_idx * stride_col
    )  # roll along dim 0
    src_flat_idx_dim1 = (
        row_idx * stride_row + src_col_idx_dim1 * stride_col
    )  # roll along dim 1

    # Select the correct source index based on target_dim
    src_flat_idx = tl.where(target_dim == 0, src_flat_idx_dim0, src_flat_idx_dim1)

    # Calculate flat destination index using strides
    dst_flat_idx = row_idx * stride_row + col_idx * stride_col

    # Load from source, store to destination
    input_ptrs = input_ptr + src_flat_idx
    output_ptrs = output_ptr + dst_flat_idx

    data = tl.load(input_ptrs, mask=flat_mask, other=0.0)
    tl.store(output_ptrs, data, mask=flat_mask)


def roll_2d(input, shift, dim):
    """
    Optimized Triton implementation of 2D roll using 1D grid.

    Args:
        input: 2D tensor
        shift: Number of positions to roll
        dim: 0 for rolling along rows, 1 for rolling along columns

    Returns:
        Tensor: Rolled tensor with same shape as input
    """
    if input.dim() != 2:
        raise ValueError(f"roll_2d expects 2D tensor, got {input.dim()}D")

    n_rows, n_cols = input.shape

    # Handle empty tensor
    if n_rows == 0 or n_cols == 0:
        return input.clone()

    # Get dimension size for target dimension
    dim_size = n_rows if dim == 0 else n_cols

    # Fast path 1: shift = 0 (no-op)
    if shift == 0:
        return input.clone()

    # Fast path 2: full rotation (shift % dim_size == 0)
    effective_shift = shift % dim_size
    if effective_shift == 0:
        return input.clone()

    # Handle negative shift (already computed as effective_shift)
    shift = effective_shift

    stride_row = input.stride(0)
    stride_col = input.stride(1)

    out = torch.empty_like(input)

    # Use 1D grid - treat tensor as flattened 1D array
    n_elements = n_rows * n_cols
    grid = lambda cfg: (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)

    roll_2d_kernel[grid](input, out, n_rows, n_cols, shift, dim, stride_row, stride_col)

    return out


# ==================== General ND Roll Kernel ====================
def roll_nd_decomposed(input, shift, dim):
    """
    Roll operation for high-dimensional tensors (6D+) using decomposition strategy.

    Decomposition strategy:
    - 6D: Decompose to 3D + 3D
    - 7D: Decompose to 3D + 4D
    - 8D: Decompose to 4D + 4D
    - 9D: Decompose to 4D + 5D
    - 10D+: Recursive decomposition

    Core idea: Reshape N-dimensional tensor to lower-dimensional tensor,
    apply roll operation, then reshape back to original shape.
    Example: 6D (d0, d1, d2, d3, d4, d5) rolling along dim=2:
    - Flatten first 2 dims and last 3 dims to get 3D tensor (d0*d1, d2, d3*d4*d5)
    - Use 3D kernel to roll along middle dimension
    - Reshape back to original shape

    Args:
        input: 6D+ tensor
        shift: Number of positions to roll
        dim: Dimension along which to roll

    Returns:
        Tensor: Rolled tensor with same shape as input
    """
    if dim < 0:
        dim = input.dim() + dim

    original_shape = input.shape
    ndim = input.dim()

    # Handle empty tensor (dimension size is 0)
    dim_size = input.shape[dim]
    if dim_size == 0:
        return input.clone()

    # Handle case when shift is 0
    if shift == 0:
        return input.clone()

    shift = shift % dim_size
    if shift == 0:
        return input.clone()

    # Select decomposition strategy based on dimension count
    if ndim == 6:
        # 6D decomposition to 3D
        # Strategy: Ensure target dimension is independent, not flattened
        if dim == 0:
            # Target is dim0, reshape to (d0, d1*d2, d3*d4*d5)
            group1_size = original_shape[0]
            group2_size = original_shape[1] * original_shape[2]
            group3_size = original_shape[3] * original_shape[4] * original_shape[5]
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 0

        elif dim == 1:
            # Target is dim1, reshape to (d0, d1, d2*d3*d4*d5)
            group1_size = original_shape[0]
            group2_size = original_shape[1]
            group3_size = (
                original_shape[2]
                * original_shape[3]
                * original_shape[4]
                * original_shape[5]
            )
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1

        elif dim == 2:
            # Target is dim2, reshape to (d0*d1, d2, d3*d4*d5)
            group1_size = original_shape[0] * original_shape[1]
            group2_size = original_shape[2]
            group3_size = original_shape[3] * original_shape[4] * original_shape[5]
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1

        elif dim == 3:
            # Target is dim3, reshape to (d0*d1*d2, d3, d4*d5)
            group1_size = original_shape[0] * original_shape[1] * original_shape[2]
            group2_size = original_shape[3]
            group3_size = original_shape[4] * original_shape[5]
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1

        elif dim == 4:
            # Target is dim4, reshape to (d0*d1*d2*d3, d4, d5)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
            )
            group2_size = original_shape[4]
            group3_size = original_shape[5]
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1

        else:  # dim == 5
            # Target is dim5, reshape to (d0*d1*d2*d3*d4, d5, 1)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
                * original_shape[4]
            )
            group2_size = original_shape[5]
            group3_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1

        result = roll_3d(reshaped, shift, new_dim)
        return result.reshape(original_shape)

    elif ndim == 7:
        # 7D decomposition to 3D or 4D
        # Strategy: Ensure target dimension is independent
        if dim == 0:
            # Target dim0: reshape to (d0, d1*d2, d3*d4*d5*d6)
            group1_size = original_shape[0]
            group2_size = original_shape[1] * original_shape[2]
            group3_size = (
                original_shape[3]
                * original_shape[4]
                * original_shape[5]
                * original_shape[6]
            )
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 0
            result = roll_3d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 1:
            # Target dim1: reshape to (d0, d1, d2*d3*d4*d5*d6)
            group1_size = original_shape[0]
            group2_size = original_shape[1]
            group3_size = (
                original_shape[2]
                * original_shape[3]
                * original_shape[4]
                * original_shape[5]
                * original_shape[6]
            )
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1
            result = roll_3d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 2:
            # Target dim2: reshape to (d0*d1, d2, d3*d4*d5*d6)
            group1_size = original_shape[0] * original_shape[1]
            group2_size = original_shape[2]
            group3_size = (
                original_shape[3]
                * original_shape[4]
                * original_shape[5]
                * original_shape[6]
            )
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1
            result = roll_3d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 3:
            # Target dim3: reshape to (d0*d1*d2, d3, d4*d5*d6)
            group1_size = original_shape[0] * original_shape[1] * original_shape[2]
            group2_size = original_shape[3]
            group3_size = original_shape[4] * original_shape[5] * original_shape[6]
            reshaped = input.reshape(group1_size, group2_size, group3_size)
            new_dim = 1
            result = roll_3d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 4:
            # Target dim4: reshape to 4D (d0*d1*d2*d3, d4, d5, d6)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
            )
            group2_size = original_shape[4]
            group3_size = original_shape[5]
            group4_size = original_shape[6]
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 5:
            # Target dim5: reshape to 4D (d0*d1*d2*d3*d4, d5, d6, 1)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
                * original_shape[4]
            )
            group2_size = original_shape[5]
            group3_size = original_shape[6]
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        else:  # dim == 6
            # Target dim6: reshape to 4D (d0*d1*d2*d3*d4*d5, d6, 1, 1)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
                * original_shape[4]
                * original_shape[5]
            )
            group2_size = original_shape[6]
            group3_size = 1
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

    elif ndim == 8:
        # 8D decomposition to 4D
        # Strategy: Ensure target dimension is independent
        if dim == 0:
            # Target dim0: reshape to (d0, d1*d2*d3, d4*d5*d6*d7, 1)
            group1_size = original_shape[0]
            group2_size = original_shape[1] * original_shape[2] * original_shape[3]
            group3_size = (
                original_shape[4]
                * original_shape[5]
                * original_shape[6]
                * original_shape[7]
            )
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 0
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 1:
            # Target dim1: reshape to (d0, d1, d2*d3, d4*d5*d6*d7)
            group1_size = original_shape[0]
            group2_size = original_shape[1]
            group3_size = original_shape[2] * original_shape[3]
            group4_size = (
                original_shape[4]
                * original_shape[5]
                * original_shape[6]
                * original_shape[7]
            )
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 2:
            # Target dim2: reshape to (d0*d1, d2, d3, d4*d5*d6*d7)
            group1_size = original_shape[0] * original_shape[1]
            group2_size = original_shape[2]
            group3_size = original_shape[3]
            group4_size = (
                original_shape[4]
                * original_shape[5]
                * original_shape[6]
                * original_shape[7]
            )
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 3:
            # Target dim3: reshape to (d0*d1*d2, d3, d4*d5*d6*d7, 1)
            group1_size = original_shape[0] * original_shape[1] * original_shape[2]
            group2_size = original_shape[3]
            group3_size = (
                original_shape[4]
                * original_shape[5]
                * original_shape[6]
                * original_shape[7]
            )
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 4:
            # Target dim4: reshape to (d0*d1*d2*d3, d4, d5*d6*d7, 1)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
            )
            group2_size = original_shape[4]
            group3_size = original_shape[5] * original_shape[6] * original_shape[7]
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 5:
            # Target dim5: reshape to (d0*d1*d2*d3*d4, d5, d6*d7, 1)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
                * original_shape[4]
            )
            group2_size = original_shape[5]
            group3_size = original_shape[6] * original_shape[7]
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        elif dim == 6:
            # Target dim6: reshape to (d0*d1*d2*d3*d4*d5, d6, d7, 1)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
                * original_shape[4]
                * original_shape[5]
            )
            group2_size = original_shape[6]
            group3_size = original_shape[7]
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

        else:  # dim == 7
            # Target dim7: reshape to (d0*d1*d2*d3*d4*d5*d6, d7, 1, 1)
            group1_size = (
                original_shape[0]
                * original_shape[1]
                * original_shape[2]
                * original_shape[3]
                * original_shape[4]
                * original_shape[5]
                * original_shape[6]
            )
            group2_size = original_shape[7]
            group3_size = 1
            group4_size = 1
            reshaped = input.reshape(group1_size, group2_size, group3_size, group4_size)
            new_dim = 1
            result = roll_4d(reshaped, shift, new_dim)
            return result.reshape(original_shape)

    else:  # ndim >= 9
        # For 9D+, use recursive decomposition
        # Split tensor into two parts: first k dims and last (ndim-k) dims
        # Choose k so both parts are in reasonable range
        if ndim == 9:
            # 9D = 4D + 5D
            split_point = 4
        elif ndim == 10:
            # 10D = 5D + 5D
            split_point = 5
        else:
            # For higher dimensions, split into (ndim//2) and (ndim - ndim//2)
            split_point = ndim // 2

        if dim < split_point:
            # Target dimension is in the first half
            # Flatten the second half
            front_dims = original_shape[:split_point]
            back_dims = original_shape[split_point:]

            # Calculate flattened size
            back_flat_size = 1
            for d in back_dims:
                back_flat_size *= d

            # reshape to (front_dims..., back_flat_size)
            new_shape = list(front_dims) + [back_flat_size]
            reshaped = input.reshape(new_shape)

            # Check dimension count after reshape, use specialized kernel if 3D/4D/5D
            new_ndim = len(new_shape)
            if new_ndim == 3:
                result = roll_3d(reshaped, shift, dim)
            elif new_ndim == 4:
                result = roll_4d(reshaped, shift, dim)
            elif new_ndim == 5:
                result = roll_5d(reshaped, shift, dim)
            else:
                # Recursive call
                result = roll_nd_decomposed(reshaped, shift, dim)
            return result.reshape(original_shape)
        else:
            # Target dimension is in the second half
            # Flatten the first half
            front_dims = original_shape[:split_point]
            back_dims = original_shape[split_point:]

            # Calculate flattened size
            front_flat_size = 1
            for d in front_dims:
                front_flat_size *= d

            # reshape to (front_flat_size, back_dims...)
            new_shape = [front_flat_size] + list(back_dims)
            reshaped = input.reshape(new_shape)

            # Adjust dimension index:
            # Relative position of dim in second half is (dim - split_point)
            # But there's one flattened dimension before, so new_dim = (dim - split_point) + 1
            new_dim = (dim - split_point) + 1

            # Check dimension count after reshape, use specialized kernel if 3D/4D/5D
            new_ndim = len(new_shape)
            if new_ndim == 3:
                result = roll_3d(reshaped, shift, new_dim)
            elif new_ndim == 4:
                result = roll_4d(reshaped, shift, new_dim)
            elif new_ndim == 5:
                result = roll_5d(reshaped, shift, new_dim)
            else:
                # Recursive call
                result = roll_nd_decomposed(reshaped, shift, new_dim)
            return result.reshape(original_shape)


# Tensor Core implementation removed
# roll_tensor_core_kernel and roll_2d_tensor_core have been removed


def roll_nd_along_dim(input, shift, dim):
    """
    Roll an N-D tensor along the specified dimension.

    For 2D tensors, use specialized 2D kernel
    For 3D/4D/5D, use specialized ND kernels
    For higher dimensions (6D+), use decomposition strategy

    Args:
        input: N-D tensor
        shift: Number of positions to roll
        dim: Dimension along which to roll

    Returns:
        Tensor: Rolled tensor with same shape as input
    """
    if dim < 0:
        dim = input.dim() + dim

    # Handle empty tensor (dimension size is 0)
    dim_size = input.shape[dim]
    if dim_size == 0:
        return input.clone()

    # Handle case when shift is 0
    if shift == 0:
        return input.clone()

    # Handle negative shift and large shift
    shift = shift % dim_size
    if shift == 0:
        return input.clone()

    # For 1D tensors, use roll_1d
    if input.dim() == 1:
        return roll_1d(input, shift)

    # For 2D tensors, use specialized 2D kernel (more efficient)
    if input.dim() == 2:
        return roll_2d(input, shift, dim)

    # For 3D/4D/5D, use specialized ND kernels
    if input.dim() == 3:
        return roll_3d(input, shift, dim)
    elif input.dim() == 4:
        return roll_4d(input, shift, dim)
    elif input.dim() == 5:
        return roll_5d(input, shift, dim)

    # For higher dimensions (6D+), use decomposition strategy
    return roll_nd_decomposed(input, shift, dim)


# ==================== 3D Roll Kernel ====================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("roll_3d"), key=["n_dim0", "n_dim1", "n_dim2"]
)
@triton.jit(do_not_specialize=["shift", "target_dim"])
def roll_3d_kernel(
    input_ptr,
    output_ptr,
    n_dim0,
    n_dim1,
    n_dim2,
    shift,
    target_dim,
    stride_dim0,
    stride_dim1,
    stride_dim2,
    BLOCK_SIZE: tl.constexpr,
    USE_BITWISE: tl.constexpr,
):
    """
    Optimized Triton kernel for 3D tensor roll using 1D grid.

    Uses a 1D grid where each program processes BLOCK_SIZE elements
    treating the tensor as a flattened 1D array. This avoids nested loops.

    Grid: flattened tensor size / BLOCK_SIZE
    """
    # Get flat index and convert to 3D coordinates
    flat_idx = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    flat_mask = flat_idx < (n_dim0 * n_dim1 * n_dim2)

    # Convert flat index to 3D coordinates
    dim0_idx = flat_idx // (n_dim1 * n_dim2)
    dim12_flat = flat_idx % (n_dim1 * n_dim2)
    dim1_idx = dim12_flat // n_dim2
    dim2_idx = dim12_flat % n_dim2

    # Calculate source indices for all three dimensions
    if USE_BITWISE:
        src_dim0_idx = (dim0_idx + n_dim0 - shift) & (n_dim0 - 1)
        src_dim1_idx = (dim1_idx + n_dim1 - shift) & (n_dim1 - 1)
        src_dim2_idx = (dim2_idx + n_dim2 - shift) & (n_dim2 - 1)
    else:
        src_dim0_idx = (dim0_idx + n_dim0 - shift) % n_dim0
        src_dim1_idx = (dim1_idx + n_dim1 - shift) % n_dim1
        src_dim2_idx = (dim2_idx + n_dim2 - shift) % n_dim2

    # Calculate flat source indices for each dimension
    src_flat_idx_dim0 = (
        src_dim0_idx * stride_dim0 + dim1_idx * stride_dim1 + dim2_idx * stride_dim2
    )
    src_flat_idx_dim1 = (
        dim0_idx * stride_dim0 + src_dim1_idx * stride_dim1 + dim2_idx * stride_dim2
    )
    src_flat_idx_dim2 = (
        dim0_idx * stride_dim0 + dim1_idx * stride_dim1 + src_dim2_idx * stride_dim2
    )

    # Select the correct source index based on target_dim
    src_flat_idx = tl.where(
        target_dim == 0,
        src_flat_idx_dim0,
        tl.where(target_dim == 1, src_flat_idx_dim1, src_flat_idx_dim2),
    )

    # Calculate flat destination index using strides
    dst_flat_idx = (
        dim0_idx * stride_dim0 + dim1_idx * stride_dim1 + dim2_idx * stride_dim2
    )

    # Load from source, store to destination
    input_ptrs = input_ptr + src_flat_idx
    output_ptrs = output_ptr + dst_flat_idx

    data = tl.load(input_ptrs, mask=flat_mask, other=0.0)
    tl.store(output_ptrs, data, mask=flat_mask)


def roll_3d(input, shift, dim):
    """
    Triton implementation of 3D roll with autotune for optimal BLOCK_SIZE.

    Args:
        input: 3D tensor
        shift: Number of positions to roll
        dim: Dimension along which to roll (0, 1, or 2)

    Returns:
        Tensor: Rolled tensor with same shape as input
    """
    if input.dim() != 3:
        raise ValueError(f"roll_3d expects 3D tensor, got {input.dim()}D")

    n_dim0, n_dim1, n_dim2 = input.shape

    # Handle empty tensor
    if n_dim0 == 0 or n_dim1 == 0 or n_dim2 == 0:
        return input.clone()

    # Get dimension size for target dimension
    dim_size = n_dim0 if dim == 0 else (n_dim1 if dim == 1 else n_dim2)

    # Fast path 1: shift = 0 (no-op)
    if shift == 0:
        return input.clone()

    # Fast path 2: full rotation (shift % dim_size == 0)
    effective_shift = shift % dim_size
    if effective_shift == 0:
        return input.clone()

    # Handle negative shift (already computed as effective_shift)
    shift = effective_shift

    stride_dim0 = input.stride(0)
    stride_dim1 = input.stride(1)
    stride_dim2 = input.stride(2)

    out = torch.empty_like(input)

    # Use 1D grid - treat tensor as flattened 1D array
    # Grid size = total tensor size / BLOCK_SIZE
    n_elements = n_dim0 * n_dim1 * n_dim2

    # For 1D grid, pass as single-element tuple
    grid = lambda cfg: (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)

    roll_3d_kernel[grid](
        input,
        out,
        n_dim0,
        n_dim1,
        n_dim2,
        shift,
        dim,
        stride_dim0,
        stride_dim1,
        stride_dim2,
    )

    return out


# ==================== 4D Roll Kernel ====================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("roll_4d"),
    key=["n_dim0", "n_dim1", "n_dim2", "n_dim3"],
)
@triton.jit(do_not_specialize=["shift", "target_dim"])
def roll_4d_kernel(
    input_ptr,
    output_ptr,
    n_dim0,
    n_dim1,
    n_dim2,
    n_dim3,
    shift,
    target_dim,
    stride_dim0,
    stride_dim1,
    stride_dim2,
    stride_dim3,
    BLOCK_SIZE: tl.constexpr,
    USE_BITWISE: tl.constexpr,
):
    """
    Optimized Triton kernel for 4D tensor roll using 1D grid.

    Uses a 1D grid where each program processes BLOCK_SIZE elements
    treating the tensor as a flattened 1D array. This avoids nested loops.

    Grid: flattened tensor size / BLOCK_SIZE
    """
    # Get flat index and convert to 4D coordinates
    flat_idx = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    flat_mask = flat_idx < (n_dim0 * n_dim1 * n_dim2 * n_dim3)

    # Convert flat index to 4D coordinates
    dim0_idx = flat_idx // (n_dim1 * n_dim2 * n_dim3)
    dim123_flat = flat_idx % (n_dim1 * n_dim2 * n_dim3)
    dim1_idx = dim123_flat // (n_dim2 * n_dim3)
    dim23_flat = dim123_flat % (n_dim2 * n_dim3)
    dim2_idx = dim23_flat // n_dim3
    dim3_idx = dim23_flat % n_dim3

    # Calculate source indices for all four dimensions
    if USE_BITWISE:
        src_dim0_idx = (dim0_idx + n_dim0 - shift) & (n_dim0 - 1)
        src_dim1_idx = (dim1_idx + n_dim1 - shift) & (n_dim1 - 1)
        src_dim2_idx = (dim2_idx + n_dim2 - shift) & (n_dim2 - 1)
        src_dim3_idx = (dim3_idx + n_dim3 - shift) & (n_dim3 - 1)
    else:
        src_dim0_idx = (dim0_idx + n_dim0 - shift) % n_dim0
        src_dim1_idx = (dim1_idx + n_dim1 - shift) % n_dim1
        src_dim2_idx = (dim2_idx + n_dim2 - shift) % n_dim2
        src_dim3_idx = (dim3_idx + n_dim3 - shift) % n_dim3

    # Calculate flat source indices for each dimension
    src_flat_idx_dim0 = (
        src_dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
    )
    src_flat_idx_dim1 = (
        dim0_idx * stride_dim0
        + src_dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
    )
    src_flat_idx_dim2 = (
        dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + src_dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
    )
    src_flat_idx_dim3 = (
        dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + src_dim3_idx * stride_dim3
    )

    # Select the correct source index based on target_dim
    src_flat_idx = tl.where(
        target_dim == 0,
        src_flat_idx_dim0,
        tl.where(
            target_dim == 1,
            src_flat_idx_dim1,
            tl.where(target_dim == 2, src_flat_idx_dim2, src_flat_idx_dim3),
        ),
    )

    # Calculate flat destination index using strides
    dst_flat_idx = (
        dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
    )

    # Load from source, store to destination
    input_ptrs = input_ptr + src_flat_idx
    output_ptrs = output_ptr + dst_flat_idx

    data = tl.load(input_ptrs, mask=flat_mask, other=0.0)
    tl.store(output_ptrs, data, mask=flat_mask)


def roll_4d(input, shift, dim):
    """
    Triton implementation of 4D roll with autotune for optimal BLOCK_SIZE.

    Args:
        input: 4D tensor (batch, channels, height, width)
        shift: Number of positions to roll
        dim: Dimension along which to roll

    Returns:
        Tensor: Rolled tensor with same shape as input
    """
    if input.dim() != 4:
        raise ValueError(f"roll_4d expects 4D tensor, got {input.dim()}D")

    n_dim0, n_dim1, n_dim2, n_dim3 = input.shape

    # Handle empty tensor
    if n_dim0 == 0 or n_dim1 == 0 or n_dim2 == 0 or n_dim3 == 0:
        return input.clone()

    # Get dimension size for target dimension
    dim_size = input.shape[dim]

    # Fast path 1: shift = 0 (no-op)
    if shift == 0:
        return input.clone()

    # Fast path 2: full rotation (shift % dim_size == 0)
    effective_shift = shift % dim_size
    if effective_shift == 0:
        return input.clone()

    # Handle negative shift (already computed as effective_shift)
    shift = effective_shift

    stride_dim0 = input.stride(0)
    stride_dim1 = input.stride(1)
    stride_dim2 = input.stride(2)
    stride_dim3 = input.stride(3)

    out = torch.empty_like(input)

    # Use 1D grid - treat tensor as flattened 1D array
    n_elements = n_dim0 * n_dim1 * n_dim2 * n_dim3
    grid = lambda cfg: (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)

    roll_4d_kernel[grid](
        input,
        out,
        n_dim0,
        n_dim1,
        n_dim2,
        n_dim3,
        shift,
        dim,
        stride_dim0,
        stride_dim1,
        stride_dim2,
        stride_dim3,
    )

    return out


# ==================== 5D Roll Kernel ====================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("roll_5d"),
    key=["n_dim0", "n_dim1", "n_dim2", "n_dim3", "n_dim4"],
)
@triton.jit(do_not_specialize=["shift", "target_dim"])
def roll_5d_kernel(
    input_ptr,
    output_ptr,
    n_dim0,
    n_dim1,
    n_dim2,
    n_dim3,
    n_dim4,
    shift,
    target_dim,
    stride_dim0,
    stride_dim1,
    stride_dim2,
    stride_dim3,
    stride_dim4,
    BLOCK_SIZE: tl.constexpr,
    USE_BITWISE: tl.constexpr,
):
    """
    Optimized Triton kernel for 5D tensor roll using 1D grid.

    Uses a 1D grid where each program processes BLOCK_SIZE elements
    treating the tensor as a flattened 1D array. This avoids nested loops.

    Grid: flattened tensor size / BLOCK_SIZE
    """
    # Get flat index and convert to 5D coordinates
    flat_idx = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    flat_mask = flat_idx < (n_dim0 * n_dim1 * n_dim2 * n_dim3 * n_dim4)

    # Convert flat index to 5D coordinates
    dim0_idx = flat_idx // (n_dim1 * n_dim2 * n_dim3 * n_dim4)
    dim1234_flat = flat_idx % (n_dim1 * n_dim2 * n_dim3 * n_dim4)
    dim1_idx = dim1234_flat // (n_dim2 * n_dim3 * n_dim4)
    dim234_flat = dim1234_flat % (n_dim2 * n_dim3 * n_dim4)
    dim2_idx = dim234_flat // (n_dim3 * n_dim4)
    dim34_flat = dim234_flat % (n_dim3 * n_dim4)
    dim3_idx = dim34_flat // n_dim4
    dim4_idx = dim34_flat % n_dim4

    # Calculate source indices for all five dimensions
    if USE_BITWISE:
        src_dim0_idx = (dim0_idx + n_dim0 - shift) & (n_dim0 - 1)
        src_dim1_idx = (dim1_idx + n_dim1 - shift) & (n_dim1 - 1)
        src_dim2_idx = (dim2_idx + n_dim2 - shift) & (n_dim2 - 1)
        src_dim3_idx = (dim3_idx + n_dim3 - shift) & (n_dim3 - 1)
        src_dim4_idx = (dim4_idx + n_dim4 - shift) & (n_dim4 - 1)
    else:
        src_dim0_idx = (dim0_idx + n_dim0 - shift) % n_dim0
        src_dim1_idx = (dim1_idx + n_dim1 - shift) % n_dim1
        src_dim2_idx = (dim2_idx + n_dim2 - shift) % n_dim2
        src_dim3_idx = (dim3_idx + n_dim3 - shift) % n_dim3
        src_dim4_idx = (dim4_idx + n_dim4 - shift) % n_dim4

    # Calculate flat source indices for each dimension
    src_flat_idx_dim0 = (
        src_dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
        + dim4_idx * stride_dim4
    )
    src_flat_idx_dim1 = (
        dim0_idx * stride_dim0
        + src_dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
        + dim4_idx * stride_dim4
    )
    src_flat_idx_dim2 = (
        dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + src_dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
        + dim4_idx * stride_dim4
    )
    src_flat_idx_dim3 = (
        dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + src_dim3_idx * stride_dim3
        + dim4_idx * stride_dim4
    )
    src_flat_idx_dim4 = (
        dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
        + src_dim4_idx * stride_dim4
    )

    # Select the correct source index based on target_dim
    src_flat_idx = tl.where(
        target_dim == 0,
        src_flat_idx_dim0,
        tl.where(
            target_dim == 1,
            src_flat_idx_dim1,
            tl.where(
                target_dim == 2,
                src_flat_idx_dim2,
                tl.where(target_dim == 3, src_flat_idx_dim3, src_flat_idx_dim4),
            ),
        ),
    )

    # Calculate flat destination index using strides
    dst_flat_idx = (
        dim0_idx * stride_dim0
        + dim1_idx * stride_dim1
        + dim2_idx * stride_dim2
        + dim3_idx * stride_dim3
        + dim4_idx * stride_dim4
    )

    # Load from source, store to destination
    input_ptrs = input_ptr + src_flat_idx
    output_ptrs = output_ptr + dst_flat_idx

    data = tl.load(input_ptrs, mask=flat_mask, other=0.0)
    tl.store(output_ptrs, data, mask=flat_mask)


def roll_5d(input, shift, dim):
    """
    Triton implementation of 5D roll with autotune for optimal BLOCK_SIZE.

    Args:
        input: 5D tensor (batch, time, channel, height, width)
        shift: Number of positions to roll
        dim: Dimension along which to roll

    Returns:
        Tensor: Rolled tensor with same shape as input
    """
    if input.dim() != 5:
        raise ValueError(f"roll_5d expects 5D tensor, got {input.dim()}D")

    n_dim0, n_dim1, n_dim2, n_dim3, n_dim4 = input.shape

    # Handle empty tensor
    if n_dim0 == 0 or n_dim1 == 0 or n_dim2 == 0 or n_dim3 == 0 or n_dim4 == 0:
        return input.clone()

    # Get dimension size for target dimension
    dim_size = input.shape[dim]

    # Fast path 1: shift = 0 (no-op)
    if shift == 0:
        return input.clone()

    # Fast path 2: full rotation (shift % dim_size == 0)
    effective_shift = shift % dim_size
    if effective_shift == 0:
        return input.clone()

    # Handle negative shift (already computed as effective_shift)
    shift = effective_shift

    stride_dim0 = input.stride(0)
    stride_dim1 = input.stride(1)
    stride_dim2 = input.stride(2)
    stride_dim3 = input.stride(3)
    stride_dim4 = input.stride(4)

    out = torch.empty_like(input)

    # Use 1D grid - treat tensor as flattened 1D array
    n_elements = n_dim0 * n_dim1 * n_dim2 * n_dim3 * n_dim4
    grid = lambda cfg: (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)

    roll_5d_kernel[grid](
        input,
        out,
        n_dim0,
        n_dim1,
        n_dim2,
        n_dim3,
        n_dim4,
        shift,
        dim,
        stride_dim0,
        stride_dim1,
        stride_dim2,
        stride_dim3,
        stride_dim4,
    )

    return out


# ==================== General Roll Function ====================


def roll(input, shifts, dims=None):
    """
    Triton implementation of torch.roll supporting arbitrary dimensions.

    This function provides the roll operation for FlagGems, optimized
    for GPU execution using native Triton kernels.

    Args:
        input: Input tensor of any dimension
        shifts: int or tuple of ints, number of positions to roll
        dims: int or tuple of ints, dimensions along which to roll

    Returns:
        Tensor: Rolled tensor with same shape as input

    Examples:
        >>> import torch
        >>> from flag_gems import roll
        >>> x = torch.randn(3, 4, 5).cuda()
        >>> roll(x, shifts=2)                    # Roll flattened
        >>> roll(x, shifts=2, dims=0)            # Roll along dimension 0
        >>> roll(x, shifts=(1, 2), dims=(0, 1))  # Roll along multiple dimensions
        >>> roll(x, shifts=3, dims=-1)           # Roll along last dimension
    """
    input = input.contiguous()
    # Handle dims=None case: roll after flattening
    if dims is None:
        flat_input = input.flatten()
        flat_output = roll_1d(flat_input, shifts)
        return flat_output.reshape(input.shape)

    # Normalize parameters to tuples
    if isinstance(dims, int):
        dims = (dims,)
    if isinstance(shifts, int):
        shifts = (shifts,)

    if len(dims) != len(shifts):
        raise ValueError(
            f"shifts and dims must have same length, got {len(shifts)} and {len(dims)}"
        )

    # Handle negative dimension indices
    normalized_dims = []
    for dim in dims:
        if dim < 0:
            dim = input.dim() + dim
        if dim < 0 or dim >= input.dim():
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-input.dim()}, {input.dim()-1}], but got {dim})"
            )
        normalized_dims.append(dim)

    # Apply roll for each dimension sequentially
    result = input
    for shift, dim in zip(shifts, normalized_dims):
        result = roll_nd_along_dim(result, shift, dim)

    return result
