"""
Grid sample operator implementation for FlagGems.

This module provides the grid sampling operation with various interpolation modes.
Grid sample computes the output using input values and pixel locations from grid.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _validate_grid_sample_input(input, grid, mode, padding_mode):
    """
    Validate input tensors and parameters for grid_sample.

    Args:
        input: Input tensor
        grid: Grid tensor
        mode: Interpolation mode
        padding_mode: Padding mode

    Raises:
        ValueError: If inputs or parameters are invalid
    """
    if input.dim() not in [4, 5]:
        raise ValueError("Input must be 4D or 5D")

    if input.dim() == 4 and grid.dim() != 4:
        raise ValueError(
            "For 4D input, grid must be 4D (N, H_out, W_out, 2), "
            f"but got {grid.dim()}D tensor"
        )

    if input.dim() == 5 and grid.dim() != 5:
        raise ValueError(
            f"For 5D input, grid must be 5D (N, D_out, H_out, W_out, 3), "
            f"but got {grid.dim()}D tensor"
        )

    if input.dim() == 4 and grid.shape[-1] != 2:
        raise ValueError(
            f"For 4D input, grid must have 2 coordinates in last dimension, "
            f"but got {grid.shape[-1]}"
        )

    if input.dim() == 5 and grid.shape[-1] != 3:
        raise ValueError(
            f"For 5D input, grid must have 3 coordinates in last dimension, "
            f"but got {grid.shape[-1]}"
        )

    if input.shape[0] != grid.shape[0]:
        raise ValueError(
            f"Input and grid must have same batch size, "
            f"but got {input.shape[0]} and {grid.shape[0]}"
        )

    valid_modes = ["bilinear", "nearest", "bicubic"]
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected one of {valid_modes}, "
            f"but note: bicubic only supports 4D input"
        )

    if mode == "bicubic" and input.dim() == 5:
        raise ValueError("Bicubic interpolation only supports 4D input")

    valid_padding_modes = ["zeros", "border", "reflection"]
    if padding_mode not in valid_padding_modes:
        raise ValueError(
            f"Invalid padding_mode '{padding_mode}'. Expected one of {valid_padding_modes}"
        )


# ============================================================================
# 2D Nearest Neighbor Kernels
# ============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_nearest"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_nearest_zeros_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D nearest neighbor interpolation with zeros padding.

    For each output pixel, this kernel:
    1. Loads the grid coordinates (normalized to [-1, 1])
    2. Transforms coordinates to pixel space
    3. Rounds to nearest pixel location
    4. Loads the input pixel (or 0 if out of bounds)
    5. Stores to output

    Args:
        ptr_output: Pointer to output tensor
        ptr_input: Pointer to input tensor
        ptr_grid: Pointer to grid tensor
        N: Batch size
        C: Number of channels
        H_in: Input height
        W_in: Input width
        H_out: Output height
        W_out: Output width
        align_corners: Whether to align corners
        BLOCK_SIZE: Block size for tuning
    """
    # Each program instance handles one output pixel (for all channels)
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates for this output location
    # Grid shape: (N, H_out, W_out, 2)
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN - use sentinel value -2.0 (outside valid grid range [-1, 1])
    # We'll detect this and return 0.0 for NaN values
    grid_x_nan = grid_x != grid_x  # True if NaN
    grid_y_nan = grid_y != grid_y  # True if NaN
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        # Pixel centers at -1 and 1
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        # Use banker's rounding (round half to even) for align_corners=True too
        x_floor = tl.floor(x)
        y_floor = tl.floor(y)
        x_frac = x - x_floor
        y_frac = y - y_floor
        x_is_half = x_frac == 0.5
        y_is_half = y_frac == 0.5
        x_floor_int = tl.cast(x_floor, tl.int32)
        y_floor_int = tl.cast(y_floor, tl.int32)
        x_is_even = x_floor_int % 2 == 0
        y_is_even = y_floor_int % 2 == 0
        x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
        y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
        x_idx = tl.cast(
            tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
            tl.int32,
        )
        y_idx = tl.cast(
            tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
            tl.int32,
        )
        # Check bounds (align_corners=True: valid range is [0, W_in) x [0, H_in))
        # Also check for NaN (sentinel value -2.0)
        mask = (
            (x_idx >= 0)
            & (x_idx < W_in)
            & (y_idx >= 0)
            & (y_idx < H_in)
            & ~grid_x_nan
            & ~grid_y_nan
        )
    else:
        # Pixel corners at -1 and 1
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        # Use banker's rounding (round half to even) for align_corners=False
        x_floor = tl.floor(x)
        y_floor = tl.floor(y)
        x_frac = x - x_floor
        y_frac = y - y_floor
        x_is_half = x_frac == 0.5
        y_is_half = y_frac == 0.5
        x_floor_int = tl.cast(x_floor, tl.int32)
        y_floor_int = tl.cast(y_floor, tl.int32)
        x_is_even = x_floor_int % 2 == 0
        y_is_even = y_floor_int % 2 == 0
        x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
        y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
        x_idx = tl.cast(
            tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
            tl.int32,
        )
        y_idx = tl.cast(
            tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
            tl.int32,
        )

        # Check bounds (align_corners=False)
        # Also check for NaN (sentinel value -2.0)
        mask = (
            (x_idx >= 0)
            & (x_idx < W_in)
            & (y_idx >= 0)
            & (y_idx < H_in)
            & ~grid_x_nan
            & ~grid_y_nan
        )

    # Input shape: (N, C, H_in, W_in)
    input_offset = n * C * H_in * W_in + c * H_in * W_in + y_idx * W_in + x_idx
    val = tl.load(ptr_input + input_offset, mask=mask, other=0.0).to(tl.float32)

    # Store output
    # Output shape: (N, C, H_out, W_out)
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_nearest"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_nearest_border_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D nearest neighbor interpolation with border padding.

    Out-of-bound coordinates are clamped to the border.
    """
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN
    grid_x = tl.where(grid_x != grid_x, -1.0, grid_x)
    grid_y = tl.where(grid_y != grid_y, -1.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        # Use banker's rounding (round half to even)
        x_floor = tl.floor(x)
        y_floor = tl.floor(y)
        x_frac = x - x_floor
        y_frac = y - y_floor
        x_is_half = x_frac == 0.5
        y_is_half = y_frac == 0.5
        x_floor_int = tl.cast(x_floor, tl.int32)
        y_floor_int = tl.cast(y_floor, tl.int32)
        x_is_even = x_floor_int % 2 == 0
        y_is_even = y_floor_int % 2 == 0
        x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
        y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
        x_idx_unclamped = tl.cast(
            tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
            tl.int32,
        )
        y_idx_unclamped = tl.cast(
            tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
            tl.int32,
        )
        # For align_corners=True: clamp to [0, W_in-1]
        x_idx = tl.maximum(0, tl.minimum(x_idx_unclamped, W_in - 1))
        y_idx = tl.maximum(0, tl.minimum(y_idx_unclamped, H_in - 1))
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        # Use banker's rounding (round half to even) for align_corners=False
        x_floor = tl.floor(x)
        y_floor = tl.floor(y)
        x_frac = x - x_floor
        y_frac = y - y_floor
        x_is_half = x_frac == 0.5
        y_is_half = y_frac == 0.5
        x_floor_int = tl.cast(x_floor, tl.int32)
        y_floor_int = tl.cast(y_floor, tl.int32)
        x_is_even = x_floor_int % 2 == 0
        y_is_even = y_floor_int % 2 == 0
        x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
        y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
        x_idx_unclamped = tl.cast(
            tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
            tl.int32,
        )
        y_idx_unclamped = tl.cast(
            tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
            tl.int32,
        )
        # For align_corners=False: clamp to [0, W_in-1]
        x_idx = tl.maximum(0, tl.minimum(x_idx_unclamped, W_in - 1))
        y_idx = tl.maximum(0, tl.minimum(y_idx_unclamped, H_in - 1))

    # Load input pixel (always in bounds due to clamping)
    input_offset = n * C * H_in * W_in + c * H_in * W_in + y_idx * W_in + x_idx
    val = tl.load(ptr_input + input_offset).to(tl.float32)

    # Store output
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_nearest"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_nearest_reflection_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D nearest neighbor interpolation with reflection padding.

    Out-of-bound coordinates are reflected back into the valid range.
    """
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN
    grid_x = tl.where(grid_x != grid_x, -1.0, grid_x)
    grid_y = tl.where(grid_y != grid_y, -1.0, grid_y)

    # Reflection padding in GRID space (before denormalizing)
    # The grid space is [-1, 1], reflect at boundaries -1 and 1
    # Triangle wave pattern with period 4

    # Shift to [0, 4) range, handling negative modulo correctly
    grid_x_shifted = grid_x + 1.0
    # Triton's % operator behaves like C's fmod for floats (preserves sign)
    # So we need to adjust: for negative values, add period to make it positive
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)

    # Triangle wave: goes up from 0 to 2, then down from 2 to 0
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    grid_x_refl = grid_x_refl_mod - 1.0  # Shift back to [-1, 1]

    # Same for y
    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    grid_y_refl = grid_y_refl_mod - 1.0

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x_refl + 1.0) * (W_in - 1) / 2.0
        y = (grid_y_refl + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x_refl + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y_refl + 1.0) * H_in / 2.0 - 0.5

    # Banker's rounding (round half to even)
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    x_frac = x - x_floor
    y_frac = y - y_floor
    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)
    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0
    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    x_idx_unclamped = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx_unclamped = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )

    # Clamp to valid bounds (should already be in bounds due to reflection, but clamp for safety)
    x_idx = tl.maximum(0, tl.minimum(x_idx_unclamped, W_in - 1))
    y_idx = tl.maximum(0, tl.minimum(y_idx_unclamped, H_in - 1))

    # Load input pixel
    input_offset = n * C * H_in * W_in + c * H_in * W_in + y_idx * W_in + x_idx
    val = tl.load(ptr_input + input_offset).to(tl.float32)

    # Store output
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


# ============================================================================
# Bilinear Interpolation Kernels (4D)
# ============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bilinear"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bilinear_zeros_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D bilinear interpolation with zeros padding.

    Each program instance handles one output pixel location (all channels).
    Loads 4 corner pixels and performs bilinear interpolation.
    """
    # Each program instance processes one output pixel (all channels)
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates for this output location
    # Grid shape: (N, H_out, W_out, 2)
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN - use sentinel value -2.0 (outside valid grid range [-1, 1])
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        # Pixel centers at -1 and 1
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        # Pixel corners at -1 and 1
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Find 4 corner indices
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Compute interpolation weights
    wx = x - x0
    wy = y - y0

    # Convert corner indices to int
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y1_int = tl.cast(y1, tl.int32)

    # Check bounds for each corner (zeros padding)
    x0_in_bounds = (x0_int >= 0) & (x0_int < W_in)
    x1_in_bounds = (x1_int >= 0) & (x1_int < W_in)
    y0_in_bounds = (y0_int >= 0) & (y0_int < H_in)
    y1_in_bounds = (y1_int >= 0) & (y1_int < H_in)

    # Load 4 corner pixels with zeros padding
    # Input shape: (N, C, H_in, W_in)
    input_base = n * C * H_in * W_in + c * H_in * W_in

    offset_00 = input_base + y0_int * W_in + x0_int
    offset_01 = input_base + y0_int * W_in + x1_int
    offset_10 = input_base + y1_int * W_in + x0_int
    offset_11 = input_base + y1_int * W_in + x1_int

    p00 = tl.load(
        ptr_input + offset_00,
        mask=x0_in_bounds & y0_in_bounds & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    p01 = tl.load(
        ptr_input + offset_01,
        mask=x1_in_bounds & y0_in_bounds & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    p10 = tl.load(
        ptr_input + offset_10,
        mask=x0_in_bounds & y1_in_bounds & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    p11 = tl.load(
        ptr_input + offset_11,
        mask=x1_in_bounds & y1_in_bounds & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)

    # Bilinear interpolation
    # Interpolate along x, then y
    # top = p00 * (1-wx) + p01 * wx
    # bottom = p10 * (1-wx) + p11 * wx
    # result = top * (1-wy) + bottom * wy
    top = p00 * (1.0 - wx) + p01 * wx
    bottom = p10 * (1.0 - wx) + p11 * wx
    val = top * (1.0 - wy) + bottom * wy

    # Store output
    # Output shape: (N, C, H_out, W_out)
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bilinear"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bilinear_border_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D bilinear interpolation with border padding.

    Clamps coordinates to valid range [0, size-1] for out-of-bound values.
    """
    # Each program instance processes one output pixel (all channels)
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates for this output location
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Find 4 corner indices
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Convert to int
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y1_int = tl.cast(y1, tl.int32)

    # Clamp to valid bounds (border padding)
    x0_int = tl.maximum(0, tl.minimum(x0_int, W_in - 1))
    x1_int = tl.maximum(0, tl.minimum(x1_int, W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(y0_int, H_in - 1))
    y1_int = tl.maximum(0, tl.minimum(y1_int, H_in - 1))

    # Compute interpolation weights
    wx = x - x0
    wy = y - y0

    # Load 4 corner pixels (no mask needed due to clamping)
    input_base = n * C * H_in * W_in + c * H_in * W_in

    offset_00 = input_base + y0_int * W_in + x0_int
    offset_01 = input_base + y0_int * W_in + x1_int
    offset_10 = input_base + y1_int * W_in + x0_int
    offset_11 = input_base + y1_int * W_in + x1_int

    # For NaN, return 0.0
    p00 = tl.load(ptr_input + offset_00)
    p01 = tl.load(ptr_input + offset_01)
    p10 = tl.load(ptr_input + offset_10)
    p11 = tl.load(ptr_input + offset_11)

    # Bilinear interpolation
    top = p00 * (1.0 - wx) + p01 * wx
    bottom = p10 * (1.0 - wx) + p11 * wx
    val = tl.where(grid_x_nan | grid_y_nan, 0.0, top * (1.0 - wy) + bottom * wy)

    # Store output
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bilinear"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bilinear_reflection_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D bilinear interpolation with reflection padding.

    Reflects coordinates at boundaries using triangle wave pattern in grid space.
    """
    # Each program instance processes one output pixel (all channels)
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Reflection padding in GRID space (before denormalizing)
    # Triangle wave pattern with period 4
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    grid_x_refl = grid_x_refl_mod - 1.0

    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    grid_y_refl = grid_y_refl_mod - 1.0

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x_refl + 1.0) * (W_in - 1) / 2.0
        y = (grid_y_refl + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x_refl + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y_refl + 1.0) * H_in / 2.0 - 0.5

    # Find 4 corner indices
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Convert to int and clamp for safety
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y1_int = tl.cast(y1, tl.int32)

    x0_int = tl.maximum(0, tl.minimum(x0_int, W_in - 1))
    x1_int = tl.maximum(0, tl.minimum(x1_int, W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(y0_int, H_in - 1))
    y1_int = tl.maximum(0, tl.minimum(y1_int, H_in - 1))

    # Compute interpolation weights
    wx = x - x0
    wy = y - y0

    # Load 4 corner pixels
    input_base = n * C * H_in * W_in + c * H_in * W_in

    offset_00 = input_base + y0_int * W_in + x0_int
    offset_01 = input_base + y0_int * W_in + x1_int
    offset_10 = input_base + y1_int * W_in + x0_int
    offset_11 = input_base + y1_int * W_in + x1_int

    p00 = tl.load(ptr_input + offset_00)
    p01 = tl.load(ptr_input + offset_01)
    p10 = tl.load(ptr_input + offset_10)
    p11 = tl.load(ptr_input + offset_11)

    # Bilinear interpolation
    top = p00 * (1.0 - wx) + p01 * wx
    bottom = p10 * (1.0 - wx) + p11 * wx
    val = tl.where(grid_x_nan | grid_y_nan, 0.0, top * (1.0 - wy) + bottom * wy)

    # Store output
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


# ============================================================================
# Bicubic Interpolation Kernels (4D)
# ============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bicubic"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bicubic_zeros_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D bicubic interpolation with zeros padding.

    Uses Keys' cubic kernel with a=-0.5. Loads 4x4 neighborhood (16 pixels).
    """
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Find 4x4 neighborhood
    x0 = tl.floor(x) - 1
    y0 = tl.floor(y) - 1

    # Convert to int
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)

    # Compute interpolation weights using Keys' cubic kernel (a = -0.75)
    # W(x) = (a+2)|x|³ - (a+3)|x|² + 1, for |x| ≤ 1
    # W(x) = a|x|³ - 5a|x|² + 8a|x| - 4a, for 1 < |x| < 2
    # W(x) = 0, otherwise
    a = -0.75

    # X weights
    dx0 = x0 - x
    wx0 = tl.abs(dx0)
    weight_x0 = tl.where(
        wx0 < 1.0,
        ((a + 2) * wx0 - (a + 3)) * wx0 * wx0 + 1,
        tl.where(wx0 < 2.0, ((wx0 - 5) * wx0 + 8) * wx0 * a - 4 * a, 0.0),
    )

    dx1 = x0 + 1 - x
    wx1 = tl.abs(dx1)
    weight_x1 = tl.where(
        wx1 < 1.0,
        ((a + 2) * wx1 - (a + 3)) * wx1 * wx1 + 1,
        tl.where(wx1 < 2.0, ((wx1 - 5) * wx1 + 8) * wx1 * a - 4 * a, 0.0),
    )

    dx2 = x0 + 2 - x
    wx2 = tl.abs(dx2)
    weight_x2 = tl.where(
        wx2 < 1.0,
        ((a + 2) * wx2 - (a + 3)) * wx2 * wx2 + 1,
        tl.where(wx2 < 2.0, ((wx2 - 5) * wx2 + 8) * wx2 * a - 4 * a, 0.0),
    )

    dx3 = x0 + 3 - x
    wx3 = tl.abs(dx3)
    weight_x3 = tl.where(
        wx3 < 1.0,
        ((a + 2) * wx3 - (a + 3)) * wx3 * wx3 + 1,
        tl.where(wx3 < 2.0, ((wx3 - 5) * wx3 + 8) * wx3 * a - 4 * a, 0.0),
    )

    # Y weights
    dy0 = y0 - y
    wy0 = tl.abs(dy0)
    weight_y0 = tl.where(
        wy0 < 1.0,
        ((a + 2) * wy0 - (a + 3)) * wy0 * wy0 + 1,
        tl.where(wy0 < 2.0, ((wy0 - 5) * wy0 + 8) * wy0 * a - 4 * a, 0.0),
    )

    dy1 = y0 + 1 - y
    wy1 = tl.abs(dy1)
    weight_y1 = tl.where(
        wy1 < 1.0,
        ((a + 2) * wy1 - (a + 3)) * wy1 * wy1 + 1,
        tl.where(wy1 < 2.0, ((wy1 - 5) * wy1 + 8) * wy1 * a - 4 * a, 0.0),
    )

    dy2 = y0 + 2 - y
    wy2 = tl.abs(dy2)
    weight_y2 = tl.where(
        wy2 < 1.0,
        ((a + 2) * wy2 - (a + 3)) * wy2 * wy2 + 1,
        tl.where(wy2 < 2.0, ((wy2 - 5) * wy2 + 8) * wy2 * a - 4 * a, 0.0),
    )

    dy3 = y0 + 3 - y
    wy3 = tl.abs(dy3)
    weight_y3 = tl.where(
        wy3 < 1.0,
        ((a + 2) * wy3 - (a + 3)) * wy3 * wy3 + 1,
        tl.where(wy3 < 2.0, ((wy3 - 5) * wy3 + 8) * wy3 * a - 4 * a, 0.0),
    )

    # Load 4x4 neighborhood with zeros padding (unrolled loop)
    input_base = n * C * H_in * W_in + c * H_in * W_in

    # Initialize accumulator
    val = 0.0

    # Row 0
    y_idx0 = y0_int
    y_in_bounds0 = (y_idx0 >= 0) & (y_idx0 < H_in)

    x_idx00 = x0_int
    x_in_bounds00 = (x_idx00 >= 0) & (x_idx00 < W_in)
    offset00 = input_base + y_idx0 * W_in + x_idx00
    val00 = tl.load(
        ptr_input + offset00,
        mask=x_in_bounds00 & y_in_bounds0 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val00 * weight_x0 * weight_y0

    x_idx01 = x0_int + 1
    x_in_bounds01 = (x_idx01 >= 0) & (x_idx01 < W_in)
    offset01 = input_base + y_idx0 * W_in + x_idx01
    val01 = tl.load(
        ptr_input + offset01,
        mask=x_in_bounds01 & y_in_bounds0 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val01 * weight_x1 * weight_y0

    x_idx02 = x0_int + 2
    x_in_bounds02 = (x_idx02 >= 0) & (x_idx02 < W_in)
    offset02 = input_base + y_idx0 * W_in + x_idx02
    val02 = tl.load(
        ptr_input + offset02,
        mask=x_in_bounds02 & y_in_bounds0 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val02 * weight_x2 * weight_y0

    x_idx03 = x0_int + 3
    x_in_bounds03 = (x_idx03 >= 0) & (x_idx03 < W_in)
    offset03 = input_base + y_idx0 * W_in + x_idx03
    val03 = tl.load(
        ptr_input + offset03,
        mask=x_in_bounds03 & y_in_bounds0 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val03 * weight_x3 * weight_y0

    # Row 1
    y_idx1 = y0_int + 1
    y_in_bounds1 = (y_idx1 >= 0) & (y_idx1 < H_in)

    x_idx10 = x0_int
    x_in_bounds10 = (x_idx10 >= 0) & (x_idx10 < W_in)
    offset10 = input_base + y_idx1 * W_in + x_idx10
    val10 = tl.load(
        ptr_input + offset10,
        mask=x_in_bounds10 & y_in_bounds1 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val10 * weight_x0 * weight_y1

    x_idx11 = x0_int + 1
    x_in_bounds11 = (x_idx11 >= 0) & (x_idx11 < W_in)
    offset11 = input_base + y_idx1 * W_in + x_idx11
    val11 = tl.load(
        ptr_input + offset11,
        mask=x_in_bounds11 & y_in_bounds1 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val11 * weight_x1 * weight_y1

    x_idx12 = x0_int + 2
    x_in_bounds12 = (x_idx12 >= 0) & (x_idx12 < W_in)
    offset12 = input_base + y_idx1 * W_in + x_idx12
    val12 = tl.load(
        ptr_input + offset12,
        mask=x_in_bounds12 & y_in_bounds1 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val12 * weight_x2 * weight_y1

    x_idx13 = x0_int + 3
    x_in_bounds13 = (x_idx13 >= 0) & (x_idx13 < W_in)
    offset13 = input_base + y_idx1 * W_in + x_idx13
    val13 = tl.load(
        ptr_input + offset13,
        mask=x_in_bounds13 & y_in_bounds1 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val13 * weight_x3 * weight_y1

    # Row 2
    y_idx2 = y0_int + 2
    y_in_bounds2 = (y_idx2 >= 0) & (y_idx2 < H_in)

    x_idx20 = x0_int
    x_in_bounds20 = (x_idx20 >= 0) & (x_idx20 < W_in)
    offset20 = input_base + y_idx2 * W_in + x_idx20
    val20 = tl.load(
        ptr_input + offset20,
        mask=x_in_bounds20 & y_in_bounds2 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val20 * weight_x0 * weight_y2

    x_idx21 = x0_int + 1
    x_in_bounds21 = (x_idx21 >= 0) & (x_idx21 < W_in)
    offset21 = input_base + y_idx2 * W_in + x_idx21
    val21 = tl.load(
        ptr_input + offset21,
        mask=x_in_bounds21 & y_in_bounds2 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val21 * weight_x1 * weight_y2

    x_idx22 = x0_int + 2
    x_in_bounds22 = (x_idx22 >= 0) & (x_idx22 < W_in)
    offset22 = input_base + y_idx2 * W_in + x_idx22
    val22 = tl.load(
        ptr_input + offset22,
        mask=x_in_bounds22 & y_in_bounds2 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val22 * weight_x2 * weight_y2

    x_idx23 = x0_int + 3
    x_in_bounds23 = (x_idx23 >= 0) & (x_idx23 < W_in)
    offset23 = input_base + y_idx2 * W_in + x_idx23
    val23 = tl.load(
        ptr_input + offset23,
        mask=x_in_bounds23 & y_in_bounds2 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val23 * weight_x3 * weight_y2

    # Row 3
    y_idx3 = y0_int + 3
    y_in_bounds3 = (y_idx3 >= 0) & (y_idx3 < H_in)

    x_idx30 = x0_int
    x_in_bounds30 = (x_idx30 >= 0) & (x_idx30 < W_in)
    offset30 = input_base + y_idx3 * W_in + x_idx30
    val30 = tl.load(
        ptr_input + offset30,
        mask=x_in_bounds30 & y_in_bounds3 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val30 * weight_x0 * weight_y3

    x_idx31 = x0_int + 1
    x_in_bounds31 = (x_idx31 >= 0) & (x_idx31 < W_in)
    offset31 = input_base + y_idx3 * W_in + x_idx31
    val31 = tl.load(
        ptr_input + offset31,
        mask=x_in_bounds31 & y_in_bounds3 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val31 * weight_x1 * weight_y3

    x_idx32 = x0_int + 2
    x_in_bounds32 = (x_idx32 >= 0) & (x_idx32 < W_in)
    offset32 = input_base + y_idx3 * W_in + x_idx32
    val32 = tl.load(
        ptr_input + offset32,
        mask=x_in_bounds32 & y_in_bounds3 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val32 * weight_x2 * weight_y3

    x_idx33 = x0_int + 3
    x_in_bounds33 = (x_idx33 >= 0) & (x_idx33 < W_in)
    offset33 = input_base + y_idx3 * W_in + x_idx33
    val33 = tl.load(
        ptr_input + offset33,
        mask=x_in_bounds33 & y_in_bounds3 & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    ).to(tl.float32)
    val += val33 * weight_x3 * weight_y3

    # Store output
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bicubic"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bicubic_border_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D bicubic interpolation with border padding.
    """
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Find 4x4 neighborhood
    x0 = tl.floor(x) - 1
    y0 = tl.floor(y) - 1
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)

    # Compute Keys' cubic weights (a = -0.75)
    a = -0.75

    # X weights - compute inline for each pixel
    # W(x) = (a+2)|x|³ - (a+3)|x|² + 1, for |x| ≤ 1
    # W(x) = a|x|³ - 5a|x|² + 8a|x| - 4a, for 1 < |x| < 2

    # Load 4x4 neighborhood with border padding
    input_base = n * C * H_in * W_in + c * H_in * W_in
    val = 0.0

    # Unrolled loop for 4x4 neighborhood
    # Row 0
    y_idx = y0_int
    y_idx_clamped = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
    dy0 = y0 - y
    wy0 = tl.abs(dy0)
    weight_y0 = tl.where(
        wy0 < 1.0,
        ((a + 2) * wy0 - (a + 3)) * wy0 * wy0 + 1,
        tl.where(wy0 < 2.0, ((wy0 - 5) * wy0 + 8) * wy0 * a - 4 * a, 0.0),
    )

    # Col 0
    x_idx = x0_int
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    dx0 = x0 - x
    wx0 = tl.abs(dx0)
    weight_x0 = tl.where(
        wx0 < 1.0,
        ((a + 2) * wx0 - (a + 3)) * wx0 * wx0 + 1,
        tl.where(wx0 < 2.0, ((wx0 - 5) * wx0 + 8) * wx0 * a - 4 * a, 0.0),
    )
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x0 * weight_y0

    # Col 1
    x_idx = x0_int + 1
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    dx1 = x0 + 1 - x
    wx1 = tl.abs(dx1)
    weight_x1 = tl.where(
        wx1 < 1.0,
        ((a + 2) * wx1 - (a + 3)) * wx1 * wx1 + 1,
        tl.where(wx1 < 2.0, ((wx1 - 5) * wx1 + 8) * wx1 * a - 4 * a, 0.0),
    )
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x1 * weight_y0

    # Col 2
    x_idx = x0_int + 2
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    dx2 = x0 + 2 - x
    wx2 = tl.abs(dx2)
    weight_x2 = tl.where(
        wx2 < 1.0,
        ((a + 2) * wx2 - (a + 3)) * wx2 * wx2 + 1,
        tl.where(wx2 < 2.0, ((wx2 - 5) * wx2 + 8) * wx2 * a - 4 * a, 0.0),
    )
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x2 * weight_y0

    # Col 3
    x_idx = x0_int + 3
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    dx3 = x0 + 3 - x
    wx3 = tl.abs(dx3)
    weight_x3 = tl.where(
        wx3 < 1.0,
        ((a + 2) * wx3 - (a + 3)) * wx3 * wx3 + 1,
        tl.where(wx3 < 2.0, ((wx3 - 5) * wx3 + 8) * wx3 * a - 4 * a, 0.0),
    )
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x3 * weight_y0

    # Row 1
    y_idx = y0_int + 1
    y_idx_clamped = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
    dy1 = y0 + 1 - y
    wy1 = tl.abs(dy1)
    weight_y1 = tl.where(
        wy1 < 1.0,
        ((a + 2) * wy1 - (a + 3)) * wy1 * wy1 + 1,
        tl.where(wy1 < 2.0, ((wy1 - 5) * wy1 + 8) * wy1 * a - 4 * a, 0.0),
    )

    x_idx = x0_int
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x0 * weight_y1

    x_idx = x0_int + 1
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x1 * weight_y1

    x_idx = x0_int + 2
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x2 * weight_y1

    x_idx = x0_int + 3
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x3 * weight_y1

    # Row 2
    y_idx = y0_int + 2
    y_idx_clamped = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
    dy2 = y0 + 2 - y
    wy2 = tl.abs(dy2)
    weight_y2 = tl.where(
        wy2 < 1.0,
        ((a + 2) * wy2 - (a + 3)) * wy2 * wy2 + 1,
        tl.where(wy2 < 2.0, ((wy2 - 5) * wy2 + 8) * wy2 * a - 4 * a, 0.0),
    )

    x_idx = x0_int
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x0 * weight_y2

    x_idx = x0_int + 1
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x1 * weight_y2

    x_idx = x0_int + 2
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x2 * weight_y2

    x_idx = x0_int + 3
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x3 * weight_y2

    # Row 3
    y_idx = y0_int + 3
    y_idx_clamped = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
    dy3 = y0 + 3 - y
    wy3 = tl.abs(dy3)
    weight_y3 = tl.where(
        wy3 < 1.0,
        ((a + 2) * wy3 - (a + 3)) * wy3 * wy3 + 1,
        tl.where(wy3 < 2.0, ((wy3 - 5) * wy3 + 8) * wy3 * a - 4 * a, 0.0),
    )

    x_idx = x0_int
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x0 * weight_y3

    x_idx = x0_int + 1
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x1 * weight_y3

    x_idx = x0_int + 2
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x2 * weight_y3

    x_idx = x0_int + 3
    x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    offset = input_base + y_idx_clamped * W_in + x_idx_clamped
    val += tl.load(ptr_input + offset).to(tl.float32) * weight_x3 * weight_y3

    # Handle NaN
    val = tl.where(grid_x_nan | grid_y_nan, 0.0, val)

    # Store output
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bicubic"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bicubic_reflection_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 2D bicubic interpolation with reflection padding.
    """
    pid = tl.program_id(0)
    nc = pid // (H_out * W_out)
    hw = pid % (H_out * W_out)

    n = nc // C
    c = nc % C
    h_out = hw // W_out
    w_out = hw % W_out

    # Load grid coordinates
    grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Reflection padding in GRID space
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    grid_x_refl = grid_x_refl_mod - 1.0

    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    grid_y_refl = grid_y_refl_mod - 1.0

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x_refl + 1.0) * (W_in - 1) / 2.0
        y = (grid_y_refl + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x_refl + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y_refl + 1.0) * H_in / 2.0 - 0.5

    # Find 4x4 neighborhood
    x0 = tl.floor(x) - 1
    y0 = tl.floor(y) - 1
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)

    # Clamp for safety
    x0_int = tl.maximum(0, tl.minimum(x0_int, W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(y0_int, H_in - 1))

    # Compute Keys' cubic weights (a = -0.75)
    a = -0.75

    # Pre-compute X weights
    dx0 = x0 - x
    wx0 = tl.abs(dx0)
    weight_x0 = tl.where(
        wx0 < 1.0,
        ((a + 2) * wx0 - (a + 3)) * wx0 * wx0 + 1,
        tl.where(wx0 < 2.0, ((wx0 - 5) * wx0 + 8) * wx0 * a - 4 * a, 0.0),
    )

    dx1 = x0 + 1 - x
    wx1 = tl.abs(dx1)
    weight_x1 = tl.where(
        wx1 < 1.0,
        ((a + 2) * wx1 - (a + 3)) * wx1 * wx1 + 1,
        tl.where(wx1 < 2.0, ((wx1 - 5) * wx1 + 8) * wx1 * a - 4 * a, 0.0),
    )

    dx2 = x0 + 2 - x
    wx2 = tl.abs(dx2)
    weight_x2 = tl.where(
        wx2 < 1.0,
        ((a + 2) * wx2 - (a + 3)) * wx2 * wx2 + 1,
        tl.where(wx2 < 2.0, ((wx2 - 5) * wx2 + 8) * wx2 * a - 4 * a, 0.0),
    )

    dx3 = x0 + 3 - x
    wx3 = tl.abs(dx3)
    weight_x3 = tl.where(
        wx3 < 1.0,
        ((a + 2) * wx3 - (a + 3)) * wx3 * wx3 + 1,
        tl.where(wx3 < 2.0, ((wx3 - 5) * wx3 + 8) * wx3 * a - 4 * a, 0.0),
    )

    # Pre-compute Y weights
    dy0 = y0 - y
    wy0 = tl.abs(dy0)
    weight_y0 = tl.where(
        wy0 < 1.0,
        ((a + 2) * wy0 - (a + 3)) * wy0 * wy0 + 1,
        tl.where(wy0 < 2.0, ((wy0 - 5) * wy0 + 8) * wy0 * a - 4 * a, 0.0),
    )

    dy1 = y0 + 1 - y
    wy1 = tl.abs(dy1)
    weight_y1 = tl.where(
        wy1 < 1.0,
        ((a + 2) * wy1 - (a + 3)) * wy1 * wy1 + 1,
        tl.where(wy1 < 2.0, ((wy1 - 5) * wy1 + 8) * wy1 * a - 4 * a, 0.0),
    )

    dy2 = y0 + 2 - y
    wy2 = tl.abs(dy2)
    weight_y2 = tl.where(
        wy2 < 1.0,
        ((a + 2) * wy2 - (a + 3)) * wy2 * wy2 + 1,
        tl.where(wy2 < 2.0, ((wy2 - 5) * wy2 + 8) * wy2 * a - 4 * a, 0.0),
    )

    dy3 = y0 + 3 - y
    wy3 = tl.abs(dy3)
    weight_y3 = tl.where(
        wy3 < 1.0,
        ((a + 2) * wy3 - (a + 3)) * wy3 * wy3 + 1,
        tl.where(wy3 < 2.0, ((wy3 - 5) * wy3 + 8) * wy3 * a - 4 * a, 0.0),
    )

    # Load 4x4 neighborhood with clamping (reflection already applied)
    input_base = n * C * H_in * W_in + c * H_in * W_in
    val = 0.0

    # Unrolled loops for 4x4 neighborhood
    for i in range(4):
        y_idx = y0_int + i
        y_idx_clamped = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
        weight_y = tl.where(
            i == 0,
            weight_y0,
            tl.where(i == 1, weight_y1, tl.where(i == 2, weight_y2, weight_y3)),
        )

        for j in range(4):
            x_idx = x0_int + j
            x_idx_clamped = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
            weight_x = tl.where(
                j == 0,
                weight_x0,
                tl.where(j == 1, weight_x1, tl.where(j == 2, weight_x2, weight_x3)),
            )

            offset = input_base + y_idx_clamped * W_in + x_idx_clamped
            pixel_val = tl.load(ptr_input + offset).to(tl.float32)
            val += pixel_val * weight_x * weight_y

    # Handle NaN
    val = tl.where(grid_x_nan | grid_y_nan, 0.0, val)

    # Store output
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(ptr_output + output_offset, val)


# ============================================================================
# 5D Support Kernels (Volumetric Data)
# ============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_nearest_zeros_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 3D nearest neighbor interpolation with zeros padding.
    Handles 5D input (N, C, D_in, H_in, W_in) and 5D grid (N, D_out, H_out, W_out, 3).
    """
    pid = tl.program_id(0)
    ncd = pid // (D_out * H_out * W_out)
    dhw = pid % (D_out * H_out * W_out)

    n = ncd // C
    c = ncd % C
    d_out = dhw // (H_out * W_out)
    hw = dhw % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    # Load 3D grid coordinates
    grid_idx = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)
    grid_z = tl.load(ptr_grid + grid_idx + 2).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Banker's rounding for all three coordinates
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    z_floor = tl.floor(z)
    x_frac = x - x_floor
    y_frac = y - y_floor
    z_frac = z - z_floor
    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    z_is_half = z_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)
    z_floor_int = tl.cast(z_floor, tl.int32)
    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0
    z_is_even = z_floor_int % 2 == 0
    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    z_round = tl.where(z_frac < 0.5, z_floor, z_floor + 1)
    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )
    z_idx = tl.cast(
        tl.where(z_is_half, tl.where(z_is_even, z_floor, z_floor + 1), z_round),
        tl.int32,
    )

    # Check bounds for 3D
    mask = (
        (x_idx >= 0)
        & (x_idx < W_in)
        & (y_idx >= 0)
        & (y_idx < H_in)
        & (z_idx >= 0)
        & (z_idx < D_in)
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan
    )

    # Load input pixel (5D tensor: N, C, D, H, W)
    input_offset = (
        n * C * D_in * H_in * W_in
        + c * D_in * H_in * W_in
        + z_idx * H_in * W_in
        + y_idx * W_in
        + x_idx
    )
    val = tl.load(ptr_input + input_offset, mask=mask, other=0.0).to(tl.float32)

    # Store output (5D tensor: N, C, D, H, W)
    output_offset = (
        n * C * D_out * H_out * W_out
        + c * D_out * H_out * W_out
        + d_out * H_out * W_out
        + h_out * W_out
        + w_out
    )
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_nearest_border_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 3D nearest neighbor interpolation with border padding.
    """
    pid = tl.program_id(0)
    ncd = pid // (D_out * H_out * W_out)
    dhw = pid % (D_out * H_out * W_out)

    n = ncd // C
    c = ncd % C
    d_out = dhw // (H_out * W_out)
    hw = dhw % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    # Load 3D grid coordinates
    grid_idx = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)
    grid_z = tl.load(ptr_grid + grid_idx + 2).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Banker's rounding
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    z_floor = tl.floor(z)
    x_frac = x - x_floor
    y_frac = y - y_floor
    z_frac = z - z_floor
    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    z_is_half = z_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)
    z_floor_int = tl.cast(z_floor, tl.int32)
    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0
    z_is_even = z_floor_int % 2 == 0
    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    z_round = tl.where(z_frac < 0.5, z_floor, z_floor + 1)
    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )
    z_idx = tl.cast(
        tl.where(z_is_half, tl.where(z_is_even, z_floor, z_floor + 1), z_round),
        tl.int32,
    )

    # Clamp to valid bounds (border padding)
    x_idx = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    y_idx = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
    z_idx = tl.maximum(0, tl.minimum(z_idx, D_in - 1))

    # Load input pixel
    val = tl.where(
        grid_x_nan | grid_y_nan | grid_z_nan,
        0.0,
        tl.load(
            ptr_input
            + n * C * D_in * H_in * W_in
            + c * D_in * H_in * W_in
            + z_idx * H_in * W_in
            + y_idx * W_in
            + x_idx
        ).to(tl.float32),
    )

    # Store output
    output_offset = (
        n * C * D_out * H_out * W_out
        + c * D_out * H_out * W_out
        + d_out * H_out * W_out
        + h_out * W_out
        + w_out
    )
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_nearest_reflection_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 3D nearest neighbor interpolation with reflection padding.
    """
    pid = tl.program_id(0)
    ncd = pid // (D_out * H_out * W_out)
    dhw = pid % (D_out * H_out * W_out)

    n = ncd // C
    c = ncd % C
    d_out = dhw // (H_out * W_out)
    hw = dhw % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    # Load 3D grid coordinates
    grid_idx = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)
    grid_z = tl.load(ptr_grid + grid_idx + 2).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Reflection padding in GRID space (triangle wave with period 4)
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    grid_x_refl = grid_x_refl_mod - 1.0

    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    grid_y_refl = grid_y_refl_mod - 1.0

    grid_z_shifted = grid_z + 1.0
    grid_z_mod = grid_z_shifted % 4.0
    grid_z_mod = tl.where(grid_z_mod < 0, grid_z_mod + 4.0, grid_z_mod)
    grid_z_refl_mod = tl.where(grid_z_mod <= 2.0, grid_z_mod, 4.0 - grid_z_mod)
    grid_z_refl = grid_z_refl_mod - 1.0

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x_refl + 1.0) * (W_in - 1) / 2.0
        y = (grid_y_refl + 1.0) * (H_in - 1) / 2.0
        z = (grid_z_refl + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x_refl + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y_refl + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z_refl + 1.0) * D_in / 2.0 - 0.5

    # Banker's rounding
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    z_floor = tl.floor(z)
    x_frac = x - x_floor
    y_frac = y - y_floor
    z_frac = z - z_floor
    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    z_is_half = z_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)
    z_floor_int = tl.cast(z_floor, tl.int32)
    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0
    z_is_even = z_floor_int % 2 == 0
    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    z_round = tl.where(z_frac < 0.5, z_floor, z_floor + 1)
    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )
    z_idx = tl.cast(
        tl.where(z_is_half, tl.where(z_is_even, z_floor, z_floor + 1), z_round),
        tl.int32,
    )

    # Clamp for safety
    x_idx = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    y_idx = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
    z_idx = tl.maximum(0, tl.minimum(z_idx, D_in - 1))

    # Load input pixel
    val = tl.where(
        grid_x_nan | grid_y_nan | grid_z_nan,
        0.0,
        tl.load(
            ptr_input
            + n * C * D_in * H_in * W_in
            + c * D_in * H_in * W_in
            + z_idx * H_in * W_in
            + y_idx * W_in
            + x_idx
        ).to(tl.float32),
    )

    # Store output
    output_offset = (
        n * C * D_out * H_out * W_out
        + c * D_out * H_out * W_out
        + d_out * H_out * W_out
        + h_out * W_out
        + w_out
    )
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_trilinear_zeros_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 3D trilinear interpolation with zeros padding.
    Loads 8 corner pixels and performs trilinear interpolation.
    """
    pid = tl.program_id(0)
    ncd = pid // (D_out * H_out * W_out)
    dhw = pid % (D_out * H_out * W_out)

    n = ncd // C
    c = ncd % C
    d_out = dhw // (H_out * W_out)
    hw = dhw % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    # Load 3D grid coordinates
    grid_idx = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)
    grid_z = tl.load(ptr_grid + grid_idx + 2).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Find 8 corner indices (2x2x2)
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Compute interpolation weights
    wx = x - x0
    wy = y - y0
    wz = z - z0

    # Convert to int
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    z0_int = tl.cast(z0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y1_int = tl.cast(y1, tl.int32)
    z1_int = tl.cast(z1, tl.int32)

    # Check bounds for each corner (zeros padding)
    x0_in = (x0_int >= 0) & (x0_int < W_in)
    x1_in = (x1_int >= 0) & (x1_int < W_in)
    y0_in = (y0_int >= 0) & (y0_int < H_in)
    y1_in = (y1_int >= 0) & (y1_int < H_in)
    z0_in = (z0_int >= 0) & (z0_int < D_in)
    z1_in = (z1_int >= 0) & (z1_int < D_in)

    # Load 8 corner pixels with zeros padding
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in

    # z=y=x=0,0,0
    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x0_int
    p000 = tl.load(
        ptr_input + offset,
        mask=x0_in & y0_in & z0_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # z=y=0, x=1
    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x1_int
    p001 = tl.load(
        ptr_input + offset,
        mask=x1_in & y0_in & z0_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # z=0, y=1, x=0
    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x0_int
    p010 = tl.load(
        ptr_input + offset,
        mask=x0_in & y1_in & z0_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # z=0, y=1, x=1
    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x1_int
    p011 = tl.load(
        ptr_input + offset,
        mask=x1_in & y1_in & z0_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # z=1, y=x=0,0
    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x0_int
    p100 = tl.load(
        ptr_input + offset,
        mask=x0_in & y0_in & z1_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # z=1, y=0, x=1
    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x1_int
    p101 = tl.load(
        ptr_input + offset,
        mask=x1_in & y0_in & z1_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # z=1, y=1, x=0
    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x0_int
    p110 = tl.load(
        ptr_input + offset,
        mask=x0_in & y1_in & z1_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # z=1, y=1, x=1
    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x1_int
    p111 = tl.load(
        ptr_input + offset,
        mask=x1_in & y1_in & z1_in & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # Trilinear interpolation
    # Interpolate along x first, then y, then z
    # Front face (z=0)
    c000 = p000 * (1.0 - wx) + p001 * wx
    c001 = p010 * (1.0 - wx) + p011 * wx
    front = c000 * (1.0 - wy) + c001 * wy

    # Back face (z=1)
    c100 = p100 * (1.0 - wx) + p101 * wx
    c101 = p110 * (1.0 - wx) + p111 * wx
    back = c100 * (1.0 - wy) + c101 * wy

    # Interpolate along z
    val = front * (1.0 - wz) + back * wz

    # Store output
    output_offset = (
        n * C * D_out * H_out * W_out
        + c * D_out * H_out * W_out
        + d_out * H_out * W_out
        + h_out * W_out
        + w_out
    )
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_trilinear_border_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 3D trilinear interpolation with border padding.
    """
    pid = tl.program_id(0)
    ncd = pid // (D_out * H_out * W_out)
    dhw = pid % (D_out * H_out * W_out)

    n = ncd // C
    c = ncd % C
    d_out = dhw // (H_out * W_out)
    hw = dhw % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    # Load 3D grid coordinates
    grid_idx = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)
    grid_z = tl.load(ptr_grid + grid_idx + 2).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Find 8 corner indices
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Compute weights
    wx = x - x0
    wy = y - y0
    wz = z - z0

    # Convert to int and clamp
    x0_int = tl.maximum(0, tl.minimum(tl.cast(x0, tl.int32), W_in - 1))
    x1_int = tl.maximum(0, tl.minimum(tl.cast(x1, tl.int32), W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(tl.cast(y0, tl.int32), H_in - 1))
    y1_int = tl.maximum(0, tl.minimum(tl.cast(y1, tl.int32), H_in - 1))
    z0_int = tl.maximum(0, tl.minimum(tl.cast(z0, tl.int32), D_in - 1))
    z1_int = tl.maximum(0, tl.minimum(tl.cast(z1, tl.int32), D_in - 1))

    # Load 8 corner pixels (no mask needed due to clamping)
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in

    p000 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y0_int * W_in + x0_int
    ).to(tl.float32)
    p001 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y0_int * W_in + x1_int
    ).to(tl.float32)
    p010 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y1_int * W_in + x0_int
    ).to(tl.float32)
    p011 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y1_int * W_in + x1_int
    ).to(tl.float32)
    p100 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y0_int * W_in + x0_int
    ).to(tl.float32)
    p101 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y0_int * W_in + x1_int
    ).to(tl.float32)
    p110 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y1_int * W_in + x0_int
    ).to(tl.float32)
    p111 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y1_int * W_in + x1_int
    ).to(tl.float32)

    # Trilinear interpolation
    c000 = p000 * (1.0 - wx) + p001 * wx
    c001 = p010 * (1.0 - wx) + p011 * wx
    front = c000 * (1.0 - wy) + c001 * wy

    c100 = p100 * (1.0 - wx) + p101 * wx
    c101 = p110 * (1.0 - wx) + p111 * wx
    back = c100 * (1.0 - wy) + c101 * wy

    val = tl.where(
        grid_x_nan | grid_y_nan | grid_z_nan, 0.0, front * (1.0 - wz) + back * wz
    )

    # Store output
    output_offset = (
        n * C * D_out * H_out * W_out
        + c * D_out * H_out * W_out
        + d_out * H_out * W_out
        + h_out * W_out
        + w_out
    )
    tl.store(ptr_output + output_offset, val)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_trilinear_reflection_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid sample kernel for 3D trilinear interpolation with reflection padding.
    """
    pid = tl.program_id(0)
    ncd = pid // (D_out * H_out * W_out)
    dhw = pid % (D_out * H_out * W_out)

    n = ncd // C
    c = ncd % C
    d_out = dhw // (H_out * W_out)
    hw = dhw % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    # Load 3D grid coordinates
    grid_idx = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    grid_x = tl.load(ptr_grid + grid_idx).to(tl.float32)
    grid_y = tl.load(ptr_grid + grid_idx + 1).to(tl.float32)
    grid_z = tl.load(ptr_grid + grid_idx + 2).to(tl.float32)

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Reflection padding in GRID space (triangle wave)
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    grid_x_refl = grid_x_refl_mod - 1.0

    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    grid_y_refl = grid_y_refl_mod - 1.0

    grid_z_shifted = grid_z + 1.0
    grid_z_mod = grid_z_shifted % 4.0
    grid_z_mod = tl.where(grid_z_mod < 0, grid_z_mod + 4.0, grid_z_mod)
    grid_z_refl_mod = tl.where(grid_z_mod <= 2.0, grid_z_mod, 4.0 - grid_z_mod)
    grid_z_refl = grid_z_refl_mod - 1.0

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x_refl + 1.0) * (W_in - 1) / 2.0
        y = (grid_y_refl + 1.0) * (H_in - 1) / 2.0
        z = (grid_z_refl + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x_refl + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y_refl + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z_refl + 1.0) * D_in / 2.0 - 0.5

    # Find 8 corner indices
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Compute weights
    wx = x - x0
    wy = y - y0
    wz = z - z0

    # Convert to int and clamp
    x0_int = tl.maximum(0, tl.minimum(tl.cast(x0, tl.int32), W_in - 1))
    x1_int = tl.maximum(0, tl.minimum(tl.cast(x1, tl.int32), W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(tl.cast(y0, tl.int32), H_in - 1))
    y1_int = tl.maximum(0, tl.minimum(tl.cast(y1, tl.int32), H_in - 1))
    z0_int = tl.maximum(0, tl.minimum(tl.cast(z0, tl.int32), D_in - 1))
    z1_int = tl.maximum(0, tl.minimum(tl.cast(z1, tl.int32), D_in - 1))

    # Load 8 corner pixels
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in

    p000 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y0_int * W_in + x0_int
    ).to(tl.float32)
    p001 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y0_int * W_in + x1_int
    ).to(tl.float32)
    p010 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y1_int * W_in + x0_int
    ).to(tl.float32)
    p011 = tl.load(
        ptr_input + input_base + z0_int * H_in * W_in + y1_int * W_in + x1_int
    ).to(tl.float32)
    p100 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y0_int * W_in + x0_int
    ).to(tl.float32)
    p101 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y0_int * W_in + x1_int
    ).to(tl.float32)
    p110 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y1_int * W_in + x0_int
    ).to(tl.float32)
    p111 = tl.load(
        ptr_input + input_base + z1_int * H_in * W_in + y1_int * W_in + x1_int
    ).to(tl.float32)

    # Trilinear interpolation
    c000 = p000 * (1.0 - wx) + p001 * wx
    c001 = p010 * (1.0 - wx) + p011 * wx
    front = c000 * (1.0 - wy) + c001 * wy

    c100 = p100 * (1.0 - wx) + p101 * wx
    c101 = p110 * (1.0 - wx) + p111 * wx
    back = c100 * (1.0 - wy) + c101 * wy

    val = tl.where(
        grid_x_nan | grid_y_nan | grid_z_nan, 0.0, front * (1.0 - wz) + back * wz
    )

    # Store output
    output_offset = (
        n * C * D_out * H_out * W_out
        + c * D_out * H_out * W_out
        + d_out * H_out * W_out
        + h_out * W_out
        + w_out
    )
    tl.store(ptr_output + output_offset, val)


# ============================================================================
# 3D Tiled Kernels for Medium-to-Large 5D Inputs (3D Blocking: D×H×W)
# ============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest_tiled"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_nearest_zeros_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 3D nearest neighbor interpolation with zeros padding (tiled version).

    This kernel processes a BLOCK_D × BLOCK_H × BLOCK_W tile of output voxels at once,
    enabling better memory coalescing and data reuse for medium-to-large 5D inputs.

    Args:
        ptr_output: Pointer to output tensor (N, C, D_out, H_out, W_out)
        ptr_input: Pointer to input tensor (N, C, D_in, H_in, W_in)
        ptr_grid: Pointer to grid tensor (N, D_out, H_out, W_out, 3)
        N: Batch size
        C: Number of channels
        D_in: Input depth
        H_in: Input height
        W_in: Input width
        D_out: Output depth
        H_out: Output height
        W_out: Output width
        align_corners: Whether to align corners
        BLOCK_D: Block depth for tiling
        BLOCK_H: Block height for tiling
        BLOCK_W: Block width for tiling
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_dhw for spatial tile
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Decompose flattened 3D tile index to (d, h, w) block indices
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    # Compute voxel offsets within tile (3D broadcasting)
    d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    # Reshape for 3D broadcasting: (BLOCK_D, BLOCK_H, BLOCK_W)
    d_out_3d = d_offsets[:, None, None]
    h_out_3d = h_offsets[None, :, None]
    w_out_3d = w_offsets[None, None, :]

    # Load 3D grid coordinates for entire tile (vectorized)
    # Grid shape: (N, D_out, H_out, W_out, 3)
    grid_base = n * D_out * H_out * W_out * 3

    # Load x, y, z coordinates: (BLOCK_D, BLOCK_H, BLOCK_W)
    grid_x_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3
    )
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_y_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 1
    )
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_z_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 2
    )
    grid_z = tl.load(ptr_grid + grid_z_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN - use sentinel value -2.0 (outside valid grid range [-1, 1])
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        # Pixel centers at -1 and 1
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        # Pixel corners at -1 and 1
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Apply banker's rounding (vectorized across tile)
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    z_floor = tl.floor(z)
    x_frac = x - x_floor
    y_frac = y - y_floor
    z_frac = z - z_floor

    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    z_is_half = z_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)
    z_floor_int = tl.cast(z_floor, tl.int32)

    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0
    z_is_even = z_floor_int % 2 == 0

    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    z_round = tl.where(z_frac < 0.5, z_floor, z_floor + 1)

    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )
    z_idx = tl.cast(
        tl.where(z_is_half, tl.where(z_is_even, z_floor, z_floor + 1), z_round),
        tl.int32,
    )

    # Check bounds (vectorized)
    x_in_bounds = (x_idx >= 0) & (x_idx < W_in)
    y_in_bounds = (y_idx >= 0) & (y_idx < H_in)
    z_in_bounds = (z_idx >= 0) & (z_idx < D_in)
    valid_mask = (
        tile_mask
        & x_in_bounds
        & y_in_bounds
        & z_in_bounds
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan
    )

    # Load input voxels for entire tile
    # Input shape: (N, C, D_in, H_in, W_in)
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in
    input_offsets = input_base + z_idx * H_in * W_in + y_idx * W_in + x_idx

    # Vectorized load: (BLOCK_D, BLOCK_H, BLOCK_W)
    vals = tl.load(ptr_input + input_offsets, mask=valid_mask, other=0.0)

    # Store to output
    # Output shape: (N, C, D_out, H_out, W_out)
    output_base = n * C * D_out * H_out * W_out + c * D_out * H_out * W_out
    output_offsets = output_base + (
        d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d
    )

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest_tiled"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_nearest_border_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 3D nearest neighbor interpolation with border padding (tiled version).

    Border padding: coordinates outside the input range are clamped to the boundary.
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_dhw for spatial tile
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Decompose flattened 3D tile index to (d, h, w) block indices
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    # Compute voxel offsets within tile (3D broadcasting)
    d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    # Reshape for 3D broadcasting: (BLOCK_D, BLOCK_H, BLOCK_W)
    d_out_3d = d_offsets[:, None, None]
    h_out_3d = h_offsets[None, :, None]
    w_out_3d = w_offsets[None, None, :]

    # Load 3D grid coordinates for entire tile (vectorized)
    grid_base = n * D_out * H_out * W_out * 3

    grid_x_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3
    )
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_y_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 1
    )
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_z_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 2
    )
    grid_z = tl.load(ptr_grid + grid_z_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Apply banker's rounding (vectorized across tile)
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    z_floor = tl.floor(z)
    x_frac = x - x_floor
    y_frac = y - y_floor
    z_frac = z - z_floor

    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    z_is_half = z_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)
    z_floor_int = tl.cast(z_floor, tl.int32)

    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0
    z_is_even = z_floor_int % 2 == 0

    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    z_round = tl.where(z_frac < 0.5, z_floor, z_floor + 1)

    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )
    z_idx = tl.cast(
        tl.where(z_is_half, tl.where(z_is_even, z_floor, z_floor + 1), z_round),
        tl.int32,
    )

    # Border padding: clamp coordinates to valid range
    x_idx = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
    y_idx = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
    z_idx = tl.maximum(0, tl.minimum(z_idx, D_in - 1))

    # Valid mask: only tile boundary and NaN (no bounds check needed for border)
    valid_mask = tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan

    # Load input voxels for entire tile
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in
    input_offsets = input_base + z_idx * H_in * W_in + y_idx * W_in + x_idx

    # Load and handle NaN separately (border padding doesn't help with NaN)
    vals_raw = tl.load(ptr_input + input_offsets, mask=valid_mask, other=0.0)
    vals = tl.where(grid_x_nan | grid_y_nan | grid_z_nan, 0.0, vals_raw)

    # Store to output
    output_base = n * C * D_out * H_out * W_out + c * D_out * H_out * W_out
    output_offsets = output_base + (
        d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d
    )

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest_tiled"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_nearest_reflection_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 3D nearest neighbor interpolation with reflection padding (tiled version).

    Reflection padding: coordinates outside the input range are reflected back into the valid range
    using a triangle wave pattern with period 4.
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_dhw for spatial tile
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Decompose flattened 3D tile index to (d, h, w) block indices
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    # Compute voxel offsets within tile (3D broadcasting)
    d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    # Reshape for 3D broadcasting: (BLOCK_D, BLOCK_H, BLOCK_W)
    d_out_3d = d_offsets[:, None, None]
    h_out_3d = h_offsets[None, :, None]
    w_out_3d = w_offsets[None, None, :]

    # Load 3D grid coordinates for entire tile (vectorized)
    grid_base = n * D_out * H_out * W_out * 3

    grid_x_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3
    )
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_y_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 1
    )
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_z_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 2
    )
    grid_z = tl.load(ptr_grid + grid_z_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Apply triangle wave reflection with period 4 (before denormalization)
    # This maps coordinates outside [-1, 1] back into this range by reflection
    # Process grid_x
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    grid_x = grid_x_refl_mod - 1.0

    # Process grid_y
    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    grid_y = grid_y_refl_mod - 1.0

    # Process grid_z
    grid_z_shifted = grid_z + 1.0
    grid_z_mod = grid_z_shifted % 4.0
    grid_z_mod = tl.where(grid_z_mod < 0, grid_z_mod + 4.0, grid_z_mod)
    grid_z_refl_mod = tl.where(grid_z_mod <= 2.0, grid_z_mod, 4.0 - grid_z_mod)
    grid_z = grid_z_refl_mod - 1.0

    # Denormalize reflected coordinates to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Apply banker's rounding (vectorized across tile)
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    z_floor = tl.floor(z)
    x_frac = x - x_floor
    y_frac = y - y_floor
    z_frac = z - z_floor

    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    z_is_half = z_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)
    z_floor_int = tl.cast(z_floor, tl.int32)

    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0
    z_is_even = z_floor_int % 2 == 0

    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    z_round = tl.where(z_frac < 0.5, z_floor, z_floor + 1)

    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )
    z_idx = tl.cast(
        tl.where(z_is_half, tl.where(z_is_even, z_floor, z_floor + 1), z_round),
        tl.int32,
    )

    # Check bounds (reflection ensures coordinates are valid, but still check)
    x_in_bounds = (x_idx >= 0) & (x_idx < W_in)
    y_in_bounds = (y_idx >= 0) & (y_idx < H_in)
    z_in_bounds = (z_idx >= 0) & (z_idx < D_in)
    valid_mask = (
        tile_mask
        & x_in_bounds
        & y_in_bounds
        & z_in_bounds
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan
    )

    # Load input voxels for entire tile
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in
    input_offsets = input_base + z_idx * H_in * W_in + y_idx * W_in + x_idx

    vals = tl.load(ptr_input + input_offsets, mask=valid_mask, other=0.0)

    # Store to output
    output_base = n * C * D_out * H_out * W_out + c * D_out * H_out * W_out
    output_offsets = output_base + (
        d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d
    )

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear_tiled"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_trilinear_zeros_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 3D trilinear interpolation with zeros padding (tiled version).

    Trilinear interpolation uses 8 corner points (2×2×2 cube) for each output voxel.
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_dhw for spatial tile
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Decompose flattened 3D tile index to (d, h, w) block indices
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    # Compute voxel offsets within tile (3D broadcasting)
    d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    # Reshape for 3D broadcasting: (BLOCK_D, BLOCK_H, BLOCK_W)
    d_out_3d = d_offsets[:, None, None]
    h_out_3d = h_offsets[None, :, None]
    w_out_3d = w_offsets[None, None, :]

    # Load 3D grid coordinates for entire tile (vectorized)
    grid_base = n * D_out * H_out * W_out * 3

    grid_x_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3
    )
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_y_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 1
    )
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_z_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 2
    )
    grid_z = tl.load(ptr_grid + grid_z_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Compute 8 corner indices for entire tile
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Interpolation weights
    wx = x - x0
    wy = y - y0
    wz = z - z0

    # Convert to integers
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    z0_int = tl.cast(z0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y1_int = tl.cast(y1, tl.int32)
    z1_int = tl.cast(z1, tl.int32)

    # Boundary checks
    x0_in = (x0_int >= 0) & (x0_int < W_in)
    x1_in = (x1_int >= 0) & (x1_int < W_in)
    y0_in = (y0_int >= 0) & (y0_int < H_in)
    y1_in = (y1_int >= 0) & (y1_int < H_in)
    z0_in = (z0_int >= 0) & (z0_int < D_in)
    z1_in = (z1_int >= 0) & (z1_int < D_in)

    # Load 8 corners (vectorized)
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in

    # p000: (x=0, y=0, z=0)
    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x0_int
    p000 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y0_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # p001: (x=1, y=0, z=0)
    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x1_int
    p001 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y0_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # p010: (x=0, y=1, z=0)
    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x0_int
    p010 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y1_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # p011: (x=1, y=1, z=0)
    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x1_int
    p011 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y1_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # p100: (x=0, y=0, z=1)
    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x0_int
    p100 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y0_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # p101: (x=1, y=0, z=1)
    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x1_int
    p101 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y0_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # p110: (x=0, y=1, z=1)
    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x0_int
    p110 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y1_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # p111: (x=1, y=1, z=1)
    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x1_int
    p111 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y1_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # 3-stage trilinear interpolation
    # Stage 1: Interpolate along x
    c000 = p000 * (1.0 - wx) + p001 * wx  # z=0, y=0
    c001 = p010 * (1.0 - wx) + p011 * wx  # z=0, y=1
    c010 = p100 * (1.0 - wx) + p101 * wx  # z=1, y=0
    c011 = p110 * (1.0 - wx) + p111 * wx  # z=1, y=1

    # Stage 2: Interpolate along y
    c00 = c000 * (1.0 - wy) + c001 * wy  # z=0
    c01 = c010 * (1.0 - wy) + c011 * wy  # z=1

    # Stage 3: Interpolate along z (final)
    vals = c00 * (1.0 - wz) + c01 * wz

    # Store to output
    output_base = n * C * D_out * H_out * W_out + c * D_out * H_out * W_out
    output_offsets = output_base + (
        d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d
    )

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear_tiled"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_trilinear_border_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 3D trilinear interpolation with border padding (tiled version).
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_dhw for spatial tile
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Decompose flattened 3D tile index to (d, h, w) block indices
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    # Compute voxel offsets within tile (3D broadcasting)
    d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    # Reshape for 3D broadcasting: (BLOCK_D, BLOCK_H, BLOCK_W)
    d_out_3d = d_offsets[:, None, None]
    h_out_3d = h_offsets[None, :, None]
    w_out_3d = w_offsets[None, None, :]

    # Load 3D grid coordinates for entire tile (vectorized)
    grid_base = n * D_out * H_out * W_out * 3

    grid_x_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3
    )
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_y_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 1
    )
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_z_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 2
    )
    grid_z = tl.load(ptr_grid + grid_z_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Compute 8 corner indices for entire tile
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Interpolation weights
    wx = x - x0
    wy = y - y0
    wz = z - z0

    # Convert to integers and clamp for border padding
    x0_int = tl.maximum(0, tl.minimum(tl.cast(x0, tl.int32), W_in - 1))
    x1_int = tl.maximum(0, tl.minimum(tl.cast(x1, tl.int32), W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(tl.cast(y0, tl.int32), H_in - 1))
    y1_int = tl.maximum(0, tl.minimum(tl.cast(y1, tl.int32), H_in - 1))
    z0_int = tl.maximum(0, tl.minimum(tl.cast(z0, tl.int32), D_in - 1))
    z1_int = tl.maximum(0, tl.minimum(tl.cast(z1, tl.int32), D_in - 1))

    # Load 8 corners (vectorized, no bounds mask needed for border)
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in

    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x0_int
    p000 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x1_int
    p001 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x0_int
    p010 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x1_int
    p011 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x0_int
    p100 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x1_int
    p101 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x0_int
    p110 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x1_int
    p111 = tl.load(
        ptr_input + offset,
        mask=tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # 3-stage trilinear interpolation
    c000 = p000 * (1.0 - wx) + p001 * wx
    c001 = p010 * (1.0 - wx) + p011 * wx
    c010 = p100 * (1.0 - wx) + p101 * wx
    c011 = p110 * (1.0 - wx) + p111 * wx

    c00 = c000 * (1.0 - wy) + c001 * wy
    c01 = c010 * (1.0 - wy) + c011 * wy

    vals = c00 * (1.0 - wz) + c01 * wz

    # Handle NaN
    vals = tl.where(grid_x_nan | grid_y_nan | grid_z_nan, 0.0, vals)

    # Store to output
    output_base = n * C * D_out * H_out * W_out + c * D_out * H_out * W_out
    output_offsets = output_base + (
        d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d
    )

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear_tiled"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_trilinear_reflection_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 3D trilinear interpolation with reflection padding (tiled version).
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_dhw for spatial tile
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Decompose flattened 3D tile index to (d, h, w) block indices
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    # Compute voxel offsets within tile (3D broadcasting)
    d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    # Reshape for 3D broadcasting: (BLOCK_D, BLOCK_H, BLOCK_W)
    d_out_3d = d_offsets[:, None, None]
    h_out_3d = h_offsets[None, :, None]
    w_out_3d = w_offsets[None, None, :]

    # Load 3D grid coordinates for entire tile (vectorized)
    grid_base = n * D_out * H_out * W_out * 3

    grid_x_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3
    )
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_y_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 1
    )
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    grid_z_offsets = (
        grid_base + (d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d) * 3 + 2
    )
    grid_z = tl.load(ptr_grid + grid_z_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    # Apply triangle wave reflection with period 4
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    grid_x = grid_x_refl_mod - 1.0

    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    grid_y = grid_y_refl_mod - 1.0

    grid_z_shifted = grid_z + 1.0
    grid_z_mod = grid_z_shifted % 4.0
    grid_z_mod = tl.where(grid_z_mod < 0, grid_z_mod + 4.0, grid_z_mod)
    grid_z_refl_mod = tl.where(grid_z_mod <= 2.0, grid_z_mod, 4.0 - grid_z_mod)
    grid_z = grid_z_refl_mod - 1.0

    # Denormalize reflected coordinates to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Compute 8 corner indices for entire tile
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Interpolation weights
    wx = x - x0
    wy = y - y0
    wz = z - z0

    # Convert to integers
    x0_int = tl.cast(x0, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    z0_int = tl.cast(z0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y1_int = tl.cast(y1, tl.int32)
    z1_int = tl.cast(z1, tl.int32)

    # Boundary checks (reflection ensures coordinates are mostly valid, but still check)
    x0_in = (x0_int >= 0) & (x0_int < W_in)
    x1_in = (x1_int >= 0) & (x1_int < W_in)
    y0_in = (y0_int >= 0) & (y0_int < H_in)
    y1_in = (y1_int >= 0) & (y1_int < H_in)
    z0_in = (z0_int >= 0) & (z0_int < D_in)
    z1_in = (z1_int >= 0) & (z1_int < D_in)

    # Load 8 corners (vectorized)
    input_base = n * C * D_in * H_in * W_in + c * D_in * H_in * W_in

    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x0_int
    p000 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y0_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z0_int * H_in * W_in + y0_int * W_in + x1_int
    p001 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y0_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x0_int
    p010 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y1_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z0_int * H_in * W_in + y1_int * W_in + x1_int
    p011 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y1_in
        & z0_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x0_int
    p100 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y0_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y0_int * W_in + x1_int
    p101 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y0_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x0_int
    p110 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x0_in
        & y1_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    offset = input_base + z1_int * H_in * W_in + y1_int * W_in + x1_int
    p111 = tl.load(
        ptr_input + offset,
        mask=tile_mask
        & x1_in
        & y1_in
        & z1_in
        & ~grid_x_nan
        & ~grid_y_nan
        & ~grid_z_nan,
        other=0.0,
    ).to(tl.float32)

    # 3-stage trilinear interpolation
    c000 = p000 * (1.0 - wx) + p001 * wx
    c001 = p010 * (1.0 - wx) + p011 * wx
    c010 = p100 * (1.0 - wx) + p101 * wx
    c011 = p110 * (1.0 - wx) + p111 * wx

    c00 = c000 * (1.0 - wy) + c001 * wy
    c01 = c010 * (1.0 - wy) + c011 * wy

    vals = c00 * (1.0 - wz) + c01 * wz

    # Store to output
    output_base = n * C * D_out * H_out * W_out + c * D_out * H_out * W_out
    output_offsets = output_base + (
        d_out_3d * H_out * W_out + h_out_3d * W_out + w_out_3d
    )

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


# ============================================================================
# Tiled Kernels for Medium-to-Large Inputs (Multi-dimensional Blocking)
# ============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_nearest_tiled"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_nearest_zeros_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 2D nearest neighbor interpolation with zeros padding (tiled version).

    This kernel processes a BLOCK_H × BLOCK_W tile of output pixels at once,
    enabling better memory coalescing and data reuse for medium-to-large inputs.

    Args:
        ptr_output: Pointer to output tensor
        ptr_input: Pointer to input tensor
        ptr_grid: Pointer to grid tensor
        N: Batch size
        C: Number of channels
        H_in: Input height
        W_in: Input width
        H_out: Output height
        W_out: Output width
        align_corners: Whether to align corners
        BLOCK_H: Block height for tiling
        BLOCK_W: Block width for tiling
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_hw for spatial tile
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Compute tile position in output grid
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    # Compute pixel offsets within tile
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = h_mask[:, None] & w_mask[None, :]

    # Reshape for broadcasting: (BLOCK_H, BLOCK_W)
    h_out_flat = h_offsets[:, None]
    w_out_flat = w_offsets[None, :]

    # Load grid coordinates for entire tile (vectorized)
    # Grid shape: (N, H_out, W_out, 2)
    grid_base = n * H_out * W_out * 2

    # Load x coordinates: (BLOCK_H, BLOCK_W)
    grid_x_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Load y coordinates: (BLOCK_H, BLOCK_W)
    grid_y_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2 + 1
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN - use sentinel value -2.0 (outside valid grid range [-1, 1])
    grid_x_nan = grid_x != grid_x  # True if NaN
    grid_y_nan = grid_y != grid_y  # True if NaN
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        # Pixel centers at -1 and 1
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        # Pixel corners at -1 and 1
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Apply banker's rounding (vectorized across tile)
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    x_frac = x - x_floor
    y_frac = y - y_floor

    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)

    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0

    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)

    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )

    # Check bounds (vectorized)
    x_in_bounds = (x_idx >= 0) & (x_idx < W_in)
    y_in_bounds = (y_idx >= 0) & (y_idx < H_in)
    valid_mask = tile_mask & x_in_bounds & y_in_bounds & ~grid_x_nan & ~grid_y_nan

    # Load input pixels for entire tile
    # Input shape: (N, C, H_in, W_in)
    input_base = n * C * H_in * W_in + c * H_in * W_in
    input_offsets = input_base + y_idx * W_in + x_idx

    # Vectorized load: (BLOCK_H, BLOCK_W)
    vals = tl.load(ptr_input + input_offsets, mask=valid_mask, other=0.0)

    # Store to output
    # Output shape: (N, C, H_out, W_out)
    output_base = n * C * H_out * W_out + c * H_out * W_out
    output_offsets = output_base + (h_out_flat * W_out + w_out_flat)

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bilinear_tiled"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bilinear_zeros_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 2D bilinear interpolation with zeros padding (tiled version).

    This kernel processes a BLOCK_H × BLOCK_W tile of output pixels at once,
    enabling better memory coalescing and data reuse for medium-to-large inputs.

    Args:
        ptr_output: Pointer to output tensor
        ptr_input: Pointer to input tensor
        ptr_grid: Pointer to grid tensor
        N: Batch size
        C: Number of channels
        H_in: Input height
        W_in: Input width
        H_out: Output height
        W_out: Output width
        align_corners: Whether to align corners
        BLOCK_H: Block height for tiling
        BLOCK_W: Block width for tiling
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_hw for spatial tile
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Compute tile position in output grid
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    # Compute pixel offsets within tile
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = h_mask[:, None] & w_mask[None, :]

    # Reshape for broadcasting: (BLOCK_H, BLOCK_W)
    h_out_flat = h_offsets[:, None]
    w_out_flat = w_offsets[None, :]

    # Load grid coordinates for entire tile (vectorized)
    # Grid shape: (N, H_out, W_out, 2)
    grid_base = n * H_out * W_out * 2

    # Load x coordinates: (BLOCK_H, BLOCK_W)
    grid_x_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Load y coordinates: (BLOCK_H, BLOCK_W)
    grid_y_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2 + 1
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN - use sentinel value -2.0
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        # Pixel centers at -1 and 1
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        # Pixel corners at -1 and 1
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Compute corner indices for entire tile (vectorized)
    x0 = tl.floor(x)
    x1 = x0 + 1
    y0 = tl.floor(y)
    y1 = y0 + 1

    # Cast to int for indexing
    x0_int = tl.cast(x0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    y1_int = tl.cast(y1, tl.int32)

    # Check bounds for all 4 corners
    x0_in = (x0_int >= 0) & (x0_int < W_in)
    x1_in = (x1_int >= 0) & (x1_int < W_in)
    y0_in = (y0_int >= 0) & (y0_int < H_in)
    y1_in = (y1_int >= 0) & (y1_int < H_in)

    # Compute interpolation weights
    wx = x - tl.cast(x0, tl.float32)
    wy = y - tl.cast(y0, tl.float32)

    # Load 4 corner pixels (vectorized)
    # Input shape: (N, C, H_in, W_in)
    input_base = n * C * H_in * W_in + c * H_in * W_in

    p00_offsets = input_base + y0_int * W_in + x0_int
    p00 = tl.load(
        ptr_input + p00_offsets,
        mask=tile_mask & x0_in & y0_in & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    )

    p01_offsets = input_base + y0_int * W_in + x1_int
    p01 = tl.load(
        ptr_input + p01_offsets,
        mask=tile_mask & x1_in & y0_in & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    )

    p10_offsets = input_base + y1_int * W_in + x0_int
    p10 = tl.load(
        ptr_input + p10_offsets,
        mask=tile_mask & x0_in & y1_in & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    )

    p11_offsets = input_base + y1_int * W_in + x1_int
    p11 = tl.load(
        ptr_input + p11_offsets,
        mask=tile_mask & x1_in & y1_in & ~grid_x_nan & ~grid_y_nan,
        other=0.0,
    )

    # Bilinear interpolation (vectorized)
    # Interpolate along x, then y
    top = p00 * (1.0 - wx) + p01 * wx
    bottom = p10 * (1.0 - wx) + p11 * wx
    vals = top * (1.0 - wy) + bottom * wy

    # Store to output
    # Output shape: (N, C, H_out, W_out)
    output_base = n * C * H_out * W_out + c * H_out * W_out
    output_offsets = output_base + (h_out_flat * W_out + w_out_flat)

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_nearest_tiled"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_nearest_border_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 2D nearest neighbor interpolation with border padding (tiled version).

    Border padding: coordinates are clamped to valid range [0, W_in) x [0, H_in).
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_hw for spatial tile
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Compute tile position in output grid
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    # Compute pixel offsets within tile
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = h_mask[:, None] & w_mask[None, :]

    # Reshape for broadcasting: (BLOCK_H, BLOCK_W)
    h_out_flat = h_offsets[:, None]
    w_out_flat = w_offsets[None, :]

    # Load grid coordinates for entire tile (vectorized)
    grid_base = n * H_out * W_out * 2

    # Load x coordinates: (BLOCK_H, BLOCK_W)
    grid_x_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Load y coordinates: (BLOCK_H, BLOCK_W)
    grid_y_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2 + 1
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN - use sentinel -1.0 like original kernel
    grid_x = tl.where(grid_x != grid_x, -1.0, grid_x)
    grid_y = tl.where(grid_y != grid_y, -1.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Apply banker's rounding (vectorized across tile)
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    x_frac = x - x_floor
    y_frac = y - y_floor

    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)

    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0

    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)

    x_idx_unclamped = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx_unclamped = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )

    # Clamp to valid range (border padding)
    x_idx = tl.maximum(0, tl.minimum(x_idx_unclamped, W_in - 1))
    y_idx = tl.maximum(0, tl.minimum(y_idx_unclamped, H_in - 1))

    # Load input pixels for entire tile (no mask needed - clamping ensures validity)
    input_base = n * C * H_in * W_in + c * H_in * W_in
    input_offsets = input_base + y_idx * W_in + x_idx

    vals = tl.load(ptr_input + input_offsets, mask=tile_mask, other=0.0)

    # Store to output
    output_base = n * C * H_out * W_out + c * H_out * W_out
    output_offsets = output_base + (h_out_flat * W_out + w_out_flat)

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bilinear_tiled"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bilinear_border_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 2D bilinear interpolation with border padding (tiled version).

    Border padding: coordinates are clamped to valid range [0, W_in) x [0, H_in).
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_hw for spatial tile
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Compute tile position in output grid
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    # Compute pixel offsets within tile
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = h_mask[:, None] & w_mask[None, :]

    # Reshape for broadcasting: (BLOCK_H, BLOCK_W)
    h_out_flat = h_offsets[:, None]
    w_out_flat = w_offsets[None, :]

    # Load grid coordinates for entire tile (vectorized)
    grid_base = n * H_out * W_out * 2

    # Load x coordinates: (BLOCK_H, BLOCK_W)
    grid_x_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Load y coordinates: (BLOCK_H, BLOCK_W)
    grid_y_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2 + 1
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Handle NaN - use sentinel -1.0 like original kernel
    grid_x = tl.where(grid_x != grid_x, -1.0, grid_x)
    grid_y = tl.where(grid_y != grid_y, -1.0, grid_y)

    # Denormalize to pixel space
    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    # Compute corner indices for entire tile (vectorized)
    x0 = tl.floor(x)
    x1 = x0 + 1
    y0 = tl.floor(y)
    y1 = y0 + 1

    # Cast to int for indexing
    x0_int = tl.cast(x0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    y1_int = tl.cast(y1, tl.int32)

    # Clamp to valid range (border padding)
    x0_int = tl.maximum(0, tl.minimum(x0_int, W_in - 1))
    x1_int = tl.maximum(0, tl.minimum(x1_int, W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(y0_int, H_in - 1))
    y1_int = tl.maximum(0, tl.minimum(y1_int, H_in - 1))

    # Compute interpolation weights
    wx = x - tl.cast(x0, tl.float32)
    wy = y - tl.cast(y0, tl.float32)

    # Load 4 corner pixels (vectorized, no mask needed - clamping ensures validity)
    input_base = n * C * H_in * W_in + c * H_in * W_in

    p00_offsets = input_base + y0_int * W_in + x0_int
    p00 = tl.load(ptr_input + p00_offsets, mask=tile_mask, other=0.0)

    p01_offsets = input_base + y0_int * W_in + x1_int
    p01 = tl.load(ptr_input + p01_offsets, mask=tile_mask, other=0.0)

    p10_offsets = input_base + y1_int * W_in + x0_int
    p10 = tl.load(ptr_input + p10_offsets, mask=tile_mask, other=0.0)

    p11_offsets = input_base + y1_int * W_in + x1_int
    p11 = tl.load(ptr_input + p11_offsets, mask=tile_mask, other=0.0)

    # Bilinear interpolation (vectorized)
    top = p00 * (1.0 - wx) + p01 * wx
    bottom = p10 * (1.0 - wx) + p11 * wx
    vals = top * (1.0 - wy) + bottom * wy

    # Store to output
    output_base = n * C * H_out * W_out + c * H_out * W_out
    output_offsets = output_base + (h_out_flat * W_out + w_out_flat)

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_nearest_tiled"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_nearest_reflection_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 2D nearest neighbor interpolation with reflection padding (tiled version).

    Reflection padding: applies triangle wave reflection in grid space.
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_hw for spatial tile
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Compute tile position in output grid
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    # Compute pixel offsets within tile
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = h_mask[:, None] & w_mask[None, :]

    # Reshape for broadcasting: (BLOCK_H, BLOCK_W)
    h_out_flat = h_offsets[:, None]
    w_out_flat = w_offsets[None, :]

    # Load grid coordinates for entire tile (vectorized)
    grid_base = n * H_out * W_out * 2

    # Load x coordinates: (BLOCK_H, BLOCK_W)
    grid_x_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Load y coordinates: (BLOCK_H, BLOCK_W)
    grid_y_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2 + 1
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Apply triangle wave reflection in grid space (vectorized)
    # Triangle wave pattern with period 4
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    x = grid_x_refl_mod - 1.0

    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    y = grid_y_refl_mod - 1.0

    # Denormalize to pixel space
    if align_corners:
        x = (x + 1.0) * (W_in - 1) / 2.0
        y = (y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (x + 1.0) * W_in / 2.0 - 0.5
        y = (y + 1.0) * H_in / 2.0 - 0.5

    # Apply banker's rounding (vectorized across tile)
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    x_frac = x - x_floor
    y_frac = y - y_floor

    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    x_floor_int = tl.cast(x_floor, tl.int32)
    y_floor_int = tl.cast(y_floor, tl.int32)

    x_is_even = x_floor_int % 2 == 0
    y_is_even = y_floor_int % 2 == 0

    x_round = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_round = tl.where(y_frac < 0.5, y_floor, y_floor + 1)

    x_idx_unclamped = tl.cast(
        tl.where(x_is_half, tl.where(x_is_even, x_floor, x_floor + 1), x_round),
        tl.int32,
    )
    y_idx_unclamped = tl.cast(
        tl.where(y_is_half, tl.where(y_is_even, y_floor, y_floor + 1), y_round),
        tl.int32,
    )

    # Clamp to valid bounds (should already be in bounds due to reflection, but clamp for safety)
    x_idx = tl.maximum(0, tl.minimum(x_idx_unclamped, W_in - 1))
    y_idx = tl.maximum(0, tl.minimum(y_idx_unclamped, H_in - 1))

    # Load input pixels for entire tile
    input_base = n * C * H_in * W_in + c * H_in * W_in
    input_offsets = input_base + y_idx * W_in + x_idx

    vals = tl.load(ptr_input + input_offsets, mask=tile_mask, other=0.0)

    # Store to output
    output_base = n * C * H_out * W_out + c * H_out * W_out
    output_offsets = output_base + (h_out_flat * W_out + w_out_flat)

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bilinear_tiled"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bilinear_reflection_tiled_kernel(
    ptr_output,
    ptr_input,
    ptr_grid,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid sample kernel for 2D bilinear interpolation with reflection padding (tiled version).

    Reflection padding: applies triangle wave reflection in grid space.
    """
    # 2D program IDs: pid_nc for (batch, channel), pid_hw for spatial tile
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute batch and channel
    n = pid_nc // C
    c = pid_nc % C

    # Compute tile position in output grid
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    # Compute pixel offsets within tile
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for boundary tiles
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = h_mask[:, None] & w_mask[None, :]

    # Reshape for broadcasting: (BLOCK_H, BLOCK_W)
    h_out_flat = h_offsets[:, None]
    w_out_flat = w_offsets[None, :]

    # Load grid coordinates for entire tile (vectorized)
    grid_base = n * H_out * W_out * 2

    # Load x coordinates: (BLOCK_H, BLOCK_W)
    grid_x_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2
    grid_x = tl.load(ptr_grid + grid_x_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Load y coordinates: (BLOCK_H, BLOCK_W)
    grid_y_offsets = grid_base + (h_out_flat * W_out + w_out_flat) * 2 + 1
    grid_y = tl.load(ptr_grid + grid_y_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )

    # Apply triangle wave reflection in grid space (vectorized)
    grid_x_shifted = grid_x + 1.0
    grid_x_mod = grid_x_shifted % 4.0
    grid_x_mod = tl.where(grid_x_mod < 0, grid_x_mod + 4.0, grid_x_mod)
    grid_x_refl_mod = tl.where(grid_x_mod <= 2.0, grid_x_mod, 4.0 - grid_x_mod)
    x = grid_x_refl_mod - 1.0

    grid_y_shifted = grid_y + 1.0
    grid_y_mod = grid_y_shifted % 4.0
    grid_y_mod = tl.where(grid_y_mod < 0, grid_y_mod + 4.0, grid_y_mod)
    grid_y_refl_mod = tl.where(grid_y_mod <= 2.0, grid_y_mod, 4.0 - grid_y_mod)
    y = grid_y_refl_mod - 1.0

    # Denormalize to pixel space
    if align_corners:
        x = (x + 1.0) * (W_in - 1) / 2.0
        y = (y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (x + 1.0) * W_in / 2.0 - 0.5
        y = (y + 1.0) * H_in / 2.0 - 0.5

    # Compute corner indices for entire tile (vectorized)
    x0 = tl.floor(x)
    x1 = x0 + 1
    y0 = tl.floor(y)
    y1 = y0 + 1

    # Cast to int for indexing
    x0_int = tl.cast(x0, tl.int32)
    x1_int = tl.cast(x1, tl.int32)
    y0_int = tl.cast(y0, tl.int32)
    y1_int = tl.cast(y1, tl.int32)

    # Clamp to valid bounds (should already be in bounds due to reflection)
    x0_int = tl.maximum(0, tl.minimum(x0_int, W_in - 1))
    x1_int = tl.maximum(0, tl.minimum(x1_int, W_in - 1))
    y0_int = tl.maximum(0, tl.minimum(y0_int, H_in - 1))
    y1_int = tl.maximum(0, tl.minimum(y1_int, H_in - 1))

    # Compute interpolation weights
    wx = x - tl.cast(x0, tl.float32)
    wy = y - tl.cast(y0, tl.float32)

    # Load 4 corner pixels (vectorized)
    input_base = n * C * H_in * W_in + c * H_in * W_in

    p00_offsets = input_base + y0_int * W_in + x0_int
    p00 = tl.load(ptr_input + p00_offsets, mask=tile_mask, other=0.0)

    p01_offsets = input_base + y0_int * W_in + x1_int
    p01 = tl.load(ptr_input + p01_offsets, mask=tile_mask, other=0.0)

    p10_offsets = input_base + y1_int * W_in + x0_int
    p10 = tl.load(ptr_input + p10_offsets, mask=tile_mask, other=0.0)

    p11_offsets = input_base + y1_int * W_in + x1_int
    p11 = tl.load(ptr_input + p11_offsets, mask=tile_mask, other=0.0)

    # Bilinear interpolation (vectorized)
    top = p00 * (1.0 - wx) + p01 * wx
    bottom = p10 * (1.0 - wx) + p11 * wx
    vals = top * (1.0 - wy) + bottom * wy

    # Store to output
    output_base = n * C * H_out * W_out + c * H_out * W_out
    output_offsets = output_base + (h_out_flat * W_out + w_out_flat)

    tl.store(ptr_output + output_offsets, vals, mask=tile_mask)


# ============================================================================
# Main Dispatch Function
# ============================================================================


def grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Grid sample operation with spatial interpolation.

    Computes the output using input values and pixel locations from grid.
    Grid specifies sampling pixel locations normalized by input spatial dimensions.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in) or (N, C, D_in, H_in, W_in)
        grid: Grid tensor of shape (N, H_out, W_out, 2) or (N, D_out, H_out, W_out, 3)
               Values should be in range [-1, 1], normalized by input spatial dimensions
        mode: Interpolation mode - 'bilinear', 'nearest', or 'bicubic' (4D only)
        padding_mode: Padding mode for out-of-bound grid locations
                     - 'zeros': use 0 for out-of-bound locations
                     - 'border': use border values
                     - 'reflection': reflect by border
        align_corners: If True, extrema (-1, 1) refer to center points of corner pixels
                      If False, extrema refer to corner points of corner pixels

    Returns:
        Output tensor of shape (N, C, H_out, W_out) or (N, C, D_out, H_out, W_out)

    Examples:
        >>> input = torch.randn(1, 3, 32, 32).cuda()
        >>> grid = torch.randn(1, 64, 64, 2).cuda()
        >>> output = grid_sample(input, grid, mode='bilinear')
        >>> print(output.shape)
        torch.Size([1, 3, 64, 64])
    """
    # Validate inputs
    _validate_grid_sample_input(input, grid, mode, padding_mode)

    # Get tensor properties
    dtype = input.dtype
    device = input.device

    is_3d = input.dim() == 5

    # Handle 4D inputs (N, C, H_in, W_in)
    if not is_3d:
        N, C, H_in, W_in = input.shape
        _, H_out, W_out, _ = grid.shape

        # Allocate output tensor
        output = torch.empty((N, C, H_out, W_out), dtype=dtype, device=device)

        # Adaptive kernel selection based on output size
        # Use tiled kernels for medium-to-large outputs (>= 32x32 = 1024 pixels)
        # Use original per-pixel kernels for small outputs (< 32x32)
        output_pixels = H_out * W_out
        USE_TILED_THRESHOLD = 1024

        use_tiled = output_pixels >= USE_TILED_THRESHOLD

        # Select kernel based on mode, padding mode, and output size
        if mode == "nearest":
            if use_tiled:
                # Use tiled kernels for medium-to-large outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_2d_nearest_zeros_tiled_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_2d_nearest_border_tiled_kernel
                else:  # reflection
                    kernel = grid_sample_2d_nearest_reflection_tiled_kernel
            else:
                # Use original kernels for small outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_2d_nearest_zeros_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_2d_nearest_border_kernel
                else:  # reflection
                    kernel = grid_sample_2d_nearest_reflection_kernel
        elif mode == "bilinear":
            if use_tiled:
                # Use tiled kernels for medium-to-large outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_2d_bilinear_zeros_tiled_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_2d_bilinear_border_tiled_kernel
                else:  # reflection
                    kernel = grid_sample_2d_bilinear_reflection_tiled_kernel
            else:
                # Use original kernels for small outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_2d_bilinear_zeros_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_2d_bilinear_border_kernel
                else:  # reflection
                    kernel = grid_sample_2d_bilinear_reflection_kernel
        elif mode == "bicubic":
            # Bicubic is already competitive, use original kernels for all sizes
            if padding_mode == "zeros":
                kernel = grid_sample_2d_bicubic_zeros_kernel
            elif padding_mode == "border":
                kernel = grid_sample_2d_bicubic_border_kernel
            else:  # reflection
                kernel = grid_sample_2d_bicubic_reflection_kernel
        else:  # unsupported mode
            logger.info(f"grid_sample mode '{mode}' not supported")
            raise NotImplementedError

        # Launch kernel with appropriate grid size
        # For very large outputs (> 512x512), fall back to original kernels to avoid grid size issues
        output_pixels = H_out * W_out
        MAX_TILED_PIXELS = 512 * 512

        if (
            use_tiled
            and mode in ["nearest", "bilinear"]
            and output_pixels <= MAX_TILED_PIXELS
        ):
            # Tiled kernels use 2D grid with adaptive tile size selection
            # Goal: Create ~100-500 blocks total for good GPU utilization
            target_total_blocks = (
                300  # Target: aim for ~300 blocks across all batches/channels
            )
            min_blocks_per_nc = 50  # Minimum: ensure enough parallelism
            max_blocks_per_nc = 1000  # Maximum: avoid too many blocks

            # Estimate blocks per (batch, channel) pair
            nc_pairs = N * C

            # Target blocks per (N, C) pair
            target_blocks_per_nc = max(
                min_blocks_per_nc,
                min(max_blocks_per_nc, target_total_blocks // max(1, nc_pairs)),
            )

            # Calculate tile dimensions to achieve target block count
            # Start with square tiles
            target_tile_pixels = output_pixels // target_blocks_per_nc
            target_tile_side = int(max(4, min(128, int(target_tile_pixels**0.5))))

            # Snap to power-of-2 for better alignment
            if target_tile_side >= 64:
                block_h = block_w = 64 if target_tile_side < 96 else 128
            elif target_tile_side >= 16:
                block_h = block_w = 32
            elif target_tile_side >= 8:
                block_h = block_w = 16
            else:
                block_h = block_w = 8

            # For bilinear, use smaller tiles due to higher memory footprint
            if mode == "bilinear":
                block_h = max(4, block_h // 2)
                block_w = max(4, block_w // 2)

            # Calculate actual grid size
            num_h_blocks = (H_out + block_h - 1) // block_h
            num_w_blocks = (W_out + block_w - 1) // block_w
            grid_size = (N * C, num_h_blocks * num_w_blocks)
        else:
            # Original kernels use 1D grid (for small outputs or very large outputs)
            grid_size = (N * C * H_out * W_out,)

        kernel[grid_size](
            output,
            input,
            grid,
            N,
            C,
            H_in,
            W_in,
            H_out,
            W_out,
            align_corners,
        )

        return output

    # Handle 5D inputs (N, C, D_in, H_in, W_in)
    else:  # is_3d == True
        N, C, D_in, H_in, W_in = input.shape
        _, D_out, H_out, W_out, _ = grid.shape

        # Allocate output tensor
        output = torch.empty((N, C, D_out, H_out, W_out), dtype=dtype, device=device)

        # Adaptive kernel selection based on output size
        # Use tiled kernels for medium-to-large outputs (>= 16x16x16 = 4096 voxels)
        # Increased from 512 to avoid tiled kernel overhead on small outputs
        output_voxels = D_out * H_out * W_out
        USE_TILED_THRESHOLD_3D = 4096  # 16x16x16

        use_tiled = output_voxels >= USE_TILED_THRESHOLD_3D

        # Select kernel based on mode, padding mode, and output size
        if mode == "nearest":
            if use_tiled:
                # Use tiled kernels for medium-to-large outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_3d_nearest_zeros_tiled_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_3d_nearest_border_tiled_kernel
                else:  # reflection
                    kernel = grid_sample_3d_nearest_reflection_tiled_kernel
            else:
                # Use original kernels for small outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_3d_nearest_zeros_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_3d_nearest_border_kernel
                else:  # reflection
                    kernel = grid_sample_3d_nearest_reflection_kernel
        elif mode == "bilinear":  # For 5D, bilinear means trilinear
            if use_tiled:
                # Use tiled kernels for medium-to-large outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_3d_trilinear_zeros_tiled_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_3d_trilinear_border_tiled_kernel
                else:  # reflection
                    kernel = grid_sample_3d_trilinear_reflection_tiled_kernel
            else:
                # Use original kernels for small outputs
                if padding_mode == "zeros":
                    kernel = grid_sample_3d_trilinear_zeros_kernel
                elif padding_mode == "border":
                    kernel = grid_sample_3d_trilinear_border_kernel
                else:  # reflection
                    kernel = grid_sample_3d_trilinear_reflection_kernel
        else:  # unsupported mode for 5D
            logger.info(f"grid_sample mode '{mode}' not supported for 5D input")
            raise NotImplementedError("Unsupported mode for 5D input")

        # Launch kernel with appropriate grid size
        # For very large outputs (> 128x128x128), fall back to original kernels
        MAX_TILED_VOXELS = 128 * 128 * 128  # ~2M voxels

        if (
            use_tiled
            and mode in ["nearest", "bilinear"]
            and output_voxels <= MAX_TILED_VOXELS
        ):
            # Tiled kernels use 2D grid with adaptive tile size selection
            # Goal: Create optimal blocks for good GPU utilization (more granular for medium outputs)
            nc_pairs = N * C

            # More granular targeting to fix 16³ and 32³ performance
            # Key: Need MORE blocks for 16³ and 32³, not fewer
            if output_voxels < 8192:  # 16³ - 20³
                target_total_blocks = 512  # MORE blocks for 16³ (was 100)
                min_blocks_per_nc = 64
                max_blocks_per_nc = 1024
            elif output_voxels < 32768:  # 20³ - 32³
                target_total_blocks = 768  # MORE blocks for 20³-32³
                min_blocks_per_nc = 128
                max_blocks_per_nc = 2048
            elif output_voxels < 131072:  # 32³ - 50³
                target_total_blocks = 1024  # More blocks for medium-large
                min_blocks_per_nc = 128
                max_blocks_per_nc = 2048
            elif output_voxels < 262144:  # 50³ - 64³
                target_total_blocks = 512  # Reduced for 64³
                min_blocks_per_nc = 64
                max_blocks_per_nc = 1024
            else:  # Large outputs (>= 64³)
                target_total_blocks = 300  # Original for very large
                min_blocks_per_nc = 50
                max_blocks_per_nc = 1000

            # Channel-aware tiling: reduce targets for high channel counts to avoid too many blocks
            # When C is large, we create too many blocks with the current formula
            # Solution: Reduce target_total_blocks proportionally
            if C > 32:
                # Scale down targets more aggressively to avoid excessive blocks when C > 32
                # Use sqrt scaling for better balance
                channel_scale = (32.0 / C) ** 0.7  # More aggressive scaling
                target_total_blocks = max(128, int(target_total_blocks * channel_scale))
                min_blocks_per_nc = max(16, int(min_blocks_per_nc * channel_scale))
                # Keep max_blocks_per_nc unchanged to prevent excessive blocks

            # Target blocks per (N, C) pair
            target_blocks_per_nc = max(
                min_blocks_per_nc,
                min(max_blocks_per_nc, target_total_blocks // max(1, nc_pairs)),
            )

            # Calculate tile dimensions to achieve target block count
            # For 3D, start with cubic tiles
            total_voxels = D_out * H_out * W_out
            target_tile_voxels = total_voxels // target_blocks_per_nc
            target_tile_side = int(
                max(4, min(64, int(target_tile_voxels ** (1.0 / 3.0))))
            )  # Reverted to min=4

            # Snap to power-of-2 for better alignment
            # Minimum tile size is 4x4x4 for small outputs, 8x8x8 for large
            if target_tile_side >= 32:
                block_d = block_h = block_w = 32 if target_tile_side < 48 else 64
            elif target_tile_side >= 16:
                block_d = block_h = block_w = 16
            elif target_tile_side >= 8:
                block_d = block_h = block_w = 8
            else:
                block_d = block_h = block_w = 4  # Reverted from 8

            # For trilinear, use smaller tiles due to higher memory footprint (8x loads)
            if mode == "bilinear":  # actually trilinear in 5D
                block_d = max(2, block_d // 2)  # Reverted from max(4, ...)
                block_h = max(2, block_h // 2)  # Reverted from max(4, ...)
                block_w = max(2, block_w // 2)  # Reverted from max(4, ...)

            # Calculate actual grid size
            num_d_blocks = (D_out + block_d - 1) // block_d
            num_h_blocks = (H_out + block_h - 1) // block_h
            num_w_blocks = (W_out + block_w - 1) // block_w
            grid_size = (N * C, num_d_blocks * num_h_blocks * num_w_blocks)
        else:
            # Original kernels use 1D grid (for small outputs or very large outputs)
            grid_size = (N * C * D_out * H_out * W_out,)

        # Kernel launch
        kernel[grid_size](
            output,
            input,
            grid,
            N,
            C,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            align_corners,
        )

        return output
