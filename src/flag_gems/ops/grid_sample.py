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


# Cache for the "does this device support bf16 hardware ops" check.  ptxas
# requires sm_80+ for native bf16 instructions; on Turing (sm_75 — RTX 20xx,
# Titan RTX) and earlier the kernel won't even compile for bf16.  We upcast
# bf16 inputs to fp32 in the forward entry to keep the same op surface.
_BF16_HW_CAP = {}


def _device_lacks_bf16_hw(device):
    """Return True iff the device's compute capability major < 8 (no bf16 hw)."""
    key = (device.type, getattr(device, "index", 0))
    if key not in _BF16_HW_CAP:
        if device.type != "cuda":
            _BF16_HW_CAP[key] = False
        else:
            cap = torch.cuda.get_device_capability(device)
            _BF16_HW_CAP[key] = cap[0] < 8
    return _BF16_HW_CAP[key]


# ============================================================================
# Grid Sample Constants
# ============================================================================

# Maximum tiled voxel count for tiled kernel usage
MAX_TILED_VOXELS = 128 * 128 * 128  # ~2M voxels

# Voxel thresholds for adaptive block targeting
# These represent approximate cube dimensions: 16³=4096, 20³=8000, 32³=32768, 50³=125000, 64³=262144
VOXEL_THRESHOLD_SMALL = 8192  # Threshold for small outputs (16³ - 20³)
VOXEL_THRESHOLD_MEDIUM = 32768  # Threshold for medium outputs (20³ - 32³)
VOXEL_THRESHOLD_LARGE = 131072  # Threshold for large outputs (32³ - 50³)
VOXEL_THRESHOLD_VERY_LARGE = 262144  # Threshold for very large outputs (50³ - 64³)

# Block target configuration for different output sizes
# Small outputs (16³ - 20³): Higher block count for better utilization
TARGET_BLOCKS_SMALL = 512
MIN_BLOCKS_NC_SMALL = 64
MAX_BLOCKS_NC_SMALL = 1024

# Medium outputs (20³ - 32³): Even higher block count
TARGET_BLOCKS_MEDIUM = 768
MIN_BLOCKS_NC_MEDIUM = 128
MAX_BLOCKS_NC_MEDIUM = 2048

# Large outputs (32³ - 50³): Maximum block targeting
TARGET_BLOCKS_LARGE = 1024
MIN_BLOCKS_NC_LARGE = 128
MAX_BLOCKS_NC_LARGE = 2048

# Very large outputs (50³ - 64³): Reduced block count
TARGET_BLOCKS_VERY_LARGE = 512
MIN_BLOCKS_NC_VERY_LARGE = 64
MAX_BLOCKS_NC_VERY_LARGE = 1024

# Extra large outputs (>= 64³): Conservative block targeting
TARGET_BLOCKS_EXTRA_LARGE = 300
MIN_BLOCKS_NC_EXTRA_LARGE = 50
MAX_BLOCKS_NC_EXTRA_LARGE = 1000

# Channel scaling constants
CHANNEL_COUNT_THRESHOLD = 32  # Channel count above which to scale down block targets
CHANNEL_SCALING_EXPONENT = 0.7  # Exponent for channel scaling factor
MIN_TARGET_TOTAL_BLOCKS = 128  # Minimum target total blocks when scaling for channels
MIN_BLOCKS_PER_NC = 16  # Minimum blocks per (N, C) pair when scaling for channels

# Tile size constants
MIN_TILE_SIDE = 4  # Minimum tile side length for 3D outputs
MAX_TILE_SIDE = 64  # Maximum tile side length for 3D outputs
LARGE_TILE_THRESHOLD = 32  # Threshold for using 32 or 64 sized tiles
VERY_LARGE_TILE_THRESHOLD = 48  # Threshold for using 64 instead of 32
MEDIUM_TILE_THRESHOLD = 16  # Threshold for using 16 sized tiles
SMALL_TILE_THRESHOLD = 8  # Threshold for using 8 sized tiles

# Trilinear reduction constants
MIN_BLOCK_DIMENSION = 2  # Minimum block dimension after halving for trilinear


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
# 3D trilinear tiled forward kernel with CHANNEL REUSE.
#
# The old 3D tiled kernels assign one Triton program per (n, c) pair, which
# means the grid coord load, denorm, mask, and weight computations are
# duplicated C times per spatial tile.  At C=32 (the largest benchmark
# config) that's a 32x compute overhead per spatial tile that contributes
# nothing.  This version takes pid_n × pid_dhw (no C in pid), then walks the
# channel axis inside the kernel with a Triton `for` loop -- coord/denorm/
# mask work happens once per (n, dhw_tile), and the C-loop just does the
# 8-corner load + lerp + store per channel.
#
# Handles all 3 padding modes via padding_mode_id constexpr to avoid kernel
# duplication.  Per-sample padding semantics are not relevant for trilinear
# (matches PyTorch behaviour where padding applies to the central coord).
# ============================================================================
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear_tiled_v2"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_trilinear_tiled_v2_kernel(
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
    padding_mode_id: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_dhw = tl.program_id(1)
    n = pid_n

    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks
    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    d_offs = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offs = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    d_mask = d_offs < D_out
    h_mask = h_offs < H_out
    w_mask = w_offs < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    d3 = d_offs[:, None, None]
    h3 = h_offs[None, :, None]
    w3 = w_offs[None, None, :]

    grid_base = n * D_out * H_out * W_out * 3
    spatial_off = d3 * H_out * W_out + h3 * W_out + w3
    grid_x = tl.load(
        ptr_grid + grid_base + spatial_off * 3, mask=tile_mask, other=0.0
    ).to(tl.float32)
    grid_y = tl.load(
        ptr_grid + grid_base + spatial_off * 3 + 1, mask=tile_mask, other=0.0
    ).to(tl.float32)
    grid_z = tl.load(
        ptr_grid + grid_base + spatial_off * 3 + 2, mask=tile_mask, other=0.0
    ).to(tl.float32)

    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    # Apply padding mode to continuous coords (matches existing per-pixel
    # kernel semantics for trilinear: border clamps the float coord and
    # bilinear weights remain unaffected; reflection folds via triangle).
    if padding_mode_id == 1:  # border
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))
    elif padding_mode_id == 2:  # reflection
        # Reflect each coord via triangle wave; the existing per-pixel
        # reflection kernels do this in *grid* space, but mathematically the
        # equivalent post-denorm formulation uses [low, high] depending on AC.
        if align_corners:
            low_x = 0.0
            high_x = W_in - 1.0
            low_y = 0.0
            high_y = H_in - 1.0
            low_z = 0.0
            high_z = D_in - 1.0
        else:
            low_x = -0.5
            high_x = W_in * 1.0 - 0.5
            low_y = -0.5
            high_y = H_in * 1.0 - 0.5
            low_z = -0.5
            high_z = D_in * 1.0 - 0.5

        # x reflection
        span = high_x - low_x
        period = 2.0 * span
        sh = x - low_x
        nr = tl.floor(sh / period)
        m = sh - nr * period
        folded = m > span
        x = tl.where(folded, period - m, m) + low_x
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        # y reflection
        span = high_y - low_y
        period = 2.0 * span
        sh = y - low_y
        nr = tl.floor(sh / period)
        m = sh - nr * period
        folded = m > span
        y = tl.where(folded, period - m, m) + low_y
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        # z reflection
        span = high_z - low_z
        period = 2.0 * span
        sh = z - low_z
        nr = tl.floor(sh / period)
        m = sh - nr * period
        folded = m > span
        z = tl.where(folded, period - m, m) + low_z
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))

    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x0i = tl.cast(x0, tl.int32)
    y0i = tl.cast(y0, tl.int32)
    z0i = tl.cast(z0, tl.int32)
    x1i = x0i + 1
    y1i = y0i + 1
    z1i = z0i + 1
    wx = x - x0
    wy = y - y0
    wz = z - z0
    mx = 1.0 - wx
    my = 1.0 - wy
    mz = 1.0 - wz

    # Per-corner bounds (zeros padding masks loads; border/reflection have
    # x/y/z already clamped into [0, size-1] but x1i may be at size — mask it).
    x0_in = (x0i >= 0) & (x0i < W_in)
    x1_in = (x1i >= 0) & (x1i < W_in)
    y0_in = (y0i >= 0) & (y0i < H_in)
    y1_in = (y1i >= 0) & (y1i < H_in)
    z0_in = (z0i >= 0) & (z0i < D_in)
    z1_in = (z1i >= 0) & (z1i < D_in)

    valid_grid = tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan
    in000 = valid_grid & z0_in & y0_in & x0_in
    in001 = valid_grid & z0_in & y0_in & x1_in
    in010 = valid_grid & z0_in & y1_in & x0_in
    in011 = valid_grid & z0_in & y1_in & x1_in
    in100 = valid_grid & z1_in & y0_in & x0_in
    in101 = valid_grid & z1_in & y0_in & x1_in
    in110 = valid_grid & z1_in & y1_in & x0_in
    in111 = valid_grid & z1_in & y1_in & x1_in

    # 8 weights (computed once, reused for every channel).
    w000 = mz * my * mx
    w001 = mz * my * wx
    w010 = mz * wy * mx
    w011 = mz * wy * wx
    w100 = wz * my * mx
    w101 = wz * my * wx
    w110 = wz * wy * mx
    w111 = wz * wy * wx

    # Voxel offsets within one channel's slab.
    off000 = z0i * H_in * W_in + y0i * W_in + x0i
    off001 = z0i * H_in * W_in + y0i * W_in + x1i
    off010 = z0i * H_in * W_in + y1i * W_in + x0i
    off011 = z0i * H_in * W_in + y1i * W_in + x1i
    off100 = z1i * H_in * W_in + y0i * W_in + x0i
    off101 = z1i * H_in * W_in + y0i * W_in + x1i
    off110 = z1i * H_in * W_in + y1i * W_in + x0i
    off111 = z1i * H_in * W_in + y1i * W_in + x1i

    in_stride_c = D_in * H_in * W_in
    out_stride_c = D_out * H_out * W_out
    output_base_n = n * C * out_stride_c
    input_base_n = n * C * in_stride_c

    # C-loop -- one channel per iteration.  The 8 corner offsets + weights
    # + masks above are reused.  num_stages > 1 lets Triton overlap load
    # latency with prior iter's compute.
    for c in tl.range(0, C, 1, num_stages=4):
        in_base = input_base_n + c * in_stride_c
        p000 = tl.load(ptr_input + in_base + off000, mask=in000, other=0.0).to(
            tl.float32
        )
        p001 = tl.load(ptr_input + in_base + off001, mask=in001, other=0.0).to(
            tl.float32
        )
        p010 = tl.load(ptr_input + in_base + off010, mask=in010, other=0.0).to(
            tl.float32
        )
        p011 = tl.load(ptr_input + in_base + off011, mask=in011, other=0.0).to(
            tl.float32
        )
        p100 = tl.load(ptr_input + in_base + off100, mask=in100, other=0.0).to(
            tl.float32
        )
        p101 = tl.load(ptr_input + in_base + off101, mask=in101, other=0.0).to(
            tl.float32
        )
        p110 = tl.load(ptr_input + in_base + off110, mask=in110, other=0.0).to(
            tl.float32
        )
        p111 = tl.load(ptr_input + in_base + off111, mask=in111, other=0.0).to(
            tl.float32
        )

        vals = (
            p000 * w000
            + p001 * w001
            + p010 * w010
            + p011 * w011
            + p100 * w100
            + p101 * w101
            + p110 * w110
            + p111 * w111
        )

        out_base = output_base_n + c * out_stride_c
        tl.store(ptr_output + out_base + spatial_off, vals, mask=tile_mask)


# ============================================================================
# 3D nearest tiled v2 (channel-reuse layout, single kernel for all paddings).
#
# Old layout: pid = (n*c, dhw_tile) — coord load + denorm + padding + rounding
# duplicated C times.  New layout: pid = (n, dhw_tile) and a C-loop walks the
# channel axis, reusing the precomputed integer index and bounds mask.  The
# nearest mode is simpler than trilinear (1 voxel load vs 8 corner loads),
# so the relative amortization of the coord work is even larger.
# ============================================================================
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest_tiled_v2"),
    key=["N", "C", "D_out", "H_out", "W_out"],
)
@triton.jit
def grid_sample_3d_nearest_tiled_v2_kernel(
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
    padding_mode_id: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_dhw = tl.program_id(1)
    n = pid_n

    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks
    d_block_idx = pid_dhw // num_hw_blocks
    hw_block_idx = pid_dhw % num_hw_blocks
    h_block_idx = hw_block_idx // num_w_blocks
    w_block_idx = hw_block_idx % num_w_blocks

    d_offs = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offs = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    d_mask = d_offs < D_out
    h_mask = h_offs < H_out
    w_mask = w_offs < W_out
    tile_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    d3 = d_offs[:, None, None]
    h3 = h_offs[None, :, None]
    w3 = w_offs[None, None, :]

    grid_base = n * D_out * H_out * W_out * 3
    spatial_off = d3 * H_out * W_out + h3 * W_out + w3
    grid_x = tl.load(
        ptr_grid + grid_base + spatial_off * 3, mask=tile_mask, other=0.0
    ).to(tl.float32)
    grid_y = tl.load(
        ptr_grid + grid_base + spatial_off * 3 + 1, mask=tile_mask, other=0.0
    ).to(tl.float32)
    grid_z = tl.load(
        ptr_grid + grid_base + spatial_off * 3 + 2, mask=tile_mask, other=0.0
    ).to(tl.float32)

    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_z_nan = grid_z != grid_z
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)
    grid_z = tl.where(grid_z_nan, -2.0, grid_z)

    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
        z = (grid_z + 1.0) * (D_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5
        z = (grid_z + 1.0) * D_in / 2.0 - 0.5

    if padding_mode_id == 1:  # border
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))
    elif padding_mode_id == 2:  # reflection
        if align_corners:
            low_x = 0.0
            high_x = W_in - 1.0
            low_y = 0.0
            high_y = H_in - 1.0
            low_z = 0.0
            high_z = D_in - 1.0
        else:
            low_x = -0.5
            high_x = W_in * 1.0 - 0.5
            low_y = -0.5
            high_y = H_in * 1.0 - 0.5
            low_z = -0.5
            high_z = D_in * 1.0 - 0.5
        span = high_x - low_x
        period = 2.0 * span
        sh = x - low_x
        nr = tl.floor(sh / period)
        m = sh - nr * period
        folded = m > span
        x = tl.where(folded, period - m, m) + low_x
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        span = high_y - low_y
        period = 2.0 * span
        sh = y - low_y
        nr = tl.floor(sh / period)
        m = sh - nr * period
        folded = m > span
        y = tl.where(folded, period - m, m) + low_y
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        span = high_z - low_z
        period = 2.0 * span
        sh = z - low_z
        nr = tl.floor(sh / period)
        m = sh - nr * period
        folded = m > span
        z = tl.where(folded, period - m, m) + low_z
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))

    # PyTorch nearbyint = banker's rounding (round half to even).
    x_floor = tl.floor(x)
    y_floor = tl.floor(y)
    z_floor = tl.floor(z)
    x_frac = x - x_floor
    y_frac = y - y_floor
    z_frac = z - z_floor
    x_is_half = x_frac == 0.5
    y_is_half = y_frac == 0.5
    z_is_half = z_frac == 0.5
    x_fi = tl.cast(x_floor, tl.int32)
    y_fi = tl.cast(y_floor, tl.int32)
    z_fi = tl.cast(z_floor, tl.int32)
    x_even = x_fi % 2 == 0
    y_even = y_fi % 2 == 0
    z_even = z_fi % 2 == 0
    x_rnd = tl.where(x_frac < 0.5, x_floor, x_floor + 1)
    y_rnd = tl.where(y_frac < 0.5, y_floor, y_floor + 1)
    z_rnd = tl.where(z_frac < 0.5, z_floor, z_floor + 1)
    x_idx = tl.cast(
        tl.where(x_is_half, tl.where(x_even, x_floor, x_floor + 1), x_rnd), tl.int32
    )
    y_idx = tl.cast(
        tl.where(y_is_half, tl.where(y_even, y_floor, y_floor + 1), y_rnd), tl.int32
    )
    z_idx = tl.cast(
        tl.where(z_is_half, tl.where(z_even, z_floor, z_floor + 1), z_rnd), tl.int32
    )

    valid_grid = tile_mask & ~grid_x_nan & ~grid_y_nan & ~grid_z_nan
    in_bounds = (
        valid_grid
        & (x_idx >= 0)
        & (x_idx < W_in)
        & (y_idx >= 0)
        & (y_idx < H_in)
        & (z_idx >= 0)
        & (z_idx < D_in)
    )

    voxel_off = z_idx * H_in * W_in + y_idx * W_in + x_idx
    in_stride_c = D_in * H_in * W_in
    out_stride_c = D_out * H_out * W_out
    output_base_n = n * C * out_stride_c
    input_base_n = n * C * in_stride_c

    for c in tl.range(0, C, 1, num_stages=4):
        in_base = input_base_n + c * in_stride_c
        val = tl.load(ptr_input + in_base + voxel_off, mask=in_bounds, other=0.0)
        out_base = output_base_n + c * out_stride_c
        tl.store(ptr_output + out_base + spatial_off, val, mask=tile_mask)


# ============================================================================
# Bicubic tiled forward kernel (handles all 3 padding modes via constexpr).
#
# The per-pixel bicubic kernels above each process ONE output pixel per Triton
# program with num_warps in {1, 4, 8} — 95%+ of warp lanes are idle.  This
# tiled version processes BLOCK_H * BLOCK_W output pixels per program, with
# accumulators on the (BLOCK_H, BLOCK_W) tile so all lanes carry useful work.
# The 4x4 = 16 sample stencil is walked with a Python-range loop that Triton
# unrolls at trace time; weight selection uses an if-elif on the Python int
# so each (i, j) compiles to a specialized code path.
# ============================================================================
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_2d_bicubic_tiled"),
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def grid_sample_2d_bicubic_tiled_kernel(
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
    padding_mode_id: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Bicubic 2D forward, tiled.

    padding_mode_id: 0=zeros, 1=border, 2=reflection.  Padding is applied
    per-sample-index (each of the 16 neighbour reads), matching PyTorch's
    bicubic semantics; the central x/y are not pre-clipped.
    """
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    tile_mask = h_mask[:, None] & w_mask[None, :]

    h_flat = h_offsets[:, None]
    w_flat = w_offsets[None, :]

    grid_base = n * H_out * W_out * 2
    grid_x = tl.load(
        ptr_grid + grid_base + (h_flat * W_out + w_flat) * 2,
        mask=tile_mask,
        other=0.0,
    ).to(tl.float32)
    grid_y = tl.load(
        ptr_grid + grid_base + (h_flat * W_out + w_flat) * 2 + 1,
        mask=tile_mask,
        other=0.0,
    ).to(tl.float32)

    # NaN sentinel — out of [-1, 1] so all 16 samples will be oob/clipped.
    grid_x_nan = grid_x != grid_x
    grid_y_nan = grid_y != grid_y
    grid_x = tl.where(grid_x_nan, -2.0, grid_x)
    grid_y = tl.where(grid_y_nan, -2.0, grid_y)

    if align_corners:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        x = (grid_x + 1.0) * W_in / 2.0 - 0.5
        y = (grid_y + 1.0) * H_in / 2.0 - 0.5

    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x0i = tl.cast(x0, tl.int32)
    y0i = tl.cast(y0, tl.int32)
    wx = x - x0
    wy = y - y0

    # Cubic-kernel weights for the 4 X-samples and 4 Y-samples.
    # K(t) = 1.25 t^3 - 2.25 t^2 + 1            for t in [0, 1]
    #      = -0.75 t^3 + 3.75 t^2 - 6 t + 3      for t in (1, 2]
    # Sample distances: { wx+1, wx, 1-wx, 2-wx } and analogously for y.
    tx0 = wx + 1.0
    tx1 = wx
    tx2 = 1.0 - wx
    tx3 = 2.0 - wx
    ty0 = wy + 1.0
    ty1 = wy
    ty2 = 1.0 - wy
    ty3 = 2.0 - wy
    Wx0 = -0.75 * tx0 * tx0 * tx0 + 3.75 * tx0 * tx0 - 6.0 * tx0 + 3.0
    Wx1 = 1.25 * tx1 * tx1 * tx1 - 2.25 * tx1 * tx1 + 1.0
    Wx2 = 1.25 * tx2 * tx2 * tx2 - 2.25 * tx2 * tx2 + 1.0
    Wx3 = -0.75 * tx3 * tx3 * tx3 + 3.75 * tx3 * tx3 - 6.0 * tx3 + 3.0
    Wy0 = -0.75 * ty0 * ty0 * ty0 + 3.75 * ty0 * ty0 - 6.0 * ty0 + 3.0
    Wy1 = 1.25 * ty1 * ty1 * ty1 - 2.25 * ty1 * ty1 + 1.0
    Wy2 = 1.25 * ty2 * ty2 * ty2 - 2.25 * ty2 * ty2 + 1.0
    Wy3 = -0.75 * ty3 * ty3 * ty3 + 3.75 * ty3 * ty3 - 6.0 * ty3 + 3.0

    # Reflection period constants (used only when padding_mode_id == 2).
    if align_corners:
        period_x = 2 * (W_in - 1)
        period_y = 2 * (H_in - 1)
    else:
        period_x = 2 * W_in
        period_y = 2 * H_in

    input_base = n * C * H_in * W_in + c * H_in * W_in
    val = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # 4x4 stencil walked with Python ints so each (i, j) specializes.
    for i_off in range(4):
        di = i_off - 1
        if i_off == 0:
            Wyi = Wy0
        elif i_off == 1:
            Wyi = Wy1
        elif i_off == 2:
            Wyi = Wy2
        else:
            Wyi = Wy3

        yy_raw = y0i + di
        if padding_mode_id == 1:  # border
            yy_eff = tl.maximum(0, tl.minimum(yy_raw, H_in - 1))
            yy_in = yy_raw == yy_raw
        elif padding_mode_id == 2:  # reflection
            ym = (yy_raw % period_y + period_y) % period_y
            if align_corners:
                yy_eff = tl.where(ym <= H_in - 1, ym, period_y - ym)
            else:
                yy_eff = tl.where(ym <= H_in - 1, ym, period_y - 1 - ym)
            yy_in = yy_raw == yy_raw
        else:  # zeros
            yy_eff = yy_raw
            yy_in = (yy_raw >= 0) & (yy_raw < H_in)

        for j_off in range(4):
            dj = j_off - 1
            if j_off == 0:
                Wxj = Wx0
            elif j_off == 1:
                Wxj = Wx1
            elif j_off == 2:
                Wxj = Wx2
            else:
                Wxj = Wx3

            xx_raw = x0i + dj
            if padding_mode_id == 1:  # border
                xx_eff = tl.maximum(0, tl.minimum(xx_raw, W_in - 1))
                in_ij = yy_in
            elif padding_mode_id == 2:  # reflection
                xm = (xx_raw % period_x + period_x) % period_x
                if align_corners:
                    xx_eff = tl.where(xm <= W_in - 1, xm, period_x - xm)
                else:
                    xx_eff = tl.where(xm <= W_in - 1, xm, period_x - 1 - xm)
                in_ij = yy_in
            else:  # zeros
                xx_eff = xx_raw
                in_ij = yy_in & (xx_raw >= 0) & (xx_raw < W_in)

            load_mask = tile_mask & in_ij & ~grid_x_nan & ~grid_y_nan
            p = tl.load(
                ptr_input + input_base + yy_eff * W_in + xx_eff,
                mask=load_mask,
                other=0.0,
            ).to(tl.float32)
            val += p * (Wxj * Wyi)

    output_base = n * C * H_out * W_out + c * H_out * W_out
    tl.store(
        ptr_output + output_base + (h_flat * W_out + w_flat),
        val,
        mask=tile_mask,
    )


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

    # ptxas on sm < 80 (Turing and earlier) has no native bf16; upcast
    # input/grid to fp32, run the kernel in fp32, cast the output back.
    if dtype == torch.bfloat16 and _device_lacks_bf16_hw(device):
        out_fp32 = grid_sample(
            input.float(), grid.float(), mode, padding_mode, align_corners
        )
        return out_fp32.to(dtype)

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
            if use_tiled:
                # Unified tiled kernel handles all 3 padding modes via constexpr.
                kernel = grid_sample_2d_bicubic_tiled_kernel
            else:
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
            and mode in ["nearest", "bilinear", "bicubic"]
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

            # For bilinear / bicubic, use smaller tiles due to higher memory
            # footprint (4 / 16 corner loads per pixel respectively).
            if mode == "bilinear":
                block_h = max(4, block_h // 2)
                block_w = max(4, block_w // 2)
            elif mode == "bicubic":
                # Bicubic's 4x4 stencil is heaviest; cap tiles at 4x4 to keep
                # register pressure manageable.
                block_h = max(4, block_h // 4)
                block_w = max(4, block_w // 4)

            # Calculate actual grid size
            num_h_blocks = (H_out + block_h - 1) // block_h
            num_w_blocks = (W_out + block_w - 1) // block_w
            grid_size = (N * C, num_h_blocks * num_w_blocks)
        else:
            # Original kernels use 1D grid (for small outputs or very large outputs)
            grid_size = (N * C * H_out * W_out,)

        # Bicubic tiled kernel needs the padding_mode_id constexpr since it
        # handles all 3 padding modes in one body; the older bilinear/nearest
        # tiled kernels are specialized per-padding and take only align_corners.
        is_bicubic_tiled = (
            use_tiled and mode == "bicubic" and output_pixels <= MAX_TILED_PIXELS
        )
        if is_bicubic_tiled:
            _pmode_id = (
                0 if padding_mode == "zeros" else (1 if padding_mode == "border" else 2)
            )
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
                _pmode_id,
            )
        else:
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
                # NOTE: a channel-reuse v2 kernel (grid_sample_3d_nearest_tiled_v2_kernel)
                # exists below but is NOT used by default.  Benchmarks on 2080Ti
                # show that nearest's single-voxel inner loop has too little work
                # to amortize against the lost spatial parallelism (N programs
                # vs. N*C programs), so the v1 per-(n,c) tiled path is faster.
                # The v2 kernel is retained for future tuning / large-C shapes.
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
                # Use channel-reuse v2 kernel: one program per (n, dhw_tile)
                # with a C-loop inside (vs the old per-(n,c) layout that
                # duplicated coord/denorm work C times).
                kernel = grid_sample_3d_trilinear_tiled_v2_kernel
            elif False:  # legacy tiled path retained but disabled (kept for fallback)
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
            if output_voxels < VOXEL_THRESHOLD_SMALL:  # 16³ - 20³
                target_total_blocks = TARGET_BLOCKS_SMALL
                min_blocks_per_nc = MIN_BLOCKS_NC_SMALL
                max_blocks_per_nc = MAX_BLOCKS_NC_SMALL
            elif output_voxels < VOXEL_THRESHOLD_MEDIUM:  # 20³ - 32³
                target_total_blocks = TARGET_BLOCKS_MEDIUM
                min_blocks_per_nc = MIN_BLOCKS_NC_MEDIUM
                max_blocks_per_nc = MAX_BLOCKS_NC_MEDIUM
            elif output_voxels < VOXEL_THRESHOLD_LARGE:  # 32³ - 50³
                target_total_blocks = TARGET_BLOCKS_LARGE
                min_blocks_per_nc = MIN_BLOCKS_NC_LARGE
                max_blocks_per_nc = MAX_BLOCKS_NC_LARGE
            elif output_voxels < VOXEL_THRESHOLD_VERY_LARGE:  # 50³ - 64³
                target_total_blocks = TARGET_BLOCKS_VERY_LARGE
                min_blocks_per_nc = MIN_BLOCKS_NC_VERY_LARGE
                max_blocks_per_nc = MAX_BLOCKS_NC_VERY_LARGE
            else:  # Large outputs (>= 64³)
                target_total_blocks = TARGET_BLOCKS_EXTRA_LARGE
                min_blocks_per_nc = MIN_BLOCKS_NC_EXTRA_LARGE
                max_blocks_per_nc = MAX_BLOCKS_NC_EXTRA_LARGE

            # Channel-aware tiling: reduce targets for high channel counts to avoid too many blocks
            # When C is large, we create too many blocks with the current formula
            # Solution: Reduce target_total_blocks proportionally
            if C > CHANNEL_COUNT_THRESHOLD:
                # Scale down targets more aggressively to avoid excessive blocks when C > threshold
                # Use sqrt scaling for better balance
                channel_scale = (
                    CHANNEL_COUNT_THRESHOLD / C
                ) ** CHANNEL_SCALING_EXPONENT
                target_total_blocks = max(
                    MIN_TARGET_TOTAL_BLOCKS, int(target_total_blocks * channel_scale)
                )
                min_blocks_per_nc = max(
                    MIN_BLOCKS_PER_NC, int(min_blocks_per_nc * channel_scale)
                )
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
                max(
                    MIN_TILE_SIDE,
                    min(MAX_TILE_SIDE, int(target_tile_voxels ** (1.0 / 3.0))),
                )
            )

            # Snap to power-of-2 for better alignment
            # Minimum tile size is 4x4x4 for small outputs, 8x8x8 for large
            if target_tile_side >= LARGE_TILE_THRESHOLD:
                block_d = block_h = block_w = (
                    LARGE_TILE_THRESHOLD
                    if target_tile_side < VERY_LARGE_TILE_THRESHOLD
                    else MAX_TILE_SIDE
                )
            elif target_tile_side >= MEDIUM_TILE_THRESHOLD:
                block_d = block_h = block_w = MEDIUM_TILE_THRESHOLD
            elif target_tile_side >= SMALL_TILE_THRESHOLD:
                block_d = block_h = block_w = SMALL_TILE_THRESHOLD
            else:
                block_d = block_h = block_w = MIN_TILE_SIDE

            # For trilinear, use smaller tiles due to higher memory footprint (8x loads)
            if mode == "bilinear":  # actually trilinear in 5D
                block_d = max(MIN_BLOCK_DIMENSION, block_d // 2)
                block_h = max(MIN_BLOCK_DIMENSION, block_h // 2)
                block_w = max(MIN_BLOCK_DIMENSION, block_w // 2)

            # Calculate actual grid size
            num_d_blocks = (D_out + block_d - 1) // block_d
            num_h_blocks = (H_out + block_h - 1) // block_h
            num_w_blocks = (W_out + block_w - 1) // block_w
            # trilinear v2 uses pid_n (no C), so its grid is (N, dhw_tiles).
            # nearest tiled kernels use pid_nc layout.
            if mode == "bilinear":  # trilinear v2
                grid_size = (N, num_d_blocks * num_h_blocks * num_w_blocks)
            else:
                grid_size = (N * C, num_d_blocks * num_h_blocks * num_w_blocks)
        else:
            # Original kernels use 1D grid (for small outputs or very large outputs)
            grid_size = (N * C * D_out * H_out * W_out,)

        # Only the trilinear v2 kernel handles all 3 padding modes via constexpr;
        # it needs an extra `padding_mode_id` argument.
        is_trilinear_v2 = (
            use_tiled and mode == "bilinear" and output_voxels <= MAX_TILED_VOXELS
        )
        if is_trilinear_v2:
            _pmode_id = (
                0 if padding_mode == "zeros" else (1 if padding_mode == "border" else 2)
            )
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
                _pmode_id,
            )
        else:
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
