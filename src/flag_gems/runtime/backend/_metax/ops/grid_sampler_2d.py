import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("grid_sampler_2d"),
    key=["N", "C", "OH", "OW"],
)
@triton.jit
def grid_sampler_2d_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one output pixel
    pid = tle.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Compute output indices
    ow = idx % OW
    oh = idx // OW % OH
    c = idx // OW // OH % C
    n = idx // OW // OH // C % N

    mask = idx < (N * C * OH * OW)

    # Load grid coordinates (x, y)
    grid_offset = ((n * OH + oh) * OW + ow) * 2
    gx = tl.load(grid_ptr + grid_offset, mask=mask)
    gy = tl.load(grid_ptr + grid_offset + 1, mask=mask)

    # Normalize coordinates based on align_corners
    if align_corners:
        # Scale to [0, IH-1] and [0, IW-1]
        x = (gx + 1) * (IW - 1) / 2
        y = (gy + 1) * (IH - 1) / 2
    else:
        # Scale to [0, IH) and [0, IW)
        x = (gx + 1) * IW / 2
        y = (gy + 1) * IH / 2

    if interpolation_mode == 0:  # Bilinear
        # Compute floor and frac
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute weights
        x0_f = x0.to(tl.float32)
        y0_f = y0.to(tl.float32)
        wx1 = x - x0_f
        wy1 = y - y0_f
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        # Clamp coordinates based on padding_mode
        if padding_mode == 0:  # Zeros
            x0_clamped = x0
            y0_clamped = y0
            x1_clamped = x1
            y1_clamped = y1
            # Check bounds
            x0_valid = (x0 >= 0) & (x0 < IW)
            y0_valid = (y0 >= 0) & (y0 < IH)
            x1_valid = (x1 >= 0) & (x1 < IW)
            y1_valid = (y1 >= 0) & (y1 < IH)
        elif padding_mode == 1:  # Border
            x0_clamped = tl.minimum(tl.maximum(x0, 0), IW - 1)
            y0_clamped = tl.minimum(tl.maximum(y0, 0), IH - 1)
            x1_clamped = tl.minimum(tl.maximum(x1, 0), IW - 1)
            y1_clamped = tl.minimum(tl.maximum(y1, 0), IH - 1)
            x0_valid = True
            y0_valid = True
            x1_valid = True
            y1_valid = True
        else:  # Reflection
            x0_ref = tl.minimum(x0, 2 * IW - 2 - x0)
            x0_ref = tl.maximum(x0_ref, -x0_ref)
            y0_ref = tl.minimum(y0, 2 * IH - 2 - y0)
            y0_ref = tl.maximum(y0_ref, -y0_ref)
            x1_ref = tl.minimum(x1, 2 * IW - 2 - x1)
            x1_ref = tl.maximum(x1_ref, -x1_ref)
            y1_ref = tl.minimum(y1, 2 * IH - 2 - y1)
            y1_ref = tl.maximum(y1_ref, -y1_ref)

            x0_clamped = tl.minimum(tl.maximum(x0_ref, 0), IW - 1)
            y0_clamped = tl.minimum(tl.maximum(y0_ref, 0), IH - 1)
            x1_clamped = tl.minimum(tl.maximum(x1_ref, 0), IW - 1)
            y1_clamped = tl.minimum(tl.maximum(y1_ref, 0), IH - 1)
            x0_valid = True
            y0_valid = True
            x1_valid = True
            y1_valid = True

        # Inline pixel loads
        v00 = tl.load(
            input_ptr + ((n * C + c) * IH + y0_clamped) * IW + x0_clamped,
            mask=mask & x0_valid & y0_valid,
            other=0.0,
        )
        v01 = tl.load(
            input_ptr + ((n * C + c) * IH + y0_clamped) * IW + x1_clamped,
            mask=mask & x1_valid & y0_valid,
            other=0.0,
        )
        v10 = tl.load(
            input_ptr + ((n * C + c) * IH + y1_clamped) * IW + x0_clamped,
            mask=mask & x0_valid & y1_valid,
            other=0.0,
        )
        v11 = tl.load(
            input_ptr + ((n * C + c) * IH + y1_clamped) * IW + x1_clamped,
            mask=mask & x1_valid & y1_valid,
            other=0.0,
        )

        # Bilinear interpolation
        if padding_mode == 0:  # Zeros - need to handle invalid pixels specially
            # Use valid masks
            v00 = tl.where(x0_valid & y0_valid, v00, 0.0)
            v01 = tl.where(x1_valid & y0_valid, v01, 0.0)
            v10 = tl.where(x0_valid & y1_valid, v10, 0.0)
            v11 = tl.where(x1_valid & y1_valid, v11, 0.0)

            # Compute final value
            result = (
                v00 * wx0 * wy0 + v01 * wx1 * wy0 + v10 * wx0 * wy1 + v11 * wx1 * wy1
            )
        else:
            # Border or Reflection - all coordinates are valid
            result = (
                v00 * wx0 * wy0 + v01 * wx1 * wy0 + v10 * wx0 * wy1 + v11 * wx1 * wy1
            )

    elif interpolation_mode == 1:  # Nearest
        # Round to nearest integer using floor(x + 0.5)
        x_nearest = tl.floor(x + 0.5).to(tl.int32)
        y_nearest = tl.floor(y + 0.5).to(tl.int32)

        if padding_mode == 0:  # Zeros — must zero out-of-bounds, not clamp
            x_valid = (x_nearest >= 0) & (x_nearest < IW)
            y_valid = (y_nearest >= 0) & (y_nearest < IH)
            x_nearest = tl.minimum(tl.maximum(x_nearest, 0), IW - 1)
            y_nearest = tl.minimum(tl.maximum(y_nearest, 0), IH - 1)
        elif padding_mode == 1:  # Border
            x_nearest = tl.minimum(tl.maximum(x_nearest, 0), IW - 1)
            y_nearest = tl.minimum(tl.maximum(y_nearest, 0), IH - 1)
        else:  # Reflection
            x_ref = tl.minimum(x_nearest, 2 * IW - 2 - x_nearest)
            x_ref = tl.maximum(x_ref, -x_ref)
            y_ref = tl.minimum(y_nearest, 2 * IH - 2 - y_nearest)
            y_ref = tl.maximum(y_ref, -y_ref)
            x_nearest = tl.minimum(tl.maximum(x_ref, 0), IW - 1)
            y_nearest = tl.minimum(tl.maximum(y_ref, 0), IH - 1)

        offset = ((n * C + c) * IH + y_nearest) * IW + x_nearest
        result = tl.load(input_ptr + offset, mask=mask)
        if padding_mode == 0:
            result = tl.where(x_valid & y_valid, result, 0.0)

    else:  # Bicubic - fallback to bilinear for now
        # Compute floor and frac
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute weights
        x0_f = x0.to(tl.float32)
        y0_f = y0.to(tl.float32)
        wx1 = x - x0_f
        wy1 = y - y0_f
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        # Clamp
        x0_clamped = tl.minimum(tl.maximum(x0, 0), IW - 1)
        y0_clamped = tl.minimum(tl.maximum(y0, 0), IH - 1)
        x1_clamped = tl.minimum(tl.maximum(x1, 0), IW - 1)
        y1_clamped = tl.minimum(tl.maximum(y1, 0), IH - 1)

        # Load and interpolate
        v00 = tl.load(
            input_ptr + ((n * C + c) * IH + y0_clamped) * IW + x0_clamped, mask=mask
        )
        v01 = tl.load(
            input_ptr + ((n * C + c) * IH + y0_clamped) * IW + x1_clamped, mask=mask
        )
        v10 = tl.load(
            input_ptr + ((n * C + c) * IH + y1_clamped) * IW + x0_clamped, mask=mask
        )
        v11 = tl.load(
            input_ptr + ((n * C + c) * IH + y1_clamped) * IW + x1_clamped, mask=mask
        )

        result = v00 * wx0 * wy0 + v01 * wx1 * wy0 + v10 * wx0 * wy1 + v11 * wx1 * wy1

    # Store result
    out_offset = ((n * C + c) * OH + oh) * OW + ow
    tl.store(output_ptr + out_offset, result, mask=mask)


def grid_sampler_2d(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
):
    """
    Performs bilinear interpolation of input tensor using the given grid.

    Args:
        input: 4D tensor of shape (N, C, IH, IW)
        grid: 4D tensor of shape (N, OH, OW, 2) containing normalized coordinates
        interpolation_mode: 0=Bilinear, 1=Nearest, 2=Bicubic
        padding_mode: 0=Zeros, 1=Border, 2=Reflection
        align_corners: if True, corner pixels are aligned
    """
    logger.debug("METAX GEMS GRID_SAMPLER_2D")

    assert input.ndim == 4, "Input must be 4D"
    assert grid.ndim == 4, "Grid must be 4D"
    assert grid.shape[-1] == 2, "Grid must have last dimension of size 2"

    N, C, IH, IW = input.shape
    OH, OW = grid.shape[1:3]

    # Validate input
    assert grid.shape[0] == N, "Batch size mismatch"

    # Allocate output
    output = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    if output.numel() == 0:
        return output

    # Make inputs contiguous
    input = input.contiguous()
    grid = grid.contiguous()

    # Launch kernel
    total_threads = N * C * OH * OW
    grid_fn = lambda meta: (triton.cdiv(total_threads, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        grid_sampler_2d_kernel[grid_fn](
            output,
            input,
            grid,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            interpolation_mode,
            padding_mode,
            align_corners,
            # BLOCK_SIZE=1024 balances occupancy and register pressure for typical grid sizes
            BLOCK_SIZE=1024,
        )

    return output
