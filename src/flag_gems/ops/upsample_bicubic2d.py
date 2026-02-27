import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

device = device.name

logger = logging.getLogger(__name__)


def compute_source_index(scale, dst_index, align_corners, input_size):
    """Compute source index from destination index based on align_corners setting."""
    if align_corners:
        return scale * dst_index
    else:
        return scale * (dst_index + 0.5) - 0.5


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bicubic2d"),
    key=["N", "C", "OH", "OW"],
)
@triton.jit
def upsample_bicubic2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    rheight,
    rwidth,
    align_corners: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """
    Triton kernel for bicubic 2D upsampling.

    Uses a 4x4 neighborhood with cubic interpolation weights.
    """
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)

    # Calculate output coordinates for this block
    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)

    # Create 2D output mask
    mask_out = (oh[:, None] < OH) & (ow[None, :] < OW)

    # Compute source coordinates
    if align_corners:
        # For align_corners=True: real_x = out_x * (IW - 1) / (OW - 1)
        real_w = ow.to(tl.float32) * rwidth
        real_h = oh.to(tl.float32) * rheight
    else:
        # For align_corners=False: real_x = (out_x + 0.5) * IW / OW - 0.5
        real_w = (ow.to(tl.float32) + 0.5) * rwidth - 0.5
        real_h = (oh.to(tl.float32) + 0.5) * rheight - 0.5

    # Get integer indices (floor of real coordinates)
    # For bicubic, we need the pixel at floor(real_coord) and 3 neighbors
    # indices: floor-1, floor, floor+1, floor+2
    iw0 = tl.floor(real_w).to(tl.int32)  # floor
    ih0 = tl.floor(real_h).to(tl.int32)  # floor

    # Fractional part for interpolation weights
    tw = real_w - tl.floor(real_w)  # fractional part in [0, 1)
    th = real_h - tl.floor(real_h)  # fractional part in [0, 1)

    # Compute cubic interpolation weights
    # The cubic kernel: f(x) for x in {-1-frac, -frac, 1-frac, 2-frac}
    # which corresponds to distances from the 4 sample points
    # Using a = -0.75 (PyTorch default for bicubic)
    A = -0.75

    # Weights for x direction (4 weights for the 4 horizontal pixels)
    # Distance from pixel at (iw0-1) is (1 + tw), at iw0 is tw,
    # at (iw0+1) is (1-tw), at (iw0+2) is (2-tw)

    # Weight for pixel at offset -1: distance = 1 + tw
    d0_w = 1.0 + tw
    wx0 = ((A * d0_w - 5.0 * A) * d0_w + 8.0 * A) * d0_w - 4.0 * A

    # Weight for pixel at offset 0: distance = tw (0 <= tw < 1)
    d1_w = tw
    wx1 = ((A + 2.0) * d1_w - (A + 3.0)) * d1_w * d1_w + 1.0

    # Weight for pixel at offset 1: distance = 1 - tw (0 < 1-tw <= 1)
    d2_w = 1.0 - tw
    wx2 = ((A + 2.0) * d2_w - (A + 3.0)) * d2_w * d2_w + 1.0

    # Weight for pixel at offset 2: distance = 2 - tw (1 < 2-tw <= 2)
    d3_w = 2.0 - tw
    wx3 = ((A * d3_w - 5.0 * A) * d3_w + 8.0 * A) * d3_w - 4.0 * A

    # Same for y direction
    d0_h = 1.0 + th
    wy0 = ((A * d0_h - 5.0 * A) * d0_h + 8.0 * A) * d0_h - 4.0 * A

    d1_h = th
    wy1 = ((A + 2.0) * d1_h - (A + 3.0)) * d1_h * d1_h + 1.0

    d2_h = 1.0 - th
    wy2 = ((A + 2.0) * d2_h - (A + 3.0)) * d2_h * d2_h + 1.0

    d3_h = 2.0 - th
    wy3 = ((A * d3_h - 5.0 * A) * d3_h + 8.0 * A) * d3_h - 4.0 * A

    # Clamp source indices to valid range [0, IW-1] and [0, IH-1]
    iw_m1 = tl.maximum(iw0 - 1, 0)
    iw_0 = tl.maximum(tl.minimum(iw0, IW - 1), 0)
    iw_p1 = tl.maximum(tl.minimum(iw0 + 1, IW - 1), 0)
    iw_p2 = tl.minimum(iw0 + 2, IW - 1)

    ih_m1 = tl.maximum(ih0 - 1, 0)
    ih_0 = tl.maximum(tl.minimum(ih0, IH - 1), 0)
    ih_p1 = tl.maximum(tl.minimum(ih0 + 1, IH - 1), 0)
    ih_p2 = tl.minimum(ih0 + 2, IH - 1)

    # Process each batch and channel
    for n in range(0, N, 1):
        for c in range(0, C, 1):
            nc_offset = (n * C + c) * IH

            # Load 4x4 neighborhood and interpolate
            # Row 0 (ih_m1)
            offset_r0 = (nc_offset + ih_m1[:, None]) * IW
            p00 = tl.load(ptr_i + offset_r0 + iw_m1[None, :], mask=mask_out, other=0.0)
            p01 = tl.load(ptr_i + offset_r0 + iw_0[None, :], mask=mask_out, other=0.0)
            p02 = tl.load(ptr_i + offset_r0 + iw_p1[None, :], mask=mask_out, other=0.0)
            p03 = tl.load(ptr_i + offset_r0 + iw_p2[None, :], mask=mask_out, other=0.0)

            # Row 1 (ih_0)
            offset_r1 = (nc_offset + ih_0[:, None]) * IW
            p10 = tl.load(ptr_i + offset_r1 + iw_m1[None, :], mask=mask_out, other=0.0)
            p11 = tl.load(ptr_i + offset_r1 + iw_0[None, :], mask=mask_out, other=0.0)
            p12 = tl.load(ptr_i + offset_r1 + iw_p1[None, :], mask=mask_out, other=0.0)
            p13 = tl.load(ptr_i + offset_r1 + iw_p2[None, :], mask=mask_out, other=0.0)

            # Row 2 (ih_p1)
            offset_r2 = (nc_offset + ih_p1[:, None]) * IW
            p20 = tl.load(ptr_i + offset_r2 + iw_m1[None, :], mask=mask_out, other=0.0)
            p21 = tl.load(ptr_i + offset_r2 + iw_0[None, :], mask=mask_out, other=0.0)
            p22 = tl.load(ptr_i + offset_r2 + iw_p1[None, :], mask=mask_out, other=0.0)
            p23 = tl.load(ptr_i + offset_r2 + iw_p2[None, :], mask=mask_out, other=0.0)

            # Row 3 (ih_p2)
            offset_r3 = (nc_offset + ih_p2[:, None]) * IW
            p30 = tl.load(ptr_i + offset_r3 + iw_m1[None, :], mask=mask_out, other=0.0)
            p31 = tl.load(ptr_i + offset_r3 + iw_0[None, :], mask=mask_out, other=0.0)
            p32 = tl.load(ptr_i + offset_r3 + iw_p1[None, :], mask=mask_out, other=0.0)
            p33 = tl.load(ptr_i + offset_r3 + iw_p2[None, :], mask=mask_out, other=0.0)

            # Interpolate along x for each row
            row0 = p00 * wx0[None, :] + p01 * wx1[None, :] + p02 * wx2[None, :] + p03 * wx3[None, :]
            row1 = p10 * wx0[None, :] + p11 * wx1[None, :] + p12 * wx2[None, :] + p13 * wx3[None, :]
            row2 = p20 * wx0[None, :] + p21 * wx1[None, :] + p22 * wx2[None, :] + p23 * wx3[None, :]
            row3 = p30 * wx0[None, :] + p31 * wx1[None, :] + p32 * wx2[None, :] + p33 * wx3[None, :]

            # Interpolate along y
            result = row0 * wy0[:, None] + row1 * wy1[:, None] + row2 * wy2[:, None] + row3 * wy3[:, None]

            # Store result
            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result, mask=mask_out)


def upsample_bicubic2d(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """
    Performs bicubic interpolation for 2D upsampling.

    Args:
        input: Input tensor of shape (N, C, H, W)
        output_size: Target output size (OH, OW)
        align_corners: If True, align corners of input and output
        scales_h: Optional scale factor for height
        scales_w: Optional scale factor for width

    Returns:
        Upsampled tensor of shape (N, C, OH, OW)
    """
    logger.debug("GEMS UPSAMPLE BICUBIC2D")

    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    N, C, IH, IW = input.shape
    OH, OW = output_size

    # Compute scale factors (reciprocal scale for coordinate mapping)
    if align_corners:
        if OH > 1:
            rheight = (IH - 1) / (OH - 1)
        else:
            rheight = 0.0
        if OW > 1:
            rwidth = (IW - 1) / (OW - 1)
        else:
            rwidth = 0.0
    else:
        rheight = IH / OH
        rwidth = IW / OW

    # Allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(OW, META["BLOCK_X"]),
        triton.cdiv(OH, META["BLOCK_Y"]),
    )

    with torch_device_fn.device(input.device):
        upsample_bicubic2d_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            rheight,
            rwidth,
            align_corners,
        )

    return output
