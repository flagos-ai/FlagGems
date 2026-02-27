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


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bilinear2d"),
    key=["N", "C", "OH", "OW"],
)
@triton.jit
def upsample_bilinear2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    ow = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % OW
    oh = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % OH

    # Compute the source coordinates
    if ALIGN_CORNERS:
        # When align_corners is True, map corners to corners
        # real_h = oh * (IH - 1) / (OH - 1) when OH > 1
        real_h = tl.where(
            OH > 1,
            oh.to(tl.float32) * (IH - 1) / (OH - 1),
            tl.zeros_like(oh).to(tl.float32),
        )
        real_w = tl.where(
            OW > 1,
            ow.to(tl.float32) * (IW - 1) / (OW - 1),
            tl.zeros_like(ow).to(tl.float32),
        )
    else:
        # When align_corners is False, use area-based scaling
        # real_h = (oh + 0.5) * scale_h - 0.5
        real_h = (oh.to(tl.float32) + 0.5) * reciprocal_scale_h - 0.5
        real_w = (ow.to(tl.float32) + 0.5) * reciprocal_scale_w - 0.5

    # Clamp to valid range
    real_h = tl.maximum(real_h, 0.0)
    real_w = tl.maximum(real_w, 0.0)

    # Get integer parts (top-left corner of the 2x2 region)
    h0 = real_h.to(tl.int32)
    w0 = real_w.to(tl.int32)

    # Clamp to ensure we don't go out of bounds
    h0 = tl.minimum(h0, IH - 1)
    w0 = tl.minimum(w0, IW - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    w1 = tl.minimum(w0 + 1, IW - 1)

    # Compute interpolation weights
    h_weight = real_h - h0.to(tl.float32)
    w_weight = real_w - w0.to(tl.float32)

    # Clamp weights to [0, 1]
    h_weight = tl.maximum(tl.minimum(h_weight, 1.0), 0.0)
    w_weight = tl.maximum(tl.minimum(w_weight, 1.0), 0.0)

    # Compute bilinear weights
    w00 = (1.0 - h_weight[:, None]) * (1.0 - w_weight[None, :])
    w01 = (1.0 - h_weight[:, None]) * w_weight[None, :]
    w10 = h_weight[:, None] * (1.0 - w_weight[None, :])
    w11 = h_weight[:, None] * w_weight[None, :]

    # Process all N*C slices
    for n in range(0, N, 1):
        for c in range(0, C, 1):
            # Compute base offset for input
            base_offset = (n * C + c) * IH

            # Load the 4 neighbors
            offset_00 = (base_offset + h0[:, None]) * IW + w0[None, :]
            offset_01 = (base_offset + h0[:, None]) * IW + w1[None, :]
            offset_10 = (base_offset + h1[:, None]) * IW + w0[None, :]
            offset_11 = (base_offset + h1[:, None]) * IW + w1[None, :]

            data_00 = tl.load(ptr_i + offset_00)
            data_01 = tl.load(ptr_i + offset_01)
            data_10 = tl.load(ptr_i + offset_10)
            data_11 = tl.load(ptr_i + offset_11)

            # Compute bilinear interpolation
            result = (
                data_00.to(tl.float32) * w00
                + data_01.to(tl.float32) * w01
                + data_10.to(tl.float32) * w10
                + data_11.to(tl.float32) * w11
            )

            # Store result
            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result)


def bilinear_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0.0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


def upsample_bilinear2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE BILINEAR2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = bilinear_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bilinear_reciprocal_scale(IW, OW, align_corners, scales_w)

    # Allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(OW, META["BLOCK_X"]),
        triton.cdiv(OH, META["BLOCK_Y"]),
    )

    with torch_device_fn.device(input.device):
        upsample_bilinear2d_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            align_corners,
        )
    return output
