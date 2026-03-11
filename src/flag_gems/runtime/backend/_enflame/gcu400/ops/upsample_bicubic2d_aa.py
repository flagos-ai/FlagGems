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
    configs=runtime.get_tuned_config("upsample_bicubic2d_aa"),
    key=["N", "C", "OH", "OW"],
)
@triton.jit
def upsample_bicubic2d_aa_kernel(
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
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    ow = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % OW
    oh = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % OH

    support_w = 2.0
    support_h = 2.0

    # _compute_weights_span
    center_w = (ow + 0.5) * reciprocal_scale_w
    center_h = (oh + 0.5) * reciprocal_scale_h
    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
    span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
    span_size_w = (tl.minimum(center_w + support_w + 0.5, IW) - span_start_w).to(
        tl.int32
    )
    span_size_h = (tl.minimum(center_h + support_h + 0.5, IH) - span_start_h).to(
        tl.int32
    )
    start_minus_center_w = span_start_w - center_w
    start_minus_center_h = span_start_h - center_h
    invscale_w = 1.0
    invscale_h = 1.0
    a = -0.5
    wy0 = tl.abs((0 + start_minus_center_h + 0.5) * invscale_h)
    weight_y0 = tl.where(
        0 < span_size_h,
        tl.where(
            wy0 < 1.0,
            ((a + 2) * wy0 - (a + 3)) * wy0 * wy0 + 1,
            tl.where(wy0 < 2.0, (((wy0 - 5) * wy0 + 8) * wy0 - 4) * a, 0),
        ),
        0,
    )
    wy1 = tl.abs((1 + start_minus_center_h + 0.5) * invscale_h)
    weight_y1 = tl.where(
        1 < span_size_h,
        tl.where(
            wy1 < 1.0,
            ((a + 2) * wy1 - (a + 3)) * wy1 * wy1 + 1,
            tl.where(wy1 < 2.0, (((wy1 - 5) * wy1 + 8) * wy1 - 4) * a, 0),
        ),
        0,
    )
    wy2 = tl.abs((2 + start_minus_center_h + 0.5) * invscale_h)
    weight_y2 = tl.where(
        2 < span_size_h,
        tl.where(
            wy2 < 1.0,
            ((a + 2) * wy2 - (a + 3)) * wy2 * wy2 + 1,
            tl.where(wy2 < 2.0, (((wy2 - 5) * wy2 + 8) * wy2 - 4) * a, 0),
        ),
        0,
    )
    wy3 = tl.abs((3 + start_minus_center_h + 0.5) * invscale_h)
    weight_y3 = tl.where(
        3 < span_size_h,
        tl.where(
            wy3 < 1.0,
            ((a + 2) * wy3 - (a + 3)) * wy3 * wy3 + 1,
            tl.where(wy3 < 2.0, (((wy3 - 5) * wy3 + 8) * wy3 - 4) * a, 0),
        ),
        0,
    )
    wy4 = tl.abs((4 + start_minus_center_h + 0.5) * invscale_h)
    weight_y4 = tl.where(
        4 < span_size_h,
        tl.where(
            wy4 < 1.0,
            ((a + 2) * wy4 - (a + 3)) * wy4 * wy4 + 1,
            tl.where(wy4 < 2.0, (((wy4 - 5) * wy4 + 8) * wy4 - 4) * a, 0),
        ),
        0,
    )
    weight_y_total = weight_y0 + weight_y1 + weight_y2 + weight_y3 + weight_y4
    weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1)
    weight_y0 /= weight_y_total
    weight_y1 /= weight_y_total
    weight_y2 /= weight_y_total
    weight_y3 /= weight_y_total
    weight_y4 /= weight_y_total

    wx0 = tl.abs((0 + start_minus_center_w + 0.5) * invscale_w)
    weight_x0 = tl.where(
        0 < span_size_w,
        tl.where(
            wx0 < 1.0,
            ((a + 2) * wx0 - (a + 3)) * wx0 * wx0 + 1,
            tl.where(wx0 < 2.0, (((wx0 - 5) * wx0 + 8) * wx0 - 4) * a, 0),
        ),
        0,
    )
    wx1 = tl.abs((1 + start_minus_center_w + 0.5) * invscale_w)
    weight_x1 = tl.where(
        1 < span_size_w,
        tl.where(
            wx1 < 1.0,
            ((a + 2) * wx1 - (a + 3)) * wx1 * wx1 + 1,
            tl.where(wx1 < 2.0, (((wx1 - 5) * wx1 + 8) * wx1 - 4) * a, 0),
        ),
        0,
    )
    wx2 = tl.abs((2 + start_minus_center_w + 0.5) * invscale_w)
    weight_x2 = tl.where(
        2 < span_size_w,
        tl.where(
            wx2 < 1.0,
            ((a + 2) * wx2 - (a + 3)) * wx2 * wx2 + 1,
            tl.where(wx2 < 2.0, (((wx2 - 5) * wx2 + 8) * wx2 - 4) * a, 0),
        ),
        0,
    )
    wx3 = tl.abs((3 + start_minus_center_w + 0.5) * invscale_w)
    weight_x3 = tl.where(
        3 < span_size_w,
        tl.where(
            wx3 < 1.0,
            ((a + 2) * wx3 - (a + 3)) * wx3 * wx3 + 1,
            tl.where(wx3 < 2.0, (((wx3 - 5) * wx3 + 8) * wx3 - 4) * a, 0),
        ),
        0,
    )
    wx4 = tl.abs((4 + start_minus_center_w + 0.5) * invscale_w)
    weight_x4 = tl.where(
        4 < span_size_w,
        tl.where(
            wx4 < 1.0,
            ((a + 2) * wx4 - (a + 3)) * wx4 * wx4 + 1,
            tl.where(wx4 < 2.0, (((wx4 - 5) * wx4 + 8) * wx4 - 4) * a, 0),
        ),
        0,
    )
    weight_x_total = weight_x0 + weight_x1 + weight_x2 + weight_x3 + weight_x4
    weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1)
    weight_x0 /= weight_x_total
    weight_x1 /= weight_x_total
    weight_x2 /= weight_x_total
    weight_x3 /= weight_x_total
    weight_x4 /= weight_x_total

    mask_y0 = span_start_h[:, None] + 0 < IH
    mask_y1 = span_start_h[:, None] + 1 < IH
    mask_y2 = span_start_h[:, None] + 2 < IH
    mask_y3 = span_start_h[:, None] + 3 < IH
    mask_y4 = span_start_h[:, None] + 4 < IH
    mask_x0 = span_start_w[None, :] + 0 < IW
    mask_x1 = span_start_w[None, :] + 1 < IW
    mask_x2 = span_start_w[None, :] + 2 < IW
    mask_x3 = span_start_w[None, :] + 3 < IW
    mask_x4 = span_start_w[None, :] + 4 < IW

    for n in range(0, N, 1):
        for c in range(0, C, 1):
            offset_base = (
                (n * C + c) * IH + span_start_h[:, None]
            ) * IW + span_start_w[None, :]

            data00 = tl.load(
                ptr_i + (offset_base + 0 * IW + 0),
                mask=(mask_y0 & mask_x0),
                other=0,
            )
            data01 = tl.load(
                ptr_i + (offset_base + 0 * IW + 1),
                mask=(mask_y0 & mask_x1),
                other=0,
            )
            data02 = tl.load(
                ptr_i + (offset_base + 0 * IW + 2),
                mask=(mask_y0 & mask_x2),
                other=0,
            )
            data03 = tl.load(
                ptr_i + (offset_base + 0 * IW + 3),
                mask=(mask_y0 & mask_x3),
                other=0,
            )
            data04 = tl.load(
                ptr_i + (offset_base + 0 * IW + 4),
                mask=(mask_y0 & mask_x4),
                other=0,
            )

            data10 = tl.load(
                ptr_i + (offset_base + 1 * IW + 0),
                mask=(mask_y1 & mask_x0),
                other=0,
            )
            data11 = tl.load(
                ptr_i + (offset_base + 1 * IW + 1),
                mask=(mask_y1 & mask_x1),
                other=0,
            )
            data12 = tl.load(
                ptr_i + (offset_base + 1 * IW + 2),
                mask=(mask_y1 & mask_x2),
                other=0,
            )
            data13 = tl.load(
                ptr_i + (offset_base + 1 * IW + 3),
                mask=(mask_y1 & mask_x3),
                other=0,
            )
            data14 = tl.load(
                ptr_i + (offset_base + 1 * IW + 4),
                mask=(mask_y1 & mask_x4),
                other=0,
            )

            data20 = tl.load(
                ptr_i + (offset_base + 2 * IW + 0),
                mask=(mask_y2 & mask_x0),
                other=0,
            )
            data21 = tl.load(
                ptr_i + (offset_base + 2 * IW + 1),
                mask=(mask_y2 & mask_x1),
                other=0,
            )
            data22 = tl.load(
                ptr_i + (offset_base + 2 * IW + 2),
                mask=(mask_y2 & mask_x2),
                other=0,
            )
            data23 = tl.load(
                ptr_i + (offset_base + 2 * IW + 3),
                mask=(mask_y2 & mask_x3),
                other=0,
            )
            data24 = tl.load(
                ptr_i + (offset_base + 2 * IW + 4),
                mask=(mask_y2 & mask_x4),
                other=0,
            )

            data30 = tl.load(
                ptr_i + (offset_base + 3 * IW + 0),
                mask=(mask_y3 & mask_x0),
                other=0,
            )
            data31 = tl.load(
                ptr_i + (offset_base + 3 * IW + 1),
                mask=(mask_y3 & mask_x1),
                other=0,
            )
            data32 = tl.load(
                ptr_i + (offset_base + 3 * IW + 2),
                mask=(mask_y3 & mask_x2),
                other=0,
            )
            data33 = tl.load(
                ptr_i + (offset_base + 3 * IW + 3),
                mask=(mask_y3 & mask_x3),
                other=0,
            )
            data34 = tl.load(
                ptr_i + (offset_base + 3 * IW + 4),
                mask=(mask_y3 & mask_x4),
                other=0,
            )

            data40 = tl.load(
                ptr_i + (offset_base + 4 * IW + 0),
                mask=(mask_y4 & mask_x0),
                other=0,
            )
            data41 = tl.load(
                ptr_i + (offset_base + 4 * IW + 1),
                mask=(mask_y4 & mask_x1),
                other=0,
            )
            data42 = tl.load(
                ptr_i + (offset_base + 4 * IW + 2),
                mask=(mask_y4 & mask_x2),
                other=0,
            )
            data43 = tl.load(
                ptr_i + (offset_base + 4 * IW + 3),
                mask=(mask_y4 & mask_x3),
                other=0,
            )
            data44 = tl.load(
                ptr_i + (offset_base + 4 * IW + 4),
                mask=(mask_y4 & mask_x4),
                other=0,
            )

            data0 = (
                data00 * weight_x0[None, :]
                + data01 * weight_x1[None, :]
                + data02 * weight_x2[None, :]
                + data03 * weight_x3[None, :]
                + data04 * weight_x4[None, :]
            )
            data1 = (
                data10 * weight_x0[None, :]
                + data11 * weight_x1[None, :]
                + data12 * weight_x2[None, :]
                + data13 * weight_x3[None, :]
                + data14 * weight_x4[None, :]
            )
            data2 = (
                data20 * weight_x0[None, :]
                + data21 * weight_x1[None, :]
                + data22 * weight_x2[None, :]
                + data23 * weight_x3[None, :]
                + data24 * weight_x4[None, :]
            )
            data3 = (
                data30 * weight_x0[None, :]
                + data31 * weight_x1[None, :]
                + data32 * weight_x2[None, :]
                + data33 * weight_x3[None, :]
                + data34 * weight_x4[None, :]
            )
            data4 = (
                data40 * weight_x0[None, :]
                + data41 * weight_x1[None, :]
                + data42 * weight_x2[None, :]
                + data43 * weight_x3[None, :]
                + data44 * weight_x4[None, :]
            )
            result = (
                data0 * weight_y0[:, None]
                + data1 * weight_y1[:, None]
                + data2 * weight_y2[:, None]
                + data3 * weight_y3[:, None]
                + data4 * weight_y4[:, None]
            )

            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result)


# upsample and downsample
@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bicubic2d_aa"),
    key=["N", "C", "OH", "OW"],
)
@triton.jit
def general_interpolate_bicubic2d_aa_kernel(
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
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    ow = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % OW
    oh = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % OH

    support_w = 2 * reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 2.0
    support_h = 2 * reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 2.0

    interpolate_w = (support_w + 0.5).to(tl.int32) * 2 + 1
    interpolate_h = (support_h + 0.5).to(tl.int32) * 2 + 1

    # _compute_weights_span
    center_w = (ow + 0.5) * reciprocal_scale_w
    center_h = (oh + 0.5) * reciprocal_scale_h
    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
    span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
    span_size_w = (tl.minimum(center_w + support_w + 0.5, IW) - span_start_w).to(
        tl.int32
    )
    span_size_h = (tl.minimum(center_h + support_h + 0.5, IH) - span_start_h).to(
        tl.int32
    )

    invscale_w = 1.0 / reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 1.0
    invscale_h = 1.0 / reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 1.0
    start_minus_center_w = span_start_w - center_w
    start_minus_center_h = span_start_h - center_h

    a = -0.5
    for n in range(0, N, 1):
        for c in range(0, C, 1):
            offset_base = ((n * C + c) * IH + span_start_h[:, None]) * IW + span_start_w
            weight_y_total = tl.zeros((BLOCK_Y,), dtype=tl.float32)
            result = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
            for y in range(0, interpolate_h, 1):
                wy = tl.abs((y + start_minus_center_h + 0.5) * invscale_h)
                weight_y = tl.where(
                    y < span_size_h,
                    tl.where(
                        wy < 1.0,
                        ((a + 2) * wy - (a + 3)) * wy * wy + 1,
                        tl.where(wy < 2.0, (((wy - 5) * wy + 8) * wy - 4) * a, 0),
                    ),
                    0,
                )
                weight_y_total += weight_y
                weight_x_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
                buffer = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
                for x in range(0, interpolate_w, 1):
                    wx = tl.abs((x + start_minus_center_w + 0.5) * invscale_w)
                    weight_x = tl.where(
                        x < span_size_w,
                        tl.where(
                            wx < 1.0,
                            ((a + 2) * wx - (a + 3)) * wx * wx + 1,
                            tl.where(wx < 2.0, (((wx - 5) * wx + 8) * wx - 4) * a, 0),
                        ),
                        0,
                    )
                    weight_x_total += weight_x
                    data = tl.load(
                        ptr_i + (offset_base + y * IW + x),
                        mask=(span_start_h[:, None] + y < IH)
                        & (span_start_w[None, :] + x < IW),
                        other=0,
                    )
                    buffer += data * weight_x[None, :]
                weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1)
                result += buffer / weight_x_total[None, :] * weight_y[:, None]
            weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1)
            result /= weight_y_total[:, None]
            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result)


def bicubic_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L12547
def _upsample_bicubic2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS UPSAMPLE BICUBIC2D AA")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = bicubic_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bicubic_reciprocal_scale(IW, OW, align_corners, scales_w)

    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    grid = lambda META: (
        triton.cdiv(OW, META["BLOCK_X"]),
        triton.cdiv(OH, META["BLOCK_Y"]),
    )
    kernel = (
        general_interpolate_bicubic2d_aa_kernel
        if (reciprocal_scale_w >= 1.0) or (reciprocal_scale_h >= 1.0)
        else upsample_bicubic2d_aa_kernel
    )
    with torch_device_fn.device(input.device):
        kernel[grid](
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
        )
    return output
