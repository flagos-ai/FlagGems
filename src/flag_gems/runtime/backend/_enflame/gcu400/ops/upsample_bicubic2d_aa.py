import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name

logger = logging.getLogger(__name__)


@triton.jit(
    do_not_specialize=[
        "NC",
        "OH",
        "OW",
        "IH",
        "IW",
        "reciprocal_scale_h",
        "reciprocal_scale_w",
        "total_rows",
    ]
)
def upsample_bicubic2d_aa_kernel_1d(
    ptr_o,
    ptr_i,
    NC,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    total_rows,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    a = -0.5

    for row_id in tl.range(pid, total_rows, num_programs):
        nc = row_id // OH
        oh = row_id - nc * OH

        center_h = (oh + 0.5) * reciprocal_scale_h
        span_start_h = tl.maximum(center_h - 2.0 + 0.5, 0.0).to(tl.int32)
        span_size_h = (tl.minimum(center_h + 2.0 + 0.5, IH) - span_start_h).to(tl.int32)
        smch = span_start_h - center_h

        wy0 = tl.abs(0 + smch + 0.5)
        weight_y0 = tl.where(
            0 < span_size_h,
            tl.where(
                wy0 < 1.0,
                ((a + 2) * wy0 - (a + 3)) * wy0 * wy0 + 1,
                tl.where(
                    wy0 < 2.0,
                    (((wy0 - 5) * wy0 + 8) * wy0 - 4) * a,
                    0.0,
                ),
            ),
            0.0,
        )
        wy1 = tl.abs(1 + smch + 0.5)
        weight_y1 = tl.where(
            1 < span_size_h,
            tl.where(
                wy1 < 1.0,
                ((a + 2) * wy1 - (a + 3)) * wy1 * wy1 + 1,
                tl.where(
                    wy1 < 2.0,
                    (((wy1 - 5) * wy1 + 8) * wy1 - 4) * a,
                    0.0,
                ),
            ),
            0.0,
        )
        wy2 = tl.abs(2 + smch + 0.5)
        weight_y2 = tl.where(
            2 < span_size_h,
            tl.where(
                wy2 < 1.0,
                ((a + 2) * wy2 - (a + 3)) * wy2 * wy2 + 1,
                tl.where(
                    wy2 < 2.0,
                    (((wy2 - 5) * wy2 + 8) * wy2 - 4) * a,
                    0.0,
                ),
            ),
            0.0,
        )
        wy3 = tl.abs(3 + smch + 0.5)
        weight_y3 = tl.where(
            3 < span_size_h,
            tl.where(
                wy3 < 1.0,
                ((a + 2) * wy3 - (a + 3)) * wy3 * wy3 + 1,
                tl.where(
                    wy3 < 2.0,
                    (((wy3 - 5) * wy3 + 8) * wy3 - 4) * a,
                    0.0,
                ),
            ),
            0.0,
        )
        wy4 = tl.abs(4 + smch + 0.5)
        weight_y4 = tl.where(
            4 < span_size_h,
            tl.where(
                wy4 < 1.0,
                ((a + 2) * wy4 - (a + 3)) * wy4 * wy4 + 1,
                tl.where(
                    wy4 < 2.0,
                    (((wy4 - 5) * wy4 + 8) * wy4 - 4) * a,
                    0.0,
                ),
            ),
            0.0,
        )

        wyt = weight_y0 + weight_y1 + weight_y2 + weight_y3 + weight_y4
        wyt = tl.where(wyt != 0, wyt, 1.0)
        weight_y0 /= wyt
        weight_y1 /= wyt
        weight_y2 /= wyt
        weight_y3 /= wyt
        weight_y4 /= wyt

        base_in = nc * IH * IW
        base_out = nc * OH * OW + oh * OW

        for col_start in tl.range(0, OW, BLOCK_X):
            ow = col_start + tl.arange(0, BLOCK_X)
            ow_mask = ow < OW

            center_w = (ow + 0.5) * reciprocal_scale_w
            span_start_w = tl.maximum(center_w - 2.0 + 0.5, 0.0).to(tl.int32)
            span_size_w = (tl.minimum(center_w + 2.0 + 0.5, IW) - span_start_w).to(
                tl.int32
            )
            smcw = span_start_w - center_w

            wx0 = tl.abs(0 + smcw + 0.5)
            weight_x0 = tl.where(
                0 < span_size_w,
                tl.where(
                    wx0 < 1.0,
                    ((a + 2) * wx0 - (a + 3)) * wx0 * wx0 + 1,
                    tl.where(
                        wx0 < 2.0,
                        (((wx0 - 5) * wx0 + 8) * wx0 - 4) * a,
                        0.0,
                    ),
                ),
                0.0,
            )
            wx1 = tl.abs(1 + smcw + 0.5)
            weight_x1 = tl.where(
                1 < span_size_w,
                tl.where(
                    wx1 < 1.0,
                    ((a + 2) * wx1 - (a + 3)) * wx1 * wx1 + 1,
                    tl.where(
                        wx1 < 2.0,
                        (((wx1 - 5) * wx1 + 8) * wx1 - 4) * a,
                        0.0,
                    ),
                ),
                0.0,
            )
            wx2 = tl.abs(2 + smcw + 0.5)
            weight_x2 = tl.where(
                2 < span_size_w,
                tl.where(
                    wx2 < 1.0,
                    ((a + 2) * wx2 - (a + 3)) * wx2 * wx2 + 1,
                    tl.where(
                        wx2 < 2.0,
                        (((wx2 - 5) * wx2 + 8) * wx2 - 4) * a,
                        0.0,
                    ),
                ),
                0.0,
            )
            wx3 = tl.abs(3 + smcw + 0.5)
            weight_x3 = tl.where(
                3 < span_size_w,
                tl.where(
                    wx3 < 1.0,
                    ((a + 2) * wx3 - (a + 3)) * wx3 * wx3 + 1,
                    tl.where(
                        wx3 < 2.0,
                        (((wx3 - 5) * wx3 + 8) * wx3 - 4) * a,
                        0.0,
                    ),
                ),
                0.0,
            )
            wx4 = tl.abs(4 + smcw + 0.5)
            weight_x4 = tl.where(
                4 < span_size_w,
                tl.where(
                    wx4 < 1.0,
                    ((a + 2) * wx4 - (a + 3)) * wx4 * wx4 + 1,
                    tl.where(
                        wx4 < 2.0,
                        (((wx4 - 5) * wx4 + 8) * wx4 - 4) * a,
                        0.0,
                    ),
                ),
                0.0,
            )

            wxt = weight_x0 + weight_x1 + weight_x2 + weight_x3 + weight_x4
            wxt = tl.where(wxt != 0, wxt, 1.0)
            weight_x0 /= wxt
            weight_x1 /= wxt
            weight_x2 /= wxt
            weight_x3 /= wxt
            weight_x4 /= wxt

            result = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for row in tl.static_range(5):
                iy = span_start_h + row
                if iy < IH:
                    row_addr = base_in + iy * IW
                    d0 = tl.load(
                        ptr_i + row_addr + span_start_w + 0,
                        mask=ow_mask & (span_start_w + 0 < IW),
                        other=0,
                    )
                    d1 = tl.load(
                        ptr_i + row_addr + span_start_w + 1,
                        mask=ow_mask & (span_start_w + 1 < IW),
                        other=0,
                    )
                    d2 = tl.load(
                        ptr_i + row_addr + span_start_w + 2,
                        mask=ow_mask & (span_start_w + 2 < IW),
                        other=0,
                    )
                    d3 = tl.load(
                        ptr_i + row_addr + span_start_w + 3,
                        mask=ow_mask & (span_start_w + 3 < IW),
                        other=0,
                    )
                    d4 = tl.load(
                        ptr_i + row_addr + span_start_w + 4,
                        mask=ow_mask & (span_start_w + 4 < IW),
                        other=0,
                    )
                    row_val = (
                        d0 * weight_x0
                        + d1 * weight_x1
                        + d2 * weight_x2
                        + d3 * weight_x3
                        + d4 * weight_x4
                    )
                else:
                    row_val = tl.zeros((BLOCK_X,), dtype=tl.float32)

                if row == 0:
                    result += row_val * weight_y0
                elif row == 1:
                    result += row_val * weight_y1
                elif row == 2:
                    result += row_val * weight_y2
                elif row == 3:
                    result += row_val * weight_y3
                elif row == 4:
                    result += row_val * weight_y4

            tl.store(
                ptr_o + base_out + ow, result.to(ptr_o.dtype.element_ty), mask=ow_mask
            )


@triton.jit(
    do_not_specialize=[
        "NC",
        "OH",
        "OW",
        "IH",
        "IW",
        "reciprocal_scale_h",
        "reciprocal_scale_w",
        "total_rows",
    ]
)
def general_interpolate_bicubic2d_aa_kernel_1d(
    ptr_o,
    ptr_i,
    NC,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    total_rows,
    BLOCK_X: tl.constexpr,
    INTERP_H: tl.constexpr,
    INTERP_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    support_w = 2 * reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 2.0
    support_h = 2 * reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 2.0
    invscale_w = 1.0 / reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 1.0
    invscale_h = 1.0 / reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 1.0
    a = -0.5

    for row_id in tl.range(pid, total_rows, num_programs):
        nc = row_id // OH
        oh = row_id - nc * OH

        center_h = (oh + 0.5) * reciprocal_scale_h
        span_start_h = tl.maximum(center_h - support_h + 0.5, 0.0).to(tl.int32)
        span_size_h = (tl.minimum(center_h + support_h + 0.5, IH) - span_start_h).to(
            tl.int32
        )
        smch = span_start_h - center_h

        base_in = nc * IH * IW
        base_out = nc * OH * OW + oh * OW

        for col_start in tl.range(0, OW, BLOCK_X):
            ow = col_start + tl.arange(0, BLOCK_X)
            ow_mask = ow < OW

            center_w = (ow + 0.5) * reciprocal_scale_w
            span_start_w = tl.maximum(center_w - support_w + 0.5, 0.0).to(tl.int32)
            span_size_w = (
                tl.minimum(center_w + support_w + 0.5, IW) - span_start_w
            ).to(tl.int32)
            smcw = span_start_w - center_w

            weight_y_total = 0.0
            result = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for y in range(0, INTERP_H, 1):
                wy = tl.abs((y + smch + 0.5) * invscale_h)
                weight_y = tl.where(
                    y < span_size_h,
                    tl.where(
                        wy < 1.0,
                        ((a + 2) * wy - (a + 3)) * wy * wy + 1,
                        tl.where(
                            wy < 2.0,
                            (((wy - 5) * wy + 8) * wy - 4) * a,
                            0.0,
                        ),
                    ),
                    0.0,
                )
                weight_y_total += weight_y

                weight_x_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
                buffer = tl.zeros((BLOCK_X,), dtype=tl.float32)
                iy = span_start_h + y
                for x in range(0, INTERP_W, 1):
                    wx = tl.abs((x + smcw + 0.5) * invscale_w)
                    weight_x = tl.where(
                        x < span_size_w,
                        tl.where(
                            wx < 1.0,
                            ((a + 2) * wx - (a + 3)) * wx * wx + 1,
                            tl.where(
                                wx < 2.0,
                                (((wx - 5) * wx + 8) * wx - 4) * a,
                                0.0,
                            ),
                        ),
                        0.0,
                    )
                    weight_x_total += weight_x
                    data = tl.load(
                        ptr_i + base_in + iy * IW + span_start_w + x,
                        mask=ow_mask & (iy < IH) & (span_start_w + x < IW),
                        other=0,
                    )
                    buffer += data * weight_x
                weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1.0)
                result += buffer / weight_x_total * weight_y

            weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1.0)
            result /= weight_y_total
            tl.store(
                ptr_o + base_out + ow, result.to(ptr_o.dtype.element_ty), mask=ow_mask
            )


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
    NC = N * C
    total_rows = NC * OH

    reciprocal_scale_h = bicubic_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bicubic_reciprocal_scale(IW, OW, align_corners, scales_w)

    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    BLOCK_X = triton.next_power_of_2(OW)
    if BLOCK_X > 2048:
        BLOCK_X = 2048
    num_programs = min(total_rows, 24 * 2)
    grid = (num_programs,)

    if (reciprocal_scale_w >= 1.0) or (reciprocal_scale_h >= 1.0):
        support_w = 2 * reciprocal_scale_w if reciprocal_scale_w >= 1.0 else 2.0
        support_h = 2 * reciprocal_scale_h if reciprocal_scale_h >= 1.0 else 2.0
        INTERP_W = int(support_w + 0.5) * 2 + 1
        INTERP_H = int(support_h + 0.5) * 2 + 1
        with torch_device_fn.device(input.device):
            general_interpolate_bicubic2d_aa_kernel_1d[grid](
                output,
                input,
                NC,
                OH,
                OW,
                IH,
                IW,
                reciprocal_scale_h,
                reciprocal_scale_w,
                total_rows,
                BLOCK_X=BLOCK_X,
                INTERP_H=INTERP_H,
                INTERP_W=INTERP_W,
                num_warps=1,
            )
    else:
        with torch_device_fn.device(input.device):
            upsample_bicubic2d_aa_kernel_1d[grid](
                output,
                input,
                NC,
                OH,
                OW,
                IH,
                IW,
                reciprocal_scale_h,
                reciprocal_scale_w,
                total_rows,
                BLOCK_X=BLOCK_X,
                num_warps=1,
            )
    return output
