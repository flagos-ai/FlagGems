# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _cubic_weight(x):
    a = -0.5
    ax = tl.abs(x)
    ax2 = ax * ax
    w0 = ((a + 2.0) * ax - (a + 3.0)) * ax2 + 1.0
    w1 = ((ax - 5.0) * ax + 8.0) * ax - 4.0
    w1 *= a
    return tl.where(ax < 1.0, w0, tl.where(ax < 2.0, w1, 0.0))


@triton.jit
def _upsample_bicubic2d_aa_horizontal_kernel(
    temp_ptr,
    in_ptr,
    total_rows,
    OW,
    IW,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    MAX_INTERP_W: tl.constexpr,
):
    pid_x = tl.program_id(0)
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = ow < OW

    support_w = tl.where(reciprocal_scale_w >= 1.0, 2.0 * reciprocal_scale_w, 2.0)
    invscale_w = tl.where(reciprocal_scale_w >= 1.0, 1.0 / reciprocal_scale_w, 1.0)
    center_w = (ow.to(tl.float32) + 0.5) * reciprocal_scale_w
    span_start_w = tl.floor(tl.maximum(center_w - support_w + 0.5, 0.0)).to(tl.int32)
    span_size_w = tl.floor(
        tl.minimum(center_w + support_w + 0.5, IW) - span_start_w
    ).to(tl.int32)
    start_minus_center_w = span_start_w.to(tl.float32) - center_w

    for row_id in range(row_start, total_rows, row_step):
        acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
        weight_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
        for x in tl.static_range(0, MAX_INTERP_W):
            weight = _cubic_weight((x + start_minus_center_w + 0.5) * invscale_w)
            weight = tl.where(x < span_size_w, weight, 0.0)
            iw = tl.maximum(0, tl.minimum(span_start_w + x, IW - 1))
            data = tl.load(in_ptr + row_id * IW + iw, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += data * weight
            weight_total += weight
        weight_total = tl.where(weight_total != 0.0, weight_total, 1.0)
        tl.store(temp_ptr + row_id * OW + ow, acc / weight_total, mask=mask)


@triton.jit
def _upsample_bicubic2d_aa_vertical_kernel(
    out_ptr,
    temp_ptr,
    total_rows,
    C,
    OH,
    OW,
    IH,
    reciprocal_scale_h,
    BLOCK_X: tl.constexpr,
    MAX_INTERP_H: tl.constexpr,
):
    pid_x = tl.program_id(0)
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = ow < OW

    for row_id in range(row_start, total_rows, row_step):
        pid_nc = row_id // OH
        n = pid_nc // C
        c = pid_nc - n * C
        oh = row_id - pid_nc * OH

        support_h = tl.where(reciprocal_scale_h >= 1.0, 2.0 * reciprocal_scale_h, 2.0)
        invscale_h = tl.where(reciprocal_scale_h >= 1.0, 1.0 / reciprocal_scale_h, 1.0)
        center_h = (oh.to(tl.float32) + 0.5) * reciprocal_scale_h
        span_start_h = tl.floor(tl.maximum(center_h - support_h + 0.5, 0.0)).to(
            tl.int32
        )
        span_size_h = tl.floor(
            tl.minimum(center_h + support_h + 0.5, IH) - span_start_h
        ).to(tl.int32)
        start_minus_center_h = span_start_h.to(tl.float32) - center_h

        acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
        weight_total = tl.full((), 0.0, dtype=tl.float32)
        temp_base = (n * C + c) * IH * OW
        for y in tl.static_range(0, MAX_INTERP_H):
            weight = _cubic_weight((y + start_minus_center_h + 0.5) * invscale_h)
            weight = tl.where(y < span_size_h, weight, 0.0)
            ih = tl.maximum(0, tl.minimum(span_start_h + y, IH - 1))
            data = tl.load(temp_ptr + temp_base + ih * OW + ow, mask=mask, other=0.0)
            acc += data * weight
            weight_total += weight
        weight_total = tl.where(weight_total != 0.0, weight_total, 1.0)
        out_base = (n * C + c) * OH * OW
        tl.store(
            out_ptr + out_base + oh * OW + ow,
            acc / weight_total,
            mask=mask,
        )


@triton.jit
def _upsample_bicubic2d_aa_horizontal_pair_kernel(
    temp_ptr,
    in_ptr,
    total_rows,
    OW,
    IW,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    ow0 = pair * 2
    ow1 = ow0 + 1
    mask0 = ow0 < OW
    mask1 = ow1 < OW

    center0 = (ow0.to(tl.float32) + 0.5) * reciprocal_scale_w
    center1 = (ow1.to(tl.float32) + 0.5) * reciprocal_scale_w
    start0 = tl.floor(tl.maximum(center0 - 1.5, 0.0)).to(tl.int32)
    start1 = tl.floor(tl.maximum(center1 - 1.5, 0.0)).to(tl.int32)
    size0 = tl.floor(tl.minimum(center0 + 2.5, IW) - start0).to(tl.int32)
    size1 = tl.floor(tl.minimum(center1 + 2.5, IW) - start1).to(tl.int32)
    union_start = tl.minimum(start0, start1)

    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    for row in range(row_start, total_rows, row_step):
        acc0 = tl.zeros((BLOCK_X,), tl.float32)
        acc1 = tl.zeros((BLOCK_X,), tl.float32)
        total0 = tl.zeros((BLOCK_X,), tl.float32)
        total1 = tl.zeros((BLOCK_X,), tl.float32)
        for k in tl.static_range(0, 5):
            iw = union_start + k
            valid_iw = (iw >= 0) & (iw < IW)
            safe_iw = tl.maximum(0, tl.minimum(iw, IW - 1))
            data = tl.load(
                in_ptr + row * IW + safe_iw,
                mask=valid_iw & (mask0 | mask1),
                other=0.0,
            ).to(tl.float32)
            weight0 = _cubic_weight(iw.to(tl.float32) - center0 + 0.5)
            weight1 = _cubic_weight(iw.to(tl.float32) - center1 + 0.5)
            weight0 = tl.where(
                mask0 & (iw >= start0) & (iw < start0 + size0), weight0, 0.0
            )
            weight1 = tl.where(
                mask1 & (iw >= start1) & (iw < start1 + size1), weight1, 0.0
            )
            acc0 += data * weight0
            acc1 += data * weight1
            total0 += weight0
            total1 += weight1
        total0 = tl.where(total0 != 0.0, total0, 1.0)
        total1 = tl.where(total1 != 0.0, total1, 1.0)
        tl.store(temp_ptr + row * OW + ow0, acc0 / total0, mask=mask0)
        tl.store(temp_ptr + row * OW + ow1, acc1 / total1, mask=mask1)


@triton.jit
def _upsample_bicubic2d_aa_vertical_pair_kernel(
    out_ptr,
    temp_ptr,
    total_pair_rows,
    OH,
    OW,
    IH,
    reciprocal_scale_h,
    BLOCK_X: tl.constexpr,
):
    ow = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    width_mask = ow < OW
    pair_count = tl.cdiv(OH, 2)
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    for row in range(row_start, total_pair_rows, row_step):
        pair = row % pair_count
        nc = row // pair_count
        oh0 = pair * 2
        oh1 = oh0 + 1
        mask0 = oh0 < OH
        mask1 = oh1 < OH
        center0 = (oh0.to(tl.float32) + 0.5) * reciprocal_scale_h
        center1 = (oh1.to(tl.float32) + 0.5) * reciprocal_scale_h
        start0 = tl.floor(tl.maximum(center0 - 1.5, 0.0)).to(tl.int32)
        start1 = tl.floor(tl.maximum(center1 - 1.5, 0.0)).to(tl.int32)
        size0 = tl.floor(tl.minimum(center0 + 2.5, IH) - start0).to(tl.int32)
        size1 = tl.floor(tl.minimum(center1 + 2.5, IH) - start1).to(tl.int32)
        union_start = tl.minimum(start0, start1)

        acc0 = tl.zeros((BLOCK_X,), tl.float32)
        acc1 = tl.zeros((BLOCK_X,), tl.float32)
        total0 = tl.full((), 0.0, tl.float32)
        total1 = tl.full((), 0.0, tl.float32)
        for k in tl.static_range(0, 5):
            ih = union_start + k
            valid_ih = (ih >= 0) & (ih < IH)
            safe_ih = tl.maximum(0, tl.minimum(ih, IH - 1))
            data = tl.load(
                temp_ptr + (nc * IH + safe_ih) * OW + ow,
                mask=valid_ih & width_mask,
                other=0.0,
            ).to(tl.float32)
            weight0 = _cubic_weight(ih.to(tl.float32) - center0 + 0.5)
            weight1 = _cubic_weight(ih.to(tl.float32) - center1 + 0.5)
            weight0 = tl.where(
                mask0 & (ih >= start0) & (ih < start0 + size0), weight0, 0.0
            )
            weight1 = tl.where(
                mask1 & (ih >= start1) & (ih < start1 + size1), weight1, 0.0
            )
            acc0 += data * weight0
            acc1 += data * weight1
            total0 += weight0
            total1 += weight1
        total0 = tl.where(total0 != 0.0, total0, 1.0)
        total1 = tl.where(total1 != 0.0, total1, 1.0)
        out_base = nc * OH * OW
        tl.store(
            out_ptr + out_base + oh0 * OW + ow,
            acc0 / total0,
            mask=mask0 & width_mask,
        )
        tl.store(
            out_ptr + out_base + oh1 * OW + ow,
            acc1 / total1,
            mask=mask1 & width_mask,
        )


@triton.jit
def _bicubic2d_aa_vertical_first_kernel(
    temp_ptr,
    in_ptr,
    total_rows,
    OH,
    IW,
    IH,
    reciprocal_scale_h,
    BLOCK_X: tl.constexpr,
    MAX_INTERP_H: tl.constexpr,
):
    ix = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = ix < IW
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    support = tl.where(reciprocal_scale_h >= 1.0, 2.0 * reciprocal_scale_h, 2.0)
    invscale = tl.where(reciprocal_scale_h >= 1.0, 1.0 / reciprocal_scale_h, 1.0)
    for row_id in range(row_start, total_rows, row_step):
        oh = row_id % OH
        nc = row_id // OH
        center = (oh.to(tl.float32) + 0.5) * reciprocal_scale_h
        start = tl.floor(tl.maximum(center - support + 0.5, 0.0)).to(tl.int32)
        size = tl.floor(tl.minimum(center + support + 0.5, IH) - start).to(tl.int32)
        delta = start.to(tl.float32) - center
        acc = tl.zeros((BLOCK_X,), tl.float32)
        total = tl.full((), 0.0, tl.float32)
        for y in tl.static_range(0, MAX_INTERP_H):
            weight = _cubic_weight((y + delta + 0.5) * invscale)
            weight = tl.where(y < size, weight, 0.0)
            iy = tl.maximum(0, tl.minimum(start + y, IH - 1))
            data = tl.load(in_ptr + (nc * IH + iy) * IW + ix, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += data * weight
            total += weight
        total = tl.where(total != 0.0, total, 1.0)
        tl.store(temp_ptr + row_id * IW + ix, acc / total, mask=mask)


@triton.jit
def _bicubic2d_aa_horizontal_second_kernel(
    out_ptr,
    temp_ptr,
    total_rows,
    OW,
    IW,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    MAX_INTERP_W: tl.constexpr,
):
    ow = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = ow < OW
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    support = tl.where(reciprocal_scale_w >= 1.0, 2.0 * reciprocal_scale_w, 2.0)
    invscale = tl.where(reciprocal_scale_w >= 1.0, 1.0 / reciprocal_scale_w, 1.0)
    center = (ow.to(tl.float32) + 0.5) * reciprocal_scale_w
    start = tl.floor(tl.maximum(center - support + 0.5, 0.0)).to(tl.int32)
    size = tl.floor(tl.minimum(center + support + 0.5, IW) - start).to(tl.int32)
    delta = start.to(tl.float32) - center
    for row_id in range(row_start, total_rows, row_step):
        acc = tl.zeros((BLOCK_X,), tl.float32)
        total = tl.zeros((BLOCK_X,), tl.float32)
        for x in tl.static_range(0, MAX_INTERP_W):
            weight = _cubic_weight((x + delta + 0.5) * invscale)
            weight = tl.where(x < size, weight, 0.0)
            ix = tl.maximum(0, tl.minimum(start + x, IW - 1))
            data = tl.load(temp_ptr + row_id * IW + ix, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += data * weight
            total += weight
        total = tl.where(total != 0.0, total, 1.0)
        tl.store(out_ptr + row_id * OW + ow, acc / total, mask=mask)


@triton.jit
def _compute_upsample_bicubic2d_aa_row(
    out_ptr,
    in_ptr,
    pid_x,
    row_id,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    MAX_INTERP_H: tl.constexpr,
    MAX_INTERP_W: tl.constexpr,
):
    pid_nc = row_id // OH
    n = pid_nc // C
    c = pid_nc - n * C
    oh = row_id - pid_nc * OH
    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = ow < OW

    support_h = tl.where(reciprocal_scale_h >= 1.0, 2.0 * reciprocal_scale_h, 2.0)
    support_w = tl.where(reciprocal_scale_w >= 1.0, 2.0 * reciprocal_scale_w, 2.0)
    interp_h = tl.floor(support_h + 0.5).to(tl.int32) * 2 + 1
    interp_w = tl.floor(support_w + 0.5).to(tl.int32) * 2 + 1
    invscale_h = tl.where(reciprocal_scale_h >= 1.0, 1.0 / reciprocal_scale_h, 1.0)
    invscale_w = tl.where(reciprocal_scale_w >= 1.0, 1.0 / reciprocal_scale_w, 1.0)

    center_h = (oh.to(tl.float32) + 0.5) * reciprocal_scale_h
    center_w = (ow.to(tl.float32) + 0.5) * reciprocal_scale_w
    span_start_h = tl.floor(tl.maximum(center_h - support_h + 0.5, 0.0)).to(tl.int32)
    span_start_w = tl.floor(tl.maximum(center_w - support_w + 0.5, 0.0)).to(tl.int32)
    span_size_h = tl.floor(
        tl.minimum(center_h + support_h + 0.5, IH) - span_start_h
    ).to(tl.int32)
    span_size_w = tl.floor(
        tl.minimum(center_w + support_w + 0.5, IW) - span_start_w
    ).to(tl.int32)
    start_minus_center_h = span_start_h.to(tl.float32) - center_h
    start_minus_center_w = span_start_w.to(tl.float32) - center_w

    in_base = ((n * C + c) * IH) * IW
    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    weight_h_total = tl.full((), 0.0, dtype=tl.float32)

    for y in tl.static_range(0, MAX_INTERP_H):
        wy = _cubic_weight((y + start_minus_center_h + 0.5) * invscale_h)
        wy = tl.where((y < span_size_h) & (y < interp_h), wy, 0.0)
        weight_h_total += wy

        row_acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
        weight_w_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
        for x in tl.static_range(0, MAX_INTERP_W):
            wx = _cubic_weight((x + start_minus_center_w + 0.5) * invscale_w)
            wx = tl.where((x < span_size_w) & (x < interp_w), wx, 0.0)
            weight_w_total += wx

            ih = tl.maximum(0, tl.minimum(span_start_h + y, IH - 1))
            iw = tl.maximum(0, tl.minimum(span_start_w + x, IW - 1))
            data = tl.load(in_ptr + in_base + ih * IW + iw, mask=mask, other=0.0)
            row_acc += data.to(tl.float32) * wx

        weight_w_total = tl.where(weight_w_total != 0.0, weight_w_total, 1.0)
        acc += row_acc / weight_w_total * wy

    weight_h_total = tl.where(weight_h_total != 0.0, weight_h_total, 1.0)
    acc /= weight_h_total

    out_base = ((n * C + c) * OH) * OW
    tl.store(out_ptr + out_base + oh * OW + ow, acc, mask=mask)


@triton.jit
def _upsample_bicubic2d_aa_kernel(
    out_ptr,
    in_ptr,
    total_rows,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    MAX_INTERP_H: tl.constexpr,
    MAX_INTERP_W: tl.constexpr,
):
    pid_x = tl.program_id(0)
    row_start = tl.program_id(1)
    row_step = tl.num_programs(1)
    for row_id in range(row_start, total_rows, row_step):
        _compute_upsample_bicubic2d_aa_row(
            out_ptr,
            in_ptr,
            pid_x,
            row_id,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            BLOCK_X,
            MAX_INTERP_H,
            MAX_INTERP_W,
        )


def _bicubic_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        return (src_size - 1) / (dst_size - 1) if dst_size > 1 else 0
    return 1.0 / scale if scale is not None and scale > 0 else src_size / dst_size


def _max_interp_size(reciprocal_scale):
    if reciprocal_scale < 1.0:
        return 4
    support = 2.0 * reciprocal_scale if reciprocal_scale >= 1.0 else 2.0
    return int(support + 0.5) * 2 + 1


def _upsample_bicubic2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS_TSINGMICRO _UPSAMPLE_BICUBIC2D_AA")
    if input.ndim != 4:
        raise ValueError("The ndim of input must be 4")
    if len(output_size) != 2:
        raise ValueError("The len of output_size must be 2")

    N, C, IH, IW = input.shape
    OH, OW = int(output_size[0]), int(output_size[1])
    reciprocal_scale_h = _bicubic_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = _bicubic_reciprocal_scale(IW, OW, align_corners, scales_w)

    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    contiguous_input = input.contiguous()
    block_x = min(128, triton.next_power_of_2(OW))
    num_warps = 2 if block_x <= 64 else 4
    grid_x = triton.cdiv(OW, block_x)
    max_interp_h = _max_interp_size(reciprocal_scale_h)
    max_interp_w = _max_interp_size(reciprocal_scale_w)

    if reciprocal_scale_h < 1.0 and reciprocal_scale_w < 1.0:
        temp = torch.empty((N, C, IH, OW), device=input.device, dtype=input.dtype)
        horizontal_rows = N * C * IH
        pair_count_w = triton.cdiv(OW, 2)
        pair_block = min(128, triton.next_power_of_2(pair_count_w))
        horizontal_grid = (
            triton.cdiv(pair_count_w, pair_block),
            min(horizontal_rows, 16384),
        )
        _upsample_bicubic2d_aa_horizontal_pair_kernel[horizontal_grid](
            temp,
            contiguous_input,
            horizontal_rows,
            OW,
            IW,
            float(reciprocal_scale_w),
            BLOCK_X=pair_block,
            num_warps=2 if pair_block <= 64 else 4,
        )

        pair_rows = N * C * triton.cdiv(OH, 2)
        vertical_grid = (grid_x, min(pair_rows, 16384))
        _upsample_bicubic2d_aa_vertical_pair_kernel[vertical_grid](
            output,
            temp,
            pair_rows,
            OH,
            OW,
            IH,
            float(reciprocal_scale_h),
            BLOCK_X=block_x,
            num_warps=num_warps,
        )
        return output

    horizontal_first_cost = IH * OW * max_interp_w + OH * OW * max_interp_h
    vertical_first_cost = OH * IW * max_interp_h + OH * OW * max_interp_w
    if vertical_first_cost < horizontal_first_cost:
        temp = torch.empty((N, C, OH, IW), device=input.device, dtype=input.dtype)
        vertical_rows = N * C * OH
        vertical_block = min(128, triton.next_power_of_2(IW))
        vertical_grid = (
            triton.cdiv(IW, vertical_block),
            min(vertical_rows, 4096),
        )
        _bicubic2d_aa_vertical_first_kernel[vertical_grid](
            temp,
            contiguous_input,
            vertical_rows,
            OH,
            IW,
            IH,
            float(reciprocal_scale_h),
            BLOCK_X=vertical_block,
            MAX_INTERP_H=max_interp_h,
            num_warps=2 if vertical_block <= 64 else 4,
        )
        horizontal_grid = (grid_x, min(vertical_rows, 4096))
        _bicubic2d_aa_horizontal_second_kernel[horizontal_grid](
            output,
            temp,
            vertical_rows,
            OW,
            IW,
            float(reciprocal_scale_w),
            BLOCK_X=block_x,
            MAX_INTERP_W=max_interp_w,
            num_warps=num_warps,
        )
        return output

    temp = torch.empty((N, C, IH, OW), device=input.device, dtype=input.dtype)
    horizontal_rows = N * C * IH
    horizontal_grid = (grid_x, min(horizontal_rows, 4096))
    _upsample_bicubic2d_aa_horizontal_kernel[horizontal_grid](
        temp,
        contiguous_input,
        horizontal_rows,
        OW,
        IW,
        float(reciprocal_scale_w),
        BLOCK_X=block_x,
        MAX_INTERP_W=max_interp_w,
        num_warps=num_warps,
    )

    vertical_rows = N * C * OH
    vertical_grid = (grid_x, min(vertical_rows, 4096))
    _upsample_bicubic2d_aa_vertical_kernel[vertical_grid](
        output,
        temp,
        vertical_rows,
        C,
        OH,
        OW,
        IH,
        float(reciprocal_scale_h),
        BLOCK_X=block_x,
        MAX_INTERP_H=max_interp_h,
        num_warps=num_warps,
    )
    return output
