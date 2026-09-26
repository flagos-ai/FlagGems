# Copyright 2026, The FlagOS Contributors.
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

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _cubic_weight(t, a: tl.constexpr):
    ad = tl.abs(t)
    ad2 = ad * ad
    ad3 = ad2 * ad
    w1 = (a + 2.0) * ad3 - (a + 3.0) * ad2 + 1.0
    w2 = a * ad3 - 5.0 * a * ad2 + 8.0 * a * ad - 4.0 * a
    return tl.where(ad <= 1.0, w1, tl.where(ad < 2.0, w2, 0.0))


@triton.jit
def _reflect_int_index(idx, n, align_corners: tl.constexpr):
    if align_corners:
        p = 2 * (n - 1)
        idx_mod = tl.where(p == 0, 0, idx - tl.floor(idx / p) * p)
        return tl.where(idx_mod >= n, p - idx_mod, idx_mod).to(tl.int32)
    else:
        p = 2 * n
        x = idx + 0.5
        x_mod = x - tl.floor(x / p) * p
        x_mod = tl.where(x_mod > n, p - x_mod, x_mod)
        return (x_mod - 0.5).to(tl.int32)


@triton.jit
def _reflect_coord(coord, n, align_corners: tl.constexpr):
    if align_corners:
        min = 0.0
        span = n - 1.0
    else:
        min = -0.5
        span = n * 1.0
    in_abs = tl.abs(coord - min)
    extra = in_abs % span
    flips = tl.floor(in_abs / span)
    ref = tl.where(flips % 2 == 0, extra, span - extra) + min
    ref = tl.where(span == 0, 0.0, ref)
    return tl.minimum(tl.maximum(ref, 0.0), n - 1)


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
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    ow = idx % OW
    oh = idx // OW % OH
    c = idx // OW // OH % C
    n = idx // OW // OH // C % N

    mask = idx < (N * C * OH * OW)

    grid_offset = ((n * OH + oh) * OW + ow) * 2
    gx = tl.load(grid_ptr + grid_offset, mask=mask)
    gy = tl.load(grid_ptr + grid_offset + 1, mask=mask)

    if align_corners:
        x = (gx + 1) * (IW - 1) / 2
        y = (gy + 1) * (IH - 1) / 2
    else:
        x = gx * (IW / 2) + (IW - 1) / 2
        y = gy * (IH / 2) + (IH - 1) / 2

    if interpolation_mode == 0:
        if padding_mode == 2:
            x_ref = _reflect_coord(x, IW, align_corners)
            y_ref = _reflect_coord(y, IH, align_corners)
            x0 = tl.floor(x_ref).to(tl.int32)
            y0 = tl.floor(y_ref).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            x0_f = x0.to(tl.float32)
            y0_f = y0.to(tl.float32)
            wx1 = x_ref - x0_f
            wy1 = y_ref - y0_f
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1
        else:
            x0 = tl.floor(x).to(tl.int32)
            y0 = tl.floor(y).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            x0_f = x0.to(tl.float32)
            y0_f = y0.to(tl.float32)
            wx1 = x - x0_f
            wy1 = y - y0_f
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

        if padding_mode == 0:
            x0_clamped = tl.minimum(tl.maximum(x0, 0), IW - 1)
            y0_clamped = tl.minimum(tl.maximum(y0, 0), IH - 1)
            x1_clamped = tl.minimum(tl.maximum(x1, 0), IW - 1)
            y1_clamped = tl.minimum(tl.maximum(y1, 0), IH - 1)
            x0_valid = (x0 >= 0) & (x0 < IW)
            y0_valid = (y0 >= 0) & (y0 < IH)
            x1_valid = (x1 >= 0) & (x1 < IW)
            y1_valid = (y1 >= 0) & (y1 < IH)
        elif padding_mode == 1:
            x0_clamped = tl.minimum(tl.maximum(x0, 0), IW - 1)
            y0_clamped = tl.minimum(tl.maximum(y0, 0), IH - 1)
            x1_clamped = tl.minimum(tl.maximum(x1, 0), IW - 1)
            y1_clamped = tl.minimum(tl.maximum(y1, 0), IH - 1)
            x0_valid = True
            y0_valid = True
            x1_valid = True
            y1_valid = True
        else:
            x0_clamped = _reflect_int_index(x0.to(tl.float32), IW, align_corners)
            y0_clamped = _reflect_int_index(y0.to(tl.float32), IH, align_corners)
            x1_clamped = _reflect_int_index(x1.to(tl.float32), IW, align_corners)
            y1_clamped = _reflect_int_index(y1.to(tl.float32), IH, align_corners)
            x0_valid = True
            y0_valid = True
            x1_valid = True
            y1_valid = True

        v00 = tl.load(
            input_ptr + ((n * C + c) * IH + y0_clamped) * IW + x0_clamped,
            mask=mask,
            other=0.0,
        )
        v01 = tl.load(
            input_ptr + ((n * C + c) * IH + y0_clamped) * IW + x1_clamped,
            mask=mask,
            other=0.0,
        )
        v10 = tl.load(
            input_ptr + ((n * C + c) * IH + y1_clamped) * IW + x0_clamped,
            mask=mask,
            other=0.0,
        )
        v11 = tl.load(
            input_ptr + ((n * C + c) * IH + y1_clamped) * IW + x1_clamped,
            mask=mask,
            other=0.0,
        )

        if padding_mode == 0:
            v00 = tl.where(x0_valid & y0_valid, v00, 0.0)
            v01 = tl.where(x1_valid & y0_valid, v01, 0.0)
            v10 = tl.where(x0_valid & y1_valid, v10, 0.0)
            v11 = tl.where(x1_valid & y1_valid, v11, 0.0)

        result = v00 * wx0 * wy0 + v01 * wx1 * wy0 + v10 * wx0 * wy1 + v11 * wx1 * wy1

    elif interpolation_mode == 1:
        if padding_mode == 2:
            x = _reflect_coord(x, IW, align_corners)
            y = _reflect_coord(y, IH, align_corners)
        x_floor = tl.floor(x)
        y_floor = tl.floor(y)
        x_frac = x - x_floor
        y_frac = y - y_floor
        x_floor_int = x_floor.to(tl.int32)
        y_floor_int = y_floor.to(tl.int32)
        x_is_half = x_frac == 0.5
        y_is_half = y_frac == 0.5
        x_up = (x_frac > 0.5) | (x_is_half & ((x_floor_int & 1) == 1))
        y_up = (y_frac > 0.5) | (y_is_half & ((y_floor_int & 1) == 1))
        x_nearest = x_floor_int + x_up.to(tl.int32)
        y_nearest = y_floor_int + y_up.to(tl.int32)

        if padding_mode == 0:
            x_in = (x_nearest >= 0) & (x_nearest < IW)
            y_in = (y_nearest >= 0) & (y_nearest < IH)
        else:
            x_in = True
            y_in = True
        x_nearest = tl.minimum(tl.maximum(x_nearest, 0), IW - 1)
        y_nearest = tl.minimum(tl.maximum(y_nearest, 0), IH - 1)

        offset = ((n * C + c) * IH + y_nearest) * IW + x_nearest
        result = tl.load(input_ptr + offset, mask=mask, other=0.0)
        if padding_mode == 0:
            result = tl.where(x_in & y_in, result, 0.0)

    else:
        a: tl.constexpr = -0.75
        x_base_f = tl.floor(x)
        y_base_f = tl.floor(y)
        x_base = x_base_f.to(tl.int32)
        y_base = y_base_f.to(tl.int32)
        tx = x - x_base_f
        ty = y - y_base_f

        wx0 = _cubic_weight(tx + 1.0, a)
        wx1 = _cubic_weight(tx, a)
        wx2 = _cubic_weight(tx - 1.0, a)
        wx3 = _cubic_weight(tx - 2.0, a)
        wy0 = _cubic_weight(ty + 1.0, a)
        wy1 = _cubic_weight(ty, a)
        wy2 = _cubic_weight(ty - 1.0, a)
        wy3 = _cubic_weight(ty - 2.0, a)

        ix0 = x_base - 1
        ix1 = x_base
        ix2 = x_base + 1
        ix3 = x_base + 2
        iy0 = y_base - 1
        iy1 = y_base
        iy2 = y_base + 1
        iy3 = y_base + 2

        if padding_mode == 0:
            ix0_v = (ix0 >= 0) & (ix0 < IW)
            ix1_v = (ix1 >= 0) & (ix1 < IW)
            ix2_v = (ix2 >= 0) & (ix2 < IW)
            ix3_v = (ix3 >= 0) & (ix3 < IW)
            iy0_v = (iy0 >= 0) & (iy0 < IH)
            iy1_v = (iy1 >= 0) & (iy1 < IH)
            iy2_v = (iy2 >= 0) & (iy2 < IH)
            iy3_v = (iy3 >= 0) & (iy3 < IH)
            ix0_c = tl.minimum(tl.maximum(ix0, 0), IW - 1)
            ix1_c = tl.minimum(tl.maximum(ix1, 0), IW - 1)
            ix2_c = tl.minimum(tl.maximum(ix2, 0), IW - 1)
            ix3_c = tl.minimum(tl.maximum(ix3, 0), IW - 1)
            iy0_c = tl.minimum(tl.maximum(iy0, 0), IH - 1)
            iy1_c = tl.minimum(tl.maximum(iy1, 0), IH - 1)
            iy2_c = tl.minimum(tl.maximum(iy2, 0), IH - 1)
            iy3_c = tl.minimum(tl.maximum(iy3, 0), IH - 1)
        elif padding_mode == 1:
            ix0_c = tl.minimum(tl.maximum(ix0, 0), IW - 1)
            ix1_c = tl.minimum(tl.maximum(ix1, 0), IW - 1)
            ix2_c = tl.minimum(tl.maximum(ix2, 0), IW - 1)
            ix3_c = tl.minimum(tl.maximum(ix3, 0), IW - 1)
            iy0_c = tl.minimum(tl.maximum(iy0, 0), IH - 1)
            iy1_c = tl.minimum(tl.maximum(iy1, 0), IH - 1)
            iy2_c = tl.minimum(tl.maximum(iy2, 0), IH - 1)
            iy3_c = tl.minimum(tl.maximum(iy3, 0), IH - 1)
            ix0_v = True
            ix1_v = True
            ix2_v = True
            ix3_v = True
            iy0_v = True
            iy1_v = True
            iy2_v = True
            iy3_v = True
        else:
            ix0_c = _reflect_int_index(ix0.to(tl.float32), IW, align_corners)
            ix1_c = _reflect_int_index(ix1.to(tl.float32), IW, align_corners)
            ix2_c = _reflect_int_index(ix2.to(tl.float32), IW, align_corners)
            ix3_c = _reflect_int_index(ix3.to(tl.float32), IW, align_corners)
            iy0_c = _reflect_int_index(iy0.to(tl.float32), IH, align_corners)
            iy1_c = _reflect_int_index(iy1.to(tl.float32), IH, align_corners)
            iy2_c = _reflect_int_index(iy2.to(tl.float32), IH, align_corners)
            iy3_c = _reflect_int_index(iy3.to(tl.float32), IH, align_corners)
            ix0_v = True
            ix1_v = True
            ix2_v = True
            ix3_v = True
            iy0_v = True
            iy1_v = True
            iy2_v = True
            iy3_v = True

        nc_off = (n * C + c).to(tl.int64) * IH * IW
        p00 = tl.load(
            input_ptr + nc_off + iy0_c.to(tl.int64) * IW + ix0_c, mask=mask, other=0.0
        )
        p01 = tl.load(
            input_ptr + nc_off + iy0_c.to(tl.int64) * IW + ix1_c, mask=mask, other=0.0
        )
        p02 = tl.load(
            input_ptr + nc_off + iy0_c.to(tl.int64) * IW + ix2_c, mask=mask, other=0.0
        )
        p03 = tl.load(
            input_ptr + nc_off + iy0_c.to(tl.int64) * IW + ix3_c, mask=mask, other=0.0
        )
        p10 = tl.load(
            input_ptr + nc_off + iy1_c.to(tl.int64) * IW + ix0_c, mask=mask, other=0.0
        )
        p11 = tl.load(
            input_ptr + nc_off + iy1_c.to(tl.int64) * IW + ix1_c, mask=mask, other=0.0
        )
        p12 = tl.load(
            input_ptr + nc_off + iy1_c.to(tl.int64) * IW + ix2_c, mask=mask, other=0.0
        )
        p13 = tl.load(
            input_ptr + nc_off + iy1_c.to(tl.int64) * IW + ix3_c, mask=mask, other=0.0
        )
        p20 = tl.load(
            input_ptr + nc_off + iy2_c.to(tl.int64) * IW + ix0_c, mask=mask, other=0.0
        )
        p21 = tl.load(
            input_ptr + nc_off + iy2_c.to(tl.int64) * IW + ix1_c, mask=mask, other=0.0
        )
        p22 = tl.load(
            input_ptr + nc_off + iy2_c.to(tl.int64) * IW + ix2_c, mask=mask, other=0.0
        )
        p23 = tl.load(
            input_ptr + nc_off + iy2_c.to(tl.int64) * IW + ix3_c, mask=mask, other=0.0
        )
        p30 = tl.load(
            input_ptr + nc_off + iy3_c.to(tl.int64) * IW + ix0_c, mask=mask, other=0.0
        )
        p31 = tl.load(
            input_ptr + nc_off + iy3_c.to(tl.int64) * IW + ix1_c, mask=mask, other=0.0
        )
        p32 = tl.load(
            input_ptr + nc_off + iy3_c.to(tl.int64) * IW + ix2_c, mask=mask, other=0.0
        )
        p33 = tl.load(
            input_ptr + nc_off + iy3_c.to(tl.int64) * IW + ix3_c, mask=mask, other=0.0
        )

        if padding_mode == 0:
            p00 = tl.where(ix0_v & iy0_v, p00, 0.0)
            p01 = tl.where(ix1_v & iy0_v, p01, 0.0)
            p02 = tl.where(ix2_v & iy0_v, p02, 0.0)
            p03 = tl.where(ix3_v & iy0_v, p03, 0.0)
            p10 = tl.where(ix0_v & iy1_v, p10, 0.0)
            p11 = tl.where(ix1_v & iy1_v, p11, 0.0)
            p12 = tl.where(ix2_v & iy1_v, p12, 0.0)
            p13 = tl.where(ix3_v & iy1_v, p13, 0.0)
            p20 = tl.where(ix0_v & iy2_v, p20, 0.0)
            p21 = tl.where(ix1_v & iy2_v, p21, 0.0)
            p22 = tl.where(ix2_v & iy2_v, p22, 0.0)
            p23 = tl.where(ix3_v & iy2_v, p23, 0.0)
            p30 = tl.where(ix0_v & iy3_v, p30, 0.0)
            p31 = tl.where(ix1_v & iy3_v, p31, 0.0)
            p32 = tl.where(ix2_v & iy3_v, p32, 0.0)
            p33 = tl.where(ix3_v & iy3_v, p33, 0.0)

        row0 = (p00 * wx0 + p01 * wx1 + p02 * wx2 + p03 * wx3) * wy0
        row1 = (p10 * wx0 + p11 * wx1 + p12 * wx2 + p13 * wx3) * wy1
        row2 = (p20 * wx0 + p21 * wx1 + p22 * wx2 + p23 * wx3) * wy2
        row3 = (p30 * wx0 + p31 * wx1 + p32 * wx2 + p33 * wx3) * wy3
        result = row0 + row1 + row2 + row3

    out_offset = ((n * C + c) * OH + oh) * OW + ow
    tl.store(output_ptr + out_offset, result, mask=mask)


def grid_sampler_2d(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
):
    """Grid sampling with bilinear/nearest/bicubic and zeros/border/reflection.

    Args:
        input: 4D tensor of shape (N, C, IH, IW).
        grid: 4D tensor of shape (N, OH, OW, 2) with normalized coords.
        interpolation_mode: 0=Bilinear, 1=Nearest, 2=Bicubic.
        padding_mode: 0=Zeros, 1=Border, 2=Reflection.
        align_corners: If True, corner pixels are aligned.
    """
    logger.debug("GEMS_KUNLUNXIN GRID_SAMPLER_2D")

    assert input.ndim == 4, "Input must be 4D"
    assert grid.ndim == 4, "Grid must be 4D"
    assert grid.shape[-1] == 2, "Grid must have last dimension of size 2"

    N, C, IH, IW = input.shape
    OH, OW = grid.shape[1:3]

    assert grid.shape[0] == N, "Batch size mismatch"

    output = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    if output.numel() == 0:
        return output

    input = input.contiguous()
    grid = grid.contiguous()

    total_threads = N * C * OH * OW
    BLOCK_SIZE = 1024
    grid_fn = (triton.cdiv(total_threads, BLOCK_SIZE),)

    with torch.cuda.device(input.device):
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
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output
