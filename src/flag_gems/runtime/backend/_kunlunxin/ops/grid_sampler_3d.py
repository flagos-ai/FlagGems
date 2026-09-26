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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _round_half_to_even_fp32(x):
    """Round half to even (banker's rounding). ``x`` must be fp32."""
    r = tl.floor(x)
    d = x - r
    is_odd = tl.abs(r - 2.0 * tl.floor(r / 2.0)) > 0.5
    return tl.where((d > 0.5) | ((tl.abs(d - 0.5) < 1e-10) & is_odd), r + 1.0, r)


@libentry()
@triton.jit
def grid_sampler_3d_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    n,
    c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    in_strides_n,
    in_strides_c,
    in_strides_d,
    in_strides_h,
    in_strides_w,
    grid_strides_n,
    grid_strides_od,
    grid_strides_oh,
    grid_strides_ow,
    grid_stride_xyz,
    out_strides_n,
    out_strides_c,
    out_strides_d,
    out_strides_h,
    out_strides_w,
    total_out,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid >= total_out:
        return

    n_idx = pid // (out_d * out_h * out_w)
    remainder = pid % (out_d * out_h * out_w)
    od_idx = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    oh_idx = remainder // out_w
    ow_idx = remainder % out_w

    grid_offset = (
        n_idx * grid_strides_n
        + od_idx * grid_strides_od
        + oh_idx * grid_strides_oh
        + ow_idx * grid_strides_ow
    )
    gx = tl.load(grid_ptr + grid_offset + 0 * grid_stride_xyz).to(tl.float32)
    gy = tl.load(grid_ptr + grid_offset + 1 * grid_stride_xyz).to(tl.float32)
    gz = tl.load(grid_ptr + grid_offset + 2 * grid_stride_xyz).to(tl.float32)

    grid_x_nan = gx != gx
    grid_y_nan = gy != gy
    grid_z_nan = gz != gz
    gx = tl.where(grid_x_nan, -2.0, gx)
    gy = tl.where(grid_y_nan, -2.0, gy)
    gz = tl.where(grid_z_nan, -2.0, gz)

    if align_corners:
        x = (gx + 1.0) * tl.cast(in_w - 1, tl.float32) * 0.5
        y = (gy + 1.0) * tl.cast(in_h - 1, tl.float32) * 0.5
        z = (gz + 1.0) * tl.cast(in_d - 1, tl.float32) * 0.5
    else:
        x = (gx + 1.0) * tl.cast(in_w, tl.float32) * 0.5 - 0.5
        y = (gy + 1.0) * tl.cast(in_h, tl.float32) * 0.5 - 0.5
        z = (gz + 1.0) * tl.cast(in_d, tl.float32) * 0.5 - 0.5

    if padding_mode == 0:
        x_pad = x
        y_pad = y
        z_pad = z
    elif padding_mode == 1:
        x_pad = tl.maximum(tl.minimum(x, tl.cast(in_w - 1, tl.float32)), 0.0)
        y_pad = tl.maximum(tl.minimum(y, tl.cast(in_h - 1, tl.float32)), 0.0)
        z_pad = tl.maximum(tl.minimum(z, tl.cast(in_d - 1, tl.float32)), 0.0)
    else:
        x_shifted = gx + 1.0
        x_mod = x_shifted % 4.0
        x_mod = tl.where(x_mod < 0, x_mod + 4.0, x_mod)
        gx_refl = tl.where(x_mod <= 2.0, x_mod, 4.0 - x_mod) - 1.0

        y_shifted = gy + 1.0
        y_mod = y_shifted % 4.0
        y_mod = tl.where(y_mod < 0, y_mod + 4.0, y_mod)
        gy_refl = tl.where(y_mod <= 2.0, y_mod, 4.0 - y_mod) - 1.0

        z_shifted = gz + 1.0
        z_mod = z_shifted % 4.0
        z_mod = tl.where(z_mod < 0, z_mod + 4.0, z_mod)
        gz_refl = tl.where(z_mod <= 2.0, z_mod, 4.0 - z_mod) - 1.0

        if align_corners:
            x_pad = (gx_refl + 1.0) * tl.cast(in_w - 1, tl.float32) * 0.5
            y_pad = (gy_refl + 1.0) * tl.cast(in_h - 1, tl.float32) * 0.5
            z_pad = (gz_refl + 1.0) * tl.cast(in_d - 1, tl.float32) * 0.5
        else:
            x_pad = (gx_refl + 1.0) * tl.cast(in_w, tl.float32) * 0.5 - 0.5
            y_pad = (gy_refl + 1.0) * tl.cast(in_h, tl.float32) * 0.5 - 0.5
            z_pad = (gz_refl + 1.0) * tl.cast(in_d, tl.float32) * 0.5 - 0.5

    if interpolation_mode == 1:
        if padding_mode == 0:
            xs, ys, zs = x, y, z
        else:
            xs, ys, zs = x_pad, y_pad, z_pad

        ix = tl.cast(_round_half_to_even_fp32(xs), tl.int32)
        iy = tl.cast(_round_half_to_even_fp32(ys), tl.int32)
        iz = tl.cast(_round_half_to_even_fp32(zs), tl.int32)

        if padding_mode != 0:
            ix = tl.maximum(0, tl.minimum(ix, in_w - 1))
            iy = tl.maximum(0, tl.minimum(iy, in_h - 1))
            iz = tl.maximum(0, tl.minimum(iz, in_d - 1))
    else:
        if padding_mode == 0:
            ix0 = tl.cast(tl.floor(x), tl.int32)
            iy0 = tl.cast(tl.floor(y), tl.int32)
            iz0 = tl.cast(tl.floor(z), tl.int32)
            fx = x - tl.floor(x)
            fy = y - tl.floor(y)
            fz = z - tl.floor(z)
        else:
            ix0 = tl.cast(tl.floor(x_pad), tl.int32)
            iy0 = tl.cast(tl.floor(y_pad), tl.int32)
            iz0 = tl.cast(tl.floor(z_pad), tl.int32)
            fx = x_pad - tl.floor(x_pad)
            fy = y_pad - tl.floor(y_pad)
            fz = z_pad - tl.floor(z_pad)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        if padding_mode != 0:
            ix0 = tl.maximum(0, tl.minimum(ix0, in_w - 1))
            iy0 = tl.maximum(0, tl.minimum(iy0, in_h - 1))
            iz0 = tl.maximum(0, tl.minimum(iz0, in_d - 1))
            ix1 = tl.maximum(0, tl.minimum(ix1, in_w - 1))
            iy1 = tl.maximum(0, tl.minimum(iy1, in_h - 1))
            iz1 = tl.maximum(0, tl.minimum(iz1, in_d - 1))

    out_offset_base = (
        n_idx * out_strides_n
        + od_idx * out_strides_d
        + oh_idx * out_strides_h
        + ow_idx * out_strides_w
    )

    for channel in range(c):
        if interpolation_mode == 1:
            if padding_mode == 0:
                valid = (
                    (iz >= 0)
                    & (iz < in_d)
                    & (iy >= 0)
                    & (iy < in_h)
                    & (ix >= 0)
                    & (ix < in_w)
                    & ~grid_x_nan
                    & ~grid_y_nan
                    & ~grid_z_nan
                )
                ixc = tl.maximum(0, tl.minimum(ix, in_w - 1))
                iyc = tl.maximum(0, tl.minimum(iy, in_h - 1))
                izc = tl.maximum(0, tl.minimum(iz, in_d - 1))
                inp_offset = (
                    n_idx * in_strides_n
                    + channel * in_strides_c
                    + izc * in_strides_d
                    + iyc * in_strides_h
                    + ixc * in_strides_w
                )
                val = tl.load(input_ptr + inp_offset)
                val = tl.where(valid, val, 0.0)
            else:
                inp_offset = (
                    n_idx * in_strides_n
                    + channel * in_strides_c
                    + iz * in_strides_d
                    + iy * in_strides_h
                    + ix * in_strides_w
                )
                val = tl.where(
                    grid_x_nan | grid_y_nan | grid_z_nan,
                    0.0,
                    tl.load(input_ptr + inp_offset),
                )
        else:
            if padding_mode == 0:
                nm = ~grid_x_nan & ~grid_y_nan & ~grid_z_nan
                iz0_v = (iz0 >= 0) & (iz0 < in_d)
                iz1_v = (iz1 >= 0) & (iz1 < in_d)
                iy0_v = (iy0 >= 0) & (iy0 < in_h)
                iy1_v = (iy1 >= 0) & (iy1 < in_h)
                ix0_v = (ix0 >= 0) & (ix0 < in_w)
                ix1_v = (ix1 >= 0) & (ix1 < in_w)

                m000 = iz0_v & iy0_v & ix0_v & nm
                m001 = iz0_v & iy0_v & ix1_v & nm
                m010 = iz0_v & iy1_v & ix0_v & nm
                m011 = iz0_v & iy1_v & ix1_v & nm
                m100 = iz1_v & iy0_v & ix0_v & nm
                m101 = iz1_v & iy0_v & ix1_v & nm
                m110 = iz1_v & iy1_v & ix0_v & nm
                m111 = iz1_v & iy1_v & ix1_v & nm

                ix0c = tl.maximum(0, tl.minimum(ix0, in_w - 1))
                ix1c = tl.maximum(0, tl.minimum(ix1, in_w - 1))
                iy0c = tl.maximum(0, tl.minimum(iy0, in_h - 1))
                iy1c = tl.maximum(0, tl.minimum(iy1, in_h - 1))
                iz0c = tl.maximum(0, tl.minimum(iz0, in_d - 1))
                iz1c = tl.maximum(0, tl.minimum(iz1, in_d - 1))

                base = n_idx * in_strides_n + channel * in_strides_c
                c000 = tl.load(
                    input_ptr
                    + base
                    + iz0c * in_strides_d
                    + iy0c * in_strides_h
                    + ix0c * in_strides_w
                )
                c001 = tl.load(
                    input_ptr
                    + base
                    + iz0c * in_strides_d
                    + iy0c * in_strides_h
                    + ix1c * in_strides_w
                )
                c010 = tl.load(
                    input_ptr
                    + base
                    + iz0c * in_strides_d
                    + iy1c * in_strides_h
                    + ix0c * in_strides_w
                )
                c011 = tl.load(
                    input_ptr
                    + base
                    + iz0c * in_strides_d
                    + iy1c * in_strides_h
                    + ix1c * in_strides_w
                )
                c100 = tl.load(
                    input_ptr
                    + base
                    + iz1c * in_strides_d
                    + iy0c * in_strides_h
                    + ix0c * in_strides_w
                )
                c101 = tl.load(
                    input_ptr
                    + base
                    + iz1c * in_strides_d
                    + iy0c * in_strides_h
                    + ix1c * in_strides_w
                )
                c110 = tl.load(
                    input_ptr
                    + base
                    + iz1c * in_strides_d
                    + iy1c * in_strides_h
                    + ix0c * in_strides_w
                )
                c111 = tl.load(
                    input_ptr
                    + base
                    + iz1c * in_strides_d
                    + iy1c * in_strides_h
                    + ix1c * in_strides_w
                )

                c000 = tl.where(m000, c000, 0.0)
                c001 = tl.where(m001, c001, 0.0)
                c010 = tl.where(m010, c010, 0.0)
                c011 = tl.where(m011, c011, 0.0)
                c100 = tl.where(m100, c100, 0.0)
                c101 = tl.where(m101, c101, 0.0)
                c110 = tl.where(m110, c110, 0.0)
                c111 = tl.where(m111, c111, 0.0)
            else:
                base = n_idx * in_strides_n + channel * in_strides_c
                c000 = tl.load(
                    input_ptr
                    + base
                    + iz0 * in_strides_d
                    + iy0 * in_strides_h
                    + ix0 * in_strides_w
                )
                c001 = tl.load(
                    input_ptr
                    + base
                    + iz0 * in_strides_d
                    + iy0 * in_strides_h
                    + ix1 * in_strides_w
                )
                c010 = tl.load(
                    input_ptr
                    + base
                    + iz0 * in_strides_d
                    + iy1 * in_strides_h
                    + ix0 * in_strides_w
                )
                c011 = tl.load(
                    input_ptr
                    + base
                    + iz0 * in_strides_d
                    + iy1 * in_strides_h
                    + ix1 * in_strides_w
                )
                c100 = tl.load(
                    input_ptr
                    + base
                    + iz1 * in_strides_d
                    + iy0 * in_strides_h
                    + ix0 * in_strides_w
                )
                c101 = tl.load(
                    input_ptr
                    + base
                    + iz1 * in_strides_d
                    + iy0 * in_strides_h
                    + ix1 * in_strides_w
                )
                c110 = tl.load(
                    input_ptr
                    + base
                    + iz1 * in_strides_d
                    + iy1 * in_strides_h
                    + ix0 * in_strides_w
                )
                c111 = tl.load(
                    input_ptr
                    + base
                    + iz1 * in_strides_d
                    + iy1 * in_strides_h
                    + ix1 * in_strides_w
                )

            c00 = c000 * (1.0 - fz) + c100 * fz
            c01 = c001 * (1.0 - fz) + c101 * fz
            c10 = c010 * (1.0 - fz) + c110 * fz
            c11 = c011 * (1.0 - fz) + c111 * fz
            c0 = c00 * (1.0 - fy) + c10 * fy
            c1 = c01 * (1.0 - fy) + c11 * fy
            val = c0 * (1.0 - fx) + c1 * fx
            if padding_mode != 0:
                val = tl.where(grid_x_nan | grid_y_nan | grid_z_nan, 0.0, val)

        out_offset = out_offset_base + channel * out_strides_c
        tl.store(output_ptr + out_offset, val)


def grid_sampler_3d(
    input,
    grid,
    interpolation_mode=0,
    padding_mode=0,
    align_corners=False,
):
    """Grid sampler 3D with trilinear or nearest interpolation (XPU override)."""
    logger.debug("GEMS_KUNLUNXIN GRID_SAMPLER_3D")

    N, C, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = grid.shape[1:4]

    output = torch.empty(
        (N, C, out_d, out_h, out_w),
        dtype=input.dtype,
        device=input.device,
    )

    if output.numel() == 0:
        return output

    total_out = N * out_d * out_h * out_w

    def get_grid(meta):
        return (total_out,)

    grid_sampler_3d_kernel[get_grid](
        input,
        grid,
        output,
        N,
        C,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        *input.stride(),
        *grid.stride(),
        *output.stride(),
        total_out,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    return output
