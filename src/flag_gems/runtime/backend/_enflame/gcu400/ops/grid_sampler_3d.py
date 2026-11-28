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

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 has 48 SPs; extra programs beyond that only pay the wave/dispatch
# tax for this gather-shaped kernel (grid_sampler_2d probe: null kernel at
# grid=2048 costs 104.7us vs 2.5us at grid=48). The host caps the launch at
# 48 programs and the kernel walks the remaining work with a grid-stride
# loop, so any output size stays launchable (P19 grid.x<=65535 included).
_MAX_PIDS = 48
_INT32_MAX = 1 << 31


@triton.jit
def _grid_sampler_3d_reflect_coord(coord, n, align_corners: tl.constexpr):
    """Reflect a continuous (pixel-space) coordinate into the valid range.

    align_corners=True:  reflect over [0, n-1] (min=0, span=n-1).
    align_corners=False: reflect over [-0.5, n-0.5] (min=-0.5, span=n).
    The triangle wave folds every span and the flip parity picks the mirror.
    Mirrors grid_sampler._grid_sampler_reflect_coord (same semantics as the
    ATen reference); the operand of the float modulo is always non-negative.
    """
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
def _grid_sampler_3d_reflect_int_index(idx, n, align_corners: tl.constexpr):
    """Reflect an integer tap index into [0, n-1] (multi-period safe)."""
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
def _grid_sampler_3d_body_points(
    pidx,
    input_ptr,
    grid_ptr,
    output_ptr,
    c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    total_points,
    ch_start,
    ch_end,
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
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    LOG_OW: tl.constexpr,
    LOG_OH: tl.constexpr,
    LOG_OD: tl.constexpr,
    SHIFT_MODE: tl.constexpr,
):
    # Decode the flat SPATIAL point index into (n, od, oh, ow). The channel
    # dim is NOT part of the index: grid loads, coordinate math and the 8
    # corner offsets are computed once per point and reused across the
    # channel loop below (the channel-folded flat form recomputed all of it
    # per (point, channel) pair, which dominated the large shapes).
    if SHIFT_MODE == 1:  # out_w/out_h/out_d are powers of two
        ow = pidx & (out_w - 1)
        oh = (pidx >> LOG_OW) & (out_h - 1)
        od = (pidx >> (LOG_OW + LOG_OH)) & (out_d - 1)
        b = pidx >> (LOG_OW + LOG_OH + LOG_OD)
    else:
        ow = pidx % out_w
        oh = pidx // out_w % out_h
        od = pidx // out_w // out_h % out_d
        b = pidx // out_w // out_h // out_d

    pmask = pidx < total_points

    grid_offset = (
        b * grid_strides_n
        + od * grid_strides_od
        + oh * grid_strides_oh
        + ow * grid_strides_ow
    )
    gx = tl.load(grid_ptr + grid_offset, mask=pmask, other=0.0).to(tl.float32)
    gy = tl.load(grid_ptr + grid_offset + grid_stride_xyz, mask=pmask, other=0.0).to(
        tl.float32
    )
    gz = tl.load(
        grid_ptr + grid_offset + 2 * grid_stride_xyz, mask=pmask, other=0.0
    ).to(tl.float32)
    # NaN grid entries: sentinel -1 (outside [-1, 1] on the grid side).
    # zeros padding then yields 0 and border/reflection fold to the
    # (0, 0, 0) edge pixel, matching the CUDA reference semantics.
    grid_nan = (gx != gx) | (gy != gy) | (gz != gz)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)
    gz = tl.where(gz != gz, -1.0, gz)

    # Unnormalize grid coordinates from [-1, 1] to input pixel space.
    if align_corners:
        x = (gx + 1) * (in_w - 1) / 2
        y = (gy + 1) * (in_h - 1) / 2
        z = (gz + 1) * (in_d - 1) / 2
    else:
        x = gx * (in_w / 2) + (in_w - 1) / 2
        y = gy * (in_h / 2) + (in_h - 1) / 2
        z = gz * (in_d / 2) + (in_d - 1) / 2

    in_base_p = b * in_strides_n
    out_base_p = (
        b * out_strides_n + od * out_strides_d + oh * out_strides_h + ow * out_strides_w
    )

    if interpolation_mode == 0:  # Trilinear
        if padding_mode == 2:
            x = _grid_sampler_3d_reflect_coord(x, in_w, align_corners)
            y = _grid_sampler_3d_reflect_coord(y, in_h, align_corners)
            z = _grid_sampler_3d_reflect_coord(z, in_d, align_corners)
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        z0 = tl.floor(z).to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        wx1 = x - x0.to(tl.float32)
        wy1 = y - y0.to(tl.float32)
        wz1 = z - z0.to(tl.float32)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1
        wz0 = 1.0 - wz1

        if padding_mode == 0:  # Zeros: each corner masked by its own coords
            x0c = x0
            y0c = y0
            z0c = z0
            x1c = x1
            y1c = y1
            z1c = z1
            # NaN grid -> every corner invalid (torch zeros+NaN yields 0).
            xv0 = (x0 >= 0) & (x0 < in_w)
            yv0 = (y0 >= 0) & (y0 < in_h)
            zv0 = (z0 >= 0) & (z0 < in_d)
            xv1 = (x1 >= 0) & (x1 < in_w)
            yv1 = (y1 >= 0) & (y1 < in_h)
            zv1 = (z1 >= 0) & (z1 < in_d)
            m000 = xv0 & yv0 & zv0 & ~grid_nan
            m100 = xv1 & yv0 & zv0 & ~grid_nan
            m010 = xv0 & yv1 & zv0 & ~grid_nan
            m110 = xv1 & yv1 & zv0 & ~grid_nan
            m001 = xv0 & yv0 & zv1 & ~grid_nan
            m101 = xv1 & yv0 & zv1 & ~grid_nan
            m011 = xv0 & yv1 & zv1 & ~grid_nan
            m111 = xv1 & yv1 & zv1 & ~grid_nan
        elif padding_mode == 1:  # Border: clamp each corner
            x0c = tl.minimum(tl.maximum(x0, 0), in_w - 1)
            y0c = tl.minimum(tl.maximum(y0, 0), in_h - 1)
            z0c = tl.minimum(tl.maximum(z0, 0), in_d - 1)
            x1c = tl.minimum(tl.maximum(x1, 0), in_w - 1)
            y1c = tl.minimum(tl.maximum(y1, 0), in_h - 1)
            z1c = tl.minimum(tl.maximum(z1, 0), in_d - 1)
            m000 = True
            m100 = True
            m010 = True
            m110 = True
            m001 = True
            m101 = True
            m011 = True
            m111 = True
        else:  # Reflection: reflect each integer tap independently
            x0c = _grid_sampler_3d_reflect_int_index(
                x0.to(tl.float32), in_w, align_corners
            )
            y0c = _grid_sampler_3d_reflect_int_index(
                y0.to(tl.float32), in_h, align_corners
            )
            z0c = _grid_sampler_3d_reflect_int_index(
                z0.to(tl.float32), in_d, align_corners
            )
            x1c = _grid_sampler_3d_reflect_int_index(
                x1.to(tl.float32), in_w, align_corners
            )
            y1c = _grid_sampler_3d_reflect_int_index(
                y1.to(tl.float32), in_h, align_corners
            )
            z1c = _grid_sampler_3d_reflect_int_index(
                z1.to(tl.float32), in_d, align_corners
            )
            m000 = True
            m100 = True
            m010 = True
            m110 = True
            m001 = True
            m101 = True
            m011 = True
            m111 = True

        c000 = z0c * in_strides_d + y0c * in_strides_h + x0c * in_strides_w
        c100 = z0c * in_strides_d + y0c * in_strides_h + x1c * in_strides_w
        c010 = z0c * in_strides_d + y1c * in_strides_h + x0c * in_strides_w
        c110 = z0c * in_strides_d + y1c * in_strides_h + x1c * in_strides_w
        c001 = z1c * in_strides_d + y0c * in_strides_h + x0c * in_strides_w
        c101 = z1c * in_strides_d + y0c * in_strides_h + x1c * in_strides_w
        c011 = z1c * in_strides_d + y1c * in_strides_h + x0c * in_strides_w
        c111 = z1c * in_strides_d + y1c * in_strides_h + x1c * in_strides_w
        w000 = wx0 * wy0 * wz0
        w100 = wx1 * wy0 * wz0
        w010 = wx0 * wy1 * wz0
        w110 = wx1 * wy1 * wz0
        w001 = wx0 * wy0 * wz1
        w101 = wx1 * wy0 * wz1
        w011 = wx0 * wy1 * wz1
        w111 = wx1 * wy1 * wz1
        # Channel loop: one trilinear gather + scatter per (point, channel).
        # The corner offsets/weights above are point-lane values reused by
        # every channel iteration; only the channel base is bumped.
        for ch_i in tl.range(ch_start, ch_end):
            in_base = in_base_p + ch_i * in_strides_c
            v000 = tl.load(input_ptr + in_base + c000, mask=pmask & m000, other=0.0)
            v100 = tl.load(input_ptr + in_base + c100, mask=pmask & m100, other=0.0)
            v010 = tl.load(input_ptr + in_base + c010, mask=pmask & m010, other=0.0)
            v110 = tl.load(input_ptr + in_base + c110, mask=pmask & m110, other=0.0)
            v001 = tl.load(input_ptr + in_base + c001, mask=pmask & m001, other=0.0)
            v101 = tl.load(input_ptr + in_base + c101, mask=pmask & m101, other=0.0)
            v011 = tl.load(input_ptr + in_base + c011, mask=pmask & m011, other=0.0)
            v111 = tl.load(input_ptr + in_base + c111, mask=pmask & m111, other=0.0)
            result = (
                v000 * w000
                + v100 * w100
                + v010 * w010
                + v110 * w110
                + v001 * w001
                + v101 * w101
                + v011 * w011
                + v111 * w111
            )
            tl.store(
                output_ptr + out_base_p + ch_i * out_strides_c,
                result,
                mask=pmask,
            )
    else:  # Nearest (interpolation_mode == 1)
        if padding_mode == 2:
            x = _grid_sampler_3d_reflect_coord(x, in_w, align_corners)
            y = _grid_sampler_3d_reflect_coord(y, in_h, align_corners)
            z = _grid_sampler_3d_reflect_coord(z, in_d, align_corners)
        x_floor = tl.floor(x)
        y_floor = tl.floor(y)
        z_floor = tl.floor(z)
        x_frac = x - x_floor
        y_frac = y - y_floor
        z_frac = z - z_floor
        x_floor_int = x_floor.to(tl.int32)
        y_floor_int = y_floor.to(tl.int32)
        z_floor_int = z_floor.to(tl.int32)
        # Round half to even (banker's rounding), matching PyTorch's
        # nearbyint.
        x_is_half = x_frac == 0.5
        y_is_half = y_frac == 0.5
        z_is_half = z_frac == 0.5
        x_up = (x_frac > 0.5) | (x_is_half & ((x_floor_int & 1) == 1))
        y_up = (y_frac > 0.5) | (y_is_half & ((y_floor_int & 1) == 1))
        z_up = (z_frac > 0.5) | (z_is_half & ((z_floor_int & 1) == 1))
        x_near = x_floor_int + x_up.to(tl.int32)
        y_near = y_floor_int + y_up.to(tl.int32)
        z_near = z_floor_int + z_up.to(tl.int32)

        if padding_mode == 0:  # Zeros
            # NaN grid -> out of range (torch zeros+NaN yields 0).
            x_in = (x_near >= 0) & (x_near < in_w) & ~grid_nan
            y_in = (y_near >= 0) & (y_near < in_h)
            z_in = (z_near >= 0) & (z_near < in_d)
            x_near = tl.minimum(tl.maximum(x_near, 0), in_w - 1)
            y_near = tl.minimum(tl.maximum(y_near, 0), in_h - 1)
            z_near = tl.minimum(tl.maximum(z_near, 0), in_d - 1)
            pixel_mask = pmask & x_in & y_in & z_in
        else:  # Border / Reflection: coordinates are already in range
            x_near = tl.minimum(tl.maximum(x_near, 0), in_w - 1)
            y_near = tl.minimum(tl.maximum(y_near, 0), in_h - 1)
            z_near = tl.minimum(tl.maximum(z_near, 0), in_d - 1)
            pixel_mask = pmask

        corner = z_near * in_strides_d + y_near * in_strides_h + x_near * in_strides_w
        for ch_i in tl.range(ch_start, ch_end):
            in_base = in_base_p + ch_i * in_strides_c
            result = tl.load(input_ptr + in_base + corner, mask=pixel_mask, other=0.0)
            tl.store(
                output_ptr + out_base_p + ch_i * out_strides_c,
                result,
                mask=pmask,
            )


@libentry()
@triton.jit
def grid_sampler_3d_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    total_points,
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
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    LOG_OW: tl.constexpr,
    LOG_OH: tl.constexpr,
    LOG_OD: tl.constexpr,
    SHIFT_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CH_GROUPS: tl.constexpr,
    CH_SPLIT: tl.constexpr,
):
    # int32 fast path: caller guarantees every flat element offset below.
    # Work items are (point_block, channel_group) pairs; with few point
    # blocks the channel dim is split across programs to fill all 48 SPs,
    # with many point blocks each program keeps the whole channel loop
    # (pure channel hoisting, best for the large shapes).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    offsets_in_block = tl.arange(0, BLOCK_SIZE)
    num_blocks = (total_points + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_items = num_blocks * CH_GROUPS
    for item in tl.range(pid, num_items, num_pids):
        pb = item // CH_GROUPS
        chg = item - pb * CH_GROUPS
        ch_start = chg * CH_SPLIT
        ch_end = tl.minimum(ch_start + CH_SPLIT, c)
        pidx = pb * BLOCK_SIZE + offsets_in_block
        _grid_sampler_3d_body_points(
            pidx,
            input_ptr,
            grid_ptr,
            output_ptr,
            c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            total_points,
            ch_start,
            ch_end,
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
            interpolation_mode,
            padding_mode,
            align_corners,
            LOG_OW,
            LOG_OH,
            LOG_OD,
            SHIFT_MODE,
        )


@libentry()
@triton.jit
def grid_sampler_3d_kernel_wide(
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
    total,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # int64 fallback for flat extents beyond int32: the previous (proven)
    # channel-folded flat form with explicit int64 offsets, grid-strided so
    # the launch never exceeds the grid-dimension limit for huge tensors.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    offsets_in_block = tl.arange(0, BLOCK_SIZE)
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid_w = pid.to(tl.int64)
    for block_id in tl.range(pid_w, num_blocks, num_pids.to(tl.int64)):
        idx = block_id * BLOCK_SIZE + offsets_in_block.to(tl.int64)
        mask = idx < total

        ow = idx % out_w
        oh = idx // out_w % out_h
        od = idx // out_w // out_h % out_d
        ch = idx // out_w // out_h // out_d % c
        b = idx // out_w // out_h // out_d // c

        grid_offset = (
            b * grid_strides_n
            + od * grid_strides_od
            + oh * grid_strides_oh
            + ow * grid_strides_ow
        )
        gx = tl.load(grid_ptr + grid_offset, mask=mask, other=0.0).to(tl.float32)
        gy = tl.load(grid_ptr + grid_offset + grid_stride_xyz, mask=mask, other=0.0).to(
            tl.float32
        )
        gz = tl.load(
            grid_ptr + grid_offset + 2 * grid_stride_xyz, mask=mask, other=0.0
        ).to(tl.float32)
        grid_nan = (gx != gx) | (gy != gy) | (gz != gz)
        gx = tl.where(gx != gx, -1.0, gx)
        gy = tl.where(gy != gy, -1.0, gy)
        gz = tl.where(gz != gz, -1.0, gz)

        if align_corners:
            x = (gx + 1) * (in_w - 1) / 2
            y = (gy + 1) * (in_h - 1) / 2
            z = (gz + 1) * (in_d - 1) / 2
        else:
            x = gx * (in_w / 2) + (in_w - 1) / 2
            y = gy * (in_h / 2) + (in_h - 1) / 2
            z = gz * (in_d / 2) + (in_d - 1) / 2

        if interpolation_mode == 0:  # Trilinear
            if padding_mode == 2:
                x = _grid_sampler_3d_reflect_coord(x, in_w, align_corners)
                y = _grid_sampler_3d_reflect_coord(y, in_h, align_corners)
                z = _grid_sampler_3d_reflect_coord(z, in_d, align_corners)
            x0 = tl.floor(x).to(tl.int32)
            y0 = tl.floor(y).to(tl.int32)
            z0 = tl.floor(z).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1
            wx1 = x - x0.to(tl.float32)
            wy1 = y - y0.to(tl.float32)
            wz1 = z - z0.to(tl.float32)
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1
            wz0 = 1.0 - wz1

            if padding_mode == 0:
                x0c = x0
                y0c = y0
                z0c = z0
                x1c = x1
                y1c = y1
                z1c = z1
                xv0 = (x0 >= 0) & (x0 < in_w)
                yv0 = (y0 >= 0) & (y0 < in_h)
                zv0 = (z0 >= 0) & (z0 < in_d)
                xv1 = (x1 >= 0) & (x1 < in_w)
                yv1 = (y1 >= 0) & (y1 < in_h)
                zv1 = (z1 >= 0) & (z1 < in_d)
                m000 = xv0 & yv0 & zv0 & ~grid_nan
                m100 = xv1 & yv0 & zv0 & ~grid_nan
                m010 = xv0 & yv1 & zv0 & ~grid_nan
                m110 = xv1 & yv1 & zv0 & ~grid_nan
                m001 = xv0 & yv0 & zv1 & ~grid_nan
                m101 = xv1 & yv0 & zv1 & ~grid_nan
                m011 = xv0 & yv1 & zv1 & ~grid_nan
                m111 = xv1 & yv1 & zv1 & ~grid_nan
            elif padding_mode == 1:
                x0c = tl.minimum(tl.maximum(x0, 0), in_w - 1)
                y0c = tl.minimum(tl.maximum(y0, 0), in_h - 1)
                z0c = tl.minimum(tl.maximum(z0, 0), in_d - 1)
                x1c = tl.minimum(tl.maximum(x1, 0), in_w - 1)
                y1c = tl.minimum(tl.maximum(y1, 0), in_h - 1)
                z1c = tl.minimum(tl.maximum(z1, 0), in_d - 1)
                m000 = True
                m100 = True
                m010 = True
                m110 = True
                m001 = True
                m101 = True
                m011 = True
                m111 = True
            else:
                x0c = _grid_sampler_3d_reflect_int_index(
                    x0.to(tl.float32), in_w, align_corners
                )
                y0c = _grid_sampler_3d_reflect_int_index(
                    y0.to(tl.float32), in_h, align_corners
                )
                z0c = _grid_sampler_3d_reflect_int_index(
                    z0.to(tl.float32), in_d, align_corners
                )
                x1c = _grid_sampler_3d_reflect_int_index(
                    x1.to(tl.float32), in_w, align_corners
                )
                y1c = _grid_sampler_3d_reflect_int_index(
                    y1.to(tl.float32), in_h, align_corners
                )
                z1c = _grid_sampler_3d_reflect_int_index(
                    z1.to(tl.float32), in_d, align_corners
                )
                m000 = True
                m100 = True
                m010 = True
                m110 = True
                m001 = True
                m101 = True
                m011 = True
                m111 = True

            base = b.to(tl.int64) * in_strides_n + ch.to(tl.int64) * in_strides_c
            v000 = tl.load(
                input_ptr
                + base
                + z0c.to(tl.int64) * in_strides_d
                + y0c.to(tl.int64) * in_strides_h
                + x0c * in_strides_w,
                mask=mask & m000,
                other=0.0,
            )
            v100 = tl.load(
                input_ptr
                + base
                + z0c.to(tl.int64) * in_strides_d
                + y0c.to(tl.int64) * in_strides_h
                + x1c * in_strides_w,
                mask=mask & m100,
                other=0.0,
            )
            v010 = tl.load(
                input_ptr
                + base
                + z0c.to(tl.int64) * in_strides_d
                + y1c.to(tl.int64) * in_strides_h
                + x0c * in_strides_w,
                mask=mask & m010,
                other=0.0,
            )
            v110 = tl.load(
                input_ptr
                + base
                + z0c.to(tl.int64) * in_strides_d
                + y1c.to(tl.int64) * in_strides_h
                + x1c * in_strides_w,
                mask=mask & m110,
                other=0.0,
            )
            v001 = tl.load(
                input_ptr
                + base
                + z1c.to(tl.int64) * in_strides_d
                + y0c.to(tl.int64) * in_strides_h
                + x0c * in_strides_w,
                mask=mask & m001,
                other=0.0,
            )
            v101 = tl.load(
                input_ptr
                + base
                + z1c.to(tl.int64) * in_strides_d
                + y0c.to(tl.int64) * in_strides_h
                + x1c * in_strides_w,
                mask=mask & m101,
                other=0.0,
            )
            v011 = tl.load(
                input_ptr
                + base
                + z1c.to(tl.int64) * in_strides_d
                + y1c.to(tl.int64) * in_strides_h
                + x0c * in_strides_w,
                mask=mask & m011,
                other=0.0,
            )
            v111 = tl.load(
                input_ptr
                + base
                + z1c.to(tl.int64) * in_strides_d
                + y1c.to(tl.int64) * in_strides_h
                + x1c * in_strides_w,
                mask=mask & m111,
                other=0.0,
            )
            result = (
                v000 * wx0 * wy0 * wz0
                + v100 * wx1 * wy0 * wz0
                + v010 * wx0 * wy1 * wz0
                + v110 * wx1 * wy1 * wz0
                + v001 * wx0 * wy0 * wz1
                + v101 * wx1 * wy0 * wz1
                + v011 * wx0 * wy1 * wz1
                + v111 * wx1 * wy1 * wz1
            )
        else:  # Nearest
            if padding_mode == 2:
                x = _grid_sampler_3d_reflect_coord(x, in_w, align_corners)
                y = _grid_sampler_3d_reflect_coord(y, in_h, align_corners)
                z = _grid_sampler_3d_reflect_coord(z, in_d, align_corners)
            x_floor = tl.floor(x)
            y_floor = tl.floor(y)
            z_floor = tl.floor(z)
            x_frac = x - x_floor
            y_frac = y - y_floor
            z_frac = z - z_floor
            x_floor_int = x_floor.to(tl.int32)
            y_floor_int = y_floor.to(tl.int32)
            z_floor_int = z_floor.to(tl.int32)
            x_is_half = x_frac == 0.5
            y_is_half = y_frac == 0.5
            z_is_half = z_frac == 0.5
            x_up = (x_frac > 0.5) | (x_is_half & ((x_floor_int & 1) == 1))
            y_up = (y_frac > 0.5) | (y_is_half & ((y_floor_int & 1) == 1))
            z_up = (z_frac > 0.5) | (z_is_half & ((z_floor_int & 1) == 1))
            x_near = x_floor_int + x_up.to(tl.int32)
            y_near = y_floor_int + y_up.to(tl.int32)
            z_near = z_floor_int + z_up.to(tl.int32)
            if padding_mode == 0:
                x_in = (x_near >= 0) & (x_near < in_w) & ~grid_nan
                y_in = (y_near >= 0) & (y_near < in_h)
                z_in = (z_near >= 0) & (z_near < in_d)
                x_near = tl.minimum(tl.maximum(x_near, 0), in_w - 1)
                y_near = tl.minimum(tl.maximum(y_near, 0), in_h - 1)
                z_near = tl.minimum(tl.maximum(z_near, 0), in_d - 1)
                pixel_mask = mask & x_in & y_in & z_in
            else:
                x_near = tl.minimum(tl.maximum(x_near, 0), in_w - 1)
                y_near = tl.minimum(tl.maximum(y_near, 0), in_h - 1)
                z_near = tl.minimum(tl.maximum(z_near, 0), in_d - 1)
                pixel_mask = mask
            inp_offset = (
                b.to(tl.int64) * in_strides_n
                + ch.to(tl.int64) * in_strides_c
                + z_near.to(tl.int64) * in_strides_d
                + y_near.to(tl.int64) * in_strides_h
                + x_near * in_strides_w
            )
            result = tl.load(input_ptr + inp_offset, mask=pixel_mask, other=0.0)

        out_offset = (
            b.to(tl.int64) * out_strides_n
            + ch.to(tl.int64) * out_strides_c
            + od.to(tl.int64) * out_strides_d
            + oh.to(tl.int64) * out_strides_h
            + ow * out_strides_w
        )
        tl.store(output_ptr + out_offset, result, mask=mask)


def _is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _log2(v):
    return v.bit_length() - 1


def _max_addr(shape, strides):
    return sum((d - 1) * s for d, s in zip(shape, strides)) + 1


def grid_sampler_3d(
    input,
    grid,
    interpolation_mode=0,
    padding_mode=0,
    align_corners=False,
):
    """Grid sampler 3D with trilinear or nearest interpolation.

    Fast path: each program owns a block of spatial output points; the grid
    coordinates and the 8 trilinear corner offsets are computed once per
    point and a channel loop gathers/scatters every channel for that point
    (channel hoisting — the channel-folded flat form spent most of its
    time recomputing coordinates per (point, channel) pair). With few point
    blocks the channel dim is split across programs to fill the 48 SPs.
    int64 tensors fall back to the flat wide kernel.
    """
    logger.debug("GEMS_ENFLAME GRID_SAMPLER_3D")

    N, C, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = grid.shape[1:4]

    output = torch.empty(
        (N, C, out_d, out_h, out_w),
        dtype=input.dtype,
        device=input.device,
    )

    if output.numel() == 0:
        return output

    # int32 fast-path guards: every flat element offset computed by the
    # kernel (input pixels, grid xyz triplets, output pixels, and the padded
    # block_id * BLOCK + arange tail) must stay below 2^31.
    in_span = _max_addr(input.shape, input.stride())
    grid_span = _max_addr(grid.shape, grid.stride()) + 2 * grid.stride()[-1]
    out_span = _max_addr(output.shape, output.stride())
    total_out = N * C * out_d * out_h * out_w
    total_points = N * out_d * out_h * out_w
    # Block size tiers by point count (sweep 2026-11-18): small grids prefer
    # 512 (more programs via the adaptive channel split), mid grids 1024,
    # and >= 2^17 points 2048 (amortizes the per-iteration coordinate setup;
    # fp16 wants >=1024 here - 512 collapses to 0.55x).
    if total_points <= 2048:
        BLOCK = 512
    elif total_points <= 65536:
        BLOCK = 1024
    else:
        BLOCK = 2048
    num_blocks = (total_points + BLOCK - 1) // BLOCK

    if (
        max(in_span, grid_span, out_span, total_out, total_points, num_blocks * BLOCK)
        < _INT32_MAX
    ):
        if _is_pow2(out_w) and _is_pow2(out_h) and _is_pow2(out_d):
            shift_mode = 1
        else:
            shift_mode = 0
        # Adaptive channel split: fill the 48 SPs when point blocks are few;
        # keep the whole channel loop per program when they are many.
        if num_blocks >= _MAX_PIDS:
            ch_split = C
        else:
            want_groups = min(C, (_MAX_PIDS + num_blocks - 1) // num_blocks)
            ch_split = max(1, (C + want_groups - 1) // want_groups)
        ch_groups = (C + ch_split - 1) // ch_split
        nprog = min(num_blocks * ch_groups, _MAX_PIDS)
        grid_sampler_3d_kernel[(nprog,)](
            input,
            grid,
            output,
            C,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            total_points,
            *input.stride(),
            *grid.stride(),
            *output.stride(),
            interpolation_mode,
            padding_mode,
            align_corners,
            _log2(out_w) if _is_pow2(out_w) else 0,
            _log2(out_h) if _is_pow2(out_h) else 0,
            _log2(out_d) if _is_pow2(out_d) else 0,
            shift_mode,
            BLOCK_SIZE=BLOCK,
            CH_GROUPS=ch_groups,
            CH_SPLIT=ch_split,
            num_warps=4,
        )
    else:
        grid_sampler_3d_kernel_wide[(min((total_out + 255) // 256, _MAX_PIDS),)](
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
            BLOCK_SIZE=256,
        )

    return output
