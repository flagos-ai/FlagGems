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

from flag_gems.ops.grid_sampler_3d_backward import (
    grid_sampler_3d_backward as default_grid_sampler_3d_backward,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _reflect_coord(v):
    # PyTorch reflect_coordinates(v, -2, 2): min=-1, span=2
    # returns (reflected value, derivative sign d(gx)/d(v))
    min_v = -1.0
    span = 2.0
    a = tl.abs(v - min_v)
    fl = tl.floor(a / span)
    extra = a - fl * span
    flips = fl.to(tl.int32)
    even = (flips % 2) == 0
    out = tl.where(even, extra + min_v, span - extra + min_v)
    d = tl.where(v >= min_v, 1.0, -1.0) * tl.where(even, 1.0, -1.0)
    return out, d


@triton.jit
def _nearbyint(v):
    # round-half-to-even
    r = tl.floor(v)
    frac = v - r
    ri = r.to(tl.int32)
    even = (ri % 2) == 0
    res = tl.where(
        frac < 0.5, r, tl.where(frac > 0.5, r + 1.0, tl.where(even, r, r + 1.0))
    )
    return res


@triton.jit
def _corner_pair(ix, size):
    # trilinear corner pair for one dimension.
    i0r_f = tl.floor(ix)
    i0r = i0r_f.to(tl.int32)
    i1r = i0r + 1
    i0c = tl.minimum(tl.maximum(i0r, 0), size - 1)
    i1c = tl.minimum(tl.maximum(i1r, 0), size - 1)
    w1 = ix - i0r_f
    w0 = 1.0 - w1
    v0 = (i0r >= 0) & (i0r < size)
    v1 = (i1r >= 0) & (i1r < size)
    return i0c, i1c, w0, w1, v0, v1


@triton.jit
def _grid_sampler_3d_bwd(
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    grad_input_ptr,
    grad_grid_ptr,
    total,
    C,
    D,
    H,
    W,
    DD,
    DH,
    DW,
    in_sN,
    in_sC,
    in_sD,
    in_sH,
    in_sW,
    go_sN,
    go_sC,
    go_sD,
    go_sH,
    go_sW,
    gr_sN,
    gr_sD,
    gr_sH,
    gr_sW,
    gr_s4,
    align_corners: tl.constexpr,
    interp: tl.constexpr,
    padding: tl.constexpr,
    need_gi: tl.constexpr,
    FULL: tl.constexpr,
    ACCUM_GG: tl.constexpr,
    BP: tl.constexpr,
    BC: tl.constexpr,
):
    # Tile layout: (BC, BP) with the position axis (BP) last so that lanes are
    # contiguous over positions -> coalesced loads and atomics.
    pid_p = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1)
    poff = pid_p * BP + tl.arange(0, BP).to(tl.int64)
    coff = pid_c * BC + tl.arange(0, BC).to(tl.int64)
    S = DD * DH * DW
    m = poff < total
    n = poff // S
    s = poff - n * S
    w = s % DW
    h = (s // DW) % DH
    d = s // (DH * DW)
    mc = coff < C
    m2 = mc[:, None] & m[None, :]

    g_off = n * gr_sN + d * gr_sD + h * gr_sH + w * gr_sW
    x = tl.load(grid_ptr + g_off, mask=m, other=0.0)
    y = tl.load(grid_ptr + g_off + gr_s4, mask=m, other=0.0)
    z = tl.load(grid_ptr + g_off + 2 * gr_s4, mask=m, other=0.0)

    if padding == 0:
        gx = x
        gy = y
        gz = z
        dgx = 1.0
        dgy = 1.0
        dgz = 1.0
    elif padding == 1:
        gx = tl.minimum(tl.maximum(x, -1.0), 1.0)
        gy = tl.minimum(tl.maximum(y, -1.0), 1.0)
        gz = tl.minimum(tl.maximum(z, -1.0), 1.0)
        dgx = tl.where((x > -1.0) & (x < 1.0), 1.0, 0.0)
        dgy = tl.where((y > -1.0) & (y < 1.0), 1.0, 0.0)
        dgz = tl.where((z > -1.0) & (z < 1.0), 1.0, 0.0)
    else:
        gx, dgx = _reflect_coord(x)
        gy, dgy = _reflect_coord(y)
        gz, dgz = _reflect_coord(z)

    if align_corners:
        ix = (gx + 1.0) * 0.5 * (W - 1)
        iy = (gy + 1.0) * 0.5 * (H - 1)
        iz = (gz + 1.0) * 0.5 * (D - 1)
    else:
        ix = (gx + 1.0) * 0.5 * W - 0.5
        iy = (gy + 1.0) * 0.5 * H - 0.5
        iz = (gz + 1.0) * 0.5 * D - 0.5

    if padding == 0:
        cgx = 1.0
        cgy = 1.0
        cgz = 1.0
    else:
        cgx = tl.where((ix > 0.0) & (ix < W - 1), 1.0, 0.0)
        cgy = tl.where((iy > 0.0) & (iy < H - 1), 1.0, 0.0)
        cgz = tl.where((iz > 0.0) & (iz < D - 1), 1.0, 0.0)

    # (BC, BP) channel/position bases
    in_cbase = coff[:, None] * in_sC + n[None, :] * in_sN
    go_cbase = coff[:, None] * go_sC + n[None, :] * go_sN
    go_off = d[None, :] * go_sD + h[None, :] * go_sH + w[None, :] * go_sW

    if interp == 0:
        # ---- trilinear ----
        x0c, x1c, w0x, w1x, vx0, vx1 = _corner_pair(ix, W)
        y0c, y1c, w0y, w1y, vy0, vy1 = _corner_pair(iy, H)
        z0c, z1c, w0z, w1z, vz0, vz1 = _corner_pair(iz, D)

        off_z0 = z0c * in_sD
        off_z1 = z1c * in_sD
        off_y0 = y0c * in_sH
        off_y1 = y1c * in_sH
        off_x0 = x0c * in_sW
        off_x1 = x1c * in_sW

        off_000 = (off_z0 + off_y0 + off_x0)[None, :]
        off_100 = (off_z0 + off_y0 + off_x1)[None, :]
        off_010 = (off_z0 + off_y1 + off_x0)[None, :]
        off_110 = (off_z0 + off_y1 + off_x1)[None, :]
        off_001 = (off_z1 + off_y0 + off_x0)[None, :]
        off_101 = (off_z1 + off_y0 + off_x1)[None, :]
        off_011 = (off_z1 + off_y1 + off_x0)[None, :]
        off_111 = (off_z1 + off_y1 + off_x1)[None, :]

        w000 = (w0x * w0y * w0z)[None, :]
        w100 = (w1x * w0y * w0z)[None, :]
        w010 = (w0x * w1y * w0z)[None, :]
        w110 = (w1x * w1y * w0z)[None, :]
        w001 = (w0x * w0y * w1z)[None, :]
        w101 = (w1x * w0y * w1z)[None, :]
        w011 = (w0x * w1y * w1z)[None, :]
        w111 = (w1x * w1y * w1z)[None, :]

        if FULL:
            go_2d = tl.load(grad_output_ptr + go_cbase + go_off)
            l000 = tl.load(input_ptr + in_cbase + off_000)
            l100 = tl.load(input_ptr + in_cbase + off_100)
            l010 = tl.load(input_ptr + in_cbase + off_010)
            l110 = tl.load(input_ptr + in_cbase + off_110)
            l001 = tl.load(input_ptr + in_cbase + off_001)
            l101 = tl.load(input_ptr + in_cbase + off_101)
            l011 = tl.load(input_ptr + in_cbase + off_011)
            l111 = tl.load(input_ptr + in_cbase + off_111)
        else:
            go_2d = tl.load(grad_output_ptr + go_cbase + go_off, mask=m2, other=0.0)
            l000 = tl.load(input_ptr + in_cbase + off_000, mask=m2, other=0.0)
            l100 = tl.load(input_ptr + in_cbase + off_100, mask=m2, other=0.0)
            l010 = tl.load(input_ptr + in_cbase + off_010, mask=m2, other=0.0)
            l110 = tl.load(input_ptr + in_cbase + off_110, mask=m2, other=0.0)
            l001 = tl.load(input_ptr + in_cbase + off_001, mask=m2, other=0.0)
            l101 = tl.load(input_ptr + in_cbase + off_101, mask=m2, other=0.0)
            l011 = tl.load(input_ptr + in_cbase + off_011, mask=m2, other=0.0)
            l111 = tl.load(input_ptr + in_cbase + off_111, mask=m2, other=0.0)

        if padding == 0:
            v000 = tl.where((vx0 & vy0 & vz0)[None, :], l000, 0.0)
            v100 = tl.where((vx1 & vy0 & vz0)[None, :], l100, 0.0)
            v010 = tl.where((vx0 & vy1 & vz0)[None, :], l010, 0.0)
            v110 = tl.where((vx1 & vy1 & vz0)[None, :], l110, 0.0)
            v001 = tl.where((vx0 & vy0 & vz1)[None, :], l001, 0.0)
            v101 = tl.where((vx1 & vy0 & vz1)[None, :], l101, 0.0)
            v011 = tl.where((vx0 & vy1 & vz1)[None, :], l011, 0.0)
            v111 = tl.where((vx1 & vy1 & vz1)[None, :], l111, 0.0)
        else:
            v000 = l000
            v100 = l100
            v010 = l010
            v110 = l110
            v001 = l001
            v101 = l101
            v011 = l011
            v111 = l111

        if need_gi:
            if FULL:
                # All position/channel lanes are valid; invalid zero-padding
                # corners contribute 0.0 through weight-zeroing, and clamped
                # corner indices keep every address in bounds, so the atomic
                # needs no mask at all.
                if padding == 0:
                    a000 = (w0x * w0y * w0z * (vx0 & vy0 & vz0))[None, :]
                    a100 = (w1x * w0y * w0z * (vx1 & vy0 & vz0))[None, :]
                    a010 = (w0x * w1y * w0z * (vx0 & vy1 & vz0))[None, :]
                    a110 = (w1x * w1y * w0z * (vx1 & vy1 & vz0))[None, :]
                    a001 = (w0x * w0y * w1z * (vx0 & vy0 & vz1))[None, :]
                    a101 = (w1x * w0y * w1z * (vx1 & vy0 & vz1))[None, :]
                    a011 = (w0x * w1y * w1z * (vx0 & vy1 & vz1))[None, :]
                    a111 = (w1x * w1y * w1z * (vx1 & vy1 & vz1))[None, :]
                else:
                    a000 = w000
                    a100 = w100
                    a010 = w010
                    a110 = w110
                    a001 = w001
                    a101 = w101
                    a011 = w011
                    a111 = w111
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_000, go_2d * a000, sem="relaxed"
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_100, go_2d * a100, sem="relaxed"
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_010, go_2d * a010, sem="relaxed"
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_110, go_2d * a110, sem="relaxed"
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_001, go_2d * a001, sem="relaxed"
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_101, go_2d * a101, sem="relaxed"
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_011, go_2d * a011, sem="relaxed"
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_111, go_2d * a111, sem="relaxed"
                )
            elif padding == 0:
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_000,
                    go_2d * w000,
                    mask=m2 & (vx0 & vy0 & vz0)[None, :],
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_100,
                    go_2d * w100,
                    mask=m2 & (vx1 & vy0 & vz0)[None, :],
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_010,
                    go_2d * w010,
                    mask=m2 & (vx0 & vy1 & vz0)[None, :],
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_110,
                    go_2d * w110,
                    mask=m2 & (vx1 & vy1 & vz0)[None, :],
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_001,
                    go_2d * w001,
                    mask=m2 & (vx0 & vy0 & vz1)[None, :],
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_101,
                    go_2d * w101,
                    mask=m2 & (vx1 & vy0 & vz1)[None, :],
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_011,
                    go_2d * w011,
                    mask=m2 & (vx0 & vy1 & vz1)[None, :],
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_111,
                    go_2d * w111,
                    mask=m2 & (vx1 & vy1 & vz1)[None, :],
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_000,
                    go_2d * w000,
                    mask=m2,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_100,
                    go_2d * w100,
                    mask=m2,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_010,
                    go_2d * w010,
                    mask=m2,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_110,
                    go_2d * w110,
                    mask=m2,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_001,
                    go_2d * w001,
                    mask=m2,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_101,
                    go_2d * w101,
                    mask=m2,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_011,
                    go_2d * w011,
                    mask=m2,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_input_ptr + in_cbase + off_111,
                    go_2d * w111,
                    mask=m2,
                    sem="relaxed",
                )

        w0y_w0z = (w0y * w0z)[None, :]
        w1y_w0z = (w1y * w0z)[None, :]
        w0y_w1z = (w0y * w1z)[None, :]
        w1y_w1z = (w1y * w1z)[None, :]
        w0x_w0z = (w0x * w0z)[None, :]
        w1x_w0z = (w1x * w0z)[None, :]
        w0x_w1z = (w0x * w1z)[None, :]
        w1x_w1z = (w1x * w1z)[None, :]
        w0x_w0y = (w0x * w0y)[None, :]
        w1x_w0y = (w1x * w0y)[None, :]
        w0x_w1y = (w0x * w1y)[None, :]
        w1x_w1y = (w1x * w1y)[None, :]

        gix_2d = go_2d * (
            w0y_w0z * (v100 - v000)
            + w1y_w0z * (v110 - v010)
            + w0y_w1z * (v101 - v001)
            + w1y_w1z * (v111 - v011)
        )
        giy_2d = go_2d * (
            w0x_w0z * (v010 - v000)
            + w1x_w0z * (v110 - v100)
            + w0x_w1z * (v011 - v001)
            + w1x_w1z * (v111 - v101)
        )
        giz_2d = go_2d * (
            w0x_w0y * (v001 - v000)
            + w1x_w0y * (v101 - v100)
            + w0x_w1y * (v011 - v010)
            + w1x_w1y * (v111 - v110)
        )
        gix = tl.sum(gix_2d, axis=0)
        giy = tl.sum(giy_2d, axis=0)
        giz = tl.sum(giz_2d, axis=0)

        if align_corners:
            d0 = (W - 1) * 0.5
            d1 = (H - 1) * 0.5
            d2 = (D - 1) * 0.5
        else:
            d0 = W * 0.5
            d1 = H * 0.5
            d2 = D * 0.5
        if ACCUM_GG:
            tl.atomic_add(
                grad_grid_ptr + g_off, gix * d0 * dgx * cgx, mask=m, sem="relaxed"
            )
            tl.atomic_add(
                grad_grid_ptr + g_off + gr_s4,
                giy * d1 * dgy * cgy,
                mask=m,
                sem="relaxed",
            )
            tl.atomic_add(
                grad_grid_ptr + g_off + 2 * gr_s4,
                giz * d2 * dgz * cgz,
                mask=m,
                sem="relaxed",
            )
        else:
            tl.store(grad_grid_ptr + g_off, gix * d0 * dgx * cgx, mask=m)
            tl.store(grad_grid_ptr + g_off + gr_s4, giy * d1 * dgy * cgy, mask=m)
            tl.store(grad_grid_ptr + g_off + 2 * gr_s4, giz * d2 * dgz * cgz, mask=m)
    else:
        # ---- nearest ----
        rxi_f = _nearbyint(ix)
        ryi_f = _nearbyint(iy)
        rzi_f = _nearbyint(iz)
        rxi = rxi_f.to(tl.int32)
        ryi = ryi_f.to(tl.int32)
        rzi = rzi_f.to(tl.int32)
        xc = tl.minimum(tl.maximum(rxi, 0), W - 1)
        yc = tl.minimum(tl.maximum(ryi, 0), H - 1)
        zc = tl.minimum(tl.maximum(rzi, 0), D - 1)
        okx = (rxi >= 0) & (rxi < W)
        oky = (ryi >= 0) & (ryi < H)
        okz = (rzi >= 0) & (rzi < D)
        gi_off = (zc * in_sD + yc * in_sH + xc * in_sW)[None, :]
        if FULL:
            go_2d = tl.load(grad_output_ptr + go_cbase + go_off)
        else:
            go_2d = tl.load(grad_output_ptr + go_cbase + go_off, mask=m2, other=0.0)
        if need_gi:
            if FULL:
                if padding == 0:
                    gv = go_2d * (okx & oky & okz)[None, :]
                else:
                    gv = go_2d
                tl.atomic_add(grad_input_ptr + in_cbase + gi_off, gv, sem="relaxed")
            elif padding == 0:
                wmask = m2 & (okx & oky & okz)[None, :]
                tl.atomic_add(
                    grad_input_ptr + in_cbase + gi_off, go_2d, mask=wmask, sem="relaxed"
                )
            else:
                tl.atomic_add(
                    grad_input_ptr + in_cbase + gi_off, go_2d, mask=m2, sem="relaxed"
                )
        z0 = x * 0.0
        tl.store(grad_grid_ptr + g_off, z0, mask=m)
        tl.store(grad_grid_ptr + g_off + gr_s4, z0, mask=m)
        tl.store(grad_grid_ptr + g_off + 2 * gr_s4, z0, mask=m)


def _pick_tiles(C, total, interp):
    if interp == 0:
        if C <= 4:
            return 32, 4, 4
        else:
            return 16, 16, 8
    else:
        return 64, 8, 4


def _specialized_grid_sampler_3d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    im = int(interpolation_mode)
    pm = int(padding_mode)
    ac = bool(align_corners)
    if isinstance(output_mask, (list, tuple)):
        need_gi = bool(output_mask[0])
    else:
        need_gi = bool(output_mask)

    N, C, D, H, W = input.shape
    ND, CD, DD, DH, DW = grad_output.shape
    N3, DD3, DH3, DW3, _ = grid.shape

    total = N * DD * DH * DW
    grad_input = torch.zeros_like(input)
    BP, BC, nw = _pick_tiles(C, total, im)
    accum_gg = (im == 0) and (triton.cdiv(C, BC) > 1)
    grad_grid = torch.zeros_like(grid) if accum_gg else torch.empty_like(grid)
    if total == 0:
        return grad_input, grad_grid

    in_sN, in_sC, in_sD, in_sH, in_sW = input.stride()
    go_sN, go_sC, go_sD, go_sH, go_sW = grad_output.stride()
    gr_sN, gr_sD, gr_sH, gr_sW, gr_s4 = grid.stride()

    full = (total % BP == 0) and (C % BC == 0)
    grid_cfg = (triton.cdiv(total, BP), triton.cdiv(C, BC))
    _grid_sampler_3d_bwd[grid_cfg](
        grad_output,
        input,
        grid,
        grad_input,
        grad_grid,
        total,
        C,
        D,
        H,
        W,
        DD,
        DH,
        DW,
        in_sN,
        in_sC,
        in_sD,
        in_sH,
        in_sW,
        go_sN,
        go_sC,
        go_sD,
        go_sH,
        go_sW,
        gr_sN,
        gr_sD,
        gr_sH,
        gr_sW,
        gr_s4,
        align_corners=ac,
        interp=im,
        padding=pm,
        need_gi=need_gi,
        FULL=full,
        ACCUM_GG=accum_gg,
        BP=BP,
        BC=BC,
        num_warps=nw,
    )
    return grad_input, grad_grid


def grid_sampler_3d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    logger.debug("GEMS_MTHREADS GRID_SAMPLER_3D_BACKWARD")
    if (
        isinstance(grad_output, torch.Tensor)
        and grad_output.device.type == "musa"
        and grad_output.dtype in _SUPPORTED_DTYPES
        and isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
        and isinstance(grid, torch.Tensor)
        and grid.device.type == "musa"
        and grid.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_grid_sampler_3d_backward(
            grad_output,
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            output_mask,
        )
    return default_grid_sampler_3d_backward(
        grad_output,
        input,
        grid,
        interpolation_mode,
        padding_mode,
        align_corners,
        output_mask,
    )
