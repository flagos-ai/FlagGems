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

logger = logging.getLogger(__name__)


@triton.jit
def _rint_half_even(x):
    """Round to nearest, ties to even (mirrors std::nearbyint)."""
    r = tl.floor(x)
    d = x - r
    odd = (r - 2.0 * tl.floor(r * 0.5)) > 0.5
    return tl.where((d > 0.5) | ((d == 0.5) & odd), r + 1.0, r)


_CONFIGS = [
    triton.Config({"BLOCK_S": 64, "BLOCK_C": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_S": 128, "BLOCK_C": 1}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_S": 256, "BLOCK_C": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_S": 512, "BLOCK_C": 1}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_S": 128, "BLOCK_C": 2}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_S": 256, "BLOCK_C": 2}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_S": 128, "BLOCK_C": 4}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_S": 64, "BLOCK_C": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_S": 256, "BLOCK_C": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_S": 128, "BLOCK_C": 8}, num_warps=4, num_stages=1),
]


@triton.autotune(configs=_CONFIGS, key=["C", "N", "OD", "OH", "OW"])
@triton.jit
def _gs3d_kernel(
    inp_ptr,
    grid_ptr,
    out_ptr,
    C,
    ID,
    IH,
    IW,
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    g_n,
    g_d,
    g_h,
    g_w,
    g_xyz,
    o_n,
    o_c,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    INTERP: tl.constexpr,
    PAD: tl.constexpr,
    ALIGN: tl.constexpr,
    IS_F64: tl.constexpr,
):
    ACC: tl.constexpr = tl.float64 if IS_F64 else tl.float32
    S_out: tl.constexpr = OD * OH * OW

    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    n = tl.program_id(2)

    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = s < S_out

    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cs < C

    # flat output spatial index -> (do, ho, wo)
    wo = s % OW
    t = s // OW
    ho = t % OH
    do = t // OH

    s64 = s.to(tl.int64)
    n64 = n.to(tl.int64)
    c64 = cs[:, None].to(tl.int64)

    g_off = (
        n64 * g_n
        + do.to(tl.int64) * g_d
        + ho.to(tl.int64) * g_h
        + wo.to(tl.int64) * g_w
    )
    gx = tl.load(grid_ptr + g_off, mask=smask, other=0.0).to(ACC)
    gy = tl.load(grid_ptr + g_off + g_xyz, mask=smask, other=0.0).to(ACC)
    gz = tl.load(grid_ptr + g_off + 2 * g_xyz, mask=smask, other=0.0).to(ACC)

    if ALIGN:
        x = ((gx + 1.0) * 0.5) * (IW.to(ACC) - 1.0)
        y = ((gy + 1.0) * 0.5) * (IH.to(ACC) - 1.0)
        z = ((gz + 1.0) * 0.5) * (ID.to(ACC) - 1.0)
    else:
        x = ((gx + 1.0) * IW.to(ACC) - 1.0) * 0.5
        y = ((gy + 1.0) * IH.to(ACC) - 1.0) * 0.5
        z = ((gz + 1.0) * ID.to(ACC) - 1.0) * 0.5

    if PAD == 0:
        x0 = x
        y0 = y
        z0 = z
    elif PAD == 1:
        x0 = tl.minimum(tl.maximum(x, 0.0), IW.to(ACC) - 1.0)
        y0 = tl.minimum(tl.maximum(y, 0.0), IH.to(ACC) - 1.0)
        z0 = tl.minimum(tl.maximum(z, 0.0), ID.to(ACC) - 1.0)
    else:
        # reflection about the unnormalized coordinate, ATen convention
        if ALIGN:
            lx = 0.0
            hx = 2.0 * (IW.to(ACC) - 1.0)
            ly = 0.0
            hy = 2.0 * (IH.to(ACC) - 1.0)
            lz = 0.0
            hz = 2.0 * (ID.to(ACC) - 1.0)
        else:
            lx = -1.0
            hx = 2.0 * IW.to(ACC) - 1.0
            ly = -1.0
            hy = 2.0 * IH.to(ACC) - 1.0
            lz = -1.0
            hz = 2.0 * ID.to(ACC) - 1.0

        sx = (hx - lx) * 0.5
        tx = tl.abs(x - lx * 0.5)
        ex = tx % sx
        flx = tl.floor(tx / sx)
        evx = (flx - 2.0 * tl.floor(flx * 0.5)) < 0.5
        xr = tl.where(evx, ex + lx * 0.5, (sx - ex) + lx * 0.5)
        x0 = tl.minimum(tl.maximum(tl.where(sx == 0.0, 0.0, xr), 0.0), IW.to(ACC) - 1.0)

        sy = (hy - ly) * 0.5
        ty = tl.abs(y - ly * 0.5)
        ey = ty % sy
        fly = tl.floor(ty / sy)
        evy = (fly - 2.0 * tl.floor(fly * 0.5)) < 0.5
        yr = tl.where(evy, ey + ly * 0.5, (sy - ey) + ly * 0.5)
        y0 = tl.minimum(tl.maximum(tl.where(sy == 0.0, 0.0, yr), 0.0), IH.to(ACC) - 1.0)

        sz = (hz - lz) * 0.5
        tz = tl.abs(z - lz * 0.5)
        ez = tz % sz
        flz = tl.floor(tz / sz)
        evz = (flz - 2.0 * tl.floor(flz * 0.5)) < 0.5
        zr = tl.where(evz, ez + lz * 0.5, (sz - ez) + lz * 0.5)
        z0 = tl.minimum(tl.maximum(tl.where(sz == 0.0, 0.0, zr), 0.0), ID.to(ACC) - 1.0)

    out_base = n64 * o_n + c64 * o_c + s64[None, :]
    out_mask = cmask[:, None] & smask[None, :]

    if INTERP == 1:
        ix = _rint_half_even(x0).to(tl.int32)
        iy = _rint_half_even(y0).to(tl.int32)
        iz = _rint_half_even(z0).to(tl.int32)
        valid = (
            smask
            & (ix >= 0)
            & (ix < IW)
            & (iy >= 0)
            & (iy < IH)
            & (iz >= 0)
            & (iz < ID)
        )
        off = (
            n64 * in_n
            + iz.to(tl.int64) * in_d
            + iy.to(tl.int64) * in_h
            + ix.to(tl.int64) * in_w
        )
        ptrs = inp_ptr + (off[None, :] + c64 * in_c)
        v = tl.load(ptrs, mask=cmask[:, None] & valid[None, :], other=0.0).to(ACC)
        tl.store(out_ptr + out_base, v, mask=out_mask)
    else:
        fz0 = tl.floor(z0)
        fy0 = tl.floor(y0)
        fx0 = tl.floor(x0)
        iz0 = fz0.to(tl.int32)
        iy0 = fy0.to(tl.int32)
        ix0 = fx0.to(tl.int32)
        iz1 = iz0 + 1
        iy1 = iy0 + 1
        ix1 = ix0 + 1
        wz1 = z0 - fz0
        wy1 = y0 - fy0
        wx1 = x0 - fx0
        wz0 = 1.0 - wz1
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1

        mz0 = (iz0 >= 0) & (iz0 < ID)
        mz1 = (iz1 >= 0) & (iz1 < ID)
        my0 = (iy0 >= 0) & (iy0 < IH)
        my1 = (iy1 >= 0) & (iy1 < IH)
        mx0 = (ix0 >= 0) & (ix0 < IW)
        mx1 = (ix1 >= 0) & (ix1 < IW)

        bz0 = (n64 * in_n + iz0.to(tl.int64) * in_d)[None, :] + c64 * in_c
        bz1 = (n64 * in_n + iz1.to(tl.int64) * in_d)[None, :] + c64 * in_c
        oy0 = (iy0.to(tl.int64) * in_h)[None, :]
        oy1 = (iy1.to(tl.int64) * in_h)[None, :]
        ox0 = (ix0.to(tl.int64) * in_w)[None, :]
        ox1 = (ix1.to(tl.int64) * in_w)[None, :]

        cm = cmask[:, None]
        m000 = cm & (mz0 & my0 & mx0)[None, :]
        m001 = cm & (mz0 & my0 & mx1)[None, :]
        m010 = cm & (mz0 & my1 & mx0)[None, :]
        m011 = cm & (mz0 & my1 & mx1)[None, :]
        m100 = cm & (mz1 & my0 & mx0)[None, :]
        m101 = cm & (mz1 & my0 & mx1)[None, :]
        m110 = cm & (mz1 & my1 & mx0)[None, :]
        m111 = cm & (mz1 & my1 & mx1)[None, :]

        p000 = inp_ptr + (bz0 + oy0 + ox0)
        p001 = inp_ptr + (bz0 + oy0 + ox1)
        p010 = inp_ptr + (bz0 + oy1 + ox0)
        p011 = inp_ptr + (bz0 + oy1 + ox1)
        p100 = inp_ptr + (bz1 + oy0 + ox0)
        p101 = inp_ptr + (bz1 + oy0 + ox1)
        p110 = inp_ptr + (bz1 + oy1 + ox0)
        p111 = inp_ptr + (bz1 + oy1 + ox1)

        v000 = tl.load(p000, mask=m000, other=0.0).to(ACC)
        v001 = tl.load(p001, mask=m001, other=0.0).to(ACC)
        v010 = tl.load(p010, mask=m010, other=0.0).to(ACC)
        v011 = tl.load(p011, mask=m011, other=0.0).to(ACC)
        v100 = tl.load(p100, mask=m100, other=0.0).to(ACC)
        v101 = tl.load(p101, mask=m101, other=0.0).to(ACC)
        v110 = tl.load(p110, mask=m110, other=0.0).to(ACC)
        v111 = tl.load(p111, mask=m111, other=0.0).to(ACC)

        wz0b = wz0[None, :]
        wz1b = wz1[None, :]
        wy0b = wy0[None, :]
        wy1b = wy1[None, :]
        wx0b = wx0[None, :]
        wx1b = wx1[None, :]

        v00 = v000 * wz0b + v100 * wz1b
        v01 = v001 * wz0b + v101 * wz1b
        v10 = v010 * wz0b + v110 * wz1b
        v11 = v011 * wz0b + v111 * wz1b
        v0 = v00 * wy0b + v10 * wy1b
        v1 = v01 * wy0b + v11 * wy1b
        val = v0 * wx0b + v1 * wx1b
        tl.store(out_ptr + out_base, val, mask=out_mask)


def grid_sampler_3d(
    input, grid, interpolation_mode=0, padding_mode=0, align_corners=False
):
    logger.debug("GEMS_HYGON GRID_SAMPLER_3D")
    N, C, ID, IH, IW = input.shape
    OD = grid.shape[1]
    OH = grid.shape[2]
    OW = grid.shape[3]

    if interpolation_mode > 1:
        raise RuntimeError(
            "bicubic interpolation is only supported for 2D grid sampling"
        )
    if padding_mode < 0 or padding_mode > 2:
        raise RuntimeError("padding_mode needs to be between 0 and 2, inclusive")

    out = torch.empty((N, C, OD, OH, OW), dtype=input.dtype, device=input.device)

    S_out = OD * OH * OW
    if N * C * S_out == 0:
        return out

    in_n, in_c, in_d, in_h, in_w = input.stride()
    g_n, g_d, g_h, g_w, g_xyz = grid.stride()
    o_n = C * S_out
    o_c = S_out

    def _grid(meta):
        return (triton.cdiv(S_out, meta["BLOCK_S"]), triton.cdiv(C, meta["BLOCK_C"]), N)

    _gs3d_kernel[_grid](
        input,
        grid,
        out,
        C,
        ID,
        IH,
        IW,
        in_n,
        in_c,
        in_d,
        in_h,
        in_w,
        g_n,
        g_d,
        g_h,
        g_w,
        g_xyz,
        o_n,
        o_c,
        OD=OD,
        OH=OH,
        OW=OW,
        INTERP=interpolation_mode,
        PAD=padding_mode,
        ALIGN=align_corners,
        IS_F64=input.dtype == torch.float64,
    )
    return out
