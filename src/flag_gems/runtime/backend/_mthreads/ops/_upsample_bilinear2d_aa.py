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

from flag_gems.ops._upsample_bilinear2d_aa import (
    _upsample_bilinear2d_aa as default__upsample_bilinear2d_aa,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ---------------------------------------------------------------------------
# upsample_bilinear2d_aa (antialias=True, bilinear mode)
#
# Semantics: replicate the flag_gems reference kernel exactly (bitwise).
# General kernel: per-output 2-tap bilinear weights computed with the exact
# flag_gems fp32 op order; 4 clamped gather loads per (BH,BW) tile.
# 2x fast path (rs_h == rs_w == 0.5, OH == 2*IH, OW == 2*IW): output-pair
# formulation with a 9-element input stencil per pair; weights are exactly
# (0.25, 0.75) / (0.75, 0.25) / boundary (1, 0), so the result is bitwise
# identical to the general path with far fewer gather lanes.
# ---------------------------------------------------------------------------


@triton.jit
def _upsample_aa_kernel(
    out_ptr,
    in_ptr,
    NC,
    OH,
    OW,
    IH,
    IW,
    rs_h,
    rs_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w = tl.cdiv(OW, BLOCK_W)
    num_h = tl.cdiv(OH, BLOCK_H)
    tiles_per_nc = num_w * num_h
    pid_nc = pid // tiles_per_nc
    rem = pid % tiles_per_nc
    pid_h = rem // num_w
    pid_w = rem % num_w

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # ---------------- width weights ----------------
    cw = (ow + 0.5) * rs_w
    s0w = tl.maximum(cw - 1.0 + 0.5, 0.0).to(tl.int32)
    sw = (tl.minimum(cw + 1.0 + 0.5, IW) - s0w).to(tl.int32)
    smc_w = s0w - cw
    wx0 = 1.0 - tl.abs(smc_w + 0.5)
    wx1 = 1.0 - tl.abs(smc_w + 1.5)
    wx0 = tl.where(0 < sw, tl.maximum(wx0, 0.0), 0.0)
    wx1 = tl.where(1 < sw, tl.maximum(wx1, 0.0), 0.0)
    wxt = wx0 + wx1
    wxt = tl.where(wxt != 0.0, wxt, 1.0)
    wx0 = wx0 / wxt
    wx1 = wx1 / wxt

    # ---------------- height weights ----------------
    ch = (oh + 0.5) * rs_h
    s0h = tl.maximum(ch - 1.0 + 0.5, 0.0).to(tl.int32)
    sh = (tl.minimum(ch + 1.0 + 0.5, IH) - s0h).to(tl.int32)
    smc_h = s0h - ch
    wy0 = 1.0 - tl.abs(smc_h + 0.5)
    wy1 = 1.0 - tl.abs(smc_h + 1.5)
    wy0 = tl.where(0 < sh, tl.maximum(wy0, 0.0), 0.0)
    wy1 = tl.where(1 < sh, tl.maximum(wy1, 0.0), 0.0)
    wyt = wy0 + wy1
    wyt = tl.where(wyt != 0.0, wyt, 1.0)
    wy0 = wy0 / wyt
    wy1 = wy1 / wyt

    # ---------------- clamped gather indices ----------------
    i0w = tl.minimum(s0w, IW - 1)
    i1w = tl.minimum(s0w + 1, IW - 1)
    i0h = tl.minimum(s0h, IH - 1)
    i1h = tl.minimum(s0h + 1, IH - 1)

    base = pid_nc.to(tl.int64) * (IH * IW)
    row0p = i0h[:, None].to(tl.int64) * IW
    row1p = i1h[:, None].to(tl.int64) * IW
    col0p = i0w[None, :].to(tl.int64)
    col1p = i1w[None, :].to(tl.int64)

    d00 = tl.load(in_ptr + base + row0p + col0p)
    d01 = tl.load(in_ptr + base + row0p + col1p)
    d10 = tl.load(in_ptr + base + row1p + col0p)
    d11 = tl.load(in_ptr + base + row1p + col1p)

    row0 = d00 * wx0[None, :] + d01 * wx1[None, :]
    row1 = d10 * wx0[None, :] + d11 * wx1[None, :]
    res = row0 * wy0[:, None] + row1 * wy1[:, None]

    omask = (oh[:, None] < OH) & (ow[None, :] < OW)
    out_off = (
        pid_nc.to(tl.int64) * (OH * OW)
        + oh[:, None].to(tl.int64) * OW
        + ow[None, :].to(tl.int64)
    )
    tl.store(out_ptr + out_off, res, mask=omask)


@triton.jit
def _upsample_aa_2x_kernel(
    out_ptr,
    in_ptr,
    NC,
    K,
    L,
    IH,
    IW,
    OH,
    OW,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
    NEED_F32_CAST: tl.constexpr,
):
    pid = tl.program_id(0)
    num_l = tl.cdiv(L, BLOCK_L)
    num_k = tl.cdiv(K, BLOCK_K)
    tpn = num_l * num_k
    pid_nc = pid // tpn
    rem = pid % tpn
    pid_k = rem // num_l
    pid_l = rem % num_l

    kk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    ll = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    # clamp load indices (stores are masked by kk < K / col < OW)
    kc = tl.minimum(kk, K - 1)
    lc = tl.minimum(ll, L - 1)
    kp = tl.minimum(kc + 1, IH - 1)
    km = tl.maximum(kc - 1, 0)
    lp = tl.minimum(lc + 1, IW - 1)
    lm = tl.maximum(lc - 1, 0)

    base = pid_nc.to(tl.int64) * (IH * IW)
    rowm = km[:, None].to(tl.int64) * IW
    row0 = kc[:, None].to(tl.int64) * IW
    rowp = kp[:, None].to(tl.int64) * IW
    colm = lm[None, :].to(tl.int64)
    col0 = lc[None, :].to(tl.int64)
    colp = lp[None, :].to(tl.int64)

    if NEED_F32_CAST:
        Gmm = tl.load(in_ptr + base + rowm + colm).to(tl.float32)
        Gm0 = tl.load(in_ptr + base + rowm + col0).to(tl.float32)
        Gmp = tl.load(in_ptr + base + rowm + colp).to(tl.float32)
        G0m = tl.load(in_ptr + base + row0 + colm).to(tl.float32)
        G00 = tl.load(in_ptr + base + row0 + col0).to(tl.float32)
        G0p = tl.load(in_ptr + base + row0 + colp).to(tl.float32)
        Gpm = tl.load(in_ptr + base + rowp + colm).to(tl.float32)
        Gp0 = tl.load(in_ptr + base + rowp + col0).to(tl.float32)
        Gpp = tl.load(in_ptr + base + rowp + colp).to(tl.float32)
    else:
        Gmm = tl.load(in_ptr + base + rowm + colm)
        Gm0 = tl.load(in_ptr + base + rowm + col0)
        Gmp = tl.load(in_ptr + base + rowm + colp)
        G0m = tl.load(in_ptr + base + row0 + colm)
        G00 = tl.load(in_ptr + base + row0 + col0)
        G0p = tl.load(in_ptr + base + row0 + colp)
        Gpm = tl.load(in_ptr + base + rowp + colm)
        Gp0 = tl.load(in_ptr + base + rowp + col0)
        Gpp = tl.load(in_ptr + base + rowp + colp)

    l_first = (ll == 0)[None, :].broadcast_to(BLOCK_K, BLOCK_L)
    l_last = (ll == L - 1)[None, :].broadcast_to(BLOCK_K, BLOCK_L)
    We_m = tl.where(l_first, Gm0, 0.25 * Gmm + 0.75 * Gm0)
    We_0 = tl.where(l_first, G00, 0.25 * G0m + 0.75 * G00)
    We_p = tl.where(l_first, Gp0, 0.25 * Gpm + 0.75 * Gp0)
    Wo_m = tl.where(l_last, Gm0, 0.75 * Gm0 + 0.25 * Gmp)
    Wo_0 = tl.where(l_last, G00, 0.75 * G00 + 0.25 * G0p)
    Wo_p = tl.where(l_last, Gp0, 0.75 * Gp0 + 0.25 * Gpp)

    k_first = (kk == 0)[:, None].broadcast_to(BLOCK_K, BLOCK_L)
    k_last = (kk == K - 1)[:, None].broadcast_to(BLOCK_K, BLOCK_L)
    EE = tl.where(k_first, We_0, 0.25 * We_m + 0.75 * We_0)
    EO = tl.where(k_first, Wo_0, 0.25 * Wo_m + 0.75 * Wo_0)
    OE = tl.where(k_last, We_0, 0.75 * We_0 + 0.25 * We_p)
    OO = tl.where(k_last, Wo_0, 0.75 * Wo_0 + 0.25 * Wo_p)

    even_row = tl.interleave(EE, EO)
    odd_row = tl.interleave(OE, OO)

    obase = pid_nc.to(tl.int64) * (OH * OW) + (2 * kk)[:, None].to(tl.int64) * OW
    ocols = 2 * pid_l * BLOCK_L + tl.arange(0, 2 * BLOCK_L)
    ocol = ocols[None, :].to(tl.int64)
    kmask = (kk[:, None] < K) & (ocols[None, :] < OW)
    tl.store(out_ptr + obase + ocol, even_row, mask=kmask)
    tl.store(out_ptr + obase + OW + ocol, odd_row, mask=kmask)


@triton.jit
def _upsample_aa_2x_fast_kernel(
    out_ptr,
    in_ptr,
    NC,
    K,
    L,
    IH,
    IW,
    OH,
    OW,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # fp16/bf16 fast variant: interior blocks (the overwhelming majority on
    # large shapes) skip all boundary selects/clamps/masks; edge blocks fall
    # back to the exact clamped path so boundary semantics are preserved.
    pid = tl.program_id(0)
    num_l = tl.cdiv(L, BLOCK_L)
    num_k = tl.cdiv(K, BLOCK_K)
    tpn = num_l * num_k
    pid_nc = pid // tpn
    rem = pid % tpn
    pid_k = rem // num_l
    pid_l = rem % num_l

    kk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    ll = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    base = pid_nc.to(tl.int64) * (IH * IW)

    interior = (
        (pid_k * BLOCK_K >= 1)
        & (pid_k * BLOCK_K + BLOCK_K - 1 <= K - 2)
        & (pid_l * BLOCK_L >= 1)
        & (pid_l * BLOCK_L + BLOCK_L - 1 <= L - 2)
    )

    if interior:
        rowm = (kk - 1)[:, None].to(tl.int64) * IW
        row0 = kk[:, None].to(tl.int64) * IW
        rowp = (kk + 1)[:, None].to(tl.int64) * IW
        colm = (ll - 1)[None, :].to(tl.int64)
        col0 = ll[None, :].to(tl.int64)
        colp = (ll + 1)[None, :].to(tl.int64)
        Gmm = tl.load(in_ptr + base + rowm + colm).to(tl.float32)
        Gm0 = tl.load(in_ptr + base + rowm + col0).to(tl.float32)
        Gmp = tl.load(in_ptr + base + rowm + colp).to(tl.float32)
        G0m = tl.load(in_ptr + base + row0 + colm).to(tl.float32)
        G00 = tl.load(in_ptr + base + row0 + col0).to(tl.float32)
        G0p = tl.load(in_ptr + base + row0 + colp).to(tl.float32)
        Gpm = tl.load(in_ptr + base + rowp + colm).to(tl.float32)
        Gp0 = tl.load(in_ptr + base + rowp + col0).to(tl.float32)
        Gpp = tl.load(in_ptr + base + rowp + colp).to(tl.float32)
        We_m = 0.25 * Gmm + 0.75 * Gm0
        We_0 = 0.25 * G0m + 0.75 * G00
        We_p = 0.25 * Gpm + 0.75 * Gp0
        Wo_m = 0.75 * Gm0 + 0.25 * Gmp
        Wo_0 = 0.75 * G00 + 0.25 * G0p
        Wo_p = 0.75 * Gp0 + 0.25 * Gpp
        EE = 0.25 * We_m + 0.75 * We_0
        EO = 0.25 * Wo_m + 0.75 * Wo_0
        OE = 0.75 * We_0 + 0.25 * We_p
        OO = 0.75 * Wo_0 + 0.25 * Wo_p
        even_row = tl.interleave(EE, EO)
        odd_row = tl.interleave(OE, OO)
        obase = pid_nc.to(tl.int64) * (OH * OW) + (2 * kk)[:, None].to(tl.int64) * OW
        ocol = (2 * pid_l * BLOCK_L + tl.arange(0, 2 * BLOCK_L))[None, :].to(tl.int64)
        tl.store(out_ptr + obase + ocol, even_row)
        tl.store(out_ptr + obase + OW + ocol, odd_row)
    else:
        kc = tl.minimum(kk, K - 1)
        lc = tl.minimum(ll, L - 1)
        kp = tl.minimum(kc + 1, IH - 1)
        km = tl.maximum(kc - 1, 0)
        lp = tl.minimum(lc + 1, IW - 1)
        lm = tl.maximum(lc - 1, 0)
        rowm = km[:, None].to(tl.int64) * IW
        row0 = kc[:, None].to(tl.int64) * IW
        rowp = kp[:, None].to(tl.int64) * IW
        colm = lm[None, :].to(tl.int64)
        col0 = lc[None, :].to(tl.int64)
        colp = lp[None, :].to(tl.int64)
        Gmm = tl.load(in_ptr + base + rowm + colm).to(tl.float32)
        Gm0 = tl.load(in_ptr + base + rowm + col0).to(tl.float32)
        Gmp = tl.load(in_ptr + base + rowm + colp).to(tl.float32)
        G0m = tl.load(in_ptr + base + row0 + colm).to(tl.float32)
        G00 = tl.load(in_ptr + base + row0 + col0).to(tl.float32)
        G0p = tl.load(in_ptr + base + row0 + colp).to(tl.float32)
        Gpm = tl.load(in_ptr + base + rowp + colm).to(tl.float32)
        Gp0 = tl.load(in_ptr + base + rowp + col0).to(tl.float32)
        Gpp = tl.load(in_ptr + base + rowp + colp).to(tl.float32)
        l_first = (ll == 0)[None, :].broadcast_to(BLOCK_K, BLOCK_L)
        l_last = (ll == L - 1)[None, :].broadcast_to(BLOCK_K, BLOCK_L)
        We_m = tl.where(l_first, Gm0, 0.25 * Gmm + 0.75 * Gm0)
        We_0 = tl.where(l_first, G00, 0.25 * G0m + 0.75 * G00)
        We_p = tl.where(l_first, Gp0, 0.25 * Gpm + 0.75 * Gp0)
        Wo_m = tl.where(l_last, Gm0, 0.75 * Gm0 + 0.25 * Gmp)
        Wo_0 = tl.where(l_last, G00, 0.75 * G00 + 0.25 * G0p)
        Wo_p = tl.where(l_last, Gp0, 0.75 * Gp0 + 0.25 * Gpp)
        k_first = (kk == 0)[:, None].broadcast_to(BLOCK_K, BLOCK_L)
        k_last = (kk == K - 1)[:, None].broadcast_to(BLOCK_K, BLOCK_L)
        EE = tl.where(k_first, We_0, 0.25 * We_m + 0.75 * We_0)
        EO = tl.where(k_first, Wo_0, 0.25 * Wo_m + 0.75 * Wo_0)
        OE = tl.where(k_last, We_0, 0.75 * We_0 + 0.25 * We_p)
        OO = tl.where(k_last, Wo_0, 0.75 * Wo_0 + 0.25 * Wo_p)
        even_row = tl.interleave(EE, EO)
        odd_row = tl.interleave(OE, OO)
        obase = pid_nc.to(tl.int64) * (OH * OW) + (2 * kk)[:, None].to(tl.int64) * OW
        ocols = 2 * pid_l * BLOCK_L + tl.arange(0, 2 * BLOCK_L)
        ocol = ocols[None, :].to(tl.int64)
        kmask = (kk[:, None] < K) & (ocols[None, :] < OW)
        tl.store(out_ptr + obase + ocol, even_row, mask=kmask)
        tl.store(out_ptr + obase + OW + ocol, odd_row, mask=kmask)


def _reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        return 0.0
    if scale is not None and scale > 0:
        return 1.0 / scale
    return src_size / dst_size


_BH, _BW, _WARPS = 16, 64, 4
# fp16/bf16 large-shape config: interior-fast-path pair kernel (square tile, 4 warps)
_2X_K, _2X_L, _2X_WARPS = 32, 32, 4
_2X_WARPS_8 = 8
# fp16/bf16 small/mid-shape config: clamped pair kernel (square tile, 8 warps)
_2X_SK, _2X_SL, _2X_SW = 32, 32, 8
# fp32 pair-tile config (wide tile favored by fp32 bandwidth-bound case)
_2X_K32, _2X_L32, _2X_W32 = 16, 64, 8


def _specialized__upsample_bilinear2d_aa(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    N, C, IH, IW = input.shape
    OH, OW = int(output_size[0]), int(output_size[1])
    if N == 0 or C == 0 or IH == 0 or IW == 0 or OH == 0 or OW == 0:
        return torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    rs_h = _reciprocal_scale(IH, OH, align_corners, scales_h)
    rs_w = _reciprocal_scale(IW, OW, align_corners, scales_w)

    out = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    NC = N * C

    if rs_h == 0.5 and rs_w == 0.5 and OH == 2 * IH and OW == 2 * IW:
        K, L = IH, IW
        if input.dtype == torch.float32:
            BK, BL, W = _2X_K32, _2X_L32, _2X_W32
            grid = (triton.cdiv(L, BL) * triton.cdiv(K, BK) * NC,)
            _upsample_aa_2x_kernel[grid](
                out,
                input,
                NC,
                K,
                L,
                IH,
                IW,
                OH,
                OW,
                BLOCK_K=BK,
                BLOCK_L=BL,
                NEED_F32_CAST=False,
                num_warps=W,
            )
        elif K >= 256 and L >= 256:
            # large shapes: interior tiles dominate, use the fast path
            BK, BL = _2X_K, _2X_L
            # warp split: fp16 uses 8 warps (latency hiding on small grids);
            # bf16 uses 8 warps only for small grids (NC < 64) and 4 warps for
            # the bandwidth-bound big grids (NC >= 64)
            if input.dtype == torch.float16 or NC < 64:
                W = _2X_WARPS_8
            else:
                W = _2X_WARPS
            grid = (triton.cdiv(L, BL) * triton.cdiv(K, BK) * NC,)
            _upsample_aa_2x_fast_kernel[grid](
                out,
                input,
                NC,
                K,
                L,
                IH,
                IW,
                OH,
                OW,
                BLOCK_K=BK,
                BLOCK_L=BL,
                num_warps=W,
            )
        else:
            # small/mid shapes: few interior tiles, use the exact clamped kernel
            BK, BL, W = _2X_SK, _2X_SL, _2X_SW
            grid = (triton.cdiv(L, BL) * triton.cdiv(K, BK) * NC,)
            _upsample_aa_2x_kernel[grid](
                out,
                input,
                NC,
                K,
                L,
                IH,
                IW,
                OH,
                OW,
                BLOCK_K=BK,
                BLOCK_L=BL,
                NEED_F32_CAST=True,
                num_warps=W,
            )
        return out

    grid = (triton.cdiv(OW, _BW) * triton.cdiv(OH, _BH) * NC,)
    _upsample_aa_kernel[grid](
        out,
        input,
        NC,
        OH,
        OW,
        IH,
        IW,
        rs_h,
        rs_w,
        BLOCK_H=_BH,
        BLOCK_W=_BW,
        num_warps=_WARPS,
    )
    return out


def _upsample_bilinear2d_aa(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    logger.debug("GEMS_MTHREADS _UPSAMPLE_BILINEAR2D_AA")
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized__upsample_bilinear2d_aa(
            input, output_size, align_corners, scales_h, scales_w
        )
    return default__upsample_bilinear2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )
