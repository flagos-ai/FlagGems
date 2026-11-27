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

import numpy as np
import torch
import triton
import triton.language as tl

_TARGET_PROG = 64
_BLOCK_M_MAX = 16
_BLOCK_M_MIN = 4
_TILE_ELEMS = 8192
_TILE_ELEMS_WIDE = 16384


def _block_m(ntok):

    blk = _BLOCK_M_MAX
    while blk > _BLOCK_M_MIN and (ntok + blk - 1) // blk < _TARGET_PROG:
        blk //= 2
    return blk


def _align(v, a):
    return (v + a - 1) // a * a


def _pow2_divisor(v, cap):
    d = 1
    while d * 2 <= cap and v % (d * 2) == 0:
        d *= 2
    return d


def _split_fp32(value):
    hi = float(np.float32(value))
    lo = float(np.float32(value - hi))
    return hi, lo


@triton.jit
def _fmadd(x, y, z):
    return tl.math.fma(x, y, z)


@triton.jit
def _divnr(x, y, yinv):

    q = x * yinv
    e = tl.math.fma(-y, q, x)
    return tl.math.fma(e, yinv, q)


@triton.jit
def _divrn(x, y):
    return tl.math.div_rn(x, y)


@triton.jit
def _group_max_i16(bits, BLOCK_M: tl.constexpr, NGRP2: tl.constexpr):

    mag = bits & tl.full((1, 1, 1), 0x7FFF, tl.int16)
    a, b = tl.split(tl.reshape(mag, (BLOCK_M, NGRP2, 16, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(tl.reshape(m, (BLOCK_M, NGRP2, 8, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(tl.reshape(m, (BLOCK_M, NGRP2, 4, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(tl.reshape(m, (BLOCK_M, NGRP2, 2, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(m)
    return tl.maximum(a, b)


@triton.jit
def _group_max_i32(bits, BLOCK_M: tl.constexpr, NGRP2: tl.constexpr):

    mag = bits & tl.full((1, 1, 1), 0x7FFFFFFF, tl.int32)
    a, b = tl.split(tl.reshape(mag, (BLOCK_M, NGRP2, 16, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(tl.reshape(m, (BLOCK_M, NGRP2, 8, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(tl.reshape(m, (BLOCK_M, NGRP2, 4, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(tl.reshape(m, (BLOCK_M, NGRP2, 2, 2)))
    m = tl.maximum(a, b)
    a, b = tl.split(m)
    return tl.maximum(a, b)


@triton.jit
def _norm_cast_kernel(
    XP,
    RESP,
    ROUTP,
    OUTP,
    SFP,
    SFWP,
    BFP,
    RSTDP,
    WP,
    ntok,
    nblk,
    x_rs,
    x_bs,
    res_rs,
    res_bs,
    rout_rs,
    rout_bs,
    out_rs,
    out_bs,
    bf_rs,
    bf_bs,
    sf_rs,
    sf_bs,
    sfw_rs,
    sfw_bs,
    out_scale,
    nprog,
    EPS_HI: tl.constexpr,
    EPS_LO: tl.constexpr,
    R0_HI: tl.constexpr,
    R0_LO: tl.constexpr,
    HIDDEN: tl.constexpr,
    NK: tl.constexpr,
    W1: tl.constexpr,
    CHUNK2: tl.constexpr,
    NK2: tl.constexpr,
    NGRP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IN_BF16: tl.constexpr,
    BF16Q: tl.constexpr,
    MASKM: tl.constexpr,
    COLMASK: tl.constexpr,
    WITH_WEIGHT: tl.constexpr,
    WITH_RESIDUAL: tl.constexpr,
    WITH_RSTD: tl.constexpr,
    WITH_OUT_BF16: tl.constexpr,
    ROUND_SF: tl.constexpr,
    PACKED_SF: tl.constexpr,
    COL_SF: tl.constexpr,
    WORD_SF: tl.constexpr,
    SCALED: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    NGRP: tl.constexpr = HIDDEN // 32
    INV448: tl.constexpr = 0.002232142857142857
    M1: tl.constexpr = MASKM or COLMASK
    NP2: tl.constexpr = NGRP2 // 2

    for blk in range(pid, nblk, nprog):
        if MASKM:
            m = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        else:
            m = tl.minimum(blk * BLOCK_M, ntok - BLOCK_M) + tl.arange(0, BLOCK_M)
        rmask = m < ntok
        xrow = XP + bid * x_bs + m[:, None] * x_rs
        rrow = RESP + bid * res_bs + m[:, None] * res_rs
        worow = ROUTP + bid * rout_bs + m[:, None] * rout_rs
        orow = OUTP + bid * out_bs + m[:, None] * out_rs
        brow = BFP + bid * bf_bs + m[:, None] * bf_rs
        sb = SFP + bid * sf_bs
        sw = SFWP + bid * sfw_bs

        # ---- phase 1: sum of squares (reference reduction order) ----
        # W1 sub-blocks of 256 columns are unrolled inside one loop body: the
        # accumulation order stays exactly per-256-element-vector, but the trip
        # count drops W1-fold and each body exposes W1 independent load/FMA
        # groups for the scheduler to overlap.
        acc = tl.zeros((BLOCK_M, 256), tl.float32)
        for k in range(NK):
            for w in tl.static_range(W1):
                cols = (k * W1 + w) * 256 + tl.arange(0, 256)
                if M1:
                    cmask = rmask[:, None] & (cols[None, :] < HIDDEN)
                    v = tl.load(xrow + cols[None, :], mask=cmask, other=0.0)
                else:
                    v = tl.load(xrow + cols[None, :])
                if WITH_RESIDUAL:
                    if M1:
                        rv = tl.load(rrow + cols[None, :], mask=cmask, other=0.0)
                    else:
                        rv = tl.load(rrow + cols[None, :])
                    if IN_BF16:
                        s = (v.to(tl.float32) + rv.to(tl.float32)).to(tl.bfloat16)
                    else:
                        s = v + rv
                    if M1:
                        tl.store(worow + cols[None, :], s, mask=cmask)
                    else:
                        tl.store(worow + cols[None, :], s)
                    vf = s.to(tl.float32)
                else:
                    vf = v.to(tl.float32)
                acc = _fmadd(vf, vf, acc)

        a4 = tl.reshape(acc, (BLOCK_M, 2, 64, 2))
        if IN_BF16:
            a4 = tl.permute(a4, (0, 1, 3, 2))
        a4 = tl.reshape(a4, (BLOCK_M, 4, 64))
        jj = tl.arange(0, 4)[None, :, None]
        a0 = tl.sum(tl.where(jj == 0, a4, 0.0), axis=1)
        a1 = tl.sum(tl.where(jj == 1, a4, 0.0), axis=1)
        a2 = tl.sum(tl.where(jj == 2, a4, 0.0), axis=1)
        a3 = tl.sum(tl.where(jj == 3, a4, 0.0), axis=1)
        ssum = tl.sum(((a0 + a1) + a2) + a3, axis=1)

        # ---- compensated rstd ----
        product = ssum * R0_HI
        value = product + EPS_HI
        virtual_c1 = value - product
        product_sum_error = product - (value - virtual_c1)
        c1_sum_error = EPS_HI - virtual_c1
        sum_error = product_sum_error + c1_sum_error
        product_error = _fmadd(ssum, R0_HI, -product)
        value_error = product_error + sum_error
        value_error = _fmadd(ssum, R0_LO, value_error)
        value_error = value_error + EPS_LO

        est = _divrn(1.0, tl.math.sqrt_rn(value))
        est2 = est * est
        sq_err = _fmadd(est, est, -est2)
        resid = 1.0
        resid = _fmadd(-value, est2, resid)
        resid = _fmadd(-value, sq_err, resid)
        resid = _fmadd(-value_error, est2, resid)
        resid = _fmadd(-value_error, sq_err, resid)
        rstd = _fmadd(est * 0.5, resid, est)

        if WITH_RSTD:
            if MASKM:
                tl.store(RSTDP + bid * ntok + m, rstd, mask=rmask)
            else:
                tl.store(RSTDP + bid * ntok + m, rstd)

        # ---- phase 2: normalize + quantize ----
        rstd2 = rstd[:, None]
        # with a residual the rounded sum already lives in residual_out
        srow = worow if WITH_RESIDUAL else xrow
        for k in range(NK2):
            cols = k * CHUNK2 + tl.arange(0, CHUNK2)
            if MASKM:
                cm2 = rmask[:, None]
                v = tl.load(srow + cols[None, :], mask=cm2, other=0.0)
            else:
                v = tl.load(srow + cols[None, :])
            y = v.to(tl.float32) * rstd2
            if WITH_WEIGHT:
                wf = tl.load(WP + cols).to(tl.float32)
                if SCALED:
                    wf = wf * out_scale
                y = y * wf[None, :]
            elif SCALED:
                y = y * out_scale
            if IN_BF16:
                # reference rounding of y to x.dtype; reused by the bf16 output,
                # the group amax and (see BF16Q) the quantizer.
                yb16 = y.to(tl.bfloat16)
                if WITH_OUT_BF16:
                    if MASKM:
                        tl.store(brow + cols[None, :], yb16, mask=cm2)
                    else:
                        tl.store(brow + cols[None, :], yb16)
                y3b = tl.reshape(yb16, (BLOCK_M, NGRP2, 32))
                m1 = _group_max_i16(y3b.to(tl.int16, bitcast=True), BLOCK_M, NGRP2)
                # widen only the reduced (BLOCK_M, NGRP2) result: the bf16 bit
                # pattern becomes the high half of the f32 pattern.
                amax = tl.maximum(
                    (m1.to(tl.int32) << 16).to(tl.float32, bitcast=True), 1e-4
                )
                if not BF16Q:
                    # only the non-power-of-two scale path still needs the f32
                    # image of the tile (for its +/-448 clamp)
                    y3 = y3b.to(tl.float32)
            else:
                y3 = tl.reshape(y, (BLOCK_M, NGRP2, 32))
                if WITH_OUT_BF16:
                    if MASKM:
                        tl.store(brow + cols[None, :], y.to(tl.bfloat16), mask=cm2)
                    else:
                        tl.store(brow + cols[None, :], y.to(tl.bfloat16))
                amax = tl.maximum(
                    _group_max_i32(y3.to(tl.int32, bitcast=True), BLOCK_M, NGRP2).to(
                        tl.float32, bitcast=True
                    ),
                    1e-4,
                )

            inv448 = tl.zeros_like(amax) + INV448
            t = _divnr(amax, 448.0, inv448)
            if ROUND_SF:
                bits = t.to(tl.int32, bitcast=True)
                sf_val = ((bits + 0x7FFFFF) & 0x7F800000).to(tl.float32, bitcast=True)
                inv = _divrn(1.0, sf_val)
            else:
                sf_val = t
                inv = _divnr(tl.zeros_like(amax) + 448.0, amax, _divrn(1.0, amax))

            if BF16Q:
                # sf is a power of two, so scaling a bf16 value by 1/sf is
                # exact (|q| <= 448 by construction, and anything that would
                # underflow bf16 already rounds to fp8 zero), hence identical
                # fp8 bytes at twice the lanes.
                inv_b = inv.to(tl.bfloat16)
                q = tl.reshape(y3b * inv_b[:, :, None], (BLOCK_M, CHUNK2))
            elif ROUND_SF:
                q = tl.reshape(y3 * inv[:, :, None], (BLOCK_M, CHUNK2))
            else:
                q = tl.minimum(
                    tl.maximum(
                        tl.reshape(y3 * inv[:, :, None], (BLOCK_M, CHUNK2)), -448.0
                    ),
                    448.0,
                )
            if MASKM:
                tl.store(orow + cols[None, :], q.to(tl.float8e4nv), mask=cm2)
            else:
                tl.store(orow + cols[None, :], q.to(tl.float8e4nv))

            if WORD_SF:
                # one uint16 word per (row, group pair): low byte = group 2p,
                # high byte = group 2p+1; inner (contiguous) axis = row index.
                byte = (((t.to(tl.int32, bitcast=True) - 1) >> 23) + 1).to(tl.uint16)
                lo, hi = tl.split(tl.reshape(byte, (BLOCK_M, NP2, 2)))
                word = lo | (hi << 8)
                pidx = k * NP2 + tl.arange(0, NP2)
                woff = pidx[:, None] * sfw_rs + m[None, :]
                if MASKM:
                    tl.store(sw + woff, tl.trans(word), mask=rmask[None, :])
                else:
                    tl.store(sw + woff, tl.trans(word))
            else:
                kg = k * NGRP2 + tl.arange(0, NGRP2)
                if COL_SF:
                    soff = (
                        (kg // 2)[None, :] * sf_rs + m[:, None] * 2 + (kg % 2)[None, :]
                    )
                else:
                    soff = m[:, None] * sf_rs + kg[None, :]
                if MASKM:
                    gmask = rmask[:, None] & (kg[None, :] < NGRP)
                else:
                    gmask = kg[None, :] < NGRP
                if PACKED_SF:
                    byte = ((t.to(tl.int32, bitcast=True) - 1) >> 23) + 1
                    tl.store(sb + soff, byte.to(tl.uint8), mask=gmask)
                else:
                    tl.store(sb + soff, sf_val, mask=gmask)


def _sf_shape(m, k, packed, col):
    nb_m = m
    nb_k = k // 32
    if packed:
        nb_k = (nb_k + 1) // 2
    if col:
        nb_m, nb_k = nb_k, nb_m
    if packed:
        nb_k = nb_k * 2
    return nb_m, nb_k


def norm_forward_and_per_token_cast(
    x,
    weight,
    eps,
    fmt,
    num_per_channels,
    residual=None,
    out_scale=1.0,
    use_tma_aligned_col_major_sf=False,
    round_sf=False,
    use_packed_ue8m0=False,
    with_out_bf16=False,
    out=None,
    out_bf16=None,
    residual_out=None,
    mega_moe_sf=None,
    mega_moe_block_m=0,
    with_rstd=False,
):
    """Fuse RMS normalization and group-32 per-token FP8 quantization."""
    if (
        fmt != "e4m3"
        or num_per_channels != 32
        or x.shape[-1] % 32
        or mega_moe_sf is not None
        or mega_moe_block_m
    ):
        raise ValueError("expected group-32 E4M3 quantization without MegaMoE output")
    assert fmt == "e4m3" and num_per_channels == 32
    assert x.ndim in (2, 3) and x.stride(-1) == 1
    if weight is not None:
        assert (
            weight.shape == (x.shape[-1],)
            and weight.dtype == x.dtype
            and weight.is_contiguous()
        )

    hidden = x.shape[-1]
    ntok = x.shape[-2]
    batch = x.shape[0] if x.ndim == 3 else 1
    device = x.device
    in_bf16 = x.dtype == torch.bfloat16
    dev = torch.float8_e4m3fn
    sf_dtype = torch.uint8 if use_packed_ue8m0 else torch.float32

    out_sf = None
    if out is None:
        out_data = torch.empty(x.shape, dtype=dev, device=device)
    else:
        out_data, out_sf = out

    ext_sf = out_sf is not None
    if use_packed_ue8m0 and ext_sf:
        out_sf = out_sf.view(torch.uint8)
    if out_sf is None:
        alloc_shape = (ntok, hidden)
        if x.ndim == 3:
            if use_tma_aligned_col_major_sf:
                sf_pack = 2 if use_packed_ue8m0 else 1
                alloc_shape = (ntok, batch * _align(hidden, 32 * sf_pack))
            else:
                alloc_shape = (batch * ntok, hidden)
        nb_m, nb_k = _sf_shape(
            alloc_shape[0],
            alloc_shape[1],
            use_packed_ue8m0,
            use_tma_aligned_col_major_sf,
        )
        out_sf = torch.empty((nb_m, nb_k), dtype=sf_dtype, device=device)
        if x.ndim == 3:
            sf_rows, sf_cols = _sf_shape(
                ntok, hidden, use_packed_ue8m0, use_tma_aligned_col_major_sf
            )
            out_sf = out_sf.as_strided(
                (batch, sf_rows, sf_cols),
                (sf_rows * out_sf.stride(0), out_sf.stride(0), 1),
            )
    sfv = out_sf

    if with_out_bf16:
        if out_bf16 is None:
            out_bf16 = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
    else:
        out_bf16 = None

    if residual is not None:
        if residual_out is None:
            residual_out = torch.empty_like(x)
        norm_input = residual_out
    else:
        residual_out = None
        norm_input = x

    if with_rstd:
        rstd = torch.empty(x.shape[:-1], dtype=torch.float32, device=device)
    else:
        rstd = None

    if ntok > 0 and hidden > 0:
        eps_hi, eps_lo = _split_fp32(float(eps))
        r0_hi, r0_lo = _split_fp32(1.0 / hidden)
        nk = _align(hidden, 256) // 256
        # unroll W1 256-column sub-blocks per phase-1 loop iteration (order and
        # therefore bits are unchanged); only when the trip count divides evenly.
        w1 = 4
        while w1 > 1 and nk % w1 != 0:
            w1 //= 2
        nk = nk // w1
        block_m = _block_m(ntok)

        prog_target = _TARGET_PROG
        if block_m >= _BLOCK_M_MAX:
            prog_target = 2 * _TARGET_PROG
        # Phase-2 stage budget.  Halving the phase-2 trips (8/10/14 -> 4/5/7) on
        # the row-block shapes measured 5-6% faster for non-residual bf16 and up
        # to 17% for the residual no-weight cases, because phase 2 has no
        # accumulation-order constraint and CHUNK2 still divides hidden.  It is
        # only legal for a 2-byte tile: the same 16384-element stage needs
        # 2524160 bits of UB against 2031616 available once the working tile is
        # 4 bytes (fp32 input), so fp32 keeps the narrow stage, as do the
        # 512-token shapes, which measured best with it.
        wide = in_bf16 and block_m >= 16
        tile_elems = _TILE_ELEMS_WIDE if wide else _TILE_ELEMS
        chunk_cap = max(32, tile_elems // block_m)
        chunk2 = _pow2_divisor(hidden, min(chunk_cap, max(hidden, 32)))
        ngrp2 = chunk2 // 32
        nk2 = hidden // chunk2
        ngroups = hidden // 32
        scaled = float(out_scale) != 1.0

        x_rs, x_bs = x.stride(-2), (x.stride(0) if x.ndim == 3 else 0)
        if residual is not None:
            res_rs, res_bs = residual.stride(-2), (
                residual.stride(0) if residual.ndim == 3 else 0
            )
            rout_rs, rout_bs = residual_out.stride(-2), (
                residual_out.stride(0) if residual_out.ndim == 3 else 0
            )
        else:
            res_rs = res_bs = rout_rs = rout_bs = 0
        out_rs, out_bs = out_data.stride(-2), (
            out_data.stride(0) if out_data.ndim == 3 else 0
        )
        if out_bf16 is not None:
            bf_rs, bf_bs = out_bf16.stride(-2), (
                out_bf16.stride(0) if out_bf16.ndim == 3 else 0
            )
        else:
            bf_rs = bf_bs = 0
        if sfv.ndim == 3:
            sf_bs = sfv.stride(0)
            sf_rs = sfv.stride(1)
        else:
            sf_bs = 0
            sf_rs = sfv.stride(0) if sfv.dim() > 1 else 0

        # packed column-major sf: emit uint16 words with a contiguous row axis
        word_sf = (
            bool(use_packed_ue8m0)
            and bool(use_tma_aligned_col_major_sf)
            and (ngroups % 2 == 0)
            and (ngrp2 % 2 == 0)
            and (sf_rs % 2 == 0)
            and (sf_bs % 2 == 0)
        )
        if word_sf:
            sfw = sfv.view(torch.int16)
            sfw_rs = sf_rs // 2
            sfw_bs = sf_bs // 2
        else:
            sfw = sfv
            sfw_rs = 0
            sfw_bs = 0

        if ntok >= block_m:
            nblk = (ntok + block_m - 1) // block_m
            maskm = False
        else:
            nblk = 1
            maskm = True
        nprog = min(nblk, max(1, prog_target // batch))

        _norm_cast_kernel[(nprog, batch)](
            x,
            residual if residual is not None else x,
            residual_out if residual_out is not None else x,
            out_data,
            sfv,
            sfw,
            out_bf16 if out_bf16 is not None else out_data,
            rstd if rstd is not None else x,
            weight if weight is not None else x,
            ntok,
            nblk,
            x_rs,
            x_bs,
            res_rs,
            res_bs,
            rout_rs,
            rout_bs,
            out_rs,
            out_bs,
            bf_rs,
            bf_bs,
            sf_rs,
            sf_bs,
            sfw_rs,
            sfw_bs,
            float(out_scale),
            nprog,
            EPS_HI=eps_hi,
            EPS_LO=eps_lo,
            R0_HI=r0_hi,
            R0_LO=r0_lo,
            HIDDEN=hidden,
            NK=nk,
            W1=w1,
            CHUNK2=chunk2,
            NK2=nk2,
            NGRP2=ngrp2,
            BLOCK_M=block_m,
            IN_BF16=in_bf16,
            BF16Q=(in_bf16 and bool(round_sf)),
            MASKM=maskm,
            COLMASK=(hidden % 256 != 0),
            WITH_WEIGHT=weight is not None,
            WITH_RESIDUAL=residual is not None,
            WITH_RSTD=with_rstd,
            WITH_OUT_BF16=with_out_bf16,
            ROUND_SF=bool(round_sf),
            PACKED_SF=bool(use_packed_ue8m0),
            COL_SF=bool(use_tma_aligned_col_major_sf),
            WORD_SF=word_sf,
            SCALED=scaled,
        )

    if x.ndim == 3 and use_tma_aligned_col_major_sf:
        sf_rows, sf_cols = sfv.shape[-2:]
        flat = sfv.as_strided((batch * sf_rows, sf_cols), (sfv.stride(1), 1))
        if use_packed_ue8m0:
            flat = flat.view(torch.int16)
        flat = flat.T
        out_sf = flat.as_strided(
            (batch, flat.shape[0], sf_rows),
            (sf_rows * flat.stride(1), flat.stride(0), flat.stride(1)),
        )
    elif use_packed_ue8m0:
        out_sf = sfv.view(torch.int16)
        if use_tma_aligned_col_major_sf:
            out_sf = out_sf.T
    else:
        out_sf = sfv.T if use_tma_aligned_col_major_sf else sfv

    return (out_data, out_sf), out_bf16, rstd, norm_input
