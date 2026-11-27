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

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------- numerics


@triton.jit
def _exact_div(a, b):

    ai = a.to(tl.int32, bitcast=True)
    bi = b.to(tl.int32, bitcast=True)
    ea = (ai >> 23) & 0xFF
    eb = (bi >> 23) & 0xFF
    ma = (ai & 0x7FFFFF) | 0x800000
    mb = (bi & 0x7FFFFF) | 0x800000
    big = ma >= mb
    num = tl.where(big, ma, ma * 2)
    e = ea - eb + tl.where(big, 0, -1)
    n = num.to(tl.int64) << 25
    mb64 = mb.to(tl.int64)
    q = n // mb64
    r = n - q * mb64
    m24 = q >> 2
    rb = (q >> 1) & 1
    sticky = ((q & 1) | (r != 0).to(tl.int64)) == 1
    up = (rb == 1) & (sticky | ((m24 & 1) == 1))
    m24 = m24 + up.to(tl.int64)
    carry = m24 >= (1 << 24)
    m24 = tl.where(carry, m24 >> 1, m24)
    e = e + carry.to(tl.int32)
    ebits = e + 127
    out = ((ebits << 23) | (m24.to(tl.int32) & 0x7FFFFF)).to(tl.float32, bitcast=True)
    return tl.where(ebits >= 255, float("inf"), out)


@triton.jit
def _dec_e2m1(c):

    s = (c >> 3) & 1
    e = (c >> 1) & 3
    m = c & 1
    mant = 1.0 + m.to(tl.float32) * 0.5
    pw = tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, tl.where(e == 3, 4.0, 0.5)))
    val = tl.where(e == 0, m.to(tl.float32) * 0.5, mant * pw)
    # sign via bit-or so that a negative zero stays -0.0 (a `-val` or a
    # lerp-based interleave canonicalises it to +0.0, but the reference keeps
    # the sign bit: input code 0x8 must quantize to output code 0x8)
    bits = val.to(tl.int32, bitcast=True) | (s << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _enc_e2m1(q):

    aq = tl.abs(q)
    lo = aq < 2.0
    md = aq < 4.0
    s = tl.where(lo, 2.0, tl.where(md, 1.0, 0.5))
    bias = tl.where(lo, 12582912.0, tl.where(md, 12582914.0, 12582916.0))
    c = (aq * s + bias).to(tl.int32, bitcast=True) - 0x4B400000
    # -0.0 keeps its sign bit, so code 0x8 stays 0x8
    return c | ((q.to(tl.int32, bitcast=True) >> 31) & 8)


@triton.jit
def _fp8b(q):

    return q.to(tl.float8e4nv).to(tl.int8, bitcast=True).to(tl.int32) & 0xFF


@triton.jit
def _sf_from_amax(
    amax, OUT_FP4: tl.constexpr, ROUND_SF: tl.constexpr, OUT_PACKED: tl.constexpr
):
    if OUT_FP4:
        amax = tl.maximum(amax, 7.052966104933725e-38)
    else:
        amax = tl.maximum(amax, 1e-4)
    if ROUND_SF:
        bits = amax.to(tl.int32, bitcast=True)
        ef = (bits >> 23) & 0xFF
        ma = bits & 0x7FFFFF
        if OUT_FP4:
            exp = ef - 127 - 2 + (ma > 0x400000).to(tl.int32)
        else:
            exp = ef - 127 - 8 + (ma > 0x600000).to(tl.int32)
        exp = tl.maximum(exp, -126)
        qsf = ((127 - exp) << 23).to(tl.float32, bitcast=True)
        if OUT_PACKED:
            sfin = exp + 127
        else:
            sfin = ((exp + 127) << 23).to(tl.float32, bitcast=True)
    else:
        if OUT_FP4:
            maxv = 6.0
        else:
            maxv = 448.0
        mv = tl.full(amax.shape, maxv, tl.float32)
        # qsf must be computed BEFORE sfr. Each quotient is bit-exact alone,
        # but with both int64 long divisions in one kernel the one computed
        # second comes out wrong for a small fraction of groups on this target
        # (measured elsewhere: true 3.549575 -> 7.549575, i.e. true + 2^(exp+1)),
        # which breaks the data while leaving the sf exact.
        qsf = _exact_div(mv, amax)
        sfr = _exact_div(amax, mv)
        sfin = sfr
    return qsf, sfin


# ------------------------------------------------------------------ kernel


@triton.jit
def _ptc_kernel(
    X,
    INSF,
    OUT,
    SF,
    M,
    NGROUPS,
    row0,
    sf_rows,
    xs0,
    y0,
    isf_sm,
    isf_sg,
    osf_sm,
    osf_sg,
    IN_KIND: tl.constexpr,
    OUT_FP4: tl.constexpr,
    ROUND_SF: tl.constexpr,
    OUT_PACKED: tl.constexpr,
    IN_PACKED: tl.constexpr,
    HAS_IN_SF: tl.constexpr,
    SF_BCAST: tl.constexpr,
    PAIR: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    BLOCK_K: tl.constexpr = BLOCK_G * 32
    NB: tl.constexpr = BLOCK_G * 16
    # When the row count divides the tile and the group count divides the group
    # tile, both masks are provably redundant; masked accesses degrade vector
    # width on this target, so the data path drops them entirely.
    FULL: tl.constexpr = EVEN_M and EVEN_K

    rm = row0 + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rg = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    row_ok = rm < M
    grp_ok = rg < NGROUPS
    ok = row_ok[:, None] & grp_ok[None, :]

    if HAS_IN_SF:
        # The input sf block is sf_rows tall, and sf_rows is read from
        # x_block_size at runtime rather than assumed to be 32. When the tile's
        # effective row range lies wholly inside one sf block, every row of the
        # tile shares one sf row, so one vector load over the group axis plus a
        # layout broadcast replaces a per-row index-driven gather. SF_BCAST is
        # decided per launch, from this launch's own row0: the tile start must be
        # a multiple of sf_rows and BLOCK_M must divide sf_rows, which together
        # guarantee no tile can straddle two blocks and that the tail launch
        # takes the branch its own origin implies. Otherwise the per-row gather
        # below is used unchanged.
        if SF_BCAST:
            ib0 = (row0 + pid_m * BLOCK_M) // sf_rows
            if IN_PACKED:
                gi = rg >> 1
                sidx = ib0 * isf_sm + gi * isf_sg
                raw = tl.load(INSF + sidx, mask=grp_ok, other=0).to(tl.int32)
                byte = (raw >> ((rg & 1) << 3)) & 0xFF
                sfl1 = (byte << 23).to(tl.float32, bitcast=True)
            else:
                sidx = ib0 * isf_sm + rg * isf_sg
                sfl1 = tl.load(INSF + sidx, mask=grp_ok, other=0.0)
            sfl = tl.broadcast_to(sfl1[None, :], (BLOCK_M, BLOCK_G))
        else:
            ib = rm // sf_rows
            if IN_PACKED:
                gi = rg >> 1
                sidx = ib[:, None] * isf_sm + gi[None, :] * isf_sg
                raw = tl.load(INSF + sidx, mask=ok, other=0).to(tl.int32)
                byte = (raw >> ((rg & 1) << 3)) & 0xFF
                sfl = (byte << 23).to(tl.float32, bitcast=True)
            else:
                sidx = ib[:, None] * isf_sm + rg[None, :] * isf_sg
                sfl = tl.load(INSF + sidx, mask=ok, other=0.0)

    if PAIR:
        # ---- even/odd lanes as two 16-wide tensors: no extent-2 axis anywhere
        if IN_KIND == 3:
            cb = pid_g * NB + tl.arange(0, NB)
            if FULL:
                xb = tl.load(X + rm[:, None] * xs0 + cb[None, :]).to(tl.int32)
            else:
                xb = tl.load(
                    X + rm[:, None] * xs0 + cb[None, :],
                    mask=row_ok[:, None] & (cb[None, :] < NGROUPS * 16),
                    other=0,
                ).to(tl.int32)
            xb3 = tl.reshape(xb, (BLOCK_M, BLOCK_G, 16))
            ve = _dec_e2m1(xb3 & 0xF)
            vo = _dec_e2m1((xb3 >> 4) & 0xF)
        else:
            # affine stride-2 columns; the group axis stays exact, so only the
            # row mask can be live
            je = (
                pid_g * BLOCK_K
                + 32 * tl.arange(0, BLOCK_G)[None, :, None]
                + 2 * tl.arange(0, 16)[None, None, :]
            )
            if FULL:
                ve = tl.load(X + rm[:, None, None] * xs0 + je).to(tl.float32)
                vo = tl.load(X + rm[:, None, None] * xs0 + (je + 1)).to(tl.float32)
            else:
                msk = row_ok[:, None, None] & (je < NGROUPS * 32)
                ve = tl.load(X + rm[:, None, None] * xs0 + je, mask=msk, other=0.0).to(
                    tl.float32
                )
                vo = tl.load(
                    X + rm[:, None, None] * xs0 + (je + 1), mask=msk, other=0.0
                ).to(tl.float32)
        if HAS_IN_SF:
            ve = ve * sfl[:, :, None]
            vo = vo * sfl[:, :, None]

        amax = tl.maximum(tl.max(tl.abs(ve), 2), tl.max(tl.abs(vo), 2))
        qsf, sfin = _sf_from_amax(amax, OUT_FP4, ROUND_SF, OUT_PACKED)
        q3 = qsf[:, :, None]
        cbo = pid_g * NB + tl.arange(0, NB)
        mko = row_ok[:, None] & (cbo[None, :] < NGROUPS * 16)
        if OUT_FP4:
            ce = _enc_e2m1(ve * q3)
            co = _enc_e2m1(vo * q3)
            packed = tl.reshape(ce | (co << 4), (BLOCK_M, NB))
            if FULL:
                tl.store(OUT + rm[:, None] * y0 + cbo[None, :], packed.to(tl.int8))
            else:
                tl.store(
                    OUT + rm[:, None] * y0 + cbo[None, :], packed.to(tl.int8), mask=mko
                )
        else:
            we = _fp8b(ve * q3)
            wo = _fp8b(vo * q3)
            word = tl.reshape(we | (wo << 8), (BLOCK_M, NB))
            if FULL:
                tl.store(OUT + rm[:, None] * y0 + cbo[None, :], word.to(tl.int16))
            else:
                tl.store(
                    OUT + rm[:, None] * y0 + cbo[None, :], word.to(tl.int16), mask=mko
                )
    else:
        # ---- contiguous 32-lane path (fp8 output from a non-fp4 input)
        rk = pid_g * BLOCK_K + tl.arange(0, BLOCK_K)
        if FULL:
            x = tl.load(X + rm[:, None] * xs0 + rk[None, :])
        else:
            x = tl.load(
                X + rm[:, None] * xs0 + rk[None, :],
                mask=row_ok[:, None] & (rk[None, :] < NGROUPS * 32),
                other=0.0,
            )
        v3 = tl.reshape(x.to(tl.float32), (BLOCK_M, BLOCK_G, 32))
        if HAS_IN_SF:
            v3 = v3 * sfl[:, :, None]
        amax = tl.max(tl.abs(v3), 2)
        qsf, sfin = _sf_from_amax(amax, OUT_FP4, ROUND_SF, OUT_PACKED)
        # no saturation and no -0.0 fixup: see _fp8b (same 448.000055 bound,
        # measured no-op below 464 over 369,098,752 exhaustive probe values)
        q3v = v3 * qsf[:, :, None]
        if FULL:
            tl.store(
                OUT + rm[:, None] * y0 + rk[None, :],
                tl.reshape(q3v, (BLOCK_M, BLOCK_K)).to(tl.float8e4nv),
            )
        else:
            tl.store(
                OUT + rm[:, None] * y0 + rk[None, :],
                tl.reshape(q3v, (BLOCK_M, BLOCK_K)).to(tl.float8e4nv),
                mask=row_ok[:, None] & (rk[None, :] < NGROUPS * 32),
            )

    # ---- sf store
    if OUT_PACKED:
        p2 = tl.reshape(sfin, (BLOCK_M, BLOCK_G // 2, 2))
        wb = (tl.arange(0, 2) * 255 + 1)[None, None, :]
        s16 = tl.sum(p2 * wb, 2)
        jj = pid_g * (BLOCK_G // 2) + tl.arange(0, BLOCK_G // 2)
        idx = rm[:, None] * osf_sm + jj[None, :] * osf_sg
        mask = row_ok[:, None] & (jj[None, :] < ((NGROUPS + 1) // 2))
        tl.store(SF + idx, s16.to(tl.int16), mask=mask)
    else:
        idx = rm[:, None] * osf_sm + rg[None, :] * osf_sg
        tl.store(SF + idx, sfin, mask=ok)


# ----------------------------------------------------------------- wrapper

_DEF_BM = 8

_DEF_BG = 32


def _pick(G, M):

    return _DEF_BM, _DEF_BG


def per_token_cast(
    x,
    fmt,
    num_per_channels=32,
    x_block_size=None,
    use_tma_aligned_col_major_sf=False,
    round_sf=False,
    use_packed_ue8m0=False,
    use_e4m3_sf=False,
    stochastic_cast=False,
    out=None,
    **kw,
):
    """Quantize or requantize each token in channel groups of 32 to E4M3 or packed E2M1."""
    if (
        fmt not in ("e4m3", "e2m1")
        or num_per_channels != 32
        or use_e4m3_sf
        or stochastic_cast
        or kw
    ):
        raise ValueError("expected deterministic group-32 E4M3 or E2M1 quantization")
    data = x[0] if isinstance(x, (tuple, list)) else x
    if data.ndim != 2 or data.dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.int8,
    ):
        raise ValueError("expected a matrix of FP32, BF16, E4M3 or packed E2M1 values")
    hidden = data.shape[1] * (2 if data.dtype == torch.int8 else 1)
    if hidden % 128 or (use_packed_ue8m0 and not round_sf):
        raise ValueError(
            "hidden size must be aligned to 128 and packed scales require round_sf"
        )
    if isinstance(x, (tuple, list)):
        xin, xsf = x[0], x[1]
    else:
        xin, xsf = x, None

    M = int(xin.shape[0])
    K = int(xin.shape[1]) * 2 if xin.dtype == torch.int8 else int(xin.shape[1])
    G = K // 32
    dev = xin.device
    out_fp4 = fmt == "e2m1"
    col = bool(use_tma_aligned_col_major_sf)
    packed = bool(use_packed_ue8m0)

    if xin.dtype == torch.float32:
        in_kind = 0
    elif xin.dtype == torch.bfloat16:
        in_kind = 1
    elif xin.dtype == torch.float8_e4m3fn:
        in_kind = 2
    else:
        in_kind = 3

    y_w = K // 2 if out_fp4 else K
    o_dtype = torch.int8 if out_fp4 else torch.float8_e4m3fn
    if out is not None and out[0] is not None:
        oxt = out[0]
    else:
        oxt = torch.empty((M, y_w), dtype=o_dtype, device=dev)

    ncol = (G + 1) // 2 if packed else G
    sfd = torch.int16 if packed else torch.float32
    if out is not None and out[1] is not None:
        sfv = out[1]
        sbase = sfv
    elif col:
        sbase = torch.empty((ncol, M), dtype=sfd, device=dev)
        sfv = sbase.T
    else:
        sbase = torch.empty((M, ncol), dtype=sfd, device=dev)
        sfv = sbase

    if M == 0 or G == 0:
        return oxt, sfv

    pair = out_fp4 or in_kind == 3
    # fp4 input -> fp8 output: store one int16 word per (even, odd) lane pair
    word_out = pair and not out_fp4
    if word_out:
        if oxt.stride(1) != 1:
            oxt = oxt.contiguous()
        out_t = oxt.view(torch.int16)
    else:
        out_t = oxt

    bm, bg = _pick(G, M)
    isf = xsf if xsf is not None else out_t
    nb_g = triton.cdiv(G, bg)
    even_k = (G % bg) == 0
    in_packed = 1 if (xsf is not None and xsf.dtype == torch.int16) else 0
    # Row extent of one input sf block, read from x_block_size at runtime rather
    # than assumed. It is the sf row stride of the pre-quantized input; with no
    # input sf the value is unused and 1 keeps it harmless.
    if xsf is None:
        sf_rows = 1
    elif x_block_size is not None:
        sf_rows = int(x_block_size[0])
    else:
        sf_rows = 32

    def _launch(rows, row0, m_arg, even_m):
        # The broadcast is legal only when this launch's tiles cannot straddle
        # two sf blocks: the launch origin must be block-aligned and BLOCK_M must
        # divide the block extent. Decided per launch, so the tail launch takes
        # the branch its own row0 implies rather than inheriting the bulk one.
        sf_bcast = (
            xsf is not None
            and sf_rows > 0
            and (sf_rows % bm) == 0
            and (row0 % sf_rows) == 0
        )
        _ptc_kernel[(rows, nb_g)](
            xin,
            isf,
            out_t,
            sbase,
            m_arg,
            G,
            row0,
            sf_rows,
            xin.stride(0),
            out_t.stride(0),
            xsf.stride(0) if xsf is not None else 0,
            xsf.stride(1) if xsf is not None else 0,
            sfv.stride(0),
            sfv.stride(1),
            IN_KIND=in_kind,
            OUT_FP4=out_fp4,
            ROUND_SF=bool(round_sf),
            OUT_PACKED=packed,
            IN_PACKED=in_packed,
            HAS_IN_SF=xsf is not None,
            SF_BCAST=sf_bcast,
            PAIR=pair,
            EVEN_M=even_m,
            EVEN_K=even_k,
            BLOCK_M=bm,
            BLOCK_G=bg,
        )

    # Rows are split so that the bulk of them run through the mask-free path:
    # a masked access lowers to dynamic-extent pad-mode DMA on this target, so
    # keeping one odd tail block from masking the other 500 costs a launch and
    # saves the predicate on every full block.
    nfull = M // bm
    if nfull:
        _launch(nfull, 0, nfull * bm, True)
    if M > nfull * bm:
        _launch(1, nfull * bm, M, False)
    return oxt, sfv
