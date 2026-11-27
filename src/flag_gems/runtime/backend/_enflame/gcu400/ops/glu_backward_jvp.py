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

"""Triton implementation of aten::glu_backward_jvp (forward-mode JVP helper for glu).

Reference (ATen composite explicit-autograd) semantics, S = grad_glu.size(dim):

    a  = x.narrow(dim, 0, S);   b  = x.narrow(dim, S, S)
    da = dx.narrow(dim, 0, S);  db = dx.narrow(dim, S, S)
    gxa = grad_x.narrow(dim, 0, S)     # grad_x.narrow(dim, S, S) is unused
    sig_b  = sigmoid(b)
    glu    = a * sig_b
    db_neg = db - db * sig_b
    dgrad_x_a = dgrad_glu * sig_b + gxa * db_neg
    dgrad_x_b = dgrad_x_a * (a - glu) + gxa * (da - da * sig_b - glu * db_neg)
    out = cat([dgrad_x_a, dgrad_x_b], dim)

``grad_glu`` only supplies the split extent S; it is not an operand of the
formula and is never loaded.

One kernel over the reduced index space (a, j, r): ``a`` is the flattened
leading index (product of the output extents before ``dim``), ``j`` covers the
split axis [0, S) and produces BOTH output halves from the same registers (two
stores), and ``r`` is the flattened trailing index.  Operands are addressed
with the general rank-5 broadcast/strided rule: an axis contributes only when
that operand's extent on it is not 1.

Target profiling of the previous revision showed the kernel is compute-bound on
integer address decode, so here (a) the grid is three-dimensional and sized
largest-extent-first, removing the per-program div/mod chain, and (b) every
index divisor is a ``constexpr`` product of output extents, so the decode
lowers to multiply-shift instead of runtime integer division.

GCU400 specialization (2026-11-17, probe-swept on the core shapes):

- **Grid hard limits**: x <= 65535 programs, y/z <= 255 (the CUDA assumption
  of a 2**31 x axis does not hold).  Any oversized launch is re-mapped onto
  two (x, y) axes carrying a flat tile id.
- **Dim-last fast path** (``R1 == 1``, ``K <= 2048`` — the layout gated
  activations use): each program covers a (BLOCK_A, BLOCK_J) tile of rows of
  the leading index decoded in vector form, instead of one row per program.
  The tile keeps a constant ~2048 lanes (``BLOCK_A = 2048 // BLOCK_J``),
  and the old mapping left ~2 lanes of ILP per thread with the CTA count
  exploded (32768..262144 tiny CTAs on the core cases); the wide tile plus a
  grid-stride loop at ~1024 CTAs is 2.4-2.5x faster on the 3-D cases
  (probe), because the warp-contiguous axis still runs along ``j``.
- **int32 address domain**: the 64-bit fallback is gated on the exact
  per-tensor bound ``sum((size_i - 1) * stride_i) + 1`` instead of the coarse
  ``numel * max(stride)``, which flagged every dense big tensor; int64
  addressing costs ~+58% on GCU400.  Exception: 16-bit inputs on the generic
  (BLOCK_J, BLOCK_R) mapping keep int64 (the int32 domain costs +22% there,
  probe 2026-11-17), while the dim-last A2D tile prefers int32 (1.7x).
- **Split-grid rowloop**: split launches stride each CTA over several flat
  tiles (~1024 CTAs total), -17%..-30% on the split cases.

``enable_fp_fusion=False`` keeps ATen's separately rounded elementwise ops from
being contracted into an FMA, and for fp16/bf16 inputs (whose ATen
intermediates are materialized in the input dtype) each step is additionally
rounded back to that dtype.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger(__name__)

# GCU400 grid axis limits: x is 65535, y/z only 255 (CUDA's x axis is 2**31
# assumption does not hold here).
_GRID_X_HARD_LIMIT = 65535
_GRID_Y_HARD_LIMIT = 255
# Threshold that triggers the 2D split and is also the grid.x block size. Alternating A/B retests (6 representative
# cases) show 1024 is about 22% faster than 65535 (28.74 ms vs 36.85 ms), and 4096 is 29.10 ms.
_GRID_AXIS_LIMIT = 1024

_RANK = 5


@triton.jit
def _sig(v):
    """Bitwise-identical to CUDA's fp32 sigmoid for the values ATen feeds it."""
    return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-v))


@triton.jit
def _rd(x, DT: tl.constexpr):
    return x.to(DT).to(tl.float32)


@triton.jit
def _dig(i, dv, di, s0, s1, s2, s3, s4, IS64: tl.constexpr):
    """Peel one digit (divisor ``dv``, extent ``di``) off ``i`` (scalar) and
    return the remaining index plus its five signed offset contributions."""
    idx = (i // dv) % di
    if IS64:
        idx = idx.to(tl.int64)
    return i - idx * dv, idx * s0, idx * s1, idx * s2, idx * s3, idx * s4


@triton.jit
def _digv(i, dv, di, s0, s1, s2, s3, s4, IS64: tl.constexpr):
    """Vector form of :func:`_dig` (``i`` is the flattened trailing index)."""
    idx = (i // dv) % di
    if IS64:
        idx = idx.to(tl.int64)
    return i - idx * dv, idx * s0, idx * s1, idx * s2, idx * s3, idx * s4


@triton.jit
def _glu_bwd_jvp_body(
    x_ptr,
    dx_ptr,
    gx_ptr,
    dg_ptr,
    out_ptr,
    x_b,
    dx_b,
    ox,
    od,
    og,
    oq,
    obase,
    oj,
    msk,
    K: tl.constexpr,
    osd: tl.constexpr,
    OUTDT: tl.constexpr,
    INDT: tl.constexpr,
    ROUND: tl.constexpr,
    C64: tl.constexpr,
):
    """Load/compute/store core shared by every mapping orientation.

    ``ox``..``oq`` are the per-operand element offsets of the tile, ``obase``
    the output base (scalar or row-vector broadcast), ``oj`` the split-axis
    output offsets (``j * osd`` in the tile's j orientation).
    """
    if C64:
        av = tl.load(x_ptr + ox, mask=msk, other=0.0).to(tl.float64)
        bv = tl.load(x_ptr + x_b + ox, mask=msk, other=0.0).to(tl.float64)
        dav = tl.load(dx_ptr + od, mask=msk, other=0.0).to(tl.float64)
        dbv = tl.load(dx_ptr + dx_b + od, mask=msk, other=0.0).to(tl.float64)
        gv = tl.load(gx_ptr + og, mask=msk, other=0.0).to(tl.float64)
        qv = tl.load(dg_ptr + oq, mask=msk, other=0.0).to(tl.float64)
        sg = 1.0 / (1.0 + tl.exp(-bv))
        glu = av * sg
        dbn = dbv - dbv * sg
        va = qv * sg + gv * dbn
        vb = va * (av - glu) + gv * (dav - dav * sg - glu * dbn)
    else:
        av = tl.load(x_ptr + ox, mask=msk, other=0.0).to(tl.float32)
        bv = tl.load(x_ptr + x_b + ox, mask=msk, other=0.0).to(tl.float32)
        dav = tl.load(dx_ptr + od, mask=msk, other=0.0).to(tl.float32)
        dbv = tl.load(dx_ptr + dx_b + od, mask=msk, other=0.0).to(tl.float32)
        gv = tl.load(gx_ptr + og, mask=msk, other=0.0).to(tl.float32)
        qv = tl.load(dg_ptr + oq, mask=msk, other=0.0).to(tl.float32)
        if ROUND:
            sg = _rd(_sig(bv), INDT)
            glu = _rd(av * sg, INDT)
            dbn = _rd(dbv - _rd(dbv * sg, INDT), INDT)
            va = _rd(_rd(qv * sg, INDT) + _rd(gv * dbn, INDT), INDT)
            t1 = _rd(av - glu, INDT)
            t2 = _rd(dav - _rd(dav * sg, INDT), INDT)
            t4 = _rd(t2 - _rd(glu * dbn, INDT), INDT)
            vb = _rd(_rd(va * t1, INDT) + _rd(gv * t4, INDT), INDT)
        else:
            sg = _sig(bv)
            glu = av * sg
            dbn = dbv - dbv * sg
            va = qv * sg + gv * dbn
            vb = va * (av - glu) + gv * (dav - dav * sg - glu * dbn)

    ova = obase + oj
    ovb = ova + K * osd
    tl.store(out_ptr + ova, va.to(OUTDT), mask=msk)
    tl.store(out_ptr + ovb, vb.to(OUTDT), mask=msk)


@triton.jit
def _tile(
    x_ptr,
    dx_ptr,
    gx_ptr,
    dg_ptr,
    out_ptr,
    x_b,
    dx_b,
    xsd: tl.constexpr,
    dsd: tl.constexpr,
    gsd: tl.constexpr,
    qsd: tl.constexpr,
    osd: tl.constexpr,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    ds0,
    ds1,
    ds2,
    ds3,
    ds4,
    gs0,
    gs1,
    gs2,
    gs3,
    gs4,
    qs0,
    qs1,
    qs2,
    qs3,
    qs4,
    os0,
    os1,
    os2,
    os3,
    os4,
    la,
    jb,
    rb,
    K: tl.constexpr,
    RMASK: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    E2: tl.constexpr,
    E3: tl.constexpr,
    E4: tl.constexpr,
    D: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
    OUTDT: tl.constexpr,
    INDT: tl.constexpr,
    ROUND: tl.constexpr,
    IS64: tl.constexpr,
    C64: tl.constexpr,
    LASTJ: tl.constexpr,
):
    """Decode one tile and run the compute body.

    ``LASTJ`` selects the tile orientation.  True keeps the generic layout
    (BLOCK_J, BLOCK_R) with the trailing vector on the last (warp-contiguous)
    axis.  False is the dim-last fast path: the ``r`` axis carries BLOCK_A
    rows of the leading index (``la`` is that same rows vector, so the
    leading-digit chain decodes per-row bases while the trailing chain is a
    zero contribution because ``R1 == 1`` forces every trailing extent to 1),
    and ``j`` covers the whole split axis on the warp-contiguous axis.
    """
    j = jb * BLOCK_J + tl.arange(0, BLOCK_J)
    r = rb * BLOCK_R + tl.arange(0, BLOCK_R)

    ax = 0
    ad = 0
    ag = 0
    aq = 0
    ao = 0
    # ---- leading digits of ``la`` (most significant first) ----
    if D == 4:
        la, ax, ad, ag, aq, ao = _dig(
            la, E1 * E2 * E3, E0, xs0, ds0, gs0, qs0, os0, IS64
        )
        la, c0, c1, c2, c3, c4 = _dig(la, E2 * E3, E1, xs1, ds1, gs1, qs1, os1, IS64)
        ax += c0
        ad += c1
        ag += c2
        aq += c3
        ao += c4
        la, c0, c1, c2, c3, c4 = _dig(la, E3, E2, xs2, ds2, gs2, qs2, os2, IS64)
        ax += c0
        ad += c1
        ag += c2
        aq += c3
        ao += c4
        la, c0, c1, c2, c3, c4 = _dig(la, 1, E3, xs3, ds3, gs3, qs3, os3, IS64)
        ax += c0
        ad += c1
        ag += c2
        aq += c3
        ao += c4
    elif D == 3:
        la, ax, ad, ag, aq, ao = _dig(la, E1 * E2, E0, xs0, ds0, gs0, qs0, os0, IS64)
        la, c0, c1, c2, c3, c4 = _dig(la, E2, E1, xs1, ds1, gs1, qs1, os1, IS64)
        ax += c0
        ad += c1
        ag += c2
        aq += c3
        ao += c4
        la, c0, c1, c2, c3, c4 = _dig(la, 1, E2, xs2, ds2, gs2, qs2, os2, IS64)
        ax += c0
        ad += c1
        ag += c2
        aq += c3
        ao += c4
    elif D == 2:
        la, ax, ad, ag, aq, ao = _dig(la, E1, E0, xs0, ds0, gs0, qs0, os0, IS64)
        la, c0, c1, c2, c3, c4 = _dig(la, 1, E1, xs1, ds1, gs1, qs1, os1, IS64)
        ax += c0
        ad += c1
        ag += c2
        aq += c3
        ao += c4
    elif D == 1:
        la, ax, ad, ag, aq, ao = _dig(la, 1, E0, xs0, ds0, gs0, qs0, os0, IS64)

    if IS64:
        ax = ax.to(tl.int64)
        ad = ad.to(tl.int64)
        ag = ag.to(tl.int64)
        aq = aq.to(tl.int64)
        ao = ao.to(tl.int64)

    zr = tl.zeros((BLOCK_R,), dtype=tl.int32)
    tr = r
    tx = zr
    td = zr
    tg = zr
    tq = zr
    to = zr
    # ---- trailing digits of ``r`` (most significant first) ----
    if D == 0:
        tr, c0, c1, c2, c3, c4 = _digv(
            tr, E2 * E3 * E4, E1, xs1, ds1, gs1, qs1, os1, IS64
        )
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
        tr, c0, c1, c2, c3, c4 = _digv(tr, E3 * E4, E2, xs2, ds2, gs2, qs2, os2, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
        tr, c0, c1, c2, c3, c4 = _digv(tr, E4, E3, xs3, ds3, gs3, qs3, os3, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
        tr, c0, c1, c2, c3, c4 = _digv(tr, 1, E4, xs4, ds4, gs4, qs4, os4, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
    elif D == 1:
        tr, c0, c1, c2, c3, c4 = _digv(tr, E3 * E4, E2, xs2, ds2, gs2, qs2, os2, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
        tr, c0, c1, c2, c3, c4 = _digv(tr, E4, E3, xs3, ds3, gs3, qs3, os3, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
        tr, c0, c1, c2, c3, c4 = _digv(tr, 1, E4, xs4, ds4, gs4, qs4, os4, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
    elif D == 2:
        tr, c0, c1, c2, c3, c4 = _digv(tr, E4, E3, xs3, ds3, gs3, qs3, os3, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
        tr, c0, c1, c2, c3, c4 = _digv(tr, 1, E4, xs4, ds4, gs4, qs4, os4, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4
    elif D == 3:
        tr, c0, c1, c2, c3, c4 = _digv(tr, 1, E4, xs4, ds4, gs4, qs4, os4, IS64)
        tx += c0
        td += c1
        tg += c2
        tq += c3
        to += c4

    if IS64:
        tx = tx.to(tl.int64)
        td = td.to(tl.int64)
        tg = tg.to(tl.int64)
        tq = tq.to(tl.int64)
        to = to.to(tl.int64)

    if LASTJ:
        msk = (j < K)[:, None] & (r < RMASK)[None, :]
        jv = j[:, None]
        ox = ax + jv * xsd + tx[None, :]
        od = ad + jv * dsd + td[None, :]
        og = ag + jv * gsd + tg[None, :]
        oq = aq + jv * qsd + tq[None, :]
        obase = ao + to[None, :]
        oj = jv * osd
    else:
        msk = (j < K)[None, :] & (r < RMASK)[:, None]
        jv = j[None, :]
        ox = ax[:, None] + jv * xsd + tx[:, None]
        od = ad[:, None] + jv * dsd + td[:, None]
        og = ag[:, None] + jv * gsd + tg[:, None]
        oq = aq[:, None] + jv * qsd + tq[:, None]
        obase = (ao + to)[:, None]
        oj = jv * osd

    _glu_bwd_jvp_body(
        x_ptr,
        dx_ptr,
        gx_ptr,
        dg_ptr,
        out_ptr,
        x_b,
        dx_b,
        ox,
        od,
        og,
        oq,
        obase,
        oj,
        msk,
        K,
        osd,
        OUTDT,
        INDT,
        ROUND,
        C64,
    )


@triton.jit
def _glu_bwd_jvp_kernel(
    x_ptr,
    dx_ptr,
    gx_ptr,
    dg_ptr,
    out_ptr,
    x_b,
    dx_b,
    xsd: tl.constexpr,
    dsd: tl.constexpr,
    gsd: tl.constexpr,
    qsd: tl.constexpr,
    osd: tl.constexpr,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    ds0,
    ds1,
    ds2,
    ds3,
    ds4,
    gs0,
    gs1,
    gs2,
    gs3,
    gs4,
    qs0,
    qs1,
    qs2,
    qs3,
    qs4,
    os0,
    os1,
    os2,
    os3,
    os4,
    K: tl.constexpr,
    R1: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    E2: tl.constexpr,
    E3: tl.constexpr,
    E4: tl.constexpr,
    D: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
    OUTDT: tl.constexpr,
    INDT: tl.constexpr,
    ROUND: tl.constexpr,
    IS64: tl.constexpr,
    C64: tl.constexpr,
    SELA: tl.constexpr,
    SELJ: tl.constexpr,
    SELR: tl.constexpr,
    GRID1D: tl.constexpr,
    NJB: tl.constexpr,
    NRB: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
    NCTAS: tl.constexpr,
    A2D: tl.constexpr,
    A_EXT: tl.constexpr,
    BLOCK_A: tl.constexpr,
    NBLKS: tl.constexpr,
):
    if A2D:
        # Dim-last fast path (R1 == 1, K <= BLOCK_J): BLOCK_A rows of the
        # leading index share one (BLOCK_A, BLOCK_J) tile, so each thread
        # carries ~8 lanes of ILP instead of the 2 the one-row-per-CTA
        # mapping left it with (probe 2026-11-17: 2.4-2.5x there, and the
        # int32 address domain another ~2x on top).  The grid-stride loop
        # keeps the CTA count near the probe sweet spot (~1024) instead of
        # one CTA per row block.
        pid = tl.program_id(0) + tl.program_id(1) * GRID_STRIDE
        for blk in range(pid, NBLKS, NCTAS):
            rows = blk * BLOCK_A + tl.arange(0, BLOCK_A)
            _tile(
                x_ptr,
                dx_ptr,
                gx_ptr,
                dg_ptr,
                out_ptr,
                x_b,
                dx_b,
                xsd,
                dsd,
                gsd,
                qsd,
                osd,
                xs0,
                xs1,
                xs2,
                xs3,
                xs4,
                ds0,
                ds1,
                ds2,
                ds3,
                ds4,
                gs0,
                gs1,
                gs2,
                gs3,
                gs4,
                qs0,
                qs1,
                qs2,
                qs3,
                qs4,
                os0,
                os1,
                os2,
                os3,
                os4,
                rows,
                0,
                0,
                K=K,
                RMASK=A_EXT,
                E0=E0,
                E1=E1,
                E2=E2,
                E3=E3,
                E4=E4,
                D=D,
                BLOCK_J=BLOCK_J,
                BLOCK_R=BLOCK_A,
                OUTDT=OUTDT,
                INDT=INDT,
                ROUND=ROUND,
                IS64=IS64,
                C64=C64,
                LASTJ=False,
            )
    elif GRID1D:
        # Split-grid path: one CTA strides over several flat tiles
        # (grid-stride loop; probe: -17% to -30% vs one CTA per tile).  The
        # loop bound subsumes the old ``pid < TOTAL_TILES`` overshoot mask.
        pid = tl.program_id(0) + tl.program_id(1) * GRID_STRIDE
        for t in range(pid, TOTAL_TILES, NCTAS):
            rb = t % NRB
            tt = t // NRB
            jb = tt % NJB
            la = tt // NJB
            _tile(
                x_ptr,
                dx_ptr,
                gx_ptr,
                dg_ptr,
                out_ptr,
                x_b,
                dx_b,
                xsd,
                dsd,
                gsd,
                qsd,
                osd,
                xs0,
                xs1,
                xs2,
                xs3,
                xs4,
                ds0,
                ds1,
                ds2,
                ds3,
                ds4,
                gs0,
                gs1,
                gs2,
                gs3,
                gs4,
                qs0,
                qs1,
                qs2,
                qs3,
                qs4,
                os0,
                os1,
                os2,
                os3,
                os4,
                la,
                jb,
                rb,
                K=K,
                RMASK=R1,
                E0=E0,
                E1=E1,
                E2=E2,
                E3=E3,
                E4=E4,
                D=D,
                BLOCK_J=BLOCK_J,
                BLOCK_R=BLOCK_R,
                OUTDT=OUTDT,
                INDT=INDT,
                ROUND=ROUND,
                IS64=IS64,
                C64=C64,
                LASTJ=True,
            )
    else:
        v0 = tl.program_id(0)
        v1 = tl.program_id(1)
        v2 = tl.program_id(2)
        if SELA == 0:
            la = v0
        elif SELA == 1:
            la = v1
        elif SELA == 2:
            la = v2
        else:
            la = v2
        if SELJ == 0:
            jb = v0
        elif SELJ == 1:
            jb = v1
        elif SELJ == 2:
            jb = v2
        if SELR == 0:
            rb = v0
        elif SELR == 1:
            rb = v1
        else:
            rb = v2
        _tile(
            x_ptr,
            dx_ptr,
            gx_ptr,
            dg_ptr,
            out_ptr,
            x_b,
            dx_b,
            xsd,
            dsd,
            gsd,
            qsd,
            osd,
            xs0,
            xs1,
            xs2,
            xs3,
            xs4,
            ds0,
            ds1,
            ds2,
            ds3,
            ds4,
            gs0,
            gs1,
            gs2,
            gs3,
            gs4,
            qs0,
            qs1,
            qs2,
            qs3,
            qs4,
            os0,
            os1,
            os2,
            os3,
            os4,
            la,
            jb,
            rb,
            K=K,
            RMASK=R1,
            E0=E0,
            E1=E1,
            E2=E2,
            E3=E3,
            E4=E4,
            D=D,
            BLOCK_J=BLOCK_J,
            BLOCK_R=BLOCK_R,
            OUTDT=OUTDT,
            INDT=INDT,
            ROUND=ROUND,
            IS64=IS64,
            C64=C64,
            LASTJ=True,
        )


_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}


def _is_float8(dt):
    return str(dt).startswith("torch.float8")


def _out_dtype_of(tensors):
    dts = []
    for t in tensors:
        dts.append(torch.float32 if not t.dtype.is_floating_point else t.dtype)
    for cand in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        if cand in dts:
            return cand
    return torch.float32


def glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim, out=None):
    logger.debug("GEMS_ENFLAME GLU_BACKWARD_JVP")

    tensors = (grad_x, grad_glu, x, dgrad_glu, dx)
    for t in tensors:
        if t.dtype == torch.bool:
            raise RuntimeError("glu_backward_jvp: bool tensors are not supported")
        if _is_float8(t.dtype):
            raise RuntimeError(
                "glu_backward_jvp: float8 tensors are not supported on this vendor"
            )

    nd = x.dim()
    d = dim
    if d < 0:
        d += nd
    if d < 0 or d >= nd:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[-{nd}, {nd - 1}], but got {dim})"
        )

    n = x.shape[d]
    S = grad_glu.shape[d]
    S2 = dgrad_glu.shape[d]
    if S < 0 or 2 * S > n:
        raise RuntimeError(
            f"glu_backward_jvp: invalid grad_glu extent {S} along dim {dim}"
        )
    if S2 != 1 and S2 != S:
        raise RuntimeError(
            f"glu_backward_jvp: grad_glu extent {S} and dgrad_glu extent {S2} "
            f"are not broadcastable along dim {dim}"
        )
    K = S

    out_dtype = _out_dtype_of(tensors)
    out_shape = list(x.shape)
    out_shape[d] = 2 * K
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=x.device)
    elif out.dtype != out_dtype or list(out.shape) != out_shape:
        raise RuntimeError("glu_backward_jvp: out tensor has wrong dtype or shape")
    if out.numel() == 0:
        return out

    R1 = 1
    for i in range(d + 1, nd):
        R1 *= out_shape[i]
    A = out.numel() // (2 * K * R1)

    def five(t):
        return [
            t.stride(i) if i < t.dim() and t.shape[i] != 1 else 0 for i in range(_RANK)
        ]

    xr, dr, gr, qr, orr = (five(x), five(dx), five(grad_x), five(dgrad_glu), five(out))
    if S == 1:
        xr[d] = 0
        dr[d] = 0
        gr[d] = 0
    if S2 == 1:
        qr[d] = 0

    # Exact per-tensor offset bound sum((size_i - 1) * stride_i) + 1: the old
    # ``numel * max(stride)`` form flags every dense big tensor (e.g. the
    # (512, 512, 512) core case at 134M elements) as needing 64-bit address
    # arithmetic even though its whole footprint fits the int32
    # element-offset domain, and int64 addressing costs ~+58% on GCU400
    # (contiguous judgment 2026-11-17).  Negative strides (never produced by
    # the aten path) fall back to the conservative bound.
    maxoff = 0
    for t in (x, dx, grad_x, grad_glu, dgrad_glu, out):
        if t.numel() == 0:
            continue
        strides = t.stride()
        if any(s < 0 for s in strides):
            bound = t.numel() * max(abs(s) for s in strides)
        else:
            bound = 1 + sum(
                (t.shape[i] - 1) * strides[i] for i in range(t.dim()) if t.shape[i] > 1
            )
        maxoff = max(maxoff, bound)
    is64 = maxoff >= 2**31
    if (
        not is64
        and out_dtype in (torch.float16, torch.bfloat16)
        and not (R1 == 1 and K <= 2048)
    ):
        # 16-bit inputs on the generic (BLOCK_J, BLOCK_R) mapping keep the
        # 64-bit address domain: a 3-round interleaved probe (2026-11-17,
        # (64, 512, 512) dim=1) shows the int32 domain costs +22% there,
        # while the same flip gains 2.3x in fp32 — the regression is
        # 16-bit-specific to that mapping.  The dim-last A2D tile behaves
        # the other way round (int32 1.7x faster), so only the generic path
        # pins 64-bit addressing for 16-bit.
        is64 = True

    a2d = R1 == 1 and K <= 2048
    if a2d:
        # Dim-last fast path: BLOCK_A rows of the leading index share one
        # (BLOCK_A, BLOCK_J) tile, and the grid-stride loop keeps the CTA
        # count near the sweet spot instead of one CTA per row.  Probe sweeps
        # (2026-11-17) over BA x nctas x nw on the (64, 512, 512),
        # (512, 512, 512) and (4096, 4096) core cases: a constant ~2048-lane
        # tile (BA = 2048 // BJ, clamped to [1, 8]) with ~1024 CTAs / nw=8
        # wins jointly, 2.4-2.5x over the one-row-per-CTA mapping on the 3-D
        # cases and ~1.5x over the split rowloop on the K=2048 case.
        BJ = triton.next_power_of_2(max(K, 1))
        BA = max(1, min(8, 2048 // BJ))
        nblks = (A + BA - 1) // BA
        nctas = min(nblks, 1024)
        grid_stride = min(
            nctas,
            min(
                _GRID_X_HARD_LIMIT,
                max(_GRID_AXIS_LIMIT, triton.cdiv(nctas, _GRID_Y_HARD_LIMIT)),
            ),
        )
        grid = (grid_stride, triton.cdiv(nctas, grid_stride))
        grid1d = True
        sel = (0, 1, 2)
        BR = 1
        njb = 1
        nrb = 1
        total_tiles = nblks
        nw = min(8, max(1, (BA * BJ + 63) // 64))
    else:
        BA = 1
        nblks = 1
        if R1 == 1:
            BJ = min(1024, triton.next_power_of_2(max(K, 1)))
            BR = 1
        elif K == 1:
            BJ = 1
            BR = min(1024, triton.next_power_of_2(R1))
        else:
            BR = min(256, triton.next_power_of_2(R1))
            BJ = max(1, min(triton.next_power_of_2(K), 4096 // BR))

        njb = (K + BJ - 1) // BJ
        nrb = (R1 + BR - 1) // BR
        if A == 0 or njb == 0 or nrb == 0:
            return out

        # A program covers BJ*BR lanes.  The dim-is-last regime has K as
        # small as 8, so a fixed 4-warp launch leaves most threads idle;
        # give one warp per up to 64 lanes (2 lanes per thread) and keep 4
        # warps for the large 2-D tiles.
        nw = min(4, max(1, (BJ * BR + 63) // 64))

        # GCU400's x-axis limit is 65535 and y/z axes only 255 (different from CUDA's assumption).
        # When any axis overflows (or is deliberately split for performance), switch to two axes (x, y) to carry flat
        # tile
        # indices; inside the kernel a grid-stride loop crosses tiles (probe: CTA count reduced to ~1024
        # makes each CTA take more tiles, -17% ~ -30%), and the loop bound naturally serves as the out-of-bounds guard.
        total_tiles = A * njb * nrb
        exts = sorted((("a", A), ("j", njb), ("r", nrb)), key=lambda kv: -kv[1])
        needs_split = (
            exts[0][1] > min(_GRID_X_HARD_LIMIT, _GRID_AXIS_LIMIT)
            or exts[1][1] > _GRID_Y_HARD_LIMIT
            or exts[2][1] > _GRID_Y_HARD_LIMIT
        )
        if needs_split:
            nctas = min(total_tiles, 1024)
            grid_stride = min(
                nctas,
                min(
                    _GRID_X_HARD_LIMIT,
                    max(_GRID_AXIS_LIMIT, triton.cdiv(nctas, _GRID_Y_HARD_LIMIT)),
                ),
            )
            grid = (grid_stride, triton.cdiv(nctas, grid_stride))
            grid1d = True
            sel = (0, 1, 2)
        else:
            grid = (exts[0][1], exts[1][1], exts[2][1])
            grid_stride = grid[0]
            nctas = total_tiles
            grid1d = False
            pos = {name: i for i, (name, _) in enumerate(exts)}
            sel = (pos["a"], pos["j"], pos["r"])

    E = [out_shape[i] if i < nd else 1 for i in range(_RANK)]

    in_dtype = torch.float32
    if out_dtype in (torch.float16, torch.bfloat16):
        for t in tensors:
            if t.dtype in (torch.float16, torch.bfloat16):
                in_dtype = t.dtype
                break

    _glu_bwd_jvp_kernel[grid](
        x,
        dx,
        grad_x,
        dgrad_glu,
        out,
        S * x.stride(d),
        S * dx.stride(d),
        x.stride(d) if S > 1 else 0,
        dx.stride(d) if S > 1 else 0,
        grad_x.stride(d) if S > 1 else 0,
        dgrad_glu.stride(d) if S2 > 1 else 0,
        out.stride(d),
        *xr,
        *dr,
        *gr,
        *qr,
        *orr,
        K=K,
        R1=R1,
        E0=E[0],
        E1=E[1],
        E2=E[2],
        E3=E[3],
        E4=E[4],
        D=d,
        RANK=_RANK,
        BLOCK_J=BJ,
        BLOCK_R=BR,
        OUTDT=_TL_DTYPE[out_dtype],
        INDT=_TL_DTYPE.get(in_dtype, tl.float32),
        ROUND=bool(out_dtype in (torch.float16, torch.bfloat16)),
        IS64=is64,
        C64=out_dtype == torch.float64,
        SELA=sel[0],
        SELJ=sel[1],
        SELR=sel[2],
        GRID1D=grid1d,
        NJB=njb,
        NRB=nrb,
        TOTAL_TILES=total_tiles,
        GRID_STRIDE=grid_stride,
        NCTAS=nctas,
        A2D=a2d,
        A_EXT=A,
        BLOCK_A=BA,
        NBLKS=nblks,
        num_warps=nw,
        enable_fp_fusion=False,
    )
    return out


# ---------------------------------------------------------------------------
# Namespace sync (2026-11-17): ``SpecOpRegistrar`` only rebinds the top-level
# ``flag_gems`` namespace, so the ``flag_gems.ops.glu_backward_jvp`` submodule
# attribute — the object benchmark/test_glu_backward_jvp.py binds as
# ``gems_op`` — keeps pointing at the generic implementation.  The generic
# launch keeps CUDA's grid.x <= 2**31 assumption and overflows GCU400's
# 65535-program hard limit (P19) on the ((512, 512, 512), dim=-1) core case:
# grid = (262144, 1, 1) -> ``OutOfResources: out of resource: grid.x``.
# Rebind the package attribute to this specialized host so every reference
# path (top level, ATen registration, ops submodule) runs the GCU400 kernel.
# ``flag_gems.ops`` is fully imported before the registrar loads this module,
# so the lookup below is a plain sys.modules hit.
try:
    import flag_gems.ops as _generic_ops_pkg
except ImportError:  # pragma: no cover - flag_gems.ops always importable
    _generic_ops_pkg = None

if (
    _generic_ops_pkg is not None
    and getattr(_generic_ops_pkg, "glu_backward_jvp", None) is not glu_backward_jvp
):
    _generic_ops_pkg.glu_backward_jvp = glu_backward_jvp
