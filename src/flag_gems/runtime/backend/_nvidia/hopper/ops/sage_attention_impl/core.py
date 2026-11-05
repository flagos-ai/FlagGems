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

"""Triton/TLE-only SM90 SageAttention; no Gluon language or layouts."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _pack(x):
    return x.reshape((4, 2, 8, 4)).permute((1, 0, 2, 3)).reshape((256,))


@triton.jit
def _step(
    qsm,
    ksm,
    vsm,
    KS,
    qs,
    maximum,
    denominator,
    accumulator,
    start,
    NK,
    NKS: tl.constexpr,
    H: tl.constexpr,
    b,
    h,
    sm,
    n,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_SCALE: tl.constexpr,
    LAST: tl.constexpr,
    KDESC,
    KBAR,
    K_SINGLE: tl.constexpr,
    EARLY_K: tl.constexpr,
    VDESC,
    VBAR,
    V_SINGLE: tl.constexpr,
):
    kn = start * BN + n
    ks = tl.load(
        KS + (b * H + h) * NKS + kn // 128 * 4 + (kn % 8) // 2, kn < NK, other=0.0
    )
    raw_token = tle.gpu.wgmma(qsm, ksm, out_dtype=tl.int32, trans_b=True)
    raw_integer = tle.gpu.wgmma_wait(0, raw_token)
    if K_SINGLE and not LAST and EARLY_K:
        tle.gpu.copy(
            KDESC, ksm, [BN, D], [(b * H + h) * NK + (start + 1) * BN, 0], barrier=KBAR
        )
    raw = raw_integer.to(tl.float32)
    if K_SINGLE and not LAST and not EARLY_K:
        # WGMMA has finished reading K. Refill this one K slot while softmax/PV
        # operate on registers and the independently double-buffered V slots.
        tle.gpu.copy(
            KDESC, ksm, [BN, D], [(b * H + h) * NK + (start + 1) * BN, 0], barrier=KBAR
        )
    if GROUP_SCALE:
        kg = tl.load(KS + (b * H + h) * NKS + start * 4 + tl.arange(0, 4))
        dqg = qs[:, None] * kg[None, :]
        sg = dqg * sm
        scores = raw
        if LAST:
            dq = (
                tl.broadcast_to(dqg[:, :, None], (BM, 4, 32))
                .reshape((BM, 4, 8, 2, 2))
                .permute((0, 2, 3, 1, 4))
                .reshape((BM, BN))
            )
            scores = raw * dq
            sg = tl.full((BM, 4), 0.0, tl.float32) + sm
        max_scores = scores
        if LAST:
            max_scores = tl.where(kn[None, :] < NK, scores, float("-inf"))
        local_max = tl.max(
            max_scores.reshape((BM, 8, 2, 4, 2))
            .permute((0, 3, 1, 2, 4))
            .reshape((BM, 4, 32)),
            2,
        )
        row_max = tl.max(tl.fma(local_max, sg, -8.807), 1)
        new_maximum = tl.maximum(maximum, row_max)
        scale = (
            tl.broadcast_to(sg[:, :, None], (BM, 4, 32))
            .reshape((BM, 4, 8, 2, 2))
            .permute((0, 2, 3, 1, 4))
            .reshape((BM, BN))
        )
        alpha = tl.exp2(maximum - new_maximum)
        probability = tl.exp2(tl.fma(scores, scale, -new_maximum[:, None]))
        if LAST:
            probability = tl.where(kn[None, :] < NK, probability, 0.0)
    else:
        dequant = qs[:, None] * ks[None, :]
        scores = raw
        scale = dequant * sm
        if LAST:
            scores = raw * dequant
            scale = tl.full((BM, BN), 0.0, tl.float32) + sm
        off = tl.fma(scores, scale, -8.807)
        off = tl.where(kn[None, :] < NK, off, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(off, axis=1))
        alpha = tl.exp2(maximum - new_maximum)
        probability = tl.exp2(tl.fma(scores, scale, -new_maximum[:, None]))
        probability = tl.where(kn[None, :] < NK, probability, 0.0)
    # Native denominator uses4 thread-local sums, each serial over8 subtiles.
    p5 = probability.reshape((BM, 8, 2, 4, 2))
    a, z = tl.split(p5)
    a0, a1 = tl.split(a.permute((0, 1, 3, 2)))
    z0, z1 = tl.split(z.permute((0, 1, 3, 2)))
    groups = ((a0 + z0) + a1) + z1
    even, odd = tl.split(groups.permute((0, 2, 1)).reshape((BM, 4, 4, 2)))
    e0, e1 = tl.split(even.reshape((BM, 4, 2, 2)))
    o0, o1 = tl.split(odd.reshape((BM, 4, 2, 2)))
    g0, g4 = tl.split(e0)
    g2, g6 = tl.split(e1)
    g1, g5 = tl.split(o0)
    g3, g7 = tl.split(o1)
    aden = _pack(tl.broadcast_to(alpha[:, None], (BM, 4)))
    denominator = tl.fma(denominator, aden, _pack(g0))
    denominator = denominator + _pack(g1)
    denominator = denominator + _pack(g2)
    denominator = denominator + _pack(g3)
    denominator = denominator + _pack(g4)
    denominator = denominator + _pack(g5)
    denominator = denominator + _pack(g6)
    denominator = denominator + _pack(g7)
    p = (
        probability.reshape((BM, 8, 2, 4, 2))
        .permute((0, 1, 3, 2, 4))
        .reshape((BM, BN))
        .to(tl.float8e4nv)
    )
    if V_SINGLE:
        tle.gpu.barrier_wait(VBAR, phaseIdx=start % 2)
    temporary = tle.gpu.wgmma(
        p, vsm, out_dtype=tl.float32, max_num_imprecise_acc=128, trans_b=True
    )
    temporary = tle.gpu.wgmma_wait(0, temporary)
    if V_SINGLE and not LAST:
        # Previous PV has finished reading V. Next V loads during the next QK
        # and softmax; consumer waits only immediately before its next PV.
        tle.gpu.copy(
            VDESC, vsm, [D, BN], [(b * H + h) * D, (start + 1) * BN], barrier=VBAR
        )
    accumulator = tl.fma(accumulator, alpha[:, None], temporary)
    maximum = new_maximum
    return maximum, denominator, accumulator


@triton.jit(do_not_specialize=["NQ", "NK"])
def _tlepeel_fwd(
    Q,
    K,
    V,
    QS,
    KS,
    VS,
    O,
    H: tl.constexpr,
    NQ,
    NK,
    NQS: tl.constexpr,
    NKS: tl.constexpr,
    NP: tl.constexpr,
    D: tl.constexpr,
    SM: tl.constexpr,
    BM: tl.constexpr = 64,
    BN: tl.constexpr = 128,
    TMA: tl.constexpr = False,
    PIPE: tl.constexpr = False,
    LOCAL_DEN: tl.constexpr = False,
    GROUP_SCALE: tl.constexpr = False,
    QDOUBLE: tl.constexpr = True,
    OUT_LAYOUT: tl.constexpr = False,
    DEN_SHFL: tl.constexpr = False,
    K_SINGLE: tl.constexpr = False,
    EARLY_K: tl.constexpr = False,
    V_SINGLE: tl.constexpr = False,
):
    h = tl.program_id(1)
    b = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    n = tl.arange(0, BN)
    q = tl.load(
        Q + ((b * H + h) * NQ + m[:, None]) * D + d[None, :], m[:, None] < NQ, other=0
    )
    qs = tl.load(QS + (b * H + h) * NQS + m // 16 * 8 + m % 8, m < NQ, other=0.0)
    qbuf = tle.gpu.alloc((2 if QDOUBLE else 1, BM, D), tl.int8, init_value=q)
    qsm = qbuf.slot(0)
    qlast = qbuf.slot(1 if QDOUBLE else 0)
    STAGES: tl.constexpr = 2 if PIPE else 1
    KSTAGES: tl.constexpr = 1 if K_SINGLE else STAGES
    kbuf = tle.gpu.alloc((KSTAGES, BN, D), tl.int8)
    VSTAGES: tl.constexpr = 1 if V_SINGLE else STAGES
    vbuf = tle.gpu.alloc((VSTAGES, D, BN), tl.float8e4nv)
    if TMA:
        kb = tle.gpu.alloc_barriers(KSTAGES, expect_bytes=BN * D)
        vb = tle.gpu.alloc_barriers(VSTAGES, expect_bytes=BN * D)
        if PIPE:
            tle.gpu.copy(K, kbuf.slot(0), [BN, D], [(b * H + h) * NK, 0], barrier=kb[0])
            tle.gpu.copy(V, vbuf.slot(0), [D, BN], [(b * H + h) * D, 0], barrier=vb[0])
    maximum = tl.full((BM,), -5000000.0, tl.float32)
    denominator = tl.full((256,), 1.0, tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    sm = tl.full((), SM, tl.float32) * 1.4426950408889634
    for start in range(tl.cdiv(NK, BN) - 1):
        slot = start % 2
        kslot = 0 if K_SINGLE else slot
        vslot = 0 if V_SINGLE else slot
        ksm = kbuf.slot(kslot)
        vsm = vbuf.slot(vslot)
        tle.gpu.barrier_wait(
            kb[kslot], phaseIdx=start % 2 if K_SINGLE else (start // 2) % 2
        )
        if not V_SINGLE:
            tle.gpu.barrier_wait(vb[vslot], phaseIdx=(start // 2) % 2)
        nxt = (start + 1) % 2
        if not K_SINGLE:
            tle.gpu.copy(
                K,
                kbuf.slot(nxt),
                [BN, D],
                [(b * H + h) * NK + (start + 1) * BN, 0],
                barrier=kb[nxt],
            )
        if not V_SINGLE:
            tle.gpu.copy(
                V,
                vbuf.slot(nxt),
                [D, BN],
                [(b * H + h) * D, (start + 1) * BN],
                barrier=vb[nxt],
            )
        maximum, denominator, accumulator = _step(
            qsm,
            ksm,
            vsm,
            KS,
            qs,
            maximum,
            denominator,
            accumulator,
            start,
            NK,
            NKS,
            H,
            b,
            h,
            sm,
            n,
            D,
            BM,
            BN,
            True,
            False,
            K,
            kb[kslot],
            K_SINGLE,
            EARLY_K,
            V,
            vb[vslot],
            V_SINGLE,
        )
    start = tl.cdiv(NK, BN) - 1
    slot = start % 2
    kslot = 0 if K_SINGLE else slot
    vslot = 0 if V_SINGLE else slot
    ksm = kbuf.slot(kslot)
    vsm = vbuf.slot(vslot)
    tle.gpu.barrier_wait(
        kb[kslot], phaseIdx=start % 2 if K_SINGLE else (start // 2) % 2
    )
    if not V_SINGLE:
        tle.gpu.barrier_wait(vb[vslot], phaseIdx=(start // 2) % 2)
    maximum, denominator, accumulator = _step(
        qlast,
        ksm,
        vsm,
        KS,
        qs,
        maximum,
        denominator,
        accumulator,
        start,
        NK,
        NKS,
        H,
        b,
        h,
        sm,
        n,
        D,
        BM,
        BN,
        True,
        True,
        K,
        kb[kslot],
        K_SINGLE,
        EARLY_K,
        V,
        vb[vslot],
        V_SINGLE,
    )
    if DEN_SHFL:
        # The packed denominator owns two rows per thread and four contiguous
        # lanes per row. Keep that ownership through the final sum and reciprocal.
        local = denominator
        total = tl.inline_asm_elementwise(
            "{ .reg .f32 t,u; shfl.sync.bfly.b32 t,$1,1,31,-1; add.rn.f32 u,$1,t; "
            "shfl.sync.bfly.b32 t,u,2,31,-1; add.rn.f32 $0,u,t; }",
            constraints="=f,f",
            args=[local],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        inv = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0,$1;",
            constraints="=f,f",
            args=[total],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        inv4 = inv.reshape((2, 4, 8, 4)).permute((1, 0, 2, 3)).reshape((BM, 4))
        inverse = (
            tl.broadcast_to(inv4[:, :, None], (BM, 4, 32))
            .reshape((BM, 4, 8, 2, 2))
            .permute((0, 2, 3, 1, 4))
            .reshape((BM, D))
        )
    else:
        den = denominator.reshape((2, 4, 8, 4)).permute((1, 0, 2, 3)).reshape((BM, 4))
        a, z = tl.split(den.reshape((BM, 2, 2)))
        a, z = tl.split(a + z)
        total = a + z
        inverse = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0,$1;",
            constraints="=f,f",
            args=[total],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )[:, None]
    vscale = tl.load(VS + (b * H + h) * D + d)
    value = (accumulator * inverse) * vscale[None, :]
    ptr = O + ((b * NQ + m[:, None]) * H + h) * D + d[None, :]
    mask = tl.broadcast_to(m[:, None] < NQ, (BM, D))
    if OUT_LAYOUT:
        # Keep the MMA ownership by emitting stores at the elementwise consumer.
        # set_layout on a reshaped tensor currently fails the WGMMA TTIR verifier.
        _ = tl.inline_asm_elementwise(
            "{ .reg .pred p; setp.ne.u32 p,$3,0; @p st.global.b16 [$1],$2; mov.u32 $0,0; }",
            constraints="=r,l,h,r",
            args=[
                ptr.to(tl.uint64),
                value.to(O.dtype.element_ty).to(tl.uint16, bitcast=True),
                mask.to(tl.int32),
            ],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.store(ptr, value, mask)


def core(
    z,
    output,
    sm_scale=None,
    *,
    out_layout=False,
    den_shfl=False,
    k_single=False,
    qdouble=True,
    maxnreg=None,
    early_k=False,
    v_single=False,
):
    """Launch the validated 64x128 INT8/FP8 path; PDL is disabled."""
    qi, qs, ki, ks, vf, vs, _ = z
    b, h, nq, d = qi.shape
    kd = TensorDescriptor(ki, [b * h * ki.shape[2], d], [d, 1], [128, d])
    vd = TensorDescriptor(vf, [b * h * d, vf.shape[-1]], [vf.shape[-1], 1], [d, 128])
    return _tlepeel_fwd[(triton.cdiv(nq, 64), h, b)](
        qi,
        kd,
        vd,
        qs,
        ks,
        vs,
        output,
        h,
        nq,
        ki.shape[2],
        qs.shape[-1],
        ks.shape[-1],
        vf.shape[-1],
        d,
        d**-0.5 if sm_scale is None else sm_scale,
        TMA=True,
        PIPE=True,
        LOCAL_DEN=False,
        GROUP_SCALE=True,
        QDOUBLE=qdouble,
        OUT_LAYOUT=out_layout,
        DEN_SHFL=den_shfl,
        K_SINGLE=k_single,
        EARLY_K=early_k,
        V_SINGLE=v_single,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        maxnreg=maxnreg,
    )
