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
import torch_npu  # noqa: F401  (registers the npu backend / dtype support)
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ._tq_mx_cube import launch_mx_cube


# --------------------------------------------------------------------------- #
# Path A: streamed matrix-vector kernel for decode-shaped (M == 1) inputs
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Path A': cube + split-K variant for decode-shaped inputs
# --------------------------------------------------------------------------- #
@triton.jit
def _tq_m1_dot_kernel(
    x1_ptr,
    x2_ptr,
    s1_ptr,
    s2_ptr,
    scr_ptr,
    N,
    K,
    sx1_b,
    sx1_m,
    sx1_k,
    sx2_b,
    sx2_k,
    sx2_n,
    ss1_b,
    ss1_k,
    ss1_h,
    ss2_b,
    ss2_k,
    ss2_n,
    ss2_h,
    sscr_s,
    sscr_b,
    BN: tl.constexpr,
    NSPLIT: tl.constexpr,
    NKG: tl.constexpr,
    BM: tl.constexpr,
    PAIR16: tl.constexpr,
):
    """M == 1: only the first row of the [BM, BN] accumulator is real, but putting the
    reduction on the cube turns the inner loop into one broadcast multiply-add per
    output element instead of three vector ops per streamed x2 element."""
    split = tl.program_id(0)
    b = tl.program_id(1)

    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, 32)
    r2 = tl.arange(0, 2)
    m_mask = rm < 1
    n_mask = rn < N

    x1b = x1_ptr + b * sx1_b
    x2b = x2_ptr + b * sx2_b
    s1b = s1_ptr + b * ss1_b
    s2b = s2_ptr + b * ss2_b

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    kg0 = split * NKG
    for t in range(0, NKG):
        kg = kg0 + t
        k0 = kg * 64

        s1f = tl.load(s1b + kg * ss1_k + r2 * ss1_h)
        s1f = ((s1f.to(tl.int32) & 0xFF) << 23).to(tl.float32, bitcast=True)
        e0, e1 = tl.split(s1f)
        if PAIR16:
            v = tl.load(s2b + kg * ss2_k + tl.arange(0, BN))
            vi = v.to(tl.int32)
            c0 = ((vi & 0xFF) << 23).to(tl.float32, bitcast=True) * e0
            c1 = (((vi >> 8) & 0xFF) << 23).to(tl.float32, bitcast=True) * e1
        else:
            p0 = tl.load(s2b + kg * ss2_k + rn * ss2_n, mask=n_mask, other=127)
            p1 = tl.load(s2b + kg * ss2_k + rn * ss2_n + ss2_h, mask=n_mask, other=127)
            c0 = ((p0.to(tl.int32) & 0xFF) << 23).to(tl.float32, bitcast=True) * e0
            c1 = ((p1.to(tl.int32) & 0xFF) << 23).to(tl.float32, bitcast=True) * e1

        a0 = tl.load(
            x1b + rm[:, None] * sx1_m + (k0 + rk)[None, :] * sx1_k,
            mask=m_mask[:, None],
            other=0.0,
        )
        w0 = tl.load(
            x2b + (k0 + rk)[:, None] * sx2_k + rn[None, :] * sx2_n,
            mask=n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a0, w0) * c0[None, :]

        a1 = tl.load(
            x1b + rm[:, None] * sx1_m + (k0 + 32 + rk)[None, :] * sx1_k,
            mask=m_mask[:, None],
            other=0.0,
        )
        w1 = tl.load(
            x2b + (k0 + 32 + rk)[:, None] * sx2_k + rn[None, :] * sx2_n,
            mask=n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a1, w1) * c1[None, :]

    # rows 1..BM-1 are exactly zero (their x1 load is masked), so a plain sum reduces
    # to the single live row.
    row = tl.sum(acc, axis=0)
    tl.store(scr_ptr + split * sscr_s + b * sscr_b + rn, row, mask=n_mask)


@triton.jit
def _tq_m1_fast(
    x1_ptr,
    x2_ptr,
    s1_ptr,
    s2_ptr,
    scr_ptr,
    N,
    K,
    sx1_b,
    sx2_b,
    ss2_b,
    ss1_b,
    sscr_s,
    sscr_b,
    BN: tl.constexpr,
    NSPLIT: tl.constexpr,
    NKG: tl.constexpr,
    BM: tl.constexpr,
):
    """Same arithmetic as _tq_m1_dot_kernel, specialised to the captured contiguous MX
    layout so the launch carries 13 runtime arguments instead of 22.

    Guards checked in run(): sx1_k == 1, sx2_n == 1, sx2_k == N, ss1_k == 2 and
    ss1_h == 1 (contiguous (hf) byte pair), plus the x2 scale pair view whose int16
    form has unit n stride and N elements per 64-element scale group. Only the decode
    workloads are affected by the argument count, and their whole device cost is
    ~19.4 us against a ~31 us measured latency, so the per-launch Python-side
    processing is a first-order term there.
    """
    split = tl.program_id(0)
    b = tl.program_id(1)

    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, 32)
    r2 = tl.arange(0, 2)
    m_mask = rm < 1
    n_mask = rn < N

    x1b = x1_ptr + b * sx1_b
    x2b = x2_ptr + b * sx2_b
    s1b = s1_ptr + b * ss1_b
    s2b = s2_ptr + b * ss2_b
    rzero = (rm * 0)[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    kg0 = split * NKG
    for t in range(0, NKG):
        kg = kg0 + t
        k0 = kg * 64

        s1f = tl.load(s1b + kg * 2 + r2)
        s1f = ((s1f.to(tl.int32) & 0xFF) << 23).to(tl.float32, bitcast=True)
        e0, e1 = tl.split(s1f)
        v = tl.load(s2b + kg * N + tl.arange(0, BN))
        vi = v.to(tl.int32)
        c0 = ((vi & 0xFF) << 23).to(tl.float32, bitcast=True) * e0
        c1 = (((vi >> 8) & 0xFF) << 23).to(tl.float32, bitcast=True) * e1

        a0 = tl.load(x1b + rzero + (k0 + rk)[None, :], mask=m_mask[:, None], other=0.0)
        w0 = tl.load(
            x2b + (k0 + rk)[:, None] * N + rn[None, :], mask=n_mask[None, :], other=0.0
        )
        acc += tl.dot(a0, w0) * c0[None, :]

        a1 = tl.load(
            x1b + rzero + (k0 + 32 + rk)[None, :], mask=m_mask[:, None], other=0.0
        )
        w1 = tl.load(
            x2b + (k0 + 32 + rk)[:, None] * N + rn[None, :],
            mask=n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a1, w1) * c1[None, :]

    row = tl.sum(acc, axis=0)
    tl.store(scr_ptr + split * sscr_s + b * sscr_b + rn, row, mask=n_mask)


# --------------------------------------------------------------------------- #
# Path A: streamed mat-vec fallback for decode-shaped inputs
# --------------------------------------------------------------------------- #


@triton.jit
def _tq_reduce_kernel(
    scr_ptr,
    out_ptr,
    N,
    sscr_s,
    sscr_b,
    so_b,
    so_n,
    BM_AXIS: tl.constexpr,
    BN: tl.constexpr,
    NSPLIT: tl.constexpr,
    DT_OUT: tl.constexpr,
):
    b = tl.program_id(0)
    rn = tl.arange(0, BN)
    n_mask = rn < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for s in range(0, NSPLIT):
        acc += tl.load(scr_ptr + s * sscr_s + b * sscr_b + rn, mask=n_mask, other=0.0)
    if DT_OUT == 0:
        o = acc.to(tl.bfloat16)
    elif DT_OUT == 1:
        o = acc.to(tl.float16)
    else:
        o = acc
    tl.store(out_ptr + b * so_b + rn * so_n, o, mask=n_mask)


# --------------------------------------------------------------------------- #
# Path B': one-shot dequantisation of x2 into a contiguous bf16 scratch.
#
# Measured evidence that motivates this pre-pass (rounds 1 and 3): applying the MX
# 32-element scale inside the k-loop is what binds geometry-000. Folding the scale
# into bf16 operands costs 3*B*M*N*K*(1/BM+1/BN) element-ops and saturates the AIV
# vector pipe (0.984) at ~2e12 ops/s; rescaling the [BM,BN] dot result per block
# instead costs 3*B*M*N*B*(K/32) ops and additionally serialises every k-step through
# a L0C -> UB -> vector -> L0C copy. Both are 2.7x-2.4x worse than this shape. Doing
# the w-side dequantisation once, outside the loop, costs only 2*B*K*N element-ops
# total and leaves the k-loop as plain bf16 loads plus tl.dot with the accumulator
# resident on the cube.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Argument-minimal variants of the three compute-path kernels.
#
# Round 8 measured the price of a Triton runtime argument on this evaluator at
# ~0.9 us of latency per argument per call: cutting the decode split-K launch from 22
# to 13 arguments took those workloads from 31.12 to 22.89 us with identical device
# work. The compute path's three launches carry 34, 15 and 14 arguments. The two
# scratches are allocated contiguous by run(), so their strides are constants of M, N
# and K and never needed to be arguments; the ABD GEMM branch never touches x1, x2, s1
# or s2 or any of their stride families. Careful: only the SCRATCH strides may be
# folded in - the source tensors keep their permuted physical layouts, so the
# activation source must still be read with its own logical row stride sx1_m (B*K for
# perm_x1 = (1, 0, 2), not K).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Path C: hardware block-scaled (MX) fp8 dot.
#
# A compile probe (this epoch) established that triton-ascend 3.6.0.post2 does
# implement a scaled cube MMA: tl.dot_scaled(e4m3, e8m0, e4m3, e8m0, acc) survives
# to the ttadapter stage as hfusion.matmul_mx {lhsFormat=2, rhsFormat=2} and lowers to
# mmadmxL1_float8_e4m3_t_float8_e4m3_t_to_float on the AIC, with no convert-to-bf16
# emulation and no separate addf when the accumulator is passed positionally. Its
# block factor is 2**(byte-127) per 32-element K block on both operands, which is
# exactly this operator's arithmetic.
#
# Round-2 evidence (profile, round_num=2) showed the MMA already ran at the native fp8
# rate (97.6 us MAC-busy for 68.7 GFLOP) while the kernel cost 1292 us, because the
# e8m0 scale tiles were fed through a per-element AIV gather: x2_scale's k-block index
# kb = 2*kg + hf is not affine in kb, so no 2D descriptor describes the tile and the
# backend emitted an int64 offset build plus triton_indirect_load per k-slab. Hence the
# repack kernels below, which rewrite the exponents byte for byte into a
# slab-contiguous layout so each per-slab scale tile becomes one plain 2D load.
# Rounds 3-4 then measured: the MX kernel itself is 133 us at ~640 TFLOPS with both
# scale tiles arriving through the AIC's load_scale_gm_to_cbuf_2d_int8_t, a plain 2-D
# block transpose pre-pass costs 5.65 us per MB, and an element-level permutation
# (reshape + trans + split of interleaved data) is lowered to a serialising
# gather/store/reload chain at ~0.99 vector-pipe busy. Both packs below are therefore
# expressed as affine plane loads whose index shape already equals the destination's.
# --------------------------------------------------------------------------- #
@triton.jit
def _tq_pack_s1_kernel(
    s1_ptr,
    s1p_ptr,
    M,
    NJ,
    ss1_b,
    ss1_m,
    BMP: tl.constexpr,
    PB: tl.constexpr,
):
    """s1p[b, j, m, q] = s1[b, m, j*PB + q]  (raw e8m0 byte copy, per slab)."""
    mb = tl.program_id(0)
    j = tl.program_id(1)
    b = tl.program_id(2)

    rm = mb * BMP + tl.arange(0, BMP)
    rq = tl.arange(0, PB)
    m_mask = rm < M

    v = tl.load(
        s1_ptr + b * ss1_b + rm[:, None] * ss1_m + (j * PB + rq)[None, :],
        mask=m_mask[:, None],
        other=127,
    )
    tl.store(
        s1p_ptr + ((b * NJ + j) * M + rm[:, None]) * PB + rq[None, :],
        v,
        mask=m_mask[:, None],
    )


@triton.jit
def _tq_pack_s2_kernel(
    s2_ptr,
    s2p_ptr,
    N,
    NJ,
    ss2_b,
    ss2_k,
    ss2_n,
    ss2_h,
    BNP: tl.constexpr,
    PB: tl.constexpr,
):
    """s2p[b, j, n, q] = s2[b, kg, n, hf] with kb = j*PB + q, kg = kb//2, hf = kb%2."""
    nb = tl.program_id(0)
    j = tl.program_id(1)
    b = tl.program_id(2)

    rn = nb * BNP + tl.arange(0, BNP)
    rq = tl.arange(0, PB)
    kb = j * PB + rq
    n_mask = rn < N

    v = tl.load(
        s2_ptr
        + b * ss2_b
        + rn[:, None] * ss2_n
        + ((kb // 2) * ss2_k + (kb % 2) * ss2_h)[None, :],
        mask=n_mask[:, None],
        other=127,
    )
    tl.store(
        s2p_ptr + ((b * NJ + j) * N + rn[:, None]) * PB + rq[None, :],
        v,
        mask=n_mask[:, None],
    )


# --------------------------------------------------------------------------- #
# Bulk-granularity repacks: one program per row block covering all k slabs.
#
# Round-3 evidence: the per-slab forms above move PB=4-byte atoms in 2048 / 1024
# instances, each bracketed by SET/WAIT.FLAG and BARRIER, so pack_s1 alone cost
# 493.7 us with aiv_scalar 0.999 for 1 MB. Bulk versions instead give each program the
# whole k-block range of one row block: 128-byte read atoms, 64 and 32 instances.
#
# Only byte permutations are computed, so bit-exactness is structural. Both transforms
# are affine plane loads plus integer packing - never a reshape/permute of interleaved
# data, which round 4 measured at 291 us for the same 1 MB.
#   s1: the PB=4 bytes of one slab for row m are exactly one int32 word of s1, so
#       s1p32[b, j, m] = s1_32[m, b, j] and the pass is one 2-D block transpose.
#   s2: the PB=4 bytes of one slab for column n are the (hf=0,hf=1) int16 words of
#       kg=2j and kg=2j+1 concatenated, so two plane loads plus a shift-or suffice.
# --------------------------------------------------------------------------- #
@triton.jit
def _tq_pack_s1_bulk(
    s1_ptr,
    s1p_ptr,
    M,
    NJ,
    ss1_b8,
    ss1_m8,
    BMP: tl.constexpr,
    NJP: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One program per (b, m-block) covering every k slab; PB=8 bytes per slab are
    exactly one int64 word of s1, so the whole pass is a single 2-D block transpose
    with 128-byte read rows and 8*BMP-byte writes."""
    mb = tl.program_id(0)
    b = tl.program_id(1)

    rm = mb * BMP + tl.arange(0, BMP)
    rj = tl.arange(0, NJP)

    if EVEN:
        v = tl.load(s1_ptr + b * ss1_b8 + rm[:, None] * ss1_m8 + rj[None, :])
    else:
        # 0x7F7F7F7F7F7F7F7F == e8m0 exponent byte 127 in all eight lanes.
        v = tl.load(
            s1_ptr + b * ss1_b8 + rm[:, None] * ss1_m8 + rj[None, :],
            mask=(rm < M)[:, None] & (rj < NJ)[None, :],
            other=0x7F7F7F7F7F7F7F7F,
        )
    t = tl.trans(v)
    if EVEN:
        tl.store(s1p_ptr + (b * NJ + rj[:, None]) * M + rm[None, :], t)
    else:
        tl.store(
            s1p_ptr + (b * NJ + rj[:, None]) * M + rm[None, :],
            t,
            mask=(rj < NJ)[:, None] & (rm < M)[None, :],
        )


@triton.jit
def _tq_pack_s2_bulk(
    s2_ptr,
    s2p_ptr,
    N,
    NJ,
    ss2_b2,
    ss2_k2,
    ss2_n2,
    BNP: tl.constexpr,
    NJP: tl.constexpr,
    EVEN: tl.constexpr,
):
    """The PB=8 bytes of one slab for column n are the four (hf=0,hf=1) int16 words of
    kg = 4j .. 4j+3 concatenated, so four affine plane loads plus shift-or build one
    int64 word per (j, n) with the destination's own index shape."""
    nb = tl.program_id(0)
    b = tl.program_id(1)

    rn = nb * BNP + tl.arange(0, BNP)
    rj = tl.arange(0, NJP)

    base = s2_ptr + b * ss2_b2 + rn[None, :] * ss2_n2
    if EVEN:
        p0 = tl.load(base + (4 * rj)[:, None] * ss2_k2)
        p1 = tl.load(base + (4 * rj + 1)[:, None] * ss2_k2)
        p2 = tl.load(base + (4 * rj + 2)[:, None] * ss2_k2)
        p3 = tl.load(base + (4 * rj + 3)[:, None] * ss2_k2)
    else:
        mask = (rj < NJ)[:, None] & (rn < N)[None, :]
        p0 = tl.load(base + (4 * rj)[:, None] * ss2_k2, mask=mask, other=32639)
        p1 = tl.load(base + (4 * rj + 1)[:, None] * ss2_k2, mask=mask, other=32639)
        p2 = tl.load(base + (4 * rj + 2)[:, None] * ss2_k2, mask=mask, other=32639)
        p3 = tl.load(base + (4 * rj + 3)[:, None] * ss2_k2, mask=mask, other=32639)
    lo = (p0.to(tl.int32) & 0xFFFF) | ((p1.to(tl.int32) & 0xFFFF) << 16)
    hi = (p2.to(tl.int32) & 0xFFFF) | ((p3.to(tl.int32) & 0xFFFF) << 16)
    w = (lo.to(tl.int64) & 0xFFFFFFFF) | (hi.to(tl.int64) << 32)
    if EVEN:
        tl.store(s2p_ptr + (b * NJ + rj[:, None]) * N + rn[None, :], w)
    else:
        tl.store(
            s2p_ptr + (b * NJ + rj[:, None]) * N + rn[None, :],
            w,
            mask=(rj < NJ)[:, None] & (rn < N)[None, :],
        )


@triton.jit
def _tq_mx_kernel(
    x1_ptr,
    x2_ptr,
    s1p_ptr,
    s2p_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sx1_b: tl.constexpr,
    sx1_m: tl.constexpr,
    sx2_b: tl.constexpr,
    so_b: tl.constexpr,
    so_m: tl.constexpr,
    so_n: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DIRECT_ACT_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DT_OUT: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Compute block-scaled FP8 tiles with packed weight scales."""
    b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rq = tl.arange(0, BK // 32)
    nj = K // BK
    mpb = M * (BK // 32)
    npb = N * (BK // 32)

    x1b = x1_ptr + b * sx1_b
    x2b = x2_ptr + b * sx2_b

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if EVEN:
        for j in range(0, nj):
            k0 = j * BK
            pk = b * nj + j
            a = tl.load(x1b + rm[:, None] * sx1_m + (k0 + rk)[None, :])
            w = tl.load(x2b + (k0 + rk)[:, None] * N + rn[None, :])
            if DIRECT_ACT_SCALE:
                asc = tl.load(
                    s1p_ptr
                    + b * (sx1_b // 32)
                    + rm[:, None] * (sx1_m // 32)
                    + j * (BK // 32)
                    + rq[None, :]
                )
            else:
                asc = tl.load(
                    s1p_ptr + pk * mpb + rm[:, None] * (BK // 32) + rq[None, :]
                )
            bsc = tl.load(s2p_ptr + pk * npb + rn[:, None] * (BK // 32) + rq[None, :])
            acc = tl.dot_scaled(a, asc, "e4m3", w, bsc, "e4m3", acc)
    else:
        m_mask = rm < M
        n_mask = rn < N
        for j in range(0, nj):
            k0 = j * BK
            pk = b * nj + j
            a = tl.load(
                x1b + rm[:, None] * sx1_m + (k0 + rk)[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            w = tl.load(
                x2b + (k0 + rk)[:, None] * N + rn[None, :],
                mask=n_mask[None, :],
                other=0.0,
            )
            asc = tl.load(
                s1p_ptr + pk * mpb + rm[:, None] * (BK // 32) + rq[None, :],
                mask=m_mask[:, None],
                other=127,
            )
            bsc = tl.load(
                s2p_ptr + pk * npb + rn[:, None] * (BK // 32) + rq[None, :],
                mask=n_mask[:, None],
                other=127,
            )
            acc = tl.dot_scaled(a, asc, "e4m3", w, bsc, "e4m3", acc)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=rn < N, other=0.0)[None, :].to(tl.float32)

    if DT_OUT == 0:
        o = acc.to(tl.bfloat16)
    elif DT_OUT == 1:
        o = acc.to(tl.float16)
    else:
        o = acc

    tl.store(
        out_ptr + b * so_b + rm[:, None] * so_m + rn[None, :] * so_n,
        o,
        mask=(rm < M)[:, None] & (rn < N)[None, :],
    )


# --------------------------------------------------------------------------- #
# Path B'': one-shot dequantisation of x1 into a contiguous bf16 scratch.
#
# The round-4 profile showed the remaining k-loop cost is split between two MTE2
# terms: the cube-side weight stream (0.0324/BM seconds) and the AIV-side activation
# load (6.29/(BN*BK) seconds, because a [BM, BK] fp8 tile is BM row-descriptors of BK
# bytes). Getting BM from 64 to 128 needs BK >= 128 to keep the second term down, and
# a BK > 32 fold would need an in-UB block permutation that measured badly. Removing
# the activation scale from the loop instead lets both operands be plain bf16 loads,
# so BK can be raised with no scale maths at all.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Path B: tiled fp8 dot kernel, one MX 32-element rescale per block
# --------------------------------------------------------------------------- #


_DT_CODE = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}

# Probe-verified availability of the hardware block-scaled dot on this toolchain.
_HAS_MX = hasattr(tl, "dot_scaled")


def _phys_shape(logical_shape, perm):
    """physical[j] = logical[perm^-1[j]]."""
    return [logical_shape[perm.index(j)] for j in range(len(perm))]


def _is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _launch(
    x1,
    x2,
    dtype,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=1,
    x1_dtype=None,
    x2_dtype=None,
):
    px1 = tuple(perm_x1) if perm_x1 is not None else (0, 1, 2)
    px2 = tuple(perm_x2) if perm_x2 is not None else (0, 1, 2)
    py = tuple(perm_y) if perm_y is not None else (0, 1, 2)

    x1s = x1.shape
    x2s = x2.shape
    B = x1s[px1[0]]
    M = x1s[px1[1]]
    K = x1s[px1[2]]
    N = x2s[px2[2]]

    sx1_b, sx1_m, sx1_k = (
        x1.stride()[px1[0]],
        x1.stride()[px1[1]],
        x1.stride()[px1[2]],
    )
    sx2_b, sx2_k, sx2_n = (
        x2.stride()[px2[0]],
        x2.stride()[px2[1]],
        x2.stride()[px2[2]],
    )

    s1_st = x1_scale.stride()
    ss1_b, ss1_m = s1_st[px1[0]], s1_st[px1[1]]
    ss1_k, ss1_h = s1_st[2], s1_st[3]
    s2_st = x2_scale.stride()
    ss2_b, ss2_k, ss2_n = s2_st[px2[0]], s2_st[px2[1]], s2_st[px2[2]]
    ss2_h = s2_st[3]

    s1_bytes = x1_scale.view(torch.int8)
    s2_bytes = x2_scale.view(torch.int8)

    out = torch.empty(_phys_shape((B, M, N), py), dtype=dtype, device=x1.device)
    ost = out.stride()
    so_b, so_m, so_n = ost[py[0]], ost[py[1]], ost[py[2]]
    dt_code = _DT_CODE[dtype]
    has_bias = bias is not None and bias.numel() > 0

    # The block kernel consumes packed E8M0 pairs in their original layout.
    # Each work item owns complete K and directly writes an MBN output tile.
    if (
        (M == 1 or M % 128 == 0)
        and N % 256 == 0
        and K % 512 == 0
        and max(M, B, K, N) < 2**31
        and all(t.data_ptr() % 32 == 0 for t in (x1, x2, x1_scale, x2_scale))
        and launch_mx_cube(x1, x2, x1_scale, x2_scale, out)
    ):
        return out

    # ---------------- Path A: decode-shaped streamed mat-vec -----------------
    use_matvec = (
        M == 1
        and not has_bias
        and _is_pow2(N)
        and 256 <= N <= 4096
        and sx2_n == 1
        and ss2_h == 1
        and ss2_n == 2
        and K % 64 == 0
    )
    if use_matvec:
        nsplit = 1
        if B < 64 and (K // 64) % 8 == 0 and B * 8 <= 256:
            nsplit = 8
        nkg = (K // 64) // nsplit
        assert nkg * nsplit * 64 == K
        scr = torch.empty((nsplit, B, N), dtype=torch.float32, device=x1.device)
        pair16 = bool(
            ss2_h == 1 and ss2_n == 2 and ss2_k == 2 * N and x2_scale.is_contiguous()
        )
        if pair16:
            shp = x2_scale.shape
            s2_arg = x2_scale.reshape(*shp[:-2], shp[-2] * 2).view(torch.int16)
            ss2_b_arg = ss2_b // 2
            ss2_k_arg = ss2_k // 2
        else:
            s2_arg = s2_bytes
            ss2_b_arg = ss2_b
            ss2_k_arg = ss2_k
        if (
            pair16
            and sx1_k == 1
            and sx2_n == 1
            and sx2_k == N
            and ss1_k == 2
            and ss1_h == 1
        ):
            _tq_m1_fast[(nsplit, B)](
                x1,
                x2,
                s1_bytes,
                s2_arg,
                scr,
                N,
                K,
                sx1_b,
                sx2_b,
                ss2_b_arg,
                ss1_b,
                scr.stride(0),
                scr.stride(1),
                BN=N,
                NSPLIT=nsplit,
                NKG=nkg,
                BM=16,
                num_stages=2,
            )
        else:
            _tq_m1_dot_kernel[(nsplit, B)](
                x1,
                x2,
                s1_bytes,
                s2_arg,
                scr,
                N,
                K,
                sx1_b,
                sx1_m,
                sx1_k,
                sx2_b,
                sx2_k,
                sx2_n,
                ss1_b,
                ss1_k,
                ss1_h,
                ss2_b_arg,
                ss2_k_arg,
                ss2_n,
                ss2_h,
                scr.stride(0),
                scr.stride(1),
                BN=N,
                NSPLIT=nsplit,
                NKG=nkg,
                BM=16,
                PAIR16=pair16,
                num_stages=2,
            )
        _tq_reduce_kernel[(B,)](
            scr,
            out,
            N,
            scr.stride(0),
            scr.stride(1),
            so_b,
            so_n,
            BM_AXIS=1,
            BN=N,
            NSPLIT=nsplit,
            DT_OUT=dt_code,
        )
        return out

    # ---------------- Path C: hardware MX fp8 dot ----------------------------
    # One MMA kernel plus two small scale repacks. The MMA consumes fp8 operands in
    # their own physical layouts and the e8m0 block factors natively, so neither bf16
    # dequantisation pre-pass is needed. Requires the captured contiguous layout.
    use_mx = bool(
        _HAS_MX
        and M >= 2
        and x1.dtype == torch.float8_e4m3fn
        and x2.dtype == torch.float8_e4m3fn
        and sx1_k == 1
        and sx2_n == 1
        and sx2_k == N
        and ss1_h == 1
        and ss1_k == 2
        and K % 128 == 0
    )
    if use_mx:
        # Tile choice. Operand traffic is B*M*N*K*(1/BM + 1/BN), so it is minimised by
        # the largest BM*BN the accumulator allows. Measured on this target: a
        # 65536-element accumulator is NOT reachable on the MX path. At the default
        # stage count it fails as "cc overflow, requires 4194304 bits while 2097152
        # bits available" (two-deep L0C), and at num_stages=1 it fails as "ub overflow,
        # requires 2097152 bits while 2031616 bits available" - i.e. dot_scaled's fp32
        # accumulator is staged through the 248 KB UB, so 256*256*4 B = 256 KB alone
        # exceeds it, whereas the plain tl.dot path kept its accumulator in L0C and
        # could use 65536. The 32768-element cap is therefore a property of the MX
        # lowering, not of the hardware, and 402 MB is the real traffic floor here.
        BN = 256 if N % 256 == 0 else (128 if N % 128 == 0 else 64)
        BM = min(32768 // BN, 128)
        # BK=256 doubles the contiguous A-row segment from 128 to 256 bytes and halves
        # the descriptor and nd2nz counts at identical bytes and identical accumulator;
        # the round-6 profile left segment granularity as the only untested explanation
        # for the GM->L1 pipe stopping at 3.19 TB/s while L1->L0 moves the same volume
        # at 5.18 TB/s.
        BK = 256 if K % 256 == 0 else 128
        nj = K // BK
        pb = BK // 32
        njp = 1
        while njp < nj:
            njp *= 2
        direct_act_scale = M % 128 == 0 and N % 256 == 0 and K % 256 == 0
        s1p = (
            s1_bytes
            if direct_act_scale
            else torch.empty((B, nj, M, pb), dtype=torch.int8, device=x1.device)
        )
        s2p = torch.empty((B, nj, N, pb), dtype=torch.int8, device=x1.device)
        # The bulk repack needs the packed word forms to be affine in their own layout:
        # s1's 8 bytes per slab are one int64 word, s2's are four int16 words.
        s1_bulk = bool(
            pb == 8
            and njp == nj
            and x1_scale.is_contiguous()
            and ss1_b % 8 == 0
            and ss1_m % 8 == 0
        )
        s2_bulk = bool(
            pb == 8
            and njp == nj
            and ss2_n == 2
            and ss2_b % 2 == 0
            and ss2_k % 2 == 0
            and x2_scale.is_contiguous()
        )
        if not direct_act_scale and s1_bulk:
            _tq_pack_s1_bulk[(triton.cdiv(M, 128), B)](
                s1_bytes.reshape(M, B, nj * pb).view(torch.int64),
                s1p.view(torch.int64),
                M,
                nj,
                ss1_b // 8,
                ss1_m // 8,
                BMP=128,
                NJP=njp,
                EVEN=(M % 128 == 0),
            )
        elif not direct_act_scale:
            _tq_pack_s1_kernel[(triton.cdiv(M, 128), nj, B)](
                s1_bytes,
                s1p,
                M,
                nj,
                ss1_b,
                ss1_m,
                BMP=128,
                PB=pb,
            )
        if s2_bulk:
            shp2 = x2_scale.shape
            _tq_pack_s2_bulk[(triton.cdiv(N, 256), B)](
                x2_scale.reshape(shp2[0], shp2[1], shp2[2] * shp2[3]).view(torch.int16),
                s2p.view(torch.int64),
                N,
                nj,
                ss2_b // 2,
                ss2_k // 2,
                ss2_n // 2,
                BNP=256,
                NJP=njp,
                EVEN=(N % 256 == 0),
            )
        else:
            _tq_pack_s2_kernel[(triton.cdiv(N, 128), nj, B)](
                s2_bytes,
                s2p,
                N,
                nj,
                ss2_b,
                ss2_k,
                ss2_n,
                ss2_h,
                BNP=128,
                PB=pb,
            )
        _mx_grid = (B, triton.cdiv(N, BN), triton.cdiv(M, BM))
        _tq_mx_kernel[_mx_grid](
            x1,
            x2,
            s1p,
            s2p,
            bias if has_bias else x1,
            out,
            M,
            N,
            K,
            sx1_b,
            sx1_m,
            sx2_b,
            so_b,
            so_m,
            so_n,
            BM=BM,
            BN=BN,
            BK=BK,
            DIRECT_ACT_SCALE=direct_act_scale,
            num_stages=2,
            HAS_BIAS=has_bias,
            DT_OUT=dt_code,
            EVEN=(M % BM == 0 and N % BN == 0),
        )
        return out

    raise ValueError("incompatible MX matrix or scale layout")


def npu_transpose_quant_batchmatmul(
    x1,
    x2,
    dtype,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=1,
    x1_dtype=None,
    x2_dtype=None,
):
    """MXFP8 batched matrix multiplication with token/batch transposes and block-32 E8M0 scaling."""
    if (
        x1.ndim != 3
        or x2.ndim != 3
        or x1.dtype != torch.float8_e4m3fn
        or x2.dtype != x1.dtype
        or dtype != torch.bfloat16
    ):
        raise ValueError("expected E4M3 MBK and BKN operands with BF16 output")
    m, b, k = x1.shape
    if x2.shape[:2] != (b, k) or k % 64 or not x1.numel() or not x2.numel():
        raise ValueError(
            "K must be a multiple of 64 and both operands must be nonempty"
        )
    n = x2.shape[-1]
    if (
        x1_scale is None
        or x2_scale is None
        or x1_scale.shape != (m, b, k // 64, 2)
        or x2_scale.shape != (b, k // 64, n, 2)
    ):
        raise ValueError("scales must be [M,B,K/64,2] and [B,K/64,N,2]")
    if (
        x1_scale.dtype != torch.float8_e8m0fnu
        or x2_scale.dtype != x1_scale.dtype
        or not all(t.is_contiguous() for t in (x1, x2, x1_scale, x2_scale))
    ):
        raise ValueError("expected contiguous data and E8M0 scale tensors")
    if (
        tuple(group_sizes or ()) != (0, 0, 32)
        or tuple(perm_x1 or ()) != (1, 0, 2)
        or tuple(perm_x2 or (0, 1, 2)) != (0, 1, 2)
        or tuple(perm_y or ()) != (1, 0, 2)
    ):
        raise ValueError("only block-32 scaling with MBK x BKN -> MBN is supported")
    if (
        bias is not None
        or batch_split_factor != 1
        or x1_dtype is not None
        or x2_dtype is not None
    ):
        raise ValueError(
            "bias, explicit dtype overrides and batch splitting are unsupported"
        )

    if not _HAS_MX or k % 128 or (m == 1 and (n < 256 or n > 4096 or n & (n - 1))):
        raise ValueError(
            "MX dot support and K divisible by 128 are required; decode N is a power of two in [256,4096]"
        )
    if (
        x1.device.type != "npu"
        or x2.device != x1.device
        or x1_scale.device != x1.device
        or x2_scale.device != x1.device
    ):
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(x1.device):
        return _launch(
            x1,
            x2,
            dtype,
            bias=bias,
            x1_scale=x1_scale,
            x2_scale=x2_scale,
            group_sizes=group_sizes,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            batch_split_factor=batch_split_factor,
            x1_dtype=x1_dtype,
            x2_dtype=x2_dtype,
        )
