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

# ---------------------------------------------------------------------------
# Bitwise-exact reproduction of the Ascend reference chain (see
# tmp/NOTES_exactness.md for the measured evidence behind every choice):
#   * division   -> one Newton-corrected reciprocal, never the plain quotient
#   * 4-wide sums -> ((a0 + a1) + (a2 + a3))
#   * scale/base  -> one rounded multiply, then one rounded add (no fusion)
#   * apply       -> one rounded multiply, then three explicit fmas
#   * bf16 output -> a single round-to-nearest-even narrowing
#
# Dispatched as two kernels: A does the per-token scalar chain (rms, mixes,
# sigmoids, 4x4 sinkhorn) over a wide token block so the vector lanes are full;
# B is a pure streaming apply whose layer_input stores are long contiguous runs.
# ---------------------------------------------------------------------------


@triton.jit
def _recip(x):
    q0 = 1.0 / x
    r = tl.math.fma(-x, q0, 1.0)
    return tl.math.fma(r, q0, q0)


@triton.jit
def _div(a, b):
    q0 = a / b
    r = tl.math.fma(-b, q0, a)
    return tl.math.fma(r, _recip(b), q0)


@triton.jit
def _s4(a0, a1, a2, a3):
    return (a0 + a1) + (a2 + a3)


@triton.jit
def _sigmoid(u):
    return _recip(tl.exp(-u) + 1.0)


@triton.jit
def _unpack8(x, TB: tl.constexpr):
    # (TB, 8) tile -> its 8 column vectors, by tl.split/tl.reshape only (pure
    # register-layout moves, no arithmetic).  Column i of x becomes argument i,
    # because reshape is row-major: evens are x[:, 0,2,4,6] and odds x[:, 1,3,5,7].
    ev, od = tl.split(tl.reshape(x, (TB, 4, 2)))
    eva, evb = tl.split(tl.reshape(ev, (TB, 2, 2)))
    oda, odb = tl.split(tl.reshape(od, (TB, 2, 2)))
    v0, v4 = tl.split(eva)
    v2, v6 = tl.split(evb)
    v1, v5 = tl.split(oda)
    v3, v7 = tl.split(odb)
    return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _mix_kernel(
    mul_ptr,  # (T, 24) fp32  (gemm_out_mul, n_splits == 1)
    sq_ptr,  # (T,)    fp32  (gemm_out_sqrsum)
    scale_ptr,  # (3,)    fp32
    base_ptr,  # (24,)   fp32
    pre_ptr,  # (T, 4)  fp32  scratch: pre_mix
    post_ptr,  # (T, 4)  fp32
    comb_ptr,  # (T, 16) fp32
    T,
    rms_eps,
    pre_eps,
    sink_eps,
    post_mult,
    sink_repeat,
    inv_group,
    res_ptr,  # (T, 4, H) bf16  (only read when DO_APPLY)
    out_ptr,  # (T, H)    bf16  (only written when DO_APPLY)
    H,
    TB: tl.constexpr,
    HC: tl.constexpr,
    DO_APPLY: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * TB + tl.arange(0, TB)
    tm = t < T

    sq = tl.load(sq_ptr + t, mask=tm, other=1.0)
    s = sq * inv_group
    s = s + rms_eps
    rms = _recip(tl.sqrt(s))

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    mrow = t * 24

    mo8 = tl.arange(0, 8)[None, :]
    mrow2 = mrow[:, None]
    ga = tl.load(mul_ptr + mrow2 + mo8, mask=tm[:, None], other=0.0)
    gb = tl.load(mul_ptr + mrow2 + 8 + mo8, mask=tm[:, None], other=0.0)
    gc = tl.load(mul_ptr + mrow2 + 16 + mo8, mask=tm[:, None], other=0.0)
    a0, a1, a2, a3, a4, a5, a6, a7 = _unpack8(ga, TB)
    b0, b1, b2, b3, b4, b5, b6, b7 = _unpack8(gb, TB)
    d0, d1, d2, d3, d4, d5, d6, d7 = _unpack8(gc, TB)

    # ---------------- mixes 0..3 -> pre_mix ----------------
    u0 = a0 * rms
    u1 = a1 * rms
    u2 = a2 * rms
    u3 = a3 * rms
    u0 = u0 * s0
    u0 = u0 + tl.load(base_ptr + 0)
    u1 = u1 * s0
    u1 = u1 + tl.load(base_ptr + 1)
    u2 = u2 * s0
    u2 = u2 + tl.load(base_ptr + 2)
    u3 = u3 * s0
    u3 = u3 + tl.load(base_ptr + 3)
    # One contiguous (TB,4) store instead of 4 single-lane strided stores.  The
    # token-major layout of pre_mix means a whole token block is one contiguous
    # run, so the transaction count per block drops 4x with no change to any
    # value: the elements are only permuted into tile positions with tl.join,
    # which is a pure register-layout operation.
    p0 = _sigmoid(u0) + pre_eps
    p1 = _sigmoid(u1) + pre_eps
    p2 = _sigmoid(u2) + pre_eps
    p3 = _sigmoid(u3) + pre_eps
    if not DO_APPLY:
        ptile = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (TB, 4))
        tl.store(
            pre_ptr + t[:, None] * 4 + tl.arange(0, 4)[None, :],
            ptile,
            mask=tm[:, None],
        )

    # ---------------- mixes 4..7 -> post_mix ----------------
    v0 = a4 * rms
    v1 = a5 * rms
    v2 = a6 * rms
    v3 = a7 * rms
    v0 = v0 * s1
    v0 = v0 + tl.load(base_ptr + 4)
    v1 = v1 * s1
    v1 = v1 + tl.load(base_ptr + 5)
    v2 = v2 * s1
    v2 = v2 + tl.load(base_ptr + 6)
    v3 = v3 * s1
    v3 = v3 + tl.load(base_ptr + 7)
    q0 = _sigmoid(v0) * post_mult
    q1 = _sigmoid(v1) * post_mult
    q2 = _sigmoid(v2) * post_mult
    q3 = _sigmoid(v3) * post_mult
    qtile = tl.reshape(tl.join(tl.join(q0, q2), tl.join(q1, q3)), (TB, 4))
    tl.store(
        post_ptr + t[:, None] * 4 + tl.arange(0, 4)[None, :],
        qtile,
        mask=tm[:, None],
    )

    # ---------------- mixes 8..23 -> 4x4 comb ----------------
    c00 = b0 * rms
    c01 = b1 * rms
    c02 = b2 * rms
    c03 = b3 * rms
    c10 = b4 * rms
    c11 = b5 * rms
    c12 = b6 * rms
    c13 = b7 * rms
    c20 = d0 * rms
    c21 = d1 * rms
    c22 = d2 * rms
    c23 = d3 * rms
    c30 = d4 * rms
    c31 = d5 * rms
    c32 = d6 * rms
    c33 = d7 * rms

    c00 = c00 * s2
    c00 = c00 + tl.load(base_ptr + 8)
    c01 = c01 * s2
    c01 = c01 + tl.load(base_ptr + 9)
    c02 = c02 * s2
    c02 = c02 + tl.load(base_ptr + 10)
    c03 = c03 * s2
    c03 = c03 + tl.load(base_ptr + 11)
    c10 = c10 * s2
    c10 = c10 + tl.load(base_ptr + 12)
    c11 = c11 * s2
    c11 = c11 + tl.load(base_ptr + 13)
    c12 = c12 * s2
    c12 = c12 + tl.load(base_ptr + 14)
    c13 = c13 * s2
    c13 = c13 + tl.load(base_ptr + 15)
    c20 = c20 * s2
    c20 = c20 + tl.load(base_ptr + 16)
    c21 = c21 * s2
    c21 = c21 + tl.load(base_ptr + 17)
    c22 = c22 * s2
    c22 = c22 + tl.load(base_ptr + 18)
    c23 = c23 * s2
    c23 = c23 + tl.load(base_ptr + 19)
    c30 = c30 * s2
    c30 = c30 + tl.load(base_ptr + 20)
    c31 = c31 * s2
    c31 = c31 + tl.load(base_ptr + 21)
    c32 = c32 * s2
    c32 = c32 + tl.load(base_ptr + 22)
    c33 = c33 * s2
    c33 = c33 + tl.load(base_ptr + 23)

    # ---------------- sinkhorn ----------------
    m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))

    c00 = tl.exp(c00 - m0)
    c01 = tl.exp(c01 - m0)
    c02 = tl.exp(c02 - m0)
    c03 = tl.exp(c03 - m0)
    c10 = tl.exp(c10 - m1)
    c11 = tl.exp(c11 - m1)
    c12 = tl.exp(c12 - m1)
    c13 = tl.exp(c13 - m1)
    c20 = tl.exp(c20 - m2)
    c21 = tl.exp(c21 - m2)
    c22 = tl.exp(c22 - m2)
    c23 = tl.exp(c23 - m2)
    c30 = tl.exp(c30 - m3)
    c31 = tl.exp(c31 - m3)
    c32 = tl.exp(c32 - m3)
    c33 = tl.exp(c33 - m3)

    rs = _s4(c00, c01, c02, c03)
    c00 = _div(c00, rs)
    c01 = _div(c01, rs)
    c02 = _div(c02, rs)
    c03 = _div(c03, rs)
    rs = _s4(c10, c11, c12, c13)
    c10 = _div(c10, rs)
    c11 = _div(c11, rs)
    c12 = _div(c12, rs)
    c13 = _div(c13, rs)
    rs = _s4(c20, c21, c22, c23)
    c20 = _div(c20, rs)
    c21 = _div(c21, rs)
    c22 = _div(c22, rs)
    c23 = _div(c23, rs)
    rs = _s4(c30, c31, c32, c33)
    c30 = _div(c30, rs)
    c31 = _div(c31, rs)
    c32 = _div(c32, rs)
    c33 = _div(c33, rs)

    c00 = c00 + sink_eps
    c01 = c01 + sink_eps
    c02 = c02 + sink_eps
    c03 = c03 + sink_eps
    c10 = c10 + sink_eps
    c11 = c11 + sink_eps
    c12 = c12 + sink_eps
    c13 = c13 + sink_eps
    c20 = c20 + sink_eps
    c21 = c21 + sink_eps
    c22 = c22 + sink_eps
    c23 = c23 + sink_eps
    c30 = c30 + sink_eps
    c31 = c31 + sink_eps
    c32 = c32 + sink_eps
    c33 = c33 + sink_eps

    cs = _s4(c00, c10, c20, c30) + sink_eps
    c00 = _div(c00, cs)
    c10 = _div(c10, cs)
    c20 = _div(c20, cs)
    c30 = _div(c30, cs)
    cs = _s4(c01, c11, c21, c31) + sink_eps
    c01 = _div(c01, cs)
    c11 = _div(c11, cs)
    c21 = _div(c21, cs)
    c31 = _div(c31, cs)
    cs = _s4(c02, c12, c22, c32) + sink_eps
    c02 = _div(c02, cs)
    c12 = _div(c12, cs)
    c22 = _div(c22, cs)
    c32 = _div(c32, cs)
    cs = _s4(c03, c13, c23, c33) + sink_eps
    c03 = _div(c03, cs)
    c13 = _div(c13, cs)
    c23 = _div(c23, cs)
    c33 = _div(c33, cs)

    nstep = sink_repeat - 1
    for _ in range(nstep):
        rs = _s4(c00, c01, c02, c03) + sink_eps
        c00 = _div(c00, rs)
        c01 = _div(c01, rs)
        c02 = _div(c02, rs)
        c03 = _div(c03, rs)
        rs = _s4(c10, c11, c12, c13) + sink_eps
        c10 = _div(c10, rs)
        c11 = _div(c11, rs)
        c12 = _div(c12, rs)
        c13 = _div(c13, rs)
        rs = _s4(c20, c21, c22, c23) + sink_eps
        c20 = _div(c20, rs)
        c21 = _div(c21, rs)
        c22 = _div(c22, rs)
        c23 = _div(c23, rs)
        rs = _s4(c30, c31, c32, c33) + sink_eps
        c30 = _div(c30, rs)
        c31 = _div(c31, rs)
        c32 = _div(c32, rs)
        c33 = _div(c33, rs)

        cs = _s4(c00, c10, c20, c30) + sink_eps
        c00 = _div(c00, cs)
        c10 = _div(c10, cs)
        c20 = _div(c20, cs)
        c30 = _div(c30, cs)
        cs = _s4(c01, c11, c21, c31) + sink_eps
        c01 = _div(c01, cs)
        c11 = _div(c11, cs)
        c21 = _div(c21, cs)
        c31 = _div(c31, cs)
        cs = _s4(c02, c12, c22, c32) + sink_eps
        c02 = _div(c02, cs)
        c12 = _div(c12, cs)
        c22 = _div(c22, cs)
        c32 = _div(c32, cs)
        cs = _s4(c03, c13, c23, c33) + sink_eps
        c03 = _div(c03, cs)
        c13 = _div(c13, cs)
        c23 = _div(c23, cs)
        c33 = _div(c33, cs)

    # comb_mix is (T,16) so a token block is one contiguous run too.  Replace the
    # 16 single-lane strided stores with two contiguous (TB,8) stores, each row
    # pair assembled by tl.join only (no arithmetic, no selects):
    #   flat positions 0..7  = c00 c01 c02 c03 c10 c11 c12 c13
    #   flat positions 8..15 = c20 c21 c22 c23 c30 c31 c32 c33
    h0 = tl.reshape(
        tl.join(
            tl.join(tl.join(c00, c10), tl.join(c02, c12)),
            tl.join(tl.join(c01, c11), tl.join(c03, c13)),
        ),
        (TB, 8),
    )
    h1 = tl.reshape(
        tl.join(
            tl.join(tl.join(c20, c30), tl.join(c22, c32)),
            tl.join(tl.join(c21, c31), tl.join(c23, c33)),
        ),
        (TB, 8),
    )
    coff = comb_ptr + t[:, None] * 16 + tl.arange(0, 8)[None, :]
    tl.store(coff, h0, mask=tm[:, None])
    tl.store(coff + 8, h1, mask=tm[:, None])

    if DO_APPLY:
        # Latency-regime path (small T only, num_tokens % TB == 0 so no token
        # mask is needed): stream the contraction here instead of in a second
        # launch, with pre kept in registers so pre_mix never round-trips through
        # GM.  Same one rounded multiply + three explicit fmas and a single bf16
        # narrowing per element, so outputs stay bitwise identical.  HC is the
        # largest power of two dividing H, capped so TB*HC stays at the 4096-
        # element body size already proven to fit UB.
        bres = res_ptr + t[:, None] * (4 * H)
        bdst = out_ptr + t[:, None] * H
        hoff = tl.arange(0, HC)[None, :]
        p0c = p0[:, None]
        p1c = p1[:, None]
        p2c = p2[:, None]
        p3c = p3[:, None]
        for i in tl.range(0, H, HC, num_stages=2):
            ho = i + hoff
            src = bres + ho
            x0 = tl.load(src + 0 * H).to(tl.float32)
            x1 = tl.load(src + 1 * H).to(tl.float32)
            x2 = tl.load(src + 2 * H).to(tl.float32)
            x3 = tl.load(src + 3 * H).to(tl.float32)
            acc = x0 * p0c
            acc = tl.math.fma(x1, p1c, acc)
            acc = tl.math.fma(x2, p2c, acc)
            acc = tl.math.fma(x3, p3c, acc)
            tl.store(bdst + ho, acc.to(tl.bfloat16))


@triton.jit
def _apply_tile(res_ptr, pre_ptr, out_ptr, t, H, HOFF: tl.constexpr, HB: tl.constexpr):
    hoff = HOFF + tl.arange(0, HB)
    base = res_ptr + t * (4 * H) + hoff
    p0 = tl.load(pre_ptr + t * 4 + 0)
    p1 = tl.load(pre_ptr + t * 4 + 1)
    p2 = tl.load(pre_ptr + t * 4 + 2)
    p3 = tl.load(pre_ptr + t * 4 + 3)
    x0 = tl.load(base + 0 * H).to(tl.float32)
    x1 = tl.load(base + 1 * H).to(tl.float32)
    x2 = tl.load(base + 2 * H).to(tl.float32)
    x3 = tl.load(base + 3 * H).to(tl.float32)
    acc = x0 * p0
    acc = tl.math.fma(x1, p1, acc)
    acc = tl.math.fma(x2, p2, acc)
    acc = tl.math.fma(x3, p3, acc)
    tl.store(out_ptr + t * H + hoff, acc.to(tl.bfloat16))


@triton.jit
def _apply_body(
    res_ptr, out_ptr, t, H, p0, p1, p2, p3, OFF: tl.constexpr, W: tl.constexpr
):
    # One unmasked 1-D column body of the contraction for token t, with the four
    # pre_mix coefficients already in registers.  Identical arithmetic to
    # _apply_tile: one rounded multiply then three explicit fmas, single bf16
    # narrowing, one contiguous burst per residual plane.
    hoff = OFF + tl.arange(0, W)
    base = res_ptr + t * (4 * H) + hoff
    x0 = tl.load(base + 0 * H).to(tl.float32)
    x1 = tl.load(base + 1 * H).to(tl.float32)
    x2 = tl.load(base + 2 * H).to(tl.float32)
    x3 = tl.load(base + 3 * H).to(tl.float32)
    acc = x0 * p0
    acc = tl.math.fma(x1, p1, acc)
    acc = tl.math.fma(x2, p2, acc)
    acc = tl.math.fma(x3, p3, acc)
    tl.store(out_ptr + t * H + hoff, acc.to(tl.bfloat16))


@triton.jit
def _apply_merged(
    res_ptr,  # (T, 4, H) bf16
    pre_ptr,  # (T, 4) fp32
    out_ptr,  # (T, H) bf16
    T,
    H,
    O0: tl.constexpr,
    W0: tl.constexpr,
    O1: tl.constexpr,
    W1: tl.constexpr,
    O2: tl.constexpr,
    W2: tl.constexpr,
):

    t = tl.program_id(0)
    p0 = tl.load(pre_ptr + t * 4 + 0)
    p1 = tl.load(pre_ptr + t * 4 + 1)
    p2 = tl.load(pre_ptr + t * 4 + 2)
    p3 = tl.load(pre_ptr + t * 4 + 3)
    _apply_body(res_ptr, out_ptr, t, H, p0, p1, p2, p3, O0, W0)
    if W1 > 0:
        _apply_body(res_ptr, out_ptr, t, H, p0, p1, p2, p3, O1, W1)
    if W2 > 0:
        _apply_body(res_ptr, out_ptr, t, H, p0, p1, p2, p3, O2, W2)


@triton.jit
def _apply_tail(
    res_ptr,  # (T, 4, H) bf16
    pre_ptr,  # (T, 4) fp32
    out_ptr,  # (T, H) bf16
    T,
    H,
    HOFF: tl.constexpr,
    HB: tl.constexpr,
    BT: tl.constexpr,
    TMASK: tl.constexpr,
):

    t = tl.program_id(1) * BT + tl.arange(0, BT)
    hoff = HOFF + tl.arange(0, HB)
    base = res_ptr + t[:, None] * (4 * H) + hoff[None, :]
    dst = out_ptr + t[:, None] * H + hoff[None, :]
    if TMASK:
        tm = t < T
        msk = tm[:, None]
        p0 = tl.load(pre_ptr + t * 4 + 0, mask=tm, other=0.0)[:, None]
        p1 = tl.load(pre_ptr + t * 4 + 1, mask=tm, other=0.0)[:, None]
        p2 = tl.load(pre_ptr + t * 4 + 2, mask=tm, other=0.0)[:, None]
        p3 = tl.load(pre_ptr + t * 4 + 3, mask=tm, other=0.0)[:, None]
        x0 = tl.load(base + 0 * H, mask=msk, other=0.0).to(tl.float32)
        x1 = tl.load(base + 1 * H, mask=msk, other=0.0).to(tl.float32)
        x2 = tl.load(base + 2 * H, mask=msk, other=0.0).to(tl.float32)
        x3 = tl.load(base + 3 * H, mask=msk, other=0.0).to(tl.float32)
        acc = x0 * p0
        acc = tl.math.fma(x1, p1, acc)
        acc = tl.math.fma(x2, p2, acc)
        acc = tl.math.fma(x3, p3, acc)
        tl.store(dst, acc.to(tl.bfloat16), mask=msk)
    else:
        p0 = tl.load(pre_ptr + t * 4 + 0)[:, None]
        p1 = tl.load(pre_ptr + t * 4 + 1)[:, None]
        p2 = tl.load(pre_ptr + t * 4 + 2)[:, None]
        p3 = tl.load(pre_ptr + t * 4 + 3)[:, None]
        x0 = tl.load(base + 0 * H).to(tl.float32)
        x1 = tl.load(base + 1 * H).to(tl.float32)
        x2 = tl.load(base + 2 * H).to(tl.float32)
        x3 = tl.load(base + 3 * H).to(tl.float32)
        acc = x0 * p0
        acc = tl.math.fma(x1, p1, acc)
        acc = tl.math.fma(x2, p2, acc)
        acc = tl.math.fma(x3, p3, acc)
        tl.store(dst, acc.to(tl.bfloat16))


@triton.jit
def _apply_tail_blocked(
    res_ptr,  # (T, 4, H) bf16
    pre_ptr,  # (T, 4) fp32
    out_ptr,  # (T, H) bf16
    T,
    H,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    BT: tl.constexpr,
    TMASK: tl.constexpr,
):
    # Uniform (BT tokens x 1024 columns) tiles for the region past the single
    # 4096-wide segment.  Only reached when H % 1024 == 0, so every tile is
    # unmasked in H; the tile's column run stays 2 KB per row exactly as in the
    # one-token-per-program form, while the descriptor count drops by BT.
    pid_h = tl.program_id(0)
    if pid_h == 0:
        _apply_tail(res_ptr, pre_ptr, out_ptr, T, H, O0, 1024, BT, TMASK)
    elif pid_h == 1:
        _apply_tail(res_ptr, pre_ptr, out_ptr, T, H, O1, 1024, BT, TMASK)
    else:
        _apply_tail(res_ptr, pre_ptr, out_ptr, T, H, O2, 1024, BT, TMASK)


@triton.jit
def _apply_kernel(
    res_ptr,  # (T, 4, H) bf16
    pre_ptr,  # (T, 4) fp32
    out_ptr,  # (T, H) bf16
    T,
    H,
    O0: tl.constexpr,
    S0: tl.constexpr,
    O1: tl.constexpr,
    S1: tl.constexpr,
    O2: tl.constexpr,
    S2: tl.constexpr,
):
    # H is covered exactly by a short list of UNMASKED power-of-two segments, so
    # every residual/layer_input access is one maximal contiguous burst with no
    # predicates: (4096, 2048, 1024) for H=7168, (4096, 1024) for H=5120, one
    # 4096 for H=4096.  pid_h selects the segment; the bodies differ only in
    # their constexpr arange width.
    pid_h = tl.program_id(0)
    t = tl.program_id(1)
    if pid_h == 0:
        _apply_tile(res_ptr, pre_ptr, out_ptr, t, H, O0, S0)
    elif pid_h == 1:
        _apply_tile(res_ptr, pre_ptr, out_ptr, t, H, O1, S1)
    else:
        _apply_tile(res_ptr, pre_ptr, out_ptr, t, H, O2, S2)


@triton.jit
def _apply_kernel_masked(
    res_ptr,  # (T, 4, H) bf16
    pre_ptr,  # (T, 4) fp32
    out_ptr,  # (T, H) bf16
    T,
    H,
    HBLK: tl.constexpr,
):
    # General fallback for an H that is not an exact sum of <=3 power-of-two
    # segments of at most 4096; only correctness shapes can reach it.
    pid_h = tl.program_id(0)
    t = tl.program_id(1)
    hoff = pid_h * HBLK + tl.arange(0, HBLK)
    msk = hoff < H
    base = res_ptr + t * (4 * H) + hoff
    p0 = tl.load(pre_ptr + t * 4 + 0)
    p1 = tl.load(pre_ptr + t * 4 + 1)
    p2 = tl.load(pre_ptr + t * 4 + 2)
    p3 = tl.load(pre_ptr + t * 4 + 3)
    x0 = tl.load(base + 0 * H, mask=msk, other=0.0).to(tl.float32)
    x1 = tl.load(base + 1 * H, mask=msk, other=0.0).to(tl.float32)
    x2 = tl.load(base + 2 * H, mask=msk, other=0.0).to(tl.float32)
    x3 = tl.load(base + 3 * H, mask=msk, other=0.0).to(tl.float32)
    acc = x0 * p0
    acc = tl.math.fma(x1, p1, acc)
    acc = tl.math.fma(x2, p2, acc)
    acc = tl.math.fma(x3, p3, acc)
    tl.store(out_ptr + t * H + hoff, acc.to(tl.bfloat16), mask=msk)


_SEG_CAP = 4096
_SEG_MAX = 3
# Shortest-segment handling: only segments of at most this many elements take the
# token-blocked form, so the 4096-wide bodies keep their current burst structure
# and UB footprint (a 4096-wide body already uses ~120 KB of the ~192 KB UB,
# whereas a 1024-wide one uses ~30 KB per token, so BT=4 there is ~120 KB).
_TAIL_MAX = 1024
_TAIL_BT = 4
# Token block for the uniform (BT x 1024) remainder tiles.  (8, 1024) = 8192
# elements is the largest token-blocked apply tile measured to compile on this
# stack; 16384 elements fails with MLIRCompilationError.
_TAIL_BT2 = 8
_WIDE_SEG = 4096


def _h_segments(H):
    segs = []
    off = 0
    rem = H
    while rem > 0 and len(segs) < _SEG_MAX:
        top = rem if rem < _SEG_CAP else _SEG_CAP
        s = 1 << (top.bit_length() - 1)
        segs.append((off, s))
        off += s
        rem -= s
    if rem > 0:
        return None
    return segs


def mhc_pre_big_fuse_fwd(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    rms_eps,
    mhc_pre_eps,
    mhc_sinkhorn_eps,
    mhc_post_mult_value,
    sinkhorn_repeat,
    n_splits=16,
):
    """Fuse four-stream mHC pre-mixing from reduced prenorm GEMM partials."""
    if (
        residual.ndim != 3
        or residual.shape[1] != 4
        or gemm_out_mul.shape != (1, residual.shape[0], 24)
    ):
        raise ValueError("expected one reduced GEMM partial and four mHC streams")

    mhc = residual.shape[-2]
    hidden = residual.shape[-1]
    num_tokens = residual.numel() // (mhc * hidden)

    pre_mix = torch.empty(
        (num_tokens, mhc), dtype=torch.float32, device=residual.device
    )
    segs = _h_segments(hidden)

    mtb = 16 if num_tokens <= 1024 else 64

    use_fused = (num_tokens <= 1024) and (num_tokens % mtb == 0) and (hidden <= 4096)
    if use_fused:
        hc = 256
        while hc > 1 and hidden % hc != 0:
            hc //= 2
    else:
        hc = 256

    _mix_kernel[(triton.cdiv(num_tokens, mtb),)](
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        pre_mix,
        post_mix,
        comb_mix,
        num_tokens,
        float(rms_eps),
        float(mhc_pre_eps),
        float(mhc_sinkhorn_eps),
        float(mhc_post_mult_value),
        int(sinkhorn_repeat),
        1.0 / float(mhc * hidden),
        residual,
        layer_input,
        hidden,
        TB=mtb,
        HC=hc,
        DO_APPLY=use_fused,
    )
    if use_fused:
        return None
    if segs is not None:

        wide = segs
        tail = None

        use_merged = (
            num_tokens > 1024 and 2 <= len(segs) <= 3 and segs[-1][1] >= _TAIL_MAX
        )
        use_blocked = (
            not use_merged
            and hidden % 1024 == 0
            and hidden > _WIDE_SEG
            and hidden <= _WIDE_SEG + 1024
            and num_tokens > 1024
        )
        if (
            not use_blocked
            and not use_merged
            and (len(segs) == 2 and segs[-1][1] <= _TAIL_MAX and num_tokens > 1024)
        ):
            wide = segs[:-1]
            tail = segs[-1]

        if use_merged:
            o = [0, 0, 0]
            w = [0, 0, 0]
            for i, (seg_off, seg_size) in enumerate(segs):
                o[i] = seg_off
                w[i] = seg_size
            _apply_merged[(num_tokens,)](
                residual,
                pre_mix,
                layer_input,
                num_tokens,
                hidden,
                O0=o[0],
                W0=w[0],
                O1=o[1],
                W1=w[1],
                O2=o[2],
                W2=w[2],
            )
        elif use_blocked:
            ntiles = hidden // 1024 - _WIDE_SEG // 1024
            offs = [(_WIDE_SEG + 1024 * i, 0, 0) for i in range(3)]
            tbt = _TAIL_BT2
            _apply_kernel[(1, num_tokens)](
                residual,
                pre_mix,
                layer_input,
                num_tokens,
                hidden,
                O0=0,
                S0=_WIDE_SEG,
                O1=0,
                S1=64,
                O2=0,
                S2=64,
            )
            _apply_tail_blocked[(ntiles, triton.cdiv(num_tokens, tbt))](
                residual,
                pre_mix,
                layer_input,
                num_tokens,
                hidden,
                O0=offs[0][0],
                O1=offs[1][0],
                O2=offs[2][0],
                BT=tbt,
                TMASK=(num_tokens % tbt != 0),
            )
        elif wide:
            n = len(wide)
            o = [0, 0, 0]
            s = [64, 64, 64]
            for i, (seg_off, seg_size) in enumerate(wide):
                o[i] = seg_off
                s[i] = seg_size
            _apply_kernel[(n, num_tokens)](
                residual,
                pre_mix,
                layer_input,
                num_tokens,
                hidden,
                O0=o[0],
                S0=s[0],
                O1=o[1],
                S1=s[1],
                O2=o[2],
                S2=s[2],
            )
        if tail is not None:
            tbt = _TAIL_BT
            _apply_tail[(1, triton.cdiv(num_tokens, tbt))](
                residual,
                pre_mix,
                layer_input,
                num_tokens,
                hidden,
                HOFF=tail[0],
                HB=tail[1],
                BT=tbt,
                TMASK=(num_tokens % tbt != 0),
            )
    else:
        fb = 1024
        _apply_kernel_masked[(triton.cdiv(hidden, fb), num_tokens)](
            residual,
            pre_mix,
            layer_input,
            num_tokens,
            hidden,
            HBLK=fb,
        )
    return None
