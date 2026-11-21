# Copyright 2026, The FlagOS Contributors.
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


"""Device softmax specialization for STATE interleaved BF16 QKV.

Cauchy bounds select a fixed-shift softmax only when every score is
within +/-55 in log2 units. The resulting exponent range stays normal
in BF16. Large-norm tiles use stable online softmax with a high/low
BF16 probability decomposition. A value-magnitude guard also selects stable
recomputation when BF16 probability rounding would be amplified by large V.
"""

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._hygon.attention_state_hip import (
    _load_state_kernel,
    launch_short_state_kernel,
    launch_state_kernel,
)


@triton.jit
def _state_key_norms(
    K,
    V,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = tl.program_id(1)
    b = tl.program_id(2)
    d = tl.arange(0, D)
    k = tl.load(
        K + b * KB + h * KH + r[:, None] * KS + d[None, :], r[:, None] < N, 0
    ).to(tl.float32)
    n = tl.max(tl.sum(k * k, 1), 0)
    v = tl.load(
        V + b * KB + h * KH + r[:, None] * KS + d[None, :], r[:, None] < N, 0
    ).to(tl.float32)
    # Large V amplifies rounding of the fast path's single BF16 probability.
    # Mark this head unsafe; the stable path uses high/low probabilities.
    vmax = tl.max(tl.max(tl.abs(v), 1), 0)
    n = tl.where(vmax <= 8.0, n, float("inf"))
    tl.store(T + (b * H + h) * triton.cdiv(N, BLOCK) + tl.program_id(0), n)


@triton.jit
def _state_fast_inner(
    q,
    kp,
    vp,
    acc,
    denom,
    shift,
    offs,
    SCALE: tl.constexpr,
    N: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BN: tl.constexpr,
    FULL: tl.constexpr,
    PRE: tl.constexpr,
    STAGES: tl.constexpr,
    RED: tl.constexpr,
):
    if FULL:
        end: tl.constexpr = N // BN * BN
        k = tl.load(kp)
        for s in range(0, end, BN):
            score = tl.dot(q, k) * SCALE - shift[:, None]
            # Fetch the next K tile while softmax/PV consume the current one.
            k = tl.load(kp + tl.minimum(s + BN, end - BN) * KS)
            p = tl.exp2(score)
            denom += tl.sum(p.reshape(p.shape[0], BN // RED, RED), 2)
            v = tl.load(vp + s * VS)
            acc = tl.dot(tl.trans(v), tl.trans(p.to(q.dtype)), acc)
    else:
        for s in range(N // BN * BN, N, BN):
            k = tl.load(kp + s * KS, s + offs[None, :] < N, 0.0)
            p = tl.exp2(tl.dot(q, k) * SCALE - shift[:, None])
            p = tl.where(s + offs[None, :] < N, p, 0.0)
            denom += tl.sum(p.reshape(p.shape[0], BN // RED, RED), 2)
            v = tl.load(vp + s * VS, s + offs[:, None] < N, 0.0)
            acc = tl.dot(tl.trans(v), tl.trans(p.to(q.dtype)), acc)
    return acc, denom


@triton.jit
def _state_stable_inner(
    q,
    kp,
    vp,
    acc,
    m,
    denom,
    offs,
    SCALE: tl.constexpr,
    N: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BN: tl.constexpr,
    FULL: tl.constexpr,
    PRE: tl.constexpr,
    STAGES: tl.constexpr,
):
    end: tl.constexpr = N // BN * BN if FULL else N
    start: tl.constexpr = 0 if FULL else N // BN * BN
    for s in tl.range(start, end, BN, num_stages=STAGES):
        if FULL:
            k = tl.load(kp + s * KS)
        else:
            k = tl.load(kp + s * KS, mask=(s + offs[None, :]) < N, other=0.0)
        if PRE:
            if FULL:
                v = tl.load(vp + s * VS)
            else:
                v = tl.load(vp + s * VS, mask=(s + offs[:, None]) < N, other=0.0)
        s2 = tl.dot(q, k) * SCALE
        if not FULL:
            s2 = tl.where(s + offs[None, :] < N, s2, -float("inf"))
        m2 = tl.maximum(m, tl.max(s2, 1))
        p = tl.exp2(s2 - m2[:, None])
        alpha = tl.exp2(m - m2)
        denom = denom * alpha + tl.sum(p, 1)
        acc = acc * alpha[None, :]
        if not PRE:
            if FULL:
                v = tl.load(vp + s * VS)
            else:
                v = tl.load(vp + s * VS, mask=(s + offs[:, None]) < N, other=0.0)
        hi = p.to(q.dtype)
        lo = (p - hi.to(tl.float32)).to(q.dtype)
        acc = tl.dot(tl.trans(v), tl.trans(hi), acc)
        acc = tl.dot(tl.trans(v), tl.trans(lo), acc)
        m = m2
    return acc, m, denom


@triton.jit
def _state_fast(
    Q,
    K,
    V,
    T,
    F,
    O,
    L,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QS: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KS: tl.constexpr,
    VB: tl.constexpr,
    VH: tl.constexpr,
    VS: tl.constexpr,
    OB: tl.constexpr,
    OH: tl.constexpr,
    OS: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PRE: tl.constexpr,
    STAGES: tl.constexpr,
    RED: tl.constexpr,
):
    pm = tl.program_id(0)
    ph = tl.program_id(1)
    pb = tl.program_id(2)
    r = pm * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    n = tl.arange(0, BN)
    q = tl.load(
        Q + pb * QB + ph * QH + r[:, None] * QS + d[None, :], r[:, None] < M, 0.0
    )
    kn = tl.arange(0, triton.next_power_of_2(triton.cdiv(N, 128)))
    kmax = tl.max(
        tl.load(
            T + (pb * H + ph) * triton.cdiv(N, 128) + kn, kn < triton.cdiv(N, 128), 0.0
        ),
        0,
    )
    qf = q.to(tl.float32)
    shift = tl.sqrt(tl.sum(qf * qf, 1) * kmax) * tl.abs(SCALE) * 1.00001
    finite = (shift.to(tl.uint32, bitcast=True) & 0x7F800000) != 0x7F800000
    shift = tl.where(finite, shift, float("inf"))
    # Bound both signs: no online rescaling or per-step row reduction.
    good = tl.max(shift, 0) <= 55.0
    tl.store(F + (pb * H + ph) * triton.cdiv(M, BM) + pm, good)
    if good:
        kp = K + pb * KB + ph * KH + n[None, :] * KS + d[:, None]
        vp = V + pb * VB + ph * VH + n[:, None] * VS + d[None, :]
        acc = tl.zeros((D, BM), tl.float32)
        denom = tl.zeros((BM, BN // RED), tl.float32)
        acc, denom = _state_fast_inner(
            q,
            kp,
            vp,
            acc,
            denom,
            shift,
            n,
            SCALE,
            N,
            KS,
            VS,
            BN,
            True,
            PRE,
            STAGES,
            RED,
        )
        if N % BN:
            acc, denom = _state_fast_inner(
                q,
                kp,
                vp,
                acc,
                denom,
                shift,
                n,
                SCALE,
                N,
                KS,
                VS,
                BN,
                False,
                PRE,
                1,
                RED,
            )
        denom = tl.sum(denom, 1)
        tl.store(
            O + pb * OB + ph * OH + r[:, None] * OS + d[None, :],
            tl.trans(acc) / denom[:, None],
            r[:, None] < M,
        )
        tl.store(
            L + (pb * H + ph) * M + r,
            (shift + tl.log2(denom)) * 0.6931471805599453,
            r < M,
        )


@triton.jit
def _state_stable(
    Q,
    K,
    V,
    F,
    O,
    L,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QS: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KS: tl.constexpr,
    VB: tl.constexpr,
    VH: tl.constexpr,
    VS: tl.constexpr,
    OB: tl.constexpr,
    OH: tl.constexpr,
    OS: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    GUARD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CHECK_GUARD: tl.constexpr = True,
):
    pm = tl.program_id(0)
    ph = tl.program_id(1)
    pb = tl.program_id(2)
    if CHECK_GUARD:
        good = tl.load(F + (pb * H + ph) * triton.cdiv(M, GUARD) + pm * BM // GUARD)
    else:
        good = 0
    if good == 0:
        r = pm * BM + tl.arange(0, BM)
        d = tl.arange(0, D)
        n = tl.arange(0, BN)
        q = tl.load(
            Q + pb * QB + ph * QH + r[:, None] * QS + d[None, :], r[:, None] < M, 0.0
        )
        kp = K + pb * KB + ph * KH + n[None, :] * KS + d[:, None]
        vp = V + pb * VB + ph * VH + n[:, None] * VS + d[None, :]
        acc = tl.zeros((D, BM), tl.float32)
        denom = tl.zeros((BM,), tl.float32)
        m = tl.full((BM,), -float("inf"), tl.float32)
        acc, m, denom = _state_stable_inner(
            q, kp, vp, acc, m, denom, n, SCALE, N, KS, VS, BN, True, False, 1
        )
        if N % BN:
            acc, m, denom = _state_stable_inner(
                q, kp, vp, acc, m, denom, n, SCALE, N, KS, VS, BN, False, False, 1
            )
        tl.store(
            O + pb * OB + ph * OH + r[:, None] * OS + d[None, :],
            tl.trans(acc) / denom[:, None],
            r[:, None] < M,
        )
        tl.store(
            L + (pb * H + ph) * M + r, (m + tl.log2(denom)) * 0.6931471805599453, r < M
        )


def state_attention(q, k, v, scale):
    # Captured explicit scale uses 1 / sqrt(D), while the default uses D**-0.5.
    # Their Python doubles differ by one ULP but have identical FP32 values.
    use_hip = (
        scale in (128**-0.5, 1.0 / math.sqrt(128)) and _load_state_kernel() is not None
    )
    bm, bn, w, s, pre, red = (192 if use_hip else 128), 32, 4, 1, False, 1
    b, h, m, d = q.shape
    n = k.shape[2]
    out = torch.empty((b, h, m, d), device=q.device, dtype=q.dtype)
    lse = torch.empty((b, h, m), device=q.device, dtype=torch.float32)
    tmp = torch.empty((b, h, triton.cdiv(n, 128)), device=q.device, dtype=torch.float32)
    flag = torch.empty((b, h, triton.cdiv(m, bm)), device=q.device, dtype=torch.uint8)
    # DTK clang-18 does not implement Triton's optional -fp-fuse=fast switch.
    _state_key_norms[(triton.cdiv(n, 128), h, b)](
        k,
        v,
        tmp,
        b,
        h,
        n,
        d,
        *k.stride()[:3],
        128,
        enable_fp_fusion=False,
        num_warps=16,
    )
    arg = (
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:3],
        h,
        m,
        n,
        d,
        scale * 1.4426950408889634,
    )
    if use_hip:
        launch_state_kernel(q, k, v, tmp, flag, out, lse)
    else:
        _state_fast[(triton.cdiv(m, bm), h, b)](
            q,
            k,
            v,
            tmp,
            flag,
            out,
            lse,
            *arg,
            bm,
            bn,
            pre,
            s,
            red,
            enable_fp_fusion=False,
            num_warps=w,
            num_stages=s,
            waves_per_eu=2,
        )
    _state_stable[(triton.cdiv(m, 64), h, b)](
        q,
        k,
        v,
        flag,
        out,
        lse,
        *arg,
        bm,
        64,
        32,
        enable_fp_fusion=False,
        num_warps=4,
        num_stages=1,
    )
    return out, lse


def small_state_attention(q, k, v, scale):
    """Short worksheet cases; every path has a stable high/low fallback."""
    b, h, m, d = q.shape
    n = k.shape[2]
    out = torch.empty((b, h, m, d), device=q.device, dtype=q.dtype)
    lse = torch.empty((b, h, m), device=q.device, dtype=torch.float32)
    bm, bn = (16, 16) if n == 3 else (16, 32) if n == 23 else (32, 64)
    arg = (
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:3],
        h,
        m,
        n,
        d,
        scale * 1.4426950408889634,
    )
    if (
        n == 362
        and scale in (128**-0.5, 1.0 / math.sqrt(128))
        and _load_state_kernel() is not None
    ):
        flag = torch.empty(
            (b, h, triton.cdiv(m, 192)), device=q.device, dtype=torch.uint8
        )
        launch_short_state_kernel(q, k, v, flag, out, lse)
        return out, lse
    _state_stable[(triton.cdiv(m, bm), h, b)](
        q,
        k,
        v,
        q,
        out,
        lse,
        *arg,
        bm,
        bm,
        bn,
        CHECK_GUARD=False,
        enable_fp_fusion=False,
        num_warps=4,
        num_stages=1,
    )
    return out, lse
