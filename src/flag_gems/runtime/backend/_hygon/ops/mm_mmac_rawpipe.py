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

"""Prefetch global operands before MMAC; convert their layout after compute."""

import triton
import triton.language as tl

from .mm_mmac_peel import (
    consume_acc,
    reverse_bits,
    split_levels,
    split_parts,
    store_acc,
)


@triton.jit
def load_raw(
    A,
    B,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    PREFETCH: tl.constexpr,
    pm,
    pn,
    step,
    FULL_K: tl.constexpr,
):
    WN: tl.constexpr = WARPS // WM
    ax = tl.arange(0, WM * TM * 64 * PREFETCH)
    aw = ax // (TM * 64 * PREFETCH)
    ar = (ax // (4 * PREFETCH)) % (TM * 16)
    ak = ax % (4 * PREFETCH)
    rr = pm * (16 * TM * WM) + aw * 16 * TM + ar
    kk = step * 4 * PREFETCH + ak
    if M % (16 * TM * WM) == 0 and FULL_K:
        av = tl.load(A + rr * AM + kk * AK)
    elif FULL_K:
        av = tl.load(A + rr * AM + kk * AK, rr < M, 0)
    else:
        av = tl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0)
    bx = tl.arange(0, WN * TN * 64 * PREFETCH)
    bw = bx // (TN * 64 * PREFETCH)
    bj = (bx // (64 * PREFETCH)) % TN
    bk = (bx // 16) % (4 * PREFETCH)
    bc = bx % 16
    cc = pn * (16 * TN * WN) + bw * 16 * TN + bj * 16 + bc
    kk = step * 4 * PREFETCH + bk
    if N % (16 * TN * WN) == 0 and FULL_K:
        bv = tl.load(B + kk * BK + cc * BN)
    elif FULL_K:
        bv = tl.load(B + kk * BK + cc * BN, cc < N, 0)
    else:
        bv = tl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0)
    return av, bv


@triton.jit
def convert_raw(
    av,
    bv,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    WN: tl.constexpr = WARPS // WM
    av = tl.reshape(av, (1, WM, TM, 16, PREFETCH, 4))
    av = tl.broadcast_to(av, (WN, WM, TM, 16, PREFETCH, 4))
    av = tl.reshape(av, (WARPS, TM, 16, PREFETCH, 4))
    av = tl.trans(av, (0, 4, 2, 1, 3))
    ap = split_parts(av, WARPS, TM, PREFETCH)

    bv = tl.reshape(bv, (WN, 1, TN, PREFETCH, 4, 16))
    bv = tl.broadcast_to(bv, (WN, WM, TN, PREFETCH, 4, 16))
    bv = tl.reshape(bv, (WARPS, TN, PREFETCH, 64))
    bv = tl.trans(bv, (0, 3, 1, 2))
    bp = split_parts(bv, WARPS, TN, PREFETCH)
    aa = ()
    bb = ()
    for ki in tl.static_range(PREFETCH):
        for ti in tl.static_range(TM):
            aa += (
                tl.reshape(
                    ap[reverse_bits(ti * PREFETCH + ki, split_levels(TM * PREFETCH))],
                    (WARPS * 64,),
                ),
            )
        for tj in tl.static_range(TN):
            bb += (
                tl.reshape(
                    bp[reverse_bits(tj * PREFETCH + ki, split_levels(TN * PREFETCH))],
                    (WARPS * 64,),
                ),
            )
    return aa, bb


@triton.jit
def compute(aa, bb, acc, PREFETCH: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    for ki in tl.static_range(PREFETCH):
        for i in tl.static_range(TM):
            for j in tl.static_range(TN):
                acc = consume_acc(aa, bb, acc, ki, i, j, TM, TN)
    return acc


@triton.jit
def mmac_tiled(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    PREFETCH: tl.constexpr,
    GROUP: tl.constexpr,
    SPLIT: tl.constexpr,
):
    tl.static_assert(SPLIT == 1)
    nm = tl.cdiv(M, 16 * TM * WM)
    nn = tl.cdiv(N, 16 * TN * (WARPS // WM))
    p = tl.program_id(0) + tl.program_id(1) * nm
    g = p // (GROUP * nn)
    gs = tl.minimum(nm - g * GROUP, GROUP)
    pm = g * GROUP + (p % (GROUP * nn)) % gs
    pn = (p % (GROUP * nn)) // gs
    acc = ()
    for i in tl.static_range(TM * TN * 4):
        acc += (tl.full((WARPS * 64,), 0, tl.float64),)
    steps: tl.constexpr = K // (4 * PREFETCH)
    if steps > 0:
        ra, rb = load_raw(
            A, B, M, N, K, AM, AK, BK, BN, WARPS, WM, TM, TN, PREFETCH, pm, pn, 0, True
        )
        aa, bb = convert_raw(ra, rb, WARPS, WM, TM, TN, PREFETCH)
        for step in range(steps - 1):
            ra, rb = load_raw(
                A,
                B,
                M,
                N,
                K,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                pm,
                pn,
                step + 1,
                True,
            )
            acc = compute(aa, bb, acc, PREFETCH, TM, TN)
            aa, bb = convert_raw(ra, rb, WARPS, WM, TM, TN, PREFETCH)
        acc = compute(aa, bb, acc, PREFETCH, TM, TN)
    if K % (4 * PREFETCH) != 0:
        ra, rb = load_raw(
            A,
            B,
            M,
            N,
            K,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            pm,
            pn,
            steps,
            False,
        )
        aa, bb = convert_raw(ra, rb, WARPS, WM, TM, TN, PREFETCH)
        acc = compute(aa, bb, acc, PREFETCH, TM, TN)
    for i in tl.static_range(TM):
        for j in tl.static_range(TN):
            store_acc(acc, C, pm, pn, M, N, WARPS, WM, TM, TN, i, j)
