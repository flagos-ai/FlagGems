# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Selected FP32 cache policy with unchanged full-precision tile arithmetic."""

import triton
import triton.language as tl


@triton.jit
def tiled_cached(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK_: tl.constexpr,
    BN_: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    UNROLL: tl.constexpr,
    TAIL: tl.constexpr,
    TRANS: tl.constexpr = False,
    LOAD: tl.constexpr = "",
    STORE: tl.constexpr = "",
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    ra = rm if M % BM == 0 else rm % M
    rb = rn if N % BN == 0 else rn % N
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in tl.range(K // BK, loop_unroll_factor=UNROLL):
        kk = step * BK + rk
        av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK, cache_modifier=LOAD)
        bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_, cache_modifier=LOAD)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if TAIL:
        for st in tl.static_range(K % BK // 8):
            kk = K // BK * BK + st * 8 + tl.arange(0, 8)
            av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK, cache_modifier=LOAD)
            bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_, cache_modifier=LOAD)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        for st in tl.static_range(K % 8):
            kk = K // 8 * 8 + st
            av = tl.load(A + ra * AM + kk * AK, cache_modifier=LOAD)
            bv = tl.load(B + kk * BK_ + rb * BN_, cache_modifier=LOAD)
            acc = tl.fma(av[:, None], bv[None, :], acc)
    elif K % BK != 0:
        kk = K // BK * BK + rk
        av = tl.load(
            A + ra[:, None] * AM + kk[None, :] * AK,
            kk[None, :] < K,
            0,
            cache_modifier=LOAD,
        )
        bv = tl.load(
            B + kk[:, None] * BK_ + rb[None, :] * BN_,
            kk[:, None] < K,
            0,
            cache_modifier=LOAD,
        )
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if TRANS:
        offsets = rm[:, None] + rn[None, :] * M
    else:
        offsets = rm[:, None] * N + rn[None, :]
    if M % BM == 0 and N % BN == 0:
        tl.store(C + offsets, acc, cache_modifier=STORE)
    else:
        tl.store(
            C + offsets,
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
            cache_modifier=STORE,
        )
