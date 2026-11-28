# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Split the existing FP64 swizzle computation along K."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from .mm_mmac_swizzle import mmac_tiled


@g.jit
def swizzle_split(
    A,
    B,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    SPLIT: gl.constexpr,
):
    gl.static_assert(K >= SPLIT)
    part = gl.program_id(2)
    chunk: gl.constexpr = K // SPLIT
    tail: gl.constexpr = K - (SPLIT - 1) * chunk
    aa = A + part * chunk * AK
    bb = B + part * chunk * BK
    pp = P + part * M * N
    if tail == chunk:
        mmac_tiled(
            aa,
            bb,
            pp,
            M,
            N,
            chunk,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            1,
            True,
            True,
            False,
        )
    else:
        if part == SPLIT - 1:
            mmac_tiled(
                aa,
                bb,
                pp,
                M,
                N,
                tail,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                1,
                True,
                True,
                False,
            )
        else:
            mmac_tiled(
                aa,
                bb,
                pp,
                M,
                N,
                chunk,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                1,
                True,
                True,
                False,
            )
