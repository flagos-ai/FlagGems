# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FP32 split-K with unmasked full K blocks and a masked final block.

Every split writes its own partial output, including empty splits. The
existing deterministic split reduction produces the contiguous public result.
"""

import triton
import triton.language as tl


@triton.jit
def tiled_fast_split(
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
    GROUP: tl.constexpr,
    SPLIT: tl.constexpr,
    TRANS: tl.constexpr,
):
    p = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    group = p // (GROUP * nn)
    size = tl.minimum(nm - group * GROUP, GROUP)
    pm = group * GROUP + (p % (GROUP * nn)) % size
    pn = (p % (GROUP * nn)) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    if M % BM == 0:
        ra = rm
    else:
        ra = rm % M
    if N % BN == 0:
        rb = rn
    else:
        rb = rn % N
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    blocks: tl.constexpr = triton.cdiv(triton.cdiv(K, BK), SPLIT)
    start = tl.program_id(1) * blocks * BK
    full = tl.minimum(blocks, tl.maximum(0, (K - start) // BK))
    for step in range(full):
        kk = start + step * BK + rk
        av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK)
        bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if (full < blocks) & (start + full * BK < K):
        kk = start + full * BK + rk
        av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK, kk[None, :] < K, 0)
        bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_, kk[:, None] < K, 0)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if TRANS:
        tl.store(
            C + tl.program_id(1) * M * N + rn[None, :] * M + rm[:, None],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
    else:
        tl.store(
            C + tl.program_id(1) * M * N + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
