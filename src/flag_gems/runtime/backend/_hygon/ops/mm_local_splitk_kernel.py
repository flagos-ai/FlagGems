# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Four logical K partitions, batched dot, and ordered FP32 partial sums."""

import triton
import triton.language as tl

from flag_gems.utils import libentry


@triton.jit
def _write_partial_sum(
    accumulator,
    C,
    mi,
    ni,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pairs = tl.reshape(tl.permute(accumulator, (1, 2, 0)), (BM * BN, 2, 2))
    even, odd = tl.split(pairs)
    low, high = tl.split(even + odd)
    value = tl.reshape(low + high, (BM, BN))
    tl.store(
        C + mi[:, None].to(tl.int64) * N + ni[None, :],
        value.to(C.dtype.element_ty),
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@libentry()
@triton.jit
def _mm_local_splitk_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    K_STEP: tl.constexpr = 256,
    B_ROW_MAJOR: tl.constexpr = False,
    M_TILES: tl.constexpr = 1,
):
    tl.static_assert(K_STEP == 128 or K_STEP == 256)
    tl.static_assert(M_TILES > 0 and BM >= 16 and BN >= 16)
    row = tl.program_id(0) * (M_TILES * BM)
    col = tl.program_id(1) * BN
    part = tl.arange(0, 4)
    offsets = tl.arange(0, BM)
    ni = col + tl.arange(0, BN)
    ki = part[:, None] * (K_STEP // 4) + tl.arange(0, K_STEP // 4)[None, :]
    accumulators = ()
    for i in tl.static_range(M_TILES):
        accumulators = accumulators + (tl.zeros((4, BM, BN), tl.float32),)
    for start in range(tl.cdiv(K, K_STEP)):
        kk = start * K_STEP + ki
        if B_ROW_MAJOR:
            bv = tl.load(
                B + kk[:, :, None].to(tl.int64) * N + ni[None, None, :],
                (kk[:, :, None] < K) & (ni[None, None, :] < N),
                other=0,
            )
        else:
            bv = tl.load(
                B + ni[None, None, :].to(tl.int64) * K + kk[:, :, None],
                (kk[:, :, None] < K) & (ni[None, None, :] < N),
                other=0,
            )
        next_accumulators = ()
        for i in tl.static_range(M_TILES):
            mi = row + i * BM + offsets
            av = tl.load(
                A + mi[None, :, None].to(tl.int64) * K + kk[:, None, :],
                (mi[None, :, None] < M) & (kk[:, None, :] < K),
                other=0,
            )
            next_accumulators = next_accumulators + (
                tl.dot(av, bv, accumulators[i], allow_tf32=False),
            )
        accumulators = next_accumulators
    for i in tl.static_range(M_TILES):
        _write_partial_sum(accumulators[i], C, row + i * BM + offsets, ni, M, N, BM, BN)
