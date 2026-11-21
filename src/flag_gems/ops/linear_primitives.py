# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared Linear computation primitives; caller owns launch configuration."""

import triton
import triton.language as tl


@triton.jit
def linear_program_ids(
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ORDER: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    if ORDER == 0:
        return tl.program_id(0), tl.program_id(1)
    pid = tl.program_id(0)
    nn = tl.cdiv(N, BLOCK_N)
    if ORDER == 1:
        return pid // nn, pid % nn
    nm = tl.cdiv(M, BLOCK_M)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_size = tl.minimum(nm - first_m, GROUP_M)
    return (
        first_m + (pid % (GROUP_M * nn)) % group_size,
        (pid % (GROUP_M * nn)) // group_size,
    )


@triton.jit
def linear_gemv_untransposed_kernel(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ki
        a = tl.load(
            X + mi[:, None] * K + k[None, :],
            (mi[:, None] < M) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        b = tl.load(
            W + ni[:, None] * K + k[None, :],
            (ni[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        products = a[:, None, :] * b[None, :, :]
        acc += tl.sum(products, axis=2)
    tl.store(
        Y + mi[:, None] * N + ni[None, :],
        acc.to(Y.dtype.element_ty),
        (mi[:, None] < M) & (ni[None, :] < N),
    )
