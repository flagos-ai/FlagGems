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

import triton
import triton.language as tl


@triton.jit
def tiled(
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
    PREC: tl.constexpr,
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
    rk = tl.program_id(1) * tl.cdiv(tl.cdiv(K, BK), SPLIT) * BK + tl.arange(0, BK)
    dtype = tl.float64 if A.dtype.element_ty == tl.float64 else tl.float32
    acc = tl.zeros((BM, BN), dtype)
    for step in range(tl.cdiv(tl.cdiv(K, BK), SPLIT)):
        kk = rk + step * BK
        a = tl.load(
            A + rm[:, None] * AM + kk[None, :] * AK,
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        )
        b = tl.load(
            B + kk[:, None] * BK_ + rn[None, :] * BN_,
            (kk[:, None] < K) & (rn[None, :] < N),
            0,
        )
        acc = tl.dot(a, b, acc, input_precision=PREC, out_dtype=dtype)
    tl.store(
        C + tl.program_id(1) * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def split_reduce(P, C, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.full(
        (BLOCK,), 0, tl.float64 if P.dtype.element_ty == tl.float64 else tl.float32
    )
    for s in range(SPLIT):
        v += tl.load(P + s * SIZE + x, x < SIZE, 0)
    tl.store(C + x, v, x < SIZE)


@triton.jit
def small_k(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK_: tl.constexpr,
    BN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // N
    col = x % N
    acc = tl.full(
        (BLOCK,), 0, tl.float64 if A.dtype.element_ty == tl.float64 else tl.float32
    )
    for k in tl.static_range(K):
        a = tl.load(A + row * AM + k * AK, x < M * N, 0)
        b = tl.load(B + k * BK_ + col * BN, x < M * N, 0)
        acc = tl.fma(a, b, acc)
    tl.store(C + x, acc, x < M * N)


@triton.jit
def reduction(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK_: tl.constexpr,
    BN: tl.constexpr,
    BO: tl.constexpr,
    RK: tl.constexpr,
):
    x = tl.program_id(0) * BO + tl.arange(0, BO)
    kk = tl.arange(0, RK)
    av = tl.load(
        A + (x // N)[:, None] * AM + kk[None, :] * AK,
        (x[:, None] < M * N) & (kk[None, :] < K),
        0,
    )
    bv = tl.load(
        B + kk[None, :] * BK_ + (x % N)[:, None] * BN,
        (x[:, None] < M * N) & (kk[None, :] < K),
        0,
    )
    out = tl.sum(av * bv, 1)
    tl.store(C + x, out, x < M * N)


@triton.jit
def transpose(
    A,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0) * B + tl.arange(0, B)
    k = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(
        A + r[:, None] * AM + k[None, :] * AK, (r[:, None] < M) & (k[None, :] < K), 0
    )
    tl.store(C + k[None, :] * M + r[:, None], x, (r[:, None] < M) & (k[None, :] < K))


@triton.jit
def simt(
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
    SPLIT: tl.constexpr,
    UNROLL: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros(
        (BM, BN), tl.float64 if A.dtype.element_ty == tl.float64 else tl.float32
    )
    chunk = tl.cdiv(K, SPLIT)
    start = tl.program_id(2) * chunk
    for ki in tl.range(0, chunk, loop_unroll_factor=UNROLL):
        kk = start + ki
        av = tl.load(A + rm * AM + kk * AK, (rm < M) & (kk < K), 0)
        bv = tl.load(B + kk * BK_ + rn * BN_, (rn < N) & (kk < K), 0)
        acc = tl.fma(av[:, None], bv[None, :], acc)
    tl.store(
        C + tl.program_id(2) * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def tiled_fast(
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
    for step in range(K // BK):
        kk = step * BK + rk
        av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK)
        bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if K % BK != 0:
        kk = K // BK * BK + rk
        av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK, kk[None, :] < K, 0)
        bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_, kk[:, None] < K, 0)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N)
    )


@triton.jit
def tiled_unrolled(
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
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    ra = rm if M % BM == 0 else rm % M
    rb = rn if N % BN == 0 else rn % N
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in tl.range(K // BK, loop_unroll_factor=UNROLL):
        kk = step * BK + rk
        av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK)
        bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if TAIL:
        for st in tl.static_range((K % BK) // 8):
            kk = K // BK * BK + st * 8 + tl.arange(0, 8)
            av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK)
            bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        for st in tl.static_range(K % 8):
            kk = K // 8 * 8 + st
            av = tl.load(A + ra * AM + kk * AK)
            bv = tl.load(B + kk * BK_ + rb * BN_)
            acc = tl.fma(av[:, None], bv[None, :], acc)
    elif K % BK != 0:
        kk = K // BK * BK + rk
        av = tl.load(A + ra[:, None] * AM + kk[None, :] * AK, kk[None, :] < K, 0)
        bv = tl.load(B + kk[:, None] * BK_ + rb[None, :] * BN_, kk[:, None] < K, 0)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if TRANS:
        offsets = rm[:, None] + rn[None, :] * M
    else:
        offsets = rm[:, None] * N + rn[None, :]
    if M % BM == 0 and N % BN == 0:
        tl.store(C + offsets, acc)
    else:
        tl.store(C + offsets, acc, (rm[:, None] < M) & (rn[None, :] < N))
