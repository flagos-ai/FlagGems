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

from flag_gems.utils import libentry

_OUTPUT_DTYPES_16 = tl.constexpr((tl.float32,) * 16)
_OUTPUT_DTYPES_8 = tl.constexpr((tl.float32,) * 8)


@triton.jit
def _native_dot(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    STAGES: tl.constexpr,
):
    # Complete native body: dtype/layout scheduling is left to the compiler.
    pid = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    gm = tl.cdiv(M, TM)
    gn = tl.cdiv(N, TN)
    first = pid // (GROUP * gn) * GROUP
    gs = tl.minimum(GROUP, gm - first)
    m0 = (first + pid % (GROUP * gn) % gs) * TM
    n0 = pid % (GROUP * gn) // gs * TN
    rm = m0 + tl.arange(0, TM)
    rn = n0 + tl.arange(0, TN)
    rk = tl.arange(0, TK)
    acc = tl.zeros((TM, TN), tl.float32)
    for off in tl.range(0, K, TK, num_stages=STAGES, loop_unroll_factor=1):
        kk = off + rk
        av = tl.load(
            A + batch * M * K + rm[:, None].to(tl.int64) * K + kk[None, :],
            (rm[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            B + batch * K * N + kk[:, None].to(tl.int64) * N + rn[None, :],
            (kk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        C + batch * M * N + rm[:, None].to(tl.int64) * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry(cache_binding=True)
@triton.jit
def regular_2_128_128_16_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 128, 128, 16, 2)


@libentry(cache_binding=True)
@triton.jit
def regular_2_64_64_128_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 128, 1)


@libentry(cache_binding=True)
@triton.jit
def regular_0_64_64_128_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 128, 1)


@libentry(cache_binding=True)
@triton.jit
def regular_1_32_64_128_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 32, 64, 128, 1)


@libentry(cache_binding=True)
@triton.jit
def regular_0_64_64_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 32, 2)


@libentry(cache_binding=True)
@triton.jit
def regular_1_64_64_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 32, 2)


@libentry(cache_binding=True)
@triton.jit
def regular_0_128_128_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 128, 128, 32, 2)


@libentry(cache_binding=True)
@triton.jit
def regular_1_128_128_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 128, 128, 32, 2)


@libentry(cache_binding=True)
@triton.jit
def regular_2_64_64_16_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 16, 2)


REGULAR_KERNELS = {
    (2, 64, 64, 16, 2, 2): regular_2_64_64_16_2_2,
    (0, 128, 128, 32, 2, 2): regular_0_128_128_32_2_2,
    (0, 64, 64, 32, 2, 2): regular_0_64_64_32_2_2,
    (1, 128, 128, 32, 2, 2): regular_1_128_128_32_2_2,
    (1, 64, 64, 32, 2, 2): regular_1_64_64_32_2_2,
    (2, 128, 128, 16, 2, 2): regular_2_128_128_16_2_2,
    (2, 64, 64, 128, 2, 2): regular_2_64_64_128_2_2,
    (0, 64, 64, 128, 2, 2): regular_0_64_64_128_2_2,
    (1, 32, 64, 128, 2, 2): regular_1_32_64_128_2_2,
}


def launch_regular(a, b, c, config):
    dt, bm, bn, bk, wm, wn = config
    batch, m, k = a.shape
    n = b.shape[2]
    return REGULAR_KERNELS[config][(triton.cdiv(m, bm) * triton.cdiv(n, bn), batch)](
        a,
        b,
        c,
        m,
        n,
        k,
        1 if max(m, n) <= 512 and k <= 512 else 8,
        num_warps=wm * wn,
        num_stages=1,
        allow_flush_denorm=True,
        enable_fp_fusion=True,
    )
