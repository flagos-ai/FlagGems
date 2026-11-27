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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _router_vector(
    X,
    W,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    XM: tl.constexpr,
    XK: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BK: tl.constexpr,
    SIZE: tl.constexpr,
    GRID: tl.constexpr,
):
    first = tl.program_id(0)
    k = tl.arange(0, BK)
    end = SIZE if GRID < SIZE else first + 1
    for pid in range(first, end, GRID):
        m = pid // N
        n = pid % N
        xp = X + m * XM + k * XK
        wp = W + n * WN + k * WK
        acc = tl.zeros((BK,), tl.float32)
        for start in range(0, K, BK):
            valid = start + k < K
            x = tl.load(xp, valid, 0).to(tl.float32)
            w = tl.load(wp, valid, 0).to(tl.float32)
            acc += x * w
            xp += BK * XK
            wp += BK * WK
        tl.store(Y + pid, tl.sum(acc, 0))


@libentry()
@triton.jit
def _router_cube(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XM: tl.constexpr,
    XK: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    z = tl.program_id(1)
    m = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    n = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    steps = tl.cdiv(K, BK * SPLIT)
    k = z * steps * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(steps):
        a = tl.load(
            X + m[:, None] * XM + k[None, :] * XK,
            (m[:, None] < M) & (k[None, :] < K),
            0,
        )
        if WN == 1:
            b = tl.load(
                W + k[:, None] * WK + n[None, :],
                (k[:, None] < K) & (n[None, :] < N),
                0,
            )
        else:
            w = tl.load(
                W + n[:, None] * WN + k[None, :] * WK,
                (n[:, None] < N) & (k[None, :] < K),
                0,
            )
            b = tl.trans(w)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        k += BK
    tl.store(
        Y + z * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@libentry()
@triton.jit
def _router_reduce(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for z in range(SPLIT):
        value += tl.load(P + z * SIZE + i, i < SIZE, 0)
    tl.store(Y + i, value, i < SIZE)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute router scores with FP32 accumulation and output."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("router_gemm expects x[M, K] and weight[N, K]")
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError("router_gemm inputs must have the same device and dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("router_gemm supports float16, bfloat16 and float32")
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    if M == 0 or N == 0:
        return out
    size = M * N
    weight_strides = weight.stride()
    strides = (*x.stride(), *weight_strides)
    dense_nn = N % 16 == 0 and weight_strides == (1, N)
    low_precision = x.dtype in (torch.float16, torch.bfloat16)
    small_short_k_nt = (
        low_precision
        and not dense_nn
        and M <= 16
        and K == 2048
        and N >= 256
        and N % 32 == 0
    )
    medium_short_k = low_precision and 512 <= M <= 1024 and K == 2048 and N % 256 == 0
    wide_bf16_nt = x.dtype == torch.bfloat16 and N >= 256 and not dense_nn
    # Cube reads either dense weight layout directly.
    vector_fallback = (
        K == 0
        or K % 128 != 0
        or (N != 8 and N % 16 != 0)
        or not x.is_contiguous()
        or not (weight.is_contiguous() or dense_nn)
        or x.data_ptr() % 16 != 0
        or weight.data_ptr() % 16 != 0
    )
    small_nt = M == 1 and not (dense_nn or wide_bf16_nt or small_short_k_nt)
    with torch_device_fn.device(x.device):
        if vector_fallback or ((small_nt or (M <= 16 and N <= 8)) and K <= 8192):
            block_k = max(1, triton.next_power_of_2(K))
            if vector_fallback:
                block_k = min(1024, block_k)
            # Ascend launch dimensions must fit in 16 bits.
            grid = size if size <= 65535 else 4096
            _router_vector[(grid,)](x, weight, out, N, K, *strides, block_k, size, grid)
            return out

        if small_short_k_nt:
            # Narrow N tiles expose parallel work without splitting K.
            bm, bn, bk = 16, 32, 512
        elif medium_short_k:
            bm, bn, bk = (64, 128, 256) if dense_nn else (128, 64, 512)
        elif low_precision and M >= 8192 and N <= 384 and N % 64 == 0 and K % 256 == 0:
            bm, bn, bk = 256, min(N, 128), 256
            if dense_nn:
                if K == 2048 and N % 256 == 0:
                    bm, bn = 128, 256
                elif N < 256:
                    bm = 128
        elif wide_bf16_nt:
            bm, bn, bk = min(128, max(16, triton.next_power_of_2(M))), 128, 512
        elif dense_nn:
            bm = 16 if M <= 16 else 32 if M < 128 else 128
            bn, bk = min(triton.next_power_of_2(N), 128), 256
        else:
            bm = 16 if M <= 32 else 32 if M < 128 else 64
            bn, bk = (16 if N <= 16 else 64), 128

        tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
        # Target 20 output tasks on 910B, with at most eight K partitions.
        split = (
            min(8, triton.next_power_of_2(triton.cdiv(20, tiles)))
            if K >= 2048 and not (small_short_k_nt or medium_short_k)
            else 1
        )
        # Avoid masked or empty K partitions in the matrix pipeline.
        while bk > 128 and K % (bk * split) != 0:
            bk //= 2
        while split > 1 and K % (bk * split) != 0:
            split //= 2
        partial = (
            torch.empty((split, M, N), device=x.device, dtype=torch.float32)
            if split > 1
            else out
        )
        _router_cube[(tiles, split)](
            x, weight, partial, M, N, K, *strides, bm, bn, bk, split
        )
        if split > 1:
            _router_reduce[(triton.cdiv(size, 1024),)](partial, out, size, split, 1024)
    return out
