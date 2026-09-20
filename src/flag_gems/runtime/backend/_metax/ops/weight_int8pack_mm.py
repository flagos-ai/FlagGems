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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _mm_kernel(
    A,
    B,
    Scales,
    Out,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_om,
    stride_on,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    if EVEN:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)), acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        s = tl.load(Scales + offs_n * stride_sn)
        acc = acc * s.to(tl.float32)[None, :]
        o_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc.to(OUT_DTYPE))
    else:
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            km = (k0 + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask[:, None] & km[None, :], other=0)
            acc = tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)), acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        s = tl.load(Scales + offs_n * stride_sn, mask=n_mask, other=0.0)
        acc = acc * s.to(tl.float32)[None, :]
        o_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc.to(OUT_DTYPE), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _w8pack_mm_kernel(
    A,
    B,
    Scales,
    Out,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_om,
    stride_on,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    if EVEN:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)), acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        s = tl.load(Scales + offs_n * stride_sn)
        acc = acc * s.to(tl.float32)[None, :]
        o_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc.to(OUT_DTYPE))
    else:
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            km = (k0 + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask[:, None] & km[None, :], other=0)
            acc = tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)), acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        s = tl.load(Scales + offs_n * stride_sn, mask=n_mask, other=0.0)
        acc = acc * s.to(tl.float32)[None, :]
        o_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc.to(OUT_DTYPE), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _w8pack_gemv1_kernel(
    A,
    B,
    Scales,
    Out,
    N,
    K,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_on,
    OUT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    if EVEN:
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            a = tl.load(A + kk * stride_ak).to(tl.float16)
            a16 = tl.broadcast_to(a[None, :], (16, BLOCK_K))
            b = tl.load(B + offs_n[:, None] * stride_bn + kk[None, :] * stride_bk)
            acc = tl.dot(a16, tl.trans(b.to(tl.float16)), acc)
        s = tl.load(Scales + offs_n * stride_sn)
        acc = acc * s.to(tl.float32)[None, :]
        offs_m0 = tl.arange(0, 16)
        tl.store(
            Out + offs_m0[:, None] * stride_on + offs_n[None, :] * stride_on,
            acc.to(OUT_DTYPE),
            mask=(offs_m0 == 0)[:, None],
        )
    else:
        n_mask = offs_n < N
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            km = kk < K
            a = tl.load(A + kk * stride_ak, mask=km, other=0.0).to(tl.float16)
            a16 = tl.broadcast_to(a[None, :], (16, BLOCK_K))
            b = tl.load(
                B + offs_n[:, None] * stride_bn + kk[None, :] * stride_bk,
                mask=n_mask[:, None] & km[None, :],
                other=0,
            )
            acc = tl.dot(a16, tl.trans(b.to(tl.float16)), acc)
        s = tl.load(Scales + offs_n * stride_sn, mask=n_mask, other=0.0)
        acc = acc * s.to(tl.float32)[None, :]
        offs_m0 = tl.arange(0, 16)
        tl.store(
            Out + offs_m0[:, None] * stride_on + offs_n[None, :] * stride_on,
            acc.to(OUT_DTYPE),
            mask=(offs_m0 == 0)[:, None] & (offs_n < N)[None, :],
        )


def weight_int8pack_mm(A, B, scales):
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, (K, K2)

    A = A.contiguous()
    B = B.contiguous()
    if scales.dim() == 2 and 1 in scales.shape:
        scales = scales.reshape(-1)
    scales = scales.contiguous()

    out = torch.empty((M, N), dtype=A.dtype, device=A.device)
    out_dtype = tl.float16 if A.dtype == torch.float16 else tl.bfloat16

    if M == 1 and N >= 16384:
        even = (N % 64 == 0) and (K % 256 == 0)
        grid = (triton.cdiv(N, 64),)
        _w8pack_gemv1_kernel[grid](
            A,
            B,
            scales,
            out,
            N,
            K,
            A.stride(1),
            B.stride(0),
            B.stride(1),
            scales.stride(0),
            out.stride(1),
            OUT_DTYPE=out_dtype,
            BLOCK_N=64,
            BLOCK_K=256,
            EVEN=even,
            num_warps=4,
            num_stages=3,
        )
        return out

    if A.dtype == torch.float16:
        if M <= 16:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 16, 64, 256, 4
        elif M <= 32:
            if K >= N:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 32, 128, 4
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 64, 128, 4
        elif M <= 64:
            if K > N:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 32, 128, 4
            elif K == N:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 128, 4
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 64, 8
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 64, 8
    else:  # bfloat16
        if M <= 16:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 16, 64, 256, 4
        elif M <= 32:
            if K >= N:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 32, 128, 4
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 64, 128, 4
        elif M <= 64:
            if K >= N:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 128, 4
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 64, 8
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 128, 64, 2

    even = (M % BLOCK_M == 0) and (N % BLOCK_N == 0) and (K % BLOCK_K == 0)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _w8pack_mm_kernel[grid](
        A,
        B,
        scales,
        out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        scales.stride(0),
        out.stride(0),
        out.stride(1),
        OUT_DTYPE=out_dtype,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=8,
        EVEN=even,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
