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


@triton.jit
def _dot_accum(a, b, acc, FP32_IEEE: tl.constexpr):
    if FP32_IEEE:
        return tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="ieee")
    else:
        return tl.dot(a, b, acc, out_dtype=tl.float32)


@triton.jit
def _mba_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_M: tl.constexpr,
    MASK_N: tl.constexpr,
    FP32_IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K and (not MASK_M) and (not MASK_N):
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = _dot_accum(a, b, acc, FP32_IEEE)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    elif EVEN_K:
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=m_mask, other=0.0)
            b = tl.load(b_ptrs, mask=n_mask, other=0.0)
            acc = _dot_accum(a, b, acc, FP32_IEEE)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_off = k * BLOCK_K + offs_k
            a = tl.load(a_ptrs, mask=m_mask & (k_off[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_off[:, None] < K) & n_mask, other=0.0)
            acc = _dot_accum(a, b, acc, FP32_IEEE)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.where(acc > 0, acc, 0.0)
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_bias_activation(input, weight, bias):
    M, K = input.shape
    K2, N = weight.shape
    assert K2 == K

    if bias.dim() > 1:
        bias = bias.reshape(-1)
    assert bias.numel() == N

    a = input
    b = weight
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_bias = bias.stride(0)

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = out.stride(0), out.stride(1)

    dtype = a.dtype
    if dtype in (torch.float16, torch.bfloat16):
        if M > 2048 and N > 2048:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 32
            num_warps, num_stages = 16, 2
        elif M > 1024 and N > 1024:
            BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 32
            num_warps, num_stages = 16, 2
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
            num_warps, num_stages = 8, 2
        fp32_ieee = False
    elif dtype == torch.float32:
        if M <= 512 and N <= 512:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            num_warps, num_stages = 4, 2
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
            num_warps, num_stages = 8, 2
        fp32_ieee = True
    else:
        raise NotImplementedError(f"unsupported dtype {dtype}")

    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _mba_kernel[grid](
        a,
        b,
        bias,
        out,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_bias,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        EVEN_K=(K % BLOCK_K == 0),
        MASK_M=(M % BLOCK_M != 0),
        MASK_N=(N % BLOCK_N != 0),
        FP32_IEEE=fp32_ieee,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
