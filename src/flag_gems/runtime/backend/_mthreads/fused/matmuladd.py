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
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.fused.matmuladd import matmuladd as default_matmuladd

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.fused.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _matmuladd_kernel(
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
    stride_bi0,
    stride_bi1,
    stride_cm,
    stride_cn,
    BIAS_MODE: tl.constexpr,  # 0 = scalar, 1 = 1D(N,), 2 = 2D(M,N)
    PREC: tl.constexpr,  # 0 = native/tf32 dot, 1 = tf32x3 (accurate fp32)
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

    for k in range(0, K, BLOCK_K):
        if EVEN_K and EVEN_M and EVEN_N:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
            b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        if PREC == 0:
            acc = tl.dot(a, b, acc)
        else:
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if BIAS_MODE == 0:
        bias_val = tl.load(bias_ptr)
        acc = acc + bias_val
    elif BIAS_MODE == 1:
        if EVEN_N:
            bias = tl.load(bias_ptr + offs_n * stride_bias)
        else:
            bias = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    else:
        bias_ptrs = (
            bias_ptr + offs_m[:, None] * stride_bi0 + offs_n[None, :] * stride_bi1
        )
        if EVEN_M and EVEN_N:
            bias = tl.load(bias_ptrs)
        else:
            bmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            bias = tl.load(bias_ptrs, mask=bmask, other=0.0)
        acc = acc + bias

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc)
    else:
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        tl.store(c_ptrs, acc, mask=m_mask & n_mask)


@triton.jit
def _matmuladd_tma_kernel(
    a_desc,
    b_desc,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_bias,
    stride_bi0,
    stride_bi1,
    stride_cm,
    stride_cn,
    BIAS_MODE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k])
        b = b_desc.load([k, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc)

    if BIAS_MODE == 0:
        bias_val = tl.load(bias_ptr)
        acc = acc + bias_val
    elif BIAS_MODE == 1:
        if EVEN_N:
            bias = tl.load(bias_ptr + offs_n * stride_bias)
        else:
            bias = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    else:
        bias_ptrs = (
            bias_ptr + offs_m[:, None] * stride_bi0 + offs_n[None, :] * stride_bi1
        )
        if EVEN_M and EVEN_N:
            bias = tl.load(bias_ptrs)
        else:
            bmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            bias = tl.load(bias_ptrs, mask=bmask, other=0.0)
        acc = acc + bias

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc)
    else:
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        tl.store(c_ptrs, acc, mask=m_mask & n_mask)


def _bias_args(bias, M, N):
    if bias.numel() == 1:
        return 0, 0, 0, 0
    if bias.dim() == 2 and bias.shape[0] == M and bias.shape[1] == N:
        return 2, 0, bias.stride(0), bias.stride(1)
    return 1, bias.stride(-1), 0, 0


def _aligned(input, other, K):
    return (
        input.is_contiguous()
        and other.is_contiguous()
        and (input.stride(0) * input.element_size()) % 16 == 0
        and (other.stride(0) * other.element_size()) % 16 == 0
        and K % 32 == 0
    )


def matmuladd(input, other, bias):
    logger.debug("GEMS_MTHREADS MATMULADD")
    if (
        not isinstance(input, torch.Tensor)
        or input.device.type != "musa"
        or input.dtype not in _SUPPORTED_DTYPES
        or other.dtype != input.dtype
        or (bias is not None and bias.dtype != input.dtype)
    ):
        return default_matmuladd(input, other, bias)
    if input.dim() != 2 or other.dim() != 2 or input.shape[-1] != other.shape[0]:
        return default_matmuladd(input, other, bias)
    M, K = input.shape
    Kb, N = other.shape

    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    bias_mode, stride_bias, stride_bi0, stride_bi1 = _bias_args(bias, M, N)
    aligned = _aligned(input, other, K)

    if input.dtype == torch.float32:
        # torch's fp32 matmul on MUSA dispatches to tf32 tensor cores for large
        # enough shapes and to accurate fp32 (FMA) for small ones; match it.
        use_tf32 = (M >= 32 and N >= 32) or (M >= 8 and N >= 4096)
        if use_tf32:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 32, 8
            num_warps, num_stages, prec = 8, 2, 0
            if aligned and N >= 1024:
                a_desc = TensorDescriptor.from_tensor(
                    input, block_shape=[BLOCK_M, BLOCK_K]
                )
                b_desc = TensorDescriptor.from_tensor(
                    other, block_shape=[BLOCK_K, BLOCK_N]
                )
                grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
                _matmuladd_tma_kernel[grid](
                    a_desc,
                    b_desc,
                    bias,
                    out,
                    M,
                    N,
                    K,
                    stride_bias,
                    stride_bi0,
                    stride_bi1,
                    out.stride(0),
                    out.stride(1),
                    BIAS_MODE=bias_mode,
                    EVEN_M=(M % BLOCK_M == 0),
                    EVEN_N=(N % BLOCK_N == 0),
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    GROUP_M=GROUP_M,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                return out
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 32, 8
            num_warps, num_stages, prec = 8, 2, 1
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _matmuladd_kernel[grid](
            input,
            other,
            bias,
            out,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            stride_bias,
            stride_bi0,
            stride_bi1,
            out.stride(0),
            out.stride(1),
            BIAS_MODE=bias_mode,
            PREC=prec,
            EVEN_M=(M % BLOCK_M == 0),
            EVEN_N=(N % BLOCK_N == 0),
            EVEN_K=(K % BLOCK_K == 0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # fp16 / bf16
    if aligned and N >= 2048:
        if N >= 4096:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 2
            num_warps, num_stages = 4, 3
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 2
            num_warps, num_stages = 4, 2
        a_desc = TensorDescriptor.from_tensor(input, block_shape=[BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(other, block_shape=[BLOCK_K, BLOCK_N])
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _matmuladd_tma_kernel[grid](
            a_desc,
            b_desc,
            bias,
            out,
            M,
            N,
            K,
            stride_bias,
            stride_bi0,
            stride_bi1,
            out.stride(0),
            out.stride(1),
            BIAS_MODE=bias_mode,
            EVEN_M=(M % BLOCK_M == 0),
            EVEN_N=(N % BLOCK_N == 0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif N >= 512:
        # medium shapes (e.g. 1024x1024): smaller tiles for more parallelism
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 8
        num_warps, num_stages = 4, 2
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _matmuladd_kernel[grid](
            input,
            other,
            bias,
            out,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            stride_bias,
            stride_bi0,
            stride_bi1,
            out.stride(0),
            out.stride(1),
            BIAS_MODE=bias_mode,
            PREC=0,
            EVEN_M=(M % BLOCK_M == 0),
            EVEN_N=(N % BLOCK_N == 0),
            EVEN_K=(K % BLOCK_K == 0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 8
        num_warps, num_stages = 4, 2
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _matmuladd_kernel[grid](
            input,
            other,
            bias,
            out,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            stride_bias,
            stride_bi0,
            stride_bi1,
            out.stride(0),
            out.stride(1),
            BIAS_MODE=bias_mode,
            PREC=0,
            EVEN_M=(M % BLOCK_M == 0),
            EVEN_N=(N % BLOCK_N == 0),
            EVEN_K=(K % BLOCK_K == 0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
