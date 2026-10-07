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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

# KLX quirk (2026-09-07): the xpu ttsdnnir pass chain rewrites any kernel
# whose symbol name starts with "matmul_bias_activation_kernel" into the
# sdnn hardware GEMM fusion, whose bf16 accumulation loses ~0.42 rms at
# K=4096 (exceeds the 1e-4*K tolerance; only bf16 is affected -- fp16/fp32
# stay within tolerance). Keep the fast sdnn path under the original name,
# and route large-K bf16 to the un-hijacked generic kernel. Remove this
# split once the vendor fixes sdnn accumulation precision.
SDNN_PRECISE_MIN_K = 4096


@libentry()
@triton.jit(
    do_not_specialize=[
        "M",
        "N",
        "K",
        "stride_am",
        "stride_ak",
        "stride_bk",
        "stride_bn",
        "stride_bias",
        "stride_cm",
        "stride_cn",
    ]
)
def matmul_bias_activation_kernel(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # Apply ReLU activation
    accumulator = tl.maximum(accumulator, 0.0)

    tl.store(c_ptrs, accumulator, mask=c_mask)


@libentry()
@triton.jit
def fused_mba_kernel(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # Apply ReLU activation
    accumulator = tl.maximum(accumulator, 0.0)

    tl.store(c_ptrs, accumulator, mask=c_mask)


@libentry()
@triton.jit
def mba_strided_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_ptrs = (
        bias_ptr + stride_bias_m * offs_cm[:, None] + stride_bias_n * offs_cn[None, :]
    )
    bias_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

    accumulator = accumulator + bias
    accumulator = tl.maximum(accumulator, 0.0)
    c = accumulator.to(c_ptr.dtype.element_ty)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=bias_mask)


def matmul_bias_activation(input, weight, bias):
    """Compute ReLU(input @ weight + bias) with broadcastable bias."""
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    M, K = input.shape
    N = weight.shape[1]
    assert broadcastable_to(bias.shape, (M, N)), "Incompatible input shape"
    if input.stride(0) > 1 and input.stride(1) > 1:
        input = input.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    expanded_bias = bias.broadcast_to((M, N))
    vector_bias = (bias.ndim == 1 and bias.shape[0] == N) or (
        bias.ndim == 2 and bias.shape == (1, N)
    )
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    with torch_device_fn.device(input.device):
        if vector_bias:
            bias = bias.reshape(-1)
            kernel = (
                fused_mba_kernel
                if input.dtype == torch.bfloat16 and K >= SDNN_PRECISE_MIN_K
                else matmul_bias_activation_kernel
            )
            kernel[grid](
                input,
                weight,
                bias,
                out,
                M,
                N,
                K,
                input.stride(0),
                input.stride(1),
                weight.stride(0),
                weight.stride(1),
                bias.stride(0),
                out.stride(0),
                out.stride(1),
                128,
                128,
                32,
            )
        else:
            mba_strided_kernel[grid](
                input,
                weight,
                out,
                expanded_bias,
                M,
                N,
                K,
                input.stride(0),
                input.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                out.stride(1),
                expanded_bias.stride(0),
                expanded_bias.stride(1),
                128,
                128,
                32,
                num_warps=4,
                num_stages=3,
            )
    return out
