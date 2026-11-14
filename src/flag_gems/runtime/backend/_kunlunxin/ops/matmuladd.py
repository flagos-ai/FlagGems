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
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


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
        "stride_cm",
        "stride_cn",
        "stride_bias_m",
        "stride_bias_n",
    ]
)
def matmuladd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias_m,
    stride_bias_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

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

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=bias_mask)


@libentry()
@triton.jit
def add_bias_strided_kernel(
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
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

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

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=bias_mask)


def _matmuladd(input, other, bias, out=None):
    assert input.shape[1] == other.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], other.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = other.shape
    if out is None:
        out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    if input.stride(0) > 1 and input.stride(1) > 1:
        input = input.contiguous()
    if other.stride(0) > 1 and other.stride(1) > 1:
        other = other.contiguous()
    bias = bias.broadcast_to(out.shape)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
        row_bias = bias.stride(0) == 0 or M == 1
        kernel = matmuladd_kernel if row_bias else add_bias_strided_kernel
        kernel[grid](
            input,
            other,
            out,
            bias,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            bias.stride(0),
            bias.stride(1),
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32,
            num_warps=4,
            num_stages=3,
        )
    return out


def matmuladd(input, other, bias):
    """
    Matrix multiplication with addition: output = matmul(input, other) + bias
    """
    logger.debug("GEMS MATMULADD")
    return _matmuladd(input, other, bias)
