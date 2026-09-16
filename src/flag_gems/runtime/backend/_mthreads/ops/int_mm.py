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

import triton
import triton.language as tl

from flag_gems.ops.int_mm import _int_mm_impl, _zero_output

logger = logging.getLogger(__name__)

_DIRECT_MAX_M = 32
_DIRECT_MAX_K = 64


@triton.jit
def _int_mm_direct_split_k_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    chunk_start = pid_k * (2 * BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_block in range(0, 2):
        current_k = chunk_start + k_block * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (current_k[None, :] < K),
            other=0,
        )
        b = tl.load(
            b_ptr + current_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(current_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, b, out_dtype=tl.int32)

    tl.atomic_add(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _int_mm_bf16_split_k_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    chunk_start = pid_k * CHUNK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, CHUNK_K, BLOCK_K):
        current_k = chunk_start + k_offset + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (current_k[None, :] < K),
            other=0,
        )
        b = tl.load(
            b_ptr + current_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(current_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        # BF16 represents every int8 operand exactly. Its product is accumulated
        # in FP32, and a 1024-element chunk has an absolute partial-sum bound of
        # 2**24, so the conversion to int32 is exact before the modular add.
        acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)

    tl.atomic_add(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.int32),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _launch(self, mat2, out, M, N, K):
    _zero_output(out, M * N)
    if M <= _DIRECT_MAX_M or K <= _DIRECT_MAX_K:
        block_m, block_n, block_k, chunk_k = 16, 16, 32, 64
        grid = (
            triton.cdiv(M, block_m),
            triton.cdiv(N, block_n),
            triton.cdiv(K, chunk_k),
        )
        _int_mm_direct_split_k_kernel[grid](
            self,
            mat2,
            out,
            M,
            N,
            K,
            self.stride(0),
            self.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            out.stride(0),
            out.stride(1),
            block_m,
            block_n,
            block_k,
            num_warps=4,
        )
    else:
        block_m, block_n, block_k, chunk_k = 64, 64, 64, 1024
        grid = (
            triton.cdiv(M, block_m),
            triton.cdiv(N, block_n),
            triton.cdiv(K, chunk_k),
        )
        _int_mm_bf16_split_k_kernel[grid](
            self,
            mat2,
            out,
            M,
            N,
            K,
            self.stride(0),
            self.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            out.stride(0),
            out.stride(1),
            block_m,
            block_n,
            block_k,
            chunk_k,
            num_warps=4,
        )


def int_mm(self, mat2):
    logger.debug("GEMS MTHREADS INT_MM")
    return _int_mm_impl(self, mat2, _launch)


def int_mm_out(self, mat2, *, out):
    logger.debug("GEMS MTHREADS INT_MM_OUT")
    return _int_mm_impl(self, mat2, _launch, out=out)


__all__ = ["int_mm", "int_mm_out"]
