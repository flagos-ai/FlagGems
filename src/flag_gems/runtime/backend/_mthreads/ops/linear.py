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

"""Triton implementation of linear: out = input @ weight.T + bias.

Reference semantics (torch.nn.functional.linear):
  input:  (..., K)
  weight: (N, K)
  bias:   (N,)
  out:    (..., N)
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linear import linear as default_linear

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@triton.jit
def _linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    USE_TF32: tl.constexpr,
    X_CG: tl.constexpr,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # weight is (N, K); load tile as (BLOCK_K, BLOCK_N) for tl.dot
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            if EVEN_M:
                if X_CG:
                    x = tl.load(x_ptrs, cache_modifier=".cg")
                else:
                    x = tl.load(x_ptrs)
            else:
                x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
            if EVEN_N:
                w = tl.load(w_ptrs)
            else:
                w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            kmask = (k * BLOCK_K + offs_k) < K
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & kmask[:, None], other=0.0)
        if FP32_INPUT:
            if USE_TF32:
                acc = tl.dot(x, w, acc, input_precision="tf32")
            else:
                acc = tl.dot(x, w, acc, input_precision="ieee")
        else:
            acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if EVEN_N:
        bias = tl.load(b_ptr + offs_n)
    else:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    if EVEN_M and EVEN_N:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty))
    else:
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def linear(input, weight, bias=None):
    logger.debug("GEMS_MTHREADS LINEAR")
    if (
        not isinstance(input, torch.Tensor)
        or input.device.type != "musa"
        or input.dtype not in _SUPPORTED_DTYPES
        or weight.dtype != input.dtype
        or weight.dim() != 2
        or input.dim() < 1
    ):
        return default_linear(input, weight, bias)

    in_shape = input.shape
    orig_dim = input.dim()
    x = input
    if orig_dim == 1:
        x = x.reshape(1, -1)
    elif orig_dim > 2:
        x = x.reshape(-1, x.shape[-1])
    M, K = x.shape
    N = weight.shape[0]

    # Tile dispatch on MTT S5000 (microbenchmark-verified):
    #  - large fp32 (tf32) squares: 256x128x32/w8/ns2/g4 (4096^3: 6.44->3.24ms)
    #  - fp16/bf16 M>=2048: 64x64x32/w4/g4 + .cg x-loads (4096^3: 1.68->1.48ms)
    #  - fp16/bf16 M=1024: 64x32x64/w4/g4 (0.0487->0.0357ms)
    #  - fp32 1024: 64x64x32/w4/g8 (unchanged)
    #  - M<1024 shapes (384s + correctness): 32x32x32 (fp32) / 32x32x64 (fp16/bf16)
    #    raise CTA count from 36 to 144, ~2x faster on the 384 half workloads.
    if input.dtype == torch.float32:
        if M >= 2048 and N >= 2048:
            bm, bn, bk, nw, ns, gm, xcg = 256, 128, 32, 8, 2, 4, False
        elif M >= 1024:
            bm, bn, bk, nw, ns, gm, xcg = 64, 64, 32, 4, 1, 8, False
        else:
            bm, bn, bk, nw, ns, gm, xcg = 32, 32, 32, 4, 1, 8, False
    else:
        if M >= 2048:
            bm, bn, bk, nw, ns, gm, xcg = 64, 64, 32, 4, 1, 4, True
        elif M >= 1024:
            bm, bn, bk, nw, ns, gm, xcg = 64, 32, 64, 4, 1, 4, False
        else:
            bm, bn, bk, nw, ns, gm, xcg = 32, 32, 64, 4, 1, 8, False

    out_shape = in_shape[:-1] + (N,)
    out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    if orig_dim == 1:
        y = out.reshape(1, N)
    elif orig_dim > 2:
        y = out.reshape(M, N)
    else:
        y = out

    if bias is None:
        b = torch.zeros(N, device=input.device, dtype=input.dtype)
    else:
        b = bias

    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _linear_kernel[grid](
        x,
        weight,
        b,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        EVEN_M=(M % bm == 0),
        EVEN_N=(N % bn == 0),
        EVEN_K=(K % bk == 0),
        FP32_INPUT=(input.dtype == torch.float32),
        USE_TF32=((input.dtype == torch.float32) and (M >= 32)),
        X_CG=xcg,
        num_warps=nw,
        num_stages=ns,
    )
    return out
