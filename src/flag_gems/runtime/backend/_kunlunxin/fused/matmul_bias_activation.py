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

# Kunlunxin(XPU) backend override for the fused matmul+bias+ReLU operator.
#
# The generic `flag_gems.fused.matmul_bias_activation` kernel (BLOCK_K=32,
# 1D bias broadcast `bias[None, :]`) fails to lower on XPU inside
# `ConvertTritonSDNNToLLVM` (compile error, all shapes/dtypes fail).
# This override reuses the structure proven in `_kunlunxin/ops/addmm.py`:
#   256/128 tiles + GROUP_M swizzle, dtype-dependent BLOCK_SIZE_K (fp16 -> 256,
#   bf16/fp32 -> 128), masked K-loop loads with other=0.0, fp32 accumulation,
#   epilogue bias load as a full 2D tile + ReLU.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def heur_block_m(args):
    M = args["M"]
    if M <= 512:
        return 128
    return 256


def heur_block_n(args):
    N = args["N"]
    if N <= 512:
        return 128
    return 256


def heur_block_k(args):
    # The wrapper passes BLOCK_K_CHOICE (fp16 -> 256, else 128).
    if args.get("BLOCK_K_CHOICE", 128) == 256:
        return 256
    return 128


def heur_warps(args):
    if args["M"] <= 512 and args["N"] <= 512:
        return 4
    return 8


@libentry()
@triton.heuristics(
    {
        "BLOCK_SIZE_M": heur_block_m,
        "BLOCK_SIZE_N": heur_block_n,
        "BLOCK_SIZE_K": heur_block_k,
        "num_warps": heur_warps,
    }
)
@triton.jit
def matmul_bias_activation_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_K_CHOICE,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # re-order program ID for better L2 reuse along the N dimension
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

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
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator + bias
    # NOTE: a ReLU (or any compare/abs/select) fused directly on the
    # fp32 tile right after tl.dot crashes the XPU compiler inside
    # `ConvertTritonSDNNToLLVM` (isolated with the same kernel body: add/mul
    # epilogue compiles, maximum/where/abs/minimum all fail). The ReLU is
    # therefore applied by a dedicated pointwise kernel afterwards.
    # Let tl.store convert to the output pointer dtype.
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def relu_kernel(
    x_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Flat 1D pass over a contiguous tensor. The previous 2D-tile
    # (BLOCK 128x128 masked) version ran at ~2.5 GB/s on XPU (13.5ms on
    # 4096^2 fp16); a flat strided-1 pass with NEED_MASK specialization is
    # ~175x faster (~0.08ms, same as the vendor pointwise relu).
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = tl.maximum(x, 0.0)
        tl.store(x_ptr + offs, x, mask=mask)
    else:
        x = tl.load(x_ptr + offs)
        x = tl.maximum(x, 0.0)
        tl.store(x_ptr + offs, x)


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Bias vector of shape (N,) or (1, N)

    Returns:
        Output tensor of shape (M, N) with ReLU activation applied
    """
    logger.debug("GEMS_KUNLUNXIN MATMUL_BIAS_ACTIVATION")
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = weight.shape

    input = input.contiguous()
    weight = weight.contiguous()
    if bias.dim() > 1:
        bias = bias.reshape(-1)
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    bias = bias.broadcast_to(out.shape)

    block_k_choice = 256 if input.dtype == torch.float16 else 128
    with torch_device_fn.device(input.device):
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        matmul_bias_activation_kernel[grid](
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
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            GROUP_M=8,
            BLOCK_K_CHOICE=block_k_choice,
            num_stages=3,
        )
        numel = M * N
        relu_block = 16384
        need_mask = numel % relu_block != 0
        relu_kernel[(triton.cdiv(numel, relu_block),)](
            out, numel, BLOCK_SIZE=relu_block, NEED_MASK=need_mask
        )
    return out
