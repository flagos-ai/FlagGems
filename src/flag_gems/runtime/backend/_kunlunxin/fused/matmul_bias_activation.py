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
# Ports the GEMM recipe proven on `_kunlunxin/ops/matmuladd.py` (2026-09-05):
#   1. bias is materialized to a contiguous (M, N) 2D tensor first -- a 1D
#      bias that broadcasts along M (stride_im == 0) runs the epilogue load
#      ~1.2-1.4x slower on this backend.
#   2. dtype/shape-adaptive tile: small square (M,N<=512) keeps 128-tile
#      warps=4; large fp16/fp32 use a wide-N tile (BM=256, BN=512); large
#      bf16 keeps the square 256-tile (BN=512 regresses bf16, and bf16
#      BN=512 + w16 is a 369ms catastrophic compile on 2048^3).
#   3. fp32 wide-N MUST run at num_warps=16 (the same tile at warps=8 is a
#      1254ms mis-compile on 4096^3 -- a sharp cliff, never route fp32
#      wide-N to w=8).
#   4. BK: fp16=256, bf16/fp32=128 (same rule as addmm/matmuladd).
#
# The activation (ReLU) is the difference vs matmuladd. Fusing `tl.maximum`
# directly on the fp32 accumulator tile only compiles for the small 128-tile
# and the bf16 square 256-tile; on the fp16/fp32 wide-N (256x512) tiles it
# aborts the XPU compiler ("out of resource: uni_sram" / "operand #1 does
# not dominate this use"). So:
#   * small shapes + large bf16: ReLU fused in the epilogue (1 launch).
#   * large fp16/fp32: ReLU applied by a dedicated flat pointwise pass
#     (~0.08ms @4096^2 fp16) after the GEMM stores raw acc+bias.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# dtype codes handed to the kernel (plain int runtime args, so the heuristics
# below can branch on the input dtype the same way matmuladd does).
_FP16, _BF16, _FP32 = 0, 1, 2


def _dtype_code(dtype):
    return {torch.float16: _FP16, torch.bfloat16: _BF16, torch.float32: _FP32}[dtype]


# Tile heuristics (P800 / XPU3, swept on the official core shapes, same
# decision rules as matmuladd -- see its header comment for the evidence).
def heur_block_m(args):
    if args["M"] <= 512:
        return 128
    return 256


def heur_block_n(args):
    if args["N"] <= 512:
        return 128
    if args.get("DTYPE_CODE", _FP16) == _BF16:
        return 256
    return 512


def heur_block_k(args):
    if args.get("DTYPE_CODE", _FP16) == _FP16:
        return 256
    return 128


def heur_warps(args):
    if args["M"] <= 512 and args["N"] <= 512:
        return 4
    if args.get("DTYPE_CODE", _FP16) == _FP32:
        return 16
    return 8


def heur_stages(args):
    return 3


autotune_decorator = triton.heuristics(
    {
        "BLOCK_SIZE_M": heur_block_m,
        "BLOCK_SIZE_N": heur_block_n,
        "BLOCK_SIZE_K": heur_block_k,
        "num_warps": heur_warps,
        "num_stages": heur_stages,
    }
)


@libentry()
@autotune_decorator
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
    DTYPE_CODE,
    FUSE_RELU: tl.constexpr,
):
    # Same GEMM structure as the kunlunxin addmm/matmuladd kernel: 1-D grid
    # with GROUP_M swizzle, masked K loop, fp32 accumulator.
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
    # ReLU fused in the epilogue where the compiler allows it (small 128-tile
    # and bf16 square 256-tile). For the fp16/fp32 wide-N tiles FUSE_RELU is
    # False and a separate pointwise pass applies ReLU afterwards.
    if FUSE_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
    # Let tl.store convert to the output pointer dtype.
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def relu_kernel(
    x_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Flat 1D pass over a contiguous tensor. This is ~175x faster than a
    # masked 2D-tile pass on XPU (~0.08ms on 4096^2 fp16).
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


def _fuse_relu(M, N, dtype):
    # Fused epilogue only where verified to compile: small square 128-tile
    # (all dtypes) and large bf16 square 256-tile. fp16/fp32 wide-N tiles
    # abort the XPU compiler -> separate ReLU pass.
    if M <= 512 and N <= 512:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    Vendor kernel reusing the matmuladd GEMM structure (see header comment):
    materialized 2D bias, dtype/shape-adaptive tile, ReLU fused in the
    epilogue for small/bf16 configs and applied by a dedicated pointwise pass
    for the fp16/fp32 wide-N configs (compiler constraint).

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Bias vector of shape (N,) or (1, N) or (M, N)

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
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    bias = bias.broadcast_to((M, N)).contiguous()

    fuse_relu = _fuse_relu(M, N, input.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
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
            DTYPE_CODE=_dtype_code(input.dtype),
            FUSE_RELU=fuse_relu,
            # NOTE: do NOT pass num_stages here; the heuristics decorator
            # supplies it (same rationale as addmm/matmuladd).
        )
        if not fuse_relu:
            numel = M * N
            relu_block = 16384
            need_mask = numel % relu_block != 0
            relu_kernel[
                (triton.cdiv(numel, relu_block),)
            ](out, numel, BLOCK_SIZE=relu_block, NEED_MASK=need_mask)
    return out
