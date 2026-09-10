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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# dtype codes handed to the kernel (plain int runtime args, so the heuristics
# below can branch on the input dtype the same way addmm branches on
# BLOCK_K_CHOICE).
_FP16, _BF16, _FP32 = 0, 1, 2


def _dtype_code(dtype):
    return {torch.float16: _FP16, torch.bfloat16: _BF16, torch.float32: _FP32}[dtype]


# Tile heuristics (P800 / XPU3, swept 2026-09-05 on the official core shapes):
#   * small square (M,N <= 512) keeps the 128-tile warps=4 config, as addmm does.
#   * large fp16/fp32 prefer a wide-N tile (BM=256, BN=512): fp16 4096^3
#     0.94ms -> 0.79ms, fp32 4096^3 2.27ms -> 1.22ms; the addmm default square
#     256-tile is ~1.2-1.9x slower on this backend.
#   * large bf16 keeps the square 256-tile (BN=512 regresses bf16, and BN=512
#     w=16 has a catastrophic compile on this backend - 369ms on 2048^3).
#   * fp32 wide-N needs num_warps=16 (the same tile at warps=8 is a 1254ms
#     mis-compile on 4096^3 - a sharp cliff, never route fp32 wide-N to w=8).
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
    # fp16 prefers BK=256, bf16/fp32 BK=128 (same rule as addmm).
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
@triton.jit(do_not_specialize=["alpha", "beta"])
def matmuladd_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
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
):
    # Same GEMM structure as the kunlunxin addmm kernel: 1-D grid with GROUP_M
    # swizzle, masked K loop, fp32 accumulator.
    pid = ext.program_id(0)
    if GROUP_M > 1:
        grid_m = tl.cdiv(M, BLOCK_SIZE_M)
        grid_n = tl.cdiv(N, BLOCK_SIZE_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
    else:
        pid_m = ext.program_id(1)
        pid_n = ext.program_id(2)

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

    accumulator = accumulator * alpha + bias * beta
    # Let tl.store convert to the output pointer dtype (fp32 out with fp16/bf16
    # inputs is allowed).
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmuladd(input, other, bias):
    """
    Matrix multiplication with addition: output = matmul(input, other) + bias

    Vendor kernel with the same GEMM structure as the kunlunxin addmm, but a
    dtype/shape-adaptive tile (see heuristics above) that is measurably faster
    than addmm's fixed square tile on the P800/XPU3 core shapes.

    The bias is materialised to a contiguous (M, N) tensor first: a bias that
    broadcasts along M (e.g. the 1-D [N] bias used by the official benchmark)
    reaches the kernel with stride_im == 0, and that broadcast-along-M epilogue
    load runs ~1.2-1.4x slower than a plain 2-D load on this backend. For an
    already-contiguous (M, N) bias the broadcast/contiguous is a no-op.
    """
    logger.debug("GEMS_KUNLUNXIN MATMULADD")
    assert input.shape[1] == other.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], other.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = other.shape

    input = input.contiguous()
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    bias = bias.broadcast_to((M, N)).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
        matmuladd_kernel[grid](
            input,
            other,
            bias,
            out,
            1.0,
            1.0,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            GROUP_M=8,
            DTYPE_CODE=_dtype_code(input.dtype),
            # NOTE: do NOT pass num_stages here; the heuristics decorator
            # supplies it (see addmm.py for the duplicate-kwarg rationale).
        )
    return out
