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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.addmm import addmm
from flag_gems.runtime.backend._enflame.gcu400.ops.mm import mm
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Tile config table (GCU400 sweep conclusions; same measurement as benchmark:
# do_bench(warmup=1000, rep=100, median), interleaved A/B retests on 2026-11-11):
#   16-bit huge shape (M*N>=2M, m%128, n%256, k%512): (BM=128, BN=256, BK=512, w=2, s=2, gm=16)
#       - 2048^3 94.8us (0.755x), 4096^3 640us (0.866x), 1536^3 50.7us (0.697x);
#         w4 with the same tile 750us -> w2 -15%; gm16 another -1% over gm8, gm32 is slower.
#   16-bit medium shape (m%256, n%128, k%256): (BM=256, BN=128, BK=256, w=2, s=2, gm=8)
#       - 1024^3 21.1us (1.06x); BK512 21.9us, (64,128,128) small tile 35.5us.
#   16-bit small shape (m%64, n%128, k%128): (BM=64, BN=128, BK=128, w=2, s=3, gm=8)
#       - 384^3 8.87us (1.36x).
#   fp32 huge shape (M*N>=16M, m%128, n%256, k%128): (BM=128, BN=256, BK=128, w=4, s=2, gm=8)
#       - 4096^3 2198us (0.925x); (128,128,256,2,1) ties at 2208us, keep this tier to avoid regression.
#   fp32 medium shape (m%128, n%128, k%256): (BM=128, BN=128, BK=256, w=2, s=1, gm=8)
#       - 1024^3 53.7us (0.973x), 2048^3 303.0us (0.813x); w4/BK128 combinations degrade 2-9%.
#   fp32/16-bit small shape share (64,128,128,2,3): fp32 384^3 11.2us (1.59x).
#   fallback (a dimension is not divisible): (BM=128, BN=128, BK=64, w=4, s=3, gm=8)
#
# Sweep coverage (~60 config points x 4 shapes x 3 dtypes + interleaved A/B): BM/BN in {64..256},
# BK in {64..512}, w in {2,4}, s in {1..6}, gm in {1,2,4,8,16,32}; the 3-argument dot accumulation
# (acc = tl.dot(a, b, acc)) ties the 2-argument form per config (<0.1%); BN=512 and 256x256
# tile families degrade across the board (the latter has 24-way single occupancy). num_warps=2 helps (128,256)/(256,128)
# large tiles, consistent with the tensordot case (GCU400 large-tile GEMM w2 > w4).
#
# Two hard constraints (not tuning preferences; violating them gives wrong results or compile failure):
#   1. When K is not divisible by BLOCK_K, BLOCK_K must be <= 64. BK>=128 tail blocks make GCU400 report
#      "out of resource: dsm memory" (compile-time failure).
#   2. When M / N are not divisible (mask needed), BLOCK_N cannot be 256. Measured (BN=256, mask)
#      produces whole wrong rows (relative error ~1.1 on shapes like 192x1024, 1024x1000).
#      All BN=128 mask combinations are correct.
# DSM reported (compiled gcu.dsm_memory_size, hard limit 896KB, <=448KB keeps dual program/SP):
#   16-bit (128,256,512,2,2)=388KB, (256,128,256,2,2)=202KB, (64,128,128,2,3)=58KB;
#   fp32 (128,256,128,4,2)=210KB, (128,128,256,2,1)=266KB. All within the dual-occupancy range.
_CONFIG_16BIT_HUGE = (
    128,
    256,
    512,
    2,
    2,
    16,
)  # BM, BN, BK, num_warps, num_stages, GROUP_M
_CONFIG_16BIT_MID = (256, 128, 256, 2, 2, 8)
_CONFIG_SMALL_EVEN = (64, 128, 128, 2, 3, 8)
_CONFIG_32BIT_EVEN256 = (128, 256, 128, 4, 2, 8)
_CONFIG_32BIT_MID = (128, 128, 256, 2, 1, 8)
_CONFIG_SAFE = (128, 128, 64, 4, 3, 8)

# 16-bit size-tier boundary: the mid-shape configuration is faster at 1024² (1M), the huge-shape configuration is faster
# at 1536² (2.36M),
# so take 2M as the boundary. The fp32 boundary stays at 16M (4096²).
_BIG16_SHAPE_ELEMENTS = 1 << 21
_HUGE_SHAPE_ELEMENTS = 1 << 24  # 16777216 == 4096²

# ---- Hybrid dispatch (2026-11-16 perf round; rationale and precedents at the top of the file) ----
# The saturated segment (M*N*K >= 2^31, including 2048³/4096³ for all dtypes and cases like (1024,1024,2048)) and
# the long-K skinny segment (K >= 2^15, mirroring mm/addmm) are routed to this backend's FlagGems
# addmm / mm; at 1024³ (=2^30) 16-bit Triton still beats native, so the dispatch line is higher than mm/addmm's
# 2^29. For the 16-bit single-launch small band, see _compose_linear.
_COMPOSE_WORK_THRESHOLD = 1 << 31
_COMPOSE_LONG_K_THRESHOLD = 1 << 15
_COMPOSE_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_COMPOSE_SMALL_BAND_DTYPES = (torch.float16, torch.bfloat16)


def _select_config(M, N, K, dtype):
    """Picks (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M) from dtype, divisibility and size.

    Order: first check the large-tile safety preconditions (dimension divisibility), then size.
    The EVEN_* precondition of each tier maps one-to-one to mask elimination in the kernel:
    BM=256 needs M%256==0, BM=128 needs M%128==0, BM=64 needs M%64==0.
    """
    is_16bit = dtype in (torch.float16, torch.bfloat16)
    even_m64 = M % 64 == 0
    even_m128 = M % 128 == 0
    even_m256 = M % 256 == 0
    even_n128 = N % 128 == 0
    even_n256 = N % 256 == 0
    even_k128 = K % 128 == 0
    even_k256 = K % 256 == 0
    even_k512 = K % 512 == 0
    out_elements = M * N

    if is_16bit:
        if (
            even_m128
            and even_n256
            and even_k512
            and out_elements >= _BIG16_SHAPE_ELEMENTS
        ):
            return _CONFIG_16BIT_HUGE
        if even_m256 and even_n128 and even_k256:
            return _CONFIG_16BIT_MID
        if even_m64 and even_n128 and even_k128:
            return _CONFIG_SMALL_EVEN
        return _CONFIG_SAFE
    if even_m128 and even_n256 and even_k128 and out_elements >= _HUGE_SHAPE_ELEMENTS:
        return _CONFIG_32BIT_EVEN256
    if even_m128 and even_n128 and even_k256:
        return _CONFIG_32BIT_MID
    if even_m64 and even_n128 and even_k128:
        return _CONFIG_SMALL_EVEN
    return _CONFIG_SAFE


@libentry()
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    # stride is tl.constexpr: address computation folds at compile time; stride=0 (broadcast views,
    # e.g. expanded tensors) must also fold, otherwise GCU aborts (seen on mm).
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    stride_bn: tl.constexpr,
    # Bias is present or not
    BIAS: tl.constexpr,
    # Divisibility flags: computed on the host, deciding at compile time whether to emit masks
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Linear kernel: y = x @ W^T + b
    - input: (M, K) where M is batch size (flattened), K is in_features
    - weight: (N, K) where N is out_features
    - bias: (N,) optional
    - output: (M, N)
    """
    # GCU hardware limits grid.y <= 255, while N can be vocabulary-sized (e.g. 248320),
    # so the tile count far exceeds 255 and cannot be placed on grid.y. Therefore the 2D (pid_m, pid_n) tile
    # space is flattened into a single grid.x (same as mm/addmm); grid.y is always 1.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # GROUP_M swizzle: only changes the scheduling order of (pid_m, pid_n) to improve A/B block reuse,
    # without changing the computation semantics of any tile.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)

    weight_ptrs = weight_ptr + (
        offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # The main loop contains only "complete K tiles"; no K-direction mask ever appears;
    # all in-loop branches must be compile-time constants, since runtime scalar branches degrade tl.dot to scalar FMA.
    for _ in range(0, K // BLOCK_SIZE_K):
        if EVEN_M:
            a = tl.load(input_ptrs)
        else:
            a = tl.load(input_ptrs, mask=(offs_m < M)[:, None], other=0.0)
        if EVEN_N:
            b = tl.load(weight_ptrs)
        else:
            b = tl.load(weight_ptrs, mask=(offs_n < N)[None, :], other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    if not EVEN_K:
        k_remaining = K % BLOCK_SIZE_K
        a = tl.load(
            input_ptrs,
            mask=(offs_m < M)[:, None] & (offs_k < k_remaining)[None, :],
            other=0.0,
        )
        b = tl.load(
            weight_ptrs,
            mask=(offs_k < k_remaining)[:, None] & (offs_n < N)[None, :],
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)

    if BIAS:
        if EVEN_N:
            bias = tl.load(bias_ptr + offs_n)
        else:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        # bias is (BLOCK_N,) 1D, broadcast along the last dimension to (BLOCK_M, BLOCK_N)
        accumulator = accumulator + bias

    output_ptrs = output_ptr + (
        offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    output = accumulator.to(output_ptr.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(output_ptrs, output)
    else:
        tl.store(
            output_ptrs, output, mask=(offs_m < M)[:, None] & (offs_n < N)[None, :]
        )


def _compose_linear(input_flat, weight, bias, M, N, K):
    """Hybrid dispatch: compose the winnable segments onto this backend FlagGems addmm / mm
    (host convention (2) composition reuse; linalg_matmul composing bmm is the same precedent).
    Returns the (M, N) output, or None when the preconditions fail, letting the caller fall back.

    Two winnable segments (rationale in the file header):
    - saturated M*N*K >= 2^31 or long-K K >= 2^15: add/mm internally delegates to the captured
      native kernel (bit-exact; 1024^3 and below stay in Triton).
    - 16-bit single-shot small band M%128==0 and N%128==0 and K<=512 and M*N<2M: lands on
      add/mm's Triton arm (128,128,512,w2,s2) as a single K-shot (384^3 -10%).
    """
    if input_flat.dtype not in _COMPOSE_FLOAT_DTYPES:
        return None
    if weight.dtype != input_flat.dtype:
        return None
    if bias is not None and bias.dtype != input_flat.dtype:
        return None
    if input_flat.device.type != runtime.device.name:
        return None
    saturated = M * N * K >= _COMPOSE_WORK_THRESHOLD or K >= _COMPOSE_LONG_K_THRESHOLD
    small_band = (
        not saturated
        and input_flat.dtype in _COMPOSE_SMALL_BAND_DTYPES
        and M % 128 == 0
        and N % 128 == 0
        and K <= 512
        and M * N < _BIG16_SHAPE_ELEMENTS
    )
    if not (saturated or small_band):
        return None
    weight_t = weight.t()
    if bias is None:
        return mm(input_flat, weight_t)
    return addmm(bias, input_flat, weight_t)


def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Args:
        input: Input tensor of shape (*, in_features) where * means any number of
               additional dimensions, including none.
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (*, out_features)
    """
    logger.debug("GEMS_ENFLAME LINEAR")

    single_1d = input.dim() == 1
    if single_1d:
        input = input.unsqueeze(0)

    # Flatten batch dimensions: (*, in_features) -> (batch, in_features)
    batch_dims = input.shape[:-1]
    M = 1
    for dim in batch_dims:
        M *= dim
    K = input.shape[-1]  # in_features
    N = weight.shape[0]  # out_features

    input_flat = input.reshape(M, K)
    # An explicit copy is needed only when both dimensions are non-contiguous; ordinary row-major inputs are zero-copy
    if input_flat.stride(0) > 1 and input_flat.stride(1) > 1:
        input_flat = input_flat.contiguous()

    # Hybrid dispatch (2026-11-16): the saturated segment / long-K segment / 16-bit single-launch small band are routed
    # to backend
    # add/mm; everything else (1024³ and below, odd shapes, small fp32 tiers) falls back to this file's Triton kernel.
    if M > 0 and N > 0 and K > 0:
        composed = _compose_linear(input_flat, weight, bias, M, N, K)
        if composed is not None:
            output = composed.view(*batch_dims, N)
            return output.squeeze(0) if single_1d else output

    weight_mat = weight
    if weight_mat.stride(0) > 1 and weight_mat.stride(1) > 1:
        weight_mat = weight_mat.contiguous()

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    if M == 0 or N == 0:
        output = output.view(*batch_dims, N)
        return output.squeeze(0) if single_1d else output

    block_m, block_n, block_k, num_warps, num_stages, group_m = _select_config(
        M, N, K, input.dtype
    )

    # Flatten the 2D (M, N) tile space into a 1D grid; grid.y is always 1
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    with torch_device_fn.device(input.device):
        linear_kernel[grid](
            input_flat,
            weight_mat,
            bias if bias is not None else weight_mat,  # Pass dummy ptr if no bias
            output,
            M,
            N,
            K,
            input_flat.stride(0),
            input_flat.stride(1),
            weight_mat.stride(0),
            weight_mat.stride(1),
            output.stride(0),
            output.stride(1),
            bias.stride(0) if bias is not None else 0,
            BIAS=bias is not None,
            EVEN_M=(M % block_m == 0),
            EVEN_N=(N % block_n == 0),
            EVEN_K=(K % block_k == 0),
            GROUP_M=group_m,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    # Reshape output: (M, N) -> (*, N)
    output = output.view(*batch_dims, N)

    # If original input was 1D, squeeze the batch dim
    if single_1d:
        output = output.squeeze(0)

    return output
