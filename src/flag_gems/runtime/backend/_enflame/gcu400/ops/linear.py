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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# tile 配置表（GCU400 扫描实测结论；测量口径与 benchmark 一致：
# do_bench(warmup=1000, rep=100, median)，2026-11-11 交错 A/B 复测）：
#   16-bit 巨形状 (M*N>=2M, m%128, n%256, k%512): (BM=128, BN=256, BK=512, w=2, s=2, gm=16)
#       —— 2048³ 94.8us (0.755x)、4096³ 640us (0.866x)、1536³ 50.7us (0.697x)；
#          w4 同 tile 750us → w2 -15%；gm16 比 gm8 再 -1%，gm32 反慢。
#   16-bit 中形状 (m%256, n%128, k%256): (BM=256, BN=128, BK=256, w=2, s=2, gm=8)
#       —— 1024³ 21.1us (1.06x)；BK512 版 21.9us，(64,128,128) 小 tile 35.5us。
#   16-bit 小形状 (m%64, n%128, k%128): (BM=64, BN=128, BK=128, w=2, s=3, gm=8)
#       —— 384³ 8.87us (1.36x)。
#   fp32 巨形状 (M*N>=16M, m%128, n%256, k%128): (BM=128, BN=256, BK=128, w=4, s=2, gm=8)
#       —— 4096³ 2198us (0.925x)；(128,128,256,2,1) 同档 2208us 打平，保留本档防回退。
#   fp32 中形状 (m%128, n%128, k%256): (BM=128, BN=128, BK=256, w=2, s=1, gm=8)
#       —— 1024³ 53.7us (0.973x)、2048³ 303.0us (0.813x)；w4/BK128 组合均劣化 2-9%。
#   fp32/16bit 小形状同用 (64,128,128,2,3)：fp32 384³ 11.2us (1.59x)。
#   兜底（存在非整除维度）: (BM=128, BN=128, BK=64, w=4, s=3, gm=8)
#
# 扫描覆盖（~60 配置点 × 4 形状 × 3 dtype + 交错 A/B）：BM/BN∈{64..256}、
# BK∈{64..512}、w∈{2,4}、s∈{1..6}、gm∈{1,2,4,8,16,32}；dot 3 参累加形式
# （acc = tl.dot(a, b, acc)）与 2 参形式逐配置打平（<0.1%），BN=512 与 256×256
# tile 族全面劣化（后者 24-way 单占用）。num_warps=2 对 (128,256)/(256,128)
# 大 tile 的优势与 tensordot 案例结论一致（GCU400 大 tile GEMM w2 > w4）。
#
# 两条硬约束（不是调优偏好，踩错就是错误结果或编译失败）：
#   1. K 不能被 BLOCK_K 整除时，BLOCK_K 必须 ≤ 64。BK≥128 的尾块会让 GCU400 报
#      "out of resource: dsm memory"（编译期失败）。
#   2. M / N 存在非整除（有 mask）时，BLOCK_N 不能取 256。实测 (BN=256, mask) 组合
#      会产出整行错误结果（192x1024、1024x1000 等 shape 上相对误差 ~1.1）。
#      BN=128 的所有 mask 组合均正确。
# DSM 上报（编译产物 gcu.dsm_memory_size，硬限 896KB，≤448KB 保双 program/SP）：
#   16-bit (128,256,512,2,2)=388KB、(256,128,256,2,2)=202KB、(64,128,128,2,3)=58KB；
#   fp32 (128,256,128,4,2)=210KB、(128,128,256,2,1)=266KB。全部处于双占用区间。
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

# 16-bit 大小档分界：1024² (1M) 上中形状配置更快，1536² (2.36M) 上巨形状配置更快，
# 取 2M 为分界。fp32 分界沿用 16M（4096²）。
_BIG16_SHAPE_ELEMENTS = 1 << 21
_HUGE_SHAPE_ELEMENTS = 1 << 24  # 16777216 == 4096²


def _select_config(M, N, K, dtype):
    """按 dtype 与整除性 / 规模确定 (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M)。

    决策顺序：先看是否满足「大 tile 的安全前提」（维度整除），再看规模。
    各档位的 EVEN_* 前提与 kernel 侧的 mask 消除一一对应：
    BM=256 档要求 M%256==0，BM=128 档要求 M%128==0，BM=64 档要求 M%64==0。
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
    # stride 取 tl.constexpr：地址计算在编译期折叠；同时 stride=0（broadcast 视图，
    # 如 expand 出来的张量）必须能被折掉，否则实测会在 GCU 上 abort（mm 上踩过）
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    stride_bn: tl.constexpr,
    # Bias is present or not
    BIAS: tl.constexpr,
    # 整除标记：host 侧算好传入，编译期决定是否生成 mask
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
    # GCU 硬件限制 grid.y <= 255，而 N 可为 vocab 级大尺寸（如 248320），
    # tile 数远超 255，不能放在 grid.y。因此将 (pid_m, pid_n) 二维 tile
    # 空间平铺编码进单维 grid.x（与 mm/addmm 相同），grid.y 恒为 1。
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # GROUP_M swizzle：仅改变 (pid_m, pid_n) 的调度顺序以提升 A/B 块复用，
    # 不改变每个 tile 的计算语义。
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

    # 主循环只包含「完整 K tile」，K 方向的 mask 一律不出现；
    # 循环内的分支必须全部是编译期常量，运行时标量分支会让 tl.dot 退化为标量 FMA。
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
        # bias 为 (BLOCK_N,) 一维，按最后一维广播到 (BLOCK_M, BLOCK_N)
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
    # 只有两个维度都不连续时才需要一次显式拷贝；常规行主序输入零拷贝
    if input_flat.stride(0) > 1 and input_flat.stride(1) > 1:
        input_flat = input_flat.contiguous()
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

    # 把 (M, N) 的二维 tile 空间展平为单维 grid，grid.y 恒为 1
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
