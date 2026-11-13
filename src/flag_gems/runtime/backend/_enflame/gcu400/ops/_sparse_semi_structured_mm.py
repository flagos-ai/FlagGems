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


@libentry()
@triton.jit
def sparse_semi_structured_mm_kernel(
    A,
    Meta,
    B,
    C,
    M,
    N,
    K4,
    stride_am,
    stride_ak,
    stride_mm,
    stride_mn,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MERGED_LOAD: tl.constexpr,
):
    # 逐 k 处理 2:4 稀疏组：每步只构造 (BLOCK_M, BLOCK_N) 的二维外积，
    # 避免公共实现 (BM, BK, BN) 三维中间张量打爆 GCU400 DSM。
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_am < M
    mask_n = offs_bn < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group in range(K4):
        select_first = tl.load(
            Meta + offs_am * stride_mm + group * stride_mn, mask=mask_m, other=0
        ).to(tl.float32)
        if MERGED_LOAD:
            # 一次 2D load 取回组内 4 个 A 值，减少独立访存指令
            pos = tl.arange(0, 4)
            a_group = tl.load(
                A
                + offs_am[:, None] * stride_am
                + (group * 4 + pos[None, :]) * stride_ak,
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            b_group = tl.load(
                B
                + (group * 4 + pos[:, None]) * stride_bk
                + offs_bn[None, :] * stride_bn,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            keep = tl.where(
                select_first[:, None] > 0.5, pos[None, :] < 2, pos[None, :] >= 2
            )
            a_group = tl.where(keep, a_group, 0.0)
            accumulator += tl.dot(a_group, b_group)
        else:
            a0 = tl.load(
                A + offs_am * stride_am + (group * 4 + 0) * stride_ak,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            a1 = tl.load(
                A + offs_am * stride_am + (group * 4 + 1) * stride_ak,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            a2 = tl.load(
                A + offs_am * stride_am + (group * 4 + 2) * stride_ak,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            a3 = tl.load(
                A + offs_am * stride_am + (group * 4 + 3) * stride_ak,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            b0 = tl.load(
                B + (group * 4 + 0) * stride_bk + offs_bn * stride_bn,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            b1 = tl.load(
                B + (group * 4 + 1) * stride_bk + offs_bn * stride_bn,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            b2 = tl.load(
                B + (group * 4 + 2) * stride_bk + offs_bn * stride_bn,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            b3 = tl.load(
                B + (group * 4 + 3) * stride_bk + offs_bn * stride_bn,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            first = select_first[:, None]
            pair_first = a0[:, None] * b0[None, :] + a1[:, None] * b1[None, :]
            pair_second = a2[:, None] * b2[None, :] + a3[:, None] * b3[None, :]
            accumulator += pair_first * first + pair_second * (1.0 - first)

    tl.store(
        C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _sparse_semi_structured_mm(mat1, mat1_meta, mat2, *, out_dtype=None):
    """2:4 半结构化稀疏矩阵乘：按 meta 选取每组 4 个元素中的 2 个与 mat2 对应行累加。"""
    logger.debug("GEMS_ENFLAME SPARSE_SEMI_STRUCTURED_MM")
    M, K4, N = mat1.shape[0], mat1_meta.shape[1], mat2.shape[1]
    assert mat1.shape == (M, 4 * K4), "mat1 shape mismatch"
    assert mat2.shape == (4 * K4, N), "mat2 shape mismatch"
    assert mat1_meta.shape == (M, K4), "mat1_meta shape mismatch"
    output_dtype = out_dtype if out_dtype is not None else mat1.dtype
    out = torch.empty((M, N), device=mat1.device, dtype=output_dtype)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    with torch_device_fn.device(mat1.device):
        sparse_semi_structured_mm_kernel[grid](
            mat1,
            mat1_meta,
            mat2,
            out,
            M,
            N,
            K4,
            mat1.stride(0),
            mat1.stride(1),
            mat1_meta.stride(0),
            mat1_meta.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M,
            BLOCK_N,
            True,
            num_warps=4,
        )
    return out
