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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 fixed grid: measured optimum across grouped-mm benchmark shapes.
# 4 DIEs x 6 SIPs; 48 concurrent programs beat 4/8/24 in every sweep.
GCU_NUM_GRID = 48


@libentry()
@triton.jit
def grouped_mm_kernel(
    A,
    B,
    C,
    offs,
    num_groups,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    GRID_DIM_X: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    group_end = 0

    for group_idx in tl.range(num_groups):
        group_end = tl.load(offs + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                # Direct row-major tile decomposition. The GROUP_M swizzle of
                # the generic kernel was measured to be a small net loss on
                # GCU400 (extra div/mod chain, no L2 benefit at grid=48).
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = group_start + tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N
                offs_bk = group_idx * K

                a_block_ptr = tl.make_block_ptr(
                    base=A,
                    shape=(M, K),
                    strides=(stride_am, stride_ak),
                    offsets=(offs_am, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )

                b_block_ptr = tl.make_block_ptr(
                    base=B,
                    shape=(num_groups * K, N),
                    strides=(stride_bk, stride_bn),
                    offsets=(offs_bk, offs_bn),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(1, 0),
                )

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    # K/N 可整除时去掉对应维度的边界检查，分支由 constexpr
                    # 在编译期裁剪；未启用时保持与原始实现完全一致的访存语句。
                    if EVEN_K and EVEN_N:
                        a = tl.load(
                            a_block_ptr, boundary_check=(0,), padding_option="zero"
                        )
                        b = tl.load(b_block_ptr)
                    elif EVEN_K:
                        a = tl.load(
                            a_block_ptr, boundary_check=(0,), padding_option="zero"
                        )
                        b = tl.load(
                            b_block_ptr, boundary_check=(1,), padding_option="zero"
                        )
                    elif EVEN_N:
                        a = tl.load(
                            a_block_ptr, boundary_check=(0, 1), padding_option="zero"
                        )
                        b = tl.load(
                            b_block_ptr, boundary_check=(0,), padding_option="zero"
                        )
                    else:
                        a = tl.load(a_block_ptr, boundary_check=(0, 1))
                        b = tl.load(b_block_ptr, boundary_check=(0, 1))
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

                c = accumulator.to(C.dtype.element_ty)

                if offs_am + BLOCK_M <= group_end:
                    c_block_ptr = tl.make_block_ptr(
                        base=C,
                        shape=(M, N),
                        strides=(stride_cm, stride_cn),
                        offsets=(offs_am, offs_bn),
                        block_shape=(BLOCK_M, BLOCK_N),
                        order=(1, 0),
                    )
                    # EVEN_N 时整块都在 C 的合法范围内，可省掉边界检查；
                    # 否则保持原始实现的双维检查。
                    if EVEN_N:
                        tl.store(c_block_ptr, c)
                    else:
                        tl.store(c_block_ptr, c, boundary_check=(0, 1))
                else:
                    offs_cm = offs_am + tl.arange(0, BLOCK_M)
                    offs_cn = offs_bn + tl.arange(0, BLOCK_N)
                    c_ptrs = (
                        C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                    )
                    c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, c, mask=c_mask)

                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


def _select_block_k(K):
    # Measured: BK=256 wins clearly when K is a multiple of 256 (fewer K-loop
    # iterations, same tile footprint); K=128 shapes must keep BK=128.
    if K >= 256 and K % 256 == 0:
        return 256
    if K >= 128:
        return 128
    if K >= 64:
        return 64
    return 32


def _select_block_n(N):
    if N <= 16:
        return 16
    if N <= 32:
        return 32
    if N <= 64:
        return 64
    return 256


@libentry()
@triton.jit
def grouped_mm_generic_kernel(
    A,
    B,
    C,
    offs,
    num_groups,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    GRID_DIM_X: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # Pointer-arithmetic fallback for shapes whose strides are not a
    # multiple of 16 bytes: the GCU backend rejects misaligned block_ptr
    # descriptors at launch time, so those shapes cannot use the block_ptr
    # fast kernel above. Math is identical; loads/stores are masked.
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    group_end = 0

    for group_idx in tl.range(num_groups):
        group_end = tl.load(offs + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_am = group_start + tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

                a_ptrs = (
                    A
                    + offs_am[:, None] * stride_am
                    + tl.arange(0, BLOCK_K)[None, :] * stride_ak
                )
                b_ptrs = (
                    B
                    + group_idx * K * stride_bk
                    + tl.arange(0, BLOCK_K)[:, None] * stride_bk
                    + offs_bn[None, :] * stride_bn
                )
                c_ptrs = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
                    a_mask = (offs_am[:, None] < group_end) & (offs_k[None, :] < K)
                    b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk

                c_mask = (offs_am[:, None] < group_end) & (offs_bn[None, :] < N)
                tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=c_mask)

                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


def _block_ptr_aligned(A, B, C) -> bool:
    # GCU block_ptr requires every stride to be a multiple of 16 bytes
    # (8 elements in bf16/fp16). Innermost strides must be unit as well.
    elem_size = A.element_size()
    align = max(1, 16 // elem_size)
    if A.stride(1) != 1 or B.stride(2) != 1 or C.stride(1) != 1:
        return False
    for s in (A.stride(0), B.stride(1), C.stride(0)):
        if s % align != 0:
            return False
    return True


def group_mm(A: torch.Tensor, B: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME GROUP_MM")
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape

    num_groups, BK, N = B.shape
    strideBK, strideBN = B.stride(1), B.stride(2)

    assert num_groups == offs.numel()

    BLOCK_M = 128
    BLOCK_N = _select_block_n(N)
    BLOCK_K = _select_block_k(K)

    # Fixed grid: cap concurrent programs at GCU_NUM_GRID and let the
    # in-kernel stride loop cover the whole tile space. The upper bound uses
    # total M, which is always >= the per-group sum of cdiv tiles, so every
    # real tile gets at least one program.
    total_tiles_upper = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(GCU_NUM_GRID, total_tiles_upper),)

    # 短 K 分组（G 多、N 大）在 num_warps=2 下更快，长 K 保持 1。
    num_warps = 2 if K <= 256 else 1
    # K / N 能整除时省掉边界 mask；N 很大的 shape 无 mask 路径会落入
    # 低算力稳态，因此只对 N<=2048 启用。
    enable_even = N <= 2048
    even_k = enable_even and K % BLOCK_K == 0
    even_n = enable_even and N % BLOCK_N == 0
    C = A.new_empty(M, N)

    launch_args = (
        A,
        B,
        C,
        offs,
        num_groups,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        strideBK,
        strideBN,
        C.stride(0),
        C.stride(1),
    )
    launch_kwargs = {
        "GRID_DIM_X": grid[0],
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "EVEN_K": even_k,
        "EVEN_N": even_n,
        "num_warps": num_warps,
        "num_stages": 3,
    }

    if _block_ptr_aligned(A, B, C):
        grouped_mm_kernel[grid](*launch_args, **launch_kwargs)
    else:
        grouped_mm_generic_kernel[grid](*launch_args, **launch_kwargs)

    return C
