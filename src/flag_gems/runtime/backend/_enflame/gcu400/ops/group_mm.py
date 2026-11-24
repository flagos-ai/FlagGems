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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    USE_WS: tl.constexpr,
):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    # Host-side constant: N / BLOCK_N tile count. Keeping it as a constexpr turns
    # the per-tile div/mod below into compile-time shift/mask (or a constant
    # multiply) instead of a runtime integer division chain.
    num_n_tiles: tl.constexpr = NUM_N_TILES
    last_problem_end = 0
    group_start = 0
    group_end = 0

    for group_idx in tl.range(num_groups, warp_specialize=USE_WS):
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
                    # When K/N divide evenly, drop that dimension's bounds check; the branch is trimmed
                    # at compile time by constexpr; when disabled, memory statements stay exactly as in the original.
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

                offs_cm = offs_am + tl.arange(0, BLOCK_M)
                offs_cn = offs_bn + tl.arange(0, BLOCK_N)
                c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)

                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


@libentry()
@triton.jit
def grouped_mm_rowtile_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    USE_WS: tl.constexpr,
):
    """Short-K shapes: one work item = one (group, M block) that reuses its A block along N.

    For short K (K <= BLOCK_K) an A row block is only BLOCK_M x K, so it can be loaded once and
    kept resident while each N tile only moves a B tile, does one dot product and writes C once.
    Compared with the generic kernel of one work item per (M,N) tile, A traffic drops
    proportionally to the N tile count, and each written C row block is contiguous along N for
    higher write bandwidth. The layout stops applying for longer K (the A block no longer fits),
    so only K <= 256 shapes are handled.

    EVEN_K / EVEN_N (K and N divisible by the block width) also drop the mask on B loads and C
    stores; the row direction keeps the row mask because group boundaries vary (hot loop free of
    predicates, measured -4.1% on G=64/N=2048/K=128 on 2026-11-16, -17% with BM=256/BN=128/nw4).
    """
    total_grid = tl.num_programs(axis=0)
    work_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    last_work_end = 0
    group_start = 0
    group_end = 0
    # Accumulate the work-item range group by group; each program advances by the grid stride within its own range
    for group_idx in tl.range(num_groups, warp_specialize=USE_WS):
        group_end = tl.load(offs + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        current_work_end = last_work_end + num_m_tiles
        if work_idx >= last_work_end and work_idx < current_work_end:
            loop_count = (current_work_end - work_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                m_tile_idx = work_idx - last_work_end
                offs_am = group_start + m_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                row_mask = offs_am < group_end
                # The A row block is moved only once and reused throughout the inner N loop
                if EVEN_K:
                    a = tl.load(
                        A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                else:
                    a = tl.load(
                        A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=row_mask[:, None] & (offs_k[None, :] < K),
                        other=0.0,
                    )
                b_base = B + group_idx * K * stride_bk
                c_base = C + offs_am[:, None] * stride_cm
                for n_tile_idx in tl.range(num_n_tiles):
                    offs_bn = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                    if EVEN_K and EVEN_N:
                        b = tl.load(
                            b_base
                            + offs_k[:, None] * stride_bk
                            + offs_bn[None, :] * stride_bn,
                        )
                    elif EVEN_K:
                        b = tl.load(
                            b_base
                            + offs_k[:, None] * stride_bk
                            + offs_bn[None, :] * stride_bn,
                            mask=offs_bn[None, :] < N,
                            other=0.0,
                        )
                    elif EVEN_N:
                        b = tl.load(
                            b_base
                            + offs_k[:, None] * stride_bk
                            + offs_bn[None, :] * stride_bn,
                            mask=offs_k[:, None] < K,
                            other=0.0,
                        )
                    else:
                        b = tl.load(
                            b_base
                            + offs_k[:, None] * stride_bk
                            + offs_bn[None, :] * stride_bn,
                            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
                            other=0.0,
                        )
                    accumulator = tl.dot(a, b, allow_tf32=False)
                    c_ptrs = c_base + offs_bn[None, :] * stride_cn
                    if EVEN_N:
                        tl.store(
                            c_ptrs,
                            accumulator.to(C.dtype.element_ty),
                            mask=row_mask[:, None],
                        )
                    else:
                        tl.store(
                            c_ptrs,
                            accumulator.to(C.dtype.element_ty),
                            mask=row_mask[:, None] & (offs_bn[None, :] < N),
                        )
                work_idx += total_grid
        last_work_end = current_work_end
        group_start = group_end


def _select_gemm_config(N, K):
    """Pick the kernel and its tile/warp config from the problem shape.

    Returns (kernel, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, use_ws) where
    `kernel` is "rowtile" or "persistent". The rowtile kernel is not selected
    until its warp-specialized path is fixed (see the K <= 256 branch).

    Measured on GCU400 (2026-11-19, per-group M=8192, M_total=131072):

      rowtile, BM=512/BN=128/nw=4 + WS:   N=2048 K=128  1.24 ms (221 TFLOPS)
      persistent + WS, BM=256/BN=256/BK=256/nw=4:
        N=512  K=2048  0.98 ms (279 TFLOPS)
        N=2560 K=2048  4.77 ms (288 TFLOPS)
      persistent without WS, BM=128/BN=256/BK=256/nw=2:
        N=512  K=2048  1.37 ms (200 TFLOPS)
        N=2560 K=2048  7.09 ms (194 TFLOPS)

    WS wins on the deep-K shapes because it splits the load pipeline (producer)
    from the MMA (consumer) instead of interleaving both in one warp group, and
    the 256-wide BM keeps the A tile resident across the whole N loop. On the
    short-K rowtile kernel WS only pays off with BM=512 (dot 512x128 is big
    enough to hide the channel round-trip); at BM=256 it loses ~11%.

    num_warps is capped at 4 because the WS pass only supports num_warps <= 4.
    """
    block_n = _select_block_n(N)
    if K <= 256:
        # Short K: rowtile is the shape match, but its warp specialization is
        # not usable yet. The WS barrier arrive is a call inside the n-tile
        # loop, and the backend spills the whole live OACC accumulator across
        # it (128 groups, 64 KiB, to the L3 stack): 5.02 ms vs 1.24 ms with the
        # barrier inlined on (64, 2048, 128). The persistent WS tile below
        # measures 1.24 ms on the same shape.
        #
        # Switch back to rowtile once the llvm fix (inline the barrier so the
        # accumulator is not live across a call) is in:
        #   return ("rowtile", 512, 128 if N % 128 == 0 else block_n,
        #           max(16, triton.next_power_of_2(K)), 4, True)
        return ("persistent", 256, block_n, _select_block_k(K), 4, True)
    if K >= 1024:
        return ("persistent", 256, block_n, _select_block_k(K), 4, True)
    # 256 < K < 1024: the K loop is too short for WS to pay for the partition.
    return ("persistent", 128, block_n, _select_block_k(K), 2, False)


def _select_block_k(K):
    # if K >= 512 and K % 512 == 0:
    #     return 512
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
    # N in (64, 128]: a 128-wide N tile covers the whole row exactly, so the dot
    # does no padding work and EVEN_N removes both the B load mask and the C
    # store mask. Measured on G=64/N=128/K=2048 bf16 (2026-11-21 probe): 3.83 ms
    # with BN=256 (one padded tile, masked store) -> 2.72 ms with BN=128.
    if N <= 128:
        return 128
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # Pointer-arithmetic fallback for shapes whose strides are not a
    # multiple of 16 bytes: the GCU backend rejects misaligned block_ptr
    # descriptors at launch time, so those shapes cannot use the block_ptr
    # fast kernel above. Math is identical; loads/stores are masked.
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles: tl.constexpr = NUM_N_TILES
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

    KERNEL, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, USE_WS = _select_gemm_config(N, K)
    C = A.new_empty(M, N)
    use_block_ptr = _block_ptr_aligned(A, B, C)

    NUM_STAGES = 2
    # Persistent grid: cap concurrent programs at GCU_NUM_GRID and let the
    # in-kernel stride loop cover the whole tile space. The upper bound uses
    # total M, which is always >= the per-group sum of cdiv tiles, so every
    # real tile gets at least one program.
    total_tiles_upper = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # WS splits each program into a producer and a consumer task, so one program
    # occupies twice the execution resources; halve the grid to keep the same
    # occupancy. Measured on case0: 24 -> 0.978 ms, 48 -> 1.003 ms.
    grid_cap = GCU_NUM_GRID // 2 if USE_WS else GCU_NUM_GRID
    grid = (min(grid_cap, total_tiles_upper),)

    # Skip boundary masks when K / N is divisible; on 2026-11-16 the EVEN path on both N=512/2560 shapes
    # was on par with or slightly better than masked (-0.5~-1.7%), and nw1+EVEN, after gap-tightened 60-round CPU-side
    # comparison, showed zero deviation (S1's EVEN path has been in production since 2026-11-03), so gating by N size is
    # no longer needed; only the divisibility requirement is retained.
    even_k = K % BLOCK_K == 0
    even_n = N % BLOCK_N == 0

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
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "NUM_N_TILES": triton.cdiv(N, BLOCK_N),
        "EVEN_K": even_k,
        "EVEN_N": even_n,
        "USE_WS": USE_WS,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
    }

    # Short K: switch to the row-tile kernel (one work item covers the N tiles of
    # a whole row block, so the A row block is moved only once).
    if KERNEL == "rowtile":
        rowtile_upper = triton.cdiv(M, BLOCK_M)
        rowtile_grid = (min(grid_cap, rowtile_upper),)
        grouped_mm_rowtile_kernel[rowtile_grid](
            *launch_args,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            EVEN_K=even_k,
            EVEN_N=even_n,
            USE_WS=USE_WS,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return C

    if use_block_ptr:
        grouped_mm_kernel[grid](*launch_args, **launch_kwargs)
    else:
        # The pointer-arithmetic fallback has no USE_WS parameter: it predates
        # warp specialization and stays on the non-WS path.
        generic_kwargs = {k: v for k, v in launch_kwargs.items() if k != "USE_WS"}
        grouped_mm_generic_kernel[grid](*launch_args, **generic_kwargs)

    return C
