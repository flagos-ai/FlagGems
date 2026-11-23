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

                if offs_am + BLOCK_M <= group_end:
                    c_block_ptr = tl.make_block_ptr(
                        base=C,
                        shape=(M, N),
                        strides=(stride_cm, stride_cn),
                        offsets=(offs_am, offs_bn),
                        block_shape=(BLOCK_M, BLOCK_N),
                        order=(1, 0),
                    )
                    # With EVEN_N the whole block is within C's valid range, so the bounds check can be dropped;
                    # otherwise keep the original implementation's dual-dimension checks.
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
    for group_idx in tl.range(num_groups):
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

    BLOCK_M = 128
    BLOCK_N = _select_block_n(N)
    # Mid-width N (256..512): the square-ish 256x128 tile with a 2-deep pipeline
    # reuses the wide A row block across more N tiles than the 128x256 tile and
    # stays inside the DSM budget, while keeping the K block at 256.
    # Measured on G=16/K=2048 bf16 (2026-11-21 probe, same-draw A/B comparison):
    # N=512 2.226 -> 2.146 ms (+3.7%), N=2560 pairwise parity (10.21 vs 10.27 ms),
    # so the wide-N shapes keep 128x256.
    mid_width_n = 256 <= N <= 512
    if mid_width_n:
        BLOCK_M, BLOCK_N = 256, 128
    C = A.new_empty(M, N)
    use_block_ptr = _block_ptr_aligned(A, B, C)

    # NOTE: a free-running one-tile-per-program kernel (grouped_mm_flat_kernel)
    # used to serve deep-K shapes here, but its K-loop dot is nondeterministic
    # on GCU400: identical inputs produce bitwise-different outputs across
    # runs (up to ~2x the accuracy tolerance with BLOCK_K=512, still ~30% of
    # runs deviating below tolerance with BLOCK_K=256). The persistent kernel
    # below is deterministic across 60+ repeated runs per shape (BK<=256), so
    # every shape now goes through it. BLOCK_K=512 was re-probed in the
    # persistent grid on 2026-11-16 (-16% latency) but is bitwise-unstable
    # there as well (1/120 runs, whole-K-chunk-sized deviations; DSM sits
    # exactly on the 458752B double-program line) -- do not raise BLOCK_K above
    # 256.
    BLOCK_K = _select_block_k(K)
    # num_warps must stay 1 on this K-loop dot kernel: nw=2 measured -9~-11%
    # on deep-K core shapes (2026-11-16 probe) BUT is bitwise-unstable on the
    # production path (~1/480 launches, above-tolerance whole-chunk errors,
    # CPU-side verified -- same disease family as the removed flat kernel and
    # the re-probed persistent BLOCK_K=512). BLOCK_K must also stay <=256
    # (BK512 persistent: -16% latency but bitwise-unstable, DSM exactly on
    # the 458752B double-program line). Do not raise either without a
    # 480+ round CPU-compared determinism gate.
    NUM_WARPS = 1
    # The 256x128 tile already fills two pipeline stages within DSM
    # ((256*256 + 256*128) * 2B * 3 stages = 576KB > 512KB budget), so its
    # pipeline depth is capped at 2.
    NUM_STAGES = 2 if mid_width_n else 3
    # Persistent grid: cap concurrent programs at GCU_NUM_GRID and let the
    # in-kernel stride loop cover the whole tile space. The upper bound uses
    # total M, which is always >= the per-group sum of cdiv tiles, so every
    # real tile gets at least one program.
    total_tiles_upper = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(GCU_NUM_GRID, total_tiles_upper),)

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
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
    }

    # Short K: switch to the row-tile kernel (one work item covers the N tiles of a whole row block, A is moved only
    # once).
    # When K <= 256 the A row block is <= 256x256 and can stay resident; the BM=256 / BN=128 / nw=4 combination
    # measured on 2026-11-16 reduced G=64/N=2048/K=128 from 3.90 to 3.24 us/k-row (-17%):
    # BM 256 halves the number of B row covers (halves b_read), and BN=128 with EVEN_K/EVEN_N makes
    # B loads and C writes mask-free. It is independent of block_ptr alignment and works with any stride.
    if K <= 256:
        rowtile_block_k = max(16, triton.next_power_of_2(K))
        rowtile_block_m = 256
        # Use 128 when N is divisible by 128 (the empirically measured sweet spot for EVEN_N mask-free), otherwise fall
        # back to
        # generic BN selection (the masked path is correct for any N).
        rowtile_block_n = 128 if N % 128 == 0 else _select_block_n(N)
        rowtile_even_k = K % rowtile_block_k == 0
        rowtile_even_n = N % rowtile_block_n == 0
        # Upper bound on the total number of work items = cdiv(M, BM) + num_groups, no need to read offs back to the CPU
        rowtile_upper = triton.cdiv(M, rowtile_block_m) + num_groups
        rowtile_grid = (min(GCU_NUM_GRID, rowtile_upper),)
        grouped_mm_rowtile_kernel[rowtile_grid](
            *launch_args,
            BLOCK_M=rowtile_block_m,
            BLOCK_N=rowtile_block_n,
            BLOCK_K=rowtile_block_k,
            EVEN_K=rowtile_even_k,
            EVEN_N=rowtile_even_n,
            num_warps=4,
            num_stages=NUM_STAGES,
        )
        return C

    if use_block_ptr:
        grouped_mm_kernel[grid](*launch_args, **launch_kwargs)
    else:
        grouped_mm_generic_kernel[grid](*launch_args, **launch_kwargs)

    return C
