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
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import has_triton_tle

HAS_TLE_GROUPED_FP8 = has_triton_tle(3, 6, 0)
if HAS_TLE_GROUPED_FP8:
    import triton.experimental.tle.language as tle
else:
    tle = None

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def grouped_fp8_gemv(
    A,
    B,
    C,
    SA,
    SB,
    Groups,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.load(Groups + row)
    kk = tl.arange(0, BK)
    scale_k = tl.arange(0, BK // 128)
    acc = tl.zeros((BN,), tl.float32)
    for base in range(tl.cdiv(K, BK)):
        offsets = base * BK + kk
        a = tl.load(A + row.to(tl.int64) * K + offsets, offsets < K, 0.0).to(tl.float32)
        b = tl.load(
            B + group.to(tl.int64) * N * K + cols[:, None] * K + offsets[None, :],
            (cols[:, None] < N) & (offsets[None, :] < K),
            0.0,
        ).to(tl.float32)
        partial = tl.sum((b * a[None, :]).reshape((BN, BK // 128, 128)), 2)
        sk = base * (BK // 128) + scale_k
        sa = tl.load(SA + row * SAM + sk * SAK, sk < K // 128, 0.0)
        sb = tl.load(
            SB + group * SBE + (cols[:, None] // 128) * SBN + sk[None, :] * SBK,
            (cols[:, None] < N) & (sk[None, :] < K // 128),
            0.0,
        )
        acc += tl.sum(partial * sa[None, :] * sb, 1)
    tl.store(C + row * N + cols, acc, cols < N)


@libentry()
@triton.jit
def grouped_fp8_tasks(
    Counts,
    Tasks,
    E: tl.constexpr,
    TILES: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # Only tile starts are materialized; A, scales, routing and C stay compact.
    expert = tl.program_id(0)
    ids = tl.arange(0, BLOCK_E)
    counts = tl.load(Counts + ids, ids < expert, 0)
    first = tl.sum(counts, 0)
    tile = tl.sum(tl.cdiv(counts, BM), 0)
    length = tl.load(Counts + expert)
    nt = tl.cdiv(length, BM)
    for chunk in range(tl.cdiv(nt, 64)):
        i = chunk * 64 + tl.arange(0, 64)
        tl.store(Tasks + tile + i, first + i * BM, i < nt)
    if expert == E - 1:
        tl.store(Tasks + TILES, tile + nt)


@libentry()
@triton.jit
def grouped_fp8_matmul(
    A,
    B,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    USE_TMA: tl.constexpr,
    GROUP_M: tl.constexpr,
    IMPRECISE: tl.constexpr,
    LOOP_UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    if GROUP_M == 0:
        tile = pid % TILES
        block_n = pid // TILES
    else:
        # Keep nearby M and N tiles together so A and B can remain in L2.
        width = GROUP_M * tl.cdiv(N, BN)
        group = pid // width
        size = tl.minimum(TILES - group * GROUP_M, GROUP_M)
        tile = group * GROUP_M + pid % width % size
        block_n = pid % width // size
    if tile < tl.load(Tasks + TILES):
        first = tl.load(Tasks + tile)
        expert = tl.load(Groups + first)
        rows = first + tl.arange(0, BM)
        cols = block_n * BN + tl.arange(0, BN)
        kk = tl.arange(0, 128)
        valid = (rows < M) & (tl.load(Groups + rows, rows < M, -1) == expert)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in tl.range(K // 128, loop_unroll_factor=LOOP_UNROLL):
            if USE_TMA:
                a = A.load([first, block * 128])
                b = B.load([expert * N + block_n * BN, block * 128]).T
            else:
                offsets = block * 128 + kk
                a = tl.load(
                    A + rows.to(tl.int64)[:, None] * K + offsets[None, :],
                    valid[:, None],
                    0.0,
                )
                b = tl.load(
                    B
                    + expert.to(tl.int64) * N * K
                    + cols.to(tl.int64)[None, :] * K
                    + offsets[:, None],
                    cols[None, :] < N,
                    0.0,
                )
            # Promote every 64 K elements, preserving the established accuracy.
            # A zero limit makes Triton lower this dot through FP16 instead.
            partial = tl.dot(a, b, max_num_imprecise_acc=IMPRECISE)
            sa = tl.load(SA + rows.to(tl.int64) * SAM + block * SAK, valid, 0.0)
            if K > 128 and GROUP_M > 0 and BN <= 128:
                # One B scale covers this N tile; retain K=128 TMA transactions
                # and 64-element FP32 promotion without a broadcast scale tile.
                sb = tl.load(
                    SB
                    + expert.to(tl.int64) * SBE
                    + (block_n * BN // 128) * SBN
                    + block * SBK
                )
                acc = tl.fma(partial, (sa * sb)[:, None], acc)
            else:
                sb = tl.load(
                    SB + expert.to(tl.int64) * SBE + (cols // 128) * SBN + block * SBK,
                    cols < N,
                    0.0,
                )
                if K > 128 and GROUP_M > 0:
                    acc = tl.fma(partial, sa[:, None] * sb[None, :], acc)
                else:
                    acc += partial * sa[:, None] * sb[None, :]
        tl.store(
            C + rows.to(tl.int64)[:, None] * N + cols[None, :],
            acc,
            valid[:, None] & (cols[None, :] < N),
        )


@libentry()
@triton.jit
def grouped_fp8_matmul_persistent(
    A,
    B,
    C,
    CDesc,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    USE_TMA: tl.constexpr,
    GROUP_M: tl.constexpr,
    IMPRECISE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    TMA_STORE: tl.constexpr,
):
    actual_tiles = tl.load(Tasks + TILES)
    for pid in tl.range(
        tl.program_id(0), actual_tiles * tl.cdiv(N, BN), PROGRAMS, num_stages=2
    ):
        if GROUP_M == 0:
            tile = pid % actual_tiles
            block_n = pid // actual_tiles
        elif GROUP_M == 1:
            # Every visited tile is valid; a one-row traversal group has no tail divisor.
            tile = pid // tl.cdiv(N, BN)
            block_n = pid % tl.cdiv(N, BN)
        else:
            width = GROUP_M * tl.cdiv(N, BN)
            group = pid // width
            if GROUP_M > 1 and (group + 1) * GROUP_M <= actual_tiles:
                tile = group * GROUP_M + pid % GROUP_M
                block_n = pid % width // GROUP_M
            else:
                size = tl.minimum(actual_tiles - group * GROUP_M, GROUP_M)
                tile = group * GROUP_M + pid % width % size
                block_n = pid % width // size
        first = tl.load(Tasks + tile)
        expert = tl.load(Groups + first)
        rows = first + tl.arange(0, BM)
        cols = block_n * BN + tl.arange(0, BN)
        kk = tl.arange(0, 128)
        valid = (rows < M) & (tl.load(Groups + rows, rows < M, -1) == expert)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(K // 128):
            if USE_TMA:
                a = A.load([first, block * 128])
                b = B.load([expert * N + block_n * BN, block * 128]).T
            else:
                offsets = block * 128 + kk
                a = tl.load(
                    A + rows.to(tl.int64)[:, None] * K + offsets[None, :],
                    valid[:, None],
                    0.0,
                )
                b = tl.load(
                    B
                    + expert.to(tl.int64) * N * K
                    + cols.to(tl.int64)[None, :] * K
                    + offsets[:, None],
                    cols[None, :] < N,
                    0.0,
                )
            # Promote every 64 K elements while retaining native FP8 Tensor Cores.
            # A zero limit makes Triton lower this dot through FP16 instead.
            partial = tl.dot(a, b, max_num_imprecise_acc=IMPRECISE)
            sa = tl.load(SA + rows.to(tl.int64) * SAM + block * SAK, valid, 0.0)
            if BN <= 128:
                sb = tl.load(
                    SB
                    + expert.to(tl.int64) * SBE
                    + (block_n * BN // 128) * SBN
                    + block * SBK
                )
                acc += partial * (sa * sb)[:, None]
            else:
                sb = tl.load(
                    SB + expert.to(tl.int64) * SBE + (cols // 128) * SBN + block * SBK,
                    cols < N,
                    0.0,
                )
                if K > 128 and GROUP_M > 0:
                    acc = tl.fma(partial, sa[:, None] * sb[None, :], acc)
                else:
                    acc += partial * sa[:, None] * sb[None, :]
        # Unaligned output passes a pointer, so descriptor calls need a constexpr guard.
        if TMA_STORE:
            # Never write a complete tile across a compact expert boundary.
            last = first + BM - 1
            full_tile = (last < M) & (tl.load(Groups + last, last < M, -1) == expert)
            if full_tile:
                CDesc.store([first, block_n * BN], acc.to(tl.bfloat16))
            else:
                tl.store(
                    C + rows.to(tl.int64)[:, None] * N + cols[None, :],
                    acc,
                    valid[:, None] & (cols[None, :] < N),
                    cache_modifier="",
                )
        else:
            tl.store(
                C + rows.to(tl.int64)[:, None] * N + cols[None, :],
                acc,
                valid[:, None] & (cols[None, :] < N),
                cache_modifier="",
            )


@triton.jit
def tle_grouped_tile(
    pid, actual_tiles, N: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr
):
    width = GROUP_M * tl.cdiv(N, BN)
    group = pid // width
    if (group + 1) * GROUP_M <= actual_tiles:
        tile = group * GROUP_M + pid % GROUP_M
        block_n = pid % width // GROUP_M
    else:
        size = tl.minimum(actual_tiles - group * GROUP_M, GROUP_M)
        tile = group * GROUP_M + pid % width % size
        block_n = pid % width // size
    return tile, block_n


@triton.jit
def tle_grouped_producer(
    A,
    B,
    Groups,
    Tasks,
    a_buf,
    b_buf,
    empty,
    full_a,
    full_b,
    N: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGES: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    actual_tiles = tl.load(Tasks + TILES)
    sequence = 0
    for pid in range(tl.program_id(0), actual_tiles * tl.cdiv(N, BN), PROGRAMS):
        tile, block_n = tle_grouped_tile(pid, actual_tiles, N, BN, GROUP_M)
        first = tl.load(Tasks + tile)
        expert = tl.load(Groups + first)
        for block in range(tl.cdiv(K, 256)):
            slot = sequence % STAGES
            phase = sequence // STAGES
            tle.gpu.barrier_wait(empty[slot], phaseIdx=phase)
            tle.gpu.copy(
                A,
                a_buf.slot(slot),
                [BM, 256],
                [first, block * 256],
                barrier=full_a[slot],
            )
            tle.gpu.copy(
                B,
                b_buf.slot(slot),
                [BN, 256],
                [expert * N + block_n * BN, block * 256],
                barrier=full_b[slot],
            )
            sequence += 1


@triton.jit
def tle_grouped_consumer(
    C,
    SA,
    SB,
    Groups,
    Tasks,
    a_buf,
    b_buf,
    empty,
    full_a,
    full_b,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGES: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    actual_tiles = tl.load(Tasks + TILES)
    sequence = 0
    for pid in range(tl.program_id(0), actual_tiles * tl.cdiv(N, BN), PROGRAMS):
        tile, block_n = tle_grouped_tile(pid, actual_tiles, N, BN, GROUP_M)
        first = tl.load(Tasks + tile)
        expert = tl.load(Groups + first)
        rows = first + tl.arange(0, BM)
        cols = block_n * BN + tl.arange(0, BN)
        valid = (rows < M) & (tl.load(Groups + rows, rows < M, -1) == expert)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, 256)):
            slot = sequence % STAGES
            phase = sequence // STAGES
            tle.gpu.barrier_wait(full_a[slot], phaseIdx=phase)
            if STAGES != 3:
                tle.gpu.barrier_wait(full_b[slot], phaseIdx=phase)
            for half in tl.static_range(2):
                scale_k = block * 2 + half
                if scale_k < K // 128:
                    a_half = a_buf.slot(slot).subslice(half * 128, 128, 1)
                    b_half = b_buf.slot(slot).subslice(half * 128, 128, 1)
                    # K128 automatic promotion reads asynchronous accumulators too early.
                    # Complete each K64 partial before the ordinary FP32 sum.
                    partial0 = tle.gpu.wgmma(
                        a_half.subslice(0, 64, 1),
                        b_half.subslice(0, 64, 1),
                        trans_b=True,
                        max_num_imprecise_acc=64,
                    )
                    sa = tl.load(
                        SA + rows.to(tl.int64) * SAM + scale_k * SAK, valid, 0.0
                    )
                    sb = tl.load(
                        SB + expert.to(tl.int64) * SBE + block_n * SBN + scale_k * SBK
                    )
                    partial0 = tle.gpu.wgmma_wait(0, partial0)
                    partial1 = tle.gpu.wgmma(
                        a_half.subslice(64, 64, 1),
                        b_half.subslice(64, 64, 1),
                        trans_b=True,
                        max_num_imprecise_acc=64,
                    )
                    partial1 = tle.gpu.wgmma_wait(0, partial1)
                    # Ordinary FP32 arithmetic must only consume completed K64 results.
                    partial = partial0 + partial1
                    acc = tl.fma(partial, (sa * sb)[:, None], acc)
            tle.gpu.barrier_arrive(empty[slot], phaseIdx=phase)
            # Preserve barrier phase across output tiles, including odd K-block counts.
            sequence += 1
        tl.store(
            C + rows.to(tl.int64)[:, None] * N + cols[None, :],
            acc,
            valid[:, None] & (cols[None, :] < N),
        )


@libentry()
@triton.jit
def grouped_fp8_matmul_tle(
    A,
    B,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGES: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    a_buf = tle.gpu.alloc([STAGES, BM, 256], tl.float8e4nv)
    b_buf = tle.gpu.alloc([STAGES, BN, 256], tl.float8e4nv)
    if STAGES == 3:
        tl.static_assert(BM == BN)
        # Each copy contributes one arrival and its own byte count.
        # Allocate four barriers because the barrier tensor requires a power of two.
        full_a = tle.gpu.alloc_barriers(4, arrive_count=2, expect_bytes=BM * 256)
        full_b = full_a
        empty = tle.gpu.alloc_barriers(4, arrive_count=1, init=tle.gpu.READY)
    else:
        full_a = tle.gpu.alloc_barriers(STAGES, expect_bytes=BM * 256)
        full_b = tle.gpu.alloc_barriers(STAGES, expect_bytes=BN * 256)
        empty = tle.gpu.alloc_barriers(STAGES, arrive_count=1, init=tle.gpu.READY)
    tle.gpu.warp_specialize(
        [
            (
                tle_grouped_consumer,
                (
                    C,
                    SA,
                    SB,
                    Groups,
                    Tasks,
                    a_buf,
                    b_buf,
                    empty,
                    full_a,
                    full_b,
                    M,
                    N,
                    K,
                    TILES,
                    SAM,
                    SAK,
                    SBE,
                    SBN,
                    SBK,
                    BM,
                    BN,
                    GROUP_M,
                    STAGES,
                    PROGRAMS,
                ),
            ),
            (
                tle_grouped_producer,
                (
                    A,
                    B,
                    Groups,
                    Tasks,
                    a_buf,
                    b_buf,
                    empty,
                    full_a,
                    full_b,
                    N,
                    K,
                    TILES,
                    BM,
                    BN,
                    GROUP_M,
                    STAGES,
                    PROGRAMS,
                ),
            ),
        ],
        [1],
        [24],
    )


def grouped_gemm_w8a8_fp8(
    a: torch.Tensor,
    sa: torch.Tensor,
    b: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    """Compute block-scaled FP8 NT GEMM on exactly M packed rows.

    Rows are grouped in ascending expert order. m_rows contains E nonnegative
    counts summing to M, and m_indices gives the matching expert for each row.
    These device metadata values are caller preconditions (no host readback).
    """
    logger.debug("GEMS_NVIDIA GROUPED_GEMM_W8A8_FP8")
    if a.ndim != 2 or b.ndim != 3 or out.ndim != 2:
        raise ValueError("expected A[M,K], B[E,N,K] and out[M,N]")
    m, k = a.shape
    experts, n, bk = b.shape
    if bk != k or out.shape != (m, n) or m_indices.shape != (m,):
        raise ValueError("incompatible GEMM or routing dimensions")
    if (
        m_rows.shape != (experts,)
        or k == 0
        or k % 128
        or n == 0
        or n % 64
        or experts == 0
    ):
        raise ValueError(
            "requires m_rows[E], positive E, K divisible by 128 and N divisible by 64"
        )
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise TypeError("A and B must use FP8 E4M3")
    if (
        out.dtype != torch.bfloat16
        or sa.dtype != torch.float32
        or sb.dtype != torch.float32
    ):
        raise TypeError("requires BF16 output and FP32 scales")
    if m_indices.dtype != torch.int32 or m_rows.dtype != torch.int32:
        raise TypeError("routing indices and counts must use INT32")
    if sa.shape != (m, k // 128) or sb.shape != (
        experts,
        triton.cdiv(n, 128),
        k // 128,
    ):
        raise ValueError("incompatible FP8 block-scale shapes")
    tensors = (a, b, sa, sb, out, m_indices, m_rows)
    if a.device.type != "cuda" or any(t.device != a.device for t in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if not all(t.is_contiguous() for t in (a, b, out, m_indices, m_rows)):
        raise ValueError("A, B, out and both routing tensors must be contiguous")
    if m == 0:
        return
    with torch_device_fn.device(a.device):
        # Small expert groups avoid padding tensor-core M tiles with inactive rows.
        # Short K remains faster in GEMV through three and a quarter rows per expert.
        use_gemv = 4 * m <= 13 * experts if k == 128 else m <= experts
        if use_gemv:
            if k == 128:
                # Wider N tiles amortize small launches and the extra GEMV rows.
                # Keep narrower tiles between those ranges to limit register use.
                if 16 <= m <= 32 or m > 512:
                    bn, bk = 128, 128
                else:
                    bn, bk = (32 if m < 320 else 64), 128
            elif m <= 32:
                bn, bk = 8, 512
            else:
                # More column tiles and a shorter K tile reduce register pressure.
                bn, bk = 16, 256
            grouped_fp8_gemv[(m, triton.cdiv(n, bn))](
                a,
                b,
                out,
                sa,
                sb,
                m_indices,
                n,
                k,
                *sa.stride(),
                *sb.stride(),
                BN=bn,
                BK=min(triton.next_power_of_2(k), bk),
                num_warps=4,
                num_stages=1,
            )
            return
        large_groups = m >= 96 * experts
        # TMA descriptors require a 16-byte base alignment. Contiguous views with
        # arbitrary offsets retain the pointer-load FP8 path instead.
        use_tma = (a.data_ptr() | b.data_ptr()) % 16 == 0
        bm, bn = (128, 128) if k == 128 and large_groups else (64, 128)
        group_m = (1 if bm == 128 else 8) if large_groups else 0
        # Larger traversal groups help short K with up to three full M tiles
        # per expert. Retain the original traversal for larger expert groups.
        if k == 128 and large_groups and m // experts <= 384:
            group_m = 32
        num_warps, num_stages = 4, 3
        loop_unroll = 1
        if k > 128 and large_groups and use_tma:
            # Larger M tiles reduce repeated B reads through L2. Keep smaller
            # tiles for short expert groups to limit masked tail computation.
            average_rows, extra_rows = divmod(m, experts)
            padded_rows = 128 * (
                (experts - extra_rows) * triton.cdiv(average_rows, 128)
                + extra_rows * triton.cdiv(average_rows + 1, 128)
            )
            bm = 128 if padded_rows * 4 <= m * 5 else 64
            group_m = 512 // bm
            num_warps, num_stages = 8, (3 if bm == 128 else 4)
            loop_unroll = 2 if bm == 128 else 1
        tle_stages = 2
        if HAS_TLE_GROUPED_FP8 and k >= 1024 and m >= 256 * experts and use_tma:
            # The explicit K64 MMA path also amortizes BM128 on larger tails.
            # Retain two buffers for those tiles; regular BM128 groups benefit
            # from a third buffer and one shared A/B completion barrier.
            if bm == 128:
                tle_stages = 3
            bm, group_m = 128, 4
        aa = TensorDescriptor(a, [m, k], [k, 1], [bm, 128]) if use_tma else a
        bb = TensorDescriptor(b, [experts * n, k], [k, 1], [bn, 128]) if use_tma else b
        tiles = min(m, triton.cdiv(m, bm) + experts - 1)
        tasks = torch.empty((tiles + 1,), dtype=torch.int32, device=a.device)
        grouped_fp8_tasks[(experts,)](
            m_rows,
            tasks,
            experts,
            tiles,
            bm,
            triton.next_power_of_2(experts),
            num_warps=4,
        )
        # Amortize the larger transfer over groups with at least two full M tiles.
        if (
            HAS_TLE_GROUPED_FP8
            and k >= 1024
            and m >= 256 * experts
            and use_tma
            and bm == 128
        ):
            aa = TensorDescriptor(a, [m, k], [k, 1], [bm, 256])
            bb = TensorDescriptor(b, [experts * n, k], [k, 1], [bn, 256])
            num_sms = torch_device_fn.get_device_properties(
                a.device
            ).multi_processor_count
            programs = min(num_sms, tiles * triton.cdiv(n, bn))
            grouped_fp8_matmul_tle[(programs,)](
                aa,
                bb,
                out,
                sa,
                sb,
                m_indices,
                tasks,
                m,
                n,
                k,
                tiles,
                *sa.stride(),
                *sb.stride(),
                BM=bm,
                BN=bn,
                GROUP_M=group_m,
                STAGES=tle_stages,
                PROGRAMS=programs,
                num_warps=8,
                num_stages=1,
            )
            return
        if k == 128 and large_groups and use_tma:
            num_sms = torch_device_fn.get_device_properties(
                a.device
            ).multi_processor_count
            programs = min(tiles * triton.cdiv(n, bn), 2 * num_sms)
            use_tma_store = out.data_ptr() % 16 == 0
            # Descriptor stores favor contiguous N traversal for larger experts.
            # Retain the established traversal when output stores are masked.
            if use_tma_store and m >= 2 * bm * experts:
                group_m = 1
            c_desc = (
                TensorDescriptor(out, [m, n], [n, 1], [bm, bn])
                if use_tma_store
                else out
            )
            grouped_fp8_matmul_persistent[(programs,)](
                aa,
                bb,
                out,
                c_desc,
                sa,
                sb,
                m_indices,
                tasks,
                m,
                n,
                k,
                tiles,
                *sa.stride(),
                *sb.stride(),
                BM=bm,
                BN=bn,
                num_warps=4,
                num_stages=3,
                USE_TMA=use_tma,
                GROUP_M=group_m,
                IMPRECISE=64,
                PROGRAMS=programs,
                TMA_STORE=use_tma_store,
            )
            return
        grouped_fp8_matmul[(tiles * triton.cdiv(n, bn),)](
            aa,
            bb,
            out,
            sa,
            sb,
            m_indices,
            tasks,
            m,
            n,
            k,
            tiles,
            *sa.stride(),
            *sb.stride(),
            BM=bm,
            BN=bn,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_TMA=use_tma,
            GROUP_M=group_m,
            IMPRECISE=64,
            LOOP_UNROLL=loop_unroll,
        )
