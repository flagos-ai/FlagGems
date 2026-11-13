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
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

try:
    from triton.experimental.tle.language.gpu.mthreads import (
        MTHREADS_GEMM_COMPILER_OPTIONS_VERSION,
        MTHREADS_TLE_CONSUMER_SYNC_VERSION,
        TMECachePolicy,
    )
except ImportError:
    MTHREADS_GEMM_COMPILER_OPTIONS_VERSION = 0
    MTHREADS_TLE_CONSUMER_SYNC_VERSION = 0
    TMECachePolicy = None

# Named readers need CTA-wide initialization and per-epoch consumer rendezvous.
MULTICAST_ON = (
    MTHREADS_GEMM_COMPILER_OPTIONS_VERSION >= 2
    and MTHREADS_TLE_CONSUMER_SYNC_VERSION >= 1
)
logger = logging.getLogger(__name__)
CONSUMER_WARPS = 8
PRODUCER_WARPS = 4


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
def grouped_fp8_single_k_kernel(
    A,
    B,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    TILES: tl.constexpr,
    M,
    N: tl.constexpr,
    SAM: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    active = pm < tl.load(Tasks + TILES)
    first = tl.load(Tasks + pm, active, 0)
    group = tl.load(Groups + first)
    aa = tle.gpu.alloc(
        (BM, 128), A.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    bb = tle.gpu.alloc(
        (BN, 128), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    full_a = tle.gpu.alloc_barriers(
        1, arrive_count=1, init=tle.gpu.PENDING, expect_bytes=BM * 128
    )
    full_b = tle.gpu.alloc_barriers(
        1, arrive_count=1, init=tle.gpu.PENDING, expect_bytes=BN * 128
    )
    if active:
        tle.gpu.copy(A, aa, (BM, 128), (first, 0), barrier=full_a[0])
        tle.gpu.copy(B, bb, (BN, 128), (group * N + pn * BN, 0), barrier=full_b[0])
        tle.gpu.barrier_wait(full_a[0], phaseIdx=0)
        tle.gpu.barrier_wait(full_b[0], phaseIdx=0)
        acc = tle.gpu.wgmma(aa, bb, tl.zeros((BM, BN), tl.float32), trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        rows = first + tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        sa = tl.load(SA + rows * SAM, rows < M, 0)
        sb = tl.load(SB + group * SBE + (cols // 128) * SBN, cols < N, 0)
        valid = tl.load(Groups + rows, rows < M, -1) == group
        tl.store(
            C + rows.to(tl.int64)[:, None] * N + cols[None, :],
            acc * sa[:, None] * sb[None, :],
            (rows[:, None] < M) & valid[:, None] & (cols[None, :] < N),
        )


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


@triton.jit
def grouped_fp8_task(
    Tasks,
    Groups,
    task,
    total,
    N: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr = 1,
):
    columns: tl.constexpr = tl.cdiv(N, BN)
    is_valid = task < total
    if GROUP_M == 1:
        tile = task // columns
        pn = task % columns
    else:
        # Adjacent M tiles reuse the same B partition before advancing in N.
        span: tl.constexpr = GROUP_M * columns
        first = task // span * GROUP_M
        width = tl.maximum(1, tl.minimum(GROUP_M, total // columns - first))
        tile = first + task % span % width
        pn = tl.where(is_valid, task % span // width, 0)
    pm = tl.load(Tasks + tile, is_valid, 0)
    expert = tl.load(Groups + pm)
    return pm, pn, expert, is_valid


@triton.jit
def grouped_fp8_producer(
    writer,
    finish,
    A,
    B,
    Groups,
    Tasks,
    M,
    TILES,
    PROGRAMS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    STAGES: tl.constexpr,
):
    total = tl.load(Tasks + TILES) * tl.cdiv(N, BN)
    sequence = 0
    for task in tl.range(
        tl.program_id(0), tl.maximum(total, PROGRAMS), PROGRAMS, num_stages=1
    ):
        pm, pn, expert, _ = grouped_fp8_task(Tasks, Groups, task, total, N, BN)
        if K_BLOCKS == 2:
            pm = tl.minimum(pm, M - BM)
        for block in tl.range(0, K // (128 * K_BLOCKS), num_stages=1):
            slot = writer.acquire(sequence)
            if K_BLOCKS == 2:
                # Each pair shares one TME transaction per operand. Cache A for the adjacent N tile.
                tle.gpu.copy(
                    A,
                    slot.a,
                    (BM, 256),
                    (pm, block * 256),
                    cache_policy=TMECachePolicy(inner="normal", outer="persist"),
                )
                tle.gpu.copy(
                    B,
                    slot.b,
                    (BN, 256),
                    (expert * N + pn * BN, block * 256),
                    cache_policy=TMECachePolicy(inner="normal", outer="normal"),
                )
            else:
                tle.gpu.copy(A, slot.a, (BM, 128), (pm, block * 128))
                tle.gpu.copy(B, slot.b, (BN, 128), (expert * N + pn * BN, block * 128))
            writer.commit(sequence)
            sequence += 1
    if K_BLOCKS == 2:
        writer.acquire(sequence + STAGES - 1)
    tle.gpu.barrier_arrive(finish[0], phaseIdx=0)
    tle.gpu.barrier_wait(finish[0], phaseIdx=0)


@triton.jit
def grouped_fp8_consumer(
    reader,
    finish,
    reader_sync,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    TILES,
    PROGRAMS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    K_BLOCKS: tl.constexpr,
):
    total = tl.load(Tasks + TILES) * tl.cdiv(N, BN)
    sequence = 0
    for task in tl.range(
        tl.program_id(0), tl.maximum(total, PROGRAMS), PROGRAMS, num_stages=1
    ):
        pm, pn, expert, is_valid = grouped_fp8_task(Tasks, Groups, task, total, N, BN)
        if K_BLOCKS == 2:
            # Keep the whole TME read inside A, including the last expert tail.
            first = tl.minimum(pm, M - BM)
        else:
            first = pm
        rows = first + tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in tl.range(0, K // (128 * K_BLOCKS), num_stages=1):
            ready = reader.wait(sequence)
            if K_BLOCKS == 2:
                # Scaling is still independent for each K=128 SQMMA, with one B scale per N tile.
                for u in tl.static_range(2):
                    sub = block * 2 + u
                    partial = tle.gpu.wgmma(
                        ready.slot.a.subslice(u * 128, 128, dim=1),
                        ready.slot.b.subslice(u * 128, 128, dim=1),
                        tl.zeros((BM, BN), tl.float32),
                        trans_b=True,
                    )
                    sa = tl.load(SA + rows * SAM + sub * SAK, rows < M, 0.0)
                    sb = tl.load(SB + expert * SBE + (pn * BN // 128) * SBN + sub * SBK)
                    partial = tle.gpu.wgmma_wait(0, partial)
                    acc = tl.fma(partial, (sa * sb)[:, None], acc)
            else:
                partial = tle.gpu.wgmma(
                    ready.slot.a,
                    ready.slot.b,
                    tl.zeros((BM, BN), tl.float32),
                    trans_b=True,
                )
                partial = tle.gpu.wgmma_wait(0, partial)
                sa = tl.load(SA + rows * SAM + block * SAK, rows < M, 0)
                sb = tl.load(
                    SB + expert * SBE + (cols // 128) * SBN + block * SBK, cols < N, 0
                )
                acc += partial * sa[:, None] * sb[None, :]
            if K_BLOCKS == 2:
                # The pipe tracks every slot release. Rendezvous once per two turns
                # of the three-slot ring to limit skew without stalling each dot pair.
                if (sequence + 1) % 6 == 0:
                    tle.gpu.barrier_arrive(reader_sync[0], phaseIdx=(sequence // 6) % 2)
                    tle.gpu.barrier_wait(reader_sync[0], phaseIdx=(sequence // 6) % 2)
            reader.release(sequence)
            sequence += 1
        valid = (rows >= pm) & (tl.load(Groups + rows, rows < M, -1) == expert)
        tl.store(
            C + rows.to(tl.int64)[:, None] * N + cols[None, :],
            acc,
            (rows[:, None] < M) & valid[:, None] & (cols[None, :] < N) & is_valid,
        )
    tle.gpu.barrier_arrive(finish[0], phaseIdx=0)
    tle.gpu.barrier_wait(finish[0], phaseIdx=0)


@triton.jit(do_not_specialize=["M", "TILES"])
def grouped_fp8_ws_kernel(
    A,
    B,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    TILES,
    PROGRAMS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    STAGES: tl.constexpr,
    CW: tl.constexpr,
    PW: tl.constexpr,
):
    aa = tle.gpu.alloc(
        (STAGES, BM, 128 * K_BLOCKS),
        A.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    bb = tle.gpu.alloc(
        (STAGES, BN, 128 * K_BLOCKS),
        B.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="ab", readers=("r",), a=aa, b=bb)
    finish = tle.gpu.alloc_barriers(1, arrive_count=CW + PW, init=tle.gpu.PENDING)
    if K_BLOCKS == 2:
        reader_sync = tle.gpu.alloc_barriers(1, arrive_count=CW, init=tle.gpu.PENDING)
    else:
        reader_sync = finish
    tle.gpu.warp_specialize(
        [
            (
                grouped_fp8_consumer,
                (
                    pipe.reader("r", fields=("a", "b")),
                    finish,
                    reader_sync,
                    C,
                    SA,
                    SB,
                    Groups,
                    Tasks,
                    M,
                    TILES,
                    PROGRAMS,
                    N,
                    K,
                    SAM,
                    SAK,
                    SBE,
                    SBN,
                    SBK,
                    BM,
                    BN,
                    K_BLOCKS,
                ),
            ),
            (
                grouped_fp8_producer,
                (
                    pipe.writer(),
                    finish,
                    A,
                    B,
                    Groups,
                    Tasks,
                    M,
                    TILES,
                    PROGRAMS,
                    N,
                    K,
                    BM,
                    BN,
                    K_BLOCKS,
                    STAGES,
                ),
            ),
        ],
        worker_num_warps=[PW],
        worker_num_regs=[24],
    )


@triton.jit
def grouped_fp8_multicast_producer(
    writer,
    A,
    B,
    Groups,
    Tasks,
    Total,
    Start,
    ITERATIONS,
    PROGRAMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
):
    total = Total
    sequence = 0
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        task = Start + iteration * PROGRAMS
        pm, pn, expert, _ = grouped_fp8_task(
            Tasks, Groups, task, total, N, 4 * BN, GROUP_M
        )
        slot = writer.acquire(sequence)
        tle.gpu.copy(A, slot.a, (BM, 128), (pm, 0))
        tle.gpu.copy(B, slot.b0, (BN, 128), (expert * N + pn * 4 * BN, 0))
        tle.gpu.copy(B, slot.b1, (BN, 128), (expert * N + pn * 4 * BN + BN, 0))
        tle.gpu.copy(
            B,
            slot.b2,
            (BN, 128),
            (expert * N + pn * 4 * BN + 2 * BN, 0),
        )
        tle.gpu.copy(
            B,
            slot.b3,
            (BN, 128),
            (expert * N + pn * 4 * BN + 3 * BN, 0),
        )
        writer.commit(sequence)
        sequence += 1
    # Keep the producer alive until every reader releases the final shared slot.
    writer.acquire(sequence + STAGES - 1)


@triton.jit
def grouped_fp8_multicast_consumer(
    reader,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    Total,
    Start,
    ITERATIONS,
    PROGRAMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    N: tl.constexpr,
    SAM: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ROLE: tl.constexpr,
):
    total = Total
    sequence = 0
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        task = Start + iteration * PROGRAMS
        pm, pn, expert, is_valid = grouped_fp8_task(
            Tasks, Groups, task, total, N, 4 * BN, GROUP_M
        )
        rows = pm + tl.arange(0, BM)
        cols = pn * 4 * BN + ROLE * BN + tl.arange(0, BN)
        ready = reader.wait(sequence)
        if ROLE == 0:
            operand_b = ready.slot.b0
        elif ROLE == 1:
            operand_b = ready.slot.b1
        elif ROLE == 2:
            operand_b = ready.slot.b2
        else:
            operand_b = ready.slot.b3
        partial = tle.gpu.wgmma(
            ready.slot.a, operand_b, tl.zeros((BM, BN), tl.float32), trans_b=True
        )
        sa = tl.load(SA + rows * SAM, rows < M, 0)
        if GROUP_M != 1:
            sb = tl.load(SB + expert * SBE + ((pn * 4 + ROLE) * BN // 128) * SBN)
        else:
            sb = tl.load(SB + expert * SBE + (cols // 128) * SBN, cols < N, 0)
        partial = tle.gpu.wgmma_wait(0, partial)
        if GROUP_M != 1:
            acc = partial * (sa * sb)[:, None]
        else:
            acc = partial * sa[:, None] * sb[None, :]
        reader.release(sequence)
        sequence += 1
        valid = tl.load(Groups + rows, rows < M, -1) == expert
        tl.store(
            C + rows.to(tl.int64)[:, None] * N + cols[None, :],
            acc,
            (rows[:, None] < M) & valid[:, None] & (cols[None, :] < N) & is_valid,
        )


@triton.jit
def grouped_fp8_multicast_kernel(
    A,
    B,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    TILES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    N: tl.constexpr,
    SAM: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    CW: tl.constexpr,
    PW: tl.constexpr,
):
    aa = tle.gpu.alloc(
        (STAGES, BM, 128), A.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    b0 = tle.gpu.alloc(
        (STAGES, BN, 128), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    b1 = tle.gpu.alloc(
        (STAGES, BN, 128), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    b2 = tle.gpu.alloc(
        (STAGES, BN, 128), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    b3 = tle.gpu.alloc(
        (STAGES, BN, 128), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="ab",
        readers=("r0", "r1", "r2", "r3"),
        a=aa,
        b0=b0,
        b1=b1,
        b2=b2,
        b3=b3,
    )
    # All reader partitions must use the same loop bound, including the last CTA wave.
    total = tl.load(Tasks + TILES) * tl.cdiv(N, 4 * BN)
    start = tl.program_id(0)
    iterations = tl.cdiv(total, PROGRAMS)
    consumer_args: tl.constexpr = (
        C,
        SA,
        SB,
        Groups,
        Tasks,
        M,
        total,
        start,
        iterations,
        PROGRAMS,
        GROUP_M,
        N,
        SAM,
        SBE,
        SBN,
        BM,
        BN,
    )
    producer_args: tl.constexpr = (
        A,
        B,
        Groups,
        Tasks,
        total,
        start,
        iterations,
        PROGRAMS,
        GROUP_M,
        N,
        BM,
        BN,
        STAGES,
    )
    tle.gpu.warp_specialize(
        [
            (
                grouped_fp8_multicast_consumer,
                (pipe.reader("r0", fields=("a", "b0")),) + consumer_args + (0,),
            ),
            (
                grouped_fp8_multicast_consumer,
                (pipe.reader("r1", fields=("a", "b1")),) + consumer_args + (1,),
            ),
            (
                grouped_fp8_multicast_consumer,
                (pipe.reader("r2", fields=("a", "b2")),) + consumer_args + (2,),
            ),
            (
                grouped_fp8_multicast_consumer,
                (pipe.reader("r3", fields=("a", "b3")),) + consumer_args + (3,),
            ),
            (grouped_fp8_multicast_producer, (pipe.writer(),) + producer_args),
        ],
        worker_num_warps=[CW, CW, CW, PW],
        worker_num_regs=[160, 160, 160, 24],
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
    logger.debug("GEMS_MTHREADS GROUPED_GEMM_W8A8_FP8")
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
    if a.device.type != "musa" or any(t.device != a.device for t in tensors):
        raise ValueError("all tensors must be on the same MUSA device")
    if not all(t.is_contiguous() for t in (a, b, out, m_indices, m_rows)):
        raise ValueError("A, B, out and both routing tensors must be contiguous")
    if m == 0:
        return
    # Use shared A tiles when enough rows amortize warp-specialization overhead.
    is_dense = m >= experts * 64 and n % 256 == 0
    is_multicast = MULTICAST_ON and is_dense and k == 128
    if is_multicast:
        # Shorten the M tile when the last tile has few useful rows on average.
        mean_rows = triton.cdiv(m, experts)
        if 0 < mean_rows % 128 <= 32 and mean_rows >= 384:
            bm, bn, group_m = 64, 128, 16
        else:
            bm, bn, group_m = 128, 64, 1
        consumer_warps = 4
    elif is_dense and k > 128:
        bm, bn = 128, 128
    else:
        bm, bn = (64, 64) if k == 128 else (64, 256)
    k_blocks = 2 if MULTICAST_ON and is_dense and m >= bm and k % 256 == 0 else 1
    with torch_device_fn.device(a.device):
        if m <= min(experts, 128 if k > 128 else 256):
            grouped_fp8_gemv[(m, triton.cdiv(n, 8 if k > 128 else 32))](
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
                BN=8 if k > 128 else 32,
                BK=min(triton.next_power_of_2(k), 512),
                num_warps=4,
                num_stages=1,
            )
            return
        a_desc = TensorDescriptor(
            a, list(a.shape), list(a.stride()), [bm, 128 * k_blocks]
        )
        flat_b = b.view(experts * n, k)
        b_desc = TensorDescriptor(
            flat_b, list(flat_b.shape), list(flat_b.stride()), [bn, 128 * k_blocks]
        )
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
        if k == 128 and not is_multicast:
            grouped_fp8_single_k_kernel[(tiles, triton.cdiv(n, bn))](
                a_desc,
                b_desc,
                out,
                sa,
                sb,
                m_indices,
                tasks,
                tiles,
                m,
                n,
                sa.stride(0),
                sb.stride(0),
                sb.stride(1),
                BM=bm,
                BN=bn,
                num_warps=4,
                num_stages=1,
            )
        else:
            num_sms = torch_device_fn.get_device_properties(
                a.device
            ).multi_processor_count
            programs = min(num_sms, tiles * triton.cdiv(n, bn))
            if is_multicast:
                grouped_fp8_multicast_kernel[(programs,)](
                    a_desc,
                    b_desc,
                    out,
                    sa,
                    sb,
                    m_indices,
                    tasks,
                    m,
                    tiles,
                    programs,
                    group_m,
                    n,
                    sa.stride(0),
                    sb.stride(0),
                    sb.stride(1),
                    BM=bm,
                    BN=bn,
                    STAGES=2,
                    CW=consumer_warps,
                    PW=PRODUCER_WARPS,
                    num_warps=consumer_warps,
                    num_stages=1,
                    enable_backend_opt=True,
                    disable_max_ilp_scheduler=True,
                    llc_options="-mtgpu-alloc-shared-memory-from-zero=1",
                )
                return
            opts = (
                dict(
                    enable_backend_opt=True,
                    llc_options="-mtgpu-alloc-shared-memory-from-zero=1",
                )
                if k_blocks == 2
                else {}
            )
            grouped_fp8_ws_kernel[(programs,)](
                a_desc,
                b_desc,
                out,
                sa,
                sb,
                m_indices,
                tasks,
                m,
                tiles,
                programs,
                n,
                k,
                *sa.stride(),
                *sb.stride(),
                BM=bm,
                BN=bn,
                STAGES=3 if k_blocks == 2 else 2,
                K_BLOCKS=k_blocks,
                CW=CONSUMER_WARPS,
                PW=PRODUCER_WARPS,
                num_warps=CONSUMER_WARPS,
                num_stages=1,
                **opts,
            )
