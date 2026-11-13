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
    )
except ImportError:
    MTHREADS_GEMM_COMPILER_OPTIONS_VERSION = 0
    MTHREADS_TLE_CONSUMER_SYNC_VERSION = 0

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
    ALIGN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * ALIGN
    stop = row + ALIGN
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.load(Groups + row)
    kk = tl.arange(0, BK)
    scale_k = tl.arange(0, BK // 128)
    while (row < stop) & (group >= 0):
        acc = tl.zeros((BN,), tl.float32)
        for base in range(tl.cdiv(K, BK)):
            offsets = base * BK + kk
            a = tl.load(A + row * K + offsets, offsets < K, 0.0).to(tl.float32)
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
        row += 1
        group = tl.load(Groups + row, row < stop, -1)


@libentry()
@triton.jit
def grouped_fp8_single_k_kernel(
    A,
    B,
    C,
    SA,
    SB,
    Groups,
    M,
    N: tl.constexpr,
    SAM: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    group = tl.load(Groups + pm * BM)
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
    if group >= 0:
        tle.gpu.copy(A, aa, (BM, 128), (pm * BM, 0), barrier=full_a[0])
        tle.gpu.copy(B, bb, (BN, 128), (group * N + pn * BN, 0), barrier=full_b[0])
        tle.gpu.barrier_wait(full_a[0], phaseIdx=0)
        tle.gpu.barrier_wait(full_b[0], phaseIdx=0)
        acc = tle.gpu.wgmma(aa, bb, tl.zeros((BM, BN), tl.float32), trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        rows = pm * BM + tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        sa = tl.load(SA + rows * SAM, rows < M, 0)
        sb = tl.load(SB + group * SBE + (cols // 128) * SBN, cols < N, 0)
        valid = tl.load(Groups + rows, rows < M, -1) >= 0
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc * sa[:, None] * sb[None, :],
            (rows[:, None] < M) & valid[:, None] & (cols[None, :] < N),
        )


@libentry()
@triton.jit
def grouped_fp8_tasks(
    Groups, Tasks, TILES: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr
):
    tiles = tl.arange(0, BLOCK)
    expert = tl.load(Groups + tiles * BM, tiles < TILES, -1)
    valid = expert >= 0
    offsets = tl.cumsum(valid.to(tl.int32), 0) - 1
    tl.store(Tasks + offsets, tiles, valid)
    tl.store(Tasks + TILES, tl.sum(valid.to(tl.int32), 0))


@triton.jit
def grouped_fp8_task(
    Tasks, Groups, task, total, N: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr
):
    columns: tl.constexpr = tl.cdiv(N, BN)
    is_valid = task < total
    pm = tl.load(Tasks + task // columns, is_valid, 0)
    pn = task % columns
    expert = tl.maximum(tl.load(Groups + pm * BM), 0)
    return pm, pn, expert, is_valid


@triton.jit
def grouped_fp8_producer(
    writer,
    finish,
    A,
    B,
    Groups,
    Tasks,
    TILES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    total = tl.load(Tasks + TILES) * tl.cdiv(N, BN)
    sequence = 0
    for task in tl.range(
        tl.program_id(0), tl.maximum(total, PROGRAMS), PROGRAMS, num_stages=1
    ):
        pm, pn, expert, _ = grouped_fp8_task(Tasks, Groups, task, total, N, BN, BM)
        for block in tl.range(0, K // 128, num_stages=1):
            slot = writer.acquire(sequence)
            tle.gpu.copy(A, slot.a, (BM, 128), (pm * BM, block * 128))
            tle.gpu.copy(B, slot.b, (BN, 128), (expert * N + pn * BN, block * 128))
            writer.commit(sequence)
            sequence += 1
    tle.gpu.barrier_arrive(finish[0], phaseIdx=0)
    tle.gpu.barrier_wait(finish[0], phaseIdx=0)


@triton.jit
def grouped_fp8_consumer(
    reader,
    finish,
    C,
    SA,
    SB,
    Groups,
    Tasks,
    M,
    TILES: tl.constexpr,
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
):
    total = tl.load(Tasks + TILES) * tl.cdiv(N, BN)
    sequence = 0
    for task in tl.range(
        tl.program_id(0), tl.maximum(total, PROGRAMS), PROGRAMS, num_stages=1
    ):
        pm, pn, expert, is_valid = grouped_fp8_task(
            Tasks, Groups, task, total, N, BN, BM
        )
        rows = pm * BM + tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in tl.range(0, K // 128, num_stages=1):
            ready = reader.wait(sequence)
            partial = tle.gpu.wgmma(
                ready.slot.a, ready.slot.b, tl.zeros((BM, BN), tl.float32), trans_b=True
            )
            partial = tle.gpu.wgmma_wait(0, partial)
            sa = tl.load(SA + rows * SAM + block * SAK, rows < M, 0)
            sb = tl.load(
                SB + expert * SBE + (cols // 128) * SBN + block * SBK, cols < N, 0
            )
            acc += partial * sa[:, None] * sb[None, :]
            reader.release(sequence)
            sequence += 1
        valid = tl.load(Groups + rows, rows < M, -1) >= 0
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc,
            (rows[:, None] < M) & valid[:, None] & (cols[None, :] < N) & is_valid,
        )
    tle.gpu.barrier_arrive(finish[0], phaseIdx=0)
    tle.gpu.barrier_wait(finish[0], phaseIdx=0)


@libentry()
@triton.jit
def grouped_fp8_ws_kernel(
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
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    CW: tl.constexpr,
    PW: tl.constexpr,
):
    aa = tle.gpu.alloc(
        (STAGES, BM, 128), A.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    bb = tle.gpu.alloc(
        (STAGES, BN, 128), B.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="ab", readers=("r",), a=aa, b=bb)
    finish = tle.gpu.alloc_barriers(1, arrive_count=CW + PW, init=tle.gpu.PENDING)
    tle.gpu.warp_specialize(
        [
            (
                grouped_fp8_consumer,
                (
                    pipe.reader("r", fields=("a", "b")),
                    finish,
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
                    TILES,
                    PROGRAMS,
                    N,
                    K,
                    BM,
                    BN,
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
    ITERATIONS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NC: tl.constexpr,
    STAGES: tl.constexpr,
):
    total = Total
    sequence = 0
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        task = Start + iteration * PROGRAMS
        pm, pn, expert, _ = grouped_fp8_task(Tasks, Groups, task, total, N, NC * BN, BM)
        for block in tl.range(0, K // 128, num_stages=1):
            slot = writer.acquire(sequence)
            tle.gpu.copy(A, slot.a, (BM, 128), (pm * BM, block * 128))
            tle.gpu.copy(
                B, slot.b0, (BN, 128), (expert * N + pn * NC * BN, block * 128)
            )
            tle.gpu.copy(
                B, slot.b1, (BN, 128), (expert * N + pn * NC * BN + BN, block * 128)
            )
            if NC == 4:
                tle.gpu.copy(
                    B,
                    slot.b2,
                    (BN, 128),
                    (expert * N + pn * NC * BN + 2 * BN, block * 128),
                )
                tle.gpu.copy(
                    B,
                    slot.b3,
                    (BN, 128),
                    (expert * N + pn * NC * BN + 3 * BN, block * 128),
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
    ITERATIONS: tl.constexpr,
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
    NC: tl.constexpr,
    ROLE: tl.constexpr,
):
    total = Total
    sequence = 0
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        task = Start + iteration * PROGRAMS
        pm, pn, expert, is_valid = grouped_fp8_task(
            Tasks, Groups, task, total, N, NC * BN, BM
        )
        rows = pm * BM + tl.arange(0, BM)
        cols = pn * NC * BN + ROLE * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in tl.range(0, K // 128, num_stages=1):
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
            sa = tl.load(SA + rows * SAM + block * SAK, rows < M, 0)
            sb = tl.load(
                SB + expert * SBE + (cols // 128) * SBN + block * SBK, cols < N, 0
            )
            partial = tle.gpu.wgmma_wait(0, partial)
            acc += partial * sa[:, None] * sb[None, :]
            reader.release(sequence)
            sequence += 1
        valid = tl.load(Groups + rows, rows < M, -1) >= 0
        tl.store(
            C + rows[:, None] * N + cols[None, :],
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
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBE: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    CW: tl.constexpr,
    PW: tl.constexpr,
    NC: tl.constexpr,
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
    if NC == 4:
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
    else:
        pipe = tle.pipe(
            capacity=STAGES,
            scope="cta",
            name="ab",
            readers=("r0", "r1"),
            a=aa,
            b0=b0,
            b1=b1,
        )
    # Named readers must execute the same number of pipe epochs, including padding.
    total = tl.load(Tasks + TILES) * tl.cdiv(N, NC * BN)
    start = tl.program_id(0)
    iterations: tl.constexpr = tl.cdiv(TILES * tl.cdiv(N, NC * BN), PROGRAMS)
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
        N,
        K,
        SAM,
        SAK,
        SBE,
        SBN,
        SBK,
        BM,
        BN,
        NC,
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
        N,
        K,
        BM,
        BN,
        NC,
        STAGES,
    )
    if NC == 4:
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
    else:
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
                (grouped_fp8_multicast_producer, (pipe.writer(),) + producer_args),
            ],
            worker_num_warps=[CW, PW],
            worker_num_regs=[160, 24],
        )


def grouped_gemm_w8a8_fp8(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    *,
    alignment_m: int = 128,
) -> None:
    """FP8 block-scaled NT GEMM with aligned contiguous expert segments.

    A/B use E4M3 and scales (1,128)/(128,128). Negative routing rows leave
    output unchanged. Each nonnegative expert segment starts at a multiple
    of alignment_m; routing values are caller-provided device metadata.
    """
    logger.debug("GEMS_MTHREADS GROUPED_GEMM_W8A8_FP8")
    a, sa = lhs
    b, sb = rhs
    if a.ndim != 2 or b.ndim != 3 or out.ndim != 2:
        raise ValueError("expected A[M,K], B[E,N,K] and out[M,N]")
    m, k = a.shape
    experts, n, bk = b.shape
    if bk != k or out.shape != (m, n) or m_indices.shape != (m,):
        raise ValueError("incompatible GEMM or routing dimensions")
    if alignment_m not in (128, 256):
        raise ValueError("alignment_m must be 128 or 256")
    if k == 0 or k % 128 or n == 0 or n % 64 or m % alignment_m or experts == 0:
        raise ValueError("requires positive K%128=0, N%64=0, E and aligned M")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise TypeError("A and B must use FP8 E4M3")
    if (
        out.dtype != torch.bfloat16
        or sa.dtype != torch.float32
        or sb.dtype != torch.float32
    ):
        raise TypeError("requires BF16 output and FP32 scales")
    if m_indices.dtype != torch.int32:
        raise TypeError("m_indices must use INT32")
    if sa.shape != (m, k // 128) or sb.shape != (
        experts,
        triton.cdiv(n, 128),
        k // 128,
    ):
        raise ValueError("incompatible FP8 block-scale shapes")
    tensors = (a, b, sa, sb, out, m_indices)
    if a.device.type != "musa" or any(t.device != a.device for t in tensors):
        raise ValueError("all tensors must be on the same MUSA device")
    if not all(t.is_contiguous() for t in (a, b, out, m_indices)):
        raise ValueError("A, B, out and m_indices must be contiguous")
    if m == 0:
        return
    # Share A tiles between independent SQMMA consumers on dense expert segments.
    is_multicast = (
        MULTICAST_ON
        and alignment_m == 256
        and m > experts * alignment_m
        and n % 256 == 0
    )
    if is_multicast:
        bm, bn, consumer_warps, consumers = (
            (128, 64, 4, 4) if k == 128 else (256, 64, 8, 2)
        )
    else:
        bm, bn = (64, 64) if k == 128 else (64, 256)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    with torch_device_fn.device(a.device):
        if k >= 2048 and n <= 256 and m <= alignment_m * 16 and m < experts * 128:
            grouped_fp8_gemv[(m // alignment_m, triton.cdiv(n, 8))](
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
                ALIGN=alignment_m,
                BN=8,
                BK=512,
                num_warps=4,
                num_stages=1,
            )
            return
        a_desc = TensorDescriptor(a, list(a.shape), list(a.stride()), [bm, 128])
        flat_b = b.view(experts * n, k)
        b_desc = TensorDescriptor(
            flat_b, list(flat_b.shape), list(flat_b.stride()), [bn, 128]
        )
        if k == 128 and not is_multicast:
            grouped_fp8_single_k_kernel[grid](
                a_desc,
                b_desc,
                out,
                sa,
                sb,
                m_indices,
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
            tiles = m // bm
            tasks = torch.empty((tiles + 1,), dtype=torch.int32, device=a.device)
            grouped_fp8_tasks[(1,)](
                m_indices, tasks, tiles, bm, triton.next_power_of_2(tiles), num_warps=4
            )
            num_sms = torch_device_fn.get_device_properties(
                a.device
            ).multi_processor_count
            programs = min(num_sms, max(1, m // alignment_m) * triton.cdiv(n, bn))
            if is_multicast:
                options = dict(
                    enable_backend_opt=True,
                    disable_max_ilp_scheduler=True,
                    disable_sqmma_m_split_preference=False,
                    llc_options="-mtgpu-alloc-shared-memory-from-zero=1",
                )
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
                    n,
                    k,
                    *sa.stride(),
                    *sb.stride(),
                    BM=bm,
                    BN=bn,
                    STAGES=2 if k == 128 else 3,
                    CW=consumer_warps,
                    PW=PRODUCER_WARPS,
                    NC=consumers,
                    num_warps=consumer_warps,
                    num_stages=1,
                    **options,
                )
                return
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
                STAGES=2,
                CW=CONSUMER_WARPS,
                PW=PRODUCER_WARPS,
                num_warps=CONSUMER_WARPS,
                num_stages=1,
            )
