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

"""Explicit-barrier BAddBMM for the M448/N7168/K256 FlagOSTune shape.

The implementation uses a full-A/full-B and aggregate-empty protocol.  It
deliberately does not use ``tle.pipe``.  Four
4-warp consumers compute disjoint 64x256 row partitions, while one 4-warp
producer fills two BM256xBK64 / BK64xBN256 shared-memory stages.
"""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
NUM_SLOTS = 2
K_TILES = 4
JIT_BLOCK_M = tl.constexpr(256)
JIT_BLOCK_N = tl.constexpr(256)

EXPLICIT_BARRIER_AVAILABLE = (
    hasattr(tle.gpu, "alloc_barriers")
    and hasattr(tle.gpu, "barrier_wait")
    and hasattr(tle.gpu, "barrier_arrive")
    and hasattr(tle.gpu, "buffered_tensor")
    and hasattr(tle.gpu.buffered_tensor, "slice")
)


@triton.jit
def _producer(
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty,
    load_m_offset,
    n_offset,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    period: tl.constexpr = 2 * NUM_SLOTS
    full_periods: tl.constexpr = K_TILES // period
    for period_idx in tl.range(0, full_periods, num_stages=1):
        for u in tl.static_range(0, period):
            tle.gpu.barrier_wait(empty[u % NUM_SLOTS], phaseIdx=u // NUM_SLOTS)
            k_iter = period_idx * period + u
            k_offset = k_iter * BLOCK_K
            tle.gpu.copy(
                a_desc,
                a_smem.slot(u % NUM_SLOTS),
                (JIT_BLOCK_M, BLOCK_K),
                (load_m_offset, k_offset),
                barrier=full_a[u % NUM_SLOTS],
            )
            # Ordinary contiguous B has physical shape [K,N], so this stage
            # is [BK,BN] and the consumer uses SQMMA without trans_b.
            tle.gpu.copy(
                b_desc,
                b_smem.slot(u % NUM_SLOTS),
                (BLOCK_K, JIT_BLOCK_N),
                (k_offset, n_offset),
                barrier=full_b[u % NUM_SLOTS],
            )


@triton.jit
def _consumer(
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty,
    bias_ptr,
    out_ptr,
    logical_m_offset,
    load_m_offset,
    n_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    columns = n_offset + tl.arange(0, JIT_BLOCK_N)
    accumulator = tl.zeros((64, JIT_BLOCK_N), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + columns,
            mask=columns < N,
            other=0.0,
        ).to(tl.float32)
        accumulator += bias[None, :]

    period: tl.constexpr = 2 * NUM_SLOTS
    full_periods: tl.constexpr = K_TILES // period
    for _period_idx in tl.range(0, full_periods, num_stages=1):
        for u in tl.static_range(0, period):
            tle.gpu.barrier_wait(full_a[u % NUM_SLOTS], phaseIdx=u // NUM_SLOTS)
            tle.gpu.barrier_wait(full_b[u % NUM_SLOTS], phaseIdx=u // NUM_SLOTS)
            accumulator = tle.gpu.wgmma(
                a_smem.slot(u % NUM_SLOTS).slice(ROW_OFFSET, 64, dim=0),
                b_smem.slot(u % NUM_SLOTS),
                accumulator,
            )
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            tle.gpu.barrier_arrive(empty[u % NUM_SLOTS], phaseIdx=u // NUM_SLOTS)

    # The final logical M tile begins at row 256 but a 256-row TME load from
    # there would overrun M=448.  Load [M-256,M)=[192,448), then store only
    # rows belonging to this logical CTA.
    rows = load_m_offset + ROW_OFFSET + tl.arange(0, 64)
    pointers = out_ptr + rows[:, None] * N + columns[None, :]
    mask = ((rows >= logical_m_offset) & (rows < M))[:, None] & (columns < N)[None, :]
    tl.store(
        pointers,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _kernel(
    a_desc,
    b_desc,
    bias_ptr,
    out_ptr,
    GRID_M: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    CONSUMER_REGS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = ext.program_id(0)
    pid_m = pid % GRID_M
    pid_n = pid // GRID_M
    logical_m_offset = (pid_m * JIT_BLOCK_M).to(tl.int32)
    n_offset = (pid_n * JIT_BLOCK_N).to(tl.int32)
    load_m_offset = tl.where(
        logical_m_offset + JIT_BLOCK_M <= M,
        logical_m_offset,
        M - JIT_BLOCK_M,
    )

    a_smem = tle.gpu.alloc(
        (NUM_SLOTS, JIT_BLOCK_M, BLOCK_K),
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (NUM_SLOTS, BLOCK_K, JIT_BLOCK_N),
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    full_a = tle.gpu.alloc_barriers(
        NUM_SLOTS,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=JIT_BLOCK_M * BLOCK_K * 2,
    )
    full_b = tle.gpu.alloc_barriers(
        NUM_SLOTS,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=BLOCK_K * JIT_BLOCK_N * 2,
    )
    empty = tle.gpu.alloc_barriers(
        NUM_SLOTS,
        arrive_count=16,
        init=tle.gpu.READY,
    )

    tle.gpu.warp_specialize(
        [
            (
                _consumer,
                (
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty,
                    bias_ptr,
                    out_ptr,
                    logical_m_offset,
                    load_m_offset,
                    n_offset,
                    M,
                    N,
                    K_TILES,
                    BLOCK_K,
                    NUM_SLOTS,
                    0,
                    HAS_BIAS,
                ),
            ),
            (
                _consumer,
                (
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty,
                    bias_ptr,
                    out_ptr,
                    logical_m_offset,
                    load_m_offset,
                    n_offset,
                    M,
                    N,
                    K_TILES,
                    BLOCK_K,
                    NUM_SLOTS,
                    64,
                    HAS_BIAS,
                ),
            ),
            (
                _consumer,
                (
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty,
                    bias_ptr,
                    out_ptr,
                    logical_m_offset,
                    load_m_offset,
                    n_offset,
                    M,
                    N,
                    K_TILES,
                    BLOCK_K,
                    NUM_SLOTS,
                    128,
                    HAS_BIAS,
                ),
            ),
            (
                _consumer,
                (
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty,
                    bias_ptr,
                    out_ptr,
                    logical_m_offset,
                    load_m_offset,
                    n_offset,
                    M,
                    N,
                    K_TILES,
                    BLOCK_K,
                    NUM_SLOTS,
                    192,
                    HAS_BIAS,
                ),
            ),
            (
                _producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty,
                    load_m_offset,
                    n_offset,
                    K_TILES,
                    BLOCK_K,
                    NUM_SLOTS,
                ),
            ),
        ],
        [4, 4, 4, 4],
        [CONSUMER_REGS, CONSUMER_REGS, CONSUMER_REGS, PRODUCER_REGS],
    )


def baddbmm_c2_explicit_out(
    bias: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not EXPLICIT_BARRIER_AVAILABLE:
        raise RuntimeError("explicit barrier APIs are unavailable")
    if tuple(batch1.shape) != (1, 448, 256):
        raise ValueError("batch1 must have shape (1,448,256)")
    if tuple(batch2.shape) != (1, 256, 7168):
        raise ValueError("batch2 must have shape (1,256,7168)")
    if tuple(bias.shape) != (7168,) or tuple(out.shape) != (1, 448, 7168):
        raise ValueError("bias/output shape mismatch")
    tensors = (bias, batch1, batch2, out)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("BF16 required")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.device != batch1.device for t in tensors):
        raise ValueError("all tensors must share one device")

    a, b, c = batch1[0], batch2[0], out[0]
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    grid_m = triton.cdiv(448, BLOCK_M)
    grid_n = triton.cdiv(7168, BLOCK_N)
    with torch_device_fn.device(a.device):
        _kernel[(grid_m * grid_n,)](
            a_desc,
            b_desc,
            bias,
            c,
            GRID_M=grid_m,
            M=448,
            N=7168,
            K_TILES=K_TILES,
            BLOCK_K=BLOCK_K,
            NUM_SLOTS=NUM_SLOTS,
            CONSUMER_REGS=192,
            PRODUCER_REGS=24,
            HAS_BIAS=True,
            num_warps=4,
            num_stages=2,
            maxnreg=192,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=False,
        )
    return out


__all__ = [
    "EXPLICIT_BARRIER_AVAILABLE",
    "baddbmm_c2_explicit_out",
]
