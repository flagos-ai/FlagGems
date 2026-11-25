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

"""FP16/BF16 MThreads BMM kernel for small-M shapes.

The kernel uses BM64 x BN64 x BK64 payloads and one CTA per (batch, N tile).
Each tile is consumed by one SQMMA. Both NN (contiguous B) and NT
(transpose-view B) are supported without materializing a copy.
"""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

BM = tl.constexpr(64)
BN = tl.constexpr(64)
BK = 64


@triton.jit
def _bm64_bn64_bk64_s2_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    pid_b,
    pid_n,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    a_row = (pid_b * M).to(tl.int32)
    n_offset = (pid_n * BN).to(tl.int32)
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        k_offset = (k_iter * BLOCK_K).to(tl.int32)
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(a_desc, a_slot.a, (BM, BLOCK_K), (a_row, k_offset))
        if TRANS_B:
            # Physical B is [batch, N, K].
            b_row = (pid_b * N + n_offset).to(tl.int32)
            tle.gpu.copy(b_desc, b_slot.b, (BN, BLOCK_K), (b_row, k_offset))
        else:
            # Physical/logical B is [batch, K, N].
            b_row = (pid_b * K + k_offset).to(tl.int32)
            tle.gpu.copy(b_desc, b_slot.b, (BLOCK_K, BN), (b_row, n_offset))
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def _bm64_bn64_bk64_s2_consumer(
    a_reader,
    b_reader,
    out_ptr,
    pid_b,
    pid_n,
    stride_ob,
    stride_om,
    stride_on,
    M: tl.constexpr,
    N: tl.constexpr,
    K_TILES: tl.constexpr,
    TRANS_B: tl.constexpr,
    UNROLL: tl.constexpr,
):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_iter in tl.range(
        0,
        K_TILES,
        num_stages=1,
        loop_unroll_factor=UNROLL,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        # A single SQMMA per tile keeps the loop-carried accumulator layout consistent.
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc, trans_b=TRANS_B)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    rows = tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    ptrs = (
        out_ptr
        + pid_b * stride_ob
        + rows[:, None] * stride_om
        + cols[None, :] * stride_on
    )
    tl.store(
        ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _bmm_bm64_bn64_bk64_s2_small_m_ws_kernel(
    a_desc,
    b_desc,
    out_ptr,
    stride_ob,
    stride_om,
    stride_on,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    TRANS_B: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    b_rows: tl.constexpr = BN if TRANS_B else BLOCK_K
    b_cols: tl.constexpr = BLOCK_K if TRANS_B else BN
    a_smem = tle.gpu.alloc(
        (NUM_SLOTS, BM, BLOCK_K),
        dtype=a_desc.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (NUM_SLOTS, b_rows, b_cols),
        dtype=b_desc.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="bmm_small_m_bm64_bn64_bk64_s2_a",
        a=a_smem,
    )
    b_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="bmm_small_m_bm64_bn64_bk64_s2_b",
        b=b_smem,
    )
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    tle.gpu.warp_specialize(
        [
            (
                _bm64_bn64_bk64_s2_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    out_ptr,
                    pid_b,
                    pid_n,
                    stride_ob,
                    stride_om,
                    stride_on,
                    M,
                    N,
                    k_tiles,
                    TRANS_B,
                    UNROLL,
                ),
            ),
            (
                _bm64_bn64_bk64_s2_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    pid_b,
                    pid_n,
                    M,
                    N,
                    K,
                    BLOCK_K,
                    k_tiles,
                    TRANS_B,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def _descriptors(a: torch.Tensor, b: torch.Tensor, *, trans_b: bool):
    batch, m, k = a.shape
    n = b.shape[2]
    a_desc = TensorDescriptor(a, [batch * m, k], [k, 1], [64, BK])
    if trans_b:
        # The logical transpose-view has storage physically arranged B,N,K.
        b_desc = TensorDescriptor(b, [batch * n, k], [k, 1], [64, BK])
    else:
        b_desc = TensorDescriptor(b, [batch * k, n], [n, 1], [BK, 64])
    return a_desc, b_desc


def _launch_with_descriptors(
    a_desc,
    b_desc,
    out: torch.Tensor,
    *,
    trans_b: bool,
    batch: int,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    grid = (triton.cdiv(n, 64), batch)
    _bmm_bm64_bn64_bk64_s2_small_m_ws_kernel[grid](
        a_desc,
        b_desc,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M=m,
        N=n,
        K=k,
        BLOCK_K=BK,
        NUM_SLOTS=2,
        TRANS_B=trans_b,
        UNROLL=1,
        num_warps=8,
        num_stages=2,
        enable_backend_opt=True,
        disable_max_ilp_scheduler=True,
    )
    return out


def bmm_bm64_bn64_bk64_s2_small_m_ws_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batch, m, k = a.shape
    n = b.shape[2]
    trans_b = b.stride(0) == n * k and b.stride(1) == 1 and b.stride(2) == k
    a_desc, b_desc = _descriptors(a, b, trans_b=trans_b)
    with torch_device_fn.device(a.device):
        _launch_with_descriptors(
            a_desc,
            b_desc,
            out,
            trans_b=trans_b,
            batch=batch,
            m=m,
            n=n,
            k=k,
        )
    return out
