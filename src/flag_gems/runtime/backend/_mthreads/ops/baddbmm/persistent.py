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

"""Persistent BF16 BAddBMM for the two large FlagOSTune shapes.

The schedule uses BM256xBN256 output tiles and 60 persistent CTAs.  One
16-warp consumer computes two BN128 accumulator halves while a four-warp
producer fills a three-slot BK64 pipe.  An explicit consumer-epoch barrier
protects output-tile transitions.
"""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

BM = tl.constexpr(256)
BN = tl.constexpr(256)
BH = tl.constexpr(128)
BK = tl.constexpr(64)


@triton.jit
def _persistent_producer(
    writer,
    a_desc,
    b_desc,
    pid,
    total_tiles: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            group_id = tile_id // group_width
            first_m = group_id * group_m
            actual_group_m = tl.minimum(grid_m - first_m, group_m)
            pid_in_group = tile_id % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (pid_m * BM).to(tl.int32)
            n_offset = (pid_n * BN).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                slot = writer.acquire(token)
                k_offset = k_iter * BK
                tle.gpu.copy(a_desc, slot.a, [BM, BK], [m_offset, k_offset])
                tle.gpu.copy(b_desc, slot.b, [BK, BN], [k_offset, n_offset])
                writer.commit(token)


@triton.jit
def _persistent_consumer(
    reader,
    consumer_epoch,
    bias_ptr,
    out_ptr,
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
    UNROLL_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    k_groups: tl.constexpr = k_tiles // UNROLL_K
    for tile_iter in range(tile_iters):
        tle.gpu.barrier_wait(consumer_epoch, phaseIdx=(tile_iter + 1) & 1)
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            group_id = tile_id // group_width
            first_m = group_id * group_m
            actual_group_m = tl.minimum(grid_m - first_m, group_m)
            pid_in_group = tile_id % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (pid_m * BM).to(tl.int32)
            n_offset = (pid_n * BN).to(tl.int32)
            cols_lo = n_offset + tl.arange(0, BH)
            cols_hi = n_offset + BH + tl.arange(0, BH)
            acc_lo = tl.zeros((BM, BH), tl.float32)
            acc_hi = tl.zeros((BM, BH), tl.float32)
            if HAS_BIAS:
                bias_lo = tl.load(bias_ptr + cols_lo, mask=cols_lo < N, other=0.0).to(
                    tl.float32
                )
                bias_hi = tl.load(bias_ptr + cols_hi, mask=cols_hi < N, other=0.0).to(
                    tl.float32
                )
                acc_lo += bias_lo[None, :]
                acc_hi += bias_hi[None, :]

            for k_group in range(k_groups):
                for k_inner in tl.static_range(UNROLL_K):
                    k_iter = k_group * UNROLL_K + k_inner
                    token = tile_iter * k_tiles + k_iter
                    ready = reader.wait(token)
                    b_lo = ready.slot.b.slice(0, BH, dim=1)
                    b_hi = ready.slot.b.slice(BH, BH, dim=1)
                    acc_lo = tle.gpu.wgmma(ready.slot.a, b_lo, acc_lo)
                    acc_hi = tle.gpu.wgmma(ready.slot.a, b_hi, acc_hi)
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    acc_hi = tle.gpu.wgmma_wait(0, acc_hi)
                    reader.release(token)

            rows = m_offset + tl.arange(0, BM)
            tl.store(
                out_ptr + rows[:, None] * N + cols_lo[None, :],
                acc_lo.to(out_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (cols_lo < N)[None, :],
            )
            tl.store(
                out_ptr + rows[:, None] * N + cols_hi[None, :],
                acc_hi.to(out_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (cols_hi < N)[None, :],
            )
        tle.gpu.barrier_arrive(consumer_epoch, phaseIdx=tile_iter & 1)


@triton.jit
def _persistent_kernel(
    a_desc,
    b_desc,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    UNROLL_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BM, BK],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, BK, BN],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="baddbmm_persistent_2x128",
        a=a_smem,
        b=b_smem,
    )
    consumer_epoch = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.PENDING)
    tile_iters: tl.constexpr = tl.cdiv(TOTAL_TILES, NUM_SMS)
    k_tiles: tl.constexpr = K // BK
    tle.gpu.warp_specialize(
        [
            (
                _persistent_consumer,
                (
                    pipe.reader(),
                    consumer_epoch,
                    bias_ptr,
                    out_ptr,
                    pid,
                    M,
                    N,
                    TOTAL_TILES,
                    GRID_M,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                    UNROLL_K,
                    HAS_BIAS,
                ),
            ),
            (
                _persistent_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    pid,
                    TOTAL_TILES,
                    GRID_M,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def persistent_mm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    num_sms: int = 60,
    group_m: int = 4,
    unroll_k: int = 1,
    num_slots: int = 2,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("a, b, out must be rank-2")
    m, k = a.shape
    n = b.shape[1]
    if b.shape[0] != k or tuple(out.shape) != (m, n):
        raise ValueError("incompatible shapes")
    if (m, n, k) not in (
        (16384, 7168, 1024),
        (16384, 2112, 7168),
    ):
        raise ValueError("unsupported persistent BAddBMM shape")
    tensors = (a, b, out) if bias is None else (a, b, out, bias)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("BF16 required")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.device != a.device for t in tensors):
        raise ValueError("all tensors must share one device")
    if k % 64 or (k // 64) % unroll_k:
        raise ValueError("K tiles must be divisible by unroll_k")
    if unroll_k not in (1, 2, 4):
        raise ValueError("unroll_k must be 1, 2, or 4")
    if group_m not in (1, 2, 4):
        raise ValueError("group_m must be 1, 2, or 4")
    if num_slots not in (2, 3):
        raise ValueError("num_slots must be 2 or 3")
    if bias is not None and (bias.ndim != 1 or bias.numel() != n):
        raise ValueError("bias must be an N-vector")

    a_desc = TensorDescriptor.from_tensor(a, [256, 64])
    b_desc = TensorDescriptor.from_tensor(b, [64, 256])
    grid_m = triton.cdiv(m, 256)
    grid_n = triton.cdiv(n, 256)
    total_tiles = grid_m * grid_n
    bias_arg = out if bias is None else bias
    with torch_device_fn.device(a.device):
        _persistent_kernel[(num_sms,)](
            a_desc,
            b_desc,
            bias_arg,
            out,
            M=m,
            N=n,
            K=k,
            TOTAL_TILES=total_tiles,
            GRID_M=grid_m,
            GRID_N=grid_n,
            NUM_SMS=num_sms,
            GROUP_M=group_m,
            NUM_SLOTS=num_slots,
            UNROLL_K=unroll_k,
            HAS_BIAS=bias is not None,
            num_warps=16,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
        )
    return out


__all__ = ["persistent_mm_out"]
