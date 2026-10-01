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

"""Batch-aware BMM kernels for the official core shapes."""

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
def _producer(
    writer,
    a_desc,
    b_desc,
    pid,
    M: tl.constexpr,
    K: tl.constexpr,
    total_tiles: tl.constexpr,
    tiles_per_batch: tl.constexpr,
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
            batch_id = tile_id // tiles_per_batch
            local_tile = tile_id - batch_id * tiles_per_batch
            group_id = local_tile // group_width
            first_m = group_id * group_m
            actual_group_m = tl.minimum(grid_m - first_m, group_m)
            pid_in_group = local_tile % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (batch_id * M + pid_m * BM).to(tl.int32)
            n_offset = (pid_n * BN).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                slot = writer.acquire(token)
                k_offset = k_iter * BK
                b_k_offset = batch_id * K + k_offset
                tle.gpu.copy(a_desc, slot.a, [BM, BK], [m_offset, k_offset])
                tle.gpu.copy(b_desc, slot.b, [BK, BN], [b_k_offset, n_offset])
                writer.commit(token)


@triton.jit
def _consumer(
    reader,
    consumer_epoch,
    out_ptr,
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
    tiles_per_batch: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
    UNROLL_K: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    k_groups: tl.constexpr = k_tiles // UNROLL_K
    for tile_iter in range(tile_iters):
        tle.gpu.barrier_wait(consumer_epoch, phaseIdx=(tile_iter + 1) & 1)
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            batch_id = tile_id // tiles_per_batch
            local_tile = tile_id - batch_id * tiles_per_batch
            group_id = local_tile // group_width
            first_m = group_id * group_m
            actual_group_m = tl.minimum(grid_m - first_m, group_m)
            pid_in_group = local_tile % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (pid_m * BM).to(tl.int32)
            n_offset = (pid_n * BN).to(tl.int32)
            cols_lo = n_offset + tl.arange(0, BH)
            cols_hi = n_offset + BH + tl.arange(0, BH)
            acc_lo = tl.zeros((BM, BH), tl.float32)
            acc_hi = tl.zeros((BM, BH), tl.float32)

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
            out_batch = out_ptr + batch_id * M * N
            tl.store(
                out_batch + rows[:, None] * N + cols_lo[None, :],
                acc_lo.to(out_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (cols_lo < N)[None, :],
            )
            tl.store(
                out_batch + rows[:, None] * N + cols_hi[None, :],
                acc_hi.to(out_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (cols_hi < N)[None, :],
            )
        tle.gpu.barrier_arrive(consumer_epoch, phaseIdx=tile_iter & 1)


@triton.jit
def _persistent_kernel(
    a_desc,
    b_desc,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    UNROLL_K: tl.constexpr,
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
        name="bmm_core_persistent",
        a=a_smem,
        b=b_smem,
    )
    consumer_epoch = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.PENDING)
    tile_iters: tl.constexpr = tl.cdiv(TOTAL_TILES, NUM_SMS)
    k_tiles: tl.constexpr = K // BK
    tle.gpu.warp_specialize(
        [
            (
                _consumer,
                (
                    pipe.reader(),
                    consumer_epoch,
                    out_ptr,
                    pid,
                    M,
                    N,
                    TOTAL_TILES,
                    TILES_PER_BATCH,
                    GRID_M,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                    UNROLL_K,
                ),
            ),
            (
                _producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    pid,
                    M,
                    K,
                    TOTAL_TILES,
                    TILES_PER_BATCH,
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


def bmm_core_persistent_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batch, m, k = a.shape
    n = b.shape[2]
    a_desc = TensorDescriptor.from_tensor(a.reshape(batch * m, k), [256, 64])
    b_desc = TensorDescriptor.from_tensor(b.reshape(batch * k, n), [64, 256])
    grid_m = triton.cdiv(m, 256)
    grid_n = triton.cdiv(n, 256)
    tiles_per_batch = grid_m * grid_n
    total_tiles = batch * tiles_per_batch
    with torch_device_fn.device(a.device):
        _persistent_kernel[(60,)](
            a_desc,
            b_desc,
            out,
            M=m,
            N=n,
            K=k,
            TOTAL_TILES=total_tiles,
            TILES_PER_BATCH=tiles_per_batch,
            GRID_M=grid_m,
            GRID_N=grid_n,
            NUM_SMS=60,
            GROUP_M=4,
            NUM_SLOTS=3,
            UNROLL_K=2,
            num_warps=16,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
        )
    return out


@triton.jit
def _tiled_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * grid_n
    group_id = pid // group_width
    first_m = group_id * GROUP_M
    group_size = tl.minimum(grid_m - first_m, GROUP_M)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % group_size
    pid_n = pid_in_group // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + pid_b * M * K + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + pid_b * K * N + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc=acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    out_ptrs = out_ptr + pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc)


def bmm_core_fp32_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batch, m, k = a.shape
    n = b.shape[2]
    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 64), batch)
    with torch_device_fn.device(a.device):
        _tiled_kernel[grid](
            a,
            b,
            out,
            M=m,
            N=n,
            K=k,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            GROUP_M=4,
            INPUT_PRECISION="tf32x3",
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
    return out


def bmm_core_small_out(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    block_m = 64
    block_n = 64
    grid = (triton.cdiv(384, block_m) * triton.cdiv(384, block_n), 2)
    with torch_device_fn.device(a.device):
        _tiled_kernel[grid](
            a,
            b,
            out,
            M=384,
            N=384,
            K=384,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=64,
            GROUP_M=4,
            INPUT_PRECISION="ieee",
            num_warps=4,
            num_stages=2,
            enable_backend_opt=True,
        )
    return out
