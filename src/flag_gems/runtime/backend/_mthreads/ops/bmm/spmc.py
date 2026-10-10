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

"""Named-reader SPMC MThreads BMM kernel.

Schedule: BM256 x BN320 x BK64, two pipeline slots and 60 persistent CTAs.
The B payload is split into power-of-two 256- and 64-column fields.  A
16-warp reader computes the 256-column field, a 4-warp reader computes the
64-column field, and four producer warps issue TMA copies.

This source requires segmented named-reader SPMC lowering from the companion
FlagTree compiler. The two consumers own independent reader endpoints, and the
pipe empty barrier waits for both partitions before the producer can reuse a
slot.
"""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

BLOCK_M = tl.constexpr(256)
BLOCK_N = tl.constexpr(320)
BLOCK_N_MAIN = tl.constexpr(256)
BLOCK_N_TAIL = tl.constexpr(64)
BLOCK_K = tl.constexpr(64)


@triton.jit
def _producer(
    writer,
    a_desc,
    b_main_desc,
    b_tail_desc,
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
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                k_offset = k_iter * BLOCK_K
                slot = writer.acquire(token)
                tle.gpu.copy(a_desc, slot.a, [BLOCK_M, BLOCK_K], [m_offset, k_offset])
                tle.gpu.copy(
                    b_main_desc,
                    slot.b_main,
                    [BLOCK_K, BLOCK_N_MAIN],
                    [k_offset, n_offset],
                )
                tle.gpu.copy(
                    b_tail_desc,
                    slot.b_tail,
                    [BLOCK_K, BLOCK_N_TAIL],
                    [k_offset, n_offset + BLOCK_N_MAIN],
                )
                writer.commit(token)


@triton.jit
def _tail_consumer(
    reader,
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
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            columns = n_offset + BLOCK_N_MAIN + tl.arange(0, BLOCK_N_TAIL)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N_TAIL), tl.float32)

            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = reader.wait(token)
                accumulator = tle.gpu.wgmma(
                    ready.slot.a, ready.slot.b_tail, accumulator
                )
                accumulator = tle.gpu.wgmma_wait(0, accumulator)
                reader.release(token)

            rows = m_offset + tl.arange(0, BLOCK_M)
            tl.store(
                out_ptr + rows[:, None] * N + columns[None, :],
                accumulator.to(out_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (columns < N)[None, :],
            )


@triton.jit
def _main_consumer(
    reader,
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
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            columns_lo = n_offset + tl.arange(0, 128)
            columns_hi = n_offset + 128 + tl.arange(0, 128)
            accumulator_lo = tl.zeros((BLOCK_M, 128), tl.float32)
            accumulator_hi = tl.zeros((BLOCK_M, 128), tl.float32)

            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = reader.wait(token)
                b_lo = ready.slot.b_main.slice(0, 128, dim=1)
                b_hi = ready.slot.b_main.slice(128, 128, dim=1)
                accumulator_lo = tle.gpu.wgmma(ready.slot.a, b_lo, accumulator_lo)
                accumulator_hi = tle.gpu.wgmma(ready.slot.a, b_hi, accumulator_hi)
                accumulator_lo = tle.gpu.wgmma_wait(0, accumulator_lo)
                accumulator_hi = tle.gpu.wgmma_wait(0, accumulator_hi)
                reader.release(token)

            rows = m_offset + tl.arange(0, BLOCK_M)
            tl.store(
                out_ptr + rows[:, None] * N + columns_lo[None, :],
                accumulator_lo.to(out_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (columns_lo < N)[None, :],
            )
            tl.store(
                out_ptr + rows[:, None] * N + columns_hi[None, :],
                accumulator_hi.to(out_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (columns_hi < N)[None, :],
            )


@triton.jit
def _kernel(
    a_desc,
    b_main_desc,
    b_tail_desc,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc(
        [2, BLOCK_M, BLOCK_K],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_main_smem = tle.gpu.alloc(
        [2, BLOCK_K, BLOCK_N_MAIN],
        dtype=b_main_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_tail_smem = tle.gpu.alloc(
        [2, BLOCK_K, BLOCK_N_TAIL],
        dtype=b_tail_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=2,
        scope="cta",
        name="bmm_c4_spmc_bm256_bn320",
        readers=("tail", "main"),
        a=a_smem,
        b_main=b_main_smem,
        b_tail=b_tail_smem,
    )
    tile_iters: tl.constexpr = tl.cdiv(TOTAL_TILES, NUM_SMS)
    k_tiles: tl.constexpr = K // BLOCK_K

    tle.gpu.warp_specialize(
        [
            (
                _tail_consumer,
                (
                    pipe.reader("tail"),
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
                ),
            ),
            (
                _main_consumer,
                (
                    pipe.reader("main"),
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
                ),
            ),
            (
                _producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_main_desc,
                    b_tail_desc,
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
        worker_num_warps=[16, 4],
        worker_num_regs=[128, 24],
    )


def bmm_spmc_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    a_desc = TensorDescriptor.from_tensor(a, [256, 64])
    b_main_desc = TensorDescriptor.from_tensor(b, [64, 256])
    b_tail_desc = TensorDescriptor.from_tensor(b, [64, 64])
    grid_m = triton.cdiv(m, 256)
    grid_n = triton.cdiv(n, 320)
    total_tiles = grid_m * grid_n
    with torch_device_fn.device(a.device):
        _kernel[(60,)](
            a_desc,
            b_main_desc,
            b_tail_desc,
            out,
            M=m,
            N=n,
            K=k,
            TOTAL_TILES=total_tiles,
            GRID_M=grid_m,
            GRID_N=grid_n,
            NUM_SMS=60,
            GROUP_M=2,
            num_warps=4,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
        )
    return out
