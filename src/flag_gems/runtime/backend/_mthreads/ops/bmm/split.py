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

"""Exact-shape split-M MThreads BMM kernel.

This keeps the current non-persistent BM384xBN256xBK32/S3 grid, static K
loop, and top16+bottom8+producer4 partitioning.  The only schedule change is
replacing padded A[512,32] with A_top[256,32] and A_bottom[128,32], combined
with B[32,256] in one three-field pipe.
"""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

BM = tl.constexpr(384)
BN = tl.constexpr(256)
BK = tl.constexpr(32)
TOP_M = tl.constexpr(256)
BOTTOM_M = tl.constexpr(128)
SLOTS = tl.constexpr(3)


@triton.jit
def _producer_u(
    writer,
    a_top_desc,
    a_bottom_desc,
    b_desc,
    m_offset,
    bottom_offset,
    n_offset,
    K_TILES: tl.constexpr,
):
    for k_iter in tl.static_range(K_TILES):
        k_offset = k_iter * BK
        slot = writer.acquire(k_iter)
        tle.gpu.copy(
            a_top_desc,
            slot.a_top,
            [TOP_M, BK],
            [m_offset, k_offset],
        )
        tle.gpu.copy(
            a_bottom_desc,
            slot.a_bottom,
            [BOTTOM_M, BK],
            [bottom_offset, k_offset],
        )
        tle.gpu.copy(
            b_desc,
            slot.b,
            [BK, BN],
            [k_offset, n_offset],
        )
        writer.commit(k_iter)


@triton.jit
def _consumer_top_u(
    reader,
    out_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((TOP_M, BN), dtype=tl.float32)
    for k_iter in tl.static_range(K_TILES):
        ready = reader.wait(k_iter)
        accumulator = tle.gpu.wgmma(
            ready.slot.a_top,
            ready.slot.b,
            accumulator,
        )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(k_iter)
    rows = m_offset + tl.arange(0, TOP_M)
    cols = n_offset + tl.arange(0, BN)
    tl.store(
        out_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _consumer_bottom_u(
    reader,
    out_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((BOTTOM_M, BN), dtype=tl.float32)
    for k_iter in tl.static_range(K_TILES):
        ready = reader.wait(k_iter)
        accumulator = tle.gpu.wgmma(
            ready.slot.a_bottom,
            ready.slot.b,
            accumulator,
        )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(k_iter)
    rows = m_offset + TOP_M + tl.arange(0, BOTTOM_M)
    cols = n_offset + tl.arange(0, BN)
    tl.store(
        out_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _kernel(
    a_top_desc,
    a_bottom_desc,
    b_desc,
    out_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid = ext.program_id(0)
    group_width: tl.constexpr = 2 * GRID_N
    group_id = pid // group_width
    first_m = group_id * 2
    actual_group_m = min(GRID_M - first_m, 2)
    inner = pid % group_width
    pid_m = first_m + inner % actual_group_m
    pid_n = inner // actual_group_m
    m_offset = (pid_m * BM).to(tl.int32)
    n_offset = (pid_n * BN).to(tl.int32)
    bottom_offset = tl.where(m_offset + TOP_M < M, m_offset + TOP_M, 0).to(tl.int32)
    a_top_smem = tle.gpu.alloc(
        [SLOTS, TOP_M, BK],
        dtype=a_top_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_bottom_smem = tle.gpu.alloc(
        [SLOTS, BOTTOM_M, BK],
        dtype=a_bottom_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [SLOTS, BK, BN],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=SLOTS,
        scope="cta",
        name="bmm_c3_nonpersistent_exact_a",
        a_top=a_top_smem,
        a_bottom=a_bottom_smem,
        b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _consumer_top_u,
                (
                    pipe.reader(),
                    out_ptr,
                    m_offset,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _consumer_bottom_u,
                (
                    pipe.reader(),
                    out_ptr,
                    m_offset,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _producer_u,
                (
                    pipe.writer(),
                    a_top_desc,
                    a_bottom_desc,
                    b_desc,
                    m_offset,
                    bottom_offset,
                    n_offset,
                    K_TILES,
                ),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[168, 24],
    )


def bmm_split_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    a_top_desc = TensorDescriptor.from_tensor(a, [256, 32])
    a_bottom_desc = TensorDescriptor.from_tensor(a, [128, 32])
    b_desc = TensorDescriptor.from_tensor(b, [32, 256])
    grid_n = triton.cdiv(n, 256)
    grid = triton.cdiv(m, 384) * grid_n
    with torch_device_fn.device(a.device):
        _kernel[(grid,)](
            a_top_desc,
            a_bottom_desc,
            b_desc,
            out,
            m,
            n,
            out.stride(0),
            out.stride(1),
            GRID_M=triton.cdiv(m, 384),
            GRID_N=grid_n,
            K_TILES=triton.cdiv(k, 32),
            num_warps=16,
            enable_backend_opt=True,
        )
    return out
