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

"""Exact-A/group-M=2 BAddBMM for the M14429/N7168/K1024 shape.

This keeps the current non-persistent BM384xBN256xBK32/S3 grid, static K
loop, and top16+bottom8+producer4 partitioning.  The only schedule change is
replacing padded A[512,32] with A_top[256,32] and A_bottom[128,32], combined
with B[32,256] in one three-field pipe.  One additional pipe token computes
``ones @ bias`` with SQMMA, so the vector bias is fused without performing
ordinary arithmetic on the opaque accumulator.
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


_SYNTHETIC_A_BASE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


@triton.jit
def _producer_u(
    writer,
    a_top_desc,
    a_bottom_desc,
    b_desc,
    synth_a_top_desc,
    synth_a_bottom_desc,
    synth_b_desc,
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

    # A synthetic final SQMMA contributes the vector bias.  Each synthetic A
    # row is [1, 0, ...], while synthetic B row zero contains the bias and the
    # descriptor zero-pads all remaining K rows.
    token: tl.constexpr = K_TILES
    slot = writer.acquire(token)
    tle.gpu.copy(
        synth_a_top_desc,
        slot.a_top,
        [TOP_M, BK],
        [0, 0],
    )
    tle.gpu.copy(
        synth_a_bottom_desc,
        slot.a_bottom,
        [BOTTOM_M, BK],
        [0, 0],
    )
    tle.gpu.copy(
        synth_b_desc,
        slot.b,
        [BK, BN],
        [0, n_offset],
    )
    writer.commit(token)


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
    for k_iter in tl.static_range(K_TILES + 1):
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
    for k_iter in tl.static_range(K_TILES + 1):
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
    synth_a_top_desc,
    synth_a_bottom_desc,
    synth_b_desc,
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
        name="baddbmm_c3_exact_a_group2_bias",
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
                    synth_a_top_desc,
                    synth_a_bottom_desc,
                    synth_b_desc,
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


def baddbmm_out(
    bias: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if bias.ndim != 1 or batch1.ndim != 3 or batch2.ndim != 3 or out.ndim != 3:
        raise ValueError("expected vector bias and rank-3 batch tensors")
    if batch1.shape[0] != 1 or batch2.shape[0] != 1 or out.shape[0] != 1:
        raise ValueError("exact-A path requires B=1")
    m, k = batch1.shape[1:]
    n = batch2.shape[2]
    if (m, k, n) != (14429, 1024, 7168):
        raise ValueError("unsupported exact-A BAddBMM shape")
    if batch2.shape[1] != k or tuple(out.shape) != (1, m, n):
        raise ValueError("incompatible matrix dimensions")
    if bias.numel() != n:
        raise ValueError("bias must contain N elements")
    if any(t.dtype != torch.bfloat16 for t in (bias, batch1, batch2, out)):
        raise TypeError("exact-A path requires BF16")
    if any(not t.is_contiguous() for t in (bias, batch1, batch2, out)):
        raise ValueError("exact-A path requires contiguous tensors")
    if any(t.device != batch1.device for t in (bias, batch1, batch2, out)):
        raise ValueError("all tensors must share a device")
    if k % 32:
        raise ValueError("K must be divisible by 32")

    a, b, c = batch1[0], batch2[0], out[0]
    a_top_desc = TensorDescriptor.from_tensor(a, [256, 32])
    a_bottom_desc = TensorDescriptor.from_tensor(a, [128, 32])
    b_desc = TensorDescriptor.from_tensor(b, [32, 256])
    cache_key = (str(a.device), a.dtype)
    synth_a_base = _SYNTHETIC_A_BASE.get(cache_key)
    if synth_a_base is None:
        # The descriptor zero-pads columns 8..31.  Column zero being one makes
        # the synthetic matrix product equal to the bias on every output row.
        synth_a_base = torch.zeros((256, 8), dtype=a.dtype, device=a.device)
        synth_a_base[:, 0] = 1.0
        _SYNTHETIC_A_BASE[cache_key] = synth_a_base
    synth_a_top_desc = TensorDescriptor.from_tensor(synth_a_base, [256, 32])
    synth_a_bottom_desc = TensorDescriptor.from_tensor(synth_a_base, [128, 32])
    synth_b_desc = TensorDescriptor.from_tensor(bias.reshape(1, n), [32, 256])
    grid_n = triton.cdiv(n, 256)
    grid = triton.cdiv(m, 384) * grid_n
    with torch_device_fn.device(a.device):
        _kernel[(grid,)](
            a_top_desc,
            a_bottom_desc,
            b_desc,
            synth_a_top_desc,
            synth_a_bottom_desc,
            synth_b_desc,
            c,
            m,
            n,
            c.stride(0),
            c.stride(1),
            GRID_M=triton.cdiv(m, 384),
            GRID_N=grid_n,
            K_TILES=triton.cdiv(k, 32),
            num_warps=16,
            enable_backend_opt=True,
        )
    return out


__all__ = ["baddbmm_out"]
