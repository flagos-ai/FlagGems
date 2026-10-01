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

"""Fused default baddbmm via a synthetic bias SQMMA K tile.

The MThreads backend treats a WGMMA accumulator as an opaque register layout.
Arithmetic on that value has caused spills, corruption, or deadlocks in prior
experiments.  This kernel never performs scalar arithmetic on the accumulator:
after the ordinary GEMM tiles, the producer emits one synthetic shared-memory
tile whose product is ``ones @ bias``.  A final WGMMA accumulates the vector
bias and the consumers perform the same raw BF16 store as the proven split384
MM kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

SUPPORTED_SHAPES = {(14429, 2112, 7168)}

_SYNTHETIC_A_BASE: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}


@triton.jit
def _producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    synth_a_desc,
    synth_b_desc,
    m_offset,
    n_offset,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * BLOCK_K
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(a_desc, a_slot.a, [512, BLOCK_K], [m_offset, k_offset])
        tle.gpu.copy(b_desc, b_slot.b, [BLOCK_K, 256], [k_offset, n_offset])
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)

    # An actual SQMMA creates the bias contribution, preserving the backend's
    # opaque accumulator encoding.  A_synth has one in its first K lane and
    # B_synth has bias in its first K row; hence their product is the bias
    # vector on every output row.
    token: tl.constexpr = K_TILES
    a_slot = a_writer.acquire(token)
    b_slot = b_writer.acquire(token)
    tle.gpu.copy(synth_a_desc, a_slot.a, [512, BLOCK_K], [0, 0])
    tle.gpu.copy(synth_b_desc, b_slot.b, [BLOCK_K, 256], [0, n_offset])
    a_writer.commit(token)
    b_writer.commit(token)


@triton.jit
def _consumer_top(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((256, 256), dtype=tl.float32)
    for token in range(K_TILES + 1):
        a_wait = a_reader.wait(token)
        b_wait = b_reader.wait(token)
        a_tile = a_wait.slot.a.slice(0, 256, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, b_wait.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(token)
        b_reader.release(token)
    rows = m_offset + tl.arange(0, 256)
    cols = n_offset + tl.arange(0, 256)
    ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(
        ptrs,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _consumer_bottom(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((128, 256), dtype=tl.float32)
    for token in range(K_TILES + 1):
        a_wait = a_reader.wait(token)
        b_wait = b_reader.wait(token)
        a_tile = a_wait.slot.a.slice(256, 128, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, b_wait.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(token)
        b_reader.release(token)
    rows = m_offset + 256 + tl.arange(0, 128)
    cols = n_offset + tl.arange(0, 256)
    ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(
        ptrs,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@libentry()
@triton.jit
def _synthetic_bias_split384_kernel(
    a_desc,
    b_desc,
    synth_a_desc,
    synth_b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // GRID_N
    pid_n = pid % GRID_N
    m_offset = (pid_m * 384).to(tl.int32)
    n_offset = (pid_n * 256).to(tl.int32)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, 512, BLOCK_K],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_K, 256],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="bias_tile_s384_a",
        a=a_smem,
    )
    b_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="bias_tile_s384_b",
        b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _consumer_top,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    m_offset,
                    n_offset,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _consumer_bottom,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    m_offset,
                    n_offset,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    synth_a_desc,
                    synth_b_desc,
                    m_offset,
                    n_offset,
                    K_TILES,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[168, 24],
    )


def _validate(bias, batch1, batch2, out):
    if bias.ndim != 1 or batch1.ndim != 3 or batch2.ndim != 3 or out.ndim != 3:
        raise ValueError("expected vector bias and three rank-3 tensors")
    if batch1.shape[0] != 1 or batch2.shape[0] != 1 or out.shape[0] != 1:
        raise ValueError("synthetic split384 requires B=1")
    m, k = batch1.shape[1:]
    n = batch2.shape[2]
    if batch2.shape[1] != k or (m, n, k) not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported shape {(m, n, k)}")
    if bias.numel() != n or tuple(out.shape) != (1, m, n):
        raise ValueError("bias/output shape mismatch")
    ts = (bias, batch1, batch2, out)
    if any(t.dtype != torch.bfloat16 for t in ts):
        raise TypeError("BF16 required")
    if any(not t.is_contiguous() for t in ts):
        raise ValueError("contiguous tensors required")
    if any(t.device != batch1.device for t in ts):
        raise ValueError("all tensors must share one device")
    return m, n, k


def baddbmm_synthetic_bias_split384_out(
    bias: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    out: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
    block_k: int = 32,
    num_slots: int = 3,
) -> torch.Tensor:
    """Default-coefficient fast path; caller must route other scalars away."""

    m, n, k = _validate(bias, batch1, batch2, out)
    if float(alpha) != 1.0 or float(beta) != 1.0:
        raise ValueError("synthetic bias fast path requires alpha=beta=1")
    if block_k not in (32, 64) or k % block_k:
        raise ValueError("block_k must be 32 or 64 and divide K")
    if num_slots not in (2, 3):
        raise ValueError("num_slots must be 2 or 3")
    a, b, c = batch1[0], batch2[0], out[0]
    a_desc = TensorDescriptor.from_tensor(a, [512, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, 256])
    cache_key = (str(a.device), a.dtype, block_k)
    synth_a_base = _SYNTHETIC_A_BASE.get(cache_key)
    if synth_a_base is None:
        # The descriptor's block is [512,BLOCK_K], while the backing tensor
        # has only eight K columns.  TMA zero padding supplies columns 8..K;
        # the first column is one and every other in-bounds element is zero.
        synth_a_base = torch.zeros((512, 8), dtype=a.dtype, device=a.device)
        synth_a_base[:, 0] = 1.0
        _SYNTHETIC_A_BASE[cache_key] = synth_a_base
    # Likewise, a [1,N] descriptor loaded as [BLOCK_K,256] supplies the bias
    # in K row zero and hardware zero padding for all remaining K rows.
    synth_b = bias.reshape(1, n)
    synth_a_desc = TensorDescriptor.from_tensor(synth_a_base, [512, block_k])
    synth_b_desc = TensorDescriptor.from_tensor(synth_b, [block_k, 256])
    grid_n = triton.cdiv(n, 256)
    grid = (triton.cdiv(m, 384) * grid_n,)
    with torch_device_fn.device(a.device):
        _synthetic_bias_split384_kernel[grid](
            a_desc,
            b_desc,
            synth_a_desc,
            synth_b_desc,
            c,
            M=m,
            N=n,
            GRID_N=grid_n,
            K_TILES=triton.cdiv(k, block_k),
            BLOCK_K=block_k,
            NUM_SLOTS=num_slots,
            num_warps=16,
            enable_backend_opt=True,
        )
    return out
