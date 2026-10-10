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

"""Fused BAddBMM for the shared M15/N160/K1024 half-precision case."""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

BLOCK_M = tl.constexpr(64)
BLOCK_N = tl.constexpr(64)
K_FRAGMENT = tl.constexpr(64)


@triton.jit
def _producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    pid_b,
    pid_n,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_row = (pid_b * 15).to(tl.int32)
    n_offset = (pid_n * BLOCK_N).to(tl.int32)
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        k_offset = (k_iter * BLOCK_K).to(tl.int32)
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(a_desc, a_slot.a, (BLOCK_M, BLOCK_K), (a_row, k_offset))
        b_row = (pid_b * 1024 + k_offset).to(tl.int32)
        tle.gpu.copy(b_desc, b_slot.b, (BLOCK_K, BLOCK_N), (b_row, n_offset))
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit(do_not_specialize=["alpha", "beta"])
def _consumer(
    a_reader,
    b_reader,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
    pid_b,
    pid_n,
    K_TILES: tl.constexpr,
    K_PARTS: tl.constexpr,
    UNROLL: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.range(
        0,
        K_TILES,
        num_stages=1,
        loop_unroll_factor=UNROLL,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        for k_part in tl.static_range(K_PARTS):
            a_fragment = a_wait.slot.a.slice(k_part * K_FRAGMENT, K_FRAGMENT, dim=1)
            b_fragment = b_wait.slot.b.slice(k_part * K_FRAGMENT, K_FRAGMENT, dim=0)
            accumulator = tle.gpu.wgmma(a_fragment, b_fragment, accumulator)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    rows = tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + cols, mask=cols < 160, other=0.0).to(tl.float32)
    result = alpha * accumulator + beta * bias[None, :]
    tl.store(
        out_ptr + pid_b * (15 * 160) + rows[:, None] * 160 + cols[None, :],
        result.to(out_ptr.dtype.element_ty),
        mask=(rows[:, None] < 15) & (cols[None, :] < 160),
    )


@triton.jit
def _kernel(
    a_desc,
    b_desc,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    a_smem = tle.gpu.alloc(
        (NUM_SLOTS, BLOCK_M, BLOCK_K),
        dtype=a_desc.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (NUM_SLOTS, BLOCK_K, BLOCK_N),
        dtype=b_desc.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="baddbmm_smallm_a",
        a=a_smem,
    )
    b_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="baddbmm_smallm_b",
        b=b_smem,
    )
    k_tiles: tl.constexpr = 1024 // BLOCK_K
    k_parts: tl.constexpr = BLOCK_K // K_FRAGMENT
    tle.gpu.warp_specialize(
        [
            (
                _consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    bias_ptr,
                    out_ptr,
                    alpha,
                    beta,
                    pid_b,
                    pid_n,
                    k_tiles,
                    k_parts,
                    UNROLL,
                ),
            ),
            (
                _producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    pid_b,
                    pid_n,
                    k_tiles,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def baddbmm_smallm_out(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    alpha,
    beta,
) -> torch.Tensor:
    """Launch the validated BM64/BN64/BK128 three-slot schedule."""
    tensors = (bias, a, b, out)
    if tuple(a.shape) != (4, 15, 1024):
        raise ValueError("a must have shape (4,15,1024)")
    if tuple(b.shape) != (4, 1024, 160):
        raise ValueError("b must have shape (4,1024,160)")
    if tuple(bias.shape) != (160,) or tuple(out.shape) != (4, 15, 160):
        raise ValueError("bias/output shape mismatch")
    if any(t.dtype not in (torch.float16, torch.bfloat16) for t in tensors):
        raise TypeError("FP16 or BF16 tensors required")
    if any(t.dtype != a.dtype for t in tensors):
        raise TypeError("all tensors must share one dtype")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.device != a.device for t in tensors):
        raise ValueError("all tensors must share one device")

    a_desc = TensorDescriptor(a, [4 * 15, 1024], [1024, 1], [64, 128])
    b_desc = TensorDescriptor(b, [4 * 1024, 160], [160, 1], [128, 64])
    with torch_device_fn.device(a.device):
        _kernel[(3, 4)](
            a_desc,
            b_desc,
            bias,
            out,
            float(alpha),
            float(beta),
            BLOCK_K=128,
            NUM_SLOTS=3,
            UNROLL=2,
            num_warps=8,
            num_stages=3,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
        )
    return out


__all__ = ["baddbmm_smallm_out"]
