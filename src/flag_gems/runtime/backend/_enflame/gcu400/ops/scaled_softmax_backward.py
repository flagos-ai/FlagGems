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

"""GCU400 scaled softmax backward.

Math: ``dS = scale * P * (dP - rowsum(dP * P))``, with P and dP shaped
(batch, heads, query, key) and the row reduction along the last (key) dim. A contiguous tensor
(M=b*h*q, N=k):

* when N fits in a single BLOCK_N, P/dP are each read once and dS written once (one tile stays in
  registers, no second read of P/dP) -- this is the main path;
* when the row length exceeds the tile budget it degrades to two passes: pass 1 computes
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

GRID_DIM_BWD = 48
# The backward pass must keep two fp32 copies resident per tile, P and dP (2x the tile elements); measured best at
# 32768:
# at 65536, bandwidth drops from ~1.7 TB/s to ~1.4 TB/s; at 131072, DSM is simply insufficient.
ROW_TILE_ELEMS = 32768
MAX_BLOCK_N = 4096


@libentry()
@triton.jit(do_not_specialize=["M", "N", "scale"])
def scaled_softmax_backward_row_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    scale,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """N fits in a single BLOCK_N: read P/dP once, compute in registers and write straight back."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        mask = (m_offsets[:, None] < M) & n_mask[None, :]
        offsets = m_offsets[:, None] * N + n_offsets[None, :]

        p_tile = tl.load(output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dp_tile = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        row_dot = tl.sum(p_tile * dp_tile, 1)
        grad_tile = p_tile * (dp_tile - row_dot[:, None]) * scale

        tl.store(
            grad_input_ptr + offsets,
            grad_tile.to(grad_input_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit(do_not_specialize=["M", "N", "scale"])
def scaled_softmax_backward_multipass_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    scale,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Row length exceeds BLOCK_N: compute the per-row D first, then recompute dS in a second pass."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    n_offsets = tl.arange(0, BLOCK_N)

    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        row_base = m_offsets[:, None] * N

        row_dot = tl.zeros([BLOCK_M], tl.float32)
        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            mask = m_mask[:, None] & (n_index[None, :] < N)
            p_tile = tl.load(
                output_ptr + row_base + n_index[None, :], mask=mask, other=0.0
            )
            dp_tile = tl.load(
                grad_output_ptr + row_base + n_index[None, :], mask=mask, other=0.0
            )
            row_dot += tl.sum(p_tile.to(tl.float32) * dp_tile.to(tl.float32), 1)

        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            mask = m_mask[:, None] & (n_index[None, :] < N)
            offsets = row_base + n_index[None, :]
            p_tile = tl.load(output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            dp_tile = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_tile = p_tile * (dp_tile - row_dot[:, None]) * scale
            tl.store(
                grad_input_ptr + offsets,
                grad_tile.to(grad_input_ptr.dtype.element_ty),
                mask=mask,
            )


def _grid_dim_for(total_elems):
    """Picks the persistent program count from the total element count.

    Program startup is expensive on GCU (measured: (1,4,64,64) with 256 programs takes 25us,
    while 16~48 programs take only 5.8~8.1us), so small problems need a narrower grid.
    """
    if total_elems <= 65536:
        return 16
    if total_elems <= 262144:
        return 32
    return GRID_DIM_BWD


def _tile_shape(num_rows, row_length, grid_dim):
    """Derives (BLOCK_M, BLOCK_N) from rows / grid width and the tile element budget.

    BLOCK_M must be a power of two (tl.arange inside the kernel), so take the largest power of
    two not exceeding cdiv(num_rows, grid_dim), guaranteeing all grid_dim programs are filled.
    """
    block_n = min(triton.next_power_of_2(row_length), MAX_BLOCK_N)
    target_rows = max(1, triton.cdiv(num_rows, grid_dim))
    block_m = 1 << (target_rows.bit_length() - 1)
    block_m = min(block_m, ROW_TILE_ELEMS // block_n, triton.next_power_of_2(num_rows))
    return max(1, block_m), block_n


def scaled_softmax_backward(
    grad_output: torch.Tensor, softmax_results: torch.Tensor, scale_factor: float
):
    logger.debug("GEMS_ENFLAME SCALED_SOFTMAX_BACKWARD")
    assert grad_output.dim() == 4, "expected 4D tensor"
    assert softmax_results.dim() == 4, "expected 4D tensor"
    assert grad_output.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"
    assert softmax_results.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"

    grad_output = (
        grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    )
    softmax_results = (
        softmax_results
        if softmax_results.is_contiguous()
        else softmax_results.contiguous()
    )

    batch_size, attn_heads, query_seq_len, key_seq_len = softmax_results.shape
    num_rows = batch_size * attn_heads * query_seq_len

    grad_input = torch.empty_like(grad_output)

    total_elems = num_rows * key_seq_len
    grid_dim = _grid_dim_for(total_elems)
    block_m, block_n = _tile_shape(num_rows, key_seq_len, grid_dim)
    num_programs = min(grid_dim, triton.cdiv(num_rows, block_m))

    if key_seq_len <= block_n:
        scaled_softmax_backward_row_kernel[(num_programs,)](
            grad_input,
            grad_output,
            softmax_results,
            scale_factor,
            num_rows,
            key_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    else:
        scaled_softmax_backward_multipass_kernel[(num_programs,)](
            grad_input,
            grad_output,
            softmax_results,
            scale_factor,
            num_rows,
            key_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return grad_input
