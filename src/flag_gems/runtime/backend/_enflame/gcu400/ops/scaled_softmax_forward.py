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

"""GCU400 scaled softmax forward: ``softmax(input * scale_factor, dim=-1)``.

Input layout is (batch, heads, query, key) with key last, so a contiguous tensor is directly a
row-wise softmax over (M=b*h*q, N=k); this reuses enflame softmax forward row-wise dataflow:

* when the row length N fits in a single BLOCK_N, one row (or several) occupies one tile, the
  input is read once and the output written once (max/exp/sum stay in registers) -- the main path
  for these test shapes 64~4096;
* when the row length exceeds the tile budget: pass 1 computes max/sum online, pass 2 recomputes exp and writes back.
The total program count is fixed at ``GRID_DIM_FWD`` and each program strides over several rows
with ``tl.range``, avoiding the CTA-count-driven startup and scheduling overhead.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# With 24 SiPs on GCU400, 48 programs is the measured bandwidth sweet spot (same criterion as the softmax forward).
GRID_DIM_FWD = 48
# Element budget for a single tile: BLOCK_M * BLOCK_N must not exceed it.
# Measured on GCU400 (fp16, do_bench): 65536 elements/block is 1.3~1.45x faster than 32768
# ((16,16,1024,1024) 0.81ms vs 1.18ms), while 131072 fails to compile due to insufficient DSM.
ROW_TILE_ELEMS = 65536
# Maximum number of columns covered by a single segment within a row, to avoid overly long rows blowing up the reduction
# width.
MAX_BLOCK_N = 4096
# exp is implemented with exp2, requiring multiplication by log2(e); it must be instantiated as a constexpr to be
# referenced inside the kernel.
LOG2E = tl.constexpr(1.4426950408889634)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "scale"])
def scaled_softmax_forward_row_kernel(
    out_ptr,
    in_ptr,
    scale,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """N fits in a single BLOCK_N: one tile per row, single read + single write."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        mask = (m_offsets[:, None] < M) & n_mask[None, :]
        offsets = m_offsets[:, None] * N + n_offsets[None, :]

        # Perform scaling in the fp32 domain to avoid extra rounding from multiplying scale in fp16.
        inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        inp = inp * scale

        row_max = tl.max(inp, 1)
        exp_val = tl.exp2((inp - row_max[:, None]) * LOG2E)
        row_sum = tl.sum(exp_val, 1)
        out = exp_val * (1.0 / row_sum)[:, None]

        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "scale"])
def scaled_softmax_forward_multipass_kernel(
    out_ptr,
    in_ptr,
    scale,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Row length exceeds BLOCK_N: pass 1 finds max/sum online, pass 2 recomputes and writes out."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    n_offsets = tl.arange(0, BLOCK_N)

    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        row_base = m_offsets[:, None] * N

        row_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        row_sum = tl.zeros([BLOCK_M], tl.float32)

        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            mask = m_mask[:, None] & (n_index[None, :] < N)
            inp = tl.load(
                in_ptr + row_base + n_index[None, :], mask=mask, other=-float("inf")
            )
            inp = inp.to(tl.float32) * scale

            m_new = tl.maximum(row_max, tl.max(inp, 1))
            # When the entire row is -inf, keep the state to avoid NaN from -inf - (-inf).
            all_neg_inf = m_new == float("-inf")
            shifted = tl.sum(tl.exp2((inp - m_new[:, None]) * LOG2E), 1)
            row_sum = tl.where(
                all_neg_inf,
                row_sum,
                row_sum * tl.exp2((row_max - m_new) * LOG2E) + shifted,
            )
            row_max = m_new

        inv_sum = 1.0 / row_sum
        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            mask = m_mask[:, None] & (n_index[None, :] < N)
            inp = tl.load(
                in_ptr + row_base + n_index[None, :], mask=mask, other=-float("inf")
            ).to(tl.float32)
            exp_val = tl.exp2((inp * scale - row_max[:, None]) * LOG2E)
            out = exp_val * inv_sum[:, None]
            tl.store(
                out_ptr + row_base + n_index[None, :],
                out.to(out_ptr.dtype.element_ty),
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
    return GRID_DIM_FWD


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


def scaled_softmax_forward(input_t: torch.Tensor, scale_factor: float):
    logger.debug("GEMS_ENFLAME SCALED_SOFTMAX_FORWARD")
    assert input_t.dim() == 4, "expected 4D tensor"
    assert input_t.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"

    batch_size, attn_heads, query_seq_len, key_seq_len = input_t.shape
    assert key_seq_len <= 16384, "Key sequence length must be 16384 or less"
    assert key_seq_len % 8 == 0, "Key sequence length must be divisible by 8"
    assert query_seq_len > 1, "Query sequence length must be greater than 1"

    # The row-based data flow requires the last dimension to be contiguous; when (b,h,q,k) is contiguous, the row index
    # is simply (b*H+h)*Q+q and can be flattened directly.
    src = input_t if input_t.is_contiguous() else input_t.contiguous()
    output_t = torch.empty_like(src)

    num_rows = batch_size * attn_heads * query_seq_len
    total_elems = num_rows * key_seq_len
    grid_dim = _grid_dim_for(total_elems)
    block_m, block_n = _tile_shape(num_rows, key_seq_len, grid_dim)
    num_programs = min(grid_dim, triton.cdiv(num_rows, block_m))

    if key_seq_len <= block_n:
        scaled_softmax_forward_row_kernel[(num_programs,)](
            output_t,
            src,
            scale_factor,
            num_rows,
            key_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    else:
        scaled_softmax_forward_multipass_kernel[(num_programs,)](
            output_t,
            src,
            scale_factor,
            num_rows,
            key_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return output_t
