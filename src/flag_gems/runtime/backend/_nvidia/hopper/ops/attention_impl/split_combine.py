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

"""Dynamic per-request Split-KV reduction for Hopper persistent attention."""

import triton
import triton.language as tl

from flag_gems.utils import libentry

from .common import _ragged_persistent_tile_coords, _split_kv_count
from .scheduling import PersistentSchedulingHeuristics


@libentry()
@triton.jit(
    do_not_specialize=[
        "batch_size",
        "total_q",
    ]
)
def _combine_persistent_split_kv_kernel(
    o_ptr,
    softmax_lse_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    scheduler_counter_ptr,
    seqused_k_ptr,
    cu_seqlens_q_ptr,
    o_row_stride,
    o_head_stride,
    batch_size,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    total_q,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    COMPACT_RAGGED: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    if COMPACT_RAGGED:
        m_block, bid, hid, work_valid = _ragged_persistent_tile_coords(
            tl.program_id(0),
            cu_seqlens_q_ptr,
            batch_size,
            num_heads,
            BLOCK_M,
            1,
            0,
        )
    else:
        m_block = tl.program_id(0)
        bid = tl.program_id(1)
        hid = tl.program_id(2)
        work_valid = True

    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    q_len = q_eos - q_bos
    k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    split_count = _split_kv_count(k_len, MAX_SPLITS)

    rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    global_rows = q_bos + rows
    valid_rows = (rows < q_len) & work_valid
    cols = tl.arange(0, BLOCK_K)

    lse_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
    for split_id in tl.static_range(0, MAX_SPLITS):
        active = split_id < split_count
        lse_base = split_id * num_heads * total_q + hid * total_q
        lse_i = tl.load(
            partial_lse_ptr + lse_base + global_rows,
            mask=valid_rows & active,
            other=float("-inf"),
        )
        lse_max = tl.maximum(lse_max, lse_i)

    lse_safe = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    weight_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
    for split_id in tl.static_range(0, MAX_SPLITS):
        active = split_id < split_count
        lse_base = split_id * num_heads * total_q + hid * total_q
        lse_i = tl.load(
            partial_lse_ptr + lse_base + global_rows,
            mask=valid_rows & active,
            other=float("-inf"),
        )
        valid_part = active & (lse_i != float("-inf"))
        weight = tl.where(valid_part, tl.exp(lse_i - lse_safe), 0.0)
        part_base = split_id * num_heads * total_q * head_dim + hid * total_q * head_dim
        part = tl.load(
            partial_out_ptr
            + part_base
            + global_rows[:, None] * head_dim
            + cols[None, :],
            mask=valid_rows[:, None] & active & (cols[None, :] < head_dim),
            other=0.0,
        )
        acc += part * weight[:, None]
        weight_sum += weight

    invalid = (weight_sum == 0.0) | (weight_sum != weight_sum)
    acc *= tl.where(invalid, 1.0, 1.0 / weight_sum)[:, None]

    o_ptrs = (
        o_ptr
        + q_bos * o_row_stride
        + rows[:, None] * o_row_stride
        + hid * o_head_stride
        + cols[None, :]
    )
    tl.store(
        o_ptrs,
        acc.to(o_ptr.dtype.element_ty),
        mask=valid_rows[:, None] & (cols[None, :] < head_dim),
    )
    if STORE_LSE:
        lse = tl.where(invalid, float("inf"), lse_max + tl.log(weight_sum))
        tl.store(
            softmax_lse_ptr + hid * total_q + global_rows,
            lse,
            mask=valid_rows,
        )
    if tl.program_id(0) == 0 and tl.program_id(1) == 0 and tl.program_id(2) == 0:
        # The combine launch is stream-ordered after every producer CTA.  Reset
        # here so the next eager invocation or graph replay starts at ticket 0.
        tl.store(scheduler_counter_ptr, 0)


def combine_persistent_split_kv(
    output,
    softmax_lse,
    partial_out,
    partial_lse,
    scheduler_counter,
    seqused_k,
    cu_seqlens_q,
    *,
    batch_size,
    num_heads,
    max_seqlen_q,
    head_dim,
    total_q,
    max_splits,
    store_lse,
):
    """Combine normalized partial O and natural-log LSE tensors."""

    plan = PersistentSchedulingHeuristics.combine_launch_plan(
        max_seqlen_q=max_seqlen_q,
        head_dim=head_dim,
        total_q=total_q,
        batch_size=batch_size,
    )
    grid = (
        (plan.compact_mblocks * num_heads, 1, 1)
        if plan.compact_ragged
        else (triton.cdiv(max_seqlen_q, plan.block_m), batch_size, num_heads)
    )
    return _combine_persistent_split_kv_kernel[grid](
        output,
        softmax_lse,
        partial_out,
        partial_lse,
        scheduler_counter,
        seqused_k,
        cu_seqlens_q,
        output.stride(-3),
        output.stride(-2),
        batch_size,
        num_heads,
        head_dim,
        total_q,
        BLOCK_M=plan.block_m,
        BLOCK_K=plan.block_k,
        MAX_SPLITS=max_splits,
        COMPACT_RAGGED=plan.compact_ragged,
        STORE_LSE=store_lse,
        num_warps=8,
    )
