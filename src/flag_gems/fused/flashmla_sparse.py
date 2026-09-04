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

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _prune_flash_mla_sparse_configs(configs, named_args, **kwargs):
    del kwargs
    block_heads = 16 if named_args["HQ"] < 64 else 64
    return [config for config in configs if config.kwargs["BH"] == block_heads]


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BH": 4}, num_warps=4, num_stages=2),
        triton.Config({"BK": 64, "BH": 8}, num_warps=4, num_stages=3),
        triton.Config({"BK": 128, "BH": 4}, num_warps=8, num_stages=2),
        triton.Config({"BK": 128, "BH": 8}, num_warps=8, num_stages=3),
    ],
    key=[
        "SQ",
        "SKV",
        "TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
        "RETURN_STATS",
    ],
)
@triton.jit
def triton_flash_mla_sparse_fwd_hq4(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    RETURN_STATS: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """Four-head specialization with heads on the MMA N dimension."""

    i_sq = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BH)
    mask_h = offs_h < 4
    offs_d = tl.arange(0, 256)
    offs_t = tl.arange(0, BK)

    q_base = q + i_sq * stride_qm
    indices_base = indices + i_sq * stride_tm
    output_base = output + i_sq * stride_om
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(
        q_ptr, mask=mask_h[:, None], other=0.0, eviction_policy="evict_first"
    )
    q_blk1 = tl.load(
        q_ptr + 256,
        mask=mask_h[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([BH], dtype=tl.float32)
    acc0 = tl.zeros([256, BH], dtype=tl.float32)
    acc1 = tl.zeros([256, BH], dtype=tl.float32)

    topk_len = tl.load(topk_length + i_sq) if HAVE_TOPK_LENGTH else TOPK
    topk_len = tl.minimum(tl.maximum(topk_len, 0), TOPK)
    num_k_blocks = tl.cdiv(topk_len, BK)
    for block_index in range(num_k_blocks):
        topk_offsets = block_index * BK + offs_t
        topk_mask = topk_offsets < topk_len
        kv_ids = tl.load(indices_base + topk_offsets, mask=topk_mask, other=-1)
        valid_ids = (kv_ids >= 0) & (kv_ids < SKV)
        safe_kv_ids = tl.where(valid_ids, kv_ids, 0)

        kv_ptr = kv + offs_d[:, None] + safe_kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
        kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".cg")

        # Keeping heads on N wastes only the hardware N=8 minimum.  Putting
        # four heads on M would execute the M=16 minimum in both dot products.
        qk_transposed = tl.dot(kv_blk0.trans(), q_blk0.trans(), out_dtype=tl.float32)
        qk_transposed = tl.dot(
            kv_blk1.trans(),
            q_blk1.trans(),
            qk_transposed,
            out_dtype=tl.float32,
        )
        qk_transposed *= sm_scale
        qk_transposed = tl.where(
            valid_ids[:, None] & mask_h[None, :],
            qk_transposed,
            float("-inf"),
        )

        new_max = tl.maximum(max_log, tl.max(qk_transposed, axis=0))
        max_for_exp = tl.where(new_max == float("-inf"), 0.0, new_max)
        exp_qk_transposed = tl.math.exp(qk_transposed - max_for_exp[None, :])
        block_sum = tl.sum(exp_qk_transposed, axis=0)
        alpha = tl.math.exp(max_log - max_for_exp)
        sum_exp = sum_exp * alpha + block_sum

        weights = exp_qk_transposed.to(tl.bfloat16)
        acc0 = tl.dot(
            kv_blk0,
            weights,
            acc0 * alpha[None, :],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            kv_blk1,
            weights,
            acc1 * alpha[None, :],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_row = max_log != float("-inf")
    if RETURN_STATS:
        max_logits_base = max_logits + i_sq * stride_mm
        lse_base = lse + i_sq * stride_lm
        tl.store(max_logits_base + offs_h, max_log, mask=mask_h)
        original_lse = max_log + tl.math.log(sum_exp)
        lse_values = tl.where(valid_row, original_lse, float("inf"))
        tl.store(lse_base + offs_h, lse_values, mask=mask_h)

    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + offs_h, mask=mask_h, other=float("-inf"))
        denominator = sum_exp + tl.math.exp(sink - max_log)
    else:
        denominator = sum_exp
    denominator = tl.where(valid_row, denominator, 1.0)
    factor = 1.0 / denominator
    out_vals0 = tl.where(valid_row[None, :], acc0 * factor[None, :], 0.0)
    out_vals1 = tl.where(valid_row[None, :], acc1 * factor[None, :], 0.0)
    output_ptr = output_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(output_ptr, out_vals0.trans().to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(
        output_ptr + 256,
        out_vals1.trans().to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.jit
def _flash_mla_sparse_hq4_pair(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    output,
    first_row,
    pair_mode,
    first_topk_length,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    SKV,
    sm_scale: tl.constexpr,
    WINDOW: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    PAIR_CACHE_CA: tl.constexpr,
    BK: tl.constexpr,
):
    """Evaluate two four-head rows from one small union of their KV IDs."""

    first_length = tl.load(topk_length + first_row)
    second_length = tl.load(topk_length + first_row + 1)
    offsets_n = tl.arange(0, 8)
    query_in_pair = offsets_n // 4
    head = offsets_n % 4
    query_row = first_row + query_in_pair
    column_length = tl.where(query_in_pair == 0, first_length, second_length)
    is_mode2 = pair_mode == 2
    is_mode3 = pair_mode == 3
    is_mode4 = pair_mode == 4
    second_topk_length = first_topk_length + 1
    union_length = second_length + tl.where(is_mode2 | is_mode4, 1, 0)

    offsets_d = tl.arange(0, 256)
    q_ptr = (
        q
        + query_row[:, None] * stride_qm
        + head[:, None] * stride_qh
        + offsets_d[None, :]
    )
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + 256, eviction_policy="evict_first")

    offsets_t = tl.arange(0, BK)
    max_log = tl.full([8], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([8], dtype=tl.float32)
    acc0 = tl.zeros([256, 8], dtype=tl.float32)
    acc1 = tl.zeros([256, 8], dtype=tl.float32)

    for block_index in range(tl.cdiv(union_length, BK)):
        union_position = block_index * BK + offsets_t
        union_mask = union_position < union_length

        # Modes 1 and 3 use row 1 directly.  A saturated, unchanged top-k
        # (mode 2) uses row 0 plus row 1's new window tail.  A saturated,
        # growing top-k (mode 4) uses row 1 plus row 0's expired window head.
        mode2_source_row = tl.where(
            union_position < first_length, first_row, first_row + 1
        )
        mode2_source_position = tl.where(
            union_position < first_length,
            union_position,
            first_length - 1,
        )
        mode4_source_row = tl.where(
            union_position == second_topk_length, first_row, first_row + 1
        )
        mode4_source_position = tl.where(
            union_position < second_topk_length,
            union_position,
            tl.where(
                union_position == second_topk_length,
                first_topk_length,
                union_position - 1,
            ),
        )
        source_row = tl.where(
            is_mode2,
            mode2_source_row,
            tl.where(is_mode4, mode4_source_row, first_row + 1),
        )
        source_position = tl.where(
            is_mode2,
            mode2_source_position,
            tl.where(is_mode4, mode4_source_position, union_position),
        )
        kv_ids = tl.load(
            indices + source_row * stride_tm + source_position,
            mask=union_mask,
            other=-1,
        )
        valid_ids = union_mask & (kv_ids >= 0) & (kv_ids < SKV)
        safe_kv_ids = tl.where(valid_ids, kv_ids, 0)
        kv_ptr = kv + offsets_d[:, None] + safe_kv_ids[None, :] * stride_kvn
        if PAIR_CACHE_CA:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".ca")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".ca")
        else:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".cg")

        scores = tl.dot(kv_blk0.trans(), q_blk0.trans(), out_dtype=tl.float32)
        scores = tl.dot(kv_blk1.trans(), q_blk1.trans(), scores, out_dtype=tl.float32)
        scores *= sm_scale

        prefix_mask = union_position[:, None] < column_length[None, :]
        mode2_mask = tl.where(
            query_in_pair[None, :] == 0,
            union_position[:, None] < first_length,
            union_position[:, None] != first_topk_length,
        )
        mode3_first_mask = (
            (union_position[:, None] < second_length)
            & (union_position[:, None] != first_topk_length)
            & (union_position[:, None] != second_length - 1)
        )
        mode3_mask = tl.where(
            query_in_pair[None, :] == 0,
            mode3_first_mask,
            union_position[:, None] < second_length,
        )
        mode4_first_mask = (union_position[:, None] < first_topk_length) | (
            (union_position[:, None] >= second_topk_length)
            & (union_position[:, None] < second_topk_length + WINDOW)
        )
        mode4_mask = tl.where(
            query_in_pair[None, :] == 0,
            mode4_first_mask,
            union_position[:, None] != second_topk_length,
        )
        pair_mask = tl.where(
            is_mode2,
            mode2_mask,
            tl.where(
                is_mode3,
                mode3_mask,
                tl.where(is_mode4, mode4_mask, prefix_mask),
            ),
        )
        scores = tl.where(valid_ids[:, None] & pair_mask, scores, float("-inf"))

        new_max = tl.maximum(max_log, tl.max(scores, axis=0))
        max_for_exp = tl.where(new_max == float("-inf"), 0.0, new_max)
        exp_scores = tl.math.exp(scores - max_for_exp[None, :])
        alpha = tl.math.exp(max_log - max_for_exp)
        sum_exp = sum_exp * alpha + tl.sum(exp_scores, axis=0)
        weights = exp_scores.to(tl.bfloat16)
        acc0 = tl.dot(
            kv_blk0,
            weights,
            acc0 * alpha[None, :],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            kv_blk1,
            weights,
            acc1 * alpha[None, :],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_row = max_log != float("-inf")
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + head)
        denominator = sum_exp + tl.math.exp(sink - max_log)
    else:
        denominator = sum_exp
    denominator = tl.where(valid_row, denominator, 1.0)
    factor = 1.0 / denominator
    out_vals0 = tl.where(valid_row[None, :], acc0 * factor[None, :], 0.0)
    out_vals1 = tl.where(valid_row[None, :], acc1 * factor[None, :], 0.0)
    output_ptr = (
        output
        + query_row[:, None] * stride_om
        + head[:, None] * stride_oh
        + offsets_d[None, :]
    )
    tl.store(output_ptr, out_vals0.trans().to(tl.bfloat16))
    tl.store(output_ptr + 256, out_vals1.trans().to(tl.bfloat16))


@triton.jit
def _flash_mla_sparse_hq4_single(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    output,
    row,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    SKV,
    sm_scale: tl.constexpr,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    SINGLE_CACHE_CA: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """Run the regular four-head path for a work item that cannot be paired."""

    offsets_h = tl.arange(0, BH)
    mask_h = offsets_h < 4
    offsets_d = tl.arange(0, 256)
    q_ptr = q + row * stride_qm + offsets_h[:, None] * stride_qh + offsets_d[None, :]
    q_blk0 = tl.load(q_ptr, mask=mask_h[:, None], other=0.0)
    q_blk1 = tl.load(q_ptr + 256, mask=mask_h[:, None], other=0.0)

    raw_length = tl.load(topk_length + row)
    length = tl.minimum(tl.maximum(raw_length, 0), TOPK)
    offsets_t = tl.arange(0, BK)
    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([BH], dtype=tl.float32)
    acc0 = tl.zeros([256, BH], dtype=tl.float32)
    acc1 = tl.zeros([256, BH], dtype=tl.float32)

    for block_index in range(tl.cdiv(length, BK)):
        positions = block_index * BK + offsets_t
        token_mask = positions < length
        kv_ids = tl.load(
            indices + row * stride_tm + positions,
            mask=token_mask,
            other=-1,
        )
        valid_ids = token_mask & (kv_ids >= 0) & (kv_ids < SKV)
        safe_kv_ids = tl.where(valid_ids, kv_ids, 0)
        kv_ptr = kv + offsets_d[:, None] + safe_kv_ids[None, :] * stride_kvn
        if SINGLE_CACHE_CA:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".ca")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".ca")
        else:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".cg")

        scores = tl.dot(kv_blk0.trans(), q_blk0.trans(), out_dtype=tl.float32)
        scores = tl.dot(kv_blk1.trans(), q_blk1.trans(), scores, out_dtype=tl.float32)
        scores *= sm_scale
        scores = tl.where(valid_ids[:, None] & mask_h[None, :], scores, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(scores, axis=0))
        max_for_exp = tl.where(new_max == float("-inf"), 0.0, new_max)
        exp_scores = tl.math.exp(scores - max_for_exp[None, :])
        alpha = tl.math.exp(max_log - max_for_exp)
        sum_exp = sum_exp * alpha + tl.sum(exp_scores, axis=0)
        weights = exp_scores.to(tl.bfloat16)
        acc0 = tl.dot(
            kv_blk0,
            weights,
            acc0 * alpha[None, :],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            kv_blk1,
            weights,
            acc1 * alpha[None, :],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_row = max_log != float("-inf")
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + offsets_h, mask=mask_h, other=float("-inf"))
        denominator = sum_exp + tl.math.exp(sink - max_log)
    else:
        denominator = sum_exp
    denominator = tl.where(valid_row, denominator, 1.0)
    factor = 1.0 / denominator
    out_vals0 = tl.where(valid_row[None, :], acc0 * factor[None, :], 0.0)
    out_vals1 = tl.where(valid_row[None, :], acc1 * factor[None, :], 0.0)
    output_ptr = (
        output + row * stride_om + offsets_h[:, None] * stride_oh + offsets_d[None, :]
    )
    tl.store(output_ptr, out_vals0.trans().to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(
        output_ptr + 256,
        out_vals1.trans().to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.jit
def triton_flash_mla_sparse_fwd_hq4_pair_work_items(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    pair_metadata,
    output,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    SQ,
    SKV,
    sm_scale: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    PAIR_CACHE_CA: tl.constexpr,
    SINGLE_CACHE_CA: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """Select an N=8 pair or an N=4 single path from producer metadata."""

    work_id = tl.program_id(0)
    pair_id = work_id // 2
    slot = work_id % 2
    first_row = (pair_id * 2).to(tl.int64)
    has_second = first_row + 1 < SQ
    packed_metadata = tl.load(pair_metadata + pair_id)
    pair_mode = packed_metadata & 7
    first_topk_length = packed_metadata >> 3
    first_length = tl.load(topk_length + first_row)
    second_length = tl.load(topk_length + first_row + 1, mask=has_second, other=0)

    valid_mode1 = (
        (pair_mode == 1)
        & has_second
        & (first_length > 0)
        & (second_length == first_length + 1)
        & (second_length <= TOPK)
        & (first_topk_length >= 0)
        & (first_length - first_topk_length > 0)
        & (first_length - first_topk_length < WINDOW)
    )
    valid_mode2 = (
        (pair_mode == 2)
        & has_second
        & (first_length >= WINDOW)
        & (first_length == second_length)
        & (first_length <= TOPK)
        & (first_length - first_topk_length == WINDOW)
    )
    valid_mode3 = (
        (pair_mode == 3)
        & has_second
        & (first_length > 0)
        & (second_length == first_length + 2)
        & (second_length <= TOPK)
        & (first_topk_length >= 0)
        & (first_length - first_topk_length > 0)
        & (first_length - first_topk_length < WINDOW)
    )
    valid_mode4 = (
        (pair_mode == 4)
        & has_second
        & (first_length > 0)
        & (second_length == first_length + 1)
        & (second_length <= TOPK)
        & (first_topk_length >= 0)
        & (first_length - first_topk_length == WINDOW)
    )
    is_pair = valid_mode1 | valid_mode2 | valid_mode3 | valid_mode4

    if is_pair:
        if slot == 0:
            _flash_mla_sparse_hq4_pair(
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                output,
                first_row,
                pair_mode,
                first_topk_length,
                stride_qh,
                stride_qm,
                stride_kvn,
                stride_tm,
                stride_oh,
                stride_om,
                SKV,
                sm_scale,
                WINDOW,
                HAVE_ATTN_SINK,
                PAIR_CACHE_CA,
                BK,
            )
    else:
        row = first_row + slot
        if row < SQ:
            _flash_mla_sparse_hq4_single(
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                output,
                row,
                stride_qh,
                stride_qm,
                stride_kvn,
                stride_tm,
                stride_oh,
                stride_om,
                SKV,
                sm_scale,
                TOPK,
                HAVE_ATTN_SINK,
                SINGLE_CACHE_CA,
                BK,
                BH,
            )


@triton.autotune(
    configs=[
        triton.Config({"BK": 32, "BH": 16}, num_warps=4, num_stages=2),
        triton.Config({"BK": 64, "BH": 16}, num_warps=4, num_stages=2),
        triton.Config({"BK": 64, "BH": 16}, num_warps=4, num_stages=3),
        triton.Config({"BK": 64, "BH": 16}, num_warps=8, num_stages=2),
        triton.Config({"BK": 128, "BH": 16}, num_warps=8, num_stages=2),
        triton.Config({"BK": 128, "BH": 16}, num_warps=8, num_stages=3),
        triton.Config({"BK": 32, "BH": 64}, num_warps=8, num_stages=2),
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=2),
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=4),
    ],
    key=[
        "SQ",
        "HQ",
        "DQK",
        "SKV",
        "TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
        "RETURN_STATS",
    ],
    prune_configs_by={"early_config_prune": _prune_flash_mla_sparse_configs},
)
@triton.jit
def triton_flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,  # s_q
    HQ: tl.constexpr,  # h_q<=128
    DQK: tl.constexpr,  # d_qk=512 or 576
    SKV,  # s_kv
    TOPK: tl.constexpr,  # topk
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    RETURN_STATS: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)  # prevent mul overflow
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    if RETURN_STATS:
        max_log_base = max_logits + i_sq * stride_mm + gbh_base
        l_base = lse + i_sq * stride_lm + gbh_base

    offs_h = tl.arange(0, BH)
    mask_h = gbh_base + offs_h < HQ
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    # `[BH, 256] x 2` delivers better performance than `[BH, 512]` when BH=64
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(
        q_ptr, mask=mask_h[:, None], other=0.0, eviction_policy="evict_first"
    )
    q_blk1 = tl.load(
        q_ptr + BDP,
        mask=mask_h[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(
            tq_ptr, mask=mask_h[:, None], other=0.0, eviction_policy="evict_first"
        )

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    topk_len = tl.minimum(tl.maximum(topk_len, 0), TOPK)
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        # step1: load indices
        t_ptr = BK * ck + offs_t  # [BK]
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        # filter invalid index that may cause overflow in mul
        kv_ids = tl.where(mask_ids, kv_ids, 0)

        # step2: gather kv with indices
        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")  # [BDP, BK]
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")  # [BDP, BK]
        # step3: (q @ kv) * sm_scale
        qk = tl.dot(
            q_blk0, kv_blk0, out_dtype=tl.float32
        )  # [BH, BDP]@[BDP, BK] -> [BH, BK]
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
            tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")  # [TDP, BK]
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale

        # step4: preprocess for logsumexp
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))  # [BH, BK]
        # step5: lse=logsumexp(qk), loop part
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))  # [BH]
        # Avoid -inf - -inf when every index in the first block is invalid.
        max_for_exp = tl.where(new_max == float("-inf"), 0.0, new_max)
        exp_qk = tl.math.exp(qk - max_for_exp[:, None])  # [BH, BK]
        sum_qk = tl.sum(exp_qk, axis=1)  # [BH]
        alpha = tl.math.exp(max_log - max_for_exp)  # [BH]
        sum_exp = sum_exp * alpha + sum_qk  # [BH]
        # step6: exp(qk-lse) @ gathered_kv.trans(), loop part
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP]->[BH, BDP]
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP]->[BH, BDP]
        max_log = new_max

    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    if RETURN_STATS:
        # Store max_logits and the final logsumexp only when the caller uses
        # them.  Inference attention consumes only the output tensor.
        tl.store(max_log_base + offs_h, max_log, mask=mask_h)
        orig_lse = max_log + tl.math.log(sum_exp)
        lse_out = tl.where(valid_mask, orig_lse, float("inf"))
        tl.store(l_base + offs_h, lse_out, mask=mask_h)

    # step9: exp(qk-lse) @ gathered_kv.trans(), final part
    if HAVE_ATTN_SINK:
        # step10: attn_sink
        sink = tl.load(attn_sink_ptr + offs_h, mask=mask_h, other=0.0)  # [BH]
        factor = 1.0 / (sum_exp + tl.math.exp(sink - max_log))
    else:
        factor = 1.0 / sum_exp

    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    # step11: store output
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]  # [BH, BDP]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16), mask=mask_h[:, None])


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_stats: bool = True,
    *,
    pair_metadata: Optional[torch.Tensor] = None,
    pair_window_size: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.
        out: optional pre-allocated output tensor, [s_q, h_q, d_v], bfloat16.
        return_stats: whether to return max_logits and lse. Set this to False
            when only the attention output is consumed.
        pair_metadata: optional packed int32 descriptors emitted by
            combine_topk_swa_indices. One descriptor covers two adjacent rows;
            structurally invalid or unsupported descriptors use the regular
            path. Descriptors must come from the same indices and topk_length
            tensors and must not be reused after either tensor is changed.
        pair_window_size: sliding-window width used to create pair_metadata.

    Returns:
        (output, max_logits, lse). max_logits and lse are None when
        return_stats is False.
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert q.stride(-1) == 1 and kv.stride(-1) == 1 and indices.stride(-1) == 1
    if type(return_stats) is not bool:
        raise TypeError("return_stats must be a bool")
    if type(pair_window_size) is not int:
        raise TypeError("pair_window_size must be an int")
    if pair_metadata is None:
        if pair_window_size != 0:
            raise ValueError("pair_window_size requires pair_metadata")
    elif not isinstance(pair_metadata, torch.Tensor):
        raise TypeError("pair_metadata must be a torch.Tensor or None")
    elif pair_window_size <= 0:
        raise ValueError("pair_window_size must be positive with pair_metadata")
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    assert q.device == kv.device and q.device == indices.device
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape
    assert SKV > 0, "kv must contain at least one row"

    assert d_v == 512, "Unsupported d_v"
    DV = d_v

    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.device == q.device
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.device == q.device
        assert topk_length.shape == (SQ,), "topk_length error shape"
    if pair_metadata is not None:
        if topk_length is None:
            raise ValueError("pair_metadata requires topk_length")
        assert pair_metadata.is_contiguous()
        assert pair_metadata.dtype == torch.int32
        assert pair_metadata.device == q.device
        assert pair_metadata.shape == (triton.cdiv(SQ, 2),), "pair_metadata error shape"

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert 0 < HQ <= 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    _ = SKV
    if out is None:
        output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    else:
        assert out.shape == (SQ, HQ, DV), "out error shape"
        assert out.dtype == q.dtype, "out error dtype"
        assert out.device == q.device, "out error device"
        assert out.stride(-1) == 1, "out must have a contiguous last dimension"
        output = out
    if return_stats:
        max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
        lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
        max_logits_ptr = max_logits
        lse_ptr = lse
        stride_mm = max_logits.stride(0)
        stride_lm = lse.stride(0)
    else:
        max_logits = None
        lse = None
        # Triton specializes RETURN_STATS at compile time, so these pointers
        # and strides are never dereferenced in the no-statistics variant.
        max_logits_ptr = output
        lse_ptr = output
        stride_mm = 0
        stride_lm = 0

    def triton_grid(META):
        return (triton.cdiv(HQ, META["BH"]) * SQ,)

    if (
        pair_metadata is not None
        and HQ == 4
        and DQK == 512
        and not return_stats
        and SQ >= 2
    ):
        pair_cache_ca = attn_sink is None
        single_cache_ca = attn_sink is not None
        triton_flash_mla_sparse_fwd_hq4_pair_work_items[(SQ,)](
            q,
            kv,
            indices,
            attn_sink,
            topk_length,
            pair_metadata,
            output,
            q.stride(1),
            q.stride(0),
            kv.stride(0),
            indices.stride(0),
            output.stride(1),
            output.stride(0),
            SQ,
            SKV,
            sm_scale,
            TOPK,
            pair_window_size,
            attn_sink is not None,
            pair_cache_ca,
            single_cache_ca,
            BK=64,
            BH=4,
            num_warps=4,
            num_stages=3,
        )
        return output, max_logits, lse

    if HQ == 4 and DQK == 512:
        triton_flash_mla_sparse_fwd_hq4[(SQ,)](
            q,
            kv,
            indices,
            attn_sink,
            topk_length,
            sm_scale,
            output,
            max_logits_ptr,
            lse_ptr,
            q.stride(1),
            q.stride(0),
            kv.stride(0),
            indices.stride(0),
            output.stride(1),
            output.stride(0),
            stride_mm,
            stride_lm,
            SQ,
            SKV,
            TOPK,
            attn_sink is not None,
            topk_length is not None,
            return_stats,
        )
        return output, max_logits, lse

    triton_flash_mla_sparse_fwd[triton_grid](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits_ptr,
        lse_ptr,
        q.stride(1),
        q.stride(0),
        kv.stride(1),
        kv.stride(0),
        indices.stride(1),
        indices.stride(0),
        output.stride(1),
        output.stride(0),
        stride_mm,
        stride_lm,
        SQ,
        HQ,
        DQK,
        SKV,
        TOPK,
        attn_sink is not None,
        topk_length is not None,
        return_stats,
    )
    return output, max_logits, lse
