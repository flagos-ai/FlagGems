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

"""Forward-only dense cross attention specialized for MetaX GPUs."""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.cross_attention import _mask_strides, _validate_inputs
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _bf16_split(x):
    hi = x.to(tl.bfloat16)
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
    return hi, lo


def _select_metax_tiling(
    query_len, key_len, head_dim, has_mask, is_fp32
):
    """Select tiles that fit the C500 register and shared-memory limits."""
    if query_len <= 32:
        if not is_fp32 and head_dim <= 64:
            return 16, 64, 8, 2
        return 16, 32, 4, 1
    if is_fp32:
        if head_dim > 64:
            return 64, 16, (8 if key_len < 1024 else 4), 1
        return 64, 32, 4, 1
    if head_dim > 128:
        return 16, 32, 4, 1
    if has_mask or head_dim > 64:
        return 64, 32, (8 if not has_mask and key_len < 1024 else 4), 1
    if head_dim <= 32 and key_len >= 1024:
        if head_dim <= 16:
            return 64, 128, 4, 1
        return 64, 128, 4, 1
    if key_len >= 1024:
        return 64, 64, 4, 1
    return 64, 32, 4, 1


@libentry()
@triton.jit
def cross_attention_metax_kernel(
    Q,
    K,
    V,
    AttnMask,
    Out,
    scale,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    stride_mask_batch,
    stride_mask_head,
    stride_mask_query,
    stride_mask_key,
    stride_o_batch,
    stride_o_head,
    stride_o_seq,
    stride_o_dim,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    EXACT_BOUNDS: tl.constexpr,
    FP32_SPLIT: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    value_tile = tl.program_id(2)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)

    query_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    qk_offsets = tl.arange(0, QK_BLOCK)
    value_offsets = value_tile * VALUE_BLOCK + tl.arange(0, VALUE_BLOCK)
    valid_queries = query_offsets < QUERY_LEN

    query_base = (
        batch.to(tl.int64) * stride_q_batch
        + query_head.to(tl.int64) * stride_q_head
    )
    key_base = (
        batch.to(tl.int64) * stride_k_batch
        + kv_head.to(tl.int64) * stride_k_head
    )
    value_base = (
        batch.to(tl.int64) * stride_v_batch
        + kv_head.to(tl.int64) * stride_v_head
    )
    output_base = (
        batch.to(tl.int64) * stride_o_batch
        + query_head.to(tl.int64) * stride_o_head
    )
    query_ptrs = (
        Q
        + query_base
        + query_offsets[:, None] * stride_q_seq
        + qk_offsets[None, :] * stride_q_dim
    )
    if EXACT_BOUNDS:
        query = tl.load(query_ptrs)
    else:
        query = tl.load(
            query_ptrs,
            mask=valid_queries[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        )
    if Q.dtype.element_ty == tl.float32 and FP32_SPLIT:
        query_hi, query_lo = _bf16_split(query)

    mask_base = 0
    if HAS_MASK:
        mask_base = (
            batch.to(tl.int64) * stride_mask_batch
            + query_head.to(tl.int64) * stride_mask_head
            + query_offsets[:, None] * stride_mask_query
        )

    accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # MetaX does not implement the TLE gpu.pipeline primitive used by the
    # generic kernel. tl.range uses the backend's native basic pipeline.
    for key_start in tl.range(0, KEY_LEN, BLOCK_N, num_stages=PIPELINE_STAGES):
        logical_keys = key_start + key_offsets
        valid_keys = logical_keys < KEY_LEN
        key_ptrs = (
            K
            + key_base
            + logical_keys[None, :] * stride_k_seq
            + qk_offsets[:, None] * stride_k_dim
        )
        if EXACT_BOUNDS:
            key = tl.load(key_ptrs)
        else:
            key = tl.load(
                key_ptrs,
                mask=(qk_offsets[:, None] < QK_DIM) & valid_keys[None, :],
                other=0.0,
            )
        if Q.dtype.element_ty == tl.float32:
            if FP32_SPLIT:
                key_hi, key_lo = _bf16_split(key)
                scores = tl.dot(query_hi, key_hi, allow_tf32=False)
                scores = tl.dot(
                    query_hi, key_lo, scores, allow_tf32=False
                )
                scores = tl.dot(
                    query_lo, key_hi, scores, allow_tf32=False
                )
            else:
                scores = tl.dot(query, key, input_precision="ieee")
        else:
            scores = tl.dot(query, key, allow_tf32=False)
        scores *= scale * _LOG2E

        if EXACT_BOUNDS:
            if HAS_MASK:
                blocked = (
                    tl.load(
                        AttnMask
                        + mask_base
                        + logical_keys[None, :] * stride_mask_key
                    )
                    != 0
                )
                score_is_valid = ~blocked
                scores = tl.where(score_is_valid, scores, -float("inf"))
        else:
            score_is_valid = valid_queries[:, None] & valid_keys[None, :]
            if HAS_MASK:
                blocked = (
                    tl.load(
                        AttnMask
                        + mask_base
                        + logical_keys[None, :] * stride_mask_key,
                        mask=score_is_valid,
                        other=1,
                    )
                    != 0
                )
                score_is_valid &= ~blocked
            scores = tl.where(score_is_valid, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        if EXACT_BOUNDS and not HAS_MASK:
            probabilities = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(row_max - new_max)
        elif not HAS_MASK:
            # Every valid query has at least one valid key. Invalid tail rows
            # are never stored, so only substitute their normalization value.
            safe_max = tl.where(valid_queries, new_max, 0.0)
            probabilities = tl.math.exp2(
                tl.where(
                    score_is_valid,
                    scores - safe_max[:, None],
                    -float("inf"),
                )
            )
            alpha = tl.math.exp2(row_max - safe_max)
        else:
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.math.exp2(
                tl.where(
                    score_is_valid,
                    scores - safe_max[:, None],
                    -float("inf"),
                )
            )
            alpha = tl.where(
                row_max == -float("inf"), 0.0, tl.math.exp2(row_max - safe_max)
            )
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)

        value_ptrs = (
            V
            + value_base
            + logical_keys[:, None] * stride_v_seq
            + value_offsets[None, :] * stride_v_dim
        )
        if EXACT_BOUNDS:
            value = tl.load(value_ptrs)
        else:
            value = tl.load(
                value_ptrs,
                mask=valid_keys[:, None] & (value_offsets[None, :] < VALUE_DIM),
                other=0.0,
            )
        accumulator *= alpha[:, None]
        if V.dtype.element_ty == tl.float32:
            if FP32_SPLIT:
                probability_hi, probability_lo = _bf16_split(probabilities)
                value_hi, value_lo = _bf16_split(value)
                accumulator = tl.dot(
                    probability_hi,
                    value_hi,
                    accumulator,
                    allow_tf32=False,
                )
                accumulator = tl.dot(
                    probability_hi,
                    value_lo,
                    accumulator,
                    allow_tf32=False,
                )
                accumulator = tl.dot(
                    probability_lo,
                    value_hi,
                    accumulator,
                    allow_tf32=False,
                )
            else:
                accumulator = tl.dot(
                    probabilities,
                    value,
                    accumulator,
                    input_precision="ieee",
                )
        else:
            accumulator = tl.dot(
                probabilities.to(V.dtype.element_ty),
                value,
                accumulator,
                allow_tf32=False,
            )
        row_max = new_max

    output = accumulator * tl.where(row_sum > 0.0, 1.0 / row_sum, 0.0)[:, None]
    output_ptrs = (
        Out
        + output_base
        + query_offsets[:, None] * stride_o_seq
        + value_offsets[None, :] * stride_o_dim
    )
    if EXACT_BOUNDS:
        tl.store(output_ptrs, output)
    else:
        tl.store(
            output_ptrs,
            output,
            mask=valid_queries[:, None] & (value_offsets[None, :] < VALUE_DIM),
        )


def cross_attention(query, key, value, attn_mask=None, scale=None):
    """Compute forward cross attention for BNSD tensors on MetaX GPUs."""
    logger.debug("GEMS_METAX CROSS_ATTENTION")
    _validate_inputs(query, key, value, attn_mask, scale)
    batch, query_heads, query_len, qk_dim = query.shape
    kv_heads, key_len, value_dim = key.shape[1], key.shape[2], value.shape[3]
    softmax_scale = 1.0 / math.sqrt(qk_dim) if scale is None else float(scale)
    output = torch.empty(
        (batch, query_heads, query_len, value_dim),
        dtype=query.dtype,
        device=query.device,
    )

    if attn_mask is None:
        mask_arg = query
        mask_strides = (0, 0, 0, 0)
    else:
        mask_arg = attn_mask
        mask_strides = _mask_strides(attn_mask)

    qk_block = max(16, triton.next_power_of_2(qk_dim))
    value_block = max(16, triton.next_power_of_2(min(value_dim, 128)))
    block_m, block_n, num_warps, pipeline_stages = _select_metax_tiling(
        query_len,
        key_len,
        max(qk_dim, value_dim),
        attn_mask is not None,
        query.dtype == torch.float32,
    )
    grid = (
        triton.cdiv(query_len, block_m),
        batch * query_heads,
        triton.cdiv(value_dim, value_block),
    )
    exact_bounds = (
        query_len % block_m == 0
        and key_len % block_n == 0
        and qk_dim == qk_block
        and value_dim % value_block == 0
    )
    with torch_device_fn.device(query.device):
        cross_attention_metax_kernel[grid](
            query,
            key,
            value,
            mask_arg,
            output,
            softmax_scale,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *mask_strides,
            *output.stride(),
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            QUERY_LEN=query_len,
            KEY_LEN=key_len,
            QK_DIM=qk_dim,
            VALUE_DIM=value_dim,
            QK_BLOCK=qk_block,
            VALUE_BLOCK=value_block,
            HAS_MASK=attn_mask is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            PIPELINE_STAGES=pipeline_stages,
            EXACT_BOUNDS=exact_bounds,
            FP32_SPLIT=query.dtype == torch.float32,
            num_stages=pipeline_stages,
            num_warps=num_warps,
        )
    return output


__all__ = ["cross_attention"]
