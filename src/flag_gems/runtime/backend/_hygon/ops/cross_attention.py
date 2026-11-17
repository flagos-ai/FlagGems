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

"""Forward-only dense cross attention specialized for Hygon DCU (BW).

Same public contract as ``flag_gems.ops.cross_attention`` (BNSD layout,
MHA/GQA/MQA, non-zero mask entries blocked, exact zeros for fully masked
rows).  Hygon-specific choices:

* Tiles are filtered against the 64 KiB LDS budget; the generic table targets
  Hopper's 227 KiB and overflows on Hygon once D >= 64.
* No TLE dependency: plain ``tl.range`` software pipelining.
* fp32 runs as three bf16 MMAs over a hi/lo decomposition (``_bf16_split``).
* Fully in-bounds launches drop every bounds predicate.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.cross_attention import _mask_strides, _validate_inputs
from flag_gems.ops.cross_attention import cross_attention as _generic_cross_attention
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Per-workgroup LDS budget on Hygon BW, minus a reserve for reduction scratch.
_HYGON_LDS_BYTES = 64 * 1024
_HYGON_LDS_RESERVE = 4 * 1024

# False -> fp32 operands use ieee dot instead of the bf16 hi/lo path.
_FP32_BF16_SPLIT = True


def _hygon_lds_bytes(bm, bn, qk_block, value_block, ns, itemsize):
    """Approximate the staged LDS: Q tile + P tile + ``ns`` K/V buffers.

    fp32's bf16 hi/lo halves cost the same bytes per element as fp32.
    """
    query_tile = bm * qk_block * itemsize
    p_tile = bm * bn * itemsize
    kv_stages = ns * bn * (qk_block + value_block) * itemsize
    return query_tile + p_tile + kv_stages


def _fits_hygon_lds(bm, bn, qk_block, value_block, ns, itemsize):
    return (
        _hygon_lds_bytes(bm, bn, qk_block, value_block, ns, itemsize)
        <= _HYGON_LDS_BYTES - _HYGON_LDS_RESERVE
    )


def _candidate_tiles(is_fp32, has_mask, query_len, key_len, head_dim):
    """Ordered (BLOCK_M, BLOCK_N, num_warps, pipeline_ns) preferences.

    The first entry that fits the 64 KiB LDS budget is used.  N shrinks as D
    grows because a stage holds ``BLOCK_N * (D + Dv)`` elements.
    """
    if is_fp32:
        if query_len <= 32:
            return [
                (32, 32, 4, 1),
                (32, 16, 4, 1),
                (16, 16, 4, 1),
            ]
        if head_dim <= 64:
            return [
                (64, 32, 4, 2),
                (64, 32, 4, 1),
                (32, 32, 4, 2),
                (32, 16, 4, 1),
                (16, 16, 4, 1),
            ]
        if head_dim <= 128:
            return [
                (64, 32, 4, 1),
                (32, 32, 4, 1),
                (32, 16, 4, 2),
                (16, 16, 4, 1),
            ]
        return [
            (32, 16, 4, 1),
            (16, 16, 4, 1),
        ]

    # fp16 / bf16.
    if query_len <= 32:
        # Single query block: latency-bound, keep N wide enough for both MMAs.
        return [
            (32, 64, 4, 3),
            (32, 32, 4, 3),
            (16, 16, 4, 2),
        ]
    if head_dim <= 32:
        if has_mask:
            return [
                (128, 32, 4, 3),
                (64, 64, 4, 2),
                (64, 32, 4, 3),
                (64, 16, 4, 3),
                (32, 16, 4, 2),
                (16, 16, 4, 2),
            ]
        # Small D: a wide N tile still fits and halves the KV loop count.
        return [
            (64, 128, 4, 1),
            (128, 128, 4, 1),
            (128, 64, 4, 1),
            (128, 32, 4, 3),
            (64, 32, 4, 4),
            (16, 16, 4, 2),
        ]
    if head_dim <= 64:
        if has_mask:
            # Masked path: narrower N halves the per-tile mask traffic.
            return [
                (128, 32, 4, 3),
                (64, 64, 4, 2),
                (64, 32, 4, 3),
                (64, 16, 4, 3),
                (32, 16, 4, 2),
                (16, 16, 4, 2),
            ]
        return [
            # 32 < D <= 64: one stage with N = 64 wins; N = 128 no longer
            # fits next to the wider P tile.
            (128, 64, 4, 1),
            (128, 32, 4, 3),
            (64, 64, 4, 2),
            (64, 32, 4, 4),
            (32, 32, 4, 3),
            (16, 16, 4, 2),
        ]
    if head_dim <= 128:
        # D = 128: 16 KiB per K/V stage, so one pipeline stage and narrow N.
        if has_mask:
            return [
                (64, 64, 4, 1),
                (64, 32, 4, 2),
                (32, 32, 4, 2),
                (16, 16, 4, 2),
            ]
        return [
            (128, 32, 4, 1),
            (64, 64, 4, 1),
            (64, 32, 4, 2),
            (32, 32, 4, 2),
            (16, 16, 4, 2),
        ]
    if head_dim <= 256:
        return [
            (32, 32, 4, 1),
            (32, 16, 4, 2),
            (16, 16, 4, 2),
        ]
    return [(16, 16, 4, 1)]


def _select_hygon_tiling(
    query_len,
    key_len,
    qk_dim,
    value_dim,
    has_mask,
    is_fp32,
    batch_size,
    query_heads,
):
    """Return (BLOCK_M, BLOCK_N, num_warps, pipeline_ns) or None if no tile fits."""
    itemsize = 4 if is_fp32 else 2
    qk_block = max(16, triton.next_power_of_2(qk_dim))
    value_block = max(16, triton.next_power_of_2(min(value_dim, 128)))
    head_dim = max(qk_dim, value_dim)

    for block_m, block_n, num_warps, pipeline_ns in _candidate_tiles(
        is_fp32, has_mask, query_len, key_len, head_dim
    ):
        # Avoid launching mostly-empty M tiles for short query sequences.
        block_m = min(block_m, max(16, triton.next_power_of_2(query_len)))
        if _fits_hygon_lds(
            block_m, block_n, qk_block, value_block, pipeline_ns, itemsize
        ):
            return block_m, block_n, num_warps, pipeline_ns
    return None


def _exact_bounds(
    query_len, key_len, qk_dim, value_dim, block_m, block_n, qk_block, value_block
):
    """True when every tile is fully in bounds, so all predicates can be dropped."""
    return (
        query_len % block_m == 0
        and key_len % block_n == 0
        and qk_dim == qk_block
        and value_dim % value_block == 0
    )


@triton.jit
def _bf16_split(x):
    """Decompose fp32 ``x`` into bf16 hi/lo halves (``x ~= hi + lo``).

    Three bf16 MMAs then reproduce a fp32 product to ~16 significant bits.
    """
    hi = x.to(tl.bfloat16)
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
    return hi, lo


@libentry()
@triton.jit
def cross_attention_hygon_kernel(
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
    PIPELINE_NS: tl.constexpr,
    FP32_SPLIT: tl.constexpr,
    EXACT_BOUNDS: tl.constexpr,
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
        batch.to(tl.int64) * stride_q_batch + query_head.to(tl.int64) * stride_q_head
    )
    key_base = (
        batch.to(tl.int64) * stride_k_batch + kv_head.to(tl.int64) * stride_k_head
    )
    value_base = (
        batch.to(tl.int64) * stride_v_batch + kv_head.to(tl.int64) * stride_v_head
    )
    output_base = (
        batch.to(tl.int64) * stride_o_batch + query_head.to(tl.int64) * stride_o_head
    )
    if EXACT_BOUNDS:
        query = tl.load(
            Q
            + query_base
            + query_offsets[:, None] * stride_q_seq
            + qk_offsets[None, :] * stride_q_dim
        )
    else:
        query = tl.load(
            Q
            + query_base
            + query_offsets[:, None] * stride_q_seq
            + qk_offsets[None, :] * stride_q_dim,
            mask=valid_queries[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        )

    if Q.dtype.element_ty == tl.float32 and FP32_SPLIT:
        q_hi, q_lo = _bf16_split(query)
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

    for key_start in tl.range(0, KEY_LEN, BLOCK_N, num_stages=PIPELINE_NS):
        logical_keys = key_start + key_offsets
        if EXACT_BOUNDS:
            key = tl.load(
                K
                + key_base
                + logical_keys[None, :] * stride_k_seq
                + qk_offsets[:, None] * stride_k_dim
            )
        else:
            valid_keys = logical_keys < KEY_LEN
            key = tl.load(
                K
                + key_base
                + logical_keys[None, :] * stride_k_seq
                + qk_offsets[:, None] * stride_k_dim,
                mask=(qk_offsets[:, None] < QK_DIM) & valid_keys[None, :],
                other=0.0,
            )
        if Q.dtype.element_ty == tl.float32:
            if FP32_SPLIT:
                k_hi, k_lo = _bf16_split(key)
                scores = tl.dot(q_hi, k_hi)
                scores = tl.dot(q_hi, k_lo, scores, allow_tf32=False)
                scores = tl.dot(q_lo, k_hi, scores, allow_tf32=False)
            else:
                scores = tl.dot(query, key, input_precision="ieee")
        else:
            scores = tl.dot(query, key, allow_tf32=False)
        scores *= scale
        if EXACT_BOUNDS:
            if HAS_MASK:
                blocked = (
                    tl.load(
                        AttnMask + mask_base + logical_keys[None, :] * stride_mask_key
                    )
                    != 0
                )
                scores = tl.where(blocked, -float("inf"), scores)
        else:
            score_is_valid = valid_queries[:, None] & valid_keys[None, :]
            if HAS_MASK:
                blocked = (
                    tl.load(
                        AttnMask + mask_base + logical_keys[None, :] * stride_mask_key,
                        mask=score_is_valid,
                        other=1,
                    )
                    != 0
                )
                score_is_valid = score_is_valid & ~blocked
            scores = tl.where(score_is_valid, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        # safe_max avoids NaN on fully blocked rows (-inf - -inf).
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        # Blocked lanes are already -inf, so exp(-inf) = 0; no extra select.
        probabilities = tl.exp(scores - safe_max[:, None])
        # row_max starts at -inf, so the first alpha is 0 without a branch.
        alpha = tl.exp(row_max - safe_max)
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
                p_hi, p_lo = _bf16_split(probabilities)
                v_hi, v_lo = _bf16_split(value)
                accumulator = tl.dot(p_hi, v_hi, accumulator, allow_tf32=False)
                accumulator = tl.dot(p_hi, v_lo, accumulator, allow_tf32=False)
                accumulator = tl.dot(p_lo, v_hi, accumulator, allow_tf32=False)
            else:
                accumulator = tl.dot(
                    probabilities, value, accumulator, input_precision="ieee"
                )
        else:
            p = probabilities.to(V.dtype.element_ty)
            accumulator = tl.dot(p, value, accumulator, allow_tf32=False)
        row_max = new_max

    output = accumulator * tl.where(row_sum > 0.0, 1.0 / row_sum, 0.0)[:, None]
    if EXACT_BOUNDS:
        tl.store(
            Out
            + output_base
            + query_offsets[:, None] * stride_o_seq
            + value_offsets[None, :] * stride_o_dim,
            output,
        )
    else:
        tl.store(
            Out
            + output_base
            + query_offsets[:, None] * stride_o_seq
            + value_offsets[None, :] * stride_o_dim,
            output,
            mask=valid_queries[:, None] & (value_offsets[None, :] < VALUE_DIM),
        )


def cross_attention(query, key, value, attn_mask=None, scale=None):
    """Compute forward cross attention for BNSD inputs (Hygon BW).

    Q and K have the same head dimension; V may have a smaller dimension. MHA,
    GQA, and MQA are supported. Non-zero bool/uint8 mask entries are blocked.
    Fully masked rows produce exact zeros. This API currently has no backward.
    """
    logger.debug("GEMS_HYGON CROSS_ATTENTION")
    _validate_inputs(query, key, value, attn_mask, scale)
    batch, query_heads, query_len, qk_dim = query.shape
    kv_heads, key_len, value_dim = key.shape[1], key.shape[2], value.shape[3]
    softmax_scale = 1.0 / math.sqrt(qk_dim) if scale is None else float(scale)

    tiling = _select_hygon_tiling(
        query_len,
        key_len,
        qk_dim,
        value_dim,
        attn_mask is not None,
        query.dtype == torch.float32,
        batch,
        query_heads,
    )
    if tiling is None:
        logger.debug(
            "GEMS_HYGON CROSS ATTENTION: no tile fits the 64 KiB LDS budget "
            "(Lq=%d Lk=%d D=%d Dv=%d); delegating to the generic kernel",
            query_len,
            key_len,
            qk_dim,
            value_dim,
        )
        return _generic_cross_attention(query, key, value, attn_mask, scale)

    block_m, block_n, num_warps, pipeline_ns = tiling
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
    grid = (
        triton.cdiv(query_len, block_m),
        batch * query_heads,
        triton.cdiv(value_dim, value_block),
    )
    # Divisible shapes: drop all per-lane predicates in the kernel.
    exact_bounds = _exact_bounds(
        query_len, key_len, qk_dim, value_dim, block_m, block_n, qk_block, value_block
    )
    with torch_device_fn.device(query.device):
        cross_attention_hygon_kernel[grid](
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
            PIPELINE_NS=pipeline_ns,
            FP32_SPLIT=query.dtype == torch.float32 and _FP32_BF16_SPLIT,
            EXACT_BOUNDS=exact_bounds,
            num_stages=max(1, pipeline_ns),
            num_warps=num_warps,
        )
    return output


__all__ = ["cross_attention"]
