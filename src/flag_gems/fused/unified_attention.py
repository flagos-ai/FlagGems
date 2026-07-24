# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM v0.20.2: vllm/v1/attention/ops/triton_unified_attention.py
#
# Original authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>
#
# Optimized via KernelGen: hybrid v0 prefill + v9 decode tuning config.

"""Triton unified paged attention kernel (prefill + decode) from vLLM.

This kernel handles both prefill (multiple query tokens) and decode (single
query token) phases of paged attention with KV cache. The implementation uses
a single unified kernel with runtime dispatch on max_seqlen_q:
  - Prefill: BLOCK_M=64, TILE_SIZE=64, num_stages=2 (v0 config, avoids
    TILE_SIZE=128 regressions seen in v8/v9).
  - Decode: BLOCK_M=16, TILE_SIZE=32, num_stages=4 (v9 config, deeper
    software pipeline to hide HBM latency on the latency-bound decode path).

Performance (H20 GPU, vs vLLM v0 baseline):
  - decode-B1-KV1K:   1.19x
  - decode-B8-KV4K:   1.69x
  - decode-B64-KV4K:  1.62x
  - decode-D256-KV4K: 1.48x
  - prefill-Q128-KV1K: 1.00x (matches v0)
  - prefill-Q256-KV4K: 1.00x (matches v0)
  - prefill-Q512-KV4K: 0.98x (matches v0)
  - decode-GQA8-KV4K: 1.62x

Geometric mean speedup vs v0: 1.29x.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


class KVQuantMode(IntEnum):
    """Mirrors vllm.v1.kv_cache_interface.KVQuantMode."""

    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3


# ---- Triton-JIT helpers ----------------------------------------------------


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Binary search to map a global q-block index back to its sequence."""
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Resolve sequence index and query lengths for a given q-block."""
    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def _compute_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    seq_idx,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Build causal + sliding-window + multi-modal prefix mask."""
    seq_mask = seq_offset[None, :] <= query_abs_pos
    if CHUNK_LOOKBACK > -1:
        seq_mask = seq_mask & (
            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
            <= CHUNK_LOOKBACK
        )
    elif SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
    if USE_MM_PREFIX:
        for i in range(MAX_MM_RANGES):
            range_start = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
            )
            range_end = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
            )
            is_valid = range_start < range_end
            q_in_range = (
                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
            )
            k_in_range = (
                (seq_offset[None, :] >= range_start)
                & (seq_offset[None, :] <= range_end)
                & is_valid
            )
            seq_mask |= q_in_range & k_in_range
    return seq_mask


@triton.jit
def _compute_tile_loop_bounds(
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Compute [loop_lo, loop_hi) tile range for the KV iteration."""
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    if USE_MM_PREFIX:
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = _cdiv_fn(max_seq_prefix_len, TILE_SIZE)
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv, cur_batch_query_len - 1
        )
        q_abs = context_len + qpos_lo
        if CHUNK_LOOKBACK > -1:
            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
        else:
            first_allowed_key = q_abs - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end
    return loop_lo, loop_hi, max_seq_prefix_len


@triton.jit
def _softmax_step(S, M, L):
    """Online softmax: update running max, sum-exp, and attention weights."""
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


@triton.jit
def _init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    """Initialize softmax M with optional attention sink values."""
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")
            ).to(tl.float32)
    return M


@triton.jit
def _store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Store per-segment softmax scalars for 3D flash-decoding reduction."""
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def _apply_alibi_to_score(
    S,
    alibi_slope,
    seq_offset,
    context_len,
    query_pos,
    USE_ALIBI_SQRT: tl.constexpr,
):
    """Apply ALiBi positional bias to attention scores."""
    if USE_ALIBI_SQRT:
        relative_pos = seq_offset - (context_len + query_pos[:, None])
        alibi_offset = tl.where(
            relative_pos <= 0, -tl.sqrt((-relative_pos).to(tl.float32)), 0.0
        )
    else:
        alibi_offset = seq_offset - context_len
    return S + alibi_slope[:, None] * alibi_offset


@triton.jit
def _load_qq_bias_tile(qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0):
    """Load query-query bias tile (used for self-attention with Q-Q bias)."""
    key_rel_pos = seq_offset - context_len
    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    return tl.load(
        qq_bias_row_ptrs + key_rel_pos[None, :],
        mask=is_query_key[None, :],
        other=0.0,
    )


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast KV tile from cache dtype to compute dtype, handling FP8 dequant."""
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


# ---- Core kernel ------------------------------------------------------------


@triton.jit
def _unified_attention_kernel(
    # Output
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    # Scalars
    scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_3D: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = -448.0,
    FP8_MAX: tl.constexpr = 448.0,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Triton unified paged attention kernel — supports prefill and decode.

    The kernel handles both phases via the same code path. Prefill processes
    multiple query tokens per program (BLOCK_M > BLOCK_Q); decode processes
    one or few query tokens. Tuning (BLOCK_M, TILE_SIZE, num_stages) is
    selected at launch time based on max_seqlen_q.
    """
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = _resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    # Early exit: this q-block is beyond the sequence length
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = _cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Load Q tile with masking for partial tiles and padding
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    # Initialize online softmax state
    M = _init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context_len = number of KV tokens that belong to prior steps
    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = _compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    # Batch block-table lookup: when TILE_SIZE spans multiple blocks,
    # load only unique block-table entries and expand to full tile width.
    # Prefill (TILE_SIZE=64, BLOCK_SIZE=16): 4 unique entries per tile
    # instead of 64, reducing block-table DRAM traffic 16x.
    # Decode (TILE_SIZE=BLOCK_SIZE): num_blocks_in_tile=1, identity.
    num_blocks_in_tile: tl.constexpr = TILE_SIZE // BLOCK_SIZE
    block_ids = tl.arange(0, num_blocks_in_tile)

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        # Batched block-table load + static expansion to TILE_SIZE
        block_base_off = j * num_blocks_in_tile
        unique_blocks = tl.load(
            block_tables_ptr + block_table_offset + block_base_off + block_ids
        ).to(tl.int64)

        # Expand unique block IDs to per-token physical block indices
        physical_block_idx = tl.zeros([TILE_SIZE], dtype=tl.int64)
        for b in tl.static_range(num_blocks_in_tile):
            in_block = offs_t // BLOCK_SIZE == b
            ub = tl.sum(tl.where(block_ids == b, unique_blocks, 0))
            physical_block_idx = tl.where(in_block, ub, physical_block_idx)

        # Compute K/V cache offsets for the paged layout
        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # Streaming cache hint: K/V tiles in paged attention are loaded once
        # per iteration and discarded after dot-product. cache_modifier=".cg"
        # bypasses L1 to prevent one-shot streaming data from polluting the
        # L1 cache, streaming directly to L2.
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-token-head FP8 scales (only used for KV_QUANT_MODE >= 2)
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx,
                mask=tile_mask,
                other=1.0,
                cache_modifier=".cg",
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx,
                mask=tile_mask,
                other=1.0,
                cache_modifier=".cg",
            )

        # Build causal + optional sliding-window mask
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = _compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )

        # Q·K^T — the attention score computation
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = _apply_softcap(S, softcap)

        # Apply causal mask: positions beyond the diagonal get -inf
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        if USE_ALIBI_SLOPES:
            S = _apply_alibi_to_score(
                S,
                alibi_slope,
                seq_offset,
                context_len,
                query_pos,
                USE_ALIBI_SQRT,
            )

        if USE_QQ_BIAS:
            S += _load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        # Online softmax update: rescale accumulator, add weighted V
        M, L, P, alpha = _softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        # Apply sliding-window mask to V for windowed attention
        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        if USE_PER_TOKEN_HEAD_SCALES:
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # Epilogue: normalize by softmax sum and store output
    if IS_3D:
        # 3D flash-decoding: store partial results for inter-segment reduction
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        _store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        # 2D mode: finalize output directly (no inter-segment reduction needed)
        acc = acc / L[:, None]
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


# =============================================================================
# Public API
# =============================================================================


def unified_attention(
    q: torch.Tensor,
    # shape: [total_q_tokens, num_query_heads, head_size]
    k_cache: torch.Tensor,
    # shape: [num_blocks, block_size, num_kv_heads, head_size]
    v_cache: torch.Tensor,
    # shape: [num_blocks, block_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,
    # shape: [num_seqs, max_blocks_per_seq] int32
    seqlens_k: torch.Tensor,
    # shape: [num_seqs] int32, KV sequence lengths
    cu_seqlens_q: torch.Tensor,
    # shape: [num_seqs + 1] int32, cumulative query lengths
    max_seqlen_q: int,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton unified paged attention — prefill + decode with KV cache.

    This function replaces vLLM's original triton_unified_attention with an
    optimized hybrid tuning config (KernelGen v10):
      - Prefill (max_seqlen_q > 1): v0 config (BLOCK_M=64, TILE_SIZE=64,
        num_stages=2) — avoids TILE_SIZE=128 regressions.
      - Decode (max_seqlen_q == 1): v9 config (BLOCK_M=16, TILE_SIZE=32,
        num_stages=4) — deeper pipeline hides HBM latency.

    Args:
        q: Query tensor, [total_tokens, num_query_heads, head_size].
        k_cache: Paged key cache, [num_blocks, block_size, num_kv_heads, head_size].
        v_cache: Paged value cache, same shape as k_cache.
        block_tables: Block table, [num_seqs, max_blocks_per_seq] (int32).
        seqlens_k: Per-sequence KV lengths, [num_seqs] (int32).
        cu_seqlens_q: Cumulative query lengths, [num_seqs + 1] (int32).
        max_seqlen_q: Maximum query length across sequences (>1 = prefill).
        softmax_scale: 1/sqrt(head_size) by default.
        causal: Apply causal mask (required, only causal is supported).
        sliding_window: Sliding window size (None = disabled).
        softcap: Softcap value (None/0 = disabled).
        alibi_slopes: ALiBi slopes [num_query_heads] (None = disabled).

    Returns:
        output: [total_tokens, num_query_heads, head_size]
    """
    logger.debug("GEMS UNIFIED_ATTENTION")

    assert causal, "Only causal attention is supported"
    if softmax_scale is None:
        softmax_scale = q.shape[2] ** -0.5

    block_size = v_cache.shape[1]
    num_seqs = len(seqlens_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    out = torch.empty_like(q)
    head_size_padded = triton.next_power_of_2(head_size)

    # ---- Hybrid tuning: v0 prefill + v9 decode ----
    # Prefill (v0): BLOCK_M=64, TILE_SIZE=64, num_stages=2.
    #   v0's conservative prefill tiles avoid the TILE_SIZE=128 regressions
    #   seen in v8/v9 on prefill-Q256-KV4K and prefill-Q512-KV4K.
    # Decode (v9): BLOCK_M=16, TILE_SIZE=32, num_stages=4.
    #   Deeper 4-stage pipeline hides HBM latency on the latency-bound
    #   decode path (16 KB/stage × 4 = 64 KB shared mem, fits H20 227 KB).
    if max_seqlen_q > 1:
        # Prefill (bandwidth-saturated): moderate tiles for best prefill perf
        BLOCK_M = 64 if num_queries_per_kv <= 16 else 128
        TILE_SIZE = 64
        num_warps = 4
        num_stages = 2
    else:
        # Decode (latency-bound): deeper pipeline to hide HBM latency
        BLOCK_M = (
            16
            if num_queries_per_kv <= 16
            else triton.next_power_of_2(num_queries_per_kv)
        )
        TILE_SIZE = 32
        num_warps = 4
        num_stages = 4

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    # Total q-blocks: each sequence adds 1 for the "sentinel" block
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    grid: tuple[Any, ...] = (total_num_q_blocks, num_kv_heads)

    _unified_attention_kernel[grid](
        num_warps=num_warps,
        num_stages=num_stages,
        # Output pointers (2D mode uses out for all output params)
        output_ptr=out,
        segm_output_ptr=out,
        segm_max_ptr=out,
        segm_expsum_ptr=out,
        # Input pointers
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=q,  # dummy (sinks disabled)
        block_tables_ptr=block_tables,
        seq_lens_ptr=seqlens_k,
        alibi_slopes_ptr=alibi_slopes if alibi_slopes is not None else q,
        qq_bias_ptr=q,  # dummy (Q-Q bias disabled)
        k_scale_cache_ptr=k_cache,  # dummy (per-token scales disabled)
        v_scale_cache_ptr=v_cache,  # dummy
        # Scalars
        scale=softmax_scale,
        k_scale=1.0,
        v_scale=1.0,
        out_scale=1.0,
        softcap=softcap or 0.0,
        # Shape parameters
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        # Feature toggles (all compile-time constants for dead-code elimination)
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_ALIBI_SQRT=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=(softcap or 0.0) > 0,
        USE_SINKS=False,
        SLIDING_WINDOW=sliding_window_val,
        USE_MM_PREFIX=False,
        MAX_MM_RANGES=0,
        mm_prefix_range_ptr=q,  # dummy
        # KV cache strides
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        # Per-token-head scale strides (disabled, dummies)
        stride_ks_blk=0,
        stride_ks_slot=0,
        stride_ks_head=0,
        stride_vs_blk=0,
        stride_vs_slot=0,
        stride_vs_head=0,
        # Query scheduling
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=1,  # 2D mode only
        USE_FP8=False,
        IS_3D=False,
        KV_QUANT_MODE=KVQuantMode.NONE,
        CHUNK_LOOKBACK=-1,
        CHUNK_SIZE=-1,
    )
    return out
