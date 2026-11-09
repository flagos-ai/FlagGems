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

"""Forward-only dense cross attention specialized for the Moore Threads MUSA.

Same public contract and numerics as ``flag_gems.ops.cross_attention`` (BNSD
layout, MHA/GQA/MQA, non-zero mask entries blocked, exact zeros for fully masked
rows), but the launch configuration and the precision path are calibrated for
the MTT S5000 (MUSA) instead of the Hopper GPU the generic implementation was
tuned for:

* the generic implementation drives its K/V loop with
  ``triton.experimental.tle.language.gpu.pipeline``, which the MUSA Triton
  backend rejects ("Only `range` and `static_range` iterators are currently
  supported"), so this kernel uses a plain ``tl.range`` loop;
* fp32 is emulated with the backend-native ``bf16x3`` input precision, which the
  backend lowers to three bf16 MMAs over a hi/lo operand split.  Plain ``tf32``
  is far too coarse on this backend, and the ``ieee``/``tf32x3`` paths either
  fall back to a slow FFMA loop or exceed the shared-memory budget;
* every global load stays predicate free.  On the S5000 a load masked by a
  computed tensor is scalarized, so the tail tile -- whose key lanes run past
  ``key_len`` -- *clamps* its addresses instead of predicating them and
  neutralizes the duplicated lanes in the scores.  The launcher rounds the key
  axis up to a whole ``BLOCK_N`` tile only when the K/V loop is short enough
  that the four padding kernels stay cheap.  Either way the key axis is aligned,
  which the mask needs because a ``[B, 1, Lq, Lk]`` bool mask is byte addressed.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.cross_attention import _mask_strides, _validate_inputs
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _select_musa_tiling(
    query_len, key_len, qk_dim, value_dim, has_mask, is_fp32, batch_size, query_heads
):
    """Clamp the calibrated table to the sequence lengths actually present.

    A tile taller than the query sequence, or wider than the key sequence, only
    wastes tensor-core lanes: the MMA still runs over the whole tile while only
    ``query_len`` rows (or ``key_len`` keys) are real, and a key tile wider than
    ``key_len`` also forces the launcher to pad K/V up to the tile width first.
    """
    block_m, block_n, num_warps, pipeline_ns, trans_k = _table_musa_tiling(
        query_len,
        key_len,
        qk_dim,
        value_dim,
        has_mask,
        is_fp32,
        batch_size,
        query_heads,
    )
    if key_len < block_n:
        block_n = max(16, triton.next_power_of_2(key_len))
    if query_len < block_m:
        block_m = max(16, triton.next_power_of_2(query_len))
        num_warps = min(num_warps, block_m // 2)
    return block_m, block_n, num_warps, pipeline_ns, trans_k


def _table_musa_tiling(
    query_len,
    key_len,
    qk_dim,
    value_dim,
    has_mask,
    is_fp32,
    batch_size,
    query_heads,
):
    """Pick (BLOCK_M, BLOCK_N, num_warps, pipeline_ns, trans_k) on the S5000.

    Calibrated by sweeping the tile/warp/pipeline space on the MTT S5000 (56 SMs,
    192 KiB of shared memory per CTA).  Two effects dominate:

    * **K stage orientation (`trans_k`).**  Loading the K tile as
      ``[BLOCK_N, QK_BLOCK]`` and letting the backend fold ``tl.trans`` into the
      MMA operand load is worth 1.2-1.6x over the direct ``[QK_BLOCK, BLOCK_N]``
      load, and it is what makes the wide column tiles pay off.
    * **Warp layout of the score tile.**  The row-wise softmax reductions stay
      warp-local only while the key axis is not split across warps, which wants
      ``BLOCK_M >= 8 * num_warps``.

    ``tl.exp`` is faster than ``tl.math.exp2`` here, so the softmax keeps
    ``tl.exp``.  The masked path is a separate regime: the per-tile mask load and
    select move the optimum, and for fp32 D = 128 the wide tile no longer fits
    the 192 KiB budget together with the mask tile.

    NOTE: every entry was picked by timing ONE config per fresh process.  The
    MUSA Triton runtime reuses a compiled kernel when only ``num_warps`` differs
    (it is missing from the cache key), so a sweep that loops over ``num_warps``
    in one process silently re-measures whichever variant was compiled first.
    """
    head_dim = max(qk_dim, value_dim)

    if has_mask:
        # --- masked regimes --------------------------------------------------
        if is_fp32:
            if head_dim >= 64:
                return 128, 32, 16, 3, 0
            if head_dim >= 32:
                if key_len >= 8192:
                    return 128, 32, 16, 3, 0
                return 64, 32, 8, 4, 0
            # D <= 16.
            if key_len >= 8192:
                return 32, 32, 4, 4, 0
            return 128, 64, 8, 2, 0
        if head_dim >= 96:
            # D = 128. 
            if key_len <= 768 and 256 <= query_len <= 1024:
                return 64, 64, 8, 2, 1
            if query_len <= 32:
                return 32, 128, 16, 1, 1
            return 64, 64, 8, 1, 1
        if head_dim >= 64:
            if query_len <= 32:
                return 32, 128, 8, 2, 1
            return 64, 32, 8, 2, 1
        if head_dim >= 32:
            return 64, 64, 8, 2, 1
        # D <= 16.
        if key_len >= 8192:
            return 32, 32, 4, 4, 0
        return 64, 32, 4, 4, 0

    # --- unmasked regimes ----------------------------------------------------
    if is_fp32:
        if head_dim > 128:
            return 64, 32, 8, 3, 0
        if head_dim >= 96:
            # D128.
            return 128, 64, 16, 2, 0
        if head_dim >= 64:
            if key_len >= 2048:
                return 64, 32, 8, 3, 0
            return 128, 32, 16, 3, 0
        if head_dim >= 32:
            return 64, 32, 8, 3, 0
        # D <= 16.
        return 128, 64, 8, 2, 0

    if head_dim > 128:
        return 128, 64, 16, 1, 1
    if head_dim >= 96:
        if query_len <= 32:
            return 64, 256, 16, 1, 1
        if key_len <= 2048 and batch_size * query_heads <= 32:
            if query_len <= 512:
                return 32, 128, 8, 1, 1
            return 64, 128, 8, 1, 1
        return 64, 64, 8, 1, 1
    if head_dim >= 64:
        if query_len <= 32:
            return 32, 128, 8, 1, 1
        return 64, 64, 8, 1, 1
    if head_dim >= 32:
        # D32: single stage, trans_k.
        return 64, 64, 8, 1, 1
    if key_len >= 8192:
        return 64, 64, 4, 2, 1
    return 64, 32, 4, 1, 0


@triton.jit
def _cross_attention_key_tile(
    accumulator,
    row_max,
    row_sum,
    query,
    valid_queries,
    qk_offsets,
    value_offsets,
    K,
    V,
    AttnMask,
    scale,
    key_base,
    value_base,
    mask_base,
    stride_k_seq,
    stride_k_dim,
    stride_v_seq,
    stride_v_dim,
    stride_mask_key,
    lane_keys,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    QK_DIM: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TAIL: tl.constexpr,
    PAD: tl.constexpr,
    TRANS_K: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    """Fold one ``BLOCK_N``-wide key tile into the online softmax state.

    Shared by the vectorized main loop and the peeled ``TAIL`` iteration so the
    two stay numerically identical.  ``TAIL`` marks the peeled final tile, whose
    key lanes run past ``KEY_LEN`` and therefore clamp their addresses; ``PAD``
    marks a tile that carries such lanes with unpredicated loads.
    """
    if TAIL:
        # Clamp instead of predicating: a load masked by a computed tensor is
        # scalarized on the S5000, which costs far more than re-reading a key
        # already in the tile.  The duplicated lanes are neutralized below.
        keys_row = tl.minimum(lane_keys, KEY_LEN - 1)
    else:
        keys_row = lane_keys

    if HAS_MASK:
        mask_ptrs = AttnMask + mask_base + keys_row[None, :] * stride_mask_key
        blocked = tl.load(mask_ptrs) != 0
    if TRANS_K:
        k_ptrs = (
            K
            + key_base
            + keys_row[:, None] * stride_k_seq
            + qk_offsets[None, :] * stride_k_dim
        )
        k_lane_ok = (qk_offsets < QK_DIM)[None, :]
    else:
        k_ptrs = (
            K
            + key_base
            + keys_row[None, :] * stride_k_seq
            + qk_offsets[:, None] * stride_k_dim
        )
        k_lane_ok = (qk_offsets < QK_DIM)[:, None]
    if QK_DIM == QK_BLOCK:
        key = tl.load(k_ptrs)
    else:
        key = tl.load(k_ptrs, mask=k_lane_ok, other=0.0)
    if TRANS_K:
        key = tl.trans(key)
    if IS_FP32:
        # bf16x3 splits each operand into bf16 hi/lo and issues three bf16 MMAs
        # with an fp32 accumulator, emulating near-fp32 precision.
        scores = tl.dot(query, key, input_precision="bf16x3")
    else:
        scores = tl.dot(query, key, allow_tf32=False)
    scores *= scale
    if HAS_MASK:
        scores = tl.where(blocked, -float("inf"), scores)
    if TAIL or PAD:
        # Lanes past KEY_LEN are only in the tile because the key axis had to be
        # rounded up; left alone they would score like a real key.
        scores = tl.where(lane_keys[None, :] < KEY_LEN, scores, -float("inf"))
    if QUERY_LEN < BLOCK_M:
        scores = tl.where(valid_queries[:, None], scores, -float("inf"))
    tile_max = tl.max(scores, axis=1)
    new_max = tl.maximum(row_max, tile_max)
    # `safe_max` keeps `-inf - -inf` (fully blocked rows) from producing NaN.
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    # Blocked and padding lanes already hold -inf, so `exp(-inf) = 0` and no
    # second select is needed.
    probabilities = tl.exp(scores - safe_max[:, None])
    # `row_max` starts at -inf, so the first iteration needs no branch: alpha
    # comes out as exp(-inf) = 0.
    alpha = tl.exp(row_max - safe_max)
    row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
    v_ptrs = (
        V
        + value_base
        + keys_row[:, None] * stride_v_seq
        + value_offsets[None, :] * stride_v_dim
    )
    if VALUE_DIM % VALUE_BLOCK == 0:
        value = tl.load(v_ptrs)
    else:
        value = tl.load(v_ptrs, mask=(value_offsets < VALUE_DIM)[None, :], other=0.0)
    accumulator *= alpha[:, None]
    if IS_FP32:
        accumulator = tl.dot(
            probabilities, value, accumulator, input_precision="bf16x3"
        )
    else:
        p = probabilities.to(value.dtype)
        accumulator = tl.dot(p, value, accumulator, allow_tf32=False)
    return accumulator, new_max, row_sum


@libentry()
@triton.jit
def cross_attention_musa_kernel(
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
    KEY_FULL: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    HAS_PAD: tl.constexpr,
    IS_FP32: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_NS: tl.constexpr,
    TRANS_K: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    value_tile = tl.program_id(2)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)

    key_offsets = tl.arange(0, BLOCK_N)
    qk_offsets = tl.arange(0, QK_BLOCK)
    value_offsets = value_tile * VALUE_BLOCK + tl.arange(0, VALUE_BLOCK)
    query_lane_raw = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    if QUERY_LEN % BLOCK_M == 0:
        query_lane = query_lane_raw
        valid_queries = query_lane_raw < QUERY_LEN
    elif QUERY_LEN >= BLOCK_M:
        query_lane = tl.minimum(query_tile * BLOCK_M, QUERY_LEN - BLOCK_M) + tl.arange(
            0, BLOCK_M
        )
        valid_queries = query_lane >= query_tile * BLOCK_M
    else:
        query_lane = tl.minimum(query_lane_raw, QUERY_LEN - 1)
        valid_queries = query_lane_raw < QUERY_LEN

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
    if QK_DIM == QK_BLOCK:
        query = tl.load(
            Q
            + query_base
            + query_lane[:, None] * stride_q_seq
            + qk_offsets[None, :] * stride_q_dim
        )
    else:
        query = tl.load(
            Q
            + query_base
            + query_lane[:, None] * stride_q_seq
            + qk_offsets[None, :] * stride_q_dim,
            mask=(qk_offsets < QK_DIM)[None, :],
            other=0.0,
        )

    mask_base = 0
    if HAS_MASK:
        mask_base = (
            batch.to(tl.int64) * stride_mask_batch
            + query_head.to(tl.int64) * stride_mask_head
            + query_lane[:, None] * stride_mask_query
        )
    accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    if KEY_FULL > 0:
        for key_start in tl.range(0, KEY_FULL, BLOCK_N, num_stages=PIPELINE_NS):
            accumulator, row_max, row_sum = _cross_attention_key_tile(
                accumulator,
                row_max,
                row_sum,
                query,
                valid_queries,
                qk_offsets,
                value_offsets,
                K,
                V,
                AttnMask,
                scale,
                key_base,
                value_base,
                mask_base,
                stride_k_seq,
                stride_k_dim,
                stride_v_seq,
                stride_v_dim,
                stride_mask_key,
                key_start + key_offsets,
                QUERY_LEN=QUERY_LEN,
                KEY_LEN=KEY_LEN,
                QK_DIM=QK_DIM,
                QK_BLOCK=QK_BLOCK,
                VALUE_DIM=VALUE_DIM,
                VALUE_BLOCK=VALUE_BLOCK,
                HAS_MASK=HAS_MASK,
                BLOCK_M=BLOCK_M,
                TAIL=False,
                PAD=HAS_PAD,
                TRANS_K=TRANS_K,
                IS_FP32=IS_FP32,
            )
    if HAS_TAIL:
        accumulator, row_max, row_sum = _cross_attention_key_tile(
            accumulator,
            row_max,
            row_sum,
            query,
            valid_queries,
            qk_offsets,
            value_offsets,
            K,
            V,
            AttnMask,
            scale,
            key_base,
            value_base,
            mask_base,
            stride_k_seq,
            stride_k_dim,
            stride_v_seq,
            stride_v_dim,
            stride_mask_key,
            KEY_FULL + key_offsets,
            QUERY_LEN=QUERY_LEN,
            KEY_LEN=KEY_LEN,
            QK_DIM=QK_DIM,
            QK_BLOCK=QK_BLOCK,
            VALUE_DIM=VALUE_DIM,
            VALUE_BLOCK=VALUE_BLOCK,
            HAS_MASK=HAS_MASK,
            BLOCK_M=BLOCK_M,
            TAIL=True,
            PAD=True,
            TRANS_K=TRANS_K,
            IS_FP32=IS_FP32,
        )

    output = accumulator * tl.where(row_sum > 0.0, 1.0 / row_sum, 0.0)[:, None]
    o_ptrs = (
        Out
        + output_base
        + query_lane[:, None] * stride_o_seq
        + value_offsets[None, :] * stride_o_dim
    )
    if QUERY_LEN % BLOCK_M == 0:
        if VALUE_DIM % VALUE_BLOCK == 0:
            tl.store(o_ptrs, output)
        else:
            tl.store(o_ptrs, output, mask=(value_offsets < VALUE_DIM)[None, :])
    else:
        tl.store(
            o_ptrs,
            output,
            mask=valid_queries[:, None] & (value_offsets < VALUE_DIM)[None, :],
        )


def _pad_axis(tensor, dim, length):
    """Return ``tensor`` zero-padded along ``dim`` up to ``length``.

    A fresh contiguous buffer is produced on purpose: the padding is what gives
    the kernel an aligned key axis and the mask load vectorized rows.
    """
    size = tensor.shape[dim]
    if size >= length:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = length
    padded = tensor.new_zeros(shape)
    padded.narrow(dim, 0, size).copy_(tensor)
    return padded

_MASK_ALIGN = 16

_PEEL_MAX_ROUNDS = 16


def cross_attention(query, key, value, attn_mask=None, scale=None):
    """Compute forward cross attention for BNSD inputs (Moore Threads MUSA).

    Q and K have the same head dimension; V may have a smaller dimension. MHA,
    GQA, and MQA are supported. Non-zero bool/uint8 mask entries are blocked.
    Fully masked rows produce exact zeros. This API currently has no backward.
    """
    logger.debug("GEMS_MTHREADS CROSS_ATTENTION")
    _validate_inputs(query, key, value, attn_mask, scale)
    batch, query_heads, query_len, qk_dim = query.shape
    kv_heads, key_len, value_dim = key.shape[1], key.shape[2], value.shape[3]
    softmax_scale = 1.0 / math.sqrt(qk_dim) if scale is None else float(scale)
    output = torch.empty(
        (batch, query_heads, query_len, value_dim),
        dtype=query.dtype,
        device=query.device,
    )
    qk_block = max(16, triton.next_power_of_2(qk_dim))
    value_block = max(16, triton.next_power_of_2(min(value_dim, 128)))
    is_fp32 = query.dtype == torch.float32
    block_m, block_n, num_warps, pipeline_ns, trans_k = _select_musa_tiling(
        query_len,
        key_len,
        qk_dim,
        value_dim,
        attn_mask is not None,
        is_fp32,
        batch,
        query_heads,
    )
    # Cover the `key_len % BLOCK_N` keys the main loop does not, either by
    # peeling one tail tile (`key_full = floor`, no K/V padding and no extra
    # device kernels) or by rounding the key axis up to `key_loop_pad`
    # (`key_full = ceil`, four padding launches plus up to BLOCK_N - 1 wasted keys
    # per tile).  Peeling wins on short loops; on long ones the tail's own operand
    # staging buffer can move the kernel onto a different shared-memory carveout
    # and costs far more than the launches.  fp32 never peels: its bf16x3 dots
    # already sit on the 192 KiB budget, which the tail's buffer overflows.
    key_loop_pad = -(-key_len // block_n) * block_n
    key_full = key_loop_pad
    has_tail = False
    has_pad = key_loop_pad != key_len
    if has_pad and not is_fp32 and key_len // block_n <= _PEEL_MAX_ROUNDS:
        key_full = (key_len // block_n) * block_n
        has_tail = True
        has_pad = False
    if has_pad:
        key = _pad_axis(key, 2, key_loop_pad)
        value = _pad_axis(value, 2, key_loop_pad)
    if attn_mask is None:
        mask_arg = query
        mask_strides = (0, 0, 0, 0)
    else:
        # The mask is the one tensor that always wants padding: a row that does
        # not start on a 16-byte boundary loses all vectorization on its load.
        mask_len = -(-key_len // _MASK_ALIGN) * _MASK_ALIGN
        if has_pad:
            mask_len = max(mask_len, key_loop_pad)
        if attn_mask.shape[-1] == key_len and mask_len != key_len:
            attn_mask = _pad_axis(attn_mask, attn_mask.ndim - 1, mask_len)
        mask_arg = attn_mask
        mask_strides = _mask_strides(attn_mask)
    grid = (
        triton.cdiv(query_len, block_m),
        batch * query_heads,
        triton.cdiv(value_dim, value_block),
    )
    with torch_device_fn.device(query.device):
        cross_attention_musa_kernel[grid](
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
            KEY_FULL=key_full,
            HAS_TAIL=has_tail,
            HAS_PAD=has_pad,
            IS_FP32=is_fp32,
            QK_DIM=qk_dim,
            VALUE_DIM=value_dim,
            QK_BLOCK=qk_block,
            VALUE_BLOCK=value_block,
            HAS_MASK=attn_mask is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            PIPELINE_NS=pipeline_ns,
            TRANS_K=trans_k,
            num_stages=pipeline_ns,
            num_warps=num_warps,
        )
    return output


__all__ = ["cross_attention"]
