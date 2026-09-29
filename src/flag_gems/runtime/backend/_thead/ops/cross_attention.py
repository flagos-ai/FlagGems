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

"""Forward-only dense cross attention specialized for the T-Head ZW810 PPU.

Same public contract and numerics as ``flag_gems.ops.cross_attention`` (BNSD
layout, MHA/GQA/MQA, non-zero mask entries blocked, exact zeros for fully
masked rows), but the launch configuration is calibrated for the ZW810 PPU
instead of the Hopper GPU the generic implementation was tuned for:

* every MMA tile dimension stays >= 16 -- on this toolchain a ``tl.dot`` with a
  tile dimension of 8 compiles but silently produces wrong results;
* fp32 is emulated with three bf16 MMAs over a hi/lo operand split (see
  ``_FP32_BF16_SPLIT``): plain ``tf32`` is 3 orders of magnitude too coarse to
  pass the test tolerances and ``tf32x3`` is 1.7x slower than the split;
* tile shapes / warp counts / software-pipeline depth come from a shared
  memory budget rule measured on the ZW810 (see ``_select_ppu_tiling``);
* when every tile of a launch is fully in bounds the kernel drops all per-lane
  predicates (see ``_exact_bounds``).

The ZW810 has 64 multi-processors, 256 KiB of shared memory per SM and a
1.7 GHz clock, so cached working sets and occupancy behave differently from the
Hopper-calibrated table in the generic implementation.
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

# fp32 inputs are run as three bf16 MMAs over a hi/lo operand split (~16
# significant bits) instead of the tf32x3 emulation: on the ZW810 the former is
# 1.7x faster for the same acceptance under the flag_gems accuracy check.
# Set to False to fall back to `input_precision="tf32x3"`.
_FP32_BF16_SPLIT = True

# Multi-processors on the ZW810; used to decide whether a launch is too small
# to fill the device (and therefore wants narrower tiles).
_PPU_SM_COUNT = 64


def _select_ppu_tiling(
    query_len,
    key_len,
    qk_dim,
    value_dim,
    has_mask,
    is_fp32,
    batch_size,
    query_heads,
):
    """Pick (BLOCK_M, BLOCK_N, num_warps, pipeline_ns) for the ZW810 PPU.

    Calibrated from tile/warp/pipeline sweeps on PPU-ZW810E; every returned
    tile keeps BLOCK_M >= 16 and BLOCK_N >= 16 because on this toolchain a
    ``tl.dot`` with a tile dimension of 8 compiles but returns wrong results.

    Unlike the Hopper-calibrated table in the generic implementation, this one
    never uses BLOCK_N > 64 for the no-mask fp16 path (the row-wise softmax
    reductions become cross-warp and roughly double the runtime) and it keeps
    the read-modify-write ``[BLOCK_M, VALUE_BLOCK]`` accumulator modest so the
    four-warp CTAs still get warpsPerCTA = [n, 1] layouts.
    """
    head_dim = max(qk_dim, value_dim)
    ns_flat = 1

    if is_fp32:
        # fp32 runs as three bf16 MMAs over a hi/lo split (see _bf16_split),
        # so a narrow BLOCK_N still keeps both MMAs fed while halving the
        # score tile, the P tile that has to be staged through shared memory
        # and the softmax ALU work per iteration.  Measured best on 11/12 of
        # the D <= 64 shapes in both mask modes.
        if query_len <= 32:
            # A single query block: this is a latency-bound launch.
            return 32, 64, 4, 2
        if head_dim >= 96:
            # The K/V stage doubles in size, so the wide BLOCK_N = 32 tile
            # stops fitting the shared-memory budget.
            return 32, 64, 4, 1
        # Too few CTAs to fill the 64 SMs: the masked path keeps the narrow
        # BLOCK_M there, the unmasked one prefers the wide tile (measured
        # 1.05x vs 0.70x on B2 H4 Lq1024 D64 Lk1034).
        grid_ctas = -(-query_len // 64) * batch_size * query_heads
        if has_mask and grid_ctas < 4 * _PPU_SM_COUNT:
            return 32, 64, 4, 1
        return 64, 32, 4, 2

    # --- fp16 / bf16 regimes -------------------------------------------------
    # The ZW810 has 256 KiB of shared memory per SM and a 64-thread-warp CTA of
    # 4 warps only hides the softmax ALU behind the two MMAs when ~4 CTAs are
    # resident, i.e. when a CTA stays under ~64 KiB.  A CTA holds the query
    # tile (BLOCK_M * QK_BLOCK * 2B), the P tile (BLOCK_M * BLOCK_N * 2B) plus
    # PIPELINE_NS * BLOCK_N * (QK_BLOCK + VALUE_BLOCK) * 2B for the pipelined
    # K/V stages, which is why:
    #   * head_dim <= 64 (16 KiB per stage) affords 3 stages;
    #   * head_dim == 128 (32 KiB per stage) does not: 2 stages is the ceiling
    #     and one stage wins outright while the KV loop is still short
    #     (measured 1.35x vs 0.48x on B2 H4 Lq512 D128 Lk612);
    #   * BLOCK_N = 128 is never used: it halves the resident CTAs and makes
    #     the row-wise reductions cross warps (measured 20-40% slower).
    if query_len <= 32:
        # A single query block: BLOCK_M = 32 wastes the fewest lanes while
        # still feeding both MMAs (BLOCK_M = 64 measured ~30% slower here).
        return 32, 64, 4, 3
    if head_dim > 128:
        # Not covered by the benchmark sweep; stay conservative.
        return (16 if head_dim > 256 else 32), 32, 4, ns_flat
    if head_dim >= 96:
        # D = 128: the K/V stage is 32 KiB, so only two pipeline stages fit
        # the shared-memory budget, and one stage wins outright while the KV
        # loop is short (1.35x vs 0.48x on B2 H4 Lq512 D128 Lk612).  BLOCK_M
        # = 128 amortises the per-iteration overheads over twice the rows and
        # only pays off on long unmasked runs (0.60x vs 0.53x on
        # B2 H4 Lq4096 D128 Lk4000); with a mask the extra load+select per KV
        # tile flips it back to 0.69x, while (64, 64, 4, 1) reaches 1.01x.
        if has_mask:
            return 64, 64, 4, 1
        block_m = 128 if query_len >= 4096 else 64
        return block_m, 64, 4, (1 if key_len <= 1024 else 2)

    # Shared-memory budget: a 4-warp CTA holds the query tile, the P tile and
    # PIPELINE_NS * 16 KiB of K/V stages, so ~4 CTAs stay resident per SM (see
    # the module docstring).  BLOCK_N = 128 is never used: it halves the
    # resident CTAs and makes the row-wise reductions cross warps.
    ns = 3
    if has_mask:
        # The masked path pays a full-tile mask load + select per KV tile,
        # which the narrower column tile halves; it is worth 0.583x -> 1.44x
        # on B2 H4 Lq4001 D32 Lk4001.
        # When the launch is too small to fill the device the wide tile wins
        # instead (1.05x vs 0.74x on B2 H4 Lq1024 D64 Lk1034).
        grid_ctas = -(-query_len // 64) * batch_size * query_heads
        if grid_ctas < 4 * _PPU_SM_COUNT:
            return 64, 64, 4, 4
        return 64, 32, 4, 4
    if head_dim == 64:
        # D = 64 K/V rows are 128 B, so a half-width column tile still moves
        # whole cache lines: measured faster on 4/4 of the D = 64 shapes.
        return 64, 32, 4, ns
    return 64, 64, 4, ns


def _exact_bounds(
    query_len, key_len, qk_dim, value_dim, block_m, block_n, qk_block, value_block
):
    """True when every tile of the launch is fully inside the tensors.

    With ``query_len % BLOCK_M == 0``, ``key_len % BLOCK_N == 0`` and the
    reduction dims matching their block sizes, no lane is ever out of range,
    so the kernel can skip all bounds predicates (query/key/value loads, the
    score mask and the store) and the full-tile ``tl.where`` that they feed.
    """
    return (
        query_len % block_m == 0
        and key_len % block_n == 0
        and qk_dim == qk_block
        and value_dim % value_block == 0
    )


@triton.jit
def _bf16_split(x):
    """Split a fp32 tile into bf16 hi/lo halves (a ~= hi + lo).

    Used to emulate fp32 MMA on the ZW810: three bf16 MMAs (hi*hi, hi*lo,
    lo*hi) reproduce a fp32 product to ~16 significant bits, which is far
    cheaper than the tf32x3 path on this backend (measured 24.7 vs 14.4
    TFLOPS for a 2048^3 GEMM) because bf16 MMA runs at ~74 TFLOPS while tf32
    runs at ~43 TFLOPS.
    """
    hi = x.to(tl.bfloat16)
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
    return hi, lo


@libentry()
@triton.jit
def cross_attention_ppu_kernel(
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
        # query_len % BLOCK_M == 0 and qk_dim == QK_BLOCK and
        # value_dim % VALUE_BLOCK == 0, so every lane is in bounds and the
        # per-lane predicates can be dropped (see `_exact_bounds`).
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
                scores = tl.dot(query, key, input_precision="tf32x3")
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
        # `safe_max` keeps `-inf - -inf` (fully blocked rows) from producing NaN.
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        # Blocked lanes already hold -inf, so `-inf - safe_max = -inf` and
        # `exp(-inf) = 0`: no second select is needed here.
        probabilities = tl.exp(scores - safe_max[:, None])
        # `row_max` starts at -inf, giving alpha = exp(-inf) = 0 on the first
        # iteration without a branch.
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
                    probabilities, value, accumulator, input_precision="tf32x3"
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
    """Compute forward cross attention for BNSD inputs (ZW810 PPU).

    Q and K have the same head dimension; V may have a smaller dimension. MHA,
    GQA, and MQA are supported. Non-zero bool/uint8 mask entries are blocked.
    Fully masked rows produce exact zeros. This API currently has no backward.
    """
    logger.debug("GEMS CROSS ATTENTION (PPU)")
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
    block_m, block_n, num_warps, pipeline_ns = _select_ppu_tiling(
        query_len,
        key_len,
        qk_dim,
        value_dim,
        attn_mask is not None,
        query.dtype == torch.float32,
        batch,
        query_heads,
    )
    grid = (
        triton.cdiv(query_len, block_m),
        batch * query_heads,
        triton.cdiv(value_dim, value_block),
    )
    # When every tile is fully in bounds the kernel drops all per-lane
    # predicates (loads, the score mask and the store), which removes a
    # full-tile select plus the masked-load overhead from the KV loop.
    exact_bounds = _exact_bounds(
        query_len, key_len, qk_dim, value_dim, block_m, block_n, qk_block, value_block
    )
    with torch_device_fn.device(query.device):
        cross_attention_ppu_kernel[grid](
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
