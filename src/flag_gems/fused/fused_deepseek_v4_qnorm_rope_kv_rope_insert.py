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

"""Fused DeepseekV4 QNorm+RoPE+KV RoPE Insert Triton Kernel (BF16).
# KernelGen

Fuses three operations for DeepSeek-V4 decode-phase inference:
  Q side:  per-head RMSNorm (no weight) + GPT-J RoPE on last 64 dims
  KV side: GPT-J RoPE on last 64 dims + bf16 paged cache insert (all 512 dims)

Uses single-store pattern: builds complete output in fp32 registers,
then stores once. num_warps=2 for faster cross-warp reduction,
num_stages=4 for better pipelining.
"""

import logging

import torch  # noqa: F401
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_qkv_kernel(
    q_ptr,
    kv_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    stride_q_tok,
    stride_q_head,
    stride_kv_tok,
    stride_cache_block,
    stride_cache_token,
    stride_cos_sin_pos,
    eps,
    num_q_items,
    total_items,
    NUM_HEADS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    item_id = pid

    while item_id < total_items:
        if item_id < num_q_items:
            # ═══ Q path: RMSNorm (no weight) + GPT-J RoPE ═══
            tok_idx = item_id // NUM_HEADS
            head_idx = item_id % NUM_HEADS
            base = tok_idx * stride_q_tok + head_idx * stride_q_head

            # Load all 512 dims bf16 → fp32 (single contiguous load)
            # HEAD_DIM=512 hardcoded for DeepSeek-V4
            offs = tl.arange(0, 512)
            x = tl.load(q_ptr + base + offs).to(tl.float32)

            # RMSNorm: rsqrt(mean(x²) + eps)
            # HEAD_DIM=512 hardcoded
            sq_sum = tl.sum(x * x, axis=0)
            rsqrt_val = tl.math.rsqrt(sq_sum / 512.0 + eps)
            x_normed = x * rsqrt_val

            # GPT-J RoPE: apply rotation to normed values in-register
            pos = tl.load(position_ids_ptr + tok_idx)
            # HALF_ROPE_DIM=32 hardcoded (ROPE_DIM=64)
            half_offs = tl.arange(0, 32)
            cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
            sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)

            # Build rope-applied output for last 64 dims in-register
            # NOPE_DIM=448 hardcoded
            rope_even = 448 + half_offs * 2
            rope_odd = rope_even + 1

            # Gather normed even/odd values from original input + multiply
            # by rsqrt (avoids scatter issue with x_normed)
            x_re = tl.load(q_ptr + base + rope_even).to(tl.float32) * rsqrt_val
            x_ro = tl.load(q_ptr + base + rope_odd).to(tl.float32) * rsqrt_val

            q_out_e = x_re * cos - x_ro * sin
            q_out_o = x_re * sin + x_ro * cos

            # Store: first 448 dims (normed, no RoPE) + rotated rope dims
            # NOPE_DIM=448 hardcoded
            nope_mask = offs < 448
            tl.store(q_ptr + base + offs, x_normed.to(tl.bfloat16), mask=nope_mask)
            tl.store(q_ptr + base + rope_even, q_out_e.to(tl.bfloat16))
            tl.store(q_ptr + base + rope_odd, q_out_o.to(tl.bfloat16))

        else:
            # ═══ KV path: RoPE + bf16 paged cache insert ═══
            kv_idx = item_id - num_q_items
            slot_id = tl.load(slot_mapping_ptr + kv_idx)

            if slot_id >= 0:
                kv_base = kv_idx * stride_kv_tok

                # Load all 512 dims
                # HEAD_DIM=512 hardcoded
                offs = tl.arange(0, 512)
                kv_data = tl.load(kv_ptr + kv_base + offs)

                # GPT-J RoPE on last 64 dims [448, 512)
                pos = tl.load(position_ids_ptr + kv_idx)
                # HALF_ROPE_DIM=32 hardcoded
                half_offs = tl.arange(0, 32)
                cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
                sin = tl.load(
                    cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs
                )

                # NOPE_DIM=448 hardcoded
                rope_even = 448 + half_offs * 2
                rope_odd = rope_even + 1

                x_e = tl.load(kv_ptr + kv_base + rope_even).to(tl.float32)
                x_o = tl.load(kv_ptr + kv_base + rope_odd).to(tl.float32)
                out_e = x_e * cos - x_o * sin
                out_o = x_e * sin + x_o * cos

                # Paged cache offset
                block_idx = slot_id // CACHE_BLOCK_SIZE
                pos_in_block = slot_id % CACHE_BLOCK_SIZE
                cache_off = (
                    block_idx * stride_cache_block + pos_in_block * stride_cache_token
                )

                # Store NoPE part (first 448 dims)
                # NOPE_DIM=448 hardcoded
                nope_mask = offs < 448
                tl.store(k_cache_ptr + cache_off + offs, kv_data, mask=nope_mask)

                # Store RoPE part (last 64 dims rotated)
                tl.store(k_cache_ptr + cache_off + rope_even, out_e.to(tl.bfloat16))
                tl.store(k_cache_ptr + cache_off + rope_odd, out_o.to(tl.bfloat16))

        item_id += tl.num_programs(0)


def fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps=1e-6,
    cache_block_size=16,
):
    """Fused QNorm+RoPE (Q) and RoPE+Insert (KV), BF16 variant.

    Args:
        q: [N, H, 512] bfloat16, modified in-place (RMSNorm + RoPE).
        kv: [N, 512] bfloat16, input KV data.
        k_cache: [num_blocks, block_size, 512] bfloat16, paged KV cache.
        slot_mapping: [N_insert] int64, slot indices for cache insertion.
        position_ids: [N] int64, position indices for RoPE.
        cos_sin_cache: [max_pos, 64] float32, precomputed cos||sin cache.
        eps: RMSNorm epsilon (default 1e-6).
        cache_block_size: KV cache page size (default 16).
    """
    logger.debug("GEMS FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_INSERT")

    N = q.shape[0]
    H = q.shape[1]
    N_ins = slot_mapping.shape[0]

    total_q = N * H
    total = total_q + N_ins

    if total == 0:
        return

    # Grid capped at 65536 for large workloads
    grid = min(total, 65536)

    _fused_qkv_kernel[(grid,)](
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        cos_sin_cache.stride(0),
        eps,
        total_q,
        total,
        H,
        cache_block_size,
        num_warps=2,
        num_stages=4,
    )
