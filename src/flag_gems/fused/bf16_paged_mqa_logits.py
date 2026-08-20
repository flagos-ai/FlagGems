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

"""BF16 Paged MQA Logits Triton Kernel.
# KernelGen

Computes multi-head weighted ReLU attention logits on paged KV cache:
  logits[b*next_n+n, pos] = sum_h( relu(q[b,n,h,:] @ K[pos,:]) * w[b*next_n+n, h] )

Uses shape-specialized kernels (_kernel_H32_D128, _kernel_H64_D128) with
hardcoded shape constants as source-level literals for zero-constexpr dispatch.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# H=32, D=128, block_size=64 — shape constants as source-level literals
# for zero-constexpr function params and trivial dispatch cache
@triton.jit
def _kernel_H32_D128(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_ctx,
    stride_bt,
):
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    # block_size=64 hardcoded for H=32 specialization
    kv_pos = pid_blk * 64
    if kv_pos >= ctx_len:
        return

    # H=32, D=128 hardcoded for this specialization
    h_range = tl.arange(0, 32)
    d_range = tl.arange(0, 128)
    # block_size=64 hardcoded
    pos_range = tl.arange(0, 64)

    # Load Q [32, 128] bf16 → transpose to [128, 32]
    q_base = pid_row * (32 * 128)
    q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
    q_block = tl.load(q_ptr + q_offs, eviction_policy="evict_last")
    q_t = tl.trans(q_block)

    # Load weights [32] fp32
    w = tl.load(weights_ptr + pid_row * 32 + h_range, eviction_policy="evict_last")

    # Load K [64, 128] bf16 from paged cache
    b_idx = pid_row // next_n
    phys_blk = tl.load(block_table_ptr + b_idx * stride_bt + pid_blk)
    # block_size=64, D=128 hardcoded
    k_offs = phys_blk * (64 * 128) + pos_range[:, None] * 128 + d_range[None, :]
    k_block = tl.load(kv_cache_ptr + k_offs, eviction_policy="evict_first")

    # GEMM: K[64,128] @ Q^T[128,32] → scores[64,32] fp32
    scores = tl.dot(k_block, q_t)

    # ReLU + weighted sum → logits[64] fp32
    scores = tl.maximum(scores, 0.0)
    logits_val = tl.sum(scores * w[None, :], axis=1)

    # Store (mask only the last partial block)
    out_base = pid_row * max_ctx + kv_pos
    # block_size=64 hardcoded
    if kv_pos + 64 <= ctx_len:
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            eviction_policy="evict_first",
        )
    else:
        mask = pos_range < (ctx_len - kv_pos)
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            mask=mask,
            eviction_policy="evict_first",
        )


# H=64, D=128, block_size=64 — shape constants as source-level literals
@triton.jit
def _kernel_H64_D128(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_ctx,
    stride_bt,
):
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    # block_size=64 hardcoded for H=64 specialization
    kv_pos = pid_blk * 64
    if kv_pos >= ctx_len:
        return

    # H=64, D=128 hardcoded for this specialization
    h_range = tl.arange(0, 64)
    d_range = tl.arange(0, 128)
    # block_size=64 hardcoded
    pos_range = tl.arange(0, 64)

    q_base = pid_row * (64 * 128)
    q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
    q_block = tl.load(q_ptr + q_offs, eviction_policy="evict_last")
    q_t = tl.trans(q_block)

    w = tl.load(weights_ptr + pid_row * 64 + h_range, eviction_policy="evict_last")

    b_idx = pid_row // next_n
    phys_blk = tl.load(block_table_ptr + b_idx * stride_bt + pid_blk)
    # block_size=64, D=128 hardcoded
    k_offs = phys_blk * (64 * 128) + pos_range[:, None] * 128 + d_range[None, :]
    k_block = tl.load(kv_cache_ptr + k_offs, eviction_policy="evict_first")

    scores = tl.dot(k_block, q_t)
    scores = tl.maximum(scores, 0.0)
    logits_val = tl.sum(scores * w[None, :], axis=1)

    out_base = pid_row * max_ctx + kv_pos
    # block_size=64 hardcoded
    if kv_pos + 64 <= ctx_len:
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            eviction_policy="evict_first",
        )
    else:
        mask = pos_range < (ctx_len - kv_pos)
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            mask=mask,
            eviction_policy="evict_first",
        )


def bf16_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_metadata,
    max_context_len,
    clean_logits=False,
    logits_dtype=torch.float32,
):
    """BF16 Paged MQA Logits — Triton implementation.

    Computes weighted ReLU attention logits on paged KV cache:
      logits[row, pos] = sum_h( relu(q[b,n,h,:] · K[pos,:]) * w[row, h] )

    Args:
        q: [B, next_n, H, D] bfloat16 query tensor.
        kv_cache: [num_blocks, block_size, 1, D] bfloat16 paged KV cache.
        weights: [B*next_n, H] float32 per-head weights.
        context_lens: [B, next_n] int32 context lengths.
        block_table: [B, max_blocks] int32 block table mapping.
        schedule_metadata: [num_sms+1, 2] int32 (unused, kept for API
            compatibility with vLLM).
        max_context_len: Maximum context length.
        clean_logits: If True, set positions beyond context length to -inf.
        logits_dtype: Output dtype (default float32).

    Returns:
        Logits tensor [total_tokens, max_context_len] in logits_dtype.
    """
    logger.debug("GEMS BF16_PAGED_MQA_LOGITS")

    B, next_n, H, D = q.shape
    total_tokens = B * next_n

    logits = torch.empty(
        total_tokens,
        max_context_len,
        dtype=logits_dtype,
        device=q.device,
    )

    if total_tokens == 0 or max_context_len == 0:
        return logits

    # block_size=64 hardcoded for both specializations
    num_kv_blocks = (max_context_len + 63) >> 6
    grid = (total_tokens, num_kv_blocks)
    stride_bt = block_table.stride(0)

    # H=32 specialization: num_warps=4 for smaller head count
    # H=64 specialization: num_warps=8 for larger head count
    if H == 32:
        _kernel_H32_D128[grid](
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            logits,
            next_n,
            max_context_len,
            stride_bt,
            num_warps=4,
            num_stages=1,
        )
    else:
        _kernel_H64_D128[grid](
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            logits,
            next_n,
            max_context_len,
            stride_bt,
            num_warps=8,
            num_stages=1,
        )

    if clean_logits:
        for b in range(B):
            for n in range(next_n):
                row_idx = b * next_n + n
                ctx_len = int(context_lens[b, n].item())
                if ctx_len < max_context_len:
                    logits[row_idx, ctx_len:] = float("-inf")

    return logits
