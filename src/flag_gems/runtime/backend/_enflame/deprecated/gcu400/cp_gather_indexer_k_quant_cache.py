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

# Adapted from vLLM v0.20.2:
# csrc/cache_kernels.cu::cp_gather_indexer_k_quant_cache_kernel
#
# GCU (gcu400) adaptation notes:
# - The generic implementation resolves the owning sequence with a
#   ``tl.static_range`` binary search once batch_size > 16. That unrolled
#   pointer chase aborts the gcu-compiler (SIGABRT inside make_gcuir) on
#   batch_size=33, so the batch lookup here always uses the single-shot scan
#   form: load the whole cu_seqlen prefix, then reduce the one-hot ownership
#   matrix. Cost is O(BATCH_SCAN) per tile and BATCH_SCAN <= 256, which is far
#   below the tile copy cost.
# - i32 addptr chains truncate element offsets >= 2**31, and paged KV caches
#   beyond that size are routine, so every cache-side offset folds into one
#   i64 expression before the load (kv_cache_stride is already an element
#   count of the flattened cache row).
# - The 128-byte quant blocks are naturally aligned: block_offset * HEAD_DIM
#   and quant_block_id * QUANT_BLOCK_SIZE are both multiples of 128, which is
#   advertised to the vectorizer with tl.multiple_of.

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


@libentry()
@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,
    kv_cache_scale_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    k_fp8_stride,
    num_quant_blocks,
    batch_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    BATCH_SCAN: tl.constexpr,
):
    token_ids = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    quant_block_id = tl.program_id(1)

    # One-shot owner lookup: every lane compares its token id against the whole
    # prefix-sum range table instead of binary searching it.
    batch_offsets = tl.arange(0, BATCH_SCAN)
    batch_mask = batch_offsets < batch_size
    seq_starts = tl.load(cu_seqlen_ptr + batch_offsets, mask=batch_mask, other=0)
    seq_ends = tl.load(
        cu_seqlen_ptr + batch_offsets + 1,
        mask=batch_mask,
        other=2147483647,
    )
    # batch_mask must stay in the conjunction: the phantom lanes up to
    # BATCH_SCAN would otherwise own every token id and win the tl.max.
    owned = (
        (token_ids[:, None] >= seq_starts[None, :])
        & (token_ids[:, None] < seq_ends[None, :])
        & batch_mask[None, :]
    )
    batch_id = tl.max(tl.where(owned, batch_offsets[None, :], -1), axis=1)

    # Tokens past cu_seqlen[-1] (padding rows of the destination) stay untouched.
    active = batch_id >= 0
    safe_batch_id = tl.maximum(batch_id, 0)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=active, other=0)
    batch_offset = token_ids - batch_start
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_id = tl.load(
        block_table_ptr + safe_batch_id * block_table_stride + block_table_id,
        mask=active,
        other=0,
    )

    inner = tl.arange(0, QUANT_BLOCK_SIZE)
    cache_row_start = block_id[:, None].to(tl.int64) * kv_cache_stride + (
        block_offset[:, None].to(tl.int64) * HEAD_DIM
    )
    quant_offset = quant_block_id * QUANT_BLOCK_SIZE
    mask = active[:, None]

    values = tl.load(
        kv_cache_ptr + cache_row_start + quant_offset + inner[None, :],
        mask=mask,
    )
    dst_rows = token_ids[:, None].to(tl.int64) * k_fp8_stride + quant_offset
    tl.store(k_fp8_ptr + dst_rows + inner[None, :], values, mask=mask)

    scale = tl.load(
        kv_cache_scale_ptr
        + block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id,
        mask=active,
        other=0.0,
    )
    tl.store(
        k_scale_ptr + token_ids * num_quant_blocks + quant_block_id,
        scale,
        mask=active,
    )


@libentry()
@triton.jit
def _cp_gather_indexer_quant_cache_words_kernel(
    cache_words_ptr,
    out_words_ptr,
    out_scale_words_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    block_size,
    block_table_stride,
    block_words,
    out_stride_words,
    batch_size: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    BATCH_SCAN: tl.constexpr,
    WORDS: tl.constexpr,
    SCALE_WORDS: tl.constexpr,
):
    """Copy whole quant rows as i32 words.

    One program owns a token tile and copies the full head_dim of every token
    in it, so the batch lookup and the block-table load are amortised over
    head_dim bytes instead of one quant block. The rows are i32-aligned (the
    wrapper only takes this path when head_dim and the cache row length are
    multiples of 4), which lets a single lane cover 4 bytes and keeps the
    per-row store contiguous.
    """
    token_ids = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    batch_offsets = tl.arange(0, BATCH_SCAN)
    batch_mask = batch_offsets < batch_size
    seq_starts = tl.load(cu_seqlen_ptr + batch_offsets, mask=batch_mask, other=0)
    seq_ends = tl.load(
        cu_seqlen_ptr + batch_offsets + 1,
        mask=batch_mask,
        other=2147483647,
    )
    owned = (
        (token_ids[:, None] >= seq_starts[None, :])
        & (token_ids[:, None] < seq_ends[None, :])
        & batch_mask[None, :]
    )
    batch_id = tl.max(tl.where(owned, batch_offsets[None, :], -1), axis=1)
    active = batch_id >= 0
    safe_batch_id = tl.maximum(batch_id, 0)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=active, other=0)
    batch_offset = token_ids - batch_start
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_id = tl.load(
        block_table_ptr + safe_batch_id * block_table_stride + block_table_id,
        mask=active,
        other=0,
    )

    mask = active[:, None]
    row_words = block_id[:, None] * block_words + block_offset[:, None] * WORDS
    inner = tl.arange(0, WORDS)
    inner = tl.max_contiguous(tl.multiple_of(inner, 4), WORDS)
    values = tl.load(cache_words_ptr + row_words + inner[None, :], mask=mask)
    out_rows = token_ids[:, None] * out_stride_words
    tl.store(out_words_ptr + out_rows + inner[None, :], values, mask=mask)

    scale_ids = tl.arange(0, SCALE_WORDS)
    scale_base = (
        block_id[:, None] * block_words
        + block_size * WORDS
        + block_offset[:, None] * SCALE_WORDS
    )
    scale = tl.load(cache_words_ptr + scale_base + scale_ids[None, :], mask=mask)
    tl.store(
        out_scale_words_ptr + token_ids[:, None] * SCALE_WORDS + scale_ids[None, :],
        scale,
        mask=mask,
    )


def _token_block_size(num_tokens):
    """Token tile of the word kernel.

    The word kernel spends one batch lookup per token tile, so wide tiles pay
    off as long as the grid still fills the machine: 128 tokens per program
    measured 441us at 65k tokens against 2890us for a 16-token tile, and the
    small-token band is launch bound rather than tile bound.
    """
    if num_tokens < 2048:
        return 32
    if num_tokens < 16384:
        return 128
    return 256


def cp_gather_indexer_k_quant_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    num_scale_bytes = k_fp8_scale.shape[-1]
    quant_block_size = head_dim * 4 // num_scale_bytes
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    k_cache_flat = k_cache.view(num_blocks, -1)
    k_cache_value = k_cache_flat[:, : block_size * head_dim]
    k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
    k_fp8 = k_fp8.view(torch.uint8)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    batch_size = block_table.shape[0]
    if num_tokens == 0:
        return

    token_block = _token_block_size(num_tokens)
    batch_scan = triton.next_power_of_2(batch_size)

    use_words = (
        head_dim % 4 == 0
        and num_scale_bytes % 4 == 0
        and (block_size * head_dim) % 4 == 0
        and k_cache_flat.stride(0) % 4 == 0
    )
    if use_words:
        cache_words = k_cache.view(num_blocks, -1).view(torch.int32)
        grid = (triton.cdiv(num_tokens, token_block),)
        _cp_gather_indexer_quant_cache_words_kernel[grid](
            cache_words,
            k_fp8.view(torch.int32),
            k_fp8_scale.view(torch.int32),
            block_table,
            cu_seqlen,
            block_size,
            block_table_stride,
            cache_words.stride(0),
            head_dim // 4,
            batch_size,
            token_block,
            batch_scan,
            head_dim // 4,
            num_scale_bytes // 4,
        )
        return

    grid = (triton.cdiv(num_tokens, token_block), num_quant_blocks)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        k_fp8.stride(0),
        num_quant_blocks,
        batch_size,
        head_dim,
        quant_block_size,
        token_block,
        batch_scan,
    )
