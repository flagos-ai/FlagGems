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

import torch
import triton
import triton.language as tl


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
    BATCH_SCAN_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    tid = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    quant_block_id = tl.program_id(1)

    if batch_size <= 16:
        batch_offsets = tl.arange(0, BATCH_SCAN_SIZE)
        batch_mask = batch_offsets < batch_size
        seq_starts = tl.load(cu_seqlen_ptr + batch_offsets, mask=batch_mask, other=0)
        seq_ends = tl.load(cu_seqlen_ptr + batch_offsets + 1, mask=batch_mask, other=0)
        in_batch = (
            (tid[:, None] >= seq_starts[None, :])
            & (tid[:, None] < seq_ends[None, :])
            & batch_mask[None, :]
        )
        batch_id = tl.max(tl.where(in_batch, batch_offsets[None, :], -1), axis=1)
    else:
        left = tl.full((TOKEN_BLOCK,), 0, dtype=tl.int32)
        right = tl.full((TOKEN_BLOCK,), batch_size + 1, dtype=tl.int32)
        for _ in tl.static_range(0, SEARCH_STEPS):
            mid = (left + right) // 2
            seq_start = tl.load(
                cu_seqlen_ptr + mid,
                mask=mid <= batch_size,
                other=2147483647,
            )
            seq_start_before_token = seq_start <= tid
            left = tl.where(seq_start_before_token, mid + 1, left)
            right = tl.where(seq_start_before_token, right, mid)
        batch_id = left - 1
    valid_batch = (batch_id >= 0) & (batch_id < batch_size)
    safe_batch_id = tl.minimum(tl.maximum(batch_id, 0), batch_size - 1)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=valid_batch, other=0)
    batch_end = tl.load(cu_seqlen_ptr + safe_batch_id + 1, mask=valid_batch, other=0)
    valid_tokens = valid_batch & (tid >= batch_start) & (tid < batch_end)
    batch_offset = tid - batch_start
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = safe_batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset, mask=valid_tokens, other=0)

    offsets = quant_block_id * QUANT_BLOCK_SIZE + tl.arange(0, QUANT_BLOCK_SIZE)
    mask = valid_tokens[:, None]
    src_cache_offset = (
        block_id[:, None].to(tl.int64) * kv_cache_stride
        + block_offset[:, None].to(tl.int64) * HEAD_DIM
    )
    src_scale_offset = (
        block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id
    )
    dst_offset = tid[:, None].to(tl.int64) * k_fp8_stride

    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset

    scale_val = tl.load(
        src_scale_ptr,
        mask=valid_tokens,
        other=0.0,
    )
    tl.store(
        k_scale_ptr + tid * num_quant_blocks + quant_block_id,
        scale_val,
        mask=valid_tokens,
    )
    val = tl.load(src_cache_ptr + offsets[None, :], mask=mask)
    tl.store(dst_k_ptr + offsets[None, :], val, mask=mask)


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
    quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    k_cache_flat = k_cache.view(num_blocks, -1)
    k_cache_value = k_cache_flat[:, : block_size * head_dim]
    k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
    k_fp8 = k_fp8.view(torch.uint8)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    batch_size = block_table.shape[0]
    if num_tokens < 32:
        token_block = 1
    elif num_tokens < 64:
        token_block = 2
    elif num_tokens < 128:
        token_block = 4
    elif num_tokens < 256:
        token_block = 8
    elif num_tokens < 512:
        token_block = 16
    else:
        token_block = 32
    if batch_size <= 16:
        batch_scan_size = triton.next_power_of_2(batch_size)
    else:
        # Unused by the binary-search path; kept as a valid constexpr placeholder.
        batch_scan_size = 1
    search_steps = batch_size.bit_length()

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
        batch_scan_size,
        search_steps,
    )
