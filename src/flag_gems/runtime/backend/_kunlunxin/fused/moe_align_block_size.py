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

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@triton.jit
def _pad_topk_ids_kernel(
    topk_ids_ptr,
    topk_ids_padded_ptr,
    numel,
    padded_numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    safe_offsets = tl.minimum(offsets, numel - 1)
    values = tl.load(topk_ids_ptr + safe_offsets)
    values = tl.where(offsets < numel, values, -1)
    tl.store(topk_ids_padded_ptr + offsets, values, mask=offsets < padded_numel)


@triton.jit
def _fill_sorted_token_ids_kernel(
    sorted_token_ids_ptr,
    numel,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(sorted_token_ids_ptr + offsets, value, mask=offsets < numel)


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk_id = pid // num_experts
    expert_id = pid % num_experts
    token_offsets = chunk_id * tokens_per_thread + tl.arange(0, BLOCK_TOKENS)
    token_experts = tl.load(topk_ids_ptr + token_offsets)
    count = tl.sum((token_experts == expert_id).to(tl.int32), axis=0)
    count_offset = (chunk_id + 1) * num_experts + expert_id
    tl.store(tokens_cnts_ptr + count_offset, count)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    tokens_prefix_ptr,
    num_experts: tl.constexpr,
    START_CHUNK: tl.constexpr,
    CHUNK_CHUNKS: tl.constexpr,
):
    expert_id = tl.program_id(0)
    if START_CHUNK == 0:
        prefix = tl.full((), 0, tl.int32)
        tl.store(tokens_prefix_ptr + expert_id, prefix)
    else:
        prefix = tl.load(tokens_prefix_ptr + START_CHUNK * num_experts + expert_id)
    for chunk_offset in tl.static_range(CHUNK_CHUNKS):
        chunk_id = START_CHUNK + chunk_offset
        offset = (chunk_id + 1) * num_experts + expert_id
        prefix += tl.load(tokens_cnts_ptr + offset)
        tl.store(tokens_prefix_ptr + offset, prefix)


@triton.jit
def moe_align_block_size_stage3(
    tokens_prefix_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    num_chunks,
    block_size: tl.constexpr,
    START_EXPERT: tl.constexpr,
    CHUNK_EXPERTS: tl.constexpr,
):
    if START_EXPERT == 0:
        prefix = tl.full((), 0, tl.int32)
        tl.store(cumsum_ptr, prefix)
    else:
        prefix = tl.load(cumsum_ptr + START_EXPERT)
    for expert_offset in tl.static_range(CHUNK_EXPERTS):
        expert_id = START_EXPERT + expert_offset
        final_count_offset = num_chunks * num_experts + expert_id
        count = tl.load(tokens_prefix_ptr + final_count_offset)
        prefix += tl.cdiv(count, block_size) * block_size
        tl.store(cumsum_ptr + expert_id + 1, prefix)


@triton.jit
def moe_align_block_size_store_total(
    total_tokens_post_pad_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
):
    total = tl.load(cumsum_ptr + num_experts)
    tl.store(total_tokens_post_pad_ptr, total)


@triton.jit
def moe_align_block_size_stage4_tokens(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    scratch_ptr,
    tokens_prefix_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk_id = pid // num_experts
    expert_id = pid % num_experts
    chunk_start = chunk_id * tokens_per_thread
    chunk_end = tl.minimum(chunk_start + tokens_per_thread, numel)
    chunk_prefix = tl.load(
        tokens_prefix_ptr + chunk_id * num_experts + expert_id
    )
    expert_start = tl.load(cumsum_ptr + expert_id)
    local_count = 0
    for i in tl.static_range(BLOCK_TOKENS):
        token_offset = chunk_start + i
        valid = token_offset < chunk_end
        token_expert = tl.load(topk_ids_ptr + token_offset)
        matches = valid & (token_expert == expert_id)
        output_offset = expert_start + chunk_prefix + local_count
        output_ptr = tl.where(
            matches,
            sorted_token_ids_ptr + output_offset,
            scratch_ptr + pid,
        )
        tl.store(output_ptr, token_offset)
        local_count += matches.to(tl.int32)


@triton.jit
def moe_align_block_size_stage4_experts(
    expert_ids_ptr,
    scratch_ptr,
    total_tokens_post_pad_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    num_blocks,
    BLOCK_EXPERTS: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * block_size
    total_tokens = tl.load(total_tokens_post_pad_ptr)
    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    valid_experts = expert_offsets < num_experts
    expert_ends = tl.load(
        cumsum_ptr + expert_offsets + 1,
        mask=valid_experts,
        other=total_tokens,
    )
    expert_id = tl.sum(
        (valid_experts & (block_start >= expert_ends)).to(tl.int32), axis=0
    )
    valid_block = (block_id < num_blocks) & (block_start < total_tokens)
    output_expert_id = tl.where(valid_block, expert_id, -1)
    tl.store(expert_ids_ptr + block_id, output_expert_id)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    tokens_per_thread = 256
    num_chunks = ceil_div(numel, tokens_per_thread)
    padded_numel = num_chunks * tokens_per_thread
    topk_ids_flat = topk_ids.reshape(-1)
    if padded_numel == numel:
        topk_ids_padded = topk_ids_flat
    else:
        topk_ids_padded = torch.empty(
            (padded_numel,), dtype=topk_ids.dtype, device=topk_ids.device
        )
        pad_block = 256
        _pad_topk_ids_kernel[(triton.cdiv(padded_numel, pad_block),)](
            topk_ids_flat,
            topk_ids_padded,
            numel,
            padded_numel,
            BLOCK_SIZE=pad_block,
            num_warps=1,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    tokens_cnts = torch.empty(
        (num_chunks + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    tokens_prefix = torch.empty_like(tokens_cnts)
    cumsum = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )
    scratch = torch.empty(
        (max(num_chunks * num_experts, expert_ids.numel()),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    init_block = 256
    _fill_sorted_token_ids_kernel[(triton.cdiv(sorted_token_ids.numel(), init_block),)](
        sorted_token_ids,
        sorted_token_ids.numel(),
        numel,
        BLOCK_SIZE=init_block,
        num_warps=1,
        isCloseVectorization=True,
        buffer_size_limit=2048,
    )

    block_tokens = tokens_per_thread
    moe_align_block_size_stage1[(num_chunks * num_experts,)](
        topk_ids_padded,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
        BLOCK_TOKENS=block_tokens,
        num_warps=1,
        buffer_size_limit=max(2048, block_tokens),
        isCloseVectorization=True,
    )
    block_experts = triton.next_power_of_2(num_experts)
    chunks_per_prefix_chunk = 128
    for start_chunk in range(0, num_chunks, chunks_per_prefix_chunk):
        chunk_chunks = min(chunks_per_prefix_chunk, num_chunks - start_chunk)
        moe_align_block_size_stage2[(num_experts,)](
            tokens_cnts,
            tokens_prefix,
            num_experts,
            START_CHUNK=start_chunk,
            CHUNK_CHUNKS=chunk_chunks,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )
    experts_per_prefix_chunk = 128
    for start_expert in range(0, num_experts, experts_per_prefix_chunk):
        chunk_experts = min(experts_per_prefix_chunk, num_experts - start_expert)
        moe_align_block_size_stage3[(1,)](
            tokens_prefix,
            cumsum,
            num_experts,
            num_chunks,
            block_size,
            START_EXPERT=start_expert,
            CHUNK_EXPERTS=chunk_experts,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )
    moe_align_block_size_store_total[(1,)](
        num_tokens_post_pad,
        cumsum,
        num_experts,
    )
    moe_align_block_size_stage4_tokens[(num_chunks * num_experts,)](
        topk_ids_padded,
        sorted_token_ids,
        scratch,
        tokens_prefix,
        cumsum,
        num_experts,
        numel,
        tokens_per_thread,
        BLOCK_TOKENS=block_tokens,
        num_warps=1,
        buffer_size_limit=max(2048, block_tokens),
        isCloseVectorization=True,
    )
    moe_align_block_size_stage4_experts[(expert_ids.numel(),)](
        expert_ids,
        scratch,
        num_tokens_post_pad,
        cumsum,
        num_experts,
        block_size,
        expert_ids.numel(),
        BLOCK_EXPERTS=block_experts,
        num_warps=1,
        buffer_size_limit=max(2048, block_experts),
        isCloseVectorization=True,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    logger.debug("GEMS_KUNLUNXIN MOE_ALIGN_BLOCK_SIZE")
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad
