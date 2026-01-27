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


def next_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit(do_not_specialize=["numel", "tokens_per_thread"])
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    # Unroll loop by 4 for better instruction-level parallelism
    UNROLL: tl.constexpr = 4
    num_full_iters = tokens_per_thread // UNROLL

    # Process full iterations with unrolling
    for iter_idx in range(num_full_iters):
        base_i = iter_idx * UNROLL
        for unroll_i in range(UNROLL):
            i = base_i + unroll_i
            if start_idx + i < numel:
                idx = tl.load(topk_ids_ptr + start_idx + i)
                token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
                tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)

    # Handle remaining iterations
    for i in range(num_full_iters * UNROLL, tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2_vec(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    cnt = tl.cumsum(token_cnt, axis=0)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, cnt)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):# Vectorized implementation using tl.cumsum
    off_cnt = num_experts * num_experts

    # Load all expert token counts at once
    expert_offsets = tl.arange(0, num_experts)
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets)

    # Align each count to block_size boundary (vectorized)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    # Compute cumulative sum (exclusive prefix sum, then shift)
    cumsum_values = tl.cumsum(aligned_cnts, axis=0)

    # Store cumsum[1:num_experts+1] = cumulative sums
    # cumsum[0] is already 0 (initialized outside)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values)

    # Store total tokens post pad (last element of cumsum)
    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    offset = tl.arange(0, tokens_per_thread) + start_idx
    mask = offset < numel
    expert_id = tl.load(topk_ids_ptr + offset, mask=mask)
    token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id, mask=mask)
    rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id, mask=mask)
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset)
    tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)
    

    # for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
    #     expert_id = tl.load(topk_ids_ptr + i)
    #     token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
    #     rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
    #     tl.store(sorted_token_ids_ptr + rank_post_pad, i)
    #     tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    numel_sorted_token_ids = sorted_token_ids.numel()
    numel_expert_ids = expert_ids.numel()
    # The tensor needs to be padded before calculating IDs,
    # to prevent out-of-bounds address access.

    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = next_power_of_2(ceil_div(numel, num_experts))

    block_size_sorted = next_power_of_2(ceil_div(numel_sorted_token_ids, num_experts))
    block_size_expert = next_power_of_2(ceil_div(numel_expert_ids, num_experts))

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
        sorted_token_ids,
        expert_ids,
        numel_sorted_token_ids,
        numel_expert_ids,
        block_size_sorted,
        block_size_expert,
    )
    if num_experts == next_power_of_2(num_experts):
        moe_align_block_size_stage2_vec[grid](
            tokens_cnts,
            num_experts,
        )
    else:
        moe_align_block_size_stage2[grid](
            tokens_cnts,
            num_experts,
        )
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    logger.debug("GEMS MOE ALIGN BLOCK SIZE")
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
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
