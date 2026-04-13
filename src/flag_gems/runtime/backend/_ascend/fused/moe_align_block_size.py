import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.fused.{__name__.split(".")[-1]}')


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@libentry()
@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    tokens_per_thread_sub: tl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_sorted_sub: tl.constexpr,
    block_size_expert: tl.constexpr,
    block_size_expert_sub: tl.constexpr,
):
    pid = tle.program_id(0)

    # Initialize sorted_token_ids in sub-blocks to avoid UB overflow
    for sub_off in range(0, block_size_sorted, block_size_sorted_sub):
        offsets_sorted = pid * block_size_sorted + sub_off + tl.arange(0, block_size_sorted_sub)
        mask_sorted = offsets_sorted < numel_sorted_token_ids
        tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    # Initialize expert_ids in sub-blocks
    for sub_off in range(0, block_size_expert, block_size_expert_sub):
        offsets_expert = pid * block_size_expert + sub_off + tl.arange(0, block_size_expert_sub)
        mask_expert = offsets_expert < numel_expert_ids
        tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    # Ascend NPU: avoid scatter-atomic, use per-expert vector counting instead
    # Use float32 for comparison as Ascend Vector CMP doesn't support int32 natively
    # Process in sub-blocks to avoid UB overflow
    for sub_off in range(0, tokens_per_thread, tokens_per_thread_sub):
        offsets = start_idx + sub_off + tl.arange(0, tokens_per_thread_sub)
        mask = offsets < numel
        expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=num_experts).to(tl.int32)
        expert_id_f = expert_id.to(tl.float32)
        for e in range(num_experts):
            e_mask = (expert_id_f == (e + 0.0)) & mask
            count = tl.sum(e_mask.to(tl.int32))
            prev = tl.load(tokens_cnts_ptr + off_c + e)
            tl.store(tokens_cnts_ptr + off_c + e, prev + count)


@libentry()
@triton.jit
def moe_align_block_size_stage2_vec(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    cnt = tl.cumsum(token_cnt, axis=0)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, cnt)


@libentry()
@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tle.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@libentry()
@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    num_experts_next_power_of_2: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts

    expert_offsets = tl.arange(0, num_experts_next_power_of_2)
    mask = expert_offsets < num_experts
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets, mask=mask)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values, mask=mask)

    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)


@libentry()
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
    tokens_per_thread_sub: tl.constexpr,
):
    pid = tle.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    # Ascend NPU: avoid scatter-atomic, use per-expert cumsum ranking instead
    # Process in sub-blocks to avoid UB overflow
    for sub_off in range(0, tokens_per_thread, tokens_per_thread_sub):
        offsets = tl.arange(0, tokens_per_thread_sub) + start_idx + sub_off
        mask = offsets < numel
        expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=num_experts).to(tl.int32)

        # Use float32 for comparison as Ascend Vector CMP doesn't support int32 natively
        expert_id_f = expert_id.to(tl.float32)
        for e in range(num_experts):
            e_mask = (expert_id_f == (e + 0.0)) & mask

            # Compute exclusive prefix sum for ranking within this expert
            e_cumsum = tl.cumsum(e_mask.to(tl.int32), axis=0)
            e_rank = e_cumsum - 1  # 0-based rank

            base_rank = tl.load(tokens_cnts_ptr + off_t + e)
            cum_offset = tl.load(cumsum_ptr + e)

            rank_post_pad = tl.where(e_mask, base_rank + e_rank + cum_offset, 0)
            tl.store(sorted_token_ids_ptr + rank_post_pad, offsets, mask=e_mask)

            # Update base count for this expert
            total_count = tl.sum(e_mask.to(tl.int32))
            tl.store(tokens_cnts_ptr + off_t + e, base_rank + total_count)


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

    grid = (num_experts,)
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))

    block_size_sorted = triton.next_power_of_2(
        ceil_div(numel_sorted_token_ids, num_experts)
    )
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))

    # Sub-block sizes to avoid UB overflow on Ascend NPU (UB = 192KB)
    MAX_SUB_BLOCK = 1024
    tokens_per_thread_sub = min(tokens_per_thread, MAX_SUB_BLOCK)
    block_size_sorted_sub = min(block_size_sorted, MAX_SUB_BLOCK)
    block_size_expert_sub = min(block_size_expert, MAX_SUB_BLOCK)

    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    num_experts_next_power_of_2 = triton.next_power_of_2(num_experts)

    with torch_device_fn.device(topk_ids.device):
        moe_align_block_size_stage1[grid](
            topk_ids,
            tokens_cnts,
            num_experts,
            numel,
            tokens_per_thread,
            tokens_per_thread_sub,
            sorted_token_ids,
            expert_ids,
            numel_sorted_token_ids,
            numel_expert_ids,
            block_size_sorted,
            block_size_sorted_sub,
            block_size_expert,
            block_size_expert_sub,
        )
        if num_experts == triton.next_power_of_2(num_experts):
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
            num_experts_next_power_of_2,
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
            tokens_per_thread_sub,
        )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    logger.debug("GEMS_ASCEND MOE_ALIGN_BLOCK_SIZE")
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
