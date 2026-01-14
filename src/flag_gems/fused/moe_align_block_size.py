import logging
from typing import Optional

import torch
import triton
import triton.language as tl
# import triton.experimental.tle.language.gpu as tle
import triton.experimental.tle.language as tle


# # shared_memory
# aaa = tl.load(tokens_cnts_ptr + off_c)
# tle.gpu.memory_space(aaa, "shared_memory")
# tl.debug_barrier()

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@triton.jit(do_not_specialize=["numel", "tokens_per_thread"])
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


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
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit(do_not_specialize=["numel", "tokens_per_thread"])
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)

@triton.jit(do_not_specialize=["numel", "tokens_per_thread"])
def moe_align_block_size_kernel(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr, # 512
    numel: tl.constexpr, # 163840
    tokens_per_thread, # (163840 + 512 - 1) // 512 = 320
    cumsum_ptr,
    block_size: tl.constexpr, # 64
    sorted_token_ids_ptr, # output
    expert_ids_ptr, # output
    total_tokens_post_pad_ptr, # output
    sync_point_ptr_0,
    sync_point_ptr_1,
    sync_point_ptr_2,
):
    pid = tl.program_id(0)

    if True:
        start_idx_0 = pid * tokens_per_thread

        off_c = (pid + 1) * num_experts
        
        for i in range(tokens_per_thread):
            # i: [0:1:320)
            if start_idx_0 + i < numel:
                idx = tl.load(topk_ids_ptr + start_idx_0 + i)
                token_cnt_0 = tl.load(tokens_cnts_ptr + off_c + idx)
                tl.store(tokens_cnts_ptr + off_c + idx, token_cnt_0 + 1)
            if i == tokens_per_thread - 1:
                tl.atomic_add(sync_point_ptr_0, 1, sem="acq_rel", scope="sys")

    # tl.debug_barrier()  # --------------------------------------------------------------------------

    stage1 = False
    while not stage1:
        if tl.load(sync_point_ptr_0) >= num_experts:
            tl.device_print(" ")
            stage1 = True

            last_cnt = 0
            for i in range(1, num_experts + 1):
                # i: [1:1:512]
                token_cnt_1 = tl.load(tokens_cnts_ptr + i * num_experts + pid)
                last_cnt = last_cnt + token_cnt_1
                tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)
                if i == num_experts:
                    tl.atomic_add(sync_point_ptr_1, 1, sem="acq_rel", scope="sys")

    # tl.debug_barrier()  # --------------------------------------------------------------------------

    stage2 = False
    while not stage2:
        if tl.load(sync_point_ptr_1) >= num_experts:
            tl.device_print(" ")
            stage2 = True

            last_cumsum = 0
            off_cnt = num_experts * num_experts
            for i in range(1, num_experts + 1):
                # i: [1:1:512]
                token_cnt_2 = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
                last_cumsum = last_cumsum + tl.cdiv(token_cnt_2, block_size) * block_size
                tl.store(cumsum_ptr + i, last_cumsum)
                if i == num_experts:
                    tl.atomic_add(sync_point_ptr_2, 1, sem="acq_rel", scope="sys")
            tl.store(total_tokens_post_pad_ptr, last_cumsum)

    # tl.debug_barrier()  # --------------------------------------------------------------------------

    stage3 = False
    while not stage3:
        if tl.load(sync_point_ptr_2) >= num_experts:
            tl.device_print(" ")
            stage3 = True

            start_idx_1 = tl.load(cumsum_ptr + pid)
            end_idx = tl.load(cumsum_ptr + pid + 1)

            for i in range(start_idx_1, end_idx, block_size):
                tl.store(expert_ids_ptr + i // block_size, pid)

            start_idx_1 = pid * tokens_per_thread
            off_t = pid * num_experts

            for i in range(start_idx_1, tl.minimum(start_idx_1 + tokens_per_thread, numel)):
                expert_id = tl.load(topk_ids_ptr + i)
                token_cnt_3 = tl.load(tokens_cnts_ptr + off_t + expert_id)
                rank_post_pad = token_cnt_3 + tl.load(cumsum_ptr + expert_id)
                tl.store(sorted_token_ids_ptr + rank_post_pad, i)
                tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt_3 + 1)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    # The tensor needs to be padded before calculating IDs,
    # to prevent out-of-bounds address access.
    sorted_token_ids.fill_(numel)
    expert_ids.fill_(0)

    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    sync_point_0 = torch.zeros(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    sync_point_1 = torch.zeros(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    sync_point_2 = torch.zeros(
        (1,), dtype=torch.int32, device=topk_ids.device
    )

    # print(f"\n")
    # print("before:")
    # print(f"sync_point_0:{sync_point_0}")
    # print(f"sync_point_1:{sync_point_1}")
    # print(f"sync_point_2:{sync_point_2}")
    # print(f"num_experts:{num_experts}")
    moe_align_block_size_kernel[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
        cumsum,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        sync_point_0,
        sync_point_1,
        sync_point_2,
    )
    # print("after:")
    # print(f"sync_point_0:{sync_point_0}")
    # print(f"sync_point_1:{sync_point_1}")
    # print(f"sync_point_2:{sync_point_2}")
    # print(f"num_experts:{num_experts}")

    # moe_align_block_size_stage1[grid](
    #     topk_ids,
    #     tokens_cnts,
    #     num_experts,
    #     numel,
    #     tokens_per_thread,
    # )
    # moe_align_block_size_stage2[grid](
    #     tokens_cnts,
    #     num_experts,
    # )
    # moe_align_block_size_stage3[(1,)](
    #     num_tokens_post_pad,
    #     tokens_cnts,
    #     cumsum,
    #     num_experts,
    #     block_size,
    # )
    # moe_align_block_size_stage4[grid](
    #     topk_ids,
    #     sorted_token_ids,
    #     expert_ids,
    #     tokens_cnts,
    #     cumsum,
    #     num_experts,
    #     block_size,
    #     numel,
    #     tokens_per_thread,
    # )


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
