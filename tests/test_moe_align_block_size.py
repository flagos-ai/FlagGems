import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# Modified from: https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_moe_align_block_size.py
def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    # if topk_ids.numel() < num_experts:
    #     max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    in_sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )

    # max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    max_num_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.zeros(max_num_blocks, dtype=torch.int32, device=topk_ids.device)

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            in_sorted_token_ids[
                current_pos : current_pos + num_expert_tokens
            ] = expert_tokens

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size

            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[
                current_block : current_block + expert_blocks_needed
            ] = expert_id_new

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    in_num_tokens_post_pad = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=topk_ids.device
    )
    sorted_token_ids.copy_(in_sorted_token_ids)
    experts_ids.copy_(expert_ids)
    num_tokens_post_pad.copy_(in_num_tokens_post_pad)

    return in_sorted_token_ids, expert_ids, num_tokens_post_pad


# ref: https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_moe.py
@pytest.mark.moe_align_block_size
@pytest.mark.parametrize("num_experts", [10, 128, 250, 512])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize(
    "topk_ids_shape",
    [
        (1024, 10),
        (6152, 10),
        (11575, 10),
        (16384, 10),
    ],
)
def test_accuracy_moe_align_block_size(num_experts, block_size, topk_ids_shape):
    device = flag_gems.device
    dtype = torch.int32
    topk_ids = torch.randint(0, num_experts, topk_ids_shape, dtype=dtype, device=device)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=dtype, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    topk_ids_vllm = topk_ids.clone()
    sorted_ids_vllm = sorted_ids.clone()
    expert_ids_vllm = expert_ids.clone()
    num_tokens_post_pad_vllm = num_tokens_post_pad.clone()

    flag_gems.moe_align_block_size_triton(
        topk_ids=topk_ids,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_pad=num_tokens_post_pad,
    )

    torch_moe_align_block_size(
        topk_ids=topk_ids_vllm,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids_vllm,
        experts_ids=expert_ids_vllm,
        num_tokens_post_pad=num_tokens_post_pad_vllm,
    )

    def _group_tokens_by_expert(
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        block_size: int,
        valid_length: int,
        total_tokens: int,
    ) -> dict:
        num_blocks = valid_length // block_size
        expert_tokens: dict[int, list[int]] = {}

        for block_idx in range(num_blocks):
            expert_id = expert_ids[block_idx].item()
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, valid_length)

            block_tokens = sorted_ids[block_start:block_end]
            valid_tokens = block_tokens[block_tokens < total_tokens]

            if expert_id not in expert_tokens:
                expert_tokens[expert_id] = []
            expert_tokens[expert_id].extend(valid_tokens.tolist())
        return expert_tokens

    def _verify_expert_level_sorting(
        actual_sorted_ids: torch.Tensor,
        golden_sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        block_size: int,
        valid_length: int,
        total_tokens: int,
    ):
        """
        Verify that actual_sorted_ids follows the correct expert-level sorting.
        The kerne limplementation may or may not preserve original token order
        in topk_ids in the final sorted_ids however this does not impact quality.
        """
        # Group tokens by expert from the golden implementation
        golden_expert_tokens = _group_tokens_by_expert(
            golden_sorted_ids, expert_ids, block_size, valid_length, total_tokens
        )

        actual_expert_tokens = _group_tokens_by_expert(
            actual_sorted_ids, expert_ids, block_size, valid_length, total_tokens
        )

        assert set(golden_expert_tokens.keys()) == set(actual_expert_tokens.keys()), (
            f"Expert IDs mismatch: golden={set(golden_expert_tokens.keys())}, "
            f"actual={set(actual_expert_tokens.keys())}"
        )

        for expert_id in golden_expert_tokens:
            golden_tokens = torch.tensor(
                golden_expert_tokens[expert_id], device=actual_sorted_ids.device
            )
            actual_tokens = torch.tensor(
                actual_expert_tokens[expert_id], device=actual_sorted_ids.device
            )
            assert torch.equal(
                torch.sort(golden_tokens)[0], torch.sort(actual_tokens)[0]
            ), (
                f"Expert {expert_id} token mismatch: "
                f"golden={golden_expert_tokens[expert_id]}, "
                f"actual={actual_expert_tokens[expert_id]}"
            )

    if flag_gems.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()

    _verify_expert_level_sorting(
        sorted_ids,
        sorted_ids_vllm,
        expert_ids_vllm,
        block_size,
        num_tokens_post_pad.item(),
        topk_ids.numel(),
    )
    utils.gems_assert_close(
        expert_ids, utils.to_reference(expert_ids_vllm), dtype=dtype
    )
    utils.gems_assert_close(
        num_tokens_post_pad, utils.to_reference(num_tokens_post_pad_vllm), dtype=dtype
    )
