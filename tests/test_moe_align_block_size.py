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

import importlib

import pytest
import torch

import flag_gems
from flag_gems.fused.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_compact,
    moe_align_block_size_ep_route_block,
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)

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
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            in_sorted_token_ids[current_pos : current_pos + num_expert_tokens] = (
                expert_tokens
            )

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size

            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[current_block : current_block + expert_blocks_needed] = (
                expert_id_new
            )

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
    The kernel implementation may or may not preserve original token order in
    topk_ids in the final sorted_ids, but this does not impact correctness.
    """
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


def _synchronize():
    if flag_gems.vendor_name == "ascend":
        torch.npu.synchronize()
    elif flag_gems.vendor_name == "sunrise":
        from flag_gems.runtime import torch_device_fn

        torch_device_fn.synchronize()
    else:
        torch.cuda.synchronize()


# ref: https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_moe.py
@pytest.mark.moe_align_block_size_triton
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

    if flag_gems.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        from flag_gems.runtime import torch_device_fn

        torch_device_fn.synchronize()

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


@pytest.mark.moe_align_block_size_triton
@pytest.mark.parametrize(
    ("num_experts", "block_size", "topk_ids_shape"),
    [
        (512, 64, (16384, 10)),
        (512, 64, (6152, 10)),
        (512, 64, (4727, 10)),
        (512, 64, (1905, 10)),
        (512, 64, (11575, 10)),
        (512, 64, (1032, 10)),
        (512, 64, (4201, 10)),
        (512, 64, (2056, 10)),
        (512, 64, (7561, 10)),
        (512, 64, (4104, 10)),
        (512, 64, (14281, 10)),
    ],
)
def test_accuracy_moe_align_block_size_triton(num_experts, block_size, topk_ids_shape):
    device = flag_gems.device
    dtype = torch.int32
    topk_ids = torch.randint(0, num_experts, topk_ids_shape, dtype=dtype, device=device)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=dtype, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    topk_ids_ref = topk_ids.clone()
    sorted_ids_ref = sorted_ids.clone()
    expert_ids_ref = expert_ids.clone()
    num_tokens_post_pad_ref = num_tokens_post_pad.clone()

    flag_gems.moe_align_block_size_triton(
        topk_ids=topk_ids,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_pad=num_tokens_post_pad,
    )

    torch_moe_align_block_size(
        topk_ids=topk_ids_ref,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids_ref,
        experts_ids=expert_ids_ref,
        num_tokens_post_pad=num_tokens_post_pad_ref,
    )

    _synchronize()

    _verify_expert_level_sorting(
        sorted_ids,
        sorted_ids_ref,
        expert_ids_ref,
        block_size,
        num_tokens_post_pad.item(),
        topk_ids.numel(),
    )
    utils.gems_assert_close(expert_ids, utils.to_reference(expert_ids_ref), dtype=dtype)
    utils.gems_assert_close(
        num_tokens_post_pad, utils.to_reference(num_tokens_post_pad_ref), dtype=dtype
    )


@pytest.mark.moe_align_block_size
@pytest.mark.parametrize("fast_path", ["singleton", "small_grouped"])
def test_accuracy_moe_align_block_size_fast_paths(fast_path):
    device = flag_gems.device
    block_size = 8
    num_experts = 8

    if fast_path == "singleton":
        topk_ids = torch.tensor([[1, 3]], dtype=torch.int32, device=device)
        actual = moe_align_block_size_singleton(topk_ids, block_size)
    else:
        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]],
            dtype=torch.int32,
            device=device,
        )
        actual = moe_align_block_size_small_grouped(topk_ids, num_experts, block_size)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    expected = (
        torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device),
        torch.empty(
            max_num_tokens_padded // block_size,
            dtype=torch.int32,
            device=device,
        ),
        torch.empty(1, dtype=torch.int32, device=device),
    )
    torch_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        expected[0],
        expected[1],
        expected[2],
    )

    num_tokens = actual[2].item()
    num_blocks = num_tokens // block_size
    torch.testing.assert_close(actual[0][:num_tokens], expected[0][:num_tokens])
    torch.testing.assert_close(actual[1][:num_blocks], expected[1][:num_blocks])
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="compact MoE alignment is currently enabled only on NVIDIA SM90",
)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ("num_experts", "topk_ids_shape", "routing", "pad_sorted_ids"),
    [
        (512, (64, 10), "uniform", False),
        (512, (64, 10), "skewed", False),
        (257, (64, 10), "uniform", False),
        (511, (64, 10), "uniform", False),
        (256, (1, 1), "uniform", False),
        (512, (128, 8), "uniform", False),
        (512, (128, 8), "skewed", False),
        (512, (1, 641), "uniform", True),
    ],
)
def test_accuracy_moe_align_block_size_compact_qwen4(
    num_experts, topk_ids_shape, routing, pad_sorted_ids, id_dtype
):
    device = flag_gems.device
    block_size = 16
    num_routes = topk_ids_shape[0] * topk_ids_shape[1]

    if routing == "uniform":
        topk_ids = torch.randint(
            0,
            num_experts,
            topk_ids_shape,
            dtype=id_dtype,
            device=device,
        )
    else:
        # Exercise MAX_BLOCKS_PER_EXPERT by sending every route to one expert.
        topk_ids = torch.full(topk_ids_shape, 17, dtype=id_dtype, device=device)

    actual = moe_align_block_size_compact(
        topk_ids, block_size, num_experts, pad_sorted_ids
    )
    selected = moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        pad_sorted_ids=pad_sorted_ids,
    )
    if pad_sorted_ids:
        assert actual[0].numel() % block_size == 0
        assert selected[0].numel() % block_size == 0

    max_num_tokens_padded = num_routes + num_experts * (block_size - 1)
    expected = (
        torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device),
        torch.empty(
            max_num_tokens_padded // block_size,
            dtype=torch.int32,
            device=device,
        ),
        torch.empty(1, dtype=torch.int32, device=device),
    )
    torch_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        expected[0],
        expected[1],
        expected[2],
    )
    _synchronize()

    total_tokens = expected[2].item()
    num_blocks = total_tokens // block_size
    for output in (actual, selected):
        _verify_expert_level_sorting(
            output[0],
            expected[0],
            expected[1],
            block_size,
            total_tokens,
            num_routes,
        )
        torch.testing.assert_close(output[1][:num_blocks], expected[1][:num_blocks])
        torch.testing.assert_close(output[2], expected[2])


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="compact MoE alignment is currently enabled only on NVIDIA SM90",
)
def test_qwen4_exact_dispatch_prefers_compact_when_tle_is_available(monkeypatch):
    """The measured Qwen4 point must not regress when FlagTree exposes TLE."""
    module = importlib.import_module("flag_gems.fused.moe_align_block_size")
    compact = module.moe_align_block_size_compact
    compact_launches = 0

    def counted_compact(*args, **kwargs):
        nonlocal compact_launches
        compact_launches += 1
        return compact(*args, **kwargs)

    monkeypatch.setattr(module, "HAS_TLE", True)
    monkeypatch.setattr(module, "moe_align_block_size_compact", counted_compact)
    topk_ids = torch.randint(
        0,
        512,
        (64, 10),
        dtype=torch.int32,
        device=flag_gems.device,
    )
    module.moe_align_block_size(topk_ids, block_size=16, num_experts=512)
    _synchronize()

    assert compact_launches == 1


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Decode compact alignment is enabled only on NVIDIA SM90",
)
def test_compact_alignment_ignores_out_of_range_int64_experts():
    device = flag_gems.device
    num_experts, block_size = 512, 16
    torch.manual_seed(20260824)
    topk_ids = torch.randint(0, num_experts, (64, 10), dtype=torch.int64, device=device)
    topk_ids.view(-1)[:3] = torch.tensor(
        [-1, num_experts, 2**32 + 3], dtype=torch.int64, device=device
    )

    actual = moe_align_block_size(topk_ids, block_size, num_experts)
    max_padded = topk_ids.numel() + num_experts * (block_size - 1)
    expected = (
        torch.empty(max_padded, dtype=torch.int32, device=device),
        torch.empty(max_padded // block_size, dtype=torch.int32, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
    )
    torch_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        expected[0],
        expected[1],
        expected[2],
    )
    _synchronize()

    total_tokens = int(expected[2].item())
    num_blocks = total_tokens // block_size
    _verify_expert_level_sorting(
        actual[0],
        expected[0],
        expected[1],
        block_size,
        total_tokens,
        topk_ids.numel(),
    )
    torch.testing.assert_close(actual[1][:num_blocks], expected[1][:num_blocks])
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="compact fused-MoE EP alignment is enabled only on NVIDIA SM90",
)
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize(
    "routing",
    ["uniform", "all_local", "no_local", "skewed", "invalid", "int64_overflow"],
)
def test_accuracy_moe_align_block_size_ep(block_size, routing):
    device = flag_gems.device
    global_experts = 288
    local_experts = 18
    topk_shape = (96, 8)
    num_routes = topk_shape[0] * topk_shape[1]
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    # Use a non-zero, contiguous global shard to exercise the global-to-local
    # mapping instead of relying on global ID == local ID.
    shard_begin = 7 * local_experts
    expert_map[shard_begin : shard_begin + local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=device
    )

    if routing == "uniform":
        torch.manual_seed(20260824)
        topk_ids = torch.randint(
            0, global_experts, topk_shape, dtype=torch.int32, device=device
        )
    elif routing == "all_local":
        topk_ids = (
            torch.arange(num_routes, dtype=torch.int32, device=device)
            .remainder(local_experts)
            .add(shard_begin)
            .view(topk_shape)
        )
    elif routing == "no_local":
        topk_ids = torch.zeros(topk_shape, dtype=torch.int32, device=device)
    elif routing == "skewed":
        topk_ids = torch.zeros(topk_shape, dtype=torch.int32, device=device)
        topk_ids[:, 0] = shard_begin + 3
    elif routing == "invalid":
        topk_ids = torch.full(topk_shape, -1, dtype=torch.int32, device=device)
        topk_ids[:, 0] = shard_begin + 3
    else:
        topk_ids = torch.full(
            topk_shape,
            2**32 + shard_begin + 3,
            dtype=torch.int64,
            device=device,
        )
        topk_ids[:, 0] = shard_begin + 3

    actual = moe_align_block_size(
        topk_ids,
        block_size,
        global_experts,
        expert_map,
        ignore_invalid_experts=True,
        local_num_experts=local_experts,
    )
    max_global_padded = num_routes + global_experts * (block_size - 1)
    expected = (
        torch.empty(max_global_padded, dtype=torch.int32, device=device),
        torch.empty(max_global_padded // block_size, dtype=torch.int32, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
    )
    torch_moe_align_block_size(
        topk_ids,
        global_experts,
        block_size,
        expected[0],
        expected[1],
        expected[2],
        expert_map,
    )
    _synchronize()

    total_tokens = int(expected[2].item())
    num_blocks = total_tokens // block_size
    assert actual[0].numel() == num_routes + local_experts * (block_size - 1)
    _verify_expert_level_sorting(
        actual[0],
        expected[0],
        expected[1],
        block_size,
        total_tokens,
        num_routes,
    )
    torch.testing.assert_close(actual[1][:num_blocks], expected[1][:num_blocks])
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.moe_align_block_size
def test_moe_align_block_size_rejects_noncontiguous_routing_inputs():
    device = flag_gems.device
    topk_ids = torch.zeros(96, 8, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="block_size must be positive"):
        moe_align_block_size(topk_ids, 0, 288)
    with pytest.raises(ValueError, match="num_experts must be positive"):
        moe_align_block_size(topk_ids, 16, 0)

    expanded_ids = torch.zeros(1, 8, device=device, dtype=torch.int32).expand(96, 8)
    expert_map = torch.arange(288, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="topk_ids must be contiguous"):
        moe_align_block_size(expanded_ids, 16, 288, expert_map)

    expanded_map = torch.zeros(1, device=device, dtype=torch.int32).expand(288)
    with pytest.raises(ValueError, match="expert_map must be contiguous"):
        moe_align_block_size(topk_ids, 16, 288, expanded_map)


def _route_block_reference(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    local_num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact stable route-block layout consumed by grouped GEMM."""
    routes = topk_ids.flatten().cpu().tolist()
    mapping = expert_map.cpu().tolist()
    num_routes = len(routes)
    sorted_ids = []
    expert_ids = []
    for route_idx, global_expert in enumerate(routes):
        if not 0 <= global_expert < len(mapping):
            continue
        local_expert = mapping[global_expert]
        if not 0 <= local_expert < local_num_experts:
            continue
        sorted_ids.extend([route_idx] + [num_routes] * (block_size - 1))
        expert_ids.append(local_expert)
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device),
    )


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or not torch.cuda.is_available(),
    reason="EP route-block alignment requires NVIDIA CUDA",
)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("map_dtype", [torch.int32, torch.int64])
def test_moe_align_block_size_ep_route_block_safe_mapping(id_dtype, map_dtype):
    """Remote, invalid, and repeated routes retain deterministic semantics."""
    device = flag_gems.device
    global_experts, local_experts, block_size = 288, 18, 16
    shard_begin = 7 * local_experts
    expert_map = torch.full((global_experts,), -1, dtype=map_dtype, device=device)
    expert_map[shard_begin : shard_begin + local_experts] = torch.arange(
        local_experts, dtype=map_dtype, device=device
    )
    if map_dtype == torch.int64:
        # In-range global IDs may still map to invalid large local IDs.
        expert_map[5] = 2**40
        expert_map[6] = -(2**40)
    else:
        expert_map[5] = local_experts
        expert_map[6] = -2

    if id_dtype == torch.int64:
        invalid_low, invalid_high = -(2**40), 2**40
    else:
        invalid_low, invalid_high = -1, global_experts
    topk_ids = torch.tensor(
        [
            [
                shard_begin + 3,
                0,
                shard_begin + 3,
                shard_begin + 7,
                invalid_low,
                invalid_high,
                5,
                6,
            ]
        ],
        dtype=id_dtype,
        device=device,
    )

    actual = moe_align_block_size_ep_route_block(
        topk_ids, expert_map, block_size, local_experts
    )
    expected_sorted, expected_experts = _route_block_reference(
        topk_ids, expert_map, local_experts, block_size
    )
    _synchronize()

    num_local_routes = expected_experts.numel()
    total_tokens = num_local_routes * block_size
    assert actual[0].numel() == topk_ids.numel() * block_size
    assert actual[1].numel() == topk_ids.numel()
    torch.testing.assert_close(actual[0][:total_tokens], expected_sorted)
    torch.testing.assert_close(actual[1][:num_local_routes], expected_experts)
    torch.testing.assert_close(
        actual[2],
        torch.tensor([total_tokens], dtype=torch.int32, device=device),
    )


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or not torch.cuda.is_available(),
    reason="CUDA Graph requires NVIDIA CUDA",
)
def test_moe_align_block_size_ep_route_block_cuda_graph_dynamic_routes():
    device = flag_gems.device
    global_experts, local_experts, block_size = 288, 18, 16
    shard_begin = 7 * local_experts
    expert_map = torch.full((global_experts,), -1, dtype=torch.int64, device=device)
    expert_map[shard_begin : shard_begin + local_experts] = torch.arange(
        local_experts, dtype=torch.int64, device=device
    )
    static_topk_ids = torch.zeros((1, 8), dtype=torch.int64, device=device)

    # Compile before capture, then verify that replay reads route values from
    # device memory rather than specializing on the capture-time contents.
    moe_align_block_size_ep_route_block(
        static_topk_ids, expert_map, block_size, local_experts
    )
    _synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = moe_align_block_size_ep_route_block(
            static_topk_ids, expert_map, block_size, local_experts
        )

    route_updates = [
        # No local routes.
        [0, 1, 2, 3, 4, 5, 6, 7],
        # Remote and repeated local experts.
        [shard_begin + 2, 0, shard_begin + 2, 287, 12, 13, 14, 15],
        # Large invalid IDs mixed with valid local routes.
        [2**40, shard_begin + 17, -(2**40), shard_begin, 288, -1, 0, 1],
        # Every route is local, including repeated experts.
        [
            shard_begin,
            shard_begin,
            shard_begin + 1,
            shard_begin + 1,
            shard_begin + 2,
            shard_begin + 3,
            shard_begin + 4,
            shard_begin + 5,
        ],
    ]
    for routes in route_updates:
        static_topk_ids.copy_(torch.tensor([routes], dtype=torch.int64, device=device))
        graph.replay()
        _synchronize()
        expected_sorted, expected_experts = _route_block_reference(
            static_topk_ids, expert_map, local_experts, block_size
        )
        num_local_routes = expected_experts.numel()
        total_tokens = num_local_routes * block_size
        torch.testing.assert_close(graph_output[0][:total_tokens], expected_sorted)
        torch.testing.assert_close(graph_output[1][:num_local_routes], expected_experts)
        torch.testing.assert_close(
            graph_output[2],
            torch.tensor([total_tokens], dtype=torch.int32, device=device),
        )
