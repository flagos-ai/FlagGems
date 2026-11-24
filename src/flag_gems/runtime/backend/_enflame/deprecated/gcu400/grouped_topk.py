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

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def topk_with_k2_triton(
    scores_ptr,
    bias_ptr,
    group_scores_ptr,
    num_experts_per_group,
    n_group,
    stride_scores_token,
    stride_group_scores_token,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    token_id = pid // n_group
    group_id = pid % n_group

    lane = tl.arange(0, BLOCK_SIZE)
    mask = lane < num_experts_per_group

    scores_offset = token_id * stride_scores_token + group_id * num_experts_per_group
    bias_offset = group_id * num_experts_per_group

    x = tl.load(
        scores_ptr + scores_offset + lane,
        mask=mask,
        other=-float("inf"),
    )

    b = tl.load(
        bias_ptr + bias_offset + lane,
        mask=mask,
        other=0.0,
    )

    x = x + b

    x_f32 = x.to(tl.float32)

    max1 = tl.max(x_f32, axis=0)
    is_max1 = (x_f32 == max1) & mask
    count_max1 = tl.sum(is_max1.to(tl.int32), axis=0)

    x2 = tl.where(
        is_max1 & (count_max1 == 1),
        -float("inf"),
        x_f32,
    )
    max2 = tl.max(x2, axis=0)

    group_scores_offset = token_id * stride_group_scores_token + group_id
    tl.store(
        group_scores_ptr + group_scores_offset,
        (max1 + max2).to(INPUT_DTYPE),
    )


@triton.jit
def group_idx_and_topk_triton(
    scores_ptr,
    group_scores_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    bias_ptr,
    num_tokens,
    n_group,
    topk_group,
    topk,
    num_experts,
    num_experts_per_group,
    routed_scaling_factor,
    stride_scores_token,
    stride_group_scores_token,
    stride_out_token,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    renormalize: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    neg_inf = -float("inf")

    group_offsets = tl.arange(0, BLOCK_GROUP)
    valid_group = group_offsets < n_group

    group_scores = tl.load(
        group_scores_ptr + pid * stride_group_scores_token + group_offsets,
        mask=valid_group,
        other=neg_inf,
    )

    group_scores_f32 = group_scores.to(tl.float32)
    is_finite = (group_scores_f32 == group_scores_f32) & (
        group_scores_f32 != float("inf")
    )
    group_scores_f32 = tl.where(is_finite & valid_group, group_scores_f32, neg_inf)

    max_group_score = tl.max(group_scores_f32, axis=0)
    if_proceed = max_group_score != neg_inf

    value = group_scores_f32
    target_num_min = BLOCK_GROUP - n_group + topk_group
    count_equal_to_top_value = BLOCK_GROUP - n_group
    pre_count_equal_to_top_value = 0
    topk_group_value = neg_inf

    for _ in range(TOPK_GROUP):
        need = count_equal_to_top_value < target_num_min
        max_val = tl.max(value, axis=0)

        is_max = need & (value == max_val)
        value = tl.where(is_max, neg_inf, value)

        newly = tl.sum(is_max.to(tl.int32), axis=0)

        pre_count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value, pre_count_equal_to_top_value
        )
        count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value + newly, count_equal_to_top_value
        )
        topk_group_value = tl.where(need, max_val, topk_group_value)

    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value

    group_gt = group_scores_f32 > topk_group_value
    group_eq = group_scores_f32 == topk_group_value

    eq_i = group_eq.to(tl.int32)
    prefix_eq = tl.cumsum(eq_i, axis=0) - eq_i

    group_selected = (
        group_gt | (group_eq & (prefix_eq < num_equalto_topkth_group))
    ) & valid_group

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    valid_expert = expert_offsets < num_experts
    expert_group = expert_offsets // num_experts_per_group

    expert_in_group = expert_group[:, None] == group_offsets[None, :]
    expert_selected = (
        tl.sum((expert_in_group & group_selected[None, :]).to(tl.int32), axis=1) > 0
    ) & valid_expert

    scored = tl.load(
        scores_ptr + pid * stride_scores_token + expert_offsets,
        mask=expert_selected,
        other=neg_inf,
    )

    expert_bias = tl.load(
        bias_ptr + expert_offsets,
        mask=valid_expert,
        other=0.0,
    )

    selection_scores_native = scored + expert_bias

    selection_scores = tl.where(
        expert_selected,
        selection_scores_native.to(tl.float32),
        neg_inf,
    )

    topk_vals = tl.full([TOPK], 0.0, tl.float32)
    topk_idx = tl.full([TOPK], 0, tl.int32)
    pos_range = tl.arange(0, TOPK)

    for i in range(TOPK):
        max_val = tl.max(selection_scores, axis=0)
        is_max = selection_scores == max_val

        candidate_idx = tl.where(is_max, expert_offsets, num_experts + 1)
        selected_idx = tl.min(candidate_idx, axis=0)

        selected_score = tl.load(
            scores_ptr + pid * stride_scores_token + selected_idx,
            mask=selected_idx < num_experts,
            other=neg_inf,
        ).to(tl.float32)

        topk_vals = tl.where(pos_range == i, selected_score, topk_vals)
        topk_idx = tl.where(pos_range == i, selected_idx.to(tl.int32), topk_idx)

        selection_scores = tl.where(
            expert_offsets == selected_idx, neg_inf, selection_scores
        )

    if renormalize == 1:
        topk_sum = tl.sum(topk_vals, axis=0) + 1e-20
        scale = routed_scaling_factor / topk_sum
    else:
        scale = routed_scaling_factor

    topk_vals = topk_vals * scale

    default_idx = pos_range.to(tl.int32)
    default_vals = tl.full([TOPK], 1.0 / topk, tl.float32)

    final_vals = tl.where(if_proceed, topk_vals, default_vals)
    final_idx = tl.where(if_proceed, topk_idx, default_idx)

    tl.store(
        topk_values_ptr + pid * stride_out_token + pos_range,
        final_vals,
        mask=pos_range < topk,
    )

    tl.store(
        topk_indices_ptr + pid * stride_out_token + pos_range,
        final_idx,
        mask=pos_range < topk,
    )


# Hard limit on GCU grid.x; when the token count exceeds it, fold into grid.y paging
GCU_GRID_X_MAX = 65535


@triton.jit
def grouped_topk_fused_kernel(
    scores_ptr,
    bias_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    num_tokens,
    n_group,
    topk_group,
    topk,
    num_experts,
    num_experts_per_group,
    routed_scaling_factor,
    stride_scores_token,
    stride_out_token,
    grid_x,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_PER_GROUP: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    renormalize: tl.constexpr,
):
    """单 program 完成一个 token 的「组打分 → 选组 → 选专家 → 归一化」。

    相比 group_scores 与 group_idx 两个 kernel 的版本：省掉一次 launch、group_scores
    的全局读写，以及 scoring_func=1 时的全量 scores 中间张量；grid 也从
    num_tokens * n_group 收缩到 num_tokens。
    """
    pid = tl.program_id(1) * grid_x + tl.program_id(0)
    if pid >= num_tokens:
        return

    neg_inf = -float("inf")

    group_offsets = tl.arange(0, BLOCK_GROUP)
    per_group_offsets = tl.arange(0, BLOCK_PER_GROUP)
    # A 2D tile over (group, expert-in-group) covers all experts of this token
    expert_tile = (
        group_offsets[:, None] * num_experts_per_group + per_group_offsets[None, :]
    )
    valid_group = group_offsets < n_group
    tile_mask = valid_group[:, None] & (
        per_group_offsets[None, :] < num_experts_per_group
    )

    scores_raw = tl.load(
        scores_ptr + pid * stride_scores_token + expert_tile,
        mask=tile_mask,
        other=neg_inf,
    )
    if SCORING_FUNC == 1:
        # Match torch.sigmoid(scores.float()).to(scores.dtype)
        scores_raw = tl.sigmoid(scores_raw.to(tl.float32)).to(INPUT_DTYPE)

    bias_raw = tl.load(bias_ptr + expert_tile, mask=tile_mask, other=0.0)
    # Add in f32 first and then round back to the input dtype: semantically equivalent to low-precision addition (IEEE
    # round-to-nearest),
    # while also avoiding the precision deviation seen with 2D bf16 tile addition on the GCU backend
    pair_dtype = (scores_raw.to(tl.float32) + bias_raw.to(tl.float32)).to(INPUT_DTYPE)
    selection_tile = tl.where(tile_mask, pair_dtype.to(tl.float32), neg_inf)

    # Per-group group score = sum of the top-2 of (scores + bias) within the group; for a tie at the max, take only one
    max1 = tl.max(selection_tile, axis=1)
    is_max1 = (selection_tile == max1[:, None]) & tile_mask
    count_max1 = tl.sum(is_max1.to(tl.int32), axis=1)
    remainder = tl.where(is_max1 & (count_max1[:, None] == 1), neg_inf, selection_tile)
    max2 = tl.max(remainder, axis=1)
    # Quantize group_scores to the input dtype (the original implementation writes in scores dtype); here quantize
    # first, then convert to f32
    group_scores = (max1 + max2).to(INPUT_DTYPE).to(tl.float32)
    group_scores = tl.where(valid_group, group_scores, neg_inf)

    max_group_score = tl.max(group_scores, axis=0)
    if_proceed = max_group_score != neg_inf

    # Select topk_group groups: by value descending, ties broken by group index ascending
    value = group_scores
    target_num_min = BLOCK_GROUP - n_group + topk_group
    count_equal_to_top_value = BLOCK_GROUP - n_group
    pre_count_equal_to_top_value = 0
    topk_group_value = neg_inf

    for _ in range(TOPK_GROUP):
        need = count_equal_to_top_value < target_num_min
        max_val = tl.max(value, axis=0)
        is_max = need & (value == max_val)
        value = tl.where(is_max, neg_inf, value)
        newly = tl.sum(is_max.to(tl.int32), axis=0)
        pre_count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value, pre_count_equal_to_top_value
        )
        count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value + newly, count_equal_to_top_value
        )
        topk_group_value = tl.where(need, max_val, topk_group_value)

    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value
    group_gt = group_scores > topk_group_value
    group_eq = group_scores == topk_group_value
    eq_i = group_eq.to(tl.int32)
    prefix_eq = tl.cumsum(eq_i, axis=0) - eq_i
    group_selected = (
        group_gt | (group_eq & (prefix_eq < num_equalto_topkth_group))
    ) & valid_group

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    valid_expert = expert_offsets < num_experts
    expert_group = expert_offsets // num_experts_per_group
    expert_in_group = expert_group[:, None] == group_offsets[None, :]
    expert_selected = (
        tl.sum((expert_in_group & group_selected[None, :]).to(tl.int32), axis=1) > 0
    ) & valid_expert

    scored_raw = tl.load(
        scores_ptr + pid * stride_scores_token + expert_offsets,
        mask=valid_expert,
        other=neg_inf,
    )
    if SCORING_FUNC == 1:
        scored_raw = tl.sigmoid(scored_raw.to(tl.float32)).to(INPUT_DTYPE)
    scored = scored_raw.to(tl.float32)

    expert_bias = tl.load(bias_ptr + expert_offsets, mask=valid_expert, other=0.0)
    expert_pair = (scored_raw.to(tl.float32) + expert_bias.to(tl.float32)).to(
        INPUT_DTYPE
    )
    selection_scores = tl.where(expert_selected, expert_pair.to(tl.float32), neg_inf)

    topk_vals = tl.full([TOPK], 0.0, tl.float32)
    topk_idx = tl.full([TOPK], 0, tl.int32)
    pos_range = tl.arange(0, TOPK)

    for i in range(TOPK):
        max_val = tl.max(selection_scores, axis=0)
        is_max = selection_scores == max_val
        candidate_idx = tl.where(is_max, expert_offsets, num_experts + 1)
        selected_idx = tl.min(candidate_idx, axis=0)
        # Take the raw score of the selected expert (without bias); with no candidate, get -inf as in the original
        # implementation
        selected_score = tl.sum(
            tl.where(expert_offsets == selected_idx, scored, 0.0), axis=0
        )
        selected_score = tl.where(selected_idx < num_experts, selected_score, neg_inf)
        topk_vals = tl.where(pos_range == i, selected_score, topk_vals)
        topk_idx = tl.where(pos_range == i, selected_idx.to(tl.int32), topk_idx)
        selection_scores = tl.where(
            expert_offsets == selected_idx, neg_inf, selection_scores
        )

    if renormalize == 1:
        topk_sum = tl.sum(topk_vals, axis=0) + 1e-20
        scale = routed_scaling_factor / topk_sum
    else:
        scale = routed_scaling_factor
    topk_vals = topk_vals * scale

    default_idx = pos_range.to(tl.int32)
    default_vals = tl.full([TOPK], 1.0 / topk, tl.float32)
    final_vals = tl.where(if_proceed, topk_vals, default_vals)
    final_idx = tl.where(if_proceed, topk_idx, default_idx)

    tl.store(
        topk_values_ptr + pid * stride_out_token + pos_range,
        final_vals,
        mask=pos_range < topk,
    )
    tl.store(
        topk_indices_ptr + pid * stride_out_token + pos_range,
        final_idx,
        mask=pos_range < topk,
    )


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    logger.debug("GEMS_ENFLAME GROUPED_TOPK")
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D Tensor")
    num_tokens, num_experts = scores.shape
    if num_experts % n_group != 0:
        raise ValueError("num_experts must be divisible by n_group")
    if n_group > 32:
        raise ValueError("n_group should be smaller than or equal to 32")
    if topk > 32:
        raise ValueError("topk should be smaller than or equal to 32 for now")
    if scoring_func not in (0, 1):
        raise ValueError("scoring_func must be 0 (none) or 1 (sigmoid)")

    if bias.dtype != scores.dtype:
        bias = bias.to(scores.dtype)
    if bias.ndim != 1:
        bias = bias.flatten()
    if len(bias) != num_experts:
        raise ValueError(
            f"bias length ({len(bias)}) must match num_experts ({num_experts})"
        )

    num_experts_per_group = num_experts // n_group

    if scores.dtype == torch.float32:
        INPUT_DTYPE = tl.float32
    elif scores.dtype == torch.float16:
        INPUT_DTYPE = tl.float16
    elif scores.dtype == torch.bfloat16:
        INPUT_DTYPE = tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {scores.dtype}")

    # Fuse sigmoid into the kernel, no longer materializing the full scores_processed intermediate tensor
    topk_values = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.float32,
    )

    topk_indices = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.int32,
    )

    grid_x = min(num_tokens, GCU_GRID_X_MAX)
    grid = (grid_x, triton.cdiv(num_tokens, grid_x))

    grouped_topk_fused_kernel[grid](
        scores,
        bias,
        topk_values,
        topk_indices,
        num_tokens,
        n_group,
        topk_group,
        topk,
        num_experts,
        num_experts_per_group,
        routed_scaling_factor,
        scores.stride(0),
        topk_values.stride(0),
        grid_x,
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        TOPK=topk,
        BLOCK_GROUP=triton.next_power_of_2(n_group),
        BLOCK_PER_GROUP=triton.next_power_of_2(num_experts_per_group),
        BLOCK_EXPERT=triton.next_power_of_2(num_experts),
        INPUT_DTYPE=INPUT_DTYPE,
        SCORING_FUNC=scoring_func,
        renormalize=int(renormalize),
    )

    return topk_values, topk_indices
