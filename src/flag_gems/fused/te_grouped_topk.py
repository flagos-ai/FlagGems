# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0
#
# TE-compatible grouped topk forward for training.
# Combines the high-performance grouped_topk selection (two-kernel approach)
# with score_function computation and TE-format output generation.
#
# This avoids fusing grouped_topk into a single kernel (which performs poorly
# in Triton due to nested loops over sub-arrays) and instead takes a two-step
# approach:
#   Step 1: Use existing grouped_topk kernel to get topk_indices [T, K]
#   Step 2: A scatter kernel computes probs, routing_map, intermediate_output
#           from logits + topk_indices + score_function parameters

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

EPSILON = 1.0e-20

_SCORE_FUNCTIONS = {
    "sigmoid": 0,
    "softmax": 1,
    "sqrtsoftplus": 2,
}


# =============================================================================
# Step 2 kernels: given topk_indices, compute probs/routing_map/intermediate
# =============================================================================


@libentry()
@triton.jit
def _te_grouped_topk_scatter_sigmoid_kernel(
    logits_ptr,
    topk_indices_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scatter kernel for sigmoid: compute outputs from logits + topk_indices."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < TOPK

    # Load logits and compute sigmoid for all experts
    x = tl.load(logits_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    act = 1.0 / (1.0 + tl.exp(-x))

    # Save sigmoid output for backward
    tl.store(intermediate_ptr + row_base + expert_offsets, act, mask=mask)

    # Load topk indices for this token
    indices = tl.load(
        topk_indices_ptr + pid * TOPK + k_offsets, mask=k_mask, other=0
    ).to(tl.int32)

    # Build routing_map: 1 at selected positions
    # For each expert, check if it's in the topk_indices
    # Use broadcast: expert_offsets[BLOCK_E] vs indices[BLOCK_K]
    is_selected = (
        tl.sum((expert_offsets[:, None] == indices[None, :]).to(tl.int32), axis=1) > 0
    )

    # Gather activations at selected positions for normalization
    selected_act = tl.where(is_selected & mask, act, 0.0)

    # Normalize (topk > 1)
    if TOPK > 1:
        sum_selected = tl.sum(selected_act, axis=0) + EPSILON
        normalized = selected_act / sum_selected
    else:
        normalized = selected_act

    # Write outputs
    probs_out = tl.where(is_selected & mask, normalized * scaling_factor, 0.0)
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, is_selected, mask=mask)


@libentry()
@triton.jit
def _te_grouped_topk_scatter_softmax_pre_kernel(
    logits_ptr,
    topk_indices_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scatter kernel for pre-softmax: softmax(all) then select topk."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < TOPK

    x = tl.load(
        logits_ptr + row_base + expert_offsets, mask=mask, other=-float("inf")
    ).to(tl.float32)

    # Softmax over all experts
    row_max = tl.max(tl.where(mask, x, -float("inf")), axis=0)
    exp_vals = tl.exp(x - row_max)
    exp_vals = tl.where(mask, exp_vals, 0.0)
    sum_exp = tl.sum(exp_vals, axis=0) + EPSILON
    softmax_out = exp_vals / sum_exp

    # Save softmax output for backward
    tl.store(intermediate_ptr + row_base + expert_offsets, softmax_out, mask=mask)

    # Load topk indices
    indices = tl.load(
        topk_indices_ptr + pid * TOPK + k_offsets, mask=k_mask, other=0
    ).to(tl.int32)

    is_selected = (
        tl.sum((expert_offsets[:, None] == indices[None, :]).to(tl.int32), axis=1) > 0
    )

    # probs = scaling * softmax at selected, 0 elsewhere
    probs_out = tl.where(is_selected & mask, softmax_out * scaling_factor, 0.0)
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, is_selected, mask=mask)


@libentry()
@triton.jit
def _te_grouped_topk_scatter_softmax_post_kernel(
    logits_ptr,
    topk_indices_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scatter kernel for post-softmax: softmax only over selected experts."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < TOPK

    x = tl.load(
        logits_ptr + row_base + expert_offsets, mask=mask, other=-float("inf")
    ).to(tl.float32)

    # Load topk indices
    indices = tl.load(
        topk_indices_ptr + pid * TOPK + k_offsets, mask=k_mask, other=0
    ).to(tl.int32)

    is_selected = (
        tl.sum((expert_offsets[:, None] == indices[None, :]).to(tl.int32), axis=1) > 0
    )

    # Softmax only over selected positions
    selected_logits = tl.where(is_selected & mask, x, -float("inf"))
    row_max = tl.max(selected_logits, axis=0)
    exp_vals = tl.exp(selected_logits - row_max)
    exp_vals = tl.where(is_selected & mask, exp_vals, 0.0)
    sum_exp = tl.sum(exp_vals, axis=0) + EPSILON
    softmax_out = exp_vals / sum_exp

    # intermediate: softmax at selected, -inf elsewhere
    intermediate_out = tl.where(is_selected & mask, softmax_out, -float("inf"))
    tl.store(intermediate_ptr + row_base + expert_offsets, intermediate_out, mask=mask)

    probs_out = tl.where(is_selected & mask, softmax_out * scaling_factor, 0.0)
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, is_selected, mask=mask)


@libentry()
@triton.jit
def _te_grouped_topk_scatter_sqrtsoftplus_kernel(
    logits_ptr,
    topk_indices_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scatter kernel for sqrtsoftplus: compute outputs from logits + topk_indices."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < TOPK

    x = tl.load(logits_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    # Save original logits for backward
    tl.store(intermediate_ptr + row_base + expert_offsets, x, mask=mask)

    # Sqrtsoftplus
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    act = tl.sqrt(sp)

    # Load topk indices
    indices = tl.load(
        topk_indices_ptr + pid * TOPK + k_offsets, mask=k_mask, other=0
    ).to(tl.int32)

    is_selected = (
        tl.sum((expert_offsets[:, None] == indices[None, :]).to(tl.int32), axis=1) > 0
    )

    # Gather activations at selected positions for normalization
    selected_act = tl.where(is_selected & mask, act, 0.0)

    # Normalize (topk > 1)
    if TOPK > 1:
        sum_selected = tl.sum(selected_act, axis=0) + EPSILON
        normalized = selected_act / sum_selected
    else:
        normalized = selected_act

    probs_out = tl.where(is_selected & mask, normalized * scaling_factor, 0.0)
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, is_selected, mask=mask)


# =============================================================================
# Python entry point
# =============================================================================


def te_grouped_topk(
    logits: torch.Tensor,
    topk: int,
    num_groups: int,
    group_topk: int,
    use_pre_softmax: bool = True,
    scaling_factor: float = 1.0,
    score_function: int | str = 1,
    expert_bias: torch.Tensor = None,
) -> tuple:
    """TE-compatible grouped topk forward for training.

    Two-step approach for performance:
      1. Use FlagGems grouped_topk kernel to get topk_indices
      2. Scatter kernel computes probs/routing_map/intermediate from topk_indices

    Args:
        logits: [num_tokens, num_experts] input logits.
        topk: total number of experts selected per token.
        num_groups: number of expert groups.
        group_topk: number of groups to select.
        use_pre_softmax: (softmax only) apply softmax before topk.
        scaling_factor: output scaling factor.
        score_function: 0=sigmoid, 1=softmax, 2=sqrtsoftplus.
        expert_bias: [num_experts] bias tensor (required for grouped_topk selection).

    Returns:
        probs: [num_tokens, num_experts] sparse output (same dtype as logits).
        routing_map: [num_tokens, num_experts] bool, True at selected positions.
        intermediate_output: [num_tokens, num_experts] float32, for backward.
    """
    from flag_gems.fused.grouped_topk import grouped_topk

    logger.debug("GEMS TE_GROUPED_TOPK")
    if isinstance(score_function, str):
        score_function = _SCORE_FUNCTIONS[score_function]
    assert score_function in (0, 1, 2)
    assert num_groups > 0 and group_topk > 0
    assert logits.shape[1] % num_groups == 0
    assert topk % group_topk == 0

    num_tokens, num_experts = logits.shape

    # Step 1: Use existing grouped_topk to get topk_indices
    # grouped_topk expects scores after score_function applied (for selection with bias)
    # and scoring_func=1 means apply sigmoid internally, scoring_func=0 means no transform.
    # For TE compatibility, we handle score_function ourselves:
    if score_function == 0:  # sigmoid
        scores_for_selection = torch.sigmoid(logits.float()).to(logits.dtype)
        scoring_func_flag = 0  # no additional transform in grouped_topk
    elif score_function == 2:  # sqrtsoftplus
        x_f = logits.float()
        sp = torch.where(x_f > 20.0, x_f, torch.log1p(torch.exp(x_f)))
        scores_for_selection = torch.sqrt(sp).to(logits.dtype)
        scoring_func_flag = 0
    else:  # softmax: grouped_topk selection on raw logits or softmax output
        if use_pre_softmax:
            scores_for_selection = torch.softmax(logits.float(), dim=-1).to(
                logits.dtype
            )
        else:
            scores_for_selection = logits
        scoring_func_flag = 0

    if expert_bias is None:
        expert_bias = torch.zeros(num_experts, dtype=logits.dtype, device=logits.device)

    # grouped_topk returns (topk_values [T,K], topk_indices [T,K])
    # renormalize=False because we do normalization ourselves in the scatter kernel
    _, topk_indices = grouped_topk(
        scores=scores_for_selection,
        n_group=num_groups,
        topk_group=group_topk,
        topk=topk,
        renormalize=False,
        routed_scaling_factor=1.0,
        bias=expert_bias,
        scoring_func=scoring_func_flag,
    )

    # Step 2: Scatter kernel to produce TE-format outputs
    probs = torch.zeros(
        num_tokens, num_experts, dtype=logits.dtype, device=logits.device
    )
    routing_map = torch.zeros(
        num_tokens, num_experts, dtype=torch.bool, device=logits.device
    )
    intermediate = torch.empty(
        num_tokens, num_experts, dtype=torch.float32, device=logits.device
    )

    BLOCK_E = triton.next_power_of_2(num_experts)
    BLOCK_K = triton.next_power_of_2(topk)
    grid = (num_tokens,)

    with torch_device_fn.device(logits.device):
        if score_function == 0:
            _te_grouped_topk_scatter_sigmoid_kernel[grid](
                logits,
                topk_indices,
                probs,
                routing_map,
                intermediate,
                num_tokens,
                num_experts,
                scaling_factor,
                TOPK=topk,
                BLOCK_E=BLOCK_E,
                BLOCK_K=BLOCK_K,
            )
        elif score_function == 1:
            if use_pre_softmax:
                _te_grouped_topk_scatter_softmax_pre_kernel[grid](
                    logits,
                    topk_indices,
                    probs,
                    routing_map,
                    intermediate,
                    num_tokens,
                    num_experts,
                    scaling_factor,
                    TOPK=topk,
                    BLOCK_E=BLOCK_E,
                    BLOCK_K=BLOCK_K,
                )
            else:
                _te_grouped_topk_scatter_softmax_post_kernel[grid](
                    logits,
                    topk_indices,
                    probs,
                    routing_map,
                    intermediate,
                    num_tokens,
                    num_experts,
                    scaling_factor,
                    TOPK=topk,
                    BLOCK_E=BLOCK_E,
                    BLOCK_K=BLOCK_K,
                )
        else:
            _te_grouped_topk_scatter_sqrtsoftplus_kernel[grid](
                logits,
                topk_indices,
                probs,
                routing_map,
                intermediate,
                num_tokens,
                num_experts,
                scaling_factor,
                TOPK=topk,
                BLOCK_E=BLOCK_E,
                BLOCK_K=BLOCK_K,
            )

    return probs, routing_map, intermediate
