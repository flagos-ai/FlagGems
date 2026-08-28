# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0
#
# Forward pass for fused top-k with score function (training path).
# Reference: NVIDIA TransformerEngine fused_router/fused_topk_with_score_function.cu
#
# This kernel produces the outputs needed for backward:
#   probs: [T, E] sparse: only topk positions have values, rest are 0
#   routing_map: [T, E] bool: True at selected expert positions
#   intermediate_output: [T, E] float32: saved activations for backward
#
# Flow per score_function:
#   sigmoid (0):      sigmoid -> [+bias] -> topk -> normalize(topk) -> *scale
#   softmax (1):      pre:  softmax(all) -> topk -> *scale
#                     post: topk -> softmax(topk) -> *scale
#   sqrtsoftplus (2): sqrtsoftplus -> [+bias] -> topk -> normalize(topk) -> *scale
#
# For grouped_topk (num_groups > 0), delegates to te_grouped_topk which uses
# the high-performance two-kernel approach.

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


@libentry()
@triton.jit
def _fused_topk_fwd_sigmoid_kernel(
    logits_ptr,
    expert_bias_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Forward: sigmoid score function."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts

    x = tl.load(logits_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    act = 1.0 / (1.0 + tl.exp(-x))
    tl.store(intermediate_ptr + row_base + expert_offsets, act, mask=mask)

    if HAS_BIAS:
        bias = tl.load(expert_bias_ptr + expert_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scores = act + bias
    else:
        scores = act

    topk_mask = _topk_selection(
        tl.where(mask, scores, -float("inf")),
        expert_offsets,
        TOPK,
        BLOCK_E=BLOCK_E,
    )
    selected_act = tl.where(topk_mask == 1, act, 0.0)

    if TOPK > 1:
        sum_selected = tl.sum(selected_act, axis=0) + EPSILON
        normalized = selected_act / sum_selected
    else:
        normalized = selected_act

    probs_out = normalized * scaling_factor
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, topk_mask == 1, mask=mask)


@libentry()
@triton.jit
def _fused_topk_fwd_softmax_pre_kernel(
    logits_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Forward: softmax pre-softmax mode."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts

    x = tl.load(
        logits_ptr + row_base + expert_offsets, mask=mask, other=-float("inf")
    ).to(tl.float32)
    row_max = tl.max(tl.where(mask, x, -float("inf")), axis=0)
    exp_vals = tl.where(mask, tl.exp(x - row_max), 0.0)
    softmax_out = exp_vals / (tl.sum(exp_vals, axis=0) + EPSILON)

    tl.store(intermediate_ptr + row_base + expert_offsets, softmax_out, mask=mask)

    topk_mask = _topk_selection(
        tl.where(mask, softmax_out, -float("inf")),
        expert_offsets,
        TOPK,
        BLOCK_E=BLOCK_E,
    )

    probs_out = tl.where(topk_mask == 1, softmax_out * scaling_factor, 0.0)
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, topk_mask == 1, mask=mask)


@libentry()
@triton.jit
def _fused_topk_fwd_softmax_post_kernel(
    logits_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Forward: softmax post-softmax mode."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts

    x = tl.load(
        logits_ptr + row_base + expert_offsets, mask=mask, other=-float("inf")
    ).to(tl.float32)

    topk_mask = _topk_selection(
        tl.where(mask, x, -float("inf")),
        expert_offsets,
        TOPK,
        BLOCK_E=BLOCK_E,
    )

    selected_logits = tl.where(topk_mask == 1, x, -float("inf"))
    row_max = tl.max(selected_logits, axis=0)
    exp_vals = tl.where(topk_mask == 1, tl.exp(selected_logits - row_max), 0.0)
    softmax_out = exp_vals / (tl.sum(exp_vals, axis=0) + EPSILON)

    intermediate_out = tl.where(topk_mask == 1, softmax_out, -float("inf"))
    tl.store(intermediate_ptr + row_base + expert_offsets, intermediate_out, mask=mask)

    probs_out = tl.where(topk_mask == 1, softmax_out * scaling_factor, 0.0)
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, topk_mask == 1, mask=mask)


@libentry()
@triton.jit
def _fused_topk_fwd_sqrtsoftplus_kernel(
    logits_ptr,
    expert_bias_ptr,
    probs_ptr,
    routing_map_ptr,
    intermediate_ptr,
    num_tokens,
    num_experts,
    scaling_factor,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Forward: sqrtsoftplus score function."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts

    x = tl.load(logits_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(intermediate_ptr + row_base + expert_offsets, x, mask=mask)

    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    act = tl.sqrt(sp)

    if HAS_BIAS:
        bias = tl.load(expert_bias_ptr + expert_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scores = act + bias
    else:
        scores = act

    topk_mask = _topk_selection(
        tl.where(mask, scores, -float("inf")),
        expert_offsets,
        TOPK,
        BLOCK_E=BLOCK_E,
    )
    selected_act = tl.where(topk_mask == 1, act, 0.0)

    if TOPK > 1:
        sum_selected = tl.sum(selected_act, axis=0) + EPSILON
        normalized = selected_act / sum_selected
    else:
        normalized = selected_act

    probs_out = normalized * scaling_factor
    tl.store(
        probs_ptr + row_base + expert_offsets,
        probs_out.to(probs_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(routing_map_ptr + row_base + expert_offsets, topk_mask == 1, mask=mask)


@triton.jit
def _topk_selection(scores, expert_offsets, TOPK: tl.constexpr, BLOCK_E: tl.constexpr):
    """Select topk positions from scores, return mask [BLOCK_E] of int32 (0/1)."""
    topk_mask = tl.zeros([BLOCK_E], dtype=tl.int32)
    s = scores
    for _k in range(TOPK):
        max_score = tl.max(s, axis=0)
        is_max = s == max_score
        match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=0)
        eidx = best_slot.to(tl.int32)
        topk_mask = tl.where(expert_offsets == eidx, 1, topk_mask)
        s = tl.where(expert_offsets == eidx, -float("inf"), s)
    return topk_mask


def te_fused_topk_with_score_function_fwd(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = True,
    num_groups: int = None,
    group_topk: int = None,
    scaling_factor: float = 1.0,
    score_function: int | str = 1,
    expert_bias: torch.Tensor = None,
    *_,
) -> tuple:
    """Forward pass for fused top-k with score function (training path).

    Produces outputs compatible with te_fused_topk_with_score_function_bwd.

    For grouped_topk (num_groups > 0), delegates to te_grouped_topk which
    uses the high-performance two-kernel approach.

    Args:
        logits: [num_tokens, num_experts] input logits.
        topk: number of experts selected per token.
        use_pre_softmax: (softmax only) apply softmax before topk.
        num_groups: number of groups for grouped topk (-1 = disabled).
        group_topk: number of groups to select (-1 = disabled).
        scaling_factor: output scaling factor.
        score_function: 0=sigmoid, 1=softmax, 2=sqrtsoftplus.
        expert_bias: optional [num_experts] bias (sigmoid/sqrtsoftplus only).

    Returns:
        probs: [num_tokens, num_experts] sparse output (same dtype as logits).
        routing_map: [num_tokens, num_experts] bool, True at selected positions.
        intermediate_output: [num_tokens, num_experts] float32, for backward.
    """
    logger.debug("GEMS TE_FUSED_TOPK_WITH_SCORE_FUNCTION FWD")
    if isinstance(score_function, str):
        score_function = _SCORE_FUNCTIONS[score_function]
    assert score_function in (
        0,
        1,
        2,
    ), f"score_function must be 0, 1, or 2, got {score_function}"
    num_groups = -1 if num_groups is None else int(num_groups)
    group_topk = -1 if group_topk is None else int(group_topk)

    # Delegate grouped_topk to the high-performance two-kernel path
    if group_topk > 0 and num_groups > 0:
        from flag_gems.fused.te_grouped_topk import te_grouped_topk

        return te_grouped_topk(
            logits,
            topk,
            num_groups,
            group_topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
            expert_bias,
        )

    num_tokens, num_experts = logits.shape
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
    grid = (num_tokens,)

    with torch_device_fn.device(logits.device):
        if score_function == 0:
            _fused_topk_fwd_sigmoid_kernel[grid](
                logits,
                expert_bias if expert_bias is not None else logits,
                probs,
                routing_map,
                intermediate,
                num_tokens,
                num_experts,
                scaling_factor,
                HAS_BIAS=expert_bias is not None,
                TOPK=topk,
                BLOCK_E=BLOCK_E,
            )
        elif score_function == 1:
            if use_pre_softmax:
                _fused_topk_fwd_softmax_pre_kernel[grid](
                    logits,
                    probs,
                    routing_map,
                    intermediate,
                    num_tokens,
                    num_experts,
                    scaling_factor,
                    TOPK=topk,
                    BLOCK_E=BLOCK_E,
                )
            else:
                _fused_topk_fwd_softmax_post_kernel[grid](
                    logits,
                    probs,
                    routing_map,
                    intermediate,
                    num_tokens,
                    num_experts,
                    scaling_factor,
                    TOPK=topk,
                    BLOCK_E=BLOCK_E,
                )
        else:
            _fused_topk_fwd_sqrtsoftplus_kernel[grid](
                logits,
                expert_bias if expert_bias is not None else logits,
                probs,
                routing_map,
                intermediate,
                num_tokens,
                num_experts,
                scaling_factor,
                HAS_BIAS=expert_bias is not None,
                TOPK=topk,
                BLOCK_E=BLOCK_E,
            )

    return probs, routing_map, intermediate
