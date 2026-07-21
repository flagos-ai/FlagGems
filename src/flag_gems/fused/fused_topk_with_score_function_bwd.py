# Copyright (c) 2024, FlagGems. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Backward pass for fused top-k with score function.
# Reference: NVIDIA TransformerEngine fused_router/fused_topk_with_score_function.cu

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

EPSILON = 1e-20


@libentry()
@triton.jit
def _fused_topk_score_fn_bwd_sigmoid_kernel(
    routing_map_ptr,
    intermediate_ptr,
    grad_probs_ptr,
    grad_logits_ptr,
    num_tokens,
    num_experts,
    topk,
    scaling_factor,
    BLOCK_E: tl.constexpr,
):
    """Backward kernel for sigmoid score function.

    intermediate stores sigmoid outputs: act_i = sigmoid(x_i).
    Forward normalization (topk > 1): prob_i = act_i / sum_selected(act).
    """
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts

    grad = (
        tl.load(grad_probs_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        * scaling_factor
    )

    act = tl.load(
        intermediate_ptr + row_base + expert_offsets, mask=mask, other=0.0
    ).to(tl.float32)

    routed = (
        tl.load(routing_map_ptr + row_base + expert_offsets, mask=mask, other=0) > 0
    )

    # Normalization backward (topk > 1): quotient rule
    g = grad
    if topk > 1:
        sum_act = tl.sum(tl.where(routed, act, 0.0), axis=0) + EPSILON
        sum_grad_act = tl.sum(tl.where(routed, grad * act, 0.0), axis=0)
        g = tl.where(routed, g / sum_act - sum_grad_act / (sum_act * sum_act), 0.0)
    else:
        g = tl.where(routed, g, 0.0)

    # Sigmoid backward: dy/dx = y * (1 - y)
    g = g * act * (1.0 - act)

    tl.store(
        grad_logits_ptr + row_base + expert_offsets,
        g.to(grad_logits_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _fused_topk_score_fn_bwd_softmax_kernel(
    routing_map_ptr,
    intermediate_ptr,
    grad_probs_ptr,
    grad_logits_ptr,
    num_tokens,
    num_experts,
    topk,
    scaling_factor,
    use_pre_softmax: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Backward kernel for softmax score function.

    intermediate stores softmax outputs.
    pre_softmax: softmax applied to all experts before topk.
    post_softmax: softmax applied only to selected experts after topk.
    """
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts

    grad = (
        tl.load(grad_probs_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        * scaling_factor
    )

    act = tl.load(
        intermediate_ptr + row_base + expert_offsets, mask=mask, other=0.0
    ).to(tl.float32)

    routed = (
        tl.load(routing_map_ptr + row_base + expert_offsets, mask=mask, other=0) > 0
    )

    if use_pre_softmax:
        # Pre-softmax: softmax was applied to all experts, then topk masked
        # sum_output_x_grad = sum(grad_routed * act_all)
        sum_output_x_grad = tl.sum(tl.where(routed, grad, 0.0) * act, axis=0)
        # Mask unselected, then apply softmax backward to all
        g = tl.where(routed, grad, 0.0)
        g = act * (g - sum_output_x_grad)
    else:
        # Post-softmax: topk selected first, then softmax on selected subset
        # sum_output_x_grad = sum(grad * act) over routed only
        sum_output_x_grad = tl.sum(tl.where(routed, grad * act, 0.0), axis=0)
        g = tl.where(routed, act * (grad - sum_output_x_grad), 0.0)

    tl.store(
        grad_logits_ptr + row_base + expert_offsets,
        g.to(grad_logits_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _fused_topk_score_fn_bwd_sqrtsoftplus_kernel(
    routing_map_ptr,
    intermediate_ptr,
    grad_probs_ptr,
    grad_logits_ptr,
    num_tokens,
    num_experts,
    topk,
    scaling_factor,
    BLOCK_E: tl.constexpr,
):
    """Backward kernel for sqrtsoftplus score function.

    intermediate stores original logits x.
    Forward: act = sqrt(softplus(x)) = sqrt(log(1 + exp(x))).
    Backward: dy/dx = sigmoid(x) / (2 * y).
    """
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < num_experts
    row_base = pid * num_experts

    grad = (
        tl.load(grad_probs_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        * scaling_factor
    )

    # intermediate stores original logits
    x = tl.load(intermediate_ptr + row_base + expert_offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    routed = (
        tl.load(routing_map_ptr + row_base + expert_offsets, mask=mask, other=0) > 0
    )

    # Recompute sqrtsoftplus
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    act_val = tl.sqrt(sp)

    # Normalization backward (topk > 1)
    g = grad
    if topk > 1:
        sum_act = tl.sum(tl.where(routed, act_val, 0.0), axis=0) + EPSILON
        sum_grad_act = tl.sum(tl.where(routed, grad * act_val, 0.0), axis=0)
        g = tl.where(routed, g / sum_act - sum_grad_act / (sum_act * sum_act), 0.0)
    else:
        g = tl.where(routed, g, 0.0)

    # Sqrtsoftplus backward: dy/dx = sigmoid(x) / (2 * y)
    sig = 1.0 / (1.0 + tl.exp(-x))
    dy_dx = sig / (2.0 * act_val + EPSILON)
    g = g * dy_dx

    tl.store(
        grad_logits_ptr + row_base + expert_offsets,
        g.to(grad_logits_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_topk_with_score_function_bwd(
    routing_map: torch.Tensor,
    intermediate: torch.Tensor,
    grad_probs: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = True,
    scaling_factor: float = 1.0,
    score_function: int = 1,
) -> torch.Tensor:
    """Backward pass for fused top-k with score function.

    Computes grad_logits from grad_probs using the routing_map and
    intermediate values saved during the forward pass.

    Args:
        routing_map: [num_tokens, num_experts] uint8/bool tensor indicating
            which experts were selected per token.
        intermediate: [num_tokens, num_experts] float32 tensor saved from forward.
            Content depends on score_function:
            - sigmoid (0): sigmoid outputs
            - softmax (1): softmax outputs
            - sqrtsoftplus (2): original logits
        grad_probs: [num_tokens, num_experts] gradient w.r.t. output probabilities.
        topk: number of experts selected per token.
        use_pre_softmax: (softmax only) whether softmax was applied before topk.
        scaling_factor: scaling factor applied in forward pass.
        score_function: 0=sigmoid, 1=softmax, 2=sqrtsoftplus.

    Returns:
        grad_logits: [num_tokens, num_experts] gradient w.r.t. input logits.
    """
    logger.debug("GEMS FUSED_TOPK_WITH_SCORE_FUNCTION BWD")
    assert score_function in (
        0,
        1,
        2,
    ), f"score_function must be 0, 1, or 2, got {score_function}"
    num_tokens, num_experts = grad_probs.shape

    grad_logits = torch.empty(
        num_tokens, num_experts, dtype=grad_probs.dtype, device=grad_probs.device
    )

    BLOCK_E = triton.next_power_of_2(num_experts)
    grid = (num_tokens,)

    with torch_device_fn.device(grad_probs.device):
        if score_function == 0:
            _fused_topk_score_fn_bwd_sigmoid_kernel[grid](
                routing_map,
                intermediate,
                grad_probs,
                grad_logits,
                num_tokens,
                num_experts,
                topk,
                scaling_factor,
                BLOCK_E=BLOCK_E,
            )
        elif score_function == 1:
            _fused_topk_score_fn_bwd_softmax_kernel[grid](
                routing_map,
                intermediate,
                grad_probs,
                grad_logits,
                num_tokens,
                num_experts,
                topk,
                scaling_factor,
                use_pre_softmax=use_pre_softmax,
                BLOCK_E=BLOCK_E,
            )
        else:
            _fused_topk_score_fn_bwd_sqrtsoftplus_kernel[grid](
                routing_map,
                intermediate,
                grad_probs,
                grad_logits,
                num_tokens,
                num_experts,
                topk,
                scaling_factor,
                BLOCK_E=BLOCK_E,
            )

    return grad_logits
