# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0

import pytest
import torch

from flag_gems.fused.te_fused_topk_with_score_function_fwd import (
    te_fused_topk_with_score_function_fwd,
)

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_FWD = getattr(tex, "fused_topk_with_score_function_fwd", None)
except ImportError:
    TE_FWD = None

pytestmark = pytest.mark.te_fused_topk_with_score_function_fwd

SCORE_NAMES = {0: "sigmoid", 1: "softmax", 2: "sqrtsoftplus"}


def _sqrtsoftplus(x):
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    return torch.sqrt(sp)


def _make_routing_map(indices, num_experts):
    routing_map = torch.zeros(
        indices.shape[0],
        num_experts,
        dtype=torch.bool,
        device=indices.device,
    )
    routing_map.scatter_(1, indices, True)
    return routing_map


def _torch_reference_fwd(
    logits,
    topk,
    use_pre_softmax,
    scaling_factor,
    score_function,
    expert_bias=None,
):
    x = logits.float()
    num_experts = x.shape[1]

    if score_function == 0:
        intermediate = torch.sigmoid(x)
        scores = intermediate
        if expert_bias is not None:
            scores = scores + expert_bias.float()
        indices = scores.topk(topk, dim=-1).indices
        routing_map = _make_routing_map(indices, num_experts)
        selected = torch.where(routing_map, intermediate, torch.zeros_like(x))
        if topk > 1:
            selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
        probs = selected * scaling_factor
        return probs.to(logits.dtype), routing_map, intermediate

    if score_function == 1 and use_pre_softmax:
        intermediate = torch.softmax(x, dim=-1)
        indices = intermediate.topk(topk, dim=-1).indices
        routing_map = _make_routing_map(indices, num_experts)
        probs = torch.where(routing_map, intermediate * scaling_factor, 0.0)
        return probs.to(logits.dtype), routing_map, intermediate

    if score_function == 1:
        indices = x.topk(topk, dim=-1).indices
        routing_map = _make_routing_map(indices, num_experts)
        selected_logits = x.gather(1, indices)
        selected_probs = torch.softmax(selected_logits, dim=-1)
        probs = torch.zeros_like(x)
        probs.scatter_(1, indices, selected_probs * scaling_factor)
        intermediate = torch.full_like(x, float("-inf"))
        intermediate.scatter_(1, indices, selected_probs)
        return probs.to(logits.dtype), routing_map, intermediate

    intermediate = x
    act = _sqrtsoftplus(x)
    scores = act
    if expert_bias is not None:
        scores = scores + expert_bias.float()
    indices = scores.topk(topk, dim=-1).indices
    routing_map = _make_routing_map(indices, num_experts)
    selected = torch.where(routing_map, act, torch.zeros_like(x))
    if topk > 1:
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
    probs = selected * scaling_factor
    return probs.to(logits.dtype), routing_map, intermediate


def _reference_fwd(
    logits,
    topk,
    use_pre_softmax,
    scaling_factor,
    score_function,
    expert_bias=None,
):
    if TE_FWD is None:
        return _torch_reference_fwd(
            logits,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
            expert_bias,
        )

    try:
        return TE_FWD(
            logits,
            topk,
            use_pre_softmax,
            None,
            None,
            scaling_factor,
            SCORE_NAMES[score_function],
            expert_bias,
            0,
            None,
        )
    except TypeError:
        return _torch_reference_fwd(
            logits,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
            expert_bias,
        )


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("num_experts", [8, 64, 256])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize(
    ("score_function", "use_pre_softmax"),
    [(0, True), (1, True), (1, False), (2, True)],
    ids=["sigmoid", "softmax_pre", "softmax_post", "sqrtsoftplus"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_te_fused_topk_with_score_function_fwd(
    num_tokens,
    num_experts,
    topk,
    score_function,
    use_pre_softmax,
    dtype,
):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(0)
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)

    result = te_fused_topk_with_score_function_fwd(
        logits,
        topk,
        use_pre_softmax=use_pre_softmax,
        scaling_factor=1.0,
        score_function=score_function,
    )
    expected = _reference_fwd(
        logits,
        topk,
        use_pre_softmax,
        1.0,
        score_function,
    )

    torch.testing.assert_close(result[0], expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result[1].bool(), expected[1].bool())
    torch.testing.assert_close(result[2], expected[2].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("score_function", [0, 2], ids=["sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_te_fused_topk_with_score_function_fwd_with_expert_bias(
    score_function,
    dtype,
):
    torch.manual_seed(0)
    num_tokens, num_experts, topk = 16, 64, 8
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)
    expert_bias = torch.randn(num_experts, device="cuda", dtype=dtype)

    result = te_fused_topk_with_score_function_fwd(
        logits,
        topk,
        scaling_factor=1.0,
        score_function=score_function,
        expert_bias=expert_bias,
    )
    expected = _reference_fwd(
        logits,
        topk,
        True,
        1.0,
        score_function,
        expert_bias,
    )

    torch.testing.assert_close(result[0], expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result[1].bool(), expected[1].bool())
    torch.testing.assert_close(result[2], expected[2].float(), rtol=1e-2, atol=1e-2)
