# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0

import pytest
import torch

from flag_gems.fused.te_fused_topk_with_score_function_fwd import (
    te_fused_topk_with_score_function_fwd,
)

from . import base

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_FWD = getattr(tex, "fused_topk_with_score_function_fwd", None)
except ImportError:
    TE_FWD = None

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
    use_pre_softmax=True,
    num_groups=None,
    group_topk=None,
    scaling_factor=1.0,
    score_function=1,
    expert_bias=None,
):
    _ = num_groups, group_topk
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
        return (selected * scaling_factor).to(logits.dtype), routing_map, intermediate

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
    return (selected * scaling_factor).to(logits.dtype), routing_map, intermediate


def _baseline_fwd(*args, **kwargs):
    if TE_FWD is None:
        return _torch_reference_fwd(*args, **kwargs)
    logits = args[0]
    topk = args[1]
    use_pre_softmax = args[2]
    num_groups = args[3]
    group_topk = args[4]
    scaling_factor = args[5]
    score_function = args[6]
    expert_bias = args[7]
    try:
        return TE_FWD(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            SCORE_NAMES[score_function],
            expert_bias,
            0,
            None,
        )
    except TypeError:
        return _torch_reference_fwd(*args, **kwargs)


class FusedTopkWithScoreFunctionFwdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, topk"

    def __init__(self, score_function, use_pre_softmax=True):
        self.score_function = score_function
        self.use_pre_softmax = use_pre_softmax
        super().__init__(
            op_name="te_fused_topk_with_score_function_fwd",
            torch_op=_baseline_fwd,
            gems_op=te_fused_topk_with_score_function_fwd,
            dtypes=[torch.float16, torch.bfloat16, torch.float32],
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            (1, 64, 8),
            (16, 64, 8),
            (128, 64, 8),
            (512, 128, 8),
            (1024, 128, 8),
            (2048, 256, 8),
            (4096, 256, 8),
        ]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        for num_tokens, num_experts, topk in self.shapes:
            torch.manual_seed(0)
            logits = torch.randn(
                num_tokens,
                num_experts,
                device=self.device,
                dtype=dtype,
            )
            yield (
                logits,
                topk,
                self.use_pre_softmax,
                None,
                None,
                1.0,
                self.score_function,
                None,
            )


@pytest.mark.te_fused_topk_with_score_function_fwd
@pytest.mark.parametrize(
    ("score_function", "use_pre_softmax"),
    [(0, True), (1, True), (1, False), (2, True)],
    ids=["sigmoid", "softmax_pre", "softmax_post", "sqrtsoftplus"],
)
def test_te_fused_topk_with_score_function_fwd_benchmark(
    score_function,
    use_pre_softmax,
):
    FusedTopkWithScoreFunctionFwdBenchmark(
        score_function=score_function,
        use_pre_softmax=use_pre_softmax,
    ).run()
