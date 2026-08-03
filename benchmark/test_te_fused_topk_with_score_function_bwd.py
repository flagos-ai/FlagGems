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

import pytest
import torch

from flag_gems.fused.te_fused_topk_with_score_function_bwd import (
    te_fused_topk_with_score_function_bwd,
)

from . import base

# Importing TransformerEngine may configure the root logger. Keep the optional
# import at module scope, consistent with the other TE benchmarks in this tree.
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "fused_topk_with_score_function_bwd", None)
except ImportError:
    TE_OP = None


def _te_fused_topk_with_score_function_bwd(
    routing_map,
    intermediate,
    grad_probs,
    topk,
    use_pre_softmax=True,
    scaling_factor=1.0,
    score_function=1,
):
    """TransformerEngine CUDA baseline with the same allocating API as Gems."""
    score_names = {0: "sigmoid", 1: "softmax", 2: "sqrtsoftplus"}
    num_tokens = routing_map.shape[0]
    num_experts = routing_map.shape[1]
    te_grad_probs = grad_probs.float()
    try:
        grad_logits = TE_OP(
            num_tokens,
            num_experts,
            routing_map,
            intermediate,
            te_grad_probs,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_names[score_function],
        )
    except TypeError:
        grad_logits = torch.empty_like(te_grad_probs)
        TE_OP(
            routing_map,
            intermediate,
            te_grad_probs,
            grad_logits,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_names[score_function],
        )
    return grad_logits.to(grad_probs.dtype)


def _pytorch_fused_topk_with_score_function_bwd(
    routing_map,
    intermediate,
    grad_probs,
    topk,
    use_pre_softmax=True,
    scaling_factor=1.0,
    score_function=1,
):
    """PyTorch reference baseline used when TransformerEngine is unavailable."""
    grad = grad_probs.float() * scaling_factor
    act = intermediate.float()
    routed = routing_map.bool()

    if score_function == 1:
        if use_pre_softmax:
            masked_grad = torch.where(routed, grad, torch.zeros_like(grad))
            dot = (masked_grad * act).sum(dim=-1, keepdim=True)
            result = act * (masked_grad - dot)
        else:
            dot = (grad * act * routed).sum(dim=-1, keepdim=True)
            result = torch.where(routed, act * (grad - dot), torch.zeros_like(grad))
        return result.to(grad_probs.dtype)

    if score_function == 0:
        act_val = act
    else:
        softplus = torch.where(act > 20.0, act, torch.log1p(torch.exp(act)))
        act_val = torch.sqrt(softplus)

    if topk > 1:
        sum_act = (act_val * routed).sum(dim=-1, keepdim=True) + 1e-20
        sum_grad_act = (grad * act_val * routed).sum(dim=-1, keepdim=True)
        result = torch.where(
            routed,
            grad / sum_act - sum_grad_act / (sum_act * sum_act),
            torch.zeros_like(grad),
        )
    else:
        result = torch.where(routed, grad, torch.zeros_like(grad))

    if score_function == 0:
        result = result * act_val * (1.0 - act_val)
    else:
        sig = 1.0 / (1.0 + torch.exp(-act))
        dy_dx = torch.where(
            act > 20.0,
            1.0 / (2.0 * act_val + 1e-20),
            sig / (2.0 * act_val + 1e-20),
        )
        result = result * dy_dx

    return result.to(grad_probs.dtype)


def _baseline_fused_topk_with_score_function_bwd(*args, **kwargs):
    if TE_OP is not None:
        return _te_fused_topk_with_score_function_bwd(*args, **kwargs)
    return _pytorch_fused_topk_with_score_function_bwd(*args, **kwargs)


BENCHMARK_CASES = [
    (0, True),
    (1, True),
    (1, False),
]
BENCHMARK_IDS = ["sigmoid", "softmax_pre", "softmax_post"]
if TE_OP is None:
    BENCHMARK_CASES.append((2, True))
    BENCHMARK_IDS.append("sqrtsoftplus")


class FusedTopkWithScoreFunctionBwdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, topk"

    def __init__(self, score_function, use_pre_softmax=True):
        self.score_function = score_function
        self.use_pre_softmax = use_pre_softmax
        super().__init__(
            op_name="te_fused_topk_with_score_function_bwd",
            torch_op=_baseline_fused_topk_with_score_function_bwd,
            gems_op=te_fused_topk_with_score_function_bwd,
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
                dtype=torch.float32,
            )
            topk_indices = logits.topk(topk, dim=-1).indices
            routing_map = torch.zeros(
                num_tokens,
                num_experts,
                device=self.device,
                dtype=torch.bool,
            )
            routing_map.scatter_(1, topk_indices, True)

            if self.score_function == 0:
                intermediate = torch.sigmoid(logits)
            elif self.score_function == 1 and self.use_pre_softmax:
                intermediate = torch.softmax(logits, dim=-1)
            elif self.score_function == 1:
                selected = logits.gather(1, topk_indices)
                selected_probs = torch.softmax(selected, dim=-1)
                intermediate = torch.zeros_like(logits)
                intermediate.scatter_(1, topk_indices, selected_probs)
            else:
                intermediate = logits

            grad_probs = torch.randn(
                num_tokens,
                num_experts,
                device=self.device,
                dtype=dtype,
            )
            yield (
                routing_map,
                intermediate,
                grad_probs,
                topk,
                self.use_pre_softmax,
                1.0,
                self.score_function,
            )


@pytest.mark.te_fused_topk_with_score_function_bwd
@pytest.mark.parametrize(
    ("score_function", "use_pre_softmax"),
    BENCHMARK_CASES,
    # With TE installed, sqrtsoftplus is deferred until the matching fwd op is
    # added. Without TE, it can still benchmark against the PyTorch reference.
    ids=BENCHMARK_IDS,
)
def test_fused_topk_with_score_function_bwd_benchmark(score_function, use_pre_softmax):
    FusedTopkWithScoreFunctionBwdBenchmark(
        score_function=score_function,
        use_pre_softmax=use_pre_softmax,
    ).run()
