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
import torch.nn.functional as F

from flag_gems.fused.fused_topk_with_score_function_bwd import (
    fused_topk_with_score_function_bwd,
)

from . import base


def _torch_fused_topk_with_score_function_bwd(
    routing_map,
    intermediate,
    grad_probs,
    topk,
    use_pre_softmax=True,
    scaling_factor=1.0,
    score_function=1,
):
    """PyTorch baseline matching TransformerEngine backward semantics."""
    grad = grad_probs.float() * scaling_factor
    act = intermediate.float()
    routed = routing_map.bool()

    if score_function == 1:
        masked_grad = torch.where(routed, grad, 0.0)
        dot = (masked_grad * act).sum(dim=-1, keepdim=True)
        if use_pre_softmax:
            grad_logits = act * (masked_grad - dot)
        else:
            grad_logits = torch.where(routed, act * (grad - dot), 0.0)
        return grad_logits.to(grad_probs.dtype)

    if score_function == 2:
        logits = act
        act = torch.sqrt(F.softplus(logits))

    if topk > 1:
        sum_act = torch.where(routed, act, 0.0).sum(dim=-1, keepdim=True) + 1e-20
        sum_grad_act = torch.where(routed, grad * act, 0.0).sum(dim=-1, keepdim=True)
        grad = torch.where(
            routed,
            grad / sum_act - sum_grad_act / (sum_act * sum_act),
            0.0,
        )
    else:
        grad = torch.where(routed, grad, 0.0)

    if score_function == 0:
        grad_logits = grad * act * (1.0 - act)
    else:
        sigmoid = torch.sigmoid(logits)
        grad_logits = grad * sigmoid / (2.0 * act + 1e-20)
    return grad_logits.to(grad_probs.dtype)


class FusedTopkWithScoreFunctionBwdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, topk"

    def __init__(self, score_function, use_pre_softmax=True):
        self.score_function = score_function
        self.use_pre_softmax = use_pre_softmax
        super().__init__(
            op_name="fused_topk_with_score_function_bwd",
            torch_op=_torch_fused_topk_with_score_function_bwd,
            gems_op=fused_topk_with_score_function_bwd,
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


@pytest.mark.fused_topk_with_score_function_bwd
@pytest.mark.parametrize(
    ("score_function", "use_pre_softmax"),
    [(0, True), (1, True), (1, False), (2, True)],
    ids=["sigmoid", "softmax_pre", "softmax_post", "sqrtsoftplus"],
)
def test_fused_topk_with_score_function_bwd_benchmark(score_function, use_pre_softmax):
    FusedTopkWithScoreFunctionBwdBenchmark(
        score_function=score_function,
        use_pre_softmax=use_pre_softmax,
    ).run()
