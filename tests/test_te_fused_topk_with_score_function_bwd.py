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

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "fused_topk_with_score_function_bwd", None)
    TE_FWD = getattr(tex, "fused_topk_with_score_function_fwd", None)
except ImportError:
    TE_OP = None
    TE_FWD = None

pytestmark = pytest.mark.te_fused_topk_with_score_function_bwd


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def _sqrtsoftplus(x):
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    return torch.sqrt(sp)


def _reference_sigmoid_bwd(routing_map, intermediate, grad_probs, topk, scaling_factor):
    """Reference backward for sigmoid score function."""
    grad = grad_probs.float() * scaling_factor
    act = intermediate.float()
    routed = routing_map.bool()

    if topk > 1:
        sum_act = (act * routed).sum(dim=-1, keepdim=True) + 1e-20
        sum_grad_act = (grad * act * routed).sum(dim=-1, keepdim=True)
        g = torch.where(
            routed,
            grad / sum_act - sum_grad_act / (sum_act * sum_act),
            torch.zeros_like(grad),
        )
    else:
        g = torch.where(routed, grad, torch.zeros_like(grad))

    # sigmoid backward: dy/dx = y * (1 - y)
    g = g * act * (1.0 - act)
    return g.to(grad_probs.dtype)


def _reference_softmax_bwd(
    routing_map, intermediate, grad_probs, topk, scaling_factor, use_pre_softmax
):
    """Reference backward for softmax score function."""
    grad = grad_probs.float() * scaling_factor
    act = intermediate.float()
    routed = routing_map.bool()

    if use_pre_softmax:
        masked_grad = torch.where(routed, grad, torch.zeros_like(grad))
        dot = (masked_grad * act).sum(dim=-1, keepdim=True)
        g = act * (masked_grad - dot)
    else:
        dot = (grad * act * routed).sum(dim=-1, keepdim=True)
        g = torch.where(routed, act * (grad - dot), torch.zeros_like(grad))

    return g.to(grad_probs.dtype)


def _reference_sqrtsoftplus_bwd(
    routing_map, intermediate, grad_probs, topk, scaling_factor
):
    """Reference backward for sqrtsoftplus score function."""
    grad = grad_probs.float() * scaling_factor
    x = intermediate.float()
    routed = routing_map.bool()

    # Recompute sqrtsoftplus
    act_val = _sqrtsoftplus(x)

    if topk > 1:
        sum_act = (act_val * routed).sum(dim=-1, keepdim=True) + 1e-20
        sum_grad_act = (grad * act_val * routed).sum(dim=-1, keepdim=True)
        g = torch.where(
            routed,
            grad / sum_act - sum_grad_act / (sum_act * sum_act),
            torch.zeros_like(grad),
        )
    else:
        g = torch.where(routed, grad, torch.zeros_like(grad))

    # sqrtsoftplus backward follows TE's large-x softplus approximation.
    sig = _sigmoid(x)
    dy_dx = torch.where(
        x > 20.0,
        1.0 / (2.0 * act_val + 1e-20),
        sig / (2.0 * act_val + 1e-20),
    )
    g = g * dy_dx
    return g.to(grad_probs.dtype)


def _make_test_inputs(num_tokens, num_experts, topk, score_function, dtype, device):
    """Generate test inputs simulating what the forward pass would produce."""
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

    # Create routing map: select topk experts per token
    _, topk_indices = logits.topk(topk, dim=-1)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.uint8, device=device)
    routing_map.scatter_(1, topk_indices, 1)

    # Create intermediate based on score function
    if score_function == 0:  # sigmoid
        intermediate = _sigmoid(logits)
    elif score_function == 1:  # softmax
        intermediate = torch.softmax(logits, dim=-1)
    else:  # sqrtsoftplus: stores original logits
        intermediate = logits.clone()

    intermediate = intermediate.float()

    # grad_probs
    grad_probs = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)

    return routing_map, intermediate, grad_probs


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("num_experts", [8, 64, 256])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
class TestFusedTopkScoreFnBwdSigmoid:
    def test_sigmoid(self, num_tokens, num_experts, topk, dtype):
        if topk > num_experts:
            pytest.skip("topk > num_experts")
        device = "cuda"
        scaling_factor = 1.0

        routing_map, intermediate, grad_probs = _make_test_inputs(
            num_tokens, num_experts, topk, 0, dtype, device
        )
        grad_probs = grad_probs.to(dtype)

        result = te_fused_topk_with_score_function_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            scaling_factor=scaling_factor,
            score_function=0,
        )

        expected = _reference_sigmoid_bwd(
            routing_map, intermediate, grad_probs, topk, scaling_factor
        )

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("num_experts", [8, 64, 256])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("use_pre_softmax", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
class TestFusedTopkScoreFnBwdSoftmax:
    def test_softmax(self, num_tokens, num_experts, topk, use_pre_softmax, dtype):
        if topk > num_experts:
            pytest.skip("topk > num_experts")
        device = "cuda"
        scaling_factor = 1.0

        routing_map, intermediate, grad_probs = _make_test_inputs(
            num_tokens, num_experts, topk, 1, dtype, device
        )
        grad_probs = grad_probs.to(dtype)

        result = te_fused_topk_with_score_function_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=scaling_factor,
            score_function=1,
        )

        expected = _reference_softmax_bwd(
            routing_map, intermediate, grad_probs, topk, scaling_factor, use_pre_softmax
        )

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("num_experts", [8, 64, 256])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
class TestFusedTopkScoreFnBwdSqrtsoftplus:
    def test_sqrtsoftplus(self, num_tokens, num_experts, topk, dtype):
        if topk > num_experts:
            pytest.skip("topk > num_experts")
        device = "cuda"
        scaling_factor = 1.0

        routing_map, intermediate, grad_probs = _make_test_inputs(
            num_tokens, num_experts, topk, 2, dtype, device
        )
        grad_probs = grad_probs.to(dtype)

        result = te_fused_topk_with_score_function_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            scaling_factor=scaling_factor,
            score_function=2,
        )

        expected = _reference_sqrtsoftplus_bwd(
            routing_map, intermediate, grad_probs, topk, scaling_factor
        )

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("scaling_factor", [0.5, 2.0, 0.1])
def test_scaling_factor(scaling_factor):
    """Test that scaling_factor is correctly applied."""
    device = "cuda"
    num_tokens, num_experts, topk = 16, 8, 2

    routing_map, intermediate, grad_probs = _make_test_inputs(
        num_tokens, num_experts, topk, 0, torch.float32, device
    )

    result = te_fused_topk_with_score_function_bwd(
        routing_map,
        intermediate,
        grad_probs,
        topk=topk,
        scaling_factor=scaling_factor,
        score_function=0,
    )

    expected = _reference_sigmoid_bwd(
        routing_map, intermediate, grad_probs, topk, scaling_factor
    )

    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


def _te_reference_bwd(
    routing_map,
    intermediate,
    grad_probs,
    topk,
    use_pre_softmax,
    scaling_factor,
    score_function,
):
    """Call TransformerEngine's CUDA kernel as reference."""
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


def _baseline_reference_bwd(
    routing_map,
    intermediate,
    grad_probs,
    topk,
    use_pre_softmax,
    scaling_factor,
    score_function,
):
    """Use TE native reference when available, otherwise PyTorch reference."""
    if TE_OP is not None:
        return _te_reference_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
        )
    if score_function == 0:
        return _reference_sigmoid_bwd(
            routing_map, intermediate, grad_probs, topk, scaling_factor
        )
    if score_function == 1:
        return _reference_softmax_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk,
            scaling_factor,
            use_pre_softmax,
        )
    return _reference_sqrtsoftplus_bwd(
        routing_map, intermediate, grad_probs, topk, scaling_factor
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512])
@pytest.mark.parametrize("num_experts", [8, 64, 256])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
class TestFusedTopkScoreFnBwdVsTE:
    """Compare FlagGems against TE native or PyTorch reference fallback."""

    def _make_inputs(
        self,
        num_tokens,
        num_experts,
        topk,
        score_function,
        dtype,
        device,
        use_pre_softmax=True,
    ):
        logits = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        score_names = {0: "sigmoid", 1: "softmax", 2: "sqrtsoftplus"}
        if TE_FWD is not None:
            try:
                _probs, routing_map, intermediate = TE_FWD(
                    logits,
                    topk,
                    use_pre_softmax,
                    None,
                    None,
                    1.0,
                    score_names[score_function],
                    None,
                    0,
                    None,
                )
            except TypeError:
                routing_map, intermediate = self._manual_inputs(
                    logits, topk, score_function, use_pre_softmax
                )
        else:
            routing_map, intermediate = self._manual_inputs(
                logits, topk, score_function, use_pre_softmax
            )
        grad_probs = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
        return routing_map, intermediate, grad_probs

    @staticmethod
    def _manual_inputs(logits, topk, score_function, use_pre_softmax):
        _, topk_indices = logits.topk(topk, dim=-1)
        routing_map = torch.zeros(
            logits.shape[0], logits.shape[1], dtype=torch.bool, device=logits.device
        )
        routing_map.scatter_(1, topk_indices, True)

        if score_function == 0:
            intermediate = torch.sigmoid(logits)
        elif score_function == 1 and use_pre_softmax:
            intermediate = torch.softmax(logits, dim=-1)
        elif score_function == 1:
            selected = logits.gather(1, topk_indices)
            selected_probs = torch.softmax(selected, dim=-1)
            intermediate = torch.full_like(logits, float("-inf"))
            intermediate.scatter_(1, topk_indices, selected_probs)
        else:
            intermediate = logits.clone()
        return routing_map, intermediate.float()

    def test_sigmoid_vs_te(self, num_tokens, num_experts, topk, dtype):
        if topk > num_experts:
            pytest.skip("topk > num_experts")
        device = "cuda"
        scaling_factor = 1.0
        routing_map, intermediate, grad_probs = self._make_inputs(
            num_tokens, num_experts, topk, 0, dtype, device
        )

        result = te_fused_topk_with_score_function_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            scaling_factor=scaling_factor,
            score_function=0,
        )
        expected = _baseline_reference_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk,
            True,
            scaling_factor,
            0,
        )

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_softmax_pre_vs_te(self, num_tokens, num_experts, topk, dtype):
        if topk > num_experts:
            pytest.skip("topk > num_experts")
        device = "cuda"
        scaling_factor = 1.0
        routing_map, intermediate, grad_probs = self._make_inputs(
            num_tokens, num_experts, topk, 1, dtype, device, True
        )

        result = te_fused_topk_with_score_function_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            use_pre_softmax=True,
            scaling_factor=scaling_factor,
            score_function=1,
        )
        expected = _baseline_reference_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk,
            True,
            scaling_factor,
            1,
        )

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_softmax_post_vs_te(self, num_tokens, num_experts, topk, dtype):
        if topk > num_experts:
            pytest.skip("topk > num_experts")
        device = "cuda"
        scaling_factor = 1.0
        routing_map, intermediate, grad_probs = self._make_inputs(
            num_tokens, num_experts, topk, 1, dtype, device, False
        )

        result = te_fused_topk_with_score_function_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            use_pre_softmax=False,
            scaling_factor=scaling_factor,
            score_function=1,
        )
        expected = _baseline_reference_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk,
            False,
            scaling_factor,
            1,
        )

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_sqrtsoftplus_vs_te(self, num_tokens, num_experts, topk, dtype):
        if topk > num_experts:
            pytest.skip("topk > num_experts")
        if TE_OP is not None:
            pytest.skip(
                "sqrtsoftplus vs TE is deferred until the matching fwd op is added; "
                "covered by the Python reference test for now"
            )
        device = "cuda"
        scaling_factor = 1.0
        routing_map, intermediate, grad_probs = self._make_inputs(
            num_tokens, num_experts, topk, 2, dtype, device, False
        )

        result = te_fused_topk_with_score_function_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            scaling_factor=scaling_factor,
            score_function=2,
        )
        expected = _baseline_reference_bwd(
            routing_map,
            intermediate,
            grad_probs,
            topk,
            False,
            scaling_factor,
            2,
        )

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
