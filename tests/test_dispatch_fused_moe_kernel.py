import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids):
    """
    Pure PyTorch reference implementation of fused MoE with SiLU activation.

    This implements the same logic as fused_experts_impl:
      1. For each token, select top-k experts
      2. GEMM1: hidden_states @ w1^T -> gate+up projections
      3. SiLU activation on gate, element-wise multiply with up
      4. GEMM2: activated @ w2^T -> down projection
      5. Weighted sum across experts
    """
    num_tokens, hidden_size = hidden_states.shape
    num_experts, intermediate_size_2x, _ = w1.shape
    intermediate_size = intermediate_size_2x // 2
    top_k = topk_ids.shape[1]

    output = torch.zeros(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )

    for i in range(num_tokens):
        for j in range(top_k):
            expert_id = topk_ids[i, j].item()
            weight = topk_weights[i, j]

            # GEMM1: [1, hidden_size] @ [intermediate_size*2, hidden_size]^T
            intermediate = hidden_states[i : i + 1] @ w1[expert_id].T

            # SiLU gate + mul
            gate = intermediate[:, :intermediate_size]
            up = intermediate[:, intermediate_size:]
            activated = torch.nn.functional.silu(gate) * up

            # GEMM2: [1, intermediate_size] @ [hidden_size, intermediate_size]^T
            down = activated @ w2[expert_id].T

            output[i] += weight * down.squeeze(0)

    return output


@pytest.mark.dispatch_fused_moe_kernel
@pytest.mark.parametrize(
    "num_tokens, num_experts, hidden_size, intermediate_size, topk",
    [
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
        (32, 8, 512, 1024, 2),
        (8, 16, 256, 512, 4),
        (1, 8, 128, 256, 2),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dispatch_fused_moe_kernel_accuracy(
    num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype
):
    """Test dispatch_fused_moe_kernel accuracy through fused_experts_impl."""
    device = flag_gems.device

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = (
        torch.randn(
            num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
        )
        * 0.01
    )
    w2 = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
        )
        * 0.01
    )

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    ref_hidden = utils.to_reference(hidden_states)
    ref_w1 = utils.to_reference(w1)
    ref_w2 = utils.to_reference(w2)
    ref_topk_weights = utils.to_reference(topk_weights)
    ref_topk_ids = utils.to_reference(topk_ids)

    ref_out = torch_fused_moe_reference(
        ref_hidden, ref_w1, ref_w2, ref_topk_weights, ref_topk_ids
    )

    res_out = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.dispatch_fused_moe_kernel
@pytest.mark.parametrize(
    "num_tokens, num_experts, hidden_size, intermediate_size, topk",
    [
        (64, 256, 512, 1024, 8),
        (16, 256, 256, 512, 8),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dispatch_fused_moe_kernel_many_experts(
    num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype
):
    """Test with many experts (DeepSeek-V3-like configuration)."""
    device = flag_gems.device

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = (
        torch.randn(
            num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
        )
        * 0.01
    )
    w2 = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
        )
        * 0.01
    )

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    ref_hidden = utils.to_reference(hidden_states)
    ref_w1 = utils.to_reference(w1)
    ref_w2 = utils.to_reference(w2)
    ref_topk_weights = utils.to_reference(topk_weights)
    ref_topk_ids = utils.to_reference(topk_ids)

    ref_out = torch_fused_moe_reference(
        ref_hidden, ref_w1, ref_w2, ref_topk_weights, ref_topk_ids
    )

    res_out = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.dispatch_fused_moe_kernel
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dispatch_fused_moe_kernel_single_token(dtype):
    """Test with a single token (edge case for block alignment)."""
    device = flag_gems.device
    num_tokens = 1
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = (
        torch.randn(
            num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
        )
        * 0.01
    )
    w2 = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
        )
        * 0.01
    )

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    ref_hidden = utils.to_reference(hidden_states)
    ref_w1 = utils.to_reference(w1)
    ref_w2 = utils.to_reference(w2)
    ref_topk_weights = utils.to_reference(topk_weights)
    ref_topk_ids = utils.to_reference(topk_ids)

    ref_out = torch_fused_moe_reference(
        ref_hidden, ref_w1, ref_w2, ref_topk_weights, ref_topk_ids
    )

    res_out = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
