import random

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE

random.seed(42)

FUSED_MOE_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (8, 4, 64, 128, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if not QUICK_MODE:
    FUSED_MOE_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        (4, 16, 512, 1024, 2),
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
        # DeepSeek-V3-like shapes (TP=8 shard)
        (1, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
    ]


def torch_fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of fused MoE."""
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            z = hidden_states[m].to(torch.float32) @ w1[e].T.to(torch.float32)
            D = z.shape[-1] // 2
            gate = z[:D]
            up = z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            r = s @ w2[e].T.to(torch.float32)
            output[m] += (weight.to(torch.float32) * r).to(output.dtype)

    return output


@pytest.mark.inplace_fused_experts
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_inplace_fused_experts_accuracy(config, dtype):
    """Test inplace_fused_experts writes correct results into hidden_states."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    # inplace_fused_experts modifies hidden_states in-place
    flag_gems.inplace_fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    if flag_gems.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()

    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)
    torch.testing.assert_close(hidden_states, ref, rtol=rtol, atol=atol)


@pytest.mark.inplace_fused_experts
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_inplace_fused_experts_matches_outplace(config, dtype):
    """Test that inplace_fused_experts produces the same result as outplace_fused_experts."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # outplace result
    outplace_result = flag_gems.outplace_fused_experts(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    # inplace result
    inplace_input = hidden_states.clone()
    flag_gems.inplace_fused_experts(
        inplace_input,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    if flag_gems.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()

    torch.testing.assert_close(inplace_input, outplace_result, rtol=0, atol=0)
