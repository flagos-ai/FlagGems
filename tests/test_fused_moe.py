import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import accuracy_utils as utils
from . import conftest as cfg

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_EXPERTS_IMPL = True
except ImportError:
    HAS_VLLM_FUSED_EXPERTS_IMPL = False


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    return True


CUDA_AVAILABLE = is_cuda_available()

COSINE_SIM_THRESHOLD = 0.999
MOE_RTOL = 1e-1
MOE_ATOL = 1e-2

FUSED_EXPERTS_IMPL_CONFIGS = [
    # Scaled benchmark/test_fused_moe.py style configs for regular accuracy CI.
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    pytest.param((1, 8, 128, 256, 2), id="mixtral_tiny_1tok"),
    pytest.param((4, 8, 128, 256, 2), id="mixtral_tiny_4tok"),
    pytest.param((16, 8, 256, 512, 2), id="mixtral_tiny_16tok"),
    pytest.param((64, 8, 256, 512, 2), id="mixtral_tiny_64tok"),
    pytest.param((128, 8, 128, 256, 2), id="mixtral_tiny_128tok"),
    pytest.param((4, 256, 512, 128, 8), id="deepseek_tiny_4tok"),
    pytest.param((16, 256, 512, 128, 8), id="deepseek_tiny_16tok"),
    pytest.param((64, 256, 512, 128, 8), id="deepseek_tiny_64tok"),
]

if not cfg.QUICK_MODE:
    FUSED_EXPERTS_IMPL_CONFIGS += [
        # Full benchmark/test_fused_moe.py Mixtral-like shapes.
        pytest.param((1, 8, 4096, 14336, 2), id="mixtral_1tok"),
        pytest.param((4, 8, 4096, 14336, 2), id="mixtral_4tok"),
        pytest.param((16, 8, 4096, 14336, 2), id="mixtral_16tok"),
        pytest.param((64, 8, 4096, 14336, 2), id="mixtral_64tok"),
        pytest.param((128, 8, 4096, 14336, 2), id="mixtral_128tok"),
        pytest.param((256, 8, 4096, 14336, 2), id="mixtral_256tok"),
        pytest.param((512, 8, 4096, 14336, 2), id="mixtral_512tok"),
        # Full benchmark/test_fused_moe.py DeepSeek-V3-like shapes (TP=8 shard).
        pytest.param((1, 256, 7168, 2048, 8), id="deepseek_1tok"),
        pytest.param((4, 256, 7168, 2048, 8), id="deepseek_4tok"),
        pytest.param((16, 256, 7168, 2048, 8), id="deepseek_16tok"),
        pytest.param((64, 256, 7168, 2048, 8), id="deepseek_64tok"),
        pytest.param((128, 256, 7168, 2048, 8), id="deepseek_128tok"),
        pytest.param((256, 256, 7168, 2048, 8), id="deepseek_256tok"),
    ]


def _make_fused_experts_impl_case(config, dtype, device):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * (1.0 / hidden_size**0.5)
    ).contiguous()
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * (1.0 / intermediate_size**0.5)
    ).contiguous()

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return hidden_states, w1, w2, topk_weights.to(dtype), topk_ids


def _torch_reference_fused_experts_impl(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
):
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for token_idx in range(M):
        for topk_idx in range(topk):
            expert_idx = topk_ids[token_idx, topk_idx].item()
            router_weight = topk_weights[token_idx, topk_idx]
            gate_up = hidden_states[token_idx].float() @ w1[expert_idx].float().T
            gate, up = gate_up.chunk(2, dim=-1)
            activated = (gate * torch.sigmoid(gate)) * up
            down = activated @ w2[expert_idx].float().T
            output[token_idx] += (router_weight.float() * down).to(output.dtype)

    return output


def _assert_cosine_similarity(actual, expected):
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    denominator = actual_flat.norm() * expected_flat.norm()
    if denominator == 0:
        torch.testing.assert_close(actual_flat, expected_flat, rtol=0, atol=0)
        return

    cosine_sim = torch.dot(actual_flat, expected_flat) / denominator
    assert cosine_sim.item() >= COSINE_SIM_THRESHOLD


def _assert_fused_experts_close(actual, expected):
    _assert_cosine_similarity(actual, expected)
    torch.testing.assert_close(actual, expected, rtol=MOE_RTOL, atol=MOE_ATOL)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_EXPERTS_IMPL_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_experts_impl_matches_reference(config, dtype):
    """Test the full fused_experts_impl pipeline against a PyTorch reference."""
    device = torch.device(flag_gems.device)
    torch.manual_seed(42)

    hidden_states, w1, w2, topk_weights, topk_ids = _make_fused_experts_impl_case(
        config, dtype, device
    )

    hidden_states_ref = utils.to_reference(hidden_states, False)
    w1_ref = utils.to_reference(w1, False)
    w2_ref = utils.to_reference(w2, False)
    topk_weights_ref = utils.to_reference(topk_weights, False)
    topk_ids_ref = utils.to_reference(topk_ids, False)

    flaggems_out = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
    )
    ref_out = _torch_reference_fused_experts_impl(
        hidden_states_ref,
        w1_ref,
        w2_ref,
        topk_weights_ref,
        topk_ids_ref,
    )
    if cfg.TO_CPU:
        ref_out = ref_out.to(device=flag_gems.device)
    torch_device_fn.synchronize()

    _assert_fused_experts_close(flaggems_out, ref_out)


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(not HAS_VLLM_FUSED_EXPERTS_IMPL, reason="vLLM is required")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("config", FUSED_EXPERTS_IMPL_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_experts_impl_matches_vllm(config, dtype):
    """Compare the full FlagGems fused_experts_impl pipeline with vLLM."""
    device = torch.device(flag_gems.device)
    torch.manual_seed(42)

    hidden_states, w1, w2, topk_weights, topk_ids = _make_fused_experts_impl_case(
        config, dtype, device
    )

    flaggems_out = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
    )
    vllm_out = vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
    )
    torch_device_fn.synchronize()

    _assert_fused_experts_close(flaggems_out, vllm_out)
