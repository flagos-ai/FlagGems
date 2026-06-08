import random
import time

import pytest
import torch

import flag_gems
from flag_gems import topk_softmax

from . import accuracy_utils as utils

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def generate_test_params():
    params = [torch.int32, torch.int64]
    if utils.SkipVersion("torch", ">2.2"):
        params.append(torch.uint32)

    return params


@pytest.mark.skipif(
    flag_gems.vendor_name == "metax", reason="Issue #2857: RuntimeError"
)
@pytest.mark.topk_softmax
@pytest.mark.parametrize("index_dtype", generate_test_params())
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    [
        (1, 4, 2),
        (4, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
        (128, 128, 16),
        (500, 255, 30),
        (512, 256, 32),
        (1024, 512, 32),
    ],
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax(
    num_tokens, num_experts, topk, input_dtype, index_dtype, renormalize
):
    if flag_gems.vendor_name == "mthreads" and index_dtype == torch.uint32:
        # Issue #2858: torch musa does not support uint32
        index_dtype = torch.int64

    try:
        from vllm._custom_ops import topk_softmax as vllm_topk_softmax
    except (ImportError, AttributeError):
        pytest.skip("vLLM topk_softmax not available")

    torch.manual_seed(42)
    device = flag_gems.device

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    vllm_weights = torch.empty(num_tokens, topk, device=device, dtype=torch.float32)
    vllm_indices = torch.empty(num_tokens, topk, device=device, dtype=index_dtype)
    vllm_token_expert = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)

    vllm_topk_softmax(
        vllm_weights,
        vllm_indices,
        vllm_token_expert,
        gating_output,
        renormalize,
    )

    gems_weights = torch.empty_like(vllm_weights)
    gems_indices = torch.empty_like(vllm_indices)
    gems_token_expert = torch.empty_like(vllm_token_expert)

    topk_softmax(
        gems_weights,
        gems_indices,
        gems_token_expert,
        gating_output,
        renormalize,
    )

    assert torch.allclose(
        gems_weights, vllm_weights, atol=1e-5
    ), "topk_weights mismatch"
    assert torch.equal(
        gems_indices.cpu(), vllm_indices.cpu()
    ), "topk_indices mismatch (fp32)"
    assert torch.equal(
        gems_token_expert.cpu(), vllm_token_expert.cpu()
    ), "token_expert_indices mismatch"

    if renormalize:
        sums = gems_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
