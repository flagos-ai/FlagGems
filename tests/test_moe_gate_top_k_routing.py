import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("num_experts", [4, 8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_moe_gate_top_k_routing(num_tokens, num_experts, topk, dtype):
    if topk > num_experts:
        pytest.skip("topk must be <= num_experts")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create gate logits
    gate_logits = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device=flag_gems.device
    )

    # Reference implementation: apply softmax then topk
    ref_gate_logits = utils.to_reference(gate_logits)
    ref_probs = torch.softmax(ref_gate_logits, dim=-1)
    ref_topk_weights, ref_topk_indices = torch.topk(ref_probs, topk, dim=-1)
    # Convert weights to float32 for comparison (our implementation returns float32)
    ref_topk_weights = ref_topk_weights.to(torch.float32)

    # FlagGems implementation
    with flag_gems.use_gems():
        res_topk_weights, res_topk_indices = flag_gems.moe_gate_top_k_routing(
            gate_logits, topk
        )

    # Compare indices - convert both to same type for comparison
    ref_topk_indices = ref_topk_indices.to(res_topk_indices.dtype)
    utils.gems_assert_equal(res_topk_indices, ref_topk_indices)

    # Compare weights (both should be float32 now)
    # Use larger tolerance for float16/bfloat16 due to numerical precision
    if dtype == torch.float32:
        utils.gems_assert_close(res_topk_weights, ref_topk_weights, torch.float32)
    elif dtype == torch.float16:
        utils.gems_assert_close(
            res_topk_weights, ref_topk_weights, torch.float32, atol=5e-4
        )
    else:  # bfloat16
        utils.gems_assert_close(
            res_topk_weights, ref_topk_weights, torch.float32, atol=2e-3
        )
