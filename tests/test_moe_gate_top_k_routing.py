import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def assert_topk_indices_match(res_indices, ref_indices, ref_weights):
    res_indices = res_indices.cpu()
    ref_indices = ref_indices.cpu()
    ref_weights = ref_weights.cpu()

    if torch.equal(res_indices, ref_indices):
        return

    for row in range(ref_indices.shape[0]):
        start = 0
        while start < ref_indices.shape[1]:
            end = start + 1
            while (
                end < ref_indices.shape[1]
                and ref_weights[row, end] == ref_weights[row, start]
            ):
                end += 1

            res_group = res_indices[row, start:end]
            ref_group = ref_indices[row, start:end]
            if end - start == 1:
                torch.testing.assert_close(res_group, ref_group, atol=0, rtol=0)
            else:
                torch.testing.assert_close(
                    torch.sort(res_group).values,
                    torch.sort(ref_group).values,
                    atol=0,
                    rtol=0,
                )
            start = end


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
    assert_topk_indices_match(res_topk_indices, ref_topk_indices, ref_topk_weights)

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
