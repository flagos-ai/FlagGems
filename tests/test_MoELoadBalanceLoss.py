import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def reference_MoELoadBalanceLoss(gate_logits):
    num_tokens, num_experts = gate_logits.shape
    softmax_probs = torch.softmax(gate_logits, dim=-1)
    expert_loads = torch.sum(softmax_probs, dim=0)
    expert_loads_normalized = expert_loads / num_tokens
    loss = num_experts * torch.sum(expert_loads_normalized**2)
    return loss


# Representative (num_tokens, num_experts) combos covering small to medium MoE layers.
MOE_LOAD_BALANCE_LOSS_SHAPES = [
    (8, 4),
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
]


@pytest.mark.MoELoadBalanceLoss
@pytest.mark.skipif(
    flag_gems.vendor_name == "metax",
    reason="MetaX backend path for MoELoadBalanceLoss is not stable in CI.",
)
@pytest.mark.parametrize("shape", MOE_LOAD_BALANCE_LOSS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_MoELoadBalanceLoss(shape, dtype):
    """Test MoE load balance loss accuracy against reference implementation"""
    num_tokens, num_experts = shape
    gate_logits = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_gate_logits = utils.to_reference(gate_logits)

    ref_out = reference_MoELoadBalanceLoss(ref_gate_logits)
    with flag_gems.use_gems():
        res_out = flag_gems.MoELoadBalanceLoss(gate_logits)

    # Use larger tolerance for float16/bfloat16
    if dtype == torch.float32:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-4)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2)
