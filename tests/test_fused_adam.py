import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fused_adam
@pytest.mark.parametrize("shape", [(1024,), (2048,), (4096,), (8192,)])
# _fused_adam requires float32 for optimizer state precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_adam(shape, dtype):
    """Test fused Adam optimizer step accuracy."""
    # Create input tensors
    param = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exp_avg = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    exp_avg_sq = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    max_exp_avg_sq = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    state_step = torch.tensor([1], dtype=torch.long, device=flag_gems.device)

    # Create reference tensors (copy for comparison)
    ref_param = param.clone()
    ref_grad = grad.clone()
    ref_exp_avg = exp_avg.clone()
    ref_exp_avg_sq = exp_avg_sq.clone()

    # Compute reference manually using Adam formula
    # bias_correction1 = 1 - beta1^step
    # bias_correction2 = 1 - beta2^step
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.0
    eps = 1e-8
    step = state_step.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    # Update first moment estimate
    ref_exp_avg = beta1 * ref_exp_avg + (1 - beta1) * ref_grad
    # Update second moment estimate
    ref_exp_avg_sq = beta2 * ref_exp_avg_sq + (1 - beta2) * ref_grad * ref_grad
    # Compute bias-corrected estimates
    corrected_exp_avg = ref_exp_avg / bias_correction1
    corrected_exp_avg_sq = ref_exp_avg_sq / bias_correction2
    # Update parameters
    if weight_decay > 0:
        ref_param = ref_param - lr * (
            corrected_exp_avg / (torch.sqrt(corrected_exp_avg_sq) + eps)
            + weight_decay * ref_param
        )
    else:
        ref_param = ref_param - lr * corrected_exp_avg / (
            torch.sqrt(corrected_exp_avg_sq) + eps
        )

    # Run gems implementation
    with flag_gems.use_gems():
        gems_result = torch.ops.aten._fused_adam(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [max_exp_avg_sq],
            [state_step],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )

    # Compare results
    ref_out = utils.to_reference(ref_param)
    gems_out = utils.to_reference(gems_result[0][0])
    utils.gems_assert_close(gems_out, ref_out, dtype)


@pytest.mark.fused_adam_
@pytest.mark.parametrize("shape", [(1024,), (2048,), (4096,), (8192,)])
# _fused_adam requires float32 for optimizer state precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_adam_(shape, dtype):
    """Test in-place fused Adam optimizer step accuracy."""
    # Create input tensors
    param = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exp_avg = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    exp_avg_sq = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    max_exp_avg_sq = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    state_step = torch.tensor([1], dtype=torch.long, device=flag_gems.device)

    # Create reference tensors (copy for comparison)
    ref_param = param.clone()
    ref_grad = grad.clone()
    ref_exp_avg = exp_avg.clone()
    ref_exp_avg_sq = exp_avg_sq.clone()

    # Compute reference manually using Adam formula
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.0
    eps = 1e-8
    step = state_step.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    # Update first moment estimate
    ref_exp_avg = beta1 * ref_exp_avg + (1 - beta1) * ref_grad
    # Update second moment estimate
    ref_exp_avg_sq = beta2 * ref_exp_avg_sq + (1 - beta2) * ref_grad * ref_grad
    # Compute bias-corrected estimates
    corrected_exp_avg = ref_exp_avg / bias_correction1
    corrected_exp_avg_sq = ref_exp_avg_sq / bias_correction2
    # Update parameters
    if weight_decay > 0:
        ref_param = ref_param - lr * (
            corrected_exp_avg / (torch.sqrt(corrected_exp_avg_sq) + eps)
            + weight_decay * ref_param
        )
    else:
        ref_param = ref_param - lr * corrected_exp_avg / (
            torch.sqrt(corrected_exp_avg_sq) + eps
        )

    # Run gems inplace implementation
    with flag_gems.use_gems():
        torch.ops.aten._fused_adam_(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [max_exp_avg_sq],
            [state_step],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )

    # Compare mutated input
    gems_out = utils.to_reference(param)
    ref_out = utils.to_reference(ref_param)
    utils.gems_assert_close(gems_out, ref_out, dtype)
