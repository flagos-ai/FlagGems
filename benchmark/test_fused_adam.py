import pytest
import torch

import flag_gems

from . import base


def fused_adam_input_fn(shape, dtype, device):
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    exp_avg = torch.zeros(shape, dtype=dtype, device=device)
    exp_avg_sq = torch.zeros(shape, dtype=dtype, device=device)
    max_exp_avg_sq = torch.zeros(shape, dtype=dtype, device=device)
    state_step = torch.tensor([1], dtype=torch.long, device=device)
    yield param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step


def torch_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step):
    # Reference: compute manually using Adam formula
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = state_step.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    # Update first moment estimate
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    # Update second moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    # Compute bias-corrected estimates
    corrected_exp_avg = exp_avg / bias_correction1
    corrected_exp_avg_sq = exp_avg_sq / bias_correction2
    # Update parameters
    param = param - lr * corrected_exp_avg / (torch.sqrt(corrected_exp_avg_sq) + eps)
    return param


@pytest.mark.fused_adam
def test_fused_adam():
    def gems_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step):
        return flag_gems._fused_adam(
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

    bench = base.GenericBenchmark(
        input_fn=fused_adam_input_fn,
        op_name="fused_adam",
        torch_op=torch_op,
        # _fused_adam only supports float32 for optimizer state precision
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.fused_adam_
def test_fused_adam_():
    def gems_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step):
        flag_gems._fused_adam_(
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
        return param

    bench = base.GenericBenchmark(
        input_fn=fused_adam_input_fn,
        op_name="fused_adam_",
        torch_op=torch_op,
        # _fused_adam only supports float32 for optimizer state precision
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_op)
    bench.run()
