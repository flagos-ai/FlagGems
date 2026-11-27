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

"""Accuracy and state update tests for fused Adam."""

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
    gems_result = flag_gems._fused_adam(
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

    # Compare mutated input
    gems_out = utils.to_reference(param)
    ref_out = utils.to_reference(ref_param)
    utils.gems_assert_close(gems_out, ref_out, dtype)


def _assert_fused_adam_log(caplog, op, name):
    prefix = utils.gems_log_prefix(op)
    assert any(
        message.startswith(prefix) and message.endswith(f" {name}")
        for message in caplog.messages
    )


@pytest.mark.fused_adam_
def test_fused_adam_amsgrad_history(caplog):
    param = torch.ones(4, dtype=torch.float32, device=flag_gems.device)
    exp_avg, exp_avg_sq, max_exp_avg_sq = (torch.zeros_like(param) for _ in range(3))
    ref_param = torch.nn.Parameter(utils.to_reference(param.clone()))
    ref_opt = torch.optim.Adam([ref_param], amsgrad=True, foreach=False, fused=False)
    gems_op = flag_gems._fused_adam_

    for step, value in enumerate((4.0, 1.0, 1.0), start=1):
        grad = torch.full_like(param, value)
        state_step = torch.tensor(float(step), device=param.device)
        ref_param.grad = torch.full_like(ref_param, value)
        ref_opt.step()

        caplog.clear()
        with caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
            gems_op(
                [param],
                [grad],
                [exp_avg],
                [exp_avg_sq],
                [max_exp_avg_sq],
                [state_step],
                amsgrad=True,
            )
        _assert_fused_adam_log(caplog, gems_op, "FUSED_ADAM_")
        utils.gems_assert_close(param, ref_param.detach(), param.dtype, atol=1e-6)
        utils.gems_assert_close(
            max_exp_avg_sq,
            ref_opt.state[ref_param]["max_exp_avg_sq"],
            param.dtype,
            atol=1e-7,
        )


@pytest.mark.fused_adam_
@pytest.mark.parametrize("found_inf_value", [0.0, 1.0], ids=["unscale", "skip"])
def test_fused_adam_grad_scale(found_inf_value, caplog):
    param = torch.ones(4, dtype=torch.float32, device=flag_gems.device)
    grad = torch.full_like(param, 16.0)
    exp_avg, exp_avg_sq = torch.zeros_like(param), torch.zeros_like(param)
    state_step = torch.tensor(1.0, device=param.device)
    grad_scale = torch.tensor(16.0, device=param.device)
    found_inf = torch.tensor(found_inf_value, device=param.device)
    inputs = (param, grad, exp_avg, exp_avg_sq)
    ref = [utils.to_reference(t.clone()) for t in inputs]
    kwargs = dict(
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=True,
    )
    torch._fused_adam_(
        *([t] for t in ref),
        [],
        [utils.to_reference(state_step)],
        **kwargs,
        grad_scale=utils.to_reference(grad_scale),
        found_inf=utils.to_reference(found_inf),
    )

    gems_op = flag_gems._fused_adam_
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
        gems_op(
            *([t] for t in inputs),
            [],
            [state_step],
            **kwargs,
            grad_scale=grad_scale,
            found_inf=found_inf,
        )
    _assert_fused_adam_log(caplog, gems_op, "FUSED_ADAM_")
    utils.gems_assert_equal(grad, ref[1])
    for result, expected in zip(inputs, ref):
        utils.gems_assert_close(result, expected, param.dtype, atol=1e-7)


@pytest.mark.fused_adam
def test_fused_adam_preserves_inputs_with_grad_scale(caplog):
    param = torch.ones(4, dtype=torch.float32, device=flag_gems.device)
    grad = torch.full_like(param, 16.0)
    exp_avg, exp_avg_sq = torch.zeros_like(param), torch.zeros_like(param)
    inputs = (param, grad, exp_avg, exp_avg_sq)
    original = [utils.to_reference(t.clone()) for t in inputs]
    state_step = torch.tensor(1.0, device=param.device)
    grad_scale = torch.tensor(16.0, device=param.device)
    gems_op = flag_gems._fused_adam

    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
        outputs = gems_op(
            *([t] for t in inputs), [], [state_step], grad_scale=grad_scale
        )
    _assert_fused_adam_log(caplog, gems_op, "FUSED_ADAM")
    utils.gems_assert_equal(outputs[1][0], utils.to_reference(torch.ones_like(grad)))
    for tensor, before in zip(inputs, original):
        utils.gems_assert_equal(tensor, before)
