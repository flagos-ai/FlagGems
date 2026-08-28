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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The two registered ATen variants are ``aten::_fused_adagrad`` (out-of-place)
# and ``aten::_fused_adagrad_`` (in-place).  pytest forbids marker names that
# start with ``_``, so the leading-underscore ATen names cannot be used as
# marks; the stripped, pytest-legal marks (``fused_adagrad`` /
# ``fused_adagrad_``) are applied to the test functions below, matching the
# ``fused_adam`` convention.

# Adagrad optimizer parameter shapes covering small/large parameter tensors
# (attention head, embedding row, MLP weight, large embedding).
FUSED_ADAGRAD_SHAPES = [
    (1024,),
    (4096,),
    (256, 256),
    (1024, 256),
    (2048, 512),
]

# Per-dtype tolerance for the Adagrad update; fp16/bf16 accumulate in fp32 but
# store back into the low-precision tensor, so allow a slightly looser bound.
TOLS = {
    torch.float32: 1e-5,
    torch.float16: 1e-3,
    torch.bfloat16: 2e-2,
}


def _reference_adagrad_step(
    param, grad, state_sum, step, *, lr, lr_decay, weight_decay, eps, maximize
):
    """Pure-PyTorch reference for one Adagrad step (matches ``aten::_fused_adagrad_``)."""
    p = param.float()
    g = grad.float()
    s = state_sum.float()
    if maximize:
        g = -g
    if weight_decay != 0:
        g = g + p * weight_decay
    s = s + g * g
    corrected_lr = lr / (1.0 + (step - 1.0) * lr_decay)
    p = p - corrected_lr * g / (torch.sqrt(s) + eps)
    return p.to(param.dtype), s.to(state_sum.dtype)


def _make_inputs(shape, dtype, device):
    """Build a single-tensor optimizer state returned as 1-element lists.

    ``aten::_fused_adagrad_`` / ``aten::_fused_adagrad`` expect lists of tensors
    (foreach semantics); we exercise the single-tensor case through a 1-element
    list and the multi-tensor case through an explicit 3-element list.
    """
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    state_sum = torch.zeros(shape, dtype=dtype, device=device)
    state_step = torch.tensor([3.0], dtype=torch.float32, device=device)
    return [param], [grad], [state_sum], [state_step]


# ---------------------------------------------------------------------------
# In-place variant: aten::_fused_adagrad_  (mutates params & state_sums)
# ---------------------------------------------------------------------------


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("shape", FUSED_ADAGRAD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fused_adagrad__basic(shape, dtype):
    """Basic in-place Adagrad step against a pure-torch reference."""
    atol = TOLS[dtype]

    torch.manual_seed(42)
    (param_ref,), (grad_ref,), (state_sum_ref,), (step_ref,) = _make_inputs(
        shape, dtype, flag_gems.device
    )
    ref_p, ref_s = _reference_adagrad_step(
        param_ref,
        grad_ref,
        state_sum_ref,
        step_ref.item(),
        lr=0.01,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
    )

    torch.manual_seed(42)
    params, grads, state_sums, steps = _make_inputs(shape, dtype, flag_gems.device)
    flag_gems._fused_adagrad_(
        params,
        grads,
        state_sums,
        steps,
        lr=0.01,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
    )

    utils.gems_assert_close(
        utils.to_reference(params[0]), utils.to_reference(ref_p), dtype, atol=atol
    )
    utils.gems_assert_close(
        utils.to_reference(state_sums[0]), utils.to_reference(ref_s), dtype, atol=atol
    )


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
# _fused_adagrad_ options/multi-tensor cases use float32 only for optimizer
# state precision (fp16/bf16 path already covered by the basic case).
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "lr_decay,weight_decay,maximize",
    [
        (0.1, 0.0, False),
        (0.0, 0.01, False),
        (0.1, 0.01, True),
        (0.5, 0.001, False),
    ],
)
def test_fused_adagrad__options(shape, dtype, lr_decay, weight_decay, maximize):
    """In-place Adagrad with various lr_decay / weight_decay / maximize combos."""
    atol = TOLS[dtype]

    torch.manual_seed(42)
    (param_ref,), (grad_ref,), (state_sum_ref,), (step_ref,) = _make_inputs(
        shape, dtype, flag_gems.device
    )
    ref_p, ref_s = _reference_adagrad_step(
        param_ref,
        grad_ref,
        state_sum_ref,
        step_ref.item(),
        lr=0.05,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=1e-10,
        maximize=maximize,
    )

    torch.manual_seed(42)
    params, grads, state_sums, steps = _make_inputs(shape, dtype, flag_gems.device)
    flag_gems._fused_adagrad_(
        params,
        grads,
        state_sums,
        steps,
        lr=0.05,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=1e-10,
        maximize=maximize,
    )

    utils.gems_assert_close(
        utils.to_reference(params[0]), utils.to_reference(ref_p), dtype, atol=atol
    )
    utils.gems_assert_close(
        utils.to_reference(state_sums[0]), utils.to_reference(ref_s), dtype, atol=atol
    )


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
# _fused_adagrad_ multi-tensor foreach case uses float32 only for optimizer
# state precision (fp16/bf16 path already covered by the basic case).
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_adagrad__multi_tensor(shape, dtype):
    """Adagrad over a list of multiple parameter tensors (foreach semantics)."""
    atol = TOLS[dtype]

    def _build():
        return (
            [
                torch.randn(shape, dtype=dtype, device=flag_gems.device)
                for _ in range(3)
            ],
            [
                torch.randn(shape, dtype=dtype, device=flag_gems.device)
                for _ in range(3)
            ],
            [
                torch.zeros(shape, dtype=dtype, device=flag_gems.device)
                for _ in range(3)
            ],
            [
                torch.tensor([2.0], dtype=torch.float32, device=flag_gems.device)
                for _ in range(3)
            ],
        )

    torch.manual_seed(42)
    params_ref, grads_ref, state_sums_ref, steps_ref = _build()
    ref_ps, ref_ss = [], []
    for p, g, s, st in zip(params_ref, grads_ref, state_sums_ref, steps_ref):
        rp, rs = _reference_adagrad_step(
            p,
            g,
            s,
            st.item(),
            lr=0.01,
            lr_decay=0.0,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
        )
        ref_ps.append(rp)
        ref_ss.append(rs)

    torch.manual_seed(42)
    params, grads, state_sums, steps = _build()
    flag_gems._fused_adagrad_(
        params,
        grads,
        state_sums,
        steps,
        lr=0.01,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
    )

    for p, rp, s, rs in zip(params, ref_ps, state_sums, ref_ss):
        utils.gems_assert_close(
            utils.to_reference(p), utils.to_reference(rp), dtype, atol=atol
        )
        utils.gems_assert_close(
            utils.to_reference(s), utils.to_reference(rs), dtype, atol=atol
        )
