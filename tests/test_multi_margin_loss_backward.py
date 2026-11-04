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

import subprocess

import sys

import pytest

import torch

import flag_gems

from . import accuracy_utils as utils

from . import conftest as cfg

_SHAPES = [(3, 7)] if cfg.QUICK_MODE else [(), (7,), (3, 7), (5, 65)]

def _forward_for_backward(input, target, p, margin, weight, reduction, *, out=None):
    if flag_gems.vendor_name != "kunlunxin":
        if out is None:
            return flag_gems.ops.multi_margin_loss(input, target, p, margin, weight, reduction)
        return flag_gems.ops.multi_margin_loss_out(input, target, p, margin, weight, reduction, out=out)
    # This forward only prepares/checks the backward case. KU's native BF16
    # forward is absent; run the same native formula with float32 opmath.
    forward_input = input.float() if input.dtype == torch.bfloat16 else input
    forward_weight = weight.float() if weight is not None and weight.dtype == torch.bfloat16 else weight
    result = torch.ops.aten.multi_margin_loss(forward_input, target, p, margin, forward_weight, reduction).to(input.dtype)
    if out is None:
        return result
    out.resize_(result.shape)
    out.copy_(result)
    return out


def _make_case(shape, dtype, use_weight, *, noncontiguous=False):
    if noncontiguous:
        N, C = shape
        input = torch.randn((N, C * 2), dtype=dtype, device=flag_gems.device)[:, ::2]
        target = torch.randint(0, C, (N * 2,), dtype=torch.int64, device=flag_gems.device)[::2]
        weight = torch.randn((C * 2,), dtype=dtype, device=flag_gems.device)[::2]
        return (input, target, weight)
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    C = 1 if input.dim() == 0 else input.shape[-1]
    target_shape = (input.shape[0],) if input.dim() == 2 else ()
    target = torch.randint(0, C, target_shape, dtype=torch.int64, device=flag_gems.device)
    weight = torch.randn((C,), dtype=dtype, device=flag_gems.device) if use_weight else None
    return (input, target, weight)

def _reference(input, target, p, margin, weight, reduction):
    ref_input = utils.to_reference(input, True)
    ref_target = utils.to_reference(target)
    ref_weight = utils.to_reference(weight, True)
    return torch.ops.aten.multi_margin_loss(ref_input, ref_target, p, margin, ref_weight, reduction)

@pytest.mark.multi_margin_loss
@pytest.mark.multi_margin_loss_backward
@pytest.mark.parametrize('shape', [(7,), (3, 7), (4, 65)])
@pytest.mark.parametrize('dtype', utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('reduction', [0, 1, 2])
@pytest.mark.parametrize('use_weight', [False, True])
def test_multi_margin_loss_backward(shape, dtype, p, reduction, use_weight):
    if flag_gems.vendor_name == "enflame" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("Enflame native multi_margin_loss backward does not support float16 or bfloat16")
    if flag_gems.vendor_name == "nvidia" and shape == (4, 65) and not utils.TO_CPU:
        pytest.skip("CUDA native float64 backward is incorrect for 65 classes; CPU reference remains covered")
    input, target, weight = _make_case(shape, dtype, use_weight)
    margin = 0.7
    output_shape = (shape[0],) if len(shape) == 2 and reduction == 0 else ()
    grad_output = torch.randn(output_shape, dtype=dtype, device=flag_gems.device)
    ref = torch.ops.aten.multi_margin_loss_backward(utils.to_reference(grad_output, True), utils.to_reference(input, True), utils.to_reference(target), p, margin, utils.to_reference(weight, True), reduction)
    result = flag_gems.multi_margin_loss_backward(grad_output, input, target, p, margin, weight, reduction)
    N = input.shape[0] if input.dim() == 2 else 1
    C = input.shape[-1]
    reduce_dim = N * C if reduction == 1 else C
    utils.gems_assert_close(result, ref, dtype, reduce_dim=reduce_dim, equal_nan=True)

@pytest.mark.multi_margin_loss
@pytest.mark.multi_margin_loss_backward
def test_multi_margin_loss_forward_backward_consistency():
    input, target, weight = _make_case((3, 7), torch.float32, True)
    ref_input = utils.to_reference(input.detach(), True).requires_grad_(True)
    ref_target = utils.to_reference(target)
    ref_weight = utils.to_reference(weight, True)
    ref_output = torch.nn.functional.multi_margin_loss(ref_input, ref_target, p=2, margin=0.7, weight=ref_weight, reduction='mean')
    ref_grad = torch.autograd.grad(ref_output, ref_input)[0]
    output = _forward_for_backward(input, target, 2, 0.7, weight, 1)
    result_grad = flag_gems.multi_margin_loss_backward(torch.ones_like(output), input, target, 2, 0.7, weight, 1)
    utils.gems_assert_close(result_grad, ref_grad, torch.float32, reduce_dim=21)

@pytest.mark.multi_margin_loss
@pytest.mark.multi_margin_loss_out
@pytest.mark.multi_margin_loss_backward_out
@pytest.mark.parametrize('dtype', utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('reduction', [0, 1, 2])
@pytest.mark.parametrize('use_weight', [False, True])
def test_multi_margin_loss_out_variants(dtype, p, reduction, use_weight):
    if flag_gems.vendor_name == "enflame" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("Enflame native multi_margin_loss backward does not support float16 or bfloat16")
    input, target, weight = _make_case((3, 7), dtype, use_weight)
    out = torch.empty((2,), dtype=input.dtype, device=input.device)
    output_shape = (3,) if reduction == 0 else ()
    grad_output = torch.randn(output_shape, dtype=input.dtype, device=input.device)
    grad_input = torch.empty((1,), dtype=input.dtype, device=input.device)
    result = _forward_for_backward(input, target, p, 0.7, weight, reduction, out=out)
    grad_result = flag_gems.multi_margin_loss_backward_out(grad_output, input, target, p, 0.7, weight, reduction, grad_input=grad_input)
    ref = _reference(input, target, p, 0.7, weight, reduction)
    ref_grad = torch.ops.aten.multi_margin_loss_backward(utils.to_reference(grad_output, True), utils.to_reference(input, True), utils.to_reference(target), p, 0.7, utils.to_reference(weight, True), reduction)
    assert result is out
    assert grad_result is grad_input
    assert out.shape == torch.Size(output_shape)
    assert grad_input.shape == input.shape
    output_reduce_dim = 7 if reduction == 0 else 21
    grad_reduce_dim = 21 if reduction == 1 else 7
    utils.gems_assert_close(out, ref, dtype, reduce_dim=output_reduce_dim)
    utils.gems_assert_close(grad_input, ref_grad, dtype, reduce_dim=grad_reduce_dim)
