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


@pytest.mark.gt_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("overload", ["tensor", "scalar"])
@pytest.mark.parametrize("shape", [(0,), (7, 9), (3, 4, 5)])
@pytest.mark.parametrize("strided", [False, True])
def test_gt_inplace_backward(dtype, overload, shape, strided):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if strided and len(shape) > 1:
        inp = inp.transpose(0, -1)
    other = torch.randn(inp.shape[-1:], dtype=dtype, device=flag_gems.device)
    gradient = torch.randn_like(inp)

    def run(reference):
        base = inp.detach().clone().requires_grad_()
        receiver = base * 2
        operand = (
            other.detach().clone().requires_grad_() if overload == "tensor" else 0.5
        )
        original_version = receiver._version
        if reference:
            out = receiver.gt_(operand)
        elif overload == "tensor":
            out = flag_gems.gt_tensor_(receiver, operand)
        else:
            out = flag_gems.gt_scalar_(receiver, operand)
        assert out is receiver
        assert receiver._version == original_version + 1
        targets = (base, operand) if overload == "tensor" else (base,)
        grads = torch.autograd.grad(out, targets, gradient)
        return out.detach(), grads

    ref_out, ref_grads = run(True)
    out, grads = run(False)
    utils.gems_assert_equal(out, utils.to_reference(ref_out))
    for grad, ref_grad in zip(grads, ref_grads):
        utils.gems_assert_equal(grad, utils.to_reference(ref_grad))


@pytest.mark.gt_backward_
@pytest.mark.parametrize("overload", ["tensor", "scalar"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_gt_inplace_version_invalidates_saved_tensor(overload, requires_grad):
    base = torch.randn((3, 4), device=flag_gems.device, requires_grad=True)
    receiver = base * 2 if requires_grad else base.detach().clone()
    multiplier = torch.randn_like(receiver, requires_grad=True)
    saved = (receiver * multiplier).sum()
    old_version = receiver._version
    with torch.no_grad():
        if overload == "tensor":
            out = flag_gems.gt_tensor_(
                receiver, torch.zeros((4,), device=flag_gems.device)
            )
        else:
            out = flag_gems.gt_scalar_(receiver, 0)
    assert out is receiver
    assert receiver._version == old_version + 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved.backward()


@pytest.mark.gt_backward_
@pytest.mark.parametrize("overload", ["tensor", "scalar"])
@pytest.mark.parametrize("view", [False, True])
def test_gt_inplace_rejects_grad_leaf(overload, view):
    receiver = torch.randn((3, 4), device=flag_gems.device, requires_grad=True)
    if view:
        receiver = receiver.T
    version = receiver._version
    values = receiver.detach().clone()
    operand = torch.zeros_like(receiver) if overload == "tensor" else 0
    op = flag_gems.gt_tensor_ if overload == "tensor" else flag_gems.gt_scalar_
    with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
        op(receiver, operand)
    assert receiver._version == version
    utils.gems_assert_equal(receiver, utils.to_reference(values))


@pytest.mark.gt_backward_
def test_gt_inplace_tensor_other_only_grad_broadcast():
    receiver = torch.randn((3, 4), device=flag_gems.device)
    operand = torch.randn((4,), device=flag_gems.device, requires_grad=True)
    ref_receiver = receiver.detach().clone()
    ref_operand = operand.detach().clone().requires_grad_()
    ref = ref_receiver.gt_(ref_operand)
    out = flag_gems.gt_tensor_(receiver, operand)
    utils.gems_assert_equal(out, utils.to_reference(ref))
    grad = torch.randn_like(out)
    (ref_grad,) = torch.autograd.grad(ref, ref_operand, grad)
    (actual_grad,) = torch.autograd.grad(out, operand, grad)
    utils.gems_assert_equal(actual_grad, utils.to_reference(ref_grad))


@pytest.mark.gt_backward_
def test_gt_inplace_version_of_non_grad_receiver():
    receiver = torch.randn((3, 4), device=flag_gems.device)
    version = receiver._version
    assert flag_gems.gt_scalar_(receiver, 0) is receiver
    assert receiver._version == version + 1
