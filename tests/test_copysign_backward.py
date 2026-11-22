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


@pytest.mark.copysign_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_backward(shape, dtype):
    """Drive the gradient through the autograd of the forward ``copysign``."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other)
    ref_grad = utils.to_reference(out_grad)

    (res_in_grad,) = torch.autograd.grad(flag_gems.copysign(inp, other), inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(
        torch.copysign(ref_inp, ref_other), ref_inp, ref_grad
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.copysign_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_backward_signed_zeros(shape, dtype):
    # `other` must contribute its sign *bit*, so -0.0 counts as negative.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    other = torch.tensor(-0.0, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_grad = utils.to_reference(out_grad)

    (res_in_grad,) = torch.autograd.grad(flag_gems.copysign(inp, other), inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(
        torch.copysign(ref_inp, -0.0), ref_inp, ref_grad
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.copysign_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("other_scalar", [-2.0, 0.5])
def test_copysign_backward_scalar(shape, dtype, other_scalar):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_grad = utils.to_reference(out_grad)

    (res_in_grad,) = torch.autograd.grad(
        flag_gems.copysign(inp, other_scalar), inp, out_grad
    )
    (ref_in_grad,) = torch.autograd.grad(
        torch.copysign(ref_inp, other_scalar), ref_inp, ref_grad
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.copysign_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_backward_expanded_grad(dtype):
    """An expanded upstream gradient is non-contiguous and must not be read linearly."""
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    other = torch.randn((7, 13), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((7, 1), dtype=dtype, device=x.device).expand_as(x)

    (dx,) = torch.autograd.grad(flag_gems.copysign(x, other), x, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.copysign(ref_x, utils.to_reference(other)),
        ref_x,
        utils.to_reference(grad),
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.copysign_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_backward_grad_other_is_zero(dtype):
    """copysign's VJP w.r.t. ``other`` is identically zero, matching ATen."""
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    other = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    grad = torch.randn(x.shape, dtype=dtype, device=x.device)

    dx, dother = torch.autograd.grad(flag_gems.copysign(x, other), (x, other), grad)

    ref_x = utils.to_reference(x.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    ref_dx, ref_dother = torch.autograd.grad(
        torch.copysign(ref_x, ref_other), (ref_x, ref_other), utils.to_reference(grad)
    )

    utils.gems_assert_close(dx, ref_dx, dtype)
    assert torch.count_nonzero(dother) == 0
    assert torch.count_nonzero(ref_dother) == 0
