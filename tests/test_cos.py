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


@pytest.mark.cos
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    res_out = flag_gems.cos(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cos_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.cos_(ref_inp)
    res_out = flag_gems.cos_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cos_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos_backward(shape, dtype):
    """Drive the gradient through the autograd of the forward ``cos``."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_grad = utils.to_reference(out_grad)

    (res_in_grad,) = torch.autograd.grad(flag_gems.cos(inp), inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(torch.cos(ref_inp), ref_inp, ref_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.cos_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos_backward_expanded_grad(dtype):
    """An expanded upstream gradient is non-contiguous and must not be read linearly."""
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn((7, 1), dtype=dtype, device=x.device).expand_as(x)

    (dx,) = torch.autograd.grad(flag_gems.cos(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.cos(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.cos_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos_backward_noncontiguous_input(dtype):
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).t().requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn(x.shape, dtype=dtype, device=x.device)

    (dx,) = torch.autograd.grad(flag_gems.cos(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.cos(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.cos_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos_backward_broadcast_grad(dtype):
    """A broadcast upstream gradient (1, N) against (M, N) must expand correctly."""
    x = torch.randn((5, 4), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn((1, 4), dtype=dtype, device=x.device).expand(5, 4)

    (dx,) = torch.autograd.grad(flag_gems.cos(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.cos(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.cos_backward
@pytest.mark.parametrize("shape", [(0,), (4, 0), (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos_backward_empty(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    (dx,) = torch.autograd.grad(flag_gems.cos(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.cos(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)
