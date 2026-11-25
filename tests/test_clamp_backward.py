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


@pytest.mark.clamp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("bounds", [(-0.5, 0.5), (-1.0, 1.0), (0.0, 0.0), (0.5, -0.5)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_backward(shape, bounds, dtype):
    """Drive the gradient through the autograd of the forward ``clamp``."""
    min_value, max_value = bounds
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_grad = utils.to_reference(out_grad)

    (res_in_grad,) = torch.autograd.grad(
        flag_gems.clamp(inp, min_value, max_value), inp, out_grad
    )
    (ref_in_grad,) = torch.autograd.grad(
        torch.clamp(ref_inp, min=min_value, max=max_value), ref_inp, ref_grad
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.clamp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bound_kwarg", ["mini", "maxi"])
def test_clamp_backward_one_sided(dtype, bound_kwarg):
    """ATen's clamp backward degrades to clamp_max/clamp_min when one bound is None."""
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn(x.shape, dtype=dtype, device=x.device)
    # ``flag_gems.clamp`` follows the ATen schema's positional ``mini``/``maxi``
    # names rather than torch.clamp's ``min``/``max`` keyword aliases.
    gems_kwargs = {bound_kwarg: 0.5}
    torch_kwargs = {"min" if bound_kwarg == "mini" else "max": 0.5}

    (dx,) = torch.autograd.grad(flag_gems.clamp(x, **gems_kwargs), x, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp(ref_x, **torch_kwargs), ref_x, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.clamp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_backward_expanded_grad(dtype):
    """An expanded upstream gradient is non-contiguous and must not be read linearly."""
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn((7, 1), dtype=dtype, device=x.device).expand_as(x)

    (dx,) = torch.autograd.grad(flag_gems.clamp(x, -0.5, 0.5), x, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp(ref_x, -0.5, 0.5), ref_x, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.clamp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_backward_noncontiguous_input(dtype):
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).t().requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn(x.shape, dtype=dtype, device=x.device)

    (dx,) = torch.autograd.grad(flag_gems.clamp(x, -0.5, 0.5), x, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp(ref_x, -0.5, 0.5), ref_x, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.clamp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_backward_nan_and_boundary(dtype):
    """NaN input gets a zero gradient; an element equal to a bound keeps it."""
    inp = torch.tensor(
        [float("nan"), -0.5, 0.5, 0.6, float("inf"), float("-inf")],
        dtype=dtype,
        device=flag_gems.device,
    ).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    grad = torch.ones_like(inp)

    (dx,) = torch.autograd.grad(flag_gems.clamp(inp, -0.5, 0.5), inp, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp(ref_inp, -0.5, 0.5), ref_inp, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)
