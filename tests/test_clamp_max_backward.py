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


@pytest.mark.clamp_max_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max_value", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_backward(shape, max_value, dtype):
    """Drive the gradient through the autograd of the forward ``clamp_max``."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_grad = utils.to_reference(out_grad)

    (res_in_grad,) = torch.autograd.grad(
        flag_gems.clamp_max(inp, max_value), inp, out_grad
    )
    (ref_in_grad,) = torch.autograd.grad(
        torch.clamp_max(ref_inp, max_value), ref_inp, ref_grad
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.clamp_max_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_backward_expanded_grad(dtype):
    """An expanded upstream gradient is non-contiguous and must not be read linearly."""
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn((7, 1), dtype=dtype, device=x.device).expand_as(x)

    (dx,) = torch.autograd.grad(flag_gems.clamp_max(x, 0.5), x, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp_max(ref_x, 0.5), ref_x, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.clamp_max_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_backward_noncontiguous_input(dtype):
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).t().requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn(x.shape, dtype=dtype, device=x.device)

    (dx,) = torch.autograd.grad(flag_gems.clamp_max(x, 0.5), x, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp_max(ref_x, 0.5), ref_x, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.clamp_max_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_backward_nan_and_boundary(dtype):
    """NaN input gets a zero gradient; an element equal to the bound keeps it."""
    inp = torch.tensor(
        [float("nan"), 0.5, 0.6, -1.0, float("inf"), float("-inf")],
        dtype=dtype,
        device=flag_gems.device,
    ).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    grad = torch.ones_like(inp)

    (dx,) = torch.autograd.grad(flag_gems.clamp_max(inp, 0.5), inp, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp_max(ref_inp, 0.5), ref_inp, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.clamp_max_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("max_value", [0.3, 0.7, 1 / 3])
def test_clamp_max_backward_bound_rounds_to_dtype(dtype, max_value):
    """The bound is rounded to the input dtype before the predicate is applied.

    The forward converts the Scalar to ``self.dtype``, so an element that
    rounds *up* past the bound still compares as equal and keeps its
    gradient. Comparing the full-precision scalar against the truncated
    element drops it instead.
    """
    bound = torch.tensor([max_value], dtype=dtype)
    eps = torch.nextafter(bound, torch.tensor([-1e9], dtype=dtype, device=bound.device))
    inp = torch.cat([bound, eps, -bound]).to(flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    grad = torch.ones_like(inp)

    (dx,) = torch.autograd.grad(flag_gems.clamp_max(inp, max_value), inp, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp_max(ref_inp, max_value), ref_inp, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.clamp_max_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_backward_empty(dtype):
    """A zero-element input returns an empty gradient instead of launching."""
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    grad = torch.empty_like(inp)

    (dx,) = torch.autograd.grad(flag_gems.clamp_max(inp, 0.5), inp, grad)
    (ref_dx,) = torch.autograd.grad(
        torch.clamp_max(ref_inp, 0.5), ref_inp, utils.to_reference(grad)
    )
    utils.gems_assert_close(dx, ref_dx, dtype)
