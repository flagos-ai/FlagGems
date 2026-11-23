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


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    res_out = flag_gems.reciprocal(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reciprocal_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reciprocal_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.reciprocal_(ref_inp)
    res_out = flag_gems.reciprocal_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


def _reciprocal_input(shape, dtype):
    magnitude = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.5
    sign = torch.where(torch.rand(shape, device=flag_gems.device) < 0.5, -1.0, 1.0).to(
        dtype
    )
    return magnitude * sign


@pytest.mark.reciprocal_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reciprocal_backward(shape, dtype):
    inp = _reciprocal_input(shape, dtype)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # ReciprocalBackward uses the rounded forward result, so the reference must
    # keep the input dtype instead of recomputing the forward in float64.
    ref_inp = utils.to_reference(inp).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.reciprocal(ref_inp), ref_inp, utils.to_reference(grad)
    )

    inp.requires_grad_()
    out = flag_gems.reciprocal(inp)
    assert "ReciprocalFunction" in type(out.grad_fn).__name__
    (result,) = torch.autograd.grad(out, inp, grad)
    utils.gems_assert_close(result, ref_grad, dtype)


@pytest.mark.reciprocal_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reciprocal_backward_functional(dtype):
    inp = _reciprocal_input((31, 47), dtype)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.reciprocal(ref_inp), ref_inp, utils.to_reference(grad)
    )

    result = flag_gems.reciprocal(inp)
    grad_input = flag_gems.reciprocal_backward(grad, result)
    utils.gems_assert_close(grad_input, ref_grad, dtype)


@pytest.mark.reciprocal_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reciprocal_backward_noncontiguous(dtype):
    base = _reciprocal_input((47, 31), dtype)
    inp = base.t()
    grad = torch.randn((47, 31), dtype=dtype, device=flag_gems.device).t()

    ref_inp = utils.to_reference(inp).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.reciprocal(ref_inp), ref_inp, utils.to_reference(grad)
    )

    result = flag_gems.reciprocal(base).t()
    grad_input = flag_gems.reciprocal_backward(grad, result)
    utils.gems_assert_close(grad_input, ref_grad, dtype)


@pytest.mark.reciprocal_backward
def test_reciprocal_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = _reciprocal_input((64, 129), torch.float32).requires_grad_()
    grad = torch.randn_like(inp)
    out = flag_gems.reciprocal(inp)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [item for item in launches if "reciprocal_backward_kernel" in item[0]]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
