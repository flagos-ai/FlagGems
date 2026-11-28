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


@pytest.mark.relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_relu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.relu(ref_inp)
    res_out = flag_gems.relu(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.relu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_relu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.relu_(ref_inp)
    res_out = flag_gems.relu_(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.relu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_relu_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.relu(ref_inp), ref_inp, utils.to_reference(grad)
    )

    inp.requires_grad_()
    out = flag_gems.relu(inp)
    assert "ReluFunction" in type(out.grad_fn).__name__
    (result,) = torch.autograd.grad(out, inp, grad)
    utils.gems_assert_close(result, ref_grad, dtype)


@pytest.mark.relu_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_relu_backward_boundary(dtype):
    inp = torch.tensor([-2.0, -0.0, 0.0, 1.5], dtype=dtype, device=flag_gems.device)
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=flag_gems.device)
    result = flag_gems.relu(inp)
    grad_input = flag_gems.relu_backward(grad, result)
    expected = torch.tensor([0.0, 0.0, 0.0, 4.0], dtype=dtype, device=flag_gems.device)
    utils.gems_assert_equal(grad_input, utils.to_reference(expected))


@pytest.mark.relu_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_relu_backward_noncontiguous(dtype):
    inp = torch.randn((47, 31), dtype=dtype, device=flag_gems.device).t()
    grad = torch.randn((47, 31), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.relu(ref_inp), ref_inp, utils.to_reference(grad)
    )

    inp.requires_grad_()
    (result,) = torch.autograd.grad(flag_gems.relu(inp), inp, grad)
    utils.gems_assert_close(result, ref_grad, dtype)


@pytest.mark.relu_backward
def test_relu_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn(
        (64, 129), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp)
    out = flag_gems.relu(inp)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [item for item in launches if "relu_backward" in item[0].lower()]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
