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
from .accuracy_utils import gems_assert_close, to_reference

FLOAT_DTYPES = utils.FLOAT_DTYPES
POINTWISE_SHAPES = utils.POINTWISE_SHAPES


@pytest.mark.rad2deg
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rad2deg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.rad2deg(ref_inp)
    res_out = flag_gems.rad2deg(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rad2deg_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rad2deg_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    ref_out = ref_inp.rad2deg_()
    res_out = flag_gems.rad2deg_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rad2deg_backward
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rad2deg_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.rad2deg(ref_inp), ref_inp, to_reference(grad)
    )

    inp.requires_grad_()
    out = flag_gems.rad2deg(inp)
    assert "Rad2DegFunction" in type(out.grad_fn).__name__
    (result,) = torch.autograd.grad(out, inp, grad)

    gems_assert_close(result, ref_grad, dtype)


@pytest.mark.rad2deg_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rad2deg_backward_functional(dtype):
    grad = torch.randn((31, 47), dtype=dtype, device=flag_gems.device)
    inp = torch.randn_like(grad, requires_grad=True)
    ref_inp = to_reference(inp, True).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.rad2deg(ref_inp), ref_inp, to_reference(grad)
    )

    result = flag_gems.rad2deg_backward(grad, inp)
    gems_assert_close(result, ref_grad, dtype)


@pytest.mark.rad2deg_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rad2deg_backward_noncontiguous(dtype):
    inp = torch.randn((47, 31), dtype=dtype, device=flag_gems.device).t()
    grad = torch.randn((47, 31), dtype=dtype, device=flag_gems.device).t()

    ref_inp = to_reference(inp, True).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.rad2deg(ref_inp), ref_inp, to_reference(grad)
    )

    inp.requires_grad_()
    (result,) = torch.autograd.grad(flag_gems.rad2deg(inp), inp, grad)
    gems_assert_close(result, ref_grad, dtype)


@pytest.mark.rad2deg_backward
def test_rad2deg_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn(
        (64, 129), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp)
    out = flag_gems.rad2deg(inp)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [item for item in launches if "rad2deg_backward_kernel" in item[0]]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
