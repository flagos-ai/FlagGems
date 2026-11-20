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


@pytest.mark.abs
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.abs(ref_inp)
    res_out = flag_gems.abs(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.abs_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.abs_(ref_inp)
    res_out = flag_gems.abs_(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.abs_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    ref_grad = torch.autograd.grad(
        torch.abs(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]
    res_grad = torch.autograd.grad(flag_gems.abs(res_inp), res_inp, grad)[0]

    utils.gems_assert_close(res_grad, ref_grad, dtype)
    res_direct = flag_gems.abs_backward(grad, res_inp)
    utils.gems_assert_close(res_direct, ref_grad, dtype)


@pytest.mark.abs_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs_backward_registered_dispatch(dtype):
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device).requires_grad_()
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    ref_grad = torch.autograd.grad(
        torch.abs(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["abs"])
        res_out = torch.abs(inp)
        assert "AbsFunction" in type(res_out.grad_fn).__name__
        res_grad = torch.autograd.grad(res_out, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.abs_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs_backward_special_values(dtype):
    values = [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    grad = torch.ones_like(inp)
    ref_inp = utils.to_reference(inp, False).requires_grad_()
    res_inp = inp.requires_grad_()

    ref_grad = torch.autograd.grad(
        torch.abs(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]
    res_grad = torch.autograd.grad(flag_gems.abs(res_inp), res_inp, grad)[0]

    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.abs_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs_backward_fused_kernel_count(dtype):
    """The backward launches one fused kernel instead of sign plus multiply."""
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((4096,), dtype=dtype, device=flag_gems.device).requires_grad_()
    out = flag_gems.abs(inp)
    grad = torch.randn_like(inp)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "abs_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
