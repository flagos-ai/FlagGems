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


@pytest.mark.rsqrt
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    res_out = flag_gems.rsqrt(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.rsqrt_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.rsqrt_(ref_inp)
    res_out = flag_gems.rsqrt_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.rsqrt_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt_backward(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    ref_out = torch.rsqrt(ref_inp)
    res_out = flag_gems.rsqrt(res_inp)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, utils.to_reference(grad))
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)
    direct_grad = flag_gems.rsqrt_backward(grad, res_out.detach())

    # The forward rsqrt approximation is cubed by the derivative, so fp16
    # differences versus the CPU reference need one extra ULP of tolerance.
    utils.gems_assert_close(res_grad, ref_grad, dtype, atol=1e-3)
    utils.gems_assert_close(direct_grad, ref_grad, dtype, atol=1e-3)


@pytest.mark.rsqrt_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt_backward_registered_dispatch(dtype):
    inp = (
        torch.rand((7, 11), dtype=dtype, device=flag_gems.device) + 1.0
    ).requires_grad_()
    grad = torch.randn((7, 11), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.rsqrt(ref_inp), ref_inp, utils.to_reference(grad)
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["rsqrt"])
        out = torch.rsqrt(inp)
        assert "RsqrtAutogradFunction" in type(out.grad_fn).__name__
        (res_grad,) = torch.autograd.grad(out, inp, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grad, ref_grad, dtype, atol=1e-3)


@pytest.mark.rsqrt_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt_backward_noncontiguous(dtype):
    inp = (torch.rand((5, 7), dtype=dtype, device=flag_gems.device) + 1.0).t()
    grad = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.rsqrt(ref_inp), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(flag_gems.rsqrt(res_inp), res_inp, grad)
    utils.gems_assert_close(res_grad, ref_grad, dtype, atol=1e-3)


@pytest.mark.rsqrt_backward
def test_rsqrt_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = (
        torch.rand((64, 96), dtype=torch.float32, device=flag_gems.device) + 1.0
    ).requires_grad_()
    grad = torch.randn((64, 96), dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.rsqrt(inp)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    kernels = [item for item in launches if "rsqrt_backward" in item[0].lower()]
    assert len(kernels) == 1 and kernels[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
