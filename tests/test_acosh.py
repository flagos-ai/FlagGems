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


@pytest.mark.acosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.SMALL_INT_DTYPES)
def test_acosh(shape, dtype):
    # acosh domain is [1, inf), so generate input in [1, 2]. Integer inputs are
    # built on CPU (GPU uniform_ has no integer implementation) and promoted by
    # acosh to float32, matching ATen.
    if dtype in utils.SMALL_INT_DTYPES:
        inp = torch.randint(1, 3, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            1.0, 2.0
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.acosh(ref_inp)
    res_out = flag_gems.acosh(inp)

    # Integer inputs must promote to a float output dtype exactly as ATen does.
    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "acosh")
    )


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 2.0)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acosh_(ref_inp)
    res_out = flag_gems.acosh_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES + utils.BOOL_TYPES)
def test_acosh__integer_rejects(shape, dtype):
    # ATen raises for in-place acosh on integer/bool inputs; FlagGems mirrors
    # that instead of silently truncating.
    if dtype is torch.bool:
        inp = torch.zeros(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp = torch.randint(0, 3, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.acosh_(inp)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.1
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    res_inp = inp.detach().clone().requires_grad_()

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    res_grad = torch.autograd.grad(flag_gems.acosh(res_inp), res_inp, grad)[0]

    utils.gems_assert_close(res_grad, ref_grad, dtype)
    direct_grad = flag_gems.acosh_backward(grad, res_inp)
    utils.gems_assert_close(direct_grad, ref_grad, dtype)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_registered_dispatch(dtype):
    # A small 2-D tensor is enough to verify the Autograd-key routing itself.
    inp = (
        torch.rand((64, 64), dtype=dtype, device=flag_gems.device) + 1.1
    ).requires_grad_()
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["acosh"])
        res_out = torch.acosh(inp)
        assert "AcoshFunction" in type(res_out.grad_fn).__name__
        res_grad = torch.autograd.grad(res_out, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_special_values(dtype):
    values = [
        -float("inf"),
        -2.0,
        -1.0,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        float("inf"),
        float("nan"),
    ]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    grad = torch.ones_like(inp)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    res_inp = inp.detach().clone().requires_grad_()

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    res_grad = torch.autograd.grad(flag_gems.acosh(res_inp), res_inp, grad)[0]

    utils.gems_assert_equal(res_grad, ref_grad, equal_nan=True)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_noncontiguous(dtype):
    # A transposed 2-D input exercises the single kernel after normalization.
    inp = (torch.rand((32, 64), dtype=dtype, device=flag_gems.device) + 1.1).transpose(
        0, 1
    )
    assert not inp.is_contiguous()
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(
        inp.detach().clone(memory_format=torch.preserve_format)
    ).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    res_inp = inp.detach().clone(memory_format=torch.preserve_format).requires_grad_()

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    res_grad = torch.autograd.grad(flag_gems.acosh(res_inp), res_inp, grad)[0]

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_empty(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.empty_like(inp)

    res_grad = torch.autograd.grad(flag_gems.acosh(inp), inp, grad)[0]

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == dtype


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_fused_kernel_count(dtype):
    """The backward replaces ATen's four-kernel formula with one fused kernel."""
    from torch.profiler import ProfilerActivity, profile

    # This size is large enough for the profiler to record the Triton launch.
    inp = (
        torch.rand((4096,), dtype=dtype, device=flag_gems.device) + 1.1
    ).requires_grad_()
    out = flag_gems.acosh(inp)
    grad = torch.randn_like(inp)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "acosh_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
