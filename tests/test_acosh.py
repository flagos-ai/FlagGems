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


@pytest.mark.acosh_backward_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.1
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    actual = inp.detach().clone().requires_grad_()
    target = actual.clone()
    target_ptr = target.data_ptr()

    # Native acosh_ explicitly has no derivative, so the mathematically
    # equivalent out-of-place acosh supplies the reference gradient.
    ref_out = torch.acosh(ref_inp)
    gems_out = flag_gems.acosh_(target)
    assert gems_out.data_ptr() == target_ptr
    ref_grad = torch.autograd.grad(ref_out, ref_inp, ref_grad_output)[0]
    gems_grad = torch.autograd.grad(gems_out, actual, grad)[0]

    utils.gems_assert_close(gems_grad, ref_grad, dtype)


@pytest.mark.acosh_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_registered_dispatch(dtype):
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
        flag_gems.only_enable(lib=lib, include=["acosh_"])
        target = inp.clone()
        res_out = target.acosh_()
        assert "AcoshInplaceFunction" in type(res_out.grad_fn).__name__
        res_grad = torch.autograd.grad(res_out, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.acosh_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_special_values(dtype):
    inp = torch.tensor(
        [1.0, 1.1, 1.5, 2.0, float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad = torch.tensor([1.0, -2.0, 0.5, -0.25, 1.0], dtype=dtype, device=inp.device)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    actual = inp.detach().clone().requires_grad_()

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    gems_grad = torch.autograd.grad(flag_gems.acosh_(actual.clone()), actual, grad)[0]

    utils.gems_assert_close(gems_grad, ref_grad, dtype)


@pytest.mark.acosh_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_noncontiguous(dtype):
    leaf = (
        torch.rand((32, 64), dtype=dtype, device=flag_gems.device) + 1.1
    ).requires_grad_()
    base = leaf.clone()
    target = base.transpose(0, 1)
    assert not target.is_contiguous()
    grad = torch.randn_like(target)
    ref_inp = utils.to_reference(target.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    gems_grad = torch.autograd.grad(flag_gems.acosh_(target), leaf, grad)[0].transpose(
        0, 1
    )

    utils.gems_assert_close(gems_grad, ref_grad, dtype)


@pytest.mark.acosh_backward_
def test_acosh_inplace_version():
    x = (torch.rand((32, 64), device=flag_gems.device) + 1.1).requires_grad_()
    saved = (x * x).sum()

    with torch.no_grad():
        old_version = x._version
        flag_gems.acosh_(x)

    assert x._version > old_version
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved.backward()


@pytest.mark.acosh_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_empty(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = flag_gems.acosh_(inp.clone())
    grad = torch.empty_like(out)

    res_grad = torch.autograd.grad(out, inp, grad)[0]

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == dtype


@pytest.mark.acosh_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_inplace_fused_forward_kernel_count(dtype):
    from torch.profiler import ProfilerActivity, profile

    inp = (
        torch.rand((4096,), dtype=dtype, device=flag_gems.device) + 1.1
    ).requires_grad_()
    target = inp.clone()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.acosh_(target)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    assert sum(count for _, count in launches) == 1, launches


@pytest.mark.acosh_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_fused_kernel_count(dtype):
    from torch.profiler import ProfilerActivity, profile

    inp = (
        torch.rand((4096,), dtype=dtype, device=flag_gems.device) + 1.1
    ).requires_grad_()
    out = flag_gems.acosh_(inp.clone())
    grad = torch.randn_like(out)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [
        (name, count) for name, count in launches if "acosh_inplace_backward" in name
    ]
    assert len(backward) == 1 and backward[0][1] == 1, launches
