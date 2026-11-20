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


@pytest.mark.asin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.SMALL_INT_DTYPES)
def test_asin(shape, dtype):
    # asin promotes integer inputs to float32, matching ATen. Integers only hit
    # {-1, 0, 1} of the valid domain, so -2/2 are kept to cover the out-of-domain
    # NaN edges as well.
    if dtype in utils.SMALL_INT_DTYPES:
        res_inp = utils.randint_in_range(shape, dtype, low=-2, high=3).to(
            flag_gems.device
        )
    else:
        res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.asin(ref_inp)
    res_out = flag_gems.asin(res_inp)
    out_dtype = utils.promoted_float_dtype(dtype, "asin")
    ref_out = ref_out.to(out_dtype)

    utils.gems_assert_close(res_out, ref_out, out_dtype, equal_nan=True)


@pytest.mark.asin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.asin_(ref_inp)
    res_out = flag_gems.asin_(inp)

    ref_out = ref_out.to(res_out.dtype)
    ref_inp = ref_inp.to(inp.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(inp, ref_inp, dtype, equal_nan=True)


# Small, empty, scalar, multi-dimensional, and tiled inputs exercise the fused
# kernel without duplicating the much larger shared pointwise shape suite.
ASIN_BACKWARD_SHAPES = [(), (0,), (1,), (257,), (19, 37), (4, 5, 7)]


@pytest.mark.asin_backward
@pytest.mark.parametrize("shape", ASIN_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_backward(shape, dtype):
    x = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    ).requires_grad_()
    grad = torch.randn_like(x, memory_format=torch.contiguous_format)
    # CUDA asin backward has device-specific rsqrt rounding, so compare the
    # fused kernel with the native device formula. Forward accuracy is already
    # covered against the configured reference device above.
    ref_x = x.detach().clone().requires_grad_()

    output = flag_gems.asin(x)
    reference = torch.asin(ref_x)
    (result,) = torch.autograd.grad(output, x, grad)
    (expected,) = torch.autograd.grad(reference, ref_x, grad)

    utils.gems_assert_close(output, reference, dtype)
    utils.gems_assert_close(result, expected, dtype)
    direct = flag_gems.asin_backward(grad, x.detach())
    utils.gems_assert_close(direct, expected, dtype)


@pytest.mark.asin_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_backward_special_values(dtype):
    x = torch.tensor(
        [-1.1, -1.0, -0.999, -0.0, 0.0, 0.999, 1.0, 1.1],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    grad = torch.tensor(
        [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_x = x.detach().clone().requires_grad_()
    (result,) = torch.autograd.grad(flag_gems.asin(x), x, grad)
    (expected,) = torch.autograd.grad(torch.asin(ref_x), ref_x, grad)
    utils.gems_assert_close(result, expected, dtype, equal_nan=True)


@pytest.mark.asin_backward
def test_asin_backward_registered_dispatch():
    x = (torch.rand((31, 47), device=flag_gems.device) * 1.8 - 0.9).requires_grad_()
    grad = torch.randn_like(x)
    ref_x = x.detach().clone().requires_grad_()
    (expected,) = torch.autograd.grad(torch.asin(ref_x), ref_x, grad)

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["asin"])
        output = torch.asin(x)
        assert "AsinFunction" in type(output.grad_fn).__name__
        (result,) = torch.autograd.grad(output, x, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(result, expected, torch.float32)


@pytest.mark.asin_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_backward_strided_and_expanded(dtype):
    x = (
        torch.rand((13, 17, 19), dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    ).transpose(0, 2)
    x.requires_grad_()
    grad = torch.randn((19, 1, 13), dtype=dtype, device=x.device).expand_as(x)
    ref_x = x.detach().clone().requires_grad_()

    (result,) = torch.autograd.grad(flag_gems.asin(x), x, grad)
    (expected,) = torch.autograd.grad(torch.asin(ref_x), ref_x, grad)
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.asin_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_backward_matches_aten_rounding(dtype):
    values = torch.rand(1 << 16, dtype=dtype, device=flag_gems.device) * 1.99 - 0.995
    gem_x = values.clone().requires_grad_()
    ref_x = values.clone().requires_grad_()
    grad = torch.randn_like(values)

    (result,) = torch.autograd.grad(flag_gems.asin(gem_x), gem_x, grad)
    (expected,) = torch.autograd.grad(torch.asin(ref_x), ref_x, grad)
    torch.testing.assert_close(result, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.asin_backward
def test_asin_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    x = torch.rand((64, 512), device=flag_gems.device) * 1.8 - 0.9
    x.requires_grad_()
    output = flag_gems.asin(x)
    grad = torch.randn_like(x)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(output, x, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "asin_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
