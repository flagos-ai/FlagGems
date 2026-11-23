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

P_LIST = [1, 2, 3, 5, 8, 12]
LOW_PRECISION_DTYPES = [
    dtype for dtype in (torch.float16, torch.bfloat16) if dtype in utils.FLOAT_DTYPES
]
FULL_PRECISION_DTYPES = [
    dtype for dtype in utils.FLOAT_DTYPES if dtype not in LOW_PRECISION_DTYPES
]


def _numel(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


LOW_PRECISION_SHAPES = [
    shape for shape in utils.POINTWISE_SHAPES if _numel(shape) <= 1024
]
MVLGAMMA_CASES = [
    (shape, dtype)
    for shape in utils.POINTWISE_SHAPES
    for dtype in FULL_PRECISION_DTYPES
] + [(shape, dtype) for shape in LOW_PRECISION_SHAPES for dtype in LOW_PRECISION_DTYPES]


@pytest.mark.mvlgamma
@pytest.mark.parametrize("shape,dtype", MVLGAMMA_CASES)
@pytest.mark.parametrize("p", P_LIST)
def test_mvlgamma(shape, dtype, p):
    torch.manual_seed(42)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    inp = inp + (p - 1) / 2 + 1.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.mvlgamma(ref_inp, p)
    res_out = flag_gems.mvlgamma(inp, p)

    # Use relaxed tolerance for float16 due to lgamma precision limitations
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


# Small, empty, scalar, multidimensional, and tiled inputs cover the fused
# backward kernel without repeating the full forward pointwise shape suite.
MVLGAMMA_BACKWARD_SHAPES = [(), (0,), (1,), (257,), (19, 37), (4, 5, 7)]


@pytest.mark.mvlgamma_backward
@pytest.mark.parametrize("shape", MVLGAMMA_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2, 5, 12])
def test_mvlgamma_backward(shape, dtype, p):
    torch.manual_seed(42)
    x = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + (p - 1) / 2 + 0.75
    ).requires_grad_()
    grad = torch.randn_like(x, memory_format=torch.contiguous_format)
    ref_x = utils.to_reference(x, True)
    ref_grad = utils.to_reference(grad, True)

    (result,) = torch.autograd.grad(flag_gems.mvlgamma(x, p), x, grad)
    (expected,) = torch.autograd.grad(torch.mvlgamma(ref_x, p), ref_x, ref_grad)

    # CUDA digamma rounds each low-precision term before the reduction.
    atol = 2e-2 if dtype in LOW_PRECISION_DTYPES else 1e-4
    utils.gems_assert_close(result, expected, dtype, atol=atol)
    direct = flag_gems.mvlgamma_backward(grad, x.detach(), p)
    utils.gems_assert_close(direct, expected, dtype, atol=atol)


@pytest.mark.mvlgamma_backward
def test_mvlgamma_backward_registered_dispatch():
    x = (torch.rand((31, 47), device=flag_gems.device) + 3.0).requires_grad_()
    grad = torch.randn_like(x)
    ref_x = utils.to_reference(x, True)
    ref_grad = utils.to_reference(grad, True)
    (expected,) = torch.autograd.grad(torch.mvlgamma(ref_x, 5), ref_x, ref_grad)

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["mvlgamma"])
        output = torch.mvlgamma(x, 5)
        assert "MvlgammaFunction" in type(output.grad_fn).__name__
        (result,) = torch.autograd.grad(output, x, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(result, expected, torch.float32)


@pytest.mark.mvlgamma_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mvlgamma_backward_strided_and_expanded(dtype):
    torch.manual_seed(42)
    x = (
        torch.rand((13, 17, 19), dtype=dtype, device=flag_gems.device) + 3.0
    ).transpose(0, 2)
    x.requires_grad_()
    grad = torch.randn((19, 1, 13), dtype=dtype, device=x.device).expand_as(x)
    ref_x = utils.to_reference(x, True)
    ref_grad = utils.to_reference(grad, True)

    (result,) = torch.autograd.grad(flag_gems.mvlgamma(x, 5), x, grad)
    (expected,) = torch.autograd.grad(torch.mvlgamma(ref_x, 5), ref_x, ref_grad)
    atol = 2e-2 if dtype in LOW_PRECISION_DTYPES else 1e-4
    utils.gems_assert_close(result, expected, dtype, atol=atol)


@pytest.mark.mvlgamma_backward
def test_mvlgamma_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    x = (torch.rand((64, 512), device=flag_gems.device) + 3.0).requires_grad_()
    output = flag_gems.mvlgamma(x, 5)
    grad = torch.randn_like(x)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(output, x, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [
        (name, count) for name, count in launches if "mvlgamma_backward" in name
    ]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
