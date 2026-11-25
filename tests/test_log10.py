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


@pytest.mark.log10
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    res_out = flag_gems.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    res_out = flag_gems.log10_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    flag_gems.log10_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, 10.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    res_out = flag_gems.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_empty(dtype):
    shapes = ((0,), (4, 0), (2, 0, 3))
    for shape in shapes:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp, True)

        ref_out = torch.log10(ref_inp)
        res_out = flag_gems.log10(inp)

        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_noncontiguous(dtype):
    inp = torch.rand((32, 64), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    res_out = flag_gems.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_log10_int_promotes_to_float(dtype):
    inp = torch.randint(1, 100, (128, 64), dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    res_out = flag_gems.log10(inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.log10_backward
@pytest.mark.parametrize("shape", [(), (0,), (257,), (19, 37)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_backward(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.5
    if x.ndim == 2:
        x = x.t()
    x.requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn_like(x, memory_format=torch.contiguous_format)

    out = flag_gems.log10(x)
    ref_out = torch.log10(ref_x)
    (dx,) = torch.autograd.grad(out, x, grad)
    (ref_dx,) = torch.autograd.grad(ref_out, ref_x, utils.to_reference(grad))

    utils.gems_assert_close(out, ref_out, dtype)
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.log10_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_backward_expanded_grad(dtype):
    x = torch.rand((7, 13), dtype=dtype, device=flag_gems.device) + 0.5
    x.requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn((7, 1), dtype=dtype, device=x.device).expand_as(x)

    (dx,) = torch.autograd.grad(flag_gems.log10(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.log10(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.log10_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_backward_special_values(dtype):
    x = torch.tensor(
        [
            0.0,
            -0.0,
            -1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            torch.finfo(dtype).max,
        ],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.ones_like(x)
    (dx,) = torch.autograd.grad(flag_gems.log10(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.log10(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype, equal_nan=True)
