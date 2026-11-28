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


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    negative_slope = 0.01
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=negative_slope)
    res_out = flag_gems.leaky_relu(res_inp, negative_slope=negative_slope)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    negative_slope = 0.01
    ref_out = torch.nn.functional.leaky_relu_(ref_inp, negative_slope=negative_slope)
    res_out = flag_gems.leaky_relu_(res_inp, negative_slope=negative_slope)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative_slope = 0.01

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.leaky_relu.out(ref_inp, negative_slope=negative_slope, out=ref_out)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    flag_gems.leaky_relu_out(inp, negative_slope=negative_slope, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.leaky_relu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)

    negative_slope = 0.01
    ref_in_grad = torch.ops.aten.leaky_relu_backward(
        ref_grad, ref_inp, negative_slope, False
    )
    res_in_grad = flag_gems.leaky_relu_backward(
        res_grad, res_inp, negative_slope, False
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.skipif(not utils.fp64_is_supported, reason="FP64 is unsupported")
@pytest.mark.parametrize("negative_slope", [0.2, 0.0, 1e40])
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("forward", marks=pytest.mark.leaky_relu),
        pytest.param("inplace", marks=pytest.mark.leaky_relu_),
        pytest.param("out", marks=pytest.mark.leaky_relu_out),
        pytest.param("backward", marks=pytest.mark.leaky_relu_backward),
    ],
)
def test_leaky_relu_fp64_slope(variant, negative_slope):
    inp = torch.tensor(
        [-3.123456789, -1.234567891, -0.987654321, 0.123456789], dtype=torch.float64
    )
    device_inp = inp.to(flag_gems.device)
    if variant == "backward":
        grad = torch.tensor([1.23456789, -1.987654321, 0.123456789, 2.0], dtype=inp.dtype)
        expected = torch.ops.aten.leaky_relu_backward(grad, inp, negative_slope, False)
        actual = flag_gems.leaky_relu_backward(
            grad.to(flag_gems.device), device_inp, negative_slope, False
        )
    else:
        expected = torch.nn.functional.leaky_relu(inp, negative_slope)
        if variant == "inplace":
            actual = flag_gems.leaky_relu_(device_inp, negative_slope)
        elif variant == "out":
            out = torch.empty_like(device_inp)
            actual = flag_gems.leaky_relu_out(device_inp, negative_slope, out=out)
        else:
            actual = flag_gems.leaky_relu(device_inp, negative_slope)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
