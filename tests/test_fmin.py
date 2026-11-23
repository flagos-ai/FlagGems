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


@pytest.mark.fmin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmin(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.fmin(ref_inp1, ref_inp2)
    res_out = flag_gems.fmin(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fmin_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmin_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.empty_like(ref_inp1)
    torch.fmin(ref_inp1, ref_inp2, out=ref_out)
    res_out = torch.empty_like(inp1)
    flag_gems.fmin_out(inp1, inp2, out=res_out)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fmin_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shapes",
    [
        ((8,), (8,)),
        ((2, 3, 4), (1, 3, 1)),
        ((3, 1), (2, 3, 1)),
        ((), (5,)),
        ((0, 3), (1, 3)),
        ((1, 3), (0, 3)),
    ],
)
def test_fmin_backward(dtype, shapes):
    a_shape, b_shape = shapes
    a = torch.randn(a_shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_a = utils.to_reference(a.detach()).requires_grad_()
    ref_b = utils.to_reference(b.detach()).requires_grad_()
    expected = torch.fmin(ref_a, ref_b)
    actual = flag_gems.fmin(a, b)
    utils.gems_assert_equal(actual, expected)
    grad = torch.randn_like(actual)
    expected.backward(utils.to_reference(grad))
    actual.backward(grad)
    utils.gems_assert_close(a.grad, ref_a.grad, dtype)
    utils.gems_assert_close(b.grad, ref_b.grad, dtype)


@pytest.mark.fmin_backward
def test_fmin_backward_nan_tie_and_strided():
    a = torch.tensor(
        [1.0, float("nan"), 2.0, float("nan"), 3.0], device=flag_gems.device
    )
    b = torch.tensor(
        [1.0, 2.0, float("nan"), float("nan"), 1.0], device=flag_gems.device
    )
    a = a.repeat(2).reshape(2, 5).T.detach().requires_grad_()
    b = b.reshape(5, 1).detach().requires_grad_()
    ref_a = utils.to_reference(a.detach()).requires_grad_()
    ref_b = utils.to_reference(b.detach()).requires_grad_()
    expected = torch.fmin(ref_a, ref_b)
    actual = flag_gems.fmin(a, b)
    torch.testing.assert_close(actual, expected, equal_nan=True)
    grad = torch.arange(1, 11, device=flag_gems.device, dtype=a.dtype).reshape(5, 2)
    expected.backward(utils.to_reference(grad))
    actual.backward(grad)
    torch.testing.assert_close(a.grad, ref_a.grad.to(a.device))
    torch.testing.assert_close(b.grad, ref_b.grad.to(b.device))


@pytest.mark.fmin_backward
def test_fmin_backward_single_input_and_dispatch():
    a = torch.tensor(
        [1.0, 2.0, float("nan")], device=flag_gems.device, requires_grad=True
    )
    b = torch.tensor([1.0, float("nan"), 3.0], device=flag_gems.device)
    with flag_gems.use_gems():
        result = torch.fmin(a, b)
    result.backward(torch.ones_like(result))
    torch.testing.assert_close(a.grad, torch.tensor([1.0, 1.0, 0.0], device=a.device))


@pytest.mark.fmin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmin_nan_forward(dtype):
    a = torch.tensor(
        [float("nan"), 1.0, float("nan")], dtype=dtype, device=flag_gems.device
    )
    b = torch.tensor(
        [2.0, float("nan"), float("nan")], dtype=dtype, device=flag_gems.device
    )
    torch.testing.assert_close(flag_gems.fmin(a, b), torch.fmin(a, b), equal_nan=True)
