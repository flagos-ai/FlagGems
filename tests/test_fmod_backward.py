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


@pytest.mark.fmod_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shapes",
    [
        ((17, 33), (17, 33)),
        ((2, 3, 5), (1, 3, 1)),
        ((1, 3, 1), (2, 3, 5)),
        ((13,), ()),
        ((), (13,)),
        ((0, 3), (1, 3)),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_fmod_tensor_backward(dtype, shapes, strided):
    x_shape, y_shape = shapes

    def operand(shape, offset):
        data = torch.arange(
            2 * torch.Size(shape).numel(), device=flag_gems.device, dtype=torch.float32
        )
        data = ((data % 11) - 5.3 + offset).to(dtype)
        return (
            data.reshape(*shape, 2)[..., 0]
            if strided and shape
            else data[::2].reshape(shape)
        )

    x = operand(x_shape, 0).detach().requires_grad_(True)
    y = operand(y_shape, 0.6).detach().requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)
    ref_y = y.detach().clone().requires_grad_(True)
    grad = torch.randn(
        torch.broadcast_shapes(x_shape, y_shape), device=x.device, dtype=dtype
    )
    actual = torch.autograd.grad(flag_gems.fmod_tensor(x, y), (x, y), grad)
    expected = torch.autograd.grad(torch.fmod(ref_x, ref_y), (ref_x, ref_y), grad)
    for result, reference in zip(actual, expected):
        utils.gems_assert_close(
            result, utils.to_reference(reference), dtype, equal_nan=True
        )


@pytest.mark.fmod_backward
@pytest.mark.parametrize("requires_x,requires_y", [(True, False), (False, True)])
def test_fmod_single_input_gradient(requires_x, requires_y):
    x = torch.tensor(
        [[-4.5], [2.25]], device=flag_gems.device, requires_grad=requires_x
    )
    y = torch.tensor(
        [[1.25, -2.0, 0.8]], device=flag_gems.device, requires_grad=requires_y
    )
    ref_x = x.detach().clone().requires_grad_(requires_x)
    ref_y = y.detach().clone().requires_grad_(requires_y)
    inputs = (x, y) if requires_x and requires_y else (x,) if requires_x else (y,)
    ref_inputs = (ref_x,) if requires_x else (ref_y,)
    grad = torch.tensor([[1.0, -0.5, 2.0], [0.25, 3.0, -1.0]], device=x.device)
    actual = torch.autograd.grad(flag_gems.fmod_tensor(x, y), inputs, grad)
    expected = torch.autograd.grad(torch.fmod(ref_x, ref_y), ref_inputs, grad)
    for result, reference in zip(actual, expected):
        utils.gems_assert_close(result, utils.to_reference(reference), torch.float32)


@pytest.mark.fmod_backward
def test_fmod_backward_grad_fn():
    x = torch.tensor([-4.5, 2.25], device=flag_gems.device, requires_grad=True)
    y = torch.tensor([1.25, -2.0], device=flag_gems.device, requires_grad=True)
    grad = torch.tensor([2.0, -3.0], device=flag_gems.device)
    result = flag_gems.fmod_tensor(x, y)
    assert type(result.grad_fn).__name__ == "_FmodTensorBackward"
    actual = torch.autograd.grad(result, (x, y), grad)
    ref_x = x.detach().clone().requires_grad_()
    ref_y = y.detach().clone().requires_grad_()
    expected = torch.autograd.grad(torch.fmod(ref_x, ref_y), (ref_x, ref_y), grad)
    for value, reference in zip(actual, expected):
        utils.gems_assert_close(value, utils.to_reference(reference), torch.float32)


@pytest.mark.fmod_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (257,), (3, 17)])
@pytest.mark.parametrize("scalar", [-2.5, 3.0])
def test_fmod_scalar_backward(dtype, shape, scalar):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype).requires_grad_(True)
    ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn(shape, device=x.device, dtype=dtype)
    actual = torch.autograd.grad(flag_gems.fmod_scalar(x, scalar), x, grad)[0]
    expected = torch.autograd.grad(torch.fmod(ref, scalar), ref, grad)[0]
    utils.gems_assert_close(actual, utils.to_reference(expected), dtype)
