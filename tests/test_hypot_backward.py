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


@pytest.mark.hypot_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shapes",
    [
        ((257,), (257,)),
        ((2, 3, 5), (1, 3, 1)),
        ((2, 3), ()),
        ((), ()),
        ((0, 3), (1, 3)),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_hypot_backward(dtype, shapes, strided):
    def inputs(shape):
        raw = torch.randn(shape + (2,), dtype=dtype, device=flag_gems.device)
        return raw[..., 1] if strided else raw[..., 0].contiguous()

    a, b = (inputs(shape) for shape in shapes)
    actual_a, actual_b = (
        a.detach().requires_grad_(),
        b.detach().requires_grad_(),
    )
    ref_a, ref_b = (
        a.detach().clone().requires_grad_(),
        b.detach().clone().requires_grad_(),
    )
    result = flag_gems.hypot(actual_a, actual_b)
    expected = torch.hypot(ref_a, ref_b)
    grad = torch.randn(result.shape + (2,), dtype=dtype, device=result.device)[..., 1]
    actual_grads = torch.autograd.grad(result, (actual_a, actual_b), grad)
    expected_grads = torch.autograd.grad(expected, (ref_a, ref_b), grad)
    for actual, reference in zip(actual_grads, expected_grads):
        utils.gems_assert_close(
            actual, utils.to_reference(reference), dtype, equal_nan=True
        )


@pytest.mark.hypot_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hypot_backward_special_values(dtype):
    a = torch.tensor(
        [0, 0, 3, float("inf"), 1, float("nan"), float("inf")],
        device=flag_gems.device,
        dtype=dtype,
    )
    b = torch.tensor(
        [0, 4, 0, 2, float("inf"), float("inf"), float("inf")],
        device=flag_gems.device,
        dtype=dtype,
    )
    grad = torch.tensor([1, 2, 3, 4, 5, 6, 7], device=flag_gems.device, dtype=dtype)
    actual_a, actual_b = (
        a.detach().requires_grad_(),
        b.detach().requires_grad_(),
    )
    ref_a, ref_b = (
        a.detach().clone().requires_grad_(),
        b.detach().clone().requires_grad_(),
    )
    actual = torch.autograd.grad(
        flag_gems.hypot(actual_a, actual_b), (actual_a, actual_b), grad
    )
    expected = torch.autograd.grad(torch.hypot(ref_a, ref_b), (ref_a, ref_b), grad)
    for output, reference in zip(actual, expected):
        utils.gems_assert_close(
            output, utils.to_reference(reference), output.dtype, equal_nan=True
        )


@pytest.mark.hypot_backward
@pytest.mark.parametrize(
    "dtype_a,dtype_b",
    [(torch.float16, torch.float32), (torch.bfloat16, torch.float32)],
)
def test_hypot_backward_mixed_dtype(dtype_a, dtype_b):
    a = torch.tensor([1, 0, float("inf")], device=flag_gems.device, dtype=dtype_a)
    b = torch.tensor([2, 0, float("nan")], device=flag_gems.device, dtype=dtype_b)
    grad = torch.tensor([0.1, 4, 2], device=flag_gems.device, dtype=dtype_b)
    actual_a, actual_b = (
        a.detach().requires_grad_(),
        b.detach().requires_grad_(),
    )
    ref_a, ref_b = (
        a.detach().clone().requires_grad_(),
        b.detach().clone().requires_grad_(),
    )
    actual = torch.autograd.grad(
        flag_gems.hypot(actual_a, actual_b), (actual_a, actual_b), grad
    )
    expected = torch.autograd.grad(torch.hypot(ref_a, ref_b), (ref_a, ref_b), grad)
    for output, reference in zip(actual, expected):
        utils.gems_assert_close(
            output, utils.to_reference(reference), output.dtype, equal_nan=True
        )


@pytest.mark.hypot_backward
def test_hypot_backward_grad_fn():
    a = torch.randn(2, 3, device=flag_gems.device, requires_grad=True)
    b = torch.randn(1, 3, device=flag_gems.device, requires_grad=True)
    grad = torch.randn(2, 3, device=flag_gems.device)
    ref_a = a.detach().clone().requires_grad_()
    ref_b = b.detach().clone().requires_grad_()
    expected = torch.autograd.grad(torch.hypot(ref_a, ref_b), (ref_a, ref_b), grad)
    result = flag_gems.hypot(a, b)
    assert type(result.grad_fn).__name__ == "_HypotAutogradBackward"
    actual = torch.autograd.grad(result, (a, b), grad)
    for output, reference in zip(actual, expected):
        utils.gems_assert_close(output, utils.to_reference(reference), a.dtype)


@pytest.mark.hypot_backward
def test_hypot_backward_single_input_grad():
    a = torch.randn(2, 3, device=flag_gems.device, requires_grad=True)
    b = torch.randn(3, device=flag_gems.device)
    grad = torch.randn(2, 3, device=flag_gems.device)
    actual = torch.autograd.grad(flag_gems.hypot(a, b), a, grad)[0]
    expected = grad * a / torch.hypot(a, b)
    utils.gems_assert_close(actual, utils.to_reference(expected), actual.dtype)
