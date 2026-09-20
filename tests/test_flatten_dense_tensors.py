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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Flatten each input in logical order and concatenate into a contiguous 1-D tensor.
_SUPPORTED_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)

_NUMERIC_DTYPES = [d for d in _SUPPORTED_DTYPES if d != torch.bool]

_FLATTEN_RANGE_CASES = [
    [(8,)],
    [(3,), (5,)],
    [(), (1,)],
    [(2, 4), (3, 3), (5,)],
]

_FLATTEN_BACKWARD_CASES = [
    [(8,)],
    [(), (3,), (1, 4)],
    [(2, 3), (4,)],
    [(2, 4), (3, 3), (5,)],
    [(16, 16), (8, 8, 8)],
]

_FLATTEN_SHAPE_CASES = tu.selected_cases(
    [
        [(2, 3)],
        [(4, 5), (4, 5), (4, 5)],
        [(2, 3), (4,), (5, 6, 7)],
        [(1024,), (64, 64), (16, 16, 16)],
        [(0, 3), (2,), (1, 1, 1)],
        [(), (3,), (1, 4)],
        [(0,), (0,)],
        [(16, 7, 57, 32, 29)],
    ],
    quick=[[(2, 19, 7)], [(2, 3), (4,), (5, 6, 7)]],
)
# Append each default shape as a singleton (once) and a pair, in that order.
if not tu.QUICK_MODE:
    for shape in tu.selected_shapes():
        if [shape] not in _FLATTEN_SHAPE_CASES:
            _FLATTEN_SHAPE_CASES.append([shape])
        _FLATTEN_SHAPE_CASES.append([shape, shape])


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("tensor_shapes", _FLATTEN_SHAPE_CASES)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_flatten_dense_tensors(tensor_shapes, dtype):
    inp = [tu.make_input(dtype, shape, ["-1", "1"]) for shape in tensor_shapes]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    assert res_out.device == inp[0].device
    tu.assert_result_equal(res_out, ref_out)
    # The op is read-only: inputs must be left untouched.
    for res_t, ref_t in zip(inp, ref_inp):
        tu.assert_result_equal(res_t, ref_t)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("tensor_shapes", _FLATTEN_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NUMERIC_DTYPES)
def test_flatten_dense_tensors_value_ranges(tensor_shapes, value_range, dtype):
    inp = [tu.make_input(dtype, shape, value_range) for shape in tensor_shapes]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_flatten_dense_tensors_non_contiguous(dtype):
    base = tu.make_input(dtype, (8, 16), ["-1", "1"])
    ref_base = tu.to_reference(base)
    views = [base[:, ::2], base.t(), base.reshape(4, 32)[:, ::3]]
    ref_views = [ref_base[:, ::2], ref_base.t(), ref_base.reshape(4, 32)[:, ::3]]
    assert all(not v.is_contiguous() for v in views)

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_views)
    res_out = flag_gems.flatten_dense_tensors(views)

    assert res_out.device == views[0].device
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES))
)
def test_flatten_dense_tensors_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    other = torch.tensor([1.0, -1.0], dtype=dtype, device=flag_gems.device)
    ref_inp = [
        tu.to_reference(values),
        tu.to_reference(other),
    ]

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    res_out = flag_gems.flatten_dense_tensors([values, other])

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("tensor_shapes", _FLATTEN_BACKWARD_CASES)
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [d for d in _SUPPORTED_DTYPES if d.is_floating_point or d.is_complex]
    ),
)
def test_flatten_dense_tensors_backward(tensor_shapes, dtype):
    inp = [
        tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
        for shape in tensor_shapes
    ]
    total_numel = sum(math.prod(shape) for shape in tensor_shapes)
    grad = tu.make_input(dtype, (total_numel,), ["-1", "1"])
    ref_inp = [tu.to_reference(t) for t in inp]
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = flag_gems.flatten_dense_tensors(inp)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
    for got, exp in zip(res_in_grads, ref_in_grads):
        tu.assert_result_equal(got, exp)


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_rejects_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.flatten_dense_tensors([])
    with pytest.raises((TypeError, ValueError, RuntimeError, IndexError)):
        flag_gems.flatten_dense_tensors([])


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_rejects_non_tensor():
    a = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_a = tu.to_reference(a)
    with pytest.raises(RuntimeError):
        torch.ops.aten.flatten_dense_tensors([ref_a, 3.14])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.flatten_dense_tensors([a, 3.14])
