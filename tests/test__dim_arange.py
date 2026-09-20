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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_dim_arange",
    MarkDecorator(Mark("_dim_arange", (), {}, _ispytest=True), _ispytest=True),
)

# Build int64 indices from one logical dimension; input values are ignored.
# Scalars have no valid dimension; (1,) already appears in the default shapes.
_DIM_ARANGE_CASES = [
    (shape, dim)
    for shape in dict.fromkeys(tu.selected_shapes() + [(1,), (5, 3), (2, 3, 4)])
    if shape
    for dim in range(-len(shape), len(shape))
]

_DIM_ARANGE_INPUT_DTYPES = (
    tu.REQUIRED_DTYPES + tu.selected_cases([torch.int16]) + [torch.bool]
)

# (view_fn, logical_shape, dim) for a (4, 8, 6) base.
_VIEW_CASES = [
    (lambda b: b.transpose(0, 1), (8, 4, 6), 0),
    (lambda b: b.transpose(0, 1), (8, 4, 6), 1),
    (lambda b: b[0:3, 2:7, 1], (3, 5), 1),
    (lambda b: b.narrow(1, 1, 5), (4, 5, 6), 1),
]


def _assert_arange_result(res_out, ref_out, inp):
    assert res_out.device == inp.device
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._dim_arange
@pytest.mark.parametrize("shape, dim", _DIM_ARANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange_value_ranges(shape, dim, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, dim)
    res_out = flag_gems._dim_arange(inp, dim)

    _assert_arange_result(res_out, ref_out, inp)


@pytest.mark._dim_arange
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange_non_contiguous(view_case, value_range, dtype):
    view_fn, expected_shape, dim = view_case
    base = tu.make_input(dtype, (4, 8, 6), value_range)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    assert tuple(inp.shape) == expected_shape
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, dim)
    res_out = flag_gems._dim_arange(inp, dim)

    _assert_arange_result(res_out, ref_out, inp)


@pytest.mark._dim_arange
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_DIM_ARANGE_INPUT_DTYPES)),
)
def test__dim_arange_nan_inf(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).expand(4, 8, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, 1)
    res_out = flag_gems._dim_arange(inp, 1)

    _assert_arange_result(res_out, ref_out, inp)


@pytest.mark._dim_arange
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__dim_arange_no_autograd(dtype):
    inp = tu.make_input(dtype, (3, 5), ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, 1)
    res_out = flag_gems._dim_arange(inp, 1)

    _assert_arange_result(res_out, ref_out, inp)
    assert res_out.grad_fn is None
    assert not res_out.requires_grad
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._dim_arange
def test__dim_arange_rejects_out_of_range_dim():
    inp = tu.make_input(torch.float32, (3, 5), ["-1", "1"])
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, 2)
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, -3)
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems._dim_arange(inp, 2)
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems._dim_arange(inp, -3)


@pytest.mark._dim_arange
def test__dim_arange_rejects_zero_dim_like():
    inp = tu.make_input(torch.float32, (), ["-1", "1"])
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, 0)
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems._dim_arange(inp, 0)


@pytest.mark._dim_arange
def test__dim_arange_rejects_non_integer_dim():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._dim_arange(inp, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._dim_arange(inp, 1.5)
