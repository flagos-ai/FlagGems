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
    "_shape_as_tensor",
    MarkDecorator(Mark("_shape_as_tensor", (), {}, _ispytest=True), _ispytest=True),
)

# Materialize the logical shape as a fresh int64 tensor on CPU.
_SHAPE_AS_TENSOR_INPUT_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
    + utils.COMPLEX_DTYPES
)

_EMPTY_SHAPES = [(0,), (0, 5), (3, 0, 4)]

# (label, view_fn, logical_shape) for a (4, 8, 6) base.
_VIEW_CASES = [
    ("transposed", lambda base: base.transpose(0, 1), (8, 4, 6)),
    ("sliced", lambda base: base[0:3, 2:7, 1], (3, 5)),
    ("narrowed", lambda base: base.narrow(1, 1, 5), (4, 5, 6)),
]


def _assert_result(res_out, ref_out, inp):
    assert res_out.device == torch.device("cpu")
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_empty(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_non_contiguous(view_case, value_range, dtype):
    _, view_fn, expected = view_case
    base = tu.make_input(dtype, (4, 8, 6), value_range)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    assert inp.shape == expected
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_SHAPE_AS_TENSOR_INPUT_DTYPES)),
)
def test__shape_as_tensor_nan_inf(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).expand(4, 8, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", [(), (1,), (2, 3, 5)])
def test__shape_as_tensor_ignores_autograd(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp.detach())

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    assert not res_out.requires_grad
    _assert_result(res_out, ref_out, inp.detach())


@pytest.mark._shape_as_tensor
def test__shape_as_tensor_rejects_non_tensor_input():
    for bad in (5, [1, 2, 3], "abc", 3.14):
        with pytest.raises(RuntimeError):
            torch.ops.aten._shape_as_tensor(bad)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._shape_as_tensor(bad)
