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
from . import test_utils as tu

# Scalars become (1, 1), vectors become (1, N), and ndim >= 2 is unchanged.
_SUPPORTED_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.bool] + utils.COMPLEX_DTYPES
)
# These cases also exercise FP64 and the small integer/complex types omitted
# by the shared device and quick selections.
if not utils.fp64_is_supported:
    _SUPPORTED_DTYPES.append(torch.float64)
if tu.QUICK_MODE:
    _SUPPORTED_DTYPES.extend([torch.int16, torch.complex32])
_SUPPORTED_DTYPES += [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
_VALUE_DTYPES = [dtype for dtype in _SUPPORTED_DTYPES if not dtype.is_complex]
_COMPLEX_DTYPES = [dtype for dtype in _SUPPORTED_DTYPES if dtype.is_complex]
_VALUE_CASES = [
    (dtype, value_range)
    for dtype in _VALUE_DTYPES
    for value_range in tu.selected_ranges()
]
_VALUE_CASE_IDS = [
    f"{str(dtype).replace('torch.', '')}-{'_'.join(value_range)}"
    for dtype, value_range in _VALUE_CASES
]
_SEQUENCE_CASES = [
    (dtype, value_range)
    for dtype, value_range in _VALUE_CASES
    if dtype in utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.bool]
]


@pytest.mark.atleast_2d
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _VALUE_CASES, ids=_VALUE_CASE_IDS)
def test_atleast_2d_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = flag_gems.atleast_2d(inp)

    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == inp.device
    # atleast_2d returns a view: it must alias the input storage.
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_2d
@pytest.mark.parametrize(
    "shape,expected",
    [
        ((), (1, 1)),
        ((1,), (1, 1)),
        ((5,), (1, 5)),
        ((3, 4), (3, 4)),
        ((2, 3, 4), (2, 3, 4)),
        ((2, 3, 4, 5), (2, 3, 4, 5)),
    ],
)
def test_atleast_2d_shape_metadata(shape, expected):
    # The dim boundary is what defines the op: <2 dims are promoted to 2 dims,
    # >= 2 dims are returned unchanged.
    inp = torch.arange(
        max(1, torch.Size(shape).numel()), dtype=torch.float32, device=flag_gems.device
    ).reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = flag_gems.atleast_2d(inp)

    assert tuple(res_out.shape) == expected
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_2d_sequence
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range",
    _SEQUENCE_CASES,
    ids=[
        f"{str(dtype).replace('torch.', '')}-{'_'.join(value_range)}"
        for dtype, value_range in _SEQUENCE_CASES
    ],
)
def test_atleast_2d_sequence(shape, dtype, value_range):
    # Mix a 0-dim scalar, a 1-dim tensor and the current shape so the sequence
    # overload exercises scalar -> (1, 1), 1-dim -> (1, N) and the >= 2-dim
    # identity path.
    inp = [
        tu.make_input(dtype, (), value_range),
        tu.make_input(dtype, (3,), value_range),
        tu.make_input(dtype, shape, value_range),
    ]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_2d.Sequence(ref_inp)
    res_out = flag_gems.atleast_2d(inp)

    assert isinstance(res_out, (list, tuple))
    assert len(res_out) == len(ref_out)
    for res, ref, src in zip(res_out, ref_out, inp):
        assert res.data_ptr() == src.data_ptr()
        tu.assert_result_equal(res, ref)


@pytest.mark.atleast_2d_sequence
def test_atleast_2d_sequence_empty():
    # A Tensor[] input may legitimately be empty: the reference returns an
    # empty list, and the candidate must return an empty list too.
    ref_out = torch.ops.aten.atleast_2d.Sequence([])
    res_out = flag_gems.atleast_2d([])
    assert len(res_out) == len(ref_out)


_NAN_INF_VALUES = [
    float("inf"),
    float("-inf"),
    float("nan"),
    0.0,
    -0.0,
    1.5,
    -2.5,
    1e30,
    -1e30,
]


@pytest.mark.atleast_2d_nan_inf
@pytest.mark.parametrize("shape", [(), (9,), (3, 3)])
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.FLOAT_DTYPES))
def test_atleast_2d_nan_inf(shape, dtype):
    values = _NAN_INF_VALUES[: 1 if shape == () else len(_NAN_INF_VALUES)]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device).reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = flag_gems.atleast_2d(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_2d_complex
@pytest.mark.parametrize("shape", [(), (5,), (2, 5), (2, 3, 4)])
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_atleast_2d_complex(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = flag_gems.atleast_2d(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_2d_backward
@pytest.mark.parametrize("shape", [(), (3,), (16, 64), (7, 13, 29)])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [d for d in _SUPPORTED_DTYPES if d.is_floating_point or d.is_complex]
    ),
)
def test_atleast_2d_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    # Explicit upstream values detect gradients that always return ones.
    grad = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.atleast_2d(inp)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.atleast_2d_negative
def test_atleast_2d_rejects_non_tensor():
    # The aten op requires a Tensor / Tensor[]; scalars hit the argument type
    # check and raise.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_2d(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_2d.Sequence(
            [torch.zeros(2, device=flag_gems.device), 3.14]
        )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.atleast_2d(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.atleast_2d([torch.zeros(2, device=flag_gems.device), 3.14])


@pytest.mark.atleast_2d
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES))
)
def test_atleast_2d_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = flag_gems.atleast_2d(inp)

    tu.assert_result_equal(res_out, ref_out)
