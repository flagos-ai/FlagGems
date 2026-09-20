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
    "_neg_view_copy",
    MarkDecorator(Mark("_neg_view_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_neg_view_copy_out",
    MarkDecorator(Mark("_neg_view_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# Materialize negated values in independent contiguous storage.
_NEG_VIEW_COPY_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ]
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
)

_UNSUPPORTED_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.bool]

_NEG_VIEW_COPY_SHAPES = [(17,), (12, 13)] + tu.selected_shapes()

_NEG_VIEW_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]

_NEG_VIEW_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]

_NEG_VIEW_COPY_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]

_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _NEG_VIEW_COPY_DTYPES
    for value_range in tu.selected_ranges()
]


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    assert res_out.is_contiguous()
    assert not res_out.is_neg()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy(shape, dtype):
    inp = tu.make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    # to_reference creates an independent snapshot for the input mutation check.
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_out(shape, dtype):
    inp = tu.make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems._neg_view_copy(inp, out=out)

    # The .out variant must write into and return the caller's buffer itself.
    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_out_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems._neg_view_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
def test__neg_view_copy_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(values)

    tu.assert_result_equal(res_out, ref_out)
    # Exact numerical equality does not distinguish the signs of zero.
    zeros = values == 0
    tu.assert_result_equal(
        torch.signbit(res_out[zeros]), torch.signbit(ref_out[ref_inp == 0])
    )


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_non_contiguous(shape, dtype):
    base = tu.make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_base = tu.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_empty(shape, dtype):
    inp = tu.make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = flag_gems._neg_view_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test__neg_view_copy_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems._neg_view_copy(inp)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__neg_view_copy_rejects_unsupported_dtypes(dtype):
    inp = torch.zeros(4, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._neg_view_copy(inp)


@pytest.mark._neg_view_copy
def test__neg_view_copy_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._neg_view_copy(3.14)
