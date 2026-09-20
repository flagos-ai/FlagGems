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
    "_neg_view",
    MarkDecorator(Mark("_neg_view", (), {}, _ispytest=True), _ispytest=True),
)

# Toggle the lazy neg bit while preserving input storage and layout.
# FP8/bool views exist, but their negated values cannot be materialized.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_UNMATERIALIZABLE_DTYPES = _FP8_DTYPES + [torch.bool]

_VALUE_DTYPES = (
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
    + [torch.complex64]
)

_ALL_TEST_DTYPES = _VALUE_DTYPES + _UNMATERIALIZABLE_DTYPES

_NEG_VIEW_SHAPES = [(17,), (12, 13), (0,), (3, 0), (2, 0, 4)] + tu.selected_shapes()

_NEG_VIEW_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]

_NEG_VIEW_TOGGLE_SHAPES = [(16, 32), (4, 8, 16)]

_NEG_VIEW_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]

_NEG_VIEW_BACKWARD_SHAPES = [(16, 64), (7, 13, 29), (0,), (3, 0)]

_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _ALL_TEST_DTYPES
    for value_range in tu.selected_ranges()
]


def _assert_values_equal(res_out, ref_out):
    # For FP8/bool, clear the neg bit on aliases to inspect unchanged storage.
    if ref_out.dtype in _UNMATERIALIZABLE_DTYPES and ref_out.is_neg():
        res_out = torch.ops.aten._neg_view(res_out)
        ref_out = torch.ops.aten._neg_view(ref_out)
    tu.assert_result_equal(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_neg() == ref_out.is_neg()
    # Empty tensors can have equal null pointers without sharing storage.
    assert torch._C._is_alias_of(res_out, inp)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__neg_view(shape, dtype):
    inp = tu.make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = flag_gems._neg_view(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)
    assert res_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = flag_gems._neg_view(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _UNMATERIALIZABLE_DTYPES)
def test__neg_view_unmaterializable_dtypes(shape, dtype):
    inp = tu.make_input(dtype, shape, ["0", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = flag_gems._neg_view(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    assert res_out.is_neg()
    _assert_values_equal(res_out, ref_out)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_non_contiguous(shape, dtype):
    base = tu.make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = flag_gems._neg_view(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_TOGGLE_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_toggle(shape, dtype):
    base = tu.make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_base = tu.to_reference(base)

    inp = torch.ops.aten._neg_view(base)
    ref_inp = torch.ops.aten._neg_view(ref_base)
    assert inp.is_neg()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = flag_gems._neg_view(inp)

    _assert_view_semantics(res_out, ref_out, base)
    _assert_values_equal(res_out, ref_out)
    assert not res_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
def test__neg_view_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = flag_gems._neg_view(values)

    _assert_view_semantics(res_out, ref_out, values)
    tu.assert_result_equal(res_out, ref_out)
    # Exact numerical equality does not distinguish the signs of zero.
    zeros = values == 0
    tu.assert_result_equal(
        torch.signbit(res_out[zeros]), torch.signbit(ref_out[ref_inp == 0])
    )


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__neg_view_mutation(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    res_out = flag_gems._neg_view(inp)
    ref_out = torch.ops.aten._neg_view(ref_inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    tu.assert_result_equal(res_out, ref_out)
    # fill_ through a neg view writes -2.5 into the base storage, so the input
    # (no neg bit) materializes to -2.5 on both sides.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._neg_view
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_FP8_DTYPES))
)
def test__neg_view_fp8_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = flag_gems._neg_view(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)
    # The view must preserve storage bytes, including signed zeros and NaNs.
    tu.assert_result_equal(inp.view(torch.uint8), ref_inp.view(torch.uint8))


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_BACKWARD_SHAPES)
# FP8 backward calls aten.neg, which has no CPU/CUDA FP8 kernel.
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES + [torch.complex64])
)
def test__neg_view_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems._neg_view(inp)
    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark._neg_view
def test__neg_view_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._neg_view(3.14)
