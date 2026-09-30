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

from . import test_utils as tu

# aten::_test_optional_intlist is CPU-registered with two probed call forms:
#   * addends=None -> zero-copy identity returning the operand object itself;
#   * addends=...  -> elementwise int32 add with int32 wraparound; the schema
#     requires a 1-D int32 operand and at least one addends entry per element
#     (extra entries are ignored, a tuple is accepted).
# A CUDA operand raises NotImplementedError, so the reference and the candidate
# both receive CPU operands. The operator has no autograd formula at all.
_CPU = torch.device("cpu")

_IDENTITY_DTYPES = list(tu.REQUIRED_DTYPES) + [
    torch.bool,
    torch.float64,
    torch.complex64,
]

# The additive form indexes the operand with a single index, so only 1-D operands
# can carry it. Both rows stay in quick mode to keep the call form covered.
_ONE_D_SHAPES = [(1,), (256,)]

# (operand length, addends) rows: single element, negative, zero, a list longer
# than the operand, an empty operand with an empty list, the identity form on an
# empty operand, and the schema's tuple form.
_ADDENDS_FORM_ROWS = [
    (1, [42]),
    (4, [-1, -2, -3, -4]),
    (4, [0, 0, 0, 0]),
    (5, [1, 2, 3, 4, 5, 6, 7]),
    (0, []),
    (0, None),
    (3, (7, 8, 9)),
]

# int32 boundary arithmetic in both wraparound directions.
_WRAP_ROWS = [
    ([2147483647], [1]),
    ([2147483647], [2147483647]),
    ([-2147483648], [-1]),
    ([0, 2147483647], [0, 1]),
]

# Strided, transposed and non-zero-offset operands: the identity form returns the
# operand object itself, so its geometry has to survive unchanged.
_LAYOUT_ROWS = [
    ("column_step", (64, 32)),
    ("transposed", (32, 64)),
    ("window", (16, 8, 4)),
]

_OUT_SHAPES = [(4,), (256,)]
_OUT_LAYOUTS = ["column_step", "offset_window"]
# The .out overload writes into a torch.int buffer; other dtypes are rejected
# before an element is written.
_OUT_REJECT_DTYPES = [torch.int64, torch.float32, torch.float8_e4m3fn, torch.bool]

_GRAD_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
_GRAD_LAYOUTS = ["column_step", "window"]

_REJECT_VALUES_DTYPES = [torch.int64, torch.float32, torch.bool]
_REJECT_VALUES_SHAPES = [(2, 2), (2, 2, 2), ()]

# Shared special-value matrix: nan-only for every floating dtype plus inf and
# nan+inf where the dtype can represent infinity (float8_e4m3fn cannot).
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_IDENTITY_DTYPES), quick=[])


def _independent_operands(dtype, shape, value_range):
    # The reference operand is an independent clone of the same parent, so it has
    # identical strides and storage offset without going through the shared
    # helper's reference-device handling (this kernel only exists on the CPU).
    values = tu.make_input(dtype, shape, value_range).to(_CPU)
    return values, values.detach().clone()


def _layout_view(tensor, layout):
    if layout == "column_step":
        return tensor[:, ::2]
    if layout == "transposed":
        return tensor.t()
    return tensor[2:6]


def _layout_operands(dtype, shape, layout):
    # Replaying the same slicing on the clone gives the reference operand the
    # exact strides and storage offset of the candidate operand.
    values = tu.make_input(dtype, shape, ["-1", "1"]).to(_CPU)
    return _layout_view(values, layout), _layout_view(values.detach().clone(), layout)


def _ramp_addends(length):
    return [(index % 7) - 3 for index in range(length)]


def _exact_upstream(shape, dtype):
    # An explicit upstream reaches the missing derivative without a reduction.
    count = 1
    for extent in shape:
        count *= extent
    steps = torch.arange(1, count + 1, dtype=torch.float32, device=_CPU)
    if dtype.is_complex:
        upstream = torch.complex(steps, steps.flip(0)).to(dtype)
    else:
        upstream = steps.to(dtype)
    return upstream.reshape(shape)


def _assert_identity(res_out, ref_out, values):
    # The result is the operand itself; the independently built native result
    # carries the geometry and dtype the candidate has to preserve.
    assert res_out is values
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.dtype == ref_out.dtype


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _IDENTITY_DTYPES)
def test_test_optional_intlist_identity(shape, value_range, dtype):
    values, ref_values = _independent_operands(dtype, shape, value_range)
    before = values.detach().clone()

    ref_out = torch.ops.aten._test_optional_intlist(ref_values, None)
    res_out = flag_gems._test_optional_intlist(values, None)

    tu.assert_result_equal(res_out, ref_out)
    _assert_identity(res_out, ref_out, values)
    # The identity form never writes to its operand, and the returned alias is
    # writable: the write lands in the operand.
    tu.assert_result_equal(values, before)
    res_out.fill_(0)
    tu.assert_result_equal(values, torch.zeros_like(values))


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("layout,shape", _LAYOUT_ROWS)
def test_test_optional_intlist_identity_layout(layout, shape):
    values, ref_values = _layout_operands(torch.float32, shape, layout)
    before = values.detach().clone()

    ref_out = torch.ops.aten._test_optional_intlist(ref_values, None)
    res_out = flag_gems._test_optional_intlist(values, None)

    tu.assert_result_equal(res_out, ref_out)
    _assert_identity(res_out, ref_out, values)
    tu.assert_result_equal(values, before)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("shape", _ONE_D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_test_optional_intlist_addends(shape, value_range):
    values, ref_values = _independent_operands(torch.int32, shape, value_range)
    addends = _ramp_addends(values.numel())

    ref_out = torch.ops.aten._test_optional_intlist(ref_values, addends)
    res_out = flag_gems._test_optional_intlist(values, addends)

    # A fresh int32 tensor following the operand's shape; the operand is read only.
    assert res_out.dtype == torch.int32
    assert res_out.shape == ref_out.shape
    assert res_out is not values
    assert res_out.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("length,addends", _ADDENDS_FORM_ROWS)
def test_test_optional_intlist_addends_forms(length, addends):
    values = torch.arange(length, dtype=torch.int32, device=_CPU) - 3
    ref_values = values.detach().clone()

    ref_out = torch.ops.aten._test_optional_intlist(ref_values, addends)
    res_out = flag_gems._test_optional_intlist(values, addends)

    if addends is None:
        assert res_out is values
    else:
        assert res_out.dtype == torch.int32
        assert res_out is not values
    assert res_out.shape == ref_out.shape
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("payload,addends", _WRAP_ROWS)
def test_test_optional_intlist_addends_wraparound(payload, addends):
    values = torch.tensor(payload, dtype=torch.int32, device=_CPU)
    ref_values = values.detach().clone()

    ref_out = torch.ops.aten._test_optional_intlist(ref_values, addends)
    res_out = flag_gems._test_optional_intlist(values, addends)

    assert res_out.dtype == torch.int32
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("shape", _OUT_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_test_optional_intlist_out(shape, value_range):
    values, ref_values = _independent_operands(torch.int32, shape, value_range)
    addends = _ramp_addends(values.numel())
    out = torch.empty_like(values)
    ref_out = torch.empty_like(ref_values)

    torch.ops.aten._test_optional_intlist.out(ref_values, addends, out=ref_out)
    res = flag_gems._test_optional_intlist(values, addends, out=out)

    # The .out overload returns the caller's buffer and fills it in place, leaving
    # the operand untouched.
    assert res is out
    assert res.data_ptr() != values.data_ptr()
    tu.assert_result_equal(out, ref_out)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("layout", _OUT_LAYOUTS)
def test_test_optional_intlist_out_strided(layout):
    values = torch.arange(6, dtype=torch.int32, device=_CPU)
    addends = [1] * values.numel()
    # A strided / non-zero-offset view of a larger int32 buffer: the .out overload
    # has to write through that geometry and leave the other elements alone.
    base = torch.zeros(12, dtype=torch.int32, device=_CPU)
    ref_base = torch.zeros(12, dtype=torch.int32, device=_CPU)
    out = base[::2] if layout == "column_step" else base[4:10]
    ref_out = ref_base[::2] if layout == "column_step" else ref_base[4:10]

    torch.ops.aten._test_optional_intlist.out(values, addends, out=ref_out)
    res = flag_gems._test_optional_intlist(values, addends, out=out)

    assert res is out
    assert res.stride() == ref_out.stride()
    assert res.storage_offset() == ref_out.storage_offset()
    tu.assert_result_equal(out, ref_out)
    tu.assert_result_equal(base, ref_base)


@pytest.mark.test_optional_intlist
def test_test_optional_intlist_out_resize():
    values = torch.arange(4, dtype=torch.int32, device=_CPU)
    addends = [1] * values.numel()
    # A plain buffer whose length differs from the result: the native .out resizes
    # it in place and still returns the caller's tensor.
    out = torch.empty(7, dtype=torch.int32, device=_CPU)
    ref_out = torch.empty(7, dtype=torch.int32, device=_CPU)

    ref_res = torch.ops.aten._test_optional_intlist.out(values, addends, out=ref_out)
    res = flag_gems._test_optional_intlist(values, addends, out=out)

    assert res is out
    assert res.shape == ref_res.shape == ref_out.shape
    tu.assert_result_equal(out, ref_out)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_test_optional_intlist_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).to(_CPU)
    ref_values = values.detach().clone()

    ref_out = torch.ops.aten._test_optional_intlist(ref_values, None)
    res_out = flag_gems._test_optional_intlist(values, None)

    # The whole payload is compared, so nan and inf have to survive untouched.
    assert res_out is values
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("layout", _GRAD_LAYOUTS)
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test_test_optional_intlist_rejects_backward(layout, dtype):
    # Native replaces a non-leaf operand's history with an unimplemented derivative.
    shape = (4, 6) if layout == "column_step" else (16, 8, 4)
    base, ref_base = _independent_operands(dtype, shape, ["-1", "1"])
    base.requires_grad_(True)
    ref_base.requires_grad_(True)
    values = _layout_view(base, layout)
    ref_values = _layout_view(ref_base, layout)
    upstream = _exact_upstream(values.shape, dtype)

    ref_out = torch.ops.aten._test_optional_intlist(ref_values, None)
    res_out = flag_gems._test_optional_intlist(values, None)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out is values
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_optional_intlist is not implemented",
    ):
        torch.autograd.grad(ref_out, ref_base, grad_outputs=upstream)
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_optional_intlist is not implemented",
    ):
        torch.autograd.grad(res_out, base, grad_outputs=upstream)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("dtype", _REJECT_VALUES_DTYPES)
def test_test_optional_intlist_addends_rejects_values_dtype(dtype):
    values = torch.zeros(4, dtype=dtype, device=_CPU)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_intlist(values, [1, 1, 1, 1])


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("shape", _REJECT_VALUES_SHAPES)
def test_test_optional_intlist_addends_rejects_values_rank(shape):
    values = torch.zeros(shape, dtype=torch.int32, device=_CPU)
    addends = _ramp_addends(values.numel())
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_intlist(values, addends)


@pytest.mark.test_optional_intlist
def test_test_optional_intlist_addends_rejects_short_list():
    values = torch.arange(4, dtype=torch.int32, device=_CPU)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._test_optional_intlist(values, [1, 1])


@pytest.mark.test_optional_intlist
def test_test_optional_intlist_rejects_non_tensor_values():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_intlist([1, 2, 3], None)


@pytest.mark.test_optional_intlist
def test_test_optional_intlist_rejects_tensor_addends():
    values = torch.arange(4, dtype=torch.int32, device=_CPU)
    tensor_addends = torch.zeros(4, dtype=torch.int32, device=_CPU)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_intlist(values, tensor_addends)


@pytest.mark.test_optional_intlist
@pytest.mark.parametrize("out_dtype", _OUT_REJECT_DTYPES)
def test_test_optional_intlist_out_rejects_dtype(out_dtype):
    values = torch.arange(4, dtype=torch.int32, device=_CPU)
    out = torch.zeros(4, dtype=out_dtype, device=_CPU)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_intlist(values, [1, 1, 1, 1], out=out)
