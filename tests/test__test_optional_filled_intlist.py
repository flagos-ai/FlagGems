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

# `_test_optional_filled_intlist(Tensor values, int[2]? addends)` dispatches on
# CPU only (CUDA and Meta raise NotImplementedError). `addends=None` is genuine
# object identity for every dtype and shape, and that form is differentiable:
# a non-leaf operand has no implemented derivative. With addends present
# `values` must be rank-1 int32 and one addend is consumed per element, so an
# explicit list must be at least as long as the tensor while the `int[2]?` fill
# only covers tensors of up to two elements. The second operand is a host int
# list, so there is no broadcast operand pair.
_IDENTITY_DTYPES = tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64]

# Rank-1 shapes only: the arithmetic path indexes `addends` once per element.
# The zero/one/two and addend-boundary rows are cheap and stay in quick mode;
# (256,) is the spec 1-D level and stays default-only.
_FILLED_INT_SHAPES = [(0,), (1,), (2,)]
_LIST_SHAPES = tu.selected_cases([(0,), (1,), (2,), (256,)], quick=[(0,), (1,), (2,)])
_SMALL_SHAPES = [(1,), (2,)]

# Positive / negative / zero plus both int32 boundaries; the boundary rows pair
# with a range that cannot overflow the addition.
_ADDEND_ROWS = [
    (3, ["-1", "1"]),
    (-3, ["-1", "1"]),
    (0, ["-1", "1"]),
    (2147483647, ["-1", "0"]),
    (-2147483648, ["0", "1"]),
]

_VIEW_CASES = [
    ("step2_offset1", lambda base: base[1::2]),
    ("tail_offset3", lambda base: base[3:]),
    ("step3", lambda base: base[2::3]),
]

_OUT_FORMS = ("filled_int", "explicit_list", "none")

# Valid `.out` buffers besides a fresh contiguous one: an offset slice, a
# stride-2 slice and fresh buffers whose size differs from the result. The
# native overload resizes a non-view buffer to the result shape, then copies.
_OUT_BUFFER_ROWS = [
    ("offset_slice", 2, "offset"),
    ("strided_slice", 2, "strided"),
    ("padded", 3, "fresh"),
    ("truncated", 1, "fresh"),
]

_ADDEND_CYCLE = (3, -4, 0, 7, -2)

# Positive special values stay default-only; derivative rejection is a negative case.
_BACKWARD_DTYPES = [
    dtype for dtype in _IDENTITY_DTYPES if dtype.is_floating_point or dtype.is_complex
]


def _cpu(values):
    """The native op dispatches on CPU; tu.make_input takes no device argument."""
    return values.to("cpu")


def _metadata(tensor):
    """Identity facts captured before the call, not compared back to an alias."""
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.untyped_storage().data_ptr(),
        tensor.is_contiguous(),
    )


def _list_addends(n):
    """One addend per element; the schema fill `int[2]?` sets the floor."""
    return [_ADDEND_CYCLE[i % len(_ADDEND_CYCLE)] for i in range(max(n, 2))]


def _assert_new_storage(res_out, inp):
    """A fresh result tensor; empty storages share the null data pointer."""
    assert res_out is not inp
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()


def _out_buffer(kind, numel):
    """A valid out buffer plus the storage it was carved from, if any."""
    if kind == "fresh":
        return torch.zeros(numel, dtype=torch.int32), None
    if kind == "offset":
        base = torch.zeros(numel + 2, dtype=torch.int32)
        return base[1 : 1 + numel], base
    base = torch.zeros(2 * numel, dtype=torch.int32)
    return base[::2], base


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("dtype", _IDENTITY_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test__test_optional_filled_intlist_none_addends(shape, value_range, dtype):
    inp = _cpu(tu.make_input(dtype, shape, value_range))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, None)
    res_out = flag_gems._test_optional_filled_intlist(inp, None)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", _FILLED_INT_SHAPES)
def test__test_optional_filled_intlist_filled_int_addends(shape, value_range):
    inp = _cpu(tu.make_input(torch.int32, shape, value_range))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, 7)
    res_out = flag_gems._test_optional_filled_intlist(inp, 7)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", _LIST_SHAPES)
def test__test_optional_filled_intlist_list_addends(shape, value_range):
    inp = _cpu(tu.make_input(torch.int32, shape, value_range))
    addends = _list_addends(inp.numel())
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, addends)
    res_out = flag_gems._test_optional_filled_intlist(inp, addends)

    assert res_out.device == inp.device
    _assert_new_storage(res_out, inp)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("addends,value_range", _ADDEND_ROWS)
@pytest.mark.parametrize("shape", _SMALL_SHAPES)
def test__test_optional_filled_intlist_addends_values(shape, value_range, addends):
    inp = _cpu(tu.make_input(torch.int32, shape, value_range))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, addends)
    res_out = flag_gems._test_optional_filled_intlist(inp, addends)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("out_form", _OUT_FORMS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", _SMALL_SHAPES)
def test__test_optional_filled_intlist_out(shape, value_range, out_form):
    inp = _cpu(tu.make_input(torch.int32, shape, value_range))
    ref_inp = tu.to_reference(inp)
    if out_form == "filled_int":
        addends = 7
    elif out_form == "explicit_list":
        addends = _list_addends(inp.numel())
    else:
        addends = None

    res_buf = torch.empty(shape, dtype=torch.int32)
    ref_buf = torch.empty(shape, dtype=torch.int32)
    ref_out = torch.ops.aten._test_optional_filled_intlist.out(
        ref_inp, addends, out=ref_buf
    )
    res_out = flag_gems._test_optional_filled_intlist(inp, addends, out=res_buf)

    assert res_out is res_buf
    if addends is None:
        # The out overload copies instead of aliasing the input the way the
        # default overload does for a missing addends list.
        _assert_new_storage(res_out, inp)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("label,out_numel,kind", _OUT_BUFFER_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__test_optional_filled_intlist_out_buffer(label, out_numel, kind, value_range):
    inp = _cpu(tu.make_input(torch.int32, (2,), value_range))
    ref_inp = tu.to_reference(inp)
    inp_before = tu.to_reference(inp)

    res_buf, res_base = _out_buffer(kind, out_numel)
    ref_buf, ref_base = _out_buffer(kind, out_numel)

    ref_out = torch.ops.aten._test_optional_filled_intlist.out(ref_inp, 7, out=ref_buf)
    res_out = flag_gems._test_optional_filled_intlist(inp, 7, out=res_buf)

    assert res_out is res_buf
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    # Exact native resize/copy semantics for a strided, offset or mismatched
    # buffer: shape and values must match the oracle whatever it does.
    tu.assert_result_equal(res_out, ref_out)
    if ref_base is not None:
        # Only the logical elements may be written; the rest of the backing
        # storage keeps its initialised fill on both sides.
        tu.assert_result_equal(res_base, ref_base)
    # The values input is read-only in both overloads.
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__test_optional_filled_intlist_list_addends_non_contiguous(
    view_case, value_range
):
    _, view_fn = view_case
    base = _cpu(tu.make_input(torch.int32, (16,), value_range))
    inp = view_fn(base)
    addends = _list_addends(inp.numel())
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, addends)
    res_out = flag_gems._test_optional_filled_intlist(inp, addends)

    _assert_new_storage(res_out, inp)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.float32, torch.uint8, torch.complex64]
)
@pytest.mark.parametrize(
    "shape",
    tu.selected_cases([(1,), (16,), (2, 19, 7)], quick=[(1,), (16,), (2, 19, 7)]),
)
def test__test_optional_filled_intlist_none_returns_input(shape, dtype):
    inp = _cpu(tu.make_input(dtype, shape, ["-1", "1"]))
    before = _metadata(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, None)
    res_out = flag_gems._test_optional_filled_intlist(inp, None)

    # Native identity means the returned object is the input object, so a later
    # in-place update of the input is visible through the result. The snapshot
    # was taken before the call, so it is not a comparison back to itself.
    assert res_out is inp
    assert _metadata(res_out) == before
    tu.assert_result_equal(res_out, ref_out)
    inp.add_(1)
    ref_inp.add_(1)
    tu.assert_result_equal(res_out, ref_inp)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("view_case", _VIEW_CASES)
def test__test_optional_filled_intlist_none_keeps_view_metadata(view_case):
    _, view_fn = view_case
    base = _cpu(tu.make_input(torch.int32, (16,), ["-1", "1"]))
    inp = view_fn(base)
    assert not inp.is_contiguous() or inp.storage_offset() != 0
    before = _metadata(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, None)
    res_out = flag_gems._test_optional_filled_intlist(inp, None)

    assert res_out is inp
    assert _metadata(res_out) == before
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize(
    "shape",
    tu.selected_cases([(1,), (256,), (2, 19, 7)], quick=[(1,), (256,), (2, 19, 7)]),
)
def test__test_optional_filled_intlist_none_addends_keyword(shape, dtype):
    inp = _cpu(tu.make_input(dtype, shape, ["-1", "1"]))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, addends=None)
    res_out = flag_gems._test_optional_filled_intlist(inp, addends=None)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(
        tu.special_value_cases([d for d in _IDENTITY_DTYPES if d.is_floating_point]),
        quick=[],
    ),
)
def test__test_optional_filled_intlist_none_addends_nan_inf(dtype, scenario):
    inp = _cpu(tu.make_special_input(dtype, scenario))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, None)
    res_out = flag_gems._test_optional_filled_intlist(inp, None)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_test_optional_filled_intlist_rejects_backward(dtype):
    # A non-leaf view reaches the operator's missing derivative; a leaf identity
    # would make autograd stop at the output and never exercise this contract.
    base = tu.make_input(dtype, (16,), ["-1", "1"]).cpu().requires_grad_(True)
    ref_base = tu.to_reference(base.detach()).requires_grad_(True)
    inp, ref_inp = base[::2], ref_base[::2]
    upstream = tu.make_input(dtype, inp.shape, ["-1", "1"]).cpu()
    ref_out = torch.ops.aten._test_optional_filled_intlist(ref_inp, None)
    res_out = flag_gems._test_optional_filled_intlist(inp, None)
    tu.assert_result_equal(res_out, ref_out)
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_optional_filled_intlist is not implemented",
    ):
        torch.autograd.grad(ref_out, ref_base, grad_outputs=upstream)
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::_test_optional_filled_intlist is not implemented",
    ):
        torch.autograd.grad(res_out, base, grad_outputs=upstream)


# An addends value must cover every element: the native loop reads addends[i]
# for i in range(numel) and the `int[2]?` fill only supplies two entries.
_SHORT_ADDENDS_CASES = [((3,), 7), ((2,), [1]), ((4,), [1, 2]), ((256,), 10)]


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("shape,addends", _SHORT_ADDENDS_CASES)
def test__test_optional_filled_intlist_rejects_short_addends(shape, addends):
    inp = _cpu(tu.make_input(torch.int32, shape, ["-1", "1"]))

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_filled_intlist(inp, addends)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize(
    "shape,addends",
    [((2, 3), [1, 2]), ((2, 3, 4), [1, 2]), ((2, 3), 7), ((2, 19, 7), [1, 2])],
)
def test__test_optional_filled_intlist_rejects_non_1d_values(shape, addends):
    inp = _cpu(tu.make_input(torch.int32, shape, ["-1", "1"]))

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_filled_intlist(inp, addends)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int64,
        torch.uint8,
        torch.bool,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test__test_optional_filled_intlist_rejects_non_int32_values(dtype):
    inp = _cpu(tu.make_input(dtype, (2,), ["-1", "1"]))

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_filled_intlist(inp, 7)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("addends", [3.5, "abc", {"a": 1}])
def test__test_optional_filled_intlist_rejects_invalid_addends_type(addends):
    inp = _cpu(tu.make_input(torch.int32, (2,), ["-1", "1"]))

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_filled_intlist(inp, addends)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize("values", [[1, 2], 5, "abc"])
def test__test_optional_filled_intlist_rejects_non_tensor_values(values):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_filled_intlist(values, None)


@pytest.mark.test_optional_filled_intlist
@pytest.mark.parametrize(
    "out_dtype", [torch.float32, torch.float64, torch.int64, torch.int16]
)
def test__test_optional_filled_intlist_out_rejects_wrong_dtype(out_dtype):
    inp = _cpu(tu.make_input(torch.int32, (2,), ["-1", "1"]))
    out = torch.empty(2, dtype=out_dtype)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_optional_filled_intlist(inp, 7, out=out)
