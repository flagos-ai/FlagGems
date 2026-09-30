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

# aten::_test_string_default(Tensor dummy, str a="\"'\\", str b="\"'\\") is an
# ATen schema self-test: it accepts only the schema's three-character default
# strings and hands `dummy` back untouched. There is no arithmetic, so every
# workload asserts exact value preservation, operand identity (`is`) and
# unchanged view metadata. The signature has a single tensor operand, so the
# broadcast grid does not apply; the scalar form is the 0-dim operand and the
# scalar-operand slots are the two strings.
_DEFAULT = "\"'\\"

# Read while this module is imported: no tensor is built and no operator runs,
# so collection and `--list-cases` stay inert. A dtype outside this map is a
# baseline type every backend handles.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


_SUPPORTED_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64]
    if _dtype_supported(dtype)
]

# Only floating and complex dtypes can carry a lazy conjugate/negation bit or an
# autograd graph.
_FLOAT_DTYPES = [
    dtype for dtype in _SUPPORTED_DTYPES if dtype.is_floating_point or dtype.is_complex
]


def _operand_state(t):
    """View/layout state that a metadata-only operator must not change."""
    return (
        tuple(t.shape),
        t.stride(),
        t.storage_offset(),
        t.dtype,
        t.layout,
        str(t.device),
        t.is_conj(),
        t.is_neg(),
        t.requires_grad,
    )


def _assert_operand_preserved(res_out, inp, before_state, before_ptr):
    # The native call returns the operand object itself, so an equal-valued copy
    # or a fresh wrapper would be wrong: identity, storage and the state
    # snapshotted before the call must all survive, matching the native clone.
    assert res_out is inp
    assert res_out.data_ptr() == before_ptr
    assert _operand_state(inp) == before_state


@pytest.mark.test_string_default
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_test_string_default_value_ranges(shape, value_range, dtype):
    # Values are never read; the grid still asserts that every dtype and shape
    # passes through unchanged (and that the schema default strings hold).
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)


# Every spelling of the two optional string operands, including the forms that
# rely on the schema default.
_CALL_FORMS = [
    pytest.param((), {}, id="schema_defaults"),
    pytest.param((_DEFAULT,), {}, id="a_positional"),
    pytest.param((_DEFAULT, _DEFAULT), {}, id="a_b_positional"),
    pytest.param((), {"a": _DEFAULT}, id="a_keyword"),
    pytest.param((), {"b": _DEFAULT}, id="b_keyword"),
    pytest.param((), {"a": _DEFAULT, "b": _DEFAULT}, id="a_b_keyword"),
]

_CALL_FORM_SHAPES = tu.selected_cases([(2, 19, 7), (20, 320, 15)], quick=[(2, 19, 7)])


@pytest.mark.test_string_default
@pytest.mark.parametrize("shape", _CALL_FORM_SHAPES)
@pytest.mark.parametrize("args, kwargs", _CALL_FORMS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_test_string_default_call_forms(shape, args, kwargs, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp, *args, **kwargs)
    res_out = flag_gems._test_string_default(inp, *args, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)


# Stride, storage-offset and zero-stride variations. tu.to_reference rebuilds the
# same view metadata on the reference side, so each row is a real comparison.
_LAYOUT_ROWS = [
    ("transposed", (12, 8)),
    ("strided_columns", (8, 24)),
    ("offset_window", (10, 16)),
    ("expanded", (1, 8)),
]


def _layout_view(base, layout):
    if layout == "transposed":
        return base.transpose(0, 1)
    if layout == "strided_columns":
        return base[:, ::2]
    if layout == "offset_window":
        return base[2:6, 3:11]
    return base[:1].expand(4, base.shape[1])


@pytest.mark.test_string_default
@pytest.mark.parametrize("layout, storage_shape", _LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_test_string_default_preserves_layout(layout, storage_shape, dtype):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    inp = _layout_view(base, layout)
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)


@pytest.mark.test_string_default
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_test_string_default_preserves_lazy_conjugate(dtype):
    # A complex input carries the lazy conjugate bit.
    inp = tu.make_input(dtype, (4, 6), ["-1", "1"]).conj()
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)


@pytest.mark.test_string_default
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_test_string_default_preserves_lazy_negative(dtype):
    inp = torch._neg_view(tu.make_input(dtype, (4, 6), ["-1", "1"]))
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)

    _assert_operand_preserved(res_out, inp, before_state, before_ptr)
    assert res_out.is_neg() and ref_out.is_neg()
    # Clearing the asserted lazy bit on both views compares the complete stored
    # payload without invoking an unavailable FP8 numeric-negation kernel.
    tu.assert_result_equal(torch._neg_view(res_out), torch._neg_view(ref_out))


_EMPTY_SHAPES = [(0,), (0, 3), (3, 0, 7)]


@pytest.mark.test_string_default
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_test_string_default_empty_input(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)


_SCALAR_SHAPES = [(), (1,)]


@pytest.mark.test_string_default
@pytest.mark.parametrize("shape", _SCALAR_SHAPES)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_test_string_default_scalar_operand(shape, dtype):
    # The scalar-operand form: a 0-dim tensor (plus its single-element 1-dim
    # neighbour). The string operands are covered by the call-form test.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp, _DEFAULT, _DEFAULT)
    res_out = flag_gems._test_string_default(inp, _DEFAULT, _DEFAULT)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)


@pytest.mark.test_string_default
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES), quick=[]),
)
def test_test_string_default_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)


@pytest.mark.test_string_default
@pytest.mark.parametrize("dtype", tu.selected_cases(_FLOAT_DTYPES, quick=[]))
def test_test_string_default_backward(dtype):
    # Differentiate through the original leaf with an independent upstream
    # gradient. The native gradient is that upstream tensor itself, so the
    # comparison stays exact for the FP8 and complex dtypes as well.
    leaf = tu.make_input(dtype, (8, 4), ["-1", "1"]).requires_grad_(True)
    ref_leaf = tu.to_reference(leaf).detach().requires_grad_(True)
    inp, ref_inp = leaf.view(4, 8), ref_leaf.view(4, 8)
    upstream = tu.make_input(dtype, (4, 8), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)
    tu.assert_result_equal(res_out, ref_out)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_leaf, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_out, leaf, grad_outputs=upstream)
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.test_string_default
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_test_string_default_alias_write_through(dtype):
    inp = tu.make_input(dtype, (3, 4), ["0", "1"])
    ref_inp = tu.to_reference(inp)
    before_state, before_ptr = _operand_state(inp), inp.data_ptr()

    ref_out = torch.ops.aten._test_string_default(ref_inp)
    res_out = flag_gems._test_string_default(inp)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operand_preserved(res_out, inp, before_state, before_ptr)

    # The result shares the operand's storage, so a write through it has to show
    # up in the operand rather than in a private copy.
    res_out.fill_(1)
    ref_out.fill_(1)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.test_string_default
def test_test_string_default_accepts_none_operand():
    # A None operand is accepted and returned unchanged, so this is positive
    # coverage; the negative rows below deliberately exclude it.
    ref = torch.ops.aten._test_string_default(None)
    assert flag_gems._test_string_default(None) is ref


_INVALID_DEFAULTS = [
    pytest.param("a", "", id="a_empty"),
    pytest.param("b", "", id="b_empty"),
    pytest.param("a", "'", id="a_missing_chars"),
    pytest.param("b", "'", id="b_missing_chars"),
    pytest.param("a", '"', id="a_quote_only"),
    pytest.param("b", '"', id="b_quote_only"),
    pytest.param("a", _DEFAULT + _DEFAULT, id="a_doubled"),
    pytest.param("b", _DEFAULT + _DEFAULT, id="b_doubled"),
    pytest.param("a", "abc", id="a_unrelated_text"),
    pytest.param("b", "abc", id="b_unrelated_text"),
]


@pytest.mark.test_string_default
@pytest.mark.parametrize("field, value", _INVALID_DEFAULTS)
def test_test_string_default_rejects_invalid_default(field, value):
    # Only the exact schema default is accepted for either string.
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_string_default(inp, **{field: value})


_INVALID_ARGUMENTS = [
    pytest.param({"a": 5}, id="a_int"),
    pytest.param({"b": 5}, id="b_int"),
    pytest.param({"a": None}, id="a_none"),
    pytest.param({"b": None}, id="b_none"),
    pytest.param({"c": _DEFAULT}, id="unknown_keyword"),
]


@pytest.mark.test_string_default
@pytest.mark.parametrize("kwargs", _INVALID_ARGUMENTS)
def test_test_string_default_rejects_invalid_arguments(kwargs):
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_string_default(inp, **kwargs)


# Rows carry a metadata label only; the invalid operand is built inside the test
# so collection allocates no tensor.
_NON_TENSOR_OPERANDS = ["int", "float", "str", "list", "tuple", "bool"]


def _non_tensor_operand(label):
    return {
        "int": 5,
        "float": 3.5,
        "str": "dummy",
        "list": [1.0, 2.0],
        "tuple": (1.0, 2.0),
        "bool": True,
    }[label]


@pytest.mark.test_string_default
@pytest.mark.parametrize("operand_label", _NON_TENSOR_OPERANDS)
def test_test_string_default_rejects_non_tensor_operand(operand_label):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_string_default(_non_tensor_operand(operand_label))
