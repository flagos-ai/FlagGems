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

"""Correctness tests for ``aten::_assert_scalar(Scalar self, str assert_msg) -> ()``.

The operator consumes a Python number plus a message string and asserts on the host: it
has no tensor operand and no tensor result, so the spec's shape / value-range / tensor
dtype grid, broadcast and backward dimensions cannot apply. Its whole operand space is
the Python truthiness matrix below, where the accepted number types stand in for the
dtype axis, plus the message contract and the negative cases.
"""

import decimal

import numpy as np
import pytest
import torch

import flag_gems

from . import test_utils as tu

MSG = "assert_scalar expectation failed"

# Native substitutes this text when the caller passes an empty message.
EMPTY_MSG_FALLBACK = "Assertion is failed"

# Native accepts a Python number and applies Python truthiness, so every non-zero row
# returns () instead of raising. The accepted types are the dtype analogue here: Python
# int within the int64 range, float (float64, incl. the smallest subnormal and the
# largest finite value), bool, complex and numpy.float64.
_ORDINARY_TRUTHY_ROWS = [
    pytest.param(1, id="int_one"),
    pytest.param(-1, id="int_minus_one"),
    pytest.param(2, id="int_two"),
    pytest.param(3, id="int_three"),
    pytest.param(7, id="int_seven"),
    pytest.param(-7, id="int_minus_seven"),
    pytest.param(42, id="int_forty_two"),
    pytest.param(-42, id="int_minus_forty_two"),
    pytest.param(255, id="int_255"),
    pytest.param(256, id="int_256"),
    pytest.param(65535, id="int_65535"),
    pytest.param(10**9, id="int_1e9"),
    pytest.param(-(10**9), id="int_minus_1e9"),
    pytest.param(2**31 - 1, id="int32_max"),
    pytest.param(-(2**31), id="int32_min"),
    pytest.param(2**32, id="int_2pow32"),
    pytest.param(2**63 - 1, id="int64_max"),
    pytest.param(-(2**63), id="int64_min"),
    pytest.param(True, id="bool_true"),
    pytest.param(0.5, id="float_half"),
    pytest.param(-0.5, id="float_minus_half"),
    pytest.param(1.0, id="float_one"),
    pytest.param(-1.0, id="float_minus_one"),
    pytest.param(2.25, id="float_2p25"),
    pytest.param(-2.25, id="float_minus_2p25"),
    pytest.param(1e-3, id="float_small_positive"),
    pytest.param(-1e-3, id="float_small_negative"),
    pytest.param(3.5, id="float_3p5"),
    pytest.param(1e30, id="float_large_positive"),
    pytest.param(-1e30, id="float_large_negative"),
    pytest.param(5e-324, id="float_min_subnormal"),
    pytest.param(1.7976931348623157e308, id="float64_max"),
    pytest.param(1 + 0j, id="complex_real_only"),
    pytest.param(1 + 2j, id="complex_real_and_imag"),
    pytest.param(0 + 1j, id="complex_pure_imaginary"),
    pytest.param(0 - 1j, id="complex_negative_imaginary"),
    pytest.param(1 - 1j, id="complex_mixed_signs"),
    pytest.param(np.float64(1.0), id="numpy_float64_one"),
    pytest.param(np.float64(-2.5), id="numpy_float64_negative"),
    pytest.param(np.float64(0.5), id="numpy_float64_half"),
]

# nan and inf are truthy for this operator, so they belong to the positive matrix but
# stay default-only, exactly like the tensor special-value cases.
_SPECIAL_TRUTHY_ROWS = [
    pytest.param(float("nan"), id="float_nan_is_truthy"),
    pytest.param(float("inf"), id="float_inf"),
    pytest.param(float("-inf"), id="float_neg_inf"),
    pytest.param(complex(float("nan"), 0.0), id="complex_nan_real_part"),
    pytest.param(complex(0.0, float("inf")), id="complex_inf_imag_part"),
    pytest.param(np.float64(float("nan")), id="numpy_float64_nan"),
    pytest.param(np.float64(float("inf")), id="numpy_float64_inf"),
]

# Every zero form is falsy: signed zeros count as zero, including inside a complex and a
# numpy.float64 value.
_FALSY_ROWS = [
    pytest.param(0, id="int_zero"),
    pytest.param(0.0, id="float_zero"),
    pytest.param(-0.0, id="float_negative_zero"),
    pytest.param(False, id="bool_false"),
    pytest.param(0j, id="complex_zero"),
    pytest.param(complex(0.0, 0.0), id="complex_zero_parts"),
    pytest.param(complex(-0.0, -0.0), id="complex_negative_zero_parts"),
    pytest.param(complex(0.0, -0.0), id="complex_zero_real_negative_zero_imag"),
    pytest.param(np.float64(0.0), id="numpy_float64_zero"),
    pytest.param(np.float64(-0.0), id="numpy_float64_negative_zero"),
    pytest.param(np.float64(0), id="numpy_float64_int_zero"),
]

# The message travels unchanged through both paths; only the empty message is rewritten.
_MESSAGE_ROWS = [
    pytest.param(MSG, id="ascii_message"),
    pytest.param("", id="empty_message_falls_back"),
    pytest.param("值必须非零", id="non_ascii_message"),
    pytest.param("tab\tnewline\nreturn\r", id="control_characters"),
    pytest.param("100% failed %s", id="printf_style_placeholders"),
    pytest.param('quote " inside', id="double_quote_message"),
    pytest.param("backslash \\ path", id="backslash_message"),
    pytest.param("x" * 512, id="long_message"),
    pytest.param("   ", id="whitespace_only"),
    pytest.param("\n", id="newline_only"),
    pytest.param("  padded  ", id="padded_message"),
    pytest.param("{braces} {0}", id="format_braces"),
]

# self and assert_msg are named schema parameters, so the keyword and mixed forms are
# part of the public call contract.
_CALL_FORM_ROWS = [
    pytest.param((1, MSG), {}, id="positional"),
    pytest.param((1,), {"assert_msg": MSG}, id="message_as_keyword"),
    pytest.param((), {"self": 1, "assert_msg": MSG}, id="all_keywords"),
]

# No tensor is a valid `self` at any rank, dtype or device, not even a 0-D, bool or empty
# one. The backend dtype flags only gate whether the fixture can be allocated at all.
_TENSOR_DTYPE_FLAGS = {
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.int64: flag_gems.runtime.device.support_int64,
}

_TENSOR_SELF_FIXTURES = [
    ((), torch.float32, "tensor_0d_float32"),
    ((1,), torch.float64, "tensor_1d_float64"),
    ((3,), torch.int64, "tensor_1d_int64"),
    ((2, 2), torch.int32, "tensor_2d_int32"),
    ((), torch.bool, "tensor_0d_bool"),
    ((1,), torch.bfloat16, "tensor_1d_bfloat16"),
    ((1,), torch.float16, "tensor_1d_float16"),
    ((0,), torch.float32, "tensor_empty_float32"),
]

TENSOR_SELF_CASES = [
    pytest.param(shape, dtype, id=case_id)
    for shape, dtype, case_id in _TENSOR_SELF_FIXTURES
    if _TENSOR_DTYPE_FLAGS.get(dtype, True)
]

# Only numpy scalars of the native-accepted types are numbers to this schema; every other
# numpy scalar, container, string and out-of-int64-range integer is rejected.
_NUMBER_SELF_ROWS = [
    pytest.param(np.int8(1), id="numpy_int8"),
    pytest.param(np.int32(1), id="numpy_int32"),
    pytest.param(np.int64(1), id="numpy_int64"),
    pytest.param(np.uint8(1), id="numpy_uint8"),
    pytest.param(np.float16(1.0), id="numpy_float16"),
    pytest.param(np.float32(1.0), id="numpy_float32"),
    pytest.param(np.longdouble(1.0), id="numpy_longdouble"),
    pytest.param(np.bool_(True), id="numpy_bool"),
    pytest.param(np.array(1.0), id="numpy_0d_array"),
    pytest.param(decimal.Decimal(0), id="decimal_zero"),
    pytest.param("1", id="str_value"),
    pytest.param(b"1", id="bytes_value"),
    pytest.param(None, id="none_value"),
    pytest.param(object(), id="plain_object"),
    pytest.param([1], id="list_value"),
    pytest.param((1,), id="tuple_value"),
    pytest.param({1: 1}, id="dict_value"),
    pytest.param(2**63, id="int_above_int64_max"),
    pytest.param(2**64 - 1, id="uint64_max"),
    pytest.param(2**64, id="int_2pow64"),
    pytest.param(-(2**63) - 1, id="int_below_int64_min"),
]

_MESSAGE_TYPE_ROWS = [
    pytest.param(5, id="int_message"),
    pytest.param(1.5, id="float_message"),
    pytest.param(float("nan"), id="nan_message"),
    pytest.param(None, id="none_message"),
    pytest.param([MSG], id="list_message"),
    pytest.param({"msg": MSG}, id="dict_message"),
]

_ARITY_ROWS = [
    pytest.param((), {}, id="missing_self"),
    pytest.param((1,), {}, id="missing_assert_msg"),
    pytest.param((1, MSG, 0), {}, id="too_many_positional_args"),
    pytest.param((), {"self": 1}, id="missing_assert_msg_with_keyword"),
]

# Quick trims only the positive truthy grid, and only its nan/inf rows; the falsy,
# message, call-form and negative rows are cheap host scalars that both modes assert in
# full. Every ordinary truthy boundary stays in quick.
TRUTHY_CASES = tu.selected_cases(
    _ORDINARY_TRUTHY_ROWS + _SPECIAL_TRUTHY_ROWS, quick=_ORDINARY_TRUTHY_ROWS
)
FALSY_CASES = _FALSY_ROWS
MESSAGE_CASES = _MESSAGE_ROWS
CALL_FORM_CASES = _CALL_FORM_ROWS
NUMBER_SELF_CASES = _NUMBER_SELF_ROWS
MESSAGE_TYPE_CASES = _MESSAGE_TYPE_ROWS
ARITY_CASES = _ARITY_ROWS


@pytest.mark.assert_scalar
@pytest.mark.parametrize("value", TRUTHY_CASES)
def test_assert_scalar_truthy_returns_none(value):
    ref = torch.ops.aten._assert_scalar(value, MSG)
    res = flag_gems._assert_scalar(value, MSG)
    # The schema returns (), so a truthy scalar must produce the native no-value result.
    assert res is ref


@pytest.mark.assert_scalar
@pytest.mark.parametrize("value", FALSY_CASES)
def test_assert_scalar_falsy_raises(value):
    with pytest.raises(RuntimeError) as excinfo:
        flag_gems._assert_scalar(value, MSG)
    assert str(excinfo.value) == MSG


@pytest.mark.assert_scalar
@pytest.mark.parametrize("msg", MESSAGE_CASES)
@pytest.mark.parametrize("value", [0, 1])
def test_assert_scalar_message_contract(msg, value):
    if value:
        ref = torch.ops.aten._assert_scalar(value, msg)
        assert flag_gems._assert_scalar(value, msg) is ref
    else:
        with pytest.raises(RuntimeError) as excinfo:
            flag_gems._assert_scalar(value, msg)
        assert str(excinfo.value) == (msg or EMPTY_MSG_FALLBACK)


@pytest.mark.assert_scalar
@pytest.mark.parametrize("args,kwargs", CALL_FORM_CASES)
def test_assert_scalar_call_forms(args, kwargs):
    ref = torch.ops.aten._assert_scalar(*args, **kwargs)
    assert flag_gems._assert_scalar(*args, **kwargs) is ref


@pytest.mark.assert_scalar
@pytest.mark.parametrize("shape,dtype", TENSOR_SELF_CASES)
def test_assert_scalar_rejects_tensor_self(shape, dtype):
    tensor = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._assert_scalar(tensor, MSG)


@pytest.mark.assert_scalar
@pytest.mark.parametrize("value", NUMBER_SELF_CASES)
def test_assert_scalar_rejects_non_number_self(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._assert_scalar(value, MSG)


@pytest.mark.assert_scalar
@pytest.mark.parametrize("message", MESSAGE_TYPE_CASES)
def test_assert_scalar_rejects_non_str_message(message):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._assert_scalar(1, message)


@pytest.mark.assert_scalar
@pytest.mark.parametrize("args,kwargs", ARITY_CASES)
def test_assert_scalar_rejects_wrong_arity(args, kwargs):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._assert_scalar(*args, **kwargs)
