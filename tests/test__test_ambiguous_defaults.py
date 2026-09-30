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

# aten::_test_ambiguous_defaults never reads its operand and returns a constant:
#   .a(Tensor, int a=1, int b=1) -> int64 1, requires a == 1 and b == 1
#   .b(Tensor, int a=2, str b="2") -> int64 2, requires a == 2 and b == "2"
# Both return a fresh CPU 0-dim int64 scalar with no autograd history. The single
# public candidate name must select the overload from its argument types exactly
# as the native packet does (int b -> .a, str b -> .b).
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    # Static capability flags read at import time: collection allocates no tensor
    # and calls no operator. A dtype outside the map is a baseline type.
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# The operand only has to be constructible on the target backend, so the required
# nine dtypes plus the other tensor dtypes the schema accepts are all exercised.
_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.int16, torch.complex64]
    if _dtype_supported(dtype)
]

# (native overload, positional parameter values); the identical call is made on
# flag_gems._test_ambiguous_defaults, where the parameter types select overload.
_CALL_FORMS = [
    pytest.param(torch.ops.aten._test_ambiguous_defaults.a, (1, 1), id="a"),
    pytest.param(torch.ops.aten._test_ambiguous_defaults.b, (2, "2"), id="b"),
]

# Named-argument form of the same two overloads.
_KEYWORD_CALLS = [
    pytest.param(torch.ops.aten._test_ambiguous_defaults.a, {"a": 1, "b": 1}, id="a"),
    pytest.param(torch.ops.aten._test_ambiguous_defaults.b, {"a": 2, "b": "2"}, id="b"),
]

# Defaulted call forms; the native packet is the analog of the public name. The
# quick shape is also part of the default suite.
_PACKET_TAILS = [(), (1,), (1, 1)]
_PACKET_SHAPES = tu.selected_cases(
    [(2, 19, 7), (1024, 1024), (20, 320, 15)], quick=[(2, 19, 7)]
)

# Operand layouts the op must accept while staying independent of the operand:
# transposed, stepped, offset, stride-0 expanded and zero-sized storage.
_OPERAND_LAYOUT_CASES = [
    ("transposed", (4, 8)),
    ("column_step", (4, 16)),
    ("offset_view", (8, 8)),
    ("expanded", (1, 1)),
    ("empty", (0, 3)),
]

# Invalid parameter values, each verified to raise through the native packet.
# "two-arg-int-b" is the unreachable .b default: the packet resolves the
# two-argument int form to .a and raises, so it is a rejection row, not a case.
_NEGATIVE_PARAM_ROWS = [
    ("a-zero", (0, 1)),
    ("a-two", (2, 1)),
    ("a-negative", (-1, 1)),
    ("b-zero", (1, 0)),
    ("b-two", (1, 2)),
    ("b-negative", (1, -1)),
    ("str-b-with-a-one", (1, "2")),
    ("str-b-with-a-zero", (0, "2")),
    ("two-arg-int-b", (2,)),
    ("b-numeric-string", (2, "22")),
    ("b-empty-string", (2, "")),
    ("b-wrong-string", (2, "x")),
    ("int-b-with-a-two", (2, 2)),
    ("three-arg-int-b", (-1, 2)),
]

_NON_TENSOR_OPERANDS = [3.14, [1.0, 2.0], "dummy", 7]

# nan/inf can only be inert operand content here; every floating dtype of _DTYPES
# is covered, with the shared representability rule per scenario. Positive special
# values stay default-only, like every other positive special-value case.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases([dtype for dtype in _DTYPES if dtype.is_floating_point]),
    quick=[],
)


def _assert_scalar_contract(res_out):
    # Native output is a CPU scalar without autograd; dtype, shape and value are
    # covered by the exact shared comparison.
    assert res_out.device.type == "cpu"
    assert not res_out.requires_grad


def _operand_variant(base, kind):
    # Applied to the candidate operand and to the reference clone alike, so both
    # sides keep identical layout facts.
    if kind == "transposed":
        return base.t()
    if kind == "column_step":
        return base[:, ::2]
    if kind == "offset_view":
        return base[3:6, 2:6]
    if kind == "expanded":
        return base.expand(4, 3)
    return base


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("native_overload,params", _CALL_FORMS)
def test__test_ambiguous_defaults_grid(
    shape, value_range, dtype, native_overload, params
):
    inp = tu.make_input(dtype, shape, value_range)
    # Independent snapshot of the operand, taken before the candidate runs: it
    # serves as the reference operand and as the read-only contract check.
    ref_inp = tu.to_reference(inp)

    ref_out = native_overload(ref_inp, *params)
    res_out = flag_gems._test_ambiguous_defaults(inp, *params)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_scalar_contract(res_out)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("shape", _PACKET_SHAPES)
@pytest.mark.parametrize("tail", _PACKET_TAILS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__test_ambiguous_defaults_packet_call_forms(shape, tail, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_ambiguous_defaults(ref_inp, *tail)
    res_out = flag_gems._test_ambiguous_defaults(inp, *tail)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_scalar_contract(res_out)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("native_overload,kwargs", _KEYWORD_CALLS)
def test__test_ambiguous_defaults_keyword_forms(native_overload, kwargs):
    # Named schema arguments must reach the same overload as the positional form.
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = native_overload(ref_inp, **kwargs)
    res_out = flag_gems._test_ambiguous_defaults(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_scalar_contract(res_out)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("native_overload,params", _CALL_FORMS)
def test__test_ambiguous_defaults_operand_is_unread(native_overload, params):
    # Identical calls on different operand contents must give the same constant,
    # leave the operand untouched and return a fresh scalar for each overload.
    zero = tu.make_input(torch.float32, (4, 6), ["0", "0"])
    one = tu.make_input(torch.float32, (4, 6), ["1", "1"])
    ref_zero = tu.to_reference(zero)
    ref_one = tu.to_reference(one)

    ref_out = native_overload(ref_zero, *params)
    res_zero = flag_gems._test_ambiguous_defaults(zero, *params)
    res_one = flag_gems._test_ambiguous_defaults(one, *params)

    tu.assert_result_equal(res_zero, ref_out)
    tu.assert_result_equal(res_one, ref_out)
    tu.assert_result_equal(zero, ref_zero)
    tu.assert_result_equal(one, ref_one)
    assert res_zero is not zero
    assert res_one is not one
    assert torch._C._is_alias_of(res_zero, zero) is False
    assert torch._C._is_alias_of(res_one, one) is False
    _assert_scalar_contract(res_zero)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("kind,shape", _OPERAND_LAYOUT_CASES)
@pytest.mark.parametrize("native_overload,params", _CALL_FORMS)
def test__test_ambiguous_defaults_operand_layouts(kind, shape, native_overload, params):
    base = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    dummy = _operand_variant(base, kind)
    ref_dummy = _operand_variant(ref_base, kind)

    ref_out = native_overload(ref_dummy, *params)
    res_out = flag_gems._test_ambiguous_defaults(dummy, *params)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(dummy, ref_dummy)
    _assert_scalar_contract(res_out)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test__test_ambiguous_defaults_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._test_ambiguous_defaults.a(ref_inp)
    res_out = flag_gems._test_ambiguous_defaults(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    _assert_scalar_contract(res_out)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("native_overload,params", _CALL_FORMS)
def test__test_ambiguous_defaults_has_no_gradient(native_overload, params):
    # Negative contract kept in every mode: the native result never requires grad,
    # so differentiating it against the original leaf operand raises instead of
    # producing a gradient, and the candidate must return an equally detached
    # scalar.
    dummy = tu.make_input(torch.float32, (4, 6), ["-1", "1"]).requires_grad_(True)
    ref_dummy = tu.to_reference(dummy)

    ref_out = native_overload(ref_dummy, *params)
    res_out = flag_gems._test_ambiguous_defaults(dummy, *params)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.grad_fn is None
    with pytest.raises(RuntimeError):
        torch.autograd.grad(res_out, dummy, allow_unused=True)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("label,params", _NEGATIVE_PARAM_ROWS)
def test__test_ambiguous_defaults_rejects_invalid_params(label, params):
    dummy = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_ambiguous_defaults(dummy, *params)


@pytest.mark.test_ambiguous_defaults
@pytest.mark.parametrize("bad_operand", _NON_TENSOR_OPERANDS)
def test__test_ambiguous_defaults_rejects_non_tensor_operand(bad_operand):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_ambiguous_defaults(bad_operand)


@pytest.mark.test_ambiguous_defaults
def test__test_ambiguous_defaults_rejects_missing_operand():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_ambiguous_defaults()
