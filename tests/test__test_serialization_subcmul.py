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

"""Correctness tests for ``aten::_test_serialization_subcmul`` (``self - alpha*other``).

The op has a single ``.default`` overload with two required tensor operands and an
optional numeric ``alpha`` (schema default 1), so every workload reaches the public
candidate name ``flag_gems._test_serialization_subcmul``. Case tables hold metadata
only: collection and ``--list-cases`` allocate no tensor, and an invalid argument is
built inside its own test body.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Static capability flags, read at import; no runtime probe and no skip.
# The native operator has no fp8 oracle on the active backend (it raises inside
# itself: "mul_cuda" not implemented for 'Float8_e4m3fn'), so fp8 is exempt at the
# operator level, not because of a comparison-helper limit. bool is rejected only in
# the `self` position (negative case); a bool `other` promotes natively and is a
# positive promotion row instead.
_FLOAT_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _FLOAT_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _FLOAT_DTYPES.append(torch.float64)

_COMPLEX_DTYPES = [torch.complex64]
if utils.fp64_is_supported:
    _COMPLEX_DTYPES.append(torch.complex128)

_INT_DTYPES = [torch.int8, torch.uint8, torch.int16, torch.int32]
if utils.int64_is_supported:
    _INT_DTYPES.append(torch.int64)

_DTYPES = _FLOAT_DTYPES + _INT_DTYPES + _COMPLEX_DTYPES

# --quick shape (tests/conftest.py QUICK_MODE); the default grid uses the shared
# tu.selected_shapes()/tu.selected_ranges() selectors, never a mode check here.
_QUICK_SHAPE = (2, 19, 7)
_LARGE_SHAPE = (20, 320, 15)
_UNIT_RANGE = ["-1", "1"]
_FINITE_ALPHAS = (0.0, -1.5, 2.5)
_SPECIAL_ALPHAS = (float("nan"), float("inf"), float("-inf"))


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test__test_serialization_subcmul(shape, value_range, dtype):
    # The argument-free call form exercises the schema default alpha=1.
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other)
    res_out = flag_gems._test_serialization_subcmul(inp, other)

    tu.assert_result_close(res_out, ref_out)


# Broadcast pairs: the 1-dim and multi-dim patterns scaled onto the large spec
# shapes (default suite) plus small logical pairs that stay in --quick.
_SMALL_BROADCAST_ROWS = [
    (torch.float16, (2, 3, 5), (5,)),
    (torch.float32, (), (2, 3)),
    (torch.float32, (2, 3), ()),
    (torch.int32, (2, 3, 5), (2, 1, 5)),
    (torch.uint8, (1, 3, 1), (2, 3, 5)),
]
_LARGE_BROADCAST_ROWS = [
    (torch.float16, (1024, 1024), (1024,)),
    (torch.float16, (1, 320, 1), (20, 320, 15)),
    (torch.uint8, (20, 320, 15), (1, 320, 1)),
    (torch.int8, (16, 128, 64, 60), (1, 128, 1, 60)),
    (torch.int32, (16, 7, 57, 32, 29), (1, 7, 1, 32, 1)),
]
if utils.int64_is_supported:
    _LARGE_BROADCAST_ROWS.append((torch.int64, (20, 320, 15), (15,)))
_BROADCAST_CASES = tu.selected_cases(
    _LARGE_BROADCAST_ROWS + _SMALL_BROADCAST_ROWS, quick=_SMALL_BROADCAST_ROWS
)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("dtype,inp_shape,other_shape", _BROADCAST_CASES)
def test__test_serialization_subcmul_broadcast(dtype, inp_shape, other_shape):
    inp = tu.make_input(dtype, inp_shape, _UNIT_RANGE)
    other = tu.make_input(dtype, other_shape, _UNIT_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other)
    res_out = flag_gems._test_serialization_subcmul(inp, other)

    tu.assert_result_close(res_out, ref_out)


# Scalar-parameter coverage: alpha is the operator's Scalar argument, so these rows
# are the required scalar form. An int alpha keeps the operand dtype, while a float
# alpha on an integer operand promotes to float32 (separate native call form).
_ALPHA_ROWS = (
    [
        (dtype, alpha, _LARGE_SHAPE)
        for dtype in _FLOAT_DTYPES
        for alpha in _FINITE_ALPHAS + _SPECIAL_ALPHAS
    ]
    + [
        (dtype, alpha, _LARGE_SHAPE)
        for dtype in _COMPLEX_DTYPES
        for alpha in _FINITE_ALPHAS + (1 + 2j,)
    ]
    + [
        (dtype, alpha, _LARGE_SHAPE)
        for dtype in _INT_DTYPES
        for alpha in (0, -3, 5, torch.iinfo(dtype).max)
    ]
    + [(dtype, 2.5, _LARGE_SHAPE) for dtype in _INT_DTYPES]
    + [
        # Finite boundary, representable only in float32/float64.
        (torch.float32, 1e30, _LARGE_SHAPE)
    ]
)
if utils.fp64_is_supported:
    _ALPHA_ROWS.append((torch.float64, 1e30, _LARGE_SHAPE))

# --quick keeps every dtype and the finite alpha sweep at the small shape; the
# nan/inf and large-boundary alphas stay in the default suite.
_ALPHA_QUICK_ROWS = (
    [
        (dtype, alpha, _QUICK_SHAPE)
        for dtype in _FLOAT_DTYPES + _COMPLEX_DTYPES
        for alpha in _FINITE_ALPHAS
    ]
    + [(dtype, alpha, _QUICK_SHAPE) for dtype in _INT_DTYPES for alpha in (0, -3, 5)]
    + [(dtype, 2.5, _QUICK_SHAPE) for dtype in _INT_DTYPES]
    + [(dtype, 1 + 2j, _QUICK_SHAPE) for dtype in _COMPLEX_DTYPES]
)
_ALPHA_CASES = tu.selected_cases(
    _ALPHA_ROWS + _ALPHA_QUICK_ROWS, quick=_ALPHA_QUICK_ROWS
)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("dtype,alpha,shape", _ALPHA_CASES)
def test__test_serialization_subcmul_alpha(dtype, alpha, shape):
    inp = tu.make_input(dtype, shape, _UNIT_RANGE)
    other = tu.make_input(dtype, shape, _UNIT_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other, alpha)
    res_out = flag_gems._test_serialization_subcmul(inp, other, alpha)

    tu.assert_result_close(res_out, ref_out)


# Mixed-dtype promotion pairs; the same pairs are replayed at the small quick shape.
def _mixed_dtype_rows(shape):
    rows = [
        (torch.int32, torch.float32, 2.5, shape),
        (torch.uint8, torch.float16, 2.5, shape),
        (torch.float16, torch.float32, 1.0, shape),
        (torch.int8, torch.int16, 3, shape),
        (torch.float32, torch.bool, 2.5, shape),
        (torch.complex64, torch.float32, 2.5, shape),
    ]
    if utils.int64_is_supported:
        rows.append((torch.int32, torch.int64, 3, shape))
    if utils.fp64_is_supported:
        rows.append((torch.int8, torch.float64, 2.5, shape))
        rows.append((torch.complex128, torch.float64, 2.5, shape))
    return rows


_MIXED_CASES = tu.selected_cases(
    _mixed_dtype_rows(_LARGE_SHAPE) + _mixed_dtype_rows(_QUICK_SHAPE),
    quick=_mixed_dtype_rows(_QUICK_SHAPE),
)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("inp_dtype,other_dtype,alpha,shape", _MIXED_CASES)
def test__test_serialization_subcmul_mixed_dtypes(inp_dtype, other_dtype, alpha, shape):
    inp = tu.make_input(inp_dtype, shape, _UNIT_RANGE)
    other = tu.make_input(other_dtype, shape, _UNIT_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other, alpha)
    res_out = flag_gems._test_serialization_subcmul(inp, other, alpha)

    tu.assert_result_close(res_out, ref_out)


# Non-trivial operand layouts: non-contiguous stride, non-zero storage offset and
# an expanded zero-stride broadcast view. Cheap rows, so they stay in --quick.
_LAYOUT_ROWS = [
    (torch.float32, "transposed"),
    (torch.float32, "storage_offset"),
    (torch.float16, "expanded"),
    (torch.float32, "expanded"),
]
_LAYOUT_CASES = tu.selected_cases(_LAYOUT_ROWS, quick=_LAYOUT_ROWS)


def _layout_operands(dtype, kind):
    base = tu.make_input(dtype, (8, 16), ["-1", "0"])
    other = tu.make_input(dtype, (8, 16), ["0", "1"])
    if kind == "transposed":
        return base.t(), other.t()
    if kind == "storage_offset":
        return base[2:6], other[2:6]
    return base, other[:1].expand(8, 16)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("dtype,kind", _LAYOUT_CASES)
def test__test_serialization_subcmul_layouts(dtype, kind):
    inp, other = _layout_operands(dtype, kind)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other)
    res_out = flag_gems._test_serialization_subcmul(inp, other)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)


# Zero-element operands are valid and must produce empty outputs.
_EMPTY_ROWS = [
    (torch.float32, (0,), (0,)),
    (torch.float32, (0, 3), (3,)),
    (torch.float32, (3, 0), (3, 0)),
    (torch.float32, (0, 3), (0, 1)),
    (torch.float16, (0,), (0,)),
]
_EMPTY_CASES = tu.selected_cases(_EMPTY_ROWS, quick=_EMPTY_ROWS)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("dtype,inp_shape,other_shape", _EMPTY_CASES)
def test__test_serialization_subcmul_empty(dtype, inp_shape, other_shape):
    inp = tu.make_input(dtype, inp_shape, _UNIT_RANGE)
    other = tu.make_input(dtype, other_shape, _UNIT_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other)
    res_out = flag_gems._test_serialization_subcmul(inp, other)

    tu.assert_result_close(res_out, ref_out)


# Gradients through the original leaves: d/dself is the upstream tensor and d/dother
# is -alpha times it, reduced over any broadcast dimensions. alpha=2.5 keeps the two
# gradients distinct and non-uniform.
_BACKWARD_ROWS = [
    (torch.float16, (20, 320, 15), (20, 320, 15)),
    (torch.float32, (), ()),
    (torch.float32, (20, 320, 15), (1, 320, 1)),
]
if utils.bf16_is_supported:
    _BACKWARD_ROWS.append((torch.bfloat16, (16, 128, 64, 60), (16, 128, 64, 60)))
if utils.fp64_is_supported:
    _BACKWARD_ROWS.append((torch.float64, (1024, 1024), (1024, 1024)))
_BACKWARD_CASES = tu.selected_cases(_BACKWARD_ROWS, quick=[])


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("dtype,inp_shape,other_shape", _BACKWARD_CASES)
def test__test_serialization_subcmul_backward(dtype, inp_shape, other_shape):
    alpha = 2.5
    inp = tu.make_input(dtype, inp_shape, _UNIT_RANGE).requires_grad_()
    other = tu.make_input(dtype, other_shape, _UNIT_RANGE).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other, alpha)
    res_out = flag_gems._test_serialization_subcmul(inp, other, alpha)
    tu.assert_result_close(res_out, ref_out)

    # The reference may live on another device (--ref cpu), so each side gets its own
    # upstream gradient; equal values keep the two gradients comparable.
    upstream = tu.make_input(dtype, tuple(res_out.shape), _UNIT_RANGE)
    ref_upstream = tu.to_reference(upstream)
    ref_grads = torch.autograd.grad(
        ref_out, [ref_inp, ref_other], grad_outputs=ref_upstream
    )
    res_grads = torch.autograd.grad(res_out, [inp, other], grad_outputs=upstream)

    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("dtype", tu.selected_cases(_COMPLEX_DTYPES, quick=[]))
@pytest.mark.parametrize("alpha", [2.5, 1 + 2j])
@pytest.mark.parametrize("other_shape", [(2, 3), (3,)])
def test__test_serialization_subcmul_complex_backward(dtype, alpha, other_shape):
    inp = tu.make_input(dtype, (2, 3), _UNIT_RANGE).requires_grad_()
    other = tu.make_input(dtype, other_shape, _UNIT_RANGE).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)
    upstream = tu.make_input(dtype, (2, 3), _UNIT_RANGE)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other, alpha)
    res_out = flag_gems._test_serialization_subcmul(inp, other, alpha)
    tu.assert_result_close(res_out, ref_out)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_inp, ref_other), grad_outputs=tu.to_reference(upstream)
    )
    res_grads = torch.autograd.grad(res_out, (inp, other), grad_outputs=upstream)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


# nan/inf scenarios for every supported floating dtype, from the shared generator.
# Positive special-value rows are default-only: --quick carries no special-value
# smoke case.
_SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario, slot)
        for dtype, scenario in tu.special_value_cases(_FLOAT_DTYPES)
        for slot in ("self", "other")
    ],
    quick=[],
)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("dtype,scenario,slot", _SPECIAL_CASES)
def test__test_serialization_subcmul_special_values(dtype, scenario, slot):
    special = tu.make_special_input(dtype, scenario)
    plain = tu.make_input(dtype, list(special.shape), _UNIT_RANGE)
    inp, other = (special, plain) if slot == "self" else (plain, special)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten._test_serialization_subcmul(ref_inp, ref_other)
    res_out = flag_gems._test_serialization_subcmul(inp, other)

    tu.assert_result_close(res_out, ref_out)


# Negative cases: the native schema rejects these arguments, so the candidate must
# raise as well. Only the candidate's own exception is asserted, every row stays
# collected in the default and the --quick suite, and the parameter rows are plain
# metadata so the invalid argument is built inside the test body.
@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize(
    "shape,other_shape", [((2, 3), (4,)), ((20, 320, 15), (20, 320, 14))]
)
def test__test_serialization_subcmul_rejects_shape_mismatch(shape, other_shape):
    inp = tu.make_input(torch.float32, shape, _UNIT_RANGE)
    other = tu.make_input(torch.float32, other_shape, _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_serialization_subcmul(inp, other)


@pytest.mark.test_serialization_subcmul
def test__test_serialization_subcmul_rejects_bool_self():
    # bool is rejected only in the `self` position; a bool `other` promotes natively
    # and is covered by the mixed-dtype rows instead.
    inp = tu.make_input(torch.bool, (2, 19, 7), _UNIT_RANGE)
    other = tu.make_input(torch.float32, (2, 19, 7), _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_serialization_subcmul(inp, other)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("alpha_kind", ["tensor", "string", "none", "list"])
def test__test_serialization_subcmul_rejects_non_numeric_alpha(alpha_kind):
    # The schema wants a number, so all of these raise natively.
    alpha = {
        "tensor": torch.tensor(2.0),
        "string": "2",
        "none": None,
        "list": [1.0],
    }[alpha_kind]
    inp = tu.make_input(torch.float32, (2, 19, 7), _UNIT_RANGE)
    other = tu.make_input(torch.float32, (2, 19, 7), _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_serialization_subcmul(inp, other, alpha)


@pytest.mark.test_serialization_subcmul
@pytest.mark.parametrize("other_kind", ["scalar", "list"])
def test__test_serialization_subcmul_rejects_non_tensor_other(other_kind):
    other = 2.0 if other_kind == "scalar" else [1.0, 2.0]
    inp = tu.make_input(torch.float32, (2, 19, 7), _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_serialization_subcmul(inp, other)


@pytest.mark.test_serialization_subcmul
def test__test_serialization_subcmul_rejects_non_tensor_self():
    other = tu.make_input(torch.float32, (2, 19, 7), _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_serialization_subcmul(2.0, other)


@pytest.mark.test_serialization_subcmul
def test__test_serialization_subcmul_rejects_missing_operand():
    inp = tu.make_input(torch.float32, (2, 19, 7), _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._test_serialization_subcmul(inp)
