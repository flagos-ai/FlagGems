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

"""Correctness tests for aten::is_neg.

is_neg reads the lazy neg bit recorded by torch._neg_view; the answer is a
Python bool independent of values, rank, layout and dtype.  Setting and reading
the bit is metadata only, so no test materializes a flagged tensor (resolving a
neg view of bool or FP8 has no neg kernel).  The operator has one operand and
its result is not differentiable, so broadcast, tensor/scalar and backward
dimensions do not apply.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_FP64_SUPPORTED = utils.fp64_is_supported

# int8/uint8/FP8/bool carry the flag like any other dtype, so only the dtypes
# that need backend support for the tensor itself are gated.
_DTYPES = (
    tu.REQUIRED_DTYPES
    + [torch.bool, torch.complex64]
    + ([torch.float64, torch.complex128] if _FP64_SUPPORTED else [])
)


_LAYOUT_SHAPES = {
    "base": (4, 3),
    "slice": (5, 3),
    "transpose": (4, 3),
    "expand": (5, 1),
    "narrow": (4, 3),
    "scalar": (),
    "empty": (0, 3),
}


def _layout_view(tensor, layout):
    if layout == "slice":
        return tensor[1:3]
    if layout == "transpose":
        return tensor.transpose(0, 1)
    if layout == "expand":
        return tensor.expand(5, 4)
    if layout == "narrow":
        return tensor.narrow(1, 1, 2)
    return tensor


def _with_flags(tensor, conj, neg_count, conj_first):
    if conj and conj_first:
        tensor = tensor.conj()
    for _ in range(neg_count):
        tensor = torch._neg_view(tensor)
    if conj and not conj_first:
        tensor = tensor.conj()
    return tensor


def _flag_form_pair(dtype, layout, conj, neg_count, conj_first):
    # The reference operand is built independently and gets the same layout and
    # flags, so no flagged tensor ever has to cross devices.
    plain = tu.make_input(dtype, _LAYOUT_SHAPES[layout], ["-1", "1"])
    inp = _with_flags(_layout_view(plain, layout), conj, neg_count, conj_first)
    ref_inp = _with_flags(
        _layout_view(tu.to_reference(plain), layout), conj, neg_count, conj_first
    )
    return inp, ref_inp


# One row per flag/layout composition. conj() only sets the conjugate bit on
# complex dtypes, so the conj rows carry complex64 (and complex128 when the
# backend supports fp64).  These rows are small, so quick keeps all of them.
_FLAG_FORMS = [
    (torch.float32, "base", False, 0, True),
    (torch.float32, "base", False, 1, True),
    (torch.float32, "base", False, 2, True),
    (torch.float32, "slice", False, 1, True),
    (torch.float32, "transpose", False, 1, True),
    (torch.float32, "expand", False, 1, True),
    (torch.float32, "narrow", False, 1, True),
    (torch.float32, "scalar", False, 1, True),
    (torch.float32, "empty", False, 1, True),
    (torch.int8, "empty", False, 1, True),
    (torch.complex64, "base", True, 0, True),
    (torch.complex64, "base", True, 1, True),
    (torch.complex64, "base", True, 1, False),
] + ([(torch.complex128, "base", True, 1, True)] if _FP64_SUPPORTED else [])

# Read-only contract: the flag read must not change stride, storage offset,
# storage identity, lazy flags, grad state or payload.
_STATE_ROWS = [
    (torch.bool, (3, 4), False, False),
    (torch.int64, (5,), False, False),
    (torch.float32, (2, 3), True, False),
    (torch.complex64, (2, 3), True, False),
    (torch.float32, (2, 3), False, True),
]

_SPECIAL_FLOAT_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if dtype != torch.float64 or _FP64_SUPPORTED
]

_SPECIAL_COMPLEX_DTYPES = [torch.complex64] + (
    [torch.complex128] if _FP64_SUPPORTED else []
)

# Positive special values are default-only.  e4m3fn contributes a nan-only row
# because it cannot represent infinity (tu.special_value_cases).
_SPECIAL_ROWS = tu.selected_cases(
    [
        (dtype, scenario, has_flag)
        for dtype, scenario in tu.special_value_cases(_SPECIAL_FLOAT_DTYPES)
        for has_flag in (False, True)
    ]
    + [
        (dtype, scenario, has_flag)
        for dtype in _SPECIAL_COMPLEX_DTYPES
        for scenario in ("nan", "inf", "mixed")
        for has_flag in (False, True)
    ],
    quick=[],
)

_NEGATIVE_ARGS = [1, 1.5, "is_neg", [1, 2]]


@pytest.mark.is_neg
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_neg_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_neg_lazy_neg_view(shape, dtype):
    plain = tu.make_input(dtype, shape, ["-1", "1"])
    inp = torch._neg_view(plain)
    ref_inp = torch._neg_view(tu.to_reference(plain))

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize("dtype,layout,conj,neg_count,conj_first", _FLAG_FORMS)
def test_is_neg_flag_forms(dtype, layout, conj, neg_count, conj_first):
    inp, ref_inp = _flag_form_pair(dtype, layout, conj, neg_count, conj_first)

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    # The bit is a parity and is independent of the conjugate bit.
    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize("dtype,shape,has_flag,requires_grad", _STATE_ROWS)
def test_is_neg_keeps_input_state(dtype, shape, has_flag, requires_grad):
    plain = tu.make_input(dtype, shape, ["-1", "1"])
    if requires_grad:
        plain.requires_grad_(True)
    ref_plain = tu.to_reference(plain)
    inp = torch._neg_view(plain) if has_flag else plain
    ref_inp = torch._neg_view(ref_plain) if has_flag else ref_plain
    plain_before = tu.to_reference(plain.detach())
    stride, offset = inp.stride(), inp.storage_offset()
    storage = inp.untyped_storage().data_ptr()
    conj = inp.is_conj()

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out
    assert inp.stride() == stride
    assert inp.storage_offset() == offset
    assert inp.untyped_storage().data_ptr() == storage
    assert inp.is_conj() == conj
    assert inp.is_neg() == has_flag
    assert inp.requires_grad == requires_grad
    # The payload snapshot comes from the unflagged tensor, so comparing it does
    # not have to materialize the negation view.
    tu.assert_result_equal(plain, plain_before)


@pytest.mark.is_neg
@pytest.mark.parametrize("dtype,scenario,has_flag", _SPECIAL_ROWS)
def test_is_neg_special_values(dtype, scenario, has_flag):
    plain = tu.make_special_input(dtype, scenario)
    ref_plain = tu.to_reference(plain)
    inp = torch._neg_view(plain) if has_flag else plain
    ref_inp = torch._neg_view(ref_plain) if has_flag else ref_plain

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
def test_is_neg_undefined_tensor():
    # An undefined tensor is a valid `self` and is not negative.
    ref_out = torch.ops.aten.is_neg(None)
    res_out = flag_gems.is_neg(None)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize("bad_arg", _NEGATIVE_ARGS)
def test_is_neg_rejects_non_tensor(bad_arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_neg(bad_arg)


@pytest.mark.is_neg
def test_is_neg_requires_input():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_neg()
