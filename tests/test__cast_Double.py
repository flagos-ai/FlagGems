# Copyright 2025 The FlagGems Authors.
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

"""Correctness tests for ``torch.ops.aten._cast_Double``.

Every positive workload returns float64, so each of them is selected out
statically when the backend capability flag reports no fp64 support. For a
float64 input the native operator is a pass-through that returns the caller's
own tensor object, so those layouts check object identity, not only values.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

FP64_IS_SUPPORTED = utils.fp64_is_supported


def _build_dtypes():
    """Build the tested dtype list from static capability flags only."""
    if not FP64_IS_SUPPORTED:
        return []
    dtypes = [torch.int8, torch.uint8]
    if utils.fp8_is_supported:
        dtypes += [torch.float8_e4m3fn, torch.float8_e5m2]
    dtypes.append(torch.float32)
    if utils.bf16_is_supported:
        dtypes.append(torch.bfloat16)
    dtypes += [torch.float16, torch.int32]
    if utils.int64_is_supported:
        dtypes.append(torch.int64)
    dtypes += [torch.float64, torch.bool, torch.complex64, torch.complex128]
    return dtypes


CAST_DTYPES = _build_dtypes()
SPECIAL_DTYPES = [dtype for dtype in CAST_DTYPES if dtype.is_floating_point]


def _layout_input(layout, dtype):
    """Build an input whose storage layout exercises the cast or pass-through."""
    device = flag_gems.device
    if layout == "contiguous":
        return torch.randn(8, 16, dtype=dtype, device=device)
    if layout == "scalar":
        return torch.randn((), dtype=dtype, device=device)
    if layout == "empty":
        return torch.randn(4, 0, dtype=dtype, device=device)
    if layout == "zero-extent":
        return torch.randn(0, 3, dtype=dtype, device=device)
    if layout == "offset":
        return torch.randn(32, dtype=dtype, device=device)[5:13]
    if layout == "stepped":
        return torch.randn(8, 16, dtype=dtype, device=device)[1:7, 1:15:3]
    if layout == "transpose":
        return torch.randn(6, 7, dtype=dtype, device=device).t()
    if layout == "channels-last":
        base = torch.randn(2, 3, 4, 5, dtype=dtype, device=device)
        return base.to(memory_format=torch.channels_last)
    raise AssertionError(f"unknown layout: {layout}")


@pytest.mark.cast_Double
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", CAST_DTYPES)
def test__cast_Double(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)

    # Widening into float64 is exact for the floating and byte inputs, while
    # int64 values above 2**53 round when they are converted, so the native
    # cast is the oracle for every row instead of a Python-side literal.
    tu.assert_result_equal(res_out, ref_out)


_IDENTITY_LAYOUTS = (
    tu.selected_cases(
        [
            "contiguous",
            "scalar",
            "empty",
            "zero-extent",
            "offset",
            "stepped",
            "transpose",
            "channels-last",
        ],
        quick=[],
    )
    if FP64_IS_SUPPORTED
    else []
)


@pytest.mark.cast_Double
@pytest.mark.parametrize("layout", _IDENTITY_LAYOUTS)
def test__cast_Double_float64_is_a_pass_through(layout):
    inp = _layout_input(layout, torch.float64)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)

    # The native contract hands back the caller's own tensor for a float64
    # input, so the candidate has to reproduce that same-object relation
    # instead of allocating an equal clone. The empty layouts are included
    # because no element value can tell an equal clone from the original there.
    assert (res_out is inp) == (ref_out is ref_inp)
    tu.assert_result_equal(res_out, ref_out)


def _strided_cases():
    if not FP64_IS_SUPPORTED:
        return []
    cases = [
        ("stepped", torch.float32),
        ("offset", torch.float16),
        ("channels-last", torch.float32),
    ]
    if utils.bf16_is_supported:
        cases.append(("transpose", torch.bfloat16))
    return tu.selected_cases(cases, quick=[])


@pytest.mark.cast_Double
@pytest.mark.parametrize("layout,dtype", _strided_cases())
def test__cast_Double_non_contiguous(layout, dtype):
    inp = _layout_input(layout, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)

    tu.assert_result_equal(res_out, ref_out)


def _backward_input(layout, dtype):
    """Source tensor of one backward row, in the originally tested geometry."""
    device = flag_gems.device
    if layout == "transpose":
        inp = torch.randn(512, 1024, dtype=dtype, device=device).t()
    elif layout == "channels-last":
        inp = torch.randn(1, 3, 4, 5, dtype=dtype, device=device).to(
            memory_format=torch.channels_last
        )
    elif layout == "offset":
        inp = torch.randn(64, dtype=dtype, device=device)[5:37]
    elif layout == "stepped":
        inp = torch.randn(16, 16, dtype=dtype, device=device)[1:15:3, 1:16:5]
    else:
        inp = torch.randn(1024, 1024, dtype=dtype, device=device)
    return inp.detach().requires_grad_(True)


def _backward_dtypes():
    if not FP64_IS_SUPPORTED:
        return []
    dtypes = [torch.float32, torch.float16]
    if utils.bf16_is_supported:
        dtypes.append(torch.bfloat16)
    dtypes.append(torch.float64)
    return dtypes


def _backward_cases():
    dtypes = _backward_dtypes()
    if not dtypes:
        # Every row builds a float64 upstream and calls an operator whose
        # result is always float64, so no row below is valid without fp64
        # support; the appended strided rows are gated the same way.
        return []
    cases = [
        (layout, dtype)
        for dtype in dtypes
        for layout in ("contiguous", "transpose", "channels-last")
    ]
    # Small strided rows keep the offset/step source paths covered too.
    cases += [("offset", torch.float32), ("stepped", torch.float32)]
    return tu.selected_cases(cases, quick=[])


@pytest.mark.cast_Double
@pytest.mark.parametrize("layout,dtype", _backward_cases())
def test__cast_Double_backward(layout, dtype):
    inp = _backward_input(layout, dtype)
    # A non-uniform upstream gradient is what makes a constant-gradient or a
    # value-reordering candidate fail here. The reference may run on another
    # device, so its own upstream follows it while the candidate keeps this one.
    upstream = torch.randn(inp.shape, dtype=torch.float64, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)
    tu.assert_result_equal(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)

    # A cast reduces nothing: the gradient is the upstream gradient cast back
    # into the input dtype, so both sides are compared exactly.
    tu.assert_result_equal(res_grad, ref_grad)


# Values that stress the widening: signed zero, the smallest normal and
# subnormal, the largest finite magnitude, the 2**53 neighbourhood and
# neighbours one ULP apart. Every row is built directly in its own source
# dtype, so the native cast is the only oracle. A Python float literal cannot
# hold 2**53 + 1, so the float64 row keeps the distinguishable pair
# 2**53 / 2**53 + 2.0 and the odd value is exercised by the int64 row below.
_BOUNDARY_VALUES = {
    torch.float32: [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0**-149,
        2.0**-126,
        3.4028234663852886e38,
        -3.4028234663852886e38,
        1.0 + 2.0**-23,
        2.0**24,
        float(2**24 + 1),
        float(2**40 + 1),
        float(-(2**40) - 1),
        float(2**53 - 1),
        float(-(2**53) + 1),
    ],
    torch.float16: [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0**-24,
        2.0**-14,
        65504.0,
        -65504.0,
        1.0 + 2.0**-10,
    ],
    torch.bfloat16: [
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.3895313892515355e38,
        1.0 + 2.0**-8,
        256.0,
    ],
    torch.float64: [
        0.0,
        -0.0,
        1.0,
        -1.0,
        5e-324,
        -5e-324,
        2.2250738585072014e-308,
        1.7976931348623157e308,
        -1.7976931348623157e308,
        1.0 + 2.0**-52,
        1.0 - 2.0**-53,
        2.0**24 + 1.0,
        2.0**40 + 1.0,
        -(2.0**40) - 1.0,
        2.0**53 - 1.0,
        2.0**53,
        2.0**53 + 2.0,
    ],
}


def _boundary_cases():
    if not FP64_IS_SUPPORTED:
        return []
    dtypes = [torch.float32, torch.float16]
    if utils.bf16_is_supported:
        dtypes.append(torch.bfloat16)
    dtypes.append(torch.float64)
    return tu.selected_cases(dtypes, quick=[])


@pytest.mark.cast_Double
@pytest.mark.parametrize("dtype", _boundary_cases())
def test__cast_Double_boundary_values(dtype):
    values = _BOUNDARY_VALUES[dtype]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)

    tu.assert_result_equal(res_out, ref_out)


# The int64 row keeps exact source integers: 2**63 - 1, 2**53 + 1 and
# -(2**53) - 1 have no float64 representation, so they round on conversion and
# the native cast supplies the rounded expectation. Collected only when the
# static int64 flag reports a kernel for this backend.
_INT64_BOUNDARY_VALUES = [2**63 - 1, -(2**63), 2**53 + 1, -(2**53) - 1, 2**62]
_INT64_BOUNDARY_CASES = (
    tu.selected_cases([torch.int64], quick=[]) if torch.int64 in CAST_DTYPES else []
)


@pytest.mark.cast_Double
@pytest.mark.parametrize("dtype", _INT64_BOUNDARY_CASES)
def test__cast_Double_int64_precision_boundaries(dtype):
    inp = torch.tensor(_INT64_BOUNDARY_VALUES, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)

    tu.assert_result_equal(res_out, ref_out)


_SPECIAL_CASES = tu.selected_cases(
    list(tu.special_value_cases(SPECIAL_DTYPES)), quick=[]
)


@pytest.mark.cast_Double
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__cast_Double_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)

    # NaN and inf survive the widening, so matching NaNs are expected here.
    tu.assert_result_equal(res_out, ref_out)


# Explicit-flag call forms on the original fp16 (1024, 1024) source. The
# omitted-argument form is already one of the main grid's dtype x shape x range
# rows, so it is not repeated here.
_NON_BLOCKING_CASES = (
    tu.selected_cases(
        [(flag, torch.float16, (1024, 1024), ["-1", "1"]) for flag in (False, True)],
        quick=[],
    )
    if FP64_IS_SUPPORTED
    else []
)


@pytest.mark.cast_Double
@pytest.mark.parametrize("non_blocking,dtype,shape,value_range", _NON_BLOCKING_CASES)
def test__cast_Double_non_blocking_positional(non_blocking, dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp, non_blocking)
    res_out = flag_gems._cast_Double(inp, non_blocking)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Double
@pytest.mark.parametrize("non_blocking,dtype,shape,value_range", _NON_BLOCKING_CASES)
def test__cast_Double_non_blocking_keyword(non_blocking, dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp, non_blocking=non_blocking)
    res_out = flag_gems._cast_Double(inp, non_blocking=non_blocking)

    tu.assert_result_equal(res_out, ref_out)


# int32 is always present, int64 only behind its static capability flag, so no
# row depends on a kernel this backend may not have.
_INTEGER_CASES = tu.selected_cases(
    [dtype for dtype in (torch.int32, torch.int64) if dtype in CAST_DTYPES],
    quick=[],
)


@pytest.mark.cast_Double
@pytest.mark.parametrize("dtype", _INTEGER_CASES)
def test__cast_Double_integer_input_is_not_differentiable(dtype):
    inp = torch.ones(4, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Double(ref_inp)
    res_out = flag_gems._cast_Double(inp)

    tu.assert_result_equal(res_out, ref_out)
    # An integer input carries no autograd relation, so the cast result must not
    # claim one; together with the forward result this rejects a candidate that
    # quietly returns a differentiable floating tensor for an integer input.
    assert res_out.requires_grad is False


@pytest.mark.cast_Double
def test__cast_Double_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cast_Double(1.5)


@pytest.mark.cast_Double
def test__cast_Double_rejects_missing_argument():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cast_Double()


@pytest.mark.cast_Double
@pytest.mark.parametrize("non_blocking", ["yes", ["a"], {"a": 1}])
def test__cast_Double_rejects_invalid_non_blocking(non_blocking):
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cast_Double(inp, non_blocking)


@pytest.mark.cast_Double
@pytest.mark.parametrize("keyword", ["dtype", "out", "unknown_flag"])
def test__cast_Double_rejects_unknown_keyword(keyword):
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cast_Double(inp, **{keyword: 1})
