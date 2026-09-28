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

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::type_as(self, other) casts ``self`` to ``other``'s dtype and places the
# result on ``other``'s device; ``other``'s shape, values and storage are
# ignored (both facts are measured natively, see the device test below). Nothing
# broadcasts here, so no workload uses a broadcast pair.
#
# (source dtype of ``self``, target dtype of ``other``) pairs of the value grid.
# All nine required dtypes occur, as source and as target. A float source is not
# paired with an integer target in this grid, and a source wider than fp8 is not
# paired with an fp8 target: over the spec's five ranges those inputs leave the
# target's range, where the conversion stops being a defined value map
# (measured natively: float32 300 -> int8 44, float32 1e30 -> int8 -1). Both
# families are covered by the fixture tests below, which use values that stay
# representable in the target after the source dtype rounds; the out-of-range
# remainder is a recorded gap, not claimed coverage.
_CAST_PAIRS_ALL = [
    (torch.int8, torch.int32),
    (torch.uint8, torch.int64),
    (torch.int32, torch.int8),
    (torch.int64, torch.int32),
    (torch.bool, torch.uint8),
    (torch.int8, torch.float8_e4m3fn),
    (torch.uint8, torch.float8_e5m2),
    (torch.float8_e4m3fn, torch.float32),
    (torch.float8_e5m2, torch.float16),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.float16),
    (torch.float32, torch.bfloat16),
    (torch.float64, torch.float32),
    (torch.float32, torch.float64),
]

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _dtype_supported(dtype):
    """Keep optional dtypes only where a static runtime flag reports them.

    The flags come from ``flag_gems.runtime.device``; nothing is probed while
    collecting cases.
    """
    if dtype == torch.float64:
        return utils.fp64_is_supported
    if dtype == torch.bfloat16:
        return utils.bf16_is_supported
    if dtype == torch.int64:
        return utils.int64_is_supported
    if dtype in _FP8_DTYPES:
        return utils.fp8_is_supported
    return True


def _pair_supported(pair):
    return all(_dtype_supported(dtype) for dtype in pair)


_CAST_PAIRS = [pair for pair in _CAST_PAIRS_ALL if _pair_supported(pair)]

# Every required dtype appears as the source of a self-cast here. A positive
# supplement, so the whole family is default-only and the quick smoke run keeps
# the value grid and the negative boundaries only.
_IDENTITY_DTYPES_ALL = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
]
_IDENTITY_DTYPES = tu.selected_cases(
    [dtype for dtype in _IDENTITY_DTYPES_ALL if _dtype_supported(dtype)],
    quick=[],
)

# Layouts of ``self``: a 0-dim view, a contiguous slice, an offset slice, a
# strided non-contiguous view and a transposed offset slice. The same matrix
# drives the identity cases and the cast cases, both default-only supplements.
_VIEW_ROWS_ALL = [
    ((12, 16), (slice(None), slice(None)), False),
    ((12, 16), (slice(2, 10), slice(1, 15, 3)), False),
    ((12, 16), (slice(None), slice(None, None, 2)), False),
    ((12, 16), (slice(4, 12), slice(None)), True),
    ((12, 16), (0, 0), False),
]
_VIEW_ROWS = tu.selected_cases(_VIEW_ROWS_ALL, quick=[])
_VIEW_PAIRS_ALL = [
    (torch.float32, torch.float16),
    (torch.float16, torch.float32),
    (torch.float32, torch.float32),
]
_VIEW_PAIRS = tu.selected_cases(
    [pair for pair in _VIEW_PAIRS_ALL if _pair_supported(pair)],
    quick=[],
)

# Shape of the ``other`` operand; none of these may influence the result.
_OTHER_SHAPES = tu.selected_cases([(), (1,), (5, 7, 9), (0,)], quick=[])
_OTHER_SHAPE_TARGETS = [
    dtype for dtype in (torch.float16, torch.bfloat16) if _dtype_supported(dtype)
]
# ``other`` as a non-contiguous, offset view of large values.
_OTHER_VIEW_ROWS = tu.selected_cases(
    [
        (target_dtype, (4, 8), (slice(1, None), slice(None, None, 3)))
        for target_dtype in _OTHER_SHAPE_TARGETS
    ],
    quick=[],
)

_SPECIAL_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        *_FP8_DTYPES,
    )
    if _dtype_supported(dtype)
]
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])

# float -> int fixtures: fractional values (truncated toward zero, so 1.5 and
# 2.5 are not rounded), signed zeros, and values close to a target boundary,
# all chosen to stay inside the target's range after the source dtype rounds.
_FLOAT_TO_INT_VALUES = {
    torch.int8: (
        0.0,
        -0.0,
        0.4,
        -0.4,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        126.5,
        -126.5,
        127.0,
        -127.0,
    ),
    torch.uint8: (0.0, -0.0, 0.4, 0.5, 1.5, 2.5, 3.5, 126.5, 127.5, 254.5, 254.9),
    torch.int32: (
        0.0,
        -0.0,
        0.4,
        -0.4,
        1.5,
        -1.5,
        2.5,
        -2.5,
        1023.5,
        -1023.5,
        1023.9,
        -1023.9,
    ),
    torch.int64: (
        0.0,
        -0.0,
        0.4,
        -0.4,
        1.5,
        -1.5,
        2.5,
        -2.5,
        1023.5,
        -1023.5,
        1023.9,
        -1023.9,
    ),
}
_FLOAT_TO_INT_ROWS = tu.selected_cases(
    [
        (source, target, values)
        for target, values in _FLOAT_TO_INT_VALUES.items()
        for source in (torch.float16, torch.bfloat16, torch.float32)
        if _pair_supported((source, target))
    ],
    quick=[],
)

# Wider float -> fp8 fixtures: exact values, subnormal quantization, a value
# below the smallest subnormal (underflow to zero), the largest finite values
# and the overflow region, where float8_e4m3fn has no infinity and yields NaN
# while float8_e5m2 yields +-inf. All of this is asserted against the native
# result, at zero tolerance with matching NaNs.
_FLOAT_TO_FP8_VALUES = (
    0.0,
    -0.0,
    0.5,
    1.5,
    2.5,
    0.001953125,
    0.001,
    1e-9,
    448.0,
    464.0,
    -464.0,
    57344.0,
    65504.0,
    float("inf"),
    float("-inf"),
    float("nan"),
)
_FLOAT_TO_FP8_ROWS = tu.selected_cases(
    [
        (source, target)
        for target in _FP8_DTYPES
        for source in (torch.float16, torch.bfloat16, torch.float32)
        if _pair_supported((source, target))
    ],
    quick=[],
)

# NaN/Inf scenarios per source dtype into each fp8 target, keeping the
# nan-only, inf-only and mixed inputs distinguishable.
_FP8_TARGET_CASES = tu.selected_cases(
    [
        (source, target, scenario)
        for target in _FP8_DTYPES
        for source in (torch.float16, torch.bfloat16, torch.float32)
        for scenario in ("nan", "inf", "mixed")
        if _pair_supported((source, target))
    ],
    quick=[],
)

_BACKWARD_PAIRS_ALL = [
    (torch.float32, torch.float16),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.float16),
]
_BACKWARD_PAIRS = tu.selected_cases(
    [pair for pair in _BACKWARD_PAIRS_ALL if _pair_supported(pair)], quick=[]
)
_BACKWARD_SHAPES = tu.selected_cases([(20, 320, 15), (4, 8, 16)], quick=[])

# Empty positives are default-only as well, so the quick smoke run stays on the
# value grid and the negative boundaries.
_EMPTY_SHAPES = tu.selected_cases([(0,), (0, 3)], quick=[])
_EMPTY_PAIRS = [
    pair
    for pair in ((torch.float32, torch.float16), (torch.int32, torch.int64))
    if _pair_supported(pair)
]

# ``other`` supplies the result device as well as its dtype.
_DEVICE_CASES = tu.selected_cases(
    [("other_on_host", False), ("other_on_device", True)],
    quick=[],
)

_OPERAND_ERRORS = (TypeError, RuntimeError)


def _make_view(base, slices, transpose):
    view = base[slices]
    return view.t() if transpose else view


def _assert_operands_unchanged(inp, ref_inp, other, ref_other):
    """A cast reads both operands and must not modify either."""
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("source_dtype,target_dtype", _CAST_PAIRS)
def test_type_as_value_ranges(shape, value_range, source_dtype, target_dtype):
    inp = tu.make_input(source_dtype, shape, value_range)
    other = tu.make_input(target_dtype, (1,), value_range)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    # A cast is a deterministic elementwise map, so the candidate output has to
    # match the native result exactly (no tolerance, matching NaNs allowed).
    tu.assert_result_equal(res_out, ref_out)
    _assert_operands_unchanged(inp, ref_inp, other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("source_dtype,target_dtype,values", _FLOAT_TO_INT_ROWS)
def test_type_as_float_to_int_truncates_toward_zero(source_dtype, target_dtype, values):
    inp = torch.tensor(values, dtype=source_dtype, device=flag_gems.device)
    other = tu.make_input(target_dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operands_unchanged(inp, ref_inp, other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("source_dtype,target_dtype", _FLOAT_TO_FP8_ROWS)
def test_type_as_float_to_fp8_rounds_and_overflows(source_dtype, target_dtype):
    inp = torch.tensor(
        _FLOAT_TO_FP8_VALUES, dtype=source_dtype, device=flag_gems.device
    )
    other = tu.make_input(target_dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    tu.assert_result_equal(res_out, ref_out)
    _assert_operands_unchanged(inp, ref_inp, other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("base_shape,slices,transpose", _VIEW_ROWS)
@pytest.mark.parametrize("dtype", _IDENTITY_DTYPES)
def test_type_as_same_dtype_returns_self(base_shape, slices, transpose, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    inp = _make_view(base, slices, transpose)
    other = tu.make_input(dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    # Native returns ``self`` itself for an equal dtype and device, so the
    # candidate has to hand back the very same tensor, view metadata included.
    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.type_as
@pytest.mark.parametrize("dtype", _IDENTITY_DTYPES)
def test_type_as_same_dtype_empty_returns_self(dtype):
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    other = tu.make_input(dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.type_as
@pytest.mark.parametrize("other_shape", _OTHER_SHAPES)
@pytest.mark.parametrize("target_dtype", _OTHER_SHAPE_TARGETS)
def test_type_as_ignores_other_shape(other_shape, target_dtype):
    inp = tu.make_input(torch.float32, (20, 320, 15), ["-1", "1"])
    other = tu.make_input(target_dtype, other_shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    assert res_out.shape == inp.shape
    tu.assert_result_equal(res_out, ref_out)
    _assert_operands_unchanged(inp, ref_inp, other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("target_dtype,base_shape,slices", _OTHER_VIEW_ROWS)
def test_type_as_ignores_other_view(target_dtype, base_shape, slices):
    inp = tu.make_input(torch.float32, (20, 320, 15), ["-1", "1"])
    base = tu.make_input(target_dtype, base_shape, ["0", "max"])
    other = base[slices]
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    # Neither ``other``'s values nor its offset, non-contiguous storage may
    # reach the result; only its dtype and device do.
    assert res_out.device == other.device
    tu.assert_result_equal(res_out, ref_out)
    _assert_operands_unchanged(inp, ref_inp, other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("base_shape,slices,transpose", _VIEW_ROWS)
@pytest.mark.parametrize("source_dtype,target_dtype", _VIEW_PAIRS)
def test_type_as_strided_input(
    base_shape, slices, transpose, source_dtype, target_dtype
):
    base = tu.make_input(source_dtype, base_shape, ["-1", "1"])
    inp = _make_view(base, slices, transpose)
    other = tu.make_input(target_dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    # The cast reads the view's elements, so the values are the asserted
    # property; the result may be compacted into a contiguous tensor.
    assert res_out.shape == inp.shape
    tu.assert_result_equal(res_out, ref_out)
    _assert_operands_unchanged(inp, ref_inp, other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_type_as_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    other = tu.make_input(torch.float32, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    # NaN/Inf survive a widening float cast; matching NaNs are part of the
    # reference semantics.
    tu.assert_result_equal(res_out, ref_out)
    _assert_operands_unchanged(inp, ref_inp, other, ref_other)


@pytest.mark.type_as
@pytest.mark.parametrize("source_dtype,target_dtype,scenario", _FP8_TARGET_CASES)
def test_type_as_special_values_into_fp8(source_dtype, target_dtype, scenario):
    inp = tu.make_special_input(source_dtype, scenario)
    other = tu.make_input(target_dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    # float8_e4m3fn has no infinity, so an overflowing input becomes NaN there
    # while float8_e5m2 yields +-inf; both are the native result, matched at
    # zero tolerance with matching NaNs.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.type_as
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("source_dtype,target_dtype", _BACKWARD_PAIRS)
def test_type_as_backward(shape, source_dtype, target_dtype):
    inp = tu.make_input(source_dtype, shape, ["-1", "1"]).requires_grad_()
    other = tu.make_input(target_dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp).requires_grad_()
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)
    tu.assert_result_equal(res_out, ref_out)

    # One independent upstream tensor feeds both graphs.
    upstream = tu.make_input(ref_out.dtype, shape, ["-1", "1"])
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )[0]
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    # The cast is linear, so the gradient is exactly the upstream gradient cast
    # back to the input dtype: an exact comparison, no tolerance.
    assert res_grad.shape == inp.shape
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.type_as
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("source_dtype,target_dtype", _EMPTY_PAIRS)
def test_type_as_empty_input(shape, source_dtype, target_dtype):
    inp = tu.make_input(source_dtype, shape, ["-1", "1"])
    other = tu.make_input(target_dtype, (1,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    assert res_out.shape == shape
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.type_as
@pytest.mark.parametrize("case", _DEVICE_CASES)
def test_type_as_follows_other_device(case):
    # ``other`` also carries the result device (probed natively: an accelerator
    # ``self`` with a host ``other`` lands on the host and a host ``self`` with
    # an accelerator ``other`` lands on the accelerator, without error). The
    # host operand is the point of this workload, so it is placed explicitly and
    # the native call runs on clones at that same placement rather than on
    # operands relocated to the configured reference device.
    _, other_on_device = case
    host = torch.device("cpu")

    inp = tu.make_input(torch.float32, (8, 16), ["-1", "1"])
    other = tu.make_input(torch.float16, (1,), ["-1", "1"])
    if other_on_device:
        inp = inp.to(host)
    else:
        other = other.to(host)

    ref_inp = inp.detach().clone()
    ref_other = other.detach().clone()
    ref_out = torch.ops.aten.type_as(ref_inp, ref_other)
    res_out = flag_gems.type_as(inp, other)

    # The result follows ``other``'s dtype and device in every case, so the
    # placement is asserted against ``other`` itself and never against the
    # reference result.
    assert res_out.dtype == other.dtype
    assert res_out.device == other.device

    # Only the value comparison follows the configured reference placement.
    tu.assert_result_equal(res_out, tu.to_reference(ref_out))


@pytest.mark.type_as
def test_type_as_rejects_non_tensor_operands():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    other = tu.make_input(torch.float16, (4,), ["-1", "1"])

    with pytest.raises(_OPERAND_ERRORS):
        flag_gems.type_as(1.0, other)
    with pytest.raises(_OPERAND_ERRORS):
        flag_gems.type_as(inp, 1.0)


@pytest.mark.type_as
def test_type_as_rejects_missing_operand():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises(_OPERAND_ERRORS):
        flag_gems.type_as(inp)
