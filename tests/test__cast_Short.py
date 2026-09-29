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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor casts to
# int16.  Coverage decisions, each settled by a probe on the active device:
# * ``torch.ops.aten._cast_Short`` has no ``out`` overload (``.out`` is not an
#   attribute), so every workload calls ``.default`` through the single public
#   ``flag_gems._cast_Short`` entry point.
# * The result carries no autograd history (``op(x).grad_fn is None`` and
#   ``autograd.grad`` refuses the input), so no backward workload applies.  The
#   operator is unary, so the two-operand broadcast baseline does not apply;
#   the second argument is a bool flag, not a tensor/scalar operand.
# * Out-of-range and non-finite float inputs are handled per device rather than
#   by a universal rule: repeated probes on the assigned CUDA device gave
#   +-inf -> -1/0, NaN -> 0 and |x| > 32768.0 -> the int16 extremes, and a CPU
#   reference gave different values for the same inputs.  Cases are therefore
#   compared exactly against ``torch.ops.aten._cast_Short`` on the inputs each
#   test builds, with the shared helper handling any reference-device transfer;
#   no clamping, masking or tolerance widening is applied anywhere.
# * Every probed dtype is accepted natively and the operator accepts any rank,
#   so the negative family is the call-schema contract: non-Tensor input, an
#   invalid ``non_blocking`` type, an unknown keyword and wrong arity.

# Dtypes are gated by the framework's static per-device capability flags, never
# by probing tensors here.  All of these are constructible on the current
# target; the gates only keep the list portable to backends that lack a dtype.
_OPTIONAL_DTYPES = [
    (torch.float8_e4m3fn, utils.fp8_is_supported),
    (torch.float8_e5m2, utils.fp8_is_supported),
    (torch.bfloat16, utils.bf16_is_supported),
    (torch.int64, utils.int64_is_supported),
    (torch.float64, utils.fp64_is_supported),
]

CAST_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.float16,
        torch.int32,
    ]
    + [dtype for dtype, supported in _OPTIONAL_DTYPES if supported]
    + [
        torch.int16,
        torch.bool,
        torch.complex64,
    ]
)

# ``non_blocking`` is a bool parameter and needs both True and False; the
# schema-default call (argument omitted) is already exercised by the main grid,
# which invokes the operator with one argument only.
_NON_BLOCKING_CASES = tu.selected_cases([True, False], quick=[])
_PARAM_DTYPES = [torch.int32] + ([torch.bfloat16] if utils.bf16_is_supported else [])

_EMPTY_SHAPES = [(0,), (0, 8), (3, 0, 4)]

_EMPTY_DTYPES = [
    torch.float32,
    torch.int8,
    torch.int32,
    torch.int16,
    torch.bool,
    torch.complex64,
]

# Fractional values truncate to distinct non-zero int16 values; the int16 limit
# block covers the ends of the target range.  e4m3fn cannot represent the limit
# values (the conversion produces NaN), so it keeps only the fractional cycle;
# the limits are covered by the float32/float16/bfloat16 boundary rows below.
_FRACTION_CYCLE = [-2.5, 1.5, -1.5, 2.75, -3.25, 0.5, -0.5, 3.5]
_INT16_EDGE_CYCLE = [32767.0, -32768.0, 32768.0, -32769.0, 65536.0, -65536.0]

# Integer cycles are chosen so that the cast is non-zero and position varying;
# the int32/int64 rows contain the values that wrap when narrowed to int16.
_INT_CYCLE = {
    torch.int8: [-128, 127, -100, 100, 5, -5, 1, -1],
    torch.uint8: [0, 255, 128, 200, 7, 1, 42, 250],
    torch.int16: [-32768, 32767, -1000, 1000, 5, -5, 1, -1],
    torch.int32: [32768, -32769, 65535, 65536, 65537, -65536, 7, -7],
    torch.int64: [2**31 - 1, -(2**31), 2**40, -(2**40), 32768, -32769, 9, -9],
    torch.bool: [True, False, True, True, False, True, False, True],
}

# Explicit int16-range boundaries, including values that round, saturate or wrap
# when narrowed from a wider integer type.  The literals below are the
# mathematical boundaries; float16 and bfloat16 cannot store 32767.0 or
# -32769.0, so their tensors hold 32768.0 / -32768.0 after construction (probed)
# and the correctness check is still the exact native result for whatever was
# actually stored.
_BOUNDARY_ROWS = [
    (torch.float32, [32767.0, -32768.0, 32768.0, -32769.0, 65536.0, -65536.0]),
    (torch.float16, [32767.0, -32768.0, 32768.0, -32769.0]),
    (torch.int8, [-128, 127, 0, -1, 1]),
    (torch.int16, [-32768, 32767, 0, -1, 1]),
    (torch.int32, [32768, -32769, 65535, 65536, -65536]),
    (torch.bool, [True, False]),
]
if utils.fp8_is_supported:
    # e4m3fn is excluded: it cannot represent these magnitudes at all (its
    # conversion of an out-of-range value yields NaN), so only e5m2 carries the
    # FP8 limit row.
    _BOUNDARY_ROWS.append((torch.float8_e5m2, [32767.0, -32768.0, 32768.0, -32769.0]))
if utils.bf16_is_supported:
    _BOUNDARY_ROWS.append((torch.bfloat16, [32767.0, -32768.0, 32768.0, -32769.0]))
if utils.fp64_is_supported:
    _BOUNDARY_ROWS.append((torch.float64, [32767.0, -32768.0, 32768.0, -32769.0]))
if utils.int64_is_supported:
    _BOUNDARY_ROWS.append((torch.int64, [2**31 - 1, -(2**31), 32768, -32769]))

_LAYOUTS = ("slice", "transpose", "offset", "expanded")


def _tile(cycle, shape):
    """Repeat a deterministic value cycle over ``shape`` on the active device."""
    numel = math.prod(shape) if len(shape) else 1
    if numel <= cycle.numel():
        flat = cycle[:numel]
    else:
        repeats = -(-numel // cycle.numel())
        flat = cycle.repeat(repeats)[:numel]
    return flat.reshape(shape)


def _defined_payload(dtype, shape):
    """Deterministic, position-varying input whose cast is non-zero.

    A candidate that writes zeros, drops values or ignores strides cannot match
    the reference on this payload, unlike a uniform range whose truncation is
    constant.
    """
    if dtype.is_complex:
        real = torch.tensor(
            [3.7, -4.2, 0.9, 15.25, -0.75, 8.5],
            device=flag_gems.device,
            dtype=torch.float32,
        )
        imag = torch.tensor(
            [9.9, -15.0, 1.5, -2.25, 6.5, -0.5],
            device=flag_gems.device,
            dtype=torch.float32,
        )
        cycle = torch.complex(real, imag)
    elif dtype in _INT_CYCLE:
        cycle = torch.tensor(_INT_CYCLE[dtype], device=flag_gems.device, dtype=dtype)
    else:
        values = _FRACTION_CYCLE
        if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            values = _INT16_EDGE_CYCLE + values
        cycle = torch.tensor(values, device=flag_gems.device, dtype=torch.float32).to(
            dtype
        )
    return _tile(cycle, shape)


def _apply_layout(tensor, layout):
    """Return a view of ``tensor`` with the requested strides/offset."""
    if layout == "slice":
        # Non-zero storage offset and a non-contiguous last dimension.
        return tensor[..., 1:][..., ::2]
    if layout == "transpose":
        return tensor.transpose(0, 1)
    if layout == "offset":
        # Contiguous storage with a non-zero storage offset.
        return tensor.reshape(-1)[3:]
    if layout == "expanded":
        # Lazy broadcast view (stride 0 in the leading dimension).
        return tensor[0:1].expand(tensor.shape[0], *tensor.shape[1:])
    return tensor


_DEFINED_CASES = tu.selected_cases(
    [(shape, "contiguous") for shape in tu.selected_shapes()]
    + [
        (shape, layout)
        for shape in tu.selected_shapes()
        if len(shape) >= 2
        for layout in _LAYOUTS
    ],
    quick=[],
)

_ALIAS_CASES = tu.selected_cases(
    [(shape, "contiguous") for shape in tu.selected_shapes()]
    + [
        (shape, layout)
        for shape in tu.selected_shapes()
        if len(shape) >= 2
        for layout in _LAYOUTS
    ]
    + [(shape, "contiguous") for shape in _EMPTY_SHAPES],
    quick=[],
)

# Positive extras (defined values, views, boundaries, empty inputs, special
# values and parameter sweeps) are default-only; quick keeps the main grid and
# the negative cases.
_BOUNDARY_CASES = tu.selected_cases(_BOUNDARY_ROWS, quick=[])
_EMPTY_CASES = tu.selected_cases(
    [(dtype, shape) for dtype in _EMPTY_DTYPES for shape in _EMPTY_SHAPES],
    quick=[],
)
_CONJ_CASES = tu.selected_cases([(4, 6)], quick=[])

_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([dtype for dtype in CAST_DTYPES if dtype.is_floating_point]),
    quick=[],
)


@pytest.mark.cast_Short
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", CAST_DTYPES)
def test__cast_Short(dtype, value_range, shape):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Short(ref_inp)
    res_out = flag_gems._cast_Short(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Short
@pytest.mark.parametrize("shape,layout", _DEFINED_CASES)
@pytest.mark.parametrize("dtype", CAST_DTYPES)
def test__cast_Short_defined_values(dtype, shape, layout):
    inp = _apply_layout(_defined_payload(dtype, shape), layout)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Short(ref_inp)
    res_out = flag_gems._cast_Short(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Short
@pytest.mark.parametrize("dtype,values", _BOUNDARY_CASES)
def test__cast_Short_int16_boundaries(dtype, values):
    inp = torch.tensor(values, device=flag_gems.device, dtype=dtype).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Short(ref_inp)
    res_out = flag_gems._cast_Short(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Short
@pytest.mark.parametrize("shape,layout", _ALIAS_CASES)
def test__cast_Short_int16_alias(shape, layout):
    # Native same-dtype contract: for an int16 input ``torch.ops.aten._cast_Short``
    # returns the input tensor itself.  A probe on the assigned device confirmed
    # ``op(x) is x`` with equal data_ptr for contiguous, slice, transpose,
    # offset-only and expanded views, for a 0-dim tensor and for empty shapes,
    # so the candidate must return the very object it was given, not a copy.
    inp = _apply_layout(_defined_payload(torch.int16, shape), layout)
    # Values are checked against an independent copy, so the assertion is not
    # satisfied by object identity alone.
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Short(ref_inp)
    res_out = flag_gems._cast_Short(inp)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Short
@pytest.mark.parametrize("shape", _CONJ_CASES)
def test__cast_Short_lazy_conjugate_view(shape):
    # The reference accepts a tensor whose lazy conjugate bit is set; casting
    # keeps only the real part, which conjugation leaves unchanged, so the check
    # is that the lazy bit does not corrupt the read.
    inp = _defined_payload(torch.complex64, shape).conj()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Short(ref_inp)
    res_out = flag_gems._cast_Short(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Short
@pytest.mark.parametrize("non_blocking", _NON_BLOCKING_CASES)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test__cast_Short_non_blocking(dtype, non_blocking):
    # A [-1, 1) range truncates to an all-zero int16 output, so this parameter
    # sweep uses the signed fractional / edge fixture instead, which keeps the
    # compared result non-trivial.
    inp = _defined_payload(dtype, (20, 320, 15))
    ref_inp = tu.to_reference(inp)
    kwargs = {"non_blocking": non_blocking}

    ref_out = torch.ops.aten._cast_Short(ref_inp, **kwargs)
    res_out = flag_gems._cast_Short(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Short
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__cast_Short_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Short(ref_inp)
    res_out = flag_gems._cast_Short(inp)

    # NaN mapping to 0 is part of the native contract on this target, so the
    # exact comparison is applied to the converted int16 results.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Short
@pytest.mark.parametrize("dtype,shape", _EMPTY_CASES)
def test__cast_Short_empty(dtype, shape):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Short(ref_inp)
    res_out = flag_gems._cast_Short(inp)

    tu.assert_result_equal(res_out, ref_out)


_NON_TENSOR_INPUTS = [[1, 2, 3], 1.5, None, "abc"]


@pytest.mark.cast_Short
@pytest.mark.parametrize("bad_input", _NON_TENSOR_INPUTS)
def test__cast_Short_rejects_non_tensor_input(bad_input):
    # A candidate backed by the native schema reports RuntimeError while a plain
    # Python candidate reports TypeError; AttributeError/LookupError stay
    # unaccepted so a missing candidate still fails.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Short(bad_input)


# Native ``non_blocking`` coercion on this backend accepts bool and the numeric
# values 0/1/1.5, so only the values the schema rejects are used as negatives.
_INVALID_FLAGS = ["yes", [True]]


@pytest.mark.cast_Short
@pytest.mark.parametrize("bad_flag", _INVALID_FLAGS)
def test__cast_Short_rejects_invalid_non_blocking(bad_flag):
    inp = _defined_payload(torch.float32, (8,))
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Short(inp, non_blocking=bad_flag)


@pytest.mark.cast_Short
def test__cast_Short_rejects_unknown_keyword():
    inp = _defined_payload(torch.float32, (8,))
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Short(inp, dtype=torch.int16)


@pytest.mark.cast_Short
def test__cast_Short_rejects_missing_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Short()


@pytest.mark.cast_Short
def test__cast_Short_rejects_extra_argument():
    inp = _defined_payload(torch.float32, (8,))
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Short(inp, False, False)
