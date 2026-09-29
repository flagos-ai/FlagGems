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

"""Correctness tests for ``aten::_cast_Char(self, non_blocking=False) -> Tensor``.

The reference and every expected value come from ``torch.ops.aten._cast_Char``;
the active-device native conversion supplies the expectation and its result is
always ``torch.int8``, so the comparison is exact and carries no tolerance.
Out-of-range and non-finite conversions are left to the native operator and are
never hard-coded here.

An int8 operand comes back as the very same tensor object - same storage,
storage offset, stride and shape - for contiguous, transposed, strided, offset,
channels-last and empty inputs.

The spec's ``[-1, 1)`` range truncates to zero for every integer dtype, so the
layout, identity, empty-view and parameter families use a deterministic
position-varying fixture with ``|value| >= 1.25`` instead: a candidate that
returns zeros, clamps, or reads the wrong stride, storage offset or parameter
value stays observably wrong there.

Exempt dimensions, verified against the native operator: there is no ``.out``
overload, the operator takes exactly one tensor operand and therefore cannot
broadcast, and an integral result carries no autograd history, so there is no
backward family.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_BF16 = utils.bf16_is_supported
_INT64 = utils.int64_is_supported
_FP8 = utils.fp8_is_supported
_FP64 = utils.fp64_is_supported

_REQUIRED_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
)


def _supported(dtype):
    """Whether the device statically advertises kernels for ``dtype``.

    Only the ``accuracy_utils`` capability flags are read here, so no dtype is
    probed at import or collection time.
    """
    if dtype == torch.bfloat16:
        return _BF16
    if dtype == torch.int64:
        return _INT64
    if dtype == torch.float64:
        return _FP64
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return _FP8
    return True


def _dtypes(*extra):
    """The required dtypes plus ``extra``, minus the unsupported ones."""
    dtypes = []
    for dtype in _REQUIRED_DTYPES + extra:
        if dtype not in dtypes and _supported(dtype):
            dtypes.append(dtype)
    return tuple(dtypes)


_GRID_DTYPES = _dtypes(torch.int16, torch.bool)
# Every bool payload is true, so a bool fixture could not distinguish a wrong
# stride, offset or parameter value; the fixture families start after bool.
_FIXTURE_DTYPES = _dtypes(torch.int16)

# Position-varying payload with alternating signs and ``|value| >= 1.25``. The
# native cast leaves every entry a non-zero integer, so a candidate that zeroes,
# clamps or misreads the stride/storage offset is observable. FP8 storage rounds
# much of the payload, so the fixture does not claim distinct stored values
# there; both sides read the identical tensor and the layout and identity rows
# carry the stride/offset sensitivity.
_FIXTURE_VALUES = (
    1.25,
    -2.75,
    3.5,
    -4.25,
    5.75,
    -6.5,
    7.25,
    -8.75,
    9.5,
    -10.25,
    11.75,
    -12.5,
    13.25,
    -14.75,
    15.5,
    -16.25,
    17.75,
    -18.5,
    19.25,
    -20.75,
    21.5,
    -22.25,
    23.75,
    -24.5,
    25.25,
    -26.75,
    27.5,
    -28.25,
    29.75,
    -30.5,
    31.25,
    -32.75,
)


def _fixture_tensor(dtype, count):
    """A deterministic non-zero payload for ``dtype`` of exactly ``count`` values."""
    values = _FIXTURE_VALUES
    if dtype == torch.uint8:
        values = tuple(abs(value) for value in values)
    elif not dtype.is_floating_point:
        # int() truncates toward zero, keeping the alternating signs.
        values = tuple(int(value) for value in values)
    pattern = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    repeats = -(-count // pattern.numel())
    return pattern.repeat(repeats)[:count]


def _fixture(dtype, shape):
    count = 1
    for dim in shape:
        count *= dim
    return _fixture_tensor(dtype, count).reshape(shape)


@pytest.mark.cast_Char
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test__cast_Char(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)


_LAYOUTS = ("transpose", "channels_last", "stride_slice", "offset_slice")
_LAYOUT_WORKLOADS = tu.selected_cases(_LAYOUTS, quick=[])


def _layout_input(layout, dtype):
    if layout == "transpose":
        return _fixture(dtype, (8, 10)).transpose(0, 1)
    if layout == "channels_last":
        return _fixture(dtype, (2, 8, 4, 4)).to(memory_format=torch.channels_last)
    if layout == "stride_slice":
        return _fixture(dtype, (8, 24))[:, 1::3]
    return _fixture(dtype, (8, 5)).reshape(-1)[4:36]


@pytest.mark.cast_Char
@pytest.mark.parametrize("layout", _LAYOUT_WORKLOADS)
@pytest.mark.parametrize("dtype", _FIXTURE_DTYPES)
def test__cast_Char_view_layouts(dtype, layout):
    inp = _layout_input(layout, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)


# Same-dtype (int8) forms, where the native operator aliases instead of copying.
_IDENTITY_FORMS = (
    "contiguous",
    "transpose",
    "stride_slice",
    "offset_slice",
    "channels_last",
    "empty_view",
    "scalar",
)
_IDENTITY_WORKLOADS = tu.selected_cases(_IDENTITY_FORMS, quick=[])


def _identity_input(form):
    if form == "contiguous":
        return _fixture(torch.int8, (8, 10))
    if form == "transpose":
        return _fixture(torch.int8, (8, 10)).transpose(0, 1)
    if form == "stride_slice":
        return _fixture(torch.int8, (8, 24))[:, 1::3]
    if form == "offset_slice":
        return _fixture(torch.int8, (8, 5)).reshape(-1)[4:36]
    if form == "channels_last":
        return _fixture(torch.int8, (2, 8, 4, 4)).to(memory_format=torch.channels_last)
    if form == "empty_view":
        return _fixture(torch.int8, (0, 10)).transpose(0, 1)
    return _fixture(torch.int8, ()).reshape(())


@pytest.mark.cast_Char
@pytest.mark.parametrize("form", _IDENTITY_WORKLOADS)
def test__cast_Char_int8_identity(form):
    inp = _identity_input(form)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)
    # Native hands back the int8 operand itself, so a de-aliased copy is not
    # equivalent even when its values, stride and storage offset match.
    assert res_out is inp


_EMPTY_EXTENT_SHAPES = tu.selected_cases(((0,), (0, 3), (3, 0), (2, 0, 5)), quick=[])


@pytest.mark.cast_Char
@pytest.mark.parametrize("shape", _EMPTY_EXTENT_SHAPES)
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test__cast_Char_empty_extent(dtype, shape):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)


_EMPTY_VIEWS = ("rows", "cols", "block", "transposed")
_EMPTY_VIEW_WORKLOADS = tu.selected_cases(_EMPTY_VIEWS, quick=[])


def _empty_view_input(kind, dtype):
    if kind == "rows":
        return _fixture(dtype, (0, 8))
    if kind == "cols":
        return _fixture(dtype, (8, 0))
    if kind == "block":
        return _fixture(dtype, (4, 6))[2:2, :]
    return _fixture(dtype, (0, 8)).transpose(0, 1)


@pytest.mark.cast_Char
@pytest.mark.parametrize("kind", _EMPTY_VIEW_WORKLOADS)
@pytest.mark.parametrize("dtype", _FIXTURE_DTYPES)
def test__cast_Char_empty_view(dtype, kind):
    inp = _empty_view_input(kind, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)


# Payloads at the int8 wrap boundary, at the float infinity edge and over the
# full integer range. The expected result is the native cast of the same payload
# in the requested dtype; nothing is clamped or widened here.
_FLOAT_BOUNDARY = (0.0, 0.5, -0.5, 127.0, 128.0, -128.0, -129.0, 300.5, -300.5, -1e-45)
_FLOAT_EXTREME = (3.4e38, -3.4e38, float("inf"), float("-inf"), 1e-45, -1e-45)
_INT_BOUNDARY = (0, 127, 128, -128, -129, 255, 256, -256, 32767, -32768)
_INT_WIDE_BOUNDARY = (
    0,
    127,
    128,
    -128,
    -129,
    1000,
    -1000,
    2**31 - 1,
    -(2**31),
    -300,
    300,
)
_UINT8_BOUNDARY = (0, 127, 128, 200, 255)
_INT8_BOUNDARY = (-128, -1, 0, 1, 127)
_BOOL_BOUNDARY = (True, False, True, True, False)


def _boundary_rows():
    rows = [
        pytest.param(torch.int8, _INT8_BOUNDARY, id="int8"),
        pytest.param(torch.uint8, _UINT8_BOUNDARY, id="uint8"),
        pytest.param(torch.bool, _BOOL_BOUNDARY, id="bool"),
        pytest.param(torch.int16, _INT_BOUNDARY, id="int16"),
        pytest.param(torch.int32, _INT_WIDE_BOUNDARY, id="int32"),
        pytest.param(torch.float16, _FLOAT_BOUNDARY, id="float16"),
        pytest.param(torch.float32, _FLOAT_BOUNDARY, id="float32"),
        pytest.param(torch.float32, _FLOAT_EXTREME, id="float32-extreme"),
    ]
    if _BF16:
        rows.append(pytest.param(torch.bfloat16, _FLOAT_BOUNDARY, id="bfloat16"))
    if _FP64:
        rows.append(pytest.param(torch.float64, _FLOAT_BOUNDARY, id="float64"))
    if _INT64:
        rows.append(pytest.param(torch.int64, _INT_WIDE_BOUNDARY, id="int64"))
    if _FP8:
        rows.append(
            pytest.param(torch.float8_e4m3fn, _FLOAT_BOUNDARY, id="float8-e4m3fn")
        )
        rows.append(pytest.param(torch.float8_e5m2, _FLOAT_BOUNDARY, id="float8-e5m2"))
    return rows


_BOUNDARY_WORKLOADS = tu.selected_cases(_boundary_rows(), quick=[])


@pytest.mark.cast_Char
@pytest.mark.parametrize("dtype,payload", _BOUNDARY_WORKLOADS)
def test__cast_Char_boundaries(dtype, payload):
    inp = torch.tensor(payload, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)


_PARAM_SHAPE = (1024, 1024)
_PARAM_DTYPES = tuple(
    dtype
    for dtype in (
        torch.uint8,
        torch.int8,
        torch.int32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.float8_e4m3fn,
    )
    if _supported(dtype)
)
_NON_BLOCKING_FORMS = ("false", "true")
_NON_BLOCKING_WORKLOADS = tu.selected_cases(_NON_BLOCKING_FORMS, quick=[])


@pytest.mark.cast_Char
@pytest.mark.parametrize("form", _NON_BLOCKING_WORKLOADS)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test__cast_Char_non_blocking(dtype, form):
    # The main grid already calls the omitted schema default; these rows pin the
    # explicit ``False``/``True`` forms on the non-zero 1024x1024 fixture.
    inp = _fixture(dtype, _PARAM_SHAPE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp, non_blocking=form == "true")
    res_out = flag_gems._cast_Char(inp, non_blocking=form == "true")

    tu.assert_result_equal(res_out, ref_out)


_SPECIAL_DTYPES = tuple(dtype for dtype in _dtypes() if dtype.is_floating_point)
_SPECIAL_WORKLOADS = tu.selected_cases(
    [
        pytest.param(
            dtype, scenario, id="{}-{}".format(str(dtype).split(".")[-1], scenario)
        )
        for dtype, scenario in tu.special_value_cases(_SPECIAL_DTYPES)
    ],
    quick=[],
)


@pytest.mark.cast_Char
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_WORKLOADS)
def test__cast_Char_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)


# Narrowing to an integral dtype drops autograd history, so there is no backward
# family to cover; these rows still pin the forward result for an input that
# requires grad, against the native contract.
_GRAD_DTYPES = tuple(
    dtype
    for dtype in _dtypes(torch.float64)
    if dtype.is_floating_point and dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
)
_GRAD_WORKLOADS = tu.selected_cases(
    [pytest.param(dtype, id=str(dtype).split(".")[-1]) for dtype in _GRAD_DTYPES],
    quick=[],
)


@pytest.mark.cast_Char
@pytest.mark.parametrize("dtype", _GRAD_WORKLOADS)
def test__cast_Char_grad_enabled_input(dtype):
    inp = _fixture(dtype, (8, 8)).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)

    ref_out = torch.ops.aten._cast_Char(ref_inp)
    res_out = flag_gems._cast_Char(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Char
def test__cast_Char_rejects_out_keyword():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    out = torch.empty((4, 4), dtype=torch.int8, device=flag_gems.device)

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Char(inp, out=out)


@pytest.mark.cast_Char
def test__cast_Char_rejects_non_bool_non_blocking():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Char(inp, non_blocking="yes")


@pytest.mark.cast_Char
def test__cast_Char_requires_operand():
    # The native wrapper reports the missing operand as a RuntimeError, so both
    # error types are accepted here.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Char()
