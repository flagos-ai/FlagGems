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

"""Correctness tests for ``aten::_cast_Long`` (``Tensor self, bool non_blocking=False``).

The native cast is a zero-copy alias for an int64 input: it returns the input
tensor object itself. Every other dtype is read out of place, so read-only
behaviour and object identity are separate properties.

The float -> int64 mapping of non-finite and out-of-int64-range values is
implementation defined, so those rows compare against the native result on the
active device instead of against a literal bound.
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_INT64_MIN = -9223372036854775808
_INT64_MAX = 9223372036854775807


def _positive(cases):
    """Statically select positive cases on the active backend.

    int64 is the only result dtype of aten::_cast_Long, so a backend without
    int64 has no valid native result to compare against.
    """
    return cases if utils.int64_is_supported else []


def _dtype_eligible(dtype):
    """Static dtype eligibility taken from the runtime capability flags."""
    if dtype == torch.bfloat16:
        return utils.bf16_is_supported
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return utils.fp8_is_supported
    if dtype == torch.float64:
        return utils.fp64_is_supported
    if dtype == torch.int64:
        return utils.int64_is_supported
    return True


# Every required spec dtype (all accepted by the aten core) plus int16 and bool.
_CAST_DTYPES = _positive(
    [
        dtype
        for dtype in tu.REQUIRED_DTYPES + [torch.int16, torch.bool, torch.float64]
        if _dtype_eligible(dtype)
    ]
)
# The special-value matrix is derived from the dtypes this operator accepts, so
# FP8 is covered at the same granularity as the wider floats; e4m3fn cannot
# represent infinity and ``tu.special_value_cases`` already encodes that.
_CAST_FLOAT_DTYPES = [dtype for dtype in _CAST_DTYPES if dtype.is_floating_point]

_VALUE_CASES = [
    (shape, value_range, dtype)
    for shape in tu.selected_shapes()
    for value_range in tu.selected_ranges()
    for dtype in _CAST_DTYPES
]

# Explicit ``non_blocking`` call forms. The omitted keyword is already covered by
# the one-argument calls of the value grid; passing the default explicitly still
# needs both spellings and both values.
_PARAM_CALLS = tu.selected_cases(
    [
        pytest.param(False, False, id="positional-false"),
        pytest.param(False, True, id="keyword-false"),
        pytest.param(True, False, id="positional-true"),
        pytest.param(True, True, id="keyword-true"),
    ],
    quick=[],
)
_ONE_SHAPE = tu.selected_cases([(20, 320, 15)], quick=[(2, 19, 7)])
# Default-only workloads use the largest spec shape.
_DEFAULT_ONLY_SHAPE = tu.selected_cases([(20, 320, 15)], quick=[])

# Raw range rows over the view layouts. The id names the base geometry and the
# logical geometry the view produces. Values in [-1, 1) truncate toward zero, so
# these rows cannot detect a reader that ignores a stride or a storage offset;
# the defined-value rows below carry that part.
_LAYOUT_ROWS = _positive(
    tu.selected_cases(
        [
            pytest.param(
                "slice", (8, 16, 32), ["-1", "1"], id="slice-(8,16,32)->(8,8,4)"
            ),
            pytest.param(
                "transpose",
                (8, 16, 32),
                ["-1", "1"],
                id="transpose-(8,16,32)->(16,8,32)",
            ),
            pytest.param("offset", (16, 16), ["-1", "1"], id="offset-(16,16)->(4,16)"),
            pytest.param("expand", (1, 16), ["-1", "1"], id="expand-(1,16)->(4,16)"),
            pytest.param("empty", (0,), ["-1", "1"], id="empty-(0,)"),
            pytest.param("empty_dim", (2, 0, 3), ["-1", "1"], id="empty_dim-(2,0,3)"),
        ],
        quick=[],
    )
)

# Exact int64 literals: 2**53 + 1 has no float64 representation and the extremes
# bound the int64 range, so a float intermediate changes at least one element.
_EXACT_INT64_VALUES = [9007199254740993, _INT64_MAX, _INT64_MIN, -1, 0, 7]

# Exact in both FP8 formats, which have no dense arithmetic to build them with.
_FP8_FRACTIONAL_VALUES = [1.5, -2.5, 0.5, -0.75, 3.25, -1.125, 2.0, -0.25]


def _coordinate_index(shape):
    """An int64 index whose value changes with every coordinate."""
    shape = tuple(shape)
    acc = torch.zeros(shape, dtype=torch.int64, device=flag_gems.device)
    for axis, size in enumerate(shape):
        view = [1] * len(shape)
        view[axis] = size
        step = torch.arange(size, device=flag_gems.device).view(view) * (3 + 2 * axis)
        acc = acc + step
    return acc


def _integer_coordinate(acc, dtype):
    """Map a non-negative int64 coordinate into ``dtype``.

    Signed dtypes wrap inside their own range so both signs are present in every
    workload; uint8 has no negative half, so its own domain is used as-is.
    """
    if dtype == torch.uint8:
        return (acc % 256).to(torch.uint8)
    if dtype == torch.int8:
        acc = (acc % 200) - 100
    elif dtype == torch.int16:
        acc = (acc % 1600) - 800
    else:
        # int32 and int64 hold the coordinate itself; alternate the sign.
        acc = torch.where(acc % 2 == 0, acc, -acc)
    return acc.to(dtype)


def _coordinate_input(shape, dtype):
    """A defined, position-varying input built in the requested dtype.

    Each axis shifts the value, so a reader that ignores a stride or a storage
    offset observes different numbers. Float payloads are fractional with both
    signs, so the truncating cast is not uniform across the tensor.
    """
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # These literals are exact in both FP8 formats, which have no dense
        # arithmetic to build the values with.
        count = math.prod(shape) if shape else 1
        repeats = count // len(_FP8_FRACTIONAL_VALUES) + 1
        values = (_FP8_FRACTIONAL_VALUES * repeats)[:count]
        return torch.tensor(values, dtype=dtype, device=flag_gems.device).reshape(shape)
    acc = _coordinate_index(shape)
    if dtype == torch.bool:
        return acc % 2 == 1
    if dtype.is_floating_point:
        return (acc.to(dtype) / 8) - 2
    return _integer_coordinate(acc, dtype)


def _precision_input(shape):
    """float64 values whose low bits vanish in float32.

    2**53 + 2k is exactly representable in float64 (the spacing is 2 there) and
    needs more significand bits than float32 has, so every element but the one at
    the coordinate origin changes when the cast is staged through float32. The
    sign alternates to cover both directions.
    """
    acc = _coordinate_index(shape)
    magnitude = 9007199254740992.0 + (2 * acc).to(torch.float64)
    return torch.where(acc % 2 == 0, magnitude, -magnitude)


def _defined_base(shape, values_kind, dtype):
    if values_kind == "precision":
        return _precision_input(shape)
    return _coordinate_input(shape, dtype)


def _int64_base(shape):
    count = math.prod(shape) if shape else 1
    repeats = count // len(_EXACT_INT64_VALUES) + 1
    values = (_EXACT_INT64_VALUES * repeats)[:count]
    flat = torch.tensor(values, dtype=torch.int64, device=flag_gems.device)
    return flat.reshape(shape)


def _fp64_boundary_input(kind):
    """float64 fixtures for the defined and the implementation-defined branch.

    ``precision`` and ``in-range`` are exactly representable in float64 and stay
    inside the int64 range, so the conversion is defined there. ``extreme`` holds
    values at or beyond the int64 range boundary plus the non-finite ones, where
    the conversion has no single defined result.
    """
    if kind == "precision":
        return _precision_input((4, 8, 8)).reshape(-1)
    if kind == "fractional":
        return _coordinate_input((7, 5, 3), torch.float64).reshape(-1)
    if kind == "in-range":
        return torch.tensor(
            [
                9007199254740992.0,  # 2**53: neighbouring float64 values are 2 apart
                9007199254740994.0,
                9007199254740996.0,
                9223372036854774784.0,  # largest float64 below 2**63, still in range
                -9223372036854774784.0,
                1.5,
                -1.5,
                0.0,
            ],
            dtype=torch.float64,
            device=flag_gems.device,
        )
    if kind == "extreme":
        return torch.tensor(
            [
                9223372036854775808.0,  # 2**63: one past the largest int64
                -9223372036854775808.0,
                float("inf"),
                float("-inf"),
                float("nan"),
            ],
            dtype=torch.float64,
            device=flag_gems.device,
        )
    raise AssertionError(f"unknown float64 fixture {kind!r}")


_VIEW_FRACTIONAL_ROWS = [
    pytest.param(
        "slice",
        (8, 16, 32),
        "fractional",
        torch.float32,
        id="fractional-slice-(8,16,32)->(8,8,4)",
    ),
    pytest.param(
        "transpose",
        (8, 16, 32),
        "fractional",
        torch.float32,
        id="fractional-transpose-(8,16,32)->(16,8,32)",
    ),
    pytest.param(
        "offset",
        (16, 16),
        "fractional",
        torch.float32,
        id="fractional-offset-(16,16)->(4,16)",
    ),
    pytest.param(
        "expand",
        (1, 16),
        "fractional",
        torch.float32,
        id="fractional-expand-(1,16)->(4,16)",
    ),
]
_VIEW_PRECISION_ROWS = (
    [
        pytest.param(
            "slice",
            (8, 16, 32),
            "precision",
            torch.float64,
            id="precision-slice-(8,16,32)->(8,8,4)",
        ),
        pytest.param(
            "transpose",
            (8, 16, 32),
            "precision",
            torch.float64,
            id="precision-transpose-(8,16,32)->(16,8,32)",
        ),
        pytest.param(
            "offset",
            (16, 16),
            "precision",
            torch.float64,
            id="precision-offset-(16,16)->(4,16)",
        ),
    ]
    if utils.fp64_is_supported
    else []
)
_DEFINED_LAYOUT_ROWS = _positive(
    tu.selected_cases(_VIEW_FRACTIONAL_ROWS + _VIEW_PRECISION_ROWS, quick=[])
)

# The native cast returns the input tensor itself for an int64 input, so identity
# is asserted for dense, viewed, offset-sliced and empty inputs.
_INT64_ALIAS_ROWS = _positive(
    tu.selected_cases(
        [
            pytest.param((4, 3), "dense", id="dense-(4,3)"),
            pytest.param((2, 5, 3), "transpose", id="transpose-(2,5,3)->(5,2,3)"),
            pytest.param((8, 16, 32), "slice", id="slice-(8,16,32)->(8,8,4)"),
            pytest.param((16, 16), "offset", id="offset-(16,16)->(4,16)"),
            pytest.param((0,), "dense", id="empty-(0,)"),
            pytest.param((2, 0, 3), "dense", id="empty_dim-(2,0,3)"),
        ],
        quick=[],
    )
)

_FP64_BOUNDARY_CASES = (
    _positive(
        tu.selected_cases(
            [
                pytest.param("precision", id="precision-near-2**53"),
                pytest.param("fractional", id="fractional"),
                pytest.param("in-range", id="in-range-boundaries"),
                pytest.param("extreme", id="out-of-range-and-non-finite"),
            ],
            quick=[],
        )
    )
    if utils.fp64_is_supported
    else []
)


def _view_of(base, layout, base_shape):
    if layout == "slice":
        return base[:, ::2, 1:5]
    if layout == "transpose":
        return base.transpose(0, 1)
    if layout == "offset":
        return base[3:7]
    if layout == "expand":
        return base.expand(base_shape[0] * 4, base_shape[1])
    if layout in ("dense", "empty", "empty_dim"):
        return base
    raise AssertionError(f"unknown layout {layout!r}")


@pytest.mark.cast_Long
@pytest.mark.parametrize("shape,value_range,dtype", _VALUE_CASES)
def test__cast_Long(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    # The one-argument call also covers the schema default for non_blocking.
    ref_out = torch.ops.aten._cast_Long(ref_inp)
    res_out = flag_gems._cast_Long(inp)

    # Truncation toward zero is exact, so the cast is compared with zero
    # tolerance rather than with the arithmetic tolerance.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Long
@pytest.mark.parametrize("layout,base_shape,value_range", _LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _CAST_DTYPES)
def test__cast_Long_layout(layout, base_shape, value_range, dtype):
    base = tu.make_input(dtype, base_shape, value_range)
    inp = _view_of(base, layout, base_shape)
    ref_base = tu.to_reference(base)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Long(ref_inp)
    res_out = flag_gems._cast_Long(inp)

    tu.assert_result_equal(res_out, ref_out)
    # The cast reads the view out of place: the viewed storage stays intact.
    tu.assert_result_equal(base, ref_base)


@pytest.mark.cast_Long
@pytest.mark.parametrize("layout,base_shape,values_kind,dtype", _DEFINED_LAYOUT_ROWS)
def test__cast_Long_layout_defined_values(layout, base_shape, values_kind, dtype):
    base = _defined_base(base_shape, values_kind, dtype)
    inp = _view_of(base, layout, base_shape)
    ref_base = tu.to_reference(base)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Long(ref_inp)
    res_out = flag_gems._cast_Long(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(base, ref_base)


@pytest.mark.cast_Long
@pytest.mark.parametrize("shape,layout", _INT64_ALIAS_ROWS)
def test__cast_Long_int64_alias(shape, layout):
    base = _int64_base(shape)
    inp = _view_of(base, layout, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Long(ref_inp)
    res_out = flag_gems._cast_Long(inp)

    # The native cast returns the input tensor object itself for an int64 input;
    # a clone of an empty tensor or a distinct view would still satisfy a
    # data_ptr comparison without being the same tensor.
    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Long
@pytest.mark.parametrize("kind", _FP64_BOUNDARY_CASES)
def test__cast_Long_fp64_boundaries(kind):
    inp = _fp64_boundary_input(kind)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Long(ref_inp)
    res_out = flag_gems._cast_Long(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Long
@pytest.mark.parametrize("non_blocking,as_keyword", _PARAM_CALLS)
@pytest.mark.parametrize("shape", _ONE_SHAPE)
@pytest.mark.parametrize("dtype", _CAST_DTYPES)
def test__cast_Long_non_blocking(non_blocking, as_keyword, shape, dtype):
    # Defined fractional values rather than the truncating [-1, 1) range, so a
    # candidate that mishandles the flag differs in value as well as in call
    # form.
    inp = _coordinate_input(shape, dtype)
    ref_inp = tu.to_reference(inp)

    if as_keyword:
        ref_out = torch.ops.aten._cast_Long(ref_inp, non_blocking=non_blocking)
        res_out = flag_gems._cast_Long(inp, non_blocking=non_blocking)
    else:
        ref_out = torch.ops.aten._cast_Long(ref_inp, non_blocking)
        res_out = flag_gems._cast_Long(inp, non_blocking)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Long
@pytest.mark.parametrize("shape", _DEFAULT_ONLY_SHAPE)
@pytest.mark.parametrize(
    "dtype", [dtype for dtype in _CAST_DTYPES if dtype != torch.int64]
)
def test__cast_Long_preserves_input(shape, dtype):
    inp = _coordinate_input(shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Long(ref_inp)
    res_out = flag_gems._cast_Long(inp)

    tu.assert_result_equal(res_out, ref_out)
    # A non-int64 input is read out of place, so the source values must survive;
    # the int64 identity case is covered by test__cast_Long_int64_alias.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.cast_Long
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_CAST_FLOAT_DTYPES), quick=[]),
)
def test__cast_Long_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Long(ref_inp)
    res_out = flag_gems._cast_Long(inp)

    # The non-finite and out-of-range mapping to int64 is implementation defined,
    # so the native result on the active device is the oracle.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Long
@pytest.mark.parametrize("shape", _ONE_SHAPE)
def test__cast_Long_rejects_non_bool_non_blocking(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Long(inp, "yes")


@pytest.mark.cast_Long
@pytest.mark.parametrize("shape", _ONE_SHAPE)
def test__cast_Long_rejects_extra_positional(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Long(inp, False, True)


@pytest.mark.cast_Long
@pytest.mark.parametrize("shape", _ONE_SHAPE)
def test__cast_Long_rejects_out_keyword(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    # A float32 buffer: no int64 tensor has to be constructed for a backend that
    # does not support it, and it is the keyword itself that must be rejected.
    out = torch.empty(inp.shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Long(inp, out=out)


@pytest.mark.cast_Long
def test__cast_Long_rejects_missing_operand():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Long()


@pytest.mark.cast_Long
@pytest.mark.parametrize(
    "bad_self", tu.selected_cases([3.14, [1.0, 2.0]], quick=[3.14])
)
def test__cast_Long_rejects_non_tensor(bad_self):
    # Only the schema-level exceptions are accepted, so a missing candidate
    # raising AttributeError cannot pass this test.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Long(bad_self)
