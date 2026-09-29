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

from . import test_utils as tu

# aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
#
# Narrowing cast to float16: the native result is the round-to-nearest-even
# float16 value of every element, subnormals included, and out-of-range
# magnitudes become +/-inf. Comparison is exact against that native result, not
# because the input values are preserved.
#
# Unary, so no broadcast dimension applies. There is no .out overload:
# torch.ops.aten._cast_Half.out(...) raises AttributeError ("has no overload
# name 'out'"), so no out= workload exists and none is simulated.
#
# Only the static device flags are read here; no dtype is probed by running
# tensors at import or collection time.
SUPPORT_FP64 = flag_gems.runtime.device.support_fp64
SUPPORT_BF16 = flag_gems.runtime.device.support_bf16
SUPPORT_INT64 = flag_gems.runtime.device.support_int64
SUPPORT_FP8 = flag_gems.runtime.device.support_fp8

_GATED_OFF = set()
if not SUPPORT_FP8:
    _GATED_OFF.update((torch.float8_e4m3fn, torch.float8_e5m2))
if not SUPPORT_BF16:
    _GATED_OFF.add(torch.bfloat16)
if not SUPPORT_INT64:
    _GATED_OFF.add(torch.int64)
if not SUPPORT_FP64:
    _GATED_OFF.update((torch.float64, torch.complex128))


def _supported(dtypes):
    """Static capability filter for dtype tables."""
    return [dtype for dtype in dtypes if dtype not in _GATED_OFF]


def _supported_rows(rows):
    """Static capability filter for case tables whose first column is a dtype."""
    return [row for row in rows if row[0] not in _GATED_OFF]


# Every dtype the native operator accepts, complex included: the cast only loads
# the source dtype and stores float16, discarding an imaginary part if present.
_MAIN_DTYPES = _supported(
    [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.float64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ]
)


def _view_input(dtype, shape, layout):
    """Build the requested logical shape with a non-trivial layout."""
    if layout == "contiguous":
        return tu.make_input(dtype, shape, ["-1", "1"])
    if layout == "offset":
        # Same logical shape and strides, but a non-zero storage offset.
        flat = tu.make_input(dtype, (math.prod(shape) + 3,), ["-1", "1"])
        return flat[3:].view(shape)
    if layout == "transposed":
        buffer = tu.make_input(dtype, (shape[1], shape[0], *shape[2:]), ["-1", "1"])
        return buffer.transpose(0, 1)
    if layout == "holed":
        # Every logical element is followed by an unused slot.
        buffer = tu.make_input(dtype, tuple(2 * dim for dim in shape), ["-1", "1"])
        stride = tuple(2 * step for step in buffer.stride())
        return buffer.as_strided(shape, stride)
    raise AssertionError("unknown layout: " + layout)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype", _MAIN_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test__cast_Half_value_range(dtype, value_range, shape):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Half(ref_inp)
    res_out = flag_gems._cast_Half(inp)

    tu.assert_result_equal(res_out, ref_out)


# Empty, offset, hole and transpose geometry; each row compares the real logical
# shape it builds. Default-only.
_LAYOUT_CASES = tu.selected_cases(
    _supported_rows(
        [
            (torch.float32, (20, 320, 15), "offset"),
            (torch.float32, (20, 320, 15), "transposed"),
            (torch.float32, (1024, 1024), "holed"),
            (torch.float16, (20, 320, 15), "transposed"),
            (torch.int32, (1024, 1024), "offset"),
            (torch.bfloat16, (2, 0, 3), "contiguous"),
            (torch.float32, (0,), "contiguous"),
            (torch.float64, (), "contiguous"),
        ]
    ),
    quick=[],
)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype,shape,layout", _LAYOUT_CASES)
def test__cast_Half_layouts(dtype, shape, layout):
    inp = _view_input(dtype, shape, layout)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Half(ref_inp)
    res_out = flag_gems._cast_Half(inp)

    tu.assert_result_equal(res_out, ref_out)


# A float16 input needs no conversion: the native cast returns the input object
# itself for dense, transposed, offset, holed-stride, 0-dim and empty inputs.
_IDENTITY_CASES = tu.selected_cases(
    _supported_rows(
        [
            (torch.float16, (20, 320, 15), "contiguous"),
            (torch.float16, (20, 320, 15), "transposed"),
            (torch.float16, (1024, 1024), "offset"),
            (torch.float16, (1024, 1024), "holed"),
            (torch.float16, (0,), "contiguous"),
            (torch.float16, (2, 0, 3), "contiguous"),
            (torch.float16, (), "contiguous"),
        ]
    ),
    quick=[],
)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype,shape,layout", _IDENTITY_CASES)
def test__cast_Half_float16_identity(dtype, shape, layout):
    inp = _view_input(dtype, shape, layout)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Half(ref_inp)
    res_out = flag_gems._cast_Half(inp)

    tu.assert_result_equal(res_out, ref_out)
    # Native-guided object identity: the native cast answers with its own input
    # object here, so the candidate has to do the same. A copy or a distinct view
    # of the same storage would still match the value, shape, pointer and offset.
    assert res_out is inp


# non_blocking coverage in both call forms; the main grid already uses the
# omitted schema default. Default-only.
_PARAM_SHAPE = (1024, 1024)
_PARAM_DTYPES = tu.selected_cases(
    _supported([torch.bfloat16, torch.int32, torch.int64, torch.float8_e4m3fn]),
    quick=[],
)
_NON_BLOCKING_FORMS = tu.selected_cases(
    [
        pytest.param((False,), {}, id="non_blocking_positional_false"),
        pytest.param((True,), {}, id="non_blocking_positional_true"),
        pytest.param((), {"non_blocking": False}, id="non_blocking_keyword_false"),
        pytest.param((), {"non_blocking": True}, id="non_blocking_keyword_true"),
    ],
    quick=[],
)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
@pytest.mark.parametrize("args,kwargs", _NON_BLOCKING_FORMS)
def test__cast_Half_non_blocking(dtype, args, kwargs):
    inp = tu.make_input(dtype, _PARAM_SHAPE, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Half(ref_inp, *args, **kwargs)
    res_out = flag_gems._cast_Half(inp, *args, **kwargs)

    tu.assert_result_equal(res_out, ref_out)


# float8_e4m3fn cannot represent infinity, so the shared generator keeps only
# its nan scenario; float8_e5m2 keeps nan, inf and the mixed payload.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(
        _supported(
            [
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]
        )
    ),
    quick=[],
)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__cast_Half_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Half(ref_inp)
    res_out = flag_gems._cast_Half(inp)

    tu.assert_result_equal(res_out, ref_out)


# Deterministic fixtures on the float16 rounding boundaries: below the smallest
# subnormal (2**-25 flushes to 0), the smallest subnormal and normal, two
# mantissa ties, and the overflow point (65504 stays finite, 65520 becomes inf).
# The integer rows straddle the 2048 spacing change and the same overflow point.
# A candidate that flushes, rounds or saturates differently fails here.
_FP16_BOUNDARY_VALUES = [
    -0.0,
    -(2.0**-25),
    2.0**-25,
    2.0**-24,
    2.0**-14,
    1.0 + 2.0**-11,
    1.0 + 3.0 * 2.0**-11,
    1.0 + 2.0**-11 + 2.0**-24,
    5.0e-8,
    6.0e-8,
    32768.0,
    32769.0,
    65504.0,
    65519.0,
    65520.0,
    -65504.0,
    -65520.0,
]

_FP64_BOUNDARY_VALUES = [
    -0.0,
    -(2.0**-25),
    1.0 + 2.0**-40,
    5e-324,
    2.0**-25,
    2.0**-14,
    65519.999999,
    65520.0,
    1e300,
]

_INT32_BOUNDARY_VALUES = [
    2047,
    2048,
    2049,
    2050,
    4095,
    4097,
    65504,
    65505,
    65519,
    65520,
    65536,
    2**24 + 1,
    2**31 - 1,
    -(2**31),
]

_INT64_BOUNDARY_VALUES = [
    2**31,
    2**40,
    2**53 + 1,
    2**62,
    -(2**62),
]

_ROUNDING_CASES = tu.selected_cases(
    _supported_rows(
        [
            (torch.float32, _FP16_BOUNDARY_VALUES),
            (torch.float64, _FP64_BOUNDARY_VALUES),
            (torch.int32, _INT32_BOUNDARY_VALUES),
            (torch.int64, _INT64_BOUNDARY_VALUES),
        ]
    ),
    quick=[],
)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype,values", _ROUNDING_CASES)
def test__cast_Half_rounding_boundaries(dtype, values):
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Half(ref_inp)
    res_out = flag_gems._cast_Half(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(torch.signbit(res_out), torch.signbit(ref_out))


# Complex sources are accepted natively with the imaginary part discarded; the
# payload carries a non-zero imaginary part, which a candidate that keeps it
# would report.
_COMPLEX_VALUES = [
    1.5 + 2.0j,
    -0.5 - 3.25j,
    65504.0 + 1000.0j,
    2.0**-25 + 0.75j,
    complex(float("inf"), 5.0),
    complex(float("nan"), -1.5),
]

_COMPLEX_CASES = tu.selected_cases(
    _supported_rows(
        [
            (torch.complex64, (6,)),
            (torch.complex64, (2, 3)),
            (torch.complex128, (6,)),
        ]
    ),
    quick=[],
)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype,shape", _COMPLEX_CASES)
def test__cast_Half_complex_payload(dtype, shape):
    inp = torch.tensor(_COMPLEX_VALUES, dtype=dtype, device=flag_gems.device).reshape(
        shape
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Half(ref_inp)
    res_out = flag_gems._cast_Half(inp)

    tu.assert_result_equal(res_out, ref_out)


# The cast gradient is the upstream gradient converted back to the input dtype,
# an exact and reduction-free conversion, so it is compared exactly.
_GRAD_CASES = tu.selected_cases(
    _supported_rows(
        [
            (torch.float32, (20, 320, 15), "contiguous"),
            (torch.float16, (20, 320, 15), "contiguous"),
            (torch.bfloat16, (20, 320, 15), "contiguous"),
            (torch.float64, (20, 320, 15), "contiguous"),
            (torch.float32, (1024, 1024), "transposed"),
            (torch.float32, (), "contiguous"),
        ]
    ),
    quick=[],
)


@pytest.mark.cast_Half
@pytest.mark.parametrize("dtype,shape,layout", _GRAD_CASES)
def test__cast_Half_backward(dtype, shape, layout):
    inp = _view_input(dtype, shape, layout)
    leaf = inp.detach().requires_grad_(True)
    reference = tu.to_reference(inp).detach().requires_grad_(True)

    ref_out = torch.ops.aten._cast_Half(reference)
    res_out = flag_gems._cast_Half(leaf)
    tu.assert_result_equal(res_out, ref_out)

    upstream = tu.make_input(torch.float16, ref_out.shape, ["-1", "1"])
    (ref_grad,) = torch.autograd.grad(
        ref_out, reference, grad_outputs=tu.to_reference(upstream)
    )
    (res_grad,) = torch.autograd.grad(res_out, leaf, grad_outputs=upstream)

    tu.assert_result_equal(res_grad, ref_grad)


# The native operator rejects these call forms, so the candidate must too.
# Non-bool values such as None or 1 are coerced by the native binding, so a
# string is the invalid non_blocking value used here.
@pytest.mark.cast_Half
def test__cast_Half_rejects_non_tensor_input():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Half(3.5)


@pytest.mark.cast_Half
def test__cast_Half_rejects_missing_input():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Half()


@pytest.mark.cast_Half
def test__cast_Half_rejects_invalid_non_blocking_type():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Half(inp, "not-a-bool")
