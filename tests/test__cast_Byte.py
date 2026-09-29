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
#
# Correctness tests for aten::_cast_Byte (cast a tensor to torch.uint8).
#
# The operator exposes a single 'default' overload, so there is no .out form to
# exercise. It is unary, so nothing broadcasts, and its result dtype is
# integral, so it has no autograd path; neither dimension applies.

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Static backend capability flags: unsupported dtypes are filtered at
# collection time instead of being probed or skipped while running.
_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# Input dtypes accepted by the real call torch.ops.aten._cast_Byte(t), each of
# which returns a torch.uint8 tensor.
SUPPORTED_DTYPES = [
    dtype
    for dtype in (
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
    )
    if _DTYPE_FLAGS.get(dtype, True)
]

# Positive auxiliary workloads are default-only: an empty quick selection drops
# the item at collection time, leaving quick with the grid and the negatives.
NON_BLOCKING_CASES = tu.selected_cases([False, True], quick=[])
PARAM_SHAPES = tu.selected_cases([(1024, 1024)], quick=[])
LAYOUT_CASES = tu.selected_cases(
    ["offset_only", "transpose", "strided", "offset_strided"], quick=[]
)
EMPTY_SHAPES = tu.selected_cases([(0,), (0, 3)], quick=[])
IDENTITY_CASES = tu.selected_cases([torch.uint8], quick=[])
BOUNDARY_CASES = tu.selected_cases([(16, 8)], quick=[])
# Float dtypes whose out-of-range and non-finite conversions are exercised.
BOUNDARY_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _DTYPE_FLAGS.get(dtype, True)
]
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[])

# Defined float cast boundaries: fractional, negative, in range, past the uint8
# range, and non-finite.
BOUNDARY_VALUES = [
    -1e30,
    -300.5,
    -1.9,
    -1.0,
    -0.5,
    -0.4,
    0.0,
    0.4,
    0.5,
    1.5,
    2.9,
    127.4,
    254.6,
    255.9,
    256.5,
    300.5,
    511.9,
    1e30,
    float("nan"),
    float("inf"),
    float("-inf"),
]


def _cast_view_base(rows, cols):
    # Coordinate-varying cast input: fractional, negative, positive and past the
    # uint8 range, and different at every coordinate, so neither an all-zero
    # result nor a read that ignores the view's strides or offset can match the
    # native cast.
    flat = torch.arange(rows * cols, dtype=torch.float32, device=flag_gems.device)
    return (flat * 7.5 - 40.5).reshape(rows, cols)


@pytest.mark.cast_Byte
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cast_byte(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Byte(ref_inp)
    res_out = flag_gems._cast_Byte(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Byte
@pytest.mark.parametrize("shape", PARAM_SHAPES)
@pytest.mark.parametrize("non_blocking", NON_BLOCKING_CASES)
def test_cast_byte_non_blocking(shape, non_blocking):
    # Values that are non-zero after the cast, so this parameter variant is not
    # satisfied by an all-zero result.
    inp = _cast_view_base(*shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Byte(ref_inp, non_blocking=non_blocking)
    res_out = flag_gems._cast_Byte(inp, non_blocking=non_blocking)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Byte
@pytest.mark.parametrize("layout", LAYOUT_CASES)
def test_cast_byte_input_layout(layout):
    base = _cast_view_base(16, 8)
    ref_base = tu.to_reference(base)
    if layout == "offset_only":
        # Non-zero storage offset, strides unchanged.
        inp, ref_inp = base[1:], ref_base[1:]
    elif layout == "transpose":
        inp, ref_inp = base.t(), ref_base.t()
    elif layout == "strided":
        inp, ref_inp = base[:, ::2], ref_base[:, ::2]
    else:
        inp, ref_inp = base[1:, ::3], ref_base[1:, ::3]

    ref_out = torch.ops.aten._cast_Byte(ref_inp)
    res_out = flag_gems._cast_Byte(inp)

    # The cast values differ per coordinate, so a candidate that ignores the
    # view's strides or storage offset cannot match the exact comparison.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Byte
@pytest.mark.parametrize("shape", BOUNDARY_CASES)
@pytest.mark.parametrize("dtype", BOUNDARY_DTYPES)
def test_cast_byte_cast_boundaries(shape, dtype):
    numel = math.prod(shape)
    repeats = (numel + len(BOUNDARY_VALUES) - 1) // len(BOUNDARY_VALUES)
    values = (BOUNDARY_VALUES * repeats)[:numel]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device).reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Byte(ref_inp)
    res_out = flag_gems._cast_Byte(inp)

    # Out-of-range and non-finite float -> uint8 results are not fixed by the
    # operator specification, so the contract is the native result on this
    # device; it is compared in full, neither clamped nor widened.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Byte
@pytest.mark.parametrize("dtype", IDENTITY_CASES)
def test_cast_byte_uint8_identity_alias(dtype):
    # The index sequence is built in int32, so this fixture does not depend on
    # the optional int64 backend capability.
    inp = (torch.arange(64, dtype=torch.int32, device=flag_gems.device) * 7 % 256).to(
        dtype
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Byte(ref_inp)
    res_out = flag_gems._cast_Byte(inp)

    tu.assert_result_equal(res_out, ref_out)
    # Native returns the input itself for an already-uint8 input (no copy); the
    # candidate must keep that same-device identity/alias contract.
    assert res_out.device == inp.device
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.cast_Byte
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_cast_byte_empty(shape):
    inp = _cast_view_base(2, 3)[:0].reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Byte(ref_inp)
    res_out = flag_gems._cast_Byte(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Byte
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_cast_byte_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._cast_Byte(ref_inp)
    res_out = flag_gems._cast_Byte(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.cast_Byte
def test_cast_byte_missing_argument():
    # The native parser reports the absent operand as RuntimeError; an explicit
    # Python candidate signature reports the same invalid call as TypeError.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Byte()


@pytest.mark.cast_Byte
def test_cast_byte_rejects_non_tensor_input():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Byte(1.0)


@pytest.mark.cast_Byte
@pytest.mark.parametrize("non_blocking", ["x", [], {}])
def test_cast_byte_rejects_invalid_non_blocking(non_blocking):
    # The native parser accepts None/0/1/1.5 for the declared bool parameter but
    # rejects a string, list or dict; those are the invalid values tested here.
    inp = _cast_view_base(2, 4)

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._cast_Byte(inp, non_blocking=non_blocking)
