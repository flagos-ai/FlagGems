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

# aten::_is_any_true(self) -> Tensor reduces a whole bool tensor to a 0-dim bool
# tensor; the call takes exactly one tensor argument and no optional parameter.
#
# Spec dimensions this operator cannot carry, with the mechanism:
#   * value ranges: tu.make_input fills bool from a random 0/1 pattern and
#     ignores the range, so the five ranges keep the shared 5-range x 7-shape
#     grid in place. They are grid labels over independent random fills, not
#     five distinct value distributions.
#   * nan/inf: bool cannot represent nan or inf, so no workload can exist and
#     tu.special_value_cases(torch.bool) is empty.
#   * backward: a bool tensor cannot require grad, so this reduction builds no
#     autograd graph and exposes no gradient to compare.
#   * broadcast: one operand reduced in full, so no operand pair can broadcast.
_BOOL = torch.bool

# Deterministic element patterns: an "any" outcome is only proven by inputs
# whose expected result is fixed without executing the reduction.
_PATTERNS = (
    "all_false",
    "all_true",
    "first_true",
    "middle_true",
    "last_true",
    "single_false",
)

_PATTERN_CASES = tu.selected_cases(
    [(shape, pattern) for shape in tu.selected_shapes() for pattern in _PATTERNS],
    # The random 0/1 grid above is True on essentially every draw, so quick mode
    # also needs guaranteed outcomes: fixed False, fixed True, a lone False
    # inside an otherwise-True tensor, and a True held only by the last element.
    quick=[
        ((2, 19, 7), "all_false"),
        ((2, 19, 7), "all_true"),
        ((2, 19, 7), "single_false"),
        ((2, 19, 7), "last_true"),
    ],
)

# Sizes around small powers of two plus the smallest rank-2/rank-3 inputs: a
# generic boundary sweep for a whole-tensor scan. No implementation exists to
# read a dispatch threshold from, so these are regression boundaries only.
_BOUNDARY_SHAPES = [
    (),
    (1,),
    (3,),
    (7,),
    (31,),
    (32,),
    (33,),
    (255,),
    (256,),
    (257,),
    (1023,),
    (1024,),
    (1025,),
    (1, 1),
    (4, 1, 1),
]

_BOUNDARY_CASES = tu.selected_cases(
    [
        (shape, pattern)
        for shape in _BOUNDARY_SHAPES
        for pattern in ("all_false", "all_true", "last_true")
    ],
    quick=[((), "all_false"), ((), "all_true"), ((1025,), "last_true")],
)

# An empty input has no element that can be true, so the result must be False.
_EMPTY_SHAPES = tu.selected_cases(
    [(0,), (0, 3), (2, 0, 4), (0, 0), (16, 0, 7), (0, 0, 0, 0)],
    quick=[(0,), (2, 0, 4)],
)

# Views whose backing storage always holds one True element: "True" rows keep it
# inside the logical view, "False" rows leave it outside, so a candidate reading
# raw storage instead of the view's elements answers wrongly on those rows.
# transpose is a permutation of the whole storage and an "any" reduction is
# invariant to it, so it is paired with a head-truncating transpose view
# (transpose_head) that really excludes stored elements. "unit_offset_*" rows
# are contiguous but offset into their storage (smallest unit-stride case).
_NONCONTIG_LAYOUTS = tu.selected_cases(
    [
        ("transpose", True),
        ("transpose_head", False),
        ("row_step", True),
        ("row_step_odd", False),
        ("column_step", True),
        ("column_step_odd", False),
        ("offset_slice", True),
        ("offset_skip", False),
        ("expand_zero_stride", True),
        ("expand_zero_stride_false", False),
        ("unit_offset_tail", True),
        ("unit_offset_head", False),
    ],
    quick=[],
)

# Rejecting non-bool input was measured on the NVIDIA backend probed here (the
# kernel asserts self.scalar_type() == at::kBool); another vendor's build is not
# assumed to reject the same inputs, so this grid is collected only on the
# vendor where the measurement holds. Each dtype is added only when the static
# capability flag says the device can construct it.
_NON_BOOL_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.float16,
        torch.int32,
        torch.complex64,
        *([torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else []),
        *([torch.bfloat16] if utils.bf16_is_supported else []),
        *([torch.int64] if utils.int64_is_supported else []),
        *([torch.float64] if utils.fp64_is_supported else []),
    ]
    if flag_gems.vendor_name == "nvidia"
    else []
)


def _make_bool_input(shape, pattern):
    """Build a bool tensor of ``shape`` whose "any" outcome is known."""
    if pattern == "all_true":
        return torch.ones(shape, dtype=_BOOL, device=flag_gems.device)

    inp = torch.zeros(shape, dtype=_BOOL, device=flag_gems.device)
    flat = inp.view(-1)
    if pattern == "first_true":
        flat[0] = True
    elif pattern == "middle_true":
        flat[flat.numel() // 2] = True
    elif pattern == "last_true":
        flat[-1] = True
    elif pattern == "single_false":
        flat.fill_(True)
        flat[0] = False
    return inp


def _make_noncontiguous_input(layout, contains_true):
    """Return the view named by ``layout``; the storage holds one True element."""
    if layout.startswith("unit_offset"):
        base = torch.zeros(16, dtype=_BOOL, device=flag_gems.device)
        base[12] = True
        return base[8:] if contains_true else base[2:8]

    base = torch.zeros((8, 12), dtype=_BOOL, device=flag_gems.device)
    base[4, 4] = True
    if layout == "transpose":
        return base.t()
    if layout == "transpose_head":
        return base.t()[:3]
    if layout == "row_step":
        return base[::2]
    if layout == "row_step_odd":
        return base[1::2]
    if layout == "column_step":
        return base[:, ::2]
    if layout == "column_step_odd":
        return base[:, 1::2]
    if layout == "offset_slice":
        return base[2:6, 3:9]
    if layout == "expand_zero_stride":
        return base[4:5, 4:5].expand(6, 7)
    if layout == "expand_zero_stride_false":
        return base[0:1, 0:1].expand(6, 7)
    return base[5:8, 0:3]


@pytest.mark.is_any_true
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__is_any_true_value_ranges(shape, value_range):
    inp = tu.make_input(_BOOL, shape, value_range)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten._is_any_true(ref_inp)
    res_out = flag_gems._is_any_true(inp)

    tu.assert_result_equal(res_out, ref_out)
    # The reduction only reads its input; writing back into it is a bug.
    tu.assert_result_equal(inp, before)


@pytest.mark.is_any_true
@pytest.mark.parametrize("shape,pattern", _PATTERN_CASES)
def test__is_any_true_patterns(shape, pattern):
    inp = _make_bool_input(shape, pattern)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten._is_any_true(ref_inp)
    res_out = flag_gems._is_any_true(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, before)


@pytest.mark.is_any_true
@pytest.mark.parametrize("shape,pattern", _BOUNDARY_CASES)
def test__is_any_true_scan_boundaries(shape, pattern):
    inp = _make_bool_input(shape, pattern)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten._is_any_true(ref_inp)
    res_out = flag_gems._is_any_true(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, before)


@pytest.mark.is_any_true
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
def test__is_any_true_empty(shape):
    inp = torch.zeros(shape, dtype=_BOOL, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten._is_any_true(ref_inp)
    res_out = flag_gems._is_any_true(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, before)


@pytest.mark.is_any_true
@pytest.mark.parametrize("layout,contains_true", _NONCONTIG_LAYOUTS)
def test__is_any_true_non_contiguous(layout, contains_true):
    inp = _make_noncontiguous_input(layout, contains_true)
    ref_inp = tu.to_reference(inp)
    before = inp.clone()

    ref_out = torch.ops.aten._is_any_true(ref_inp)
    res_out = flag_gems._is_any_true(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, before)


@pytest.mark.is_any_true
@pytest.mark.parametrize("dtype", _NON_BOOL_DTYPES)
def test__is_any_true_rejects_non_bool_dtype(dtype):
    # Non-bool input: the native kernel rejects it, so the candidate must fail
    # rather than reinterpret the bytes (flag_gems dtype guards assert).
    inp = tu.make_input(dtype, (4, 8), ["-1", "1"])
    with pytest.raises((AssertionError, RuntimeError, TypeError)):
        flag_gems._is_any_true(inp)


@pytest.mark.is_any_true
@pytest.mark.parametrize("value", [3.14, "x"])
def test__is_any_true_rejects_non_tensor(value):
    # Schema negative: a non-tensor argument is rejected by the typed dispatcher
    # (RuntimeError: Expected a value of type 'Tensor' for argument 'self' on
    # the probed backend) or by a candidate type guard (TypeError). AttributeError
    # would mean the candidate is missing, not that the argument was rejected.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._is_any_true(value)


@pytest.mark.is_any_true
@pytest.mark.parametrize("num_args", [0, 2])
def test__is_any_true_rejects_wrong_arity(num_args):
    # Schema negative: aten::_is_any_true takes exactly one tensor argument, and
    # the probed native call raises RuntimeError for both a missing and an extra
    # argument.
    inp = torch.zeros((4,), dtype=_BOOL, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._is_any_true(*((inp,) * num_args))
