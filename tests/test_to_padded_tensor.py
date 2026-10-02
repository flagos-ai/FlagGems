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
from . import conftest as cfg
from . import test_utils as tu

# aten::to_padded_tensor(Tensor self, float padding, SymInt[]? output_size=None)
# densifies a nested tensor, filling every cell past a row length with padding.
# The two nested layouts take different kernels and different output_size rules
# and are therefore covered separately: jagged
# (torch.nested.nested_tensor(parts, layout=torch.jagged)) and ordinary strided
# nested storage (torch.nested.nested_tensor(parts)). The candidate is always
# called as flag_gems.to_padded_tensor(...).
#
# Exemptions, each measured on the active backend (see the proposal summary):
#  * fp8 and complex64 are rejected by the jagged densify kernel
#    ("'jagged_to_padded_dense' not implemented for 'Float8_e4m3fn'" /
#    "'ComplexFloat'"), so they are covered through strided rows, which accept
#    them, instead of being dropped.
#  * the rank-0 shape () cannot form a nested constituent: the jagged
#    constructor raises "Cannot construct a nested tensor from a list of
#    zero-dim tensors" and strided 0-dim parts fail in the operator with
#    "sizes_size_1 > 0 INTERNAL ASSERT FAILED". Rank >= 1 is fully covered.
_PAD = 3.0

_FLOAT_DTYPE_OPTIONS = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    _FLOAT_DTYPE_OPTIONS.append(torch.bfloat16)
if utils.fp64_is_supported:
    _FLOAT_DTYPE_OPTIONS.append(torch.float64)

_INT_DTYPE_OPTIONS = [torch.int8, torch.uint8, torch.int32]
if utils.int64_is_supported:
    _INT_DTYPE_OPTIONS.append(torch.int64)

_FP8_DTYPE_OPTIONS = (
    [torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else []
)

_JAGGED_DTYPES = _INT_DTYPE_OPTIONS + _FLOAT_DTYPE_OPTIONS + [torch.bool]

_STRIDED_ONLY_DTYPES = list(_FP8_DTYPE_OPTIONS)
if torch.complex64 in utils.COMPLEX_DTYPES:
    _STRIDED_ONLY_DTYPES.append(torch.complex64)

# The value-range grid splits dtypes over the two layouts: jagged rows carry every
# dtype the jagged kernel accepts, strided rows carry the dtypes only that layout
# can densify plus float32 as the shared representative.
_LAYOUT_DTYPE_ROWS = [("jagged", dtype) for dtype in _JAGGED_DTYPES] + [
    ("strided", dtype) for dtype in _STRIDED_ONLY_DTYPES + [torch.float32]
]
# Quick keeps every supported dtype and both layout branches, and only trims the
# shape/value-range axes of the collection.
_LAYOUT_DTYPE_CASES = _LAYOUT_DTYPE_ROWS


def _offsets_from_lengths(lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return offsets


def _make_values(dtype, lengths, trailing, value_range):
    # Values buffer of shape (sum(lengths), *trailing) holding every ragged row.
    shape = (sum(int(length) for length in lengths),) + tuple(trailing)
    return tu.make_input(dtype, shape, value_range)


def _make_nested(layout, values, lengths):
    parts = list(torch.split(values, [int(length) for length in lengths]))
    if layout == "jagged":
        return torch.nested.nested_tensor(parts, layout=torch.jagged)
    return torch.nested.nested_tensor(parts)


def _expected_padded_shape(num_parts, lengths, trailing):
    # NestedTensor.lengths() returns None for contiguous ragged storage, so the
    # ragged extent comes from the row lengths recorded by this module.
    return (int(num_parts), max(int(length) for length in lengths)) + tuple(trailing)


def _assert_padding_cells(res_out, ref_out, lengths):
    # Cells past each row length must carry the padding fill. The shared exact
    # comparison is NaN-aware, so nan/inf padding fills are comparable.
    for row, length in enumerate(lengths):
        if int(length) < res_out.shape[1]:
            tu.assert_result_equal(
                res_out[row, int(length) :], ref_out[row, int(length) :]
            )


def _ragged_lengths(lead):
    # Unequal, non-empty ragged rows for a leading dim.
    if lead <= 2:
        return [lead]
    third = lead // 3
    return [third, third, lead - 2 * third]


def _shape_cases():
    cases = []
    for shape in tu.selected_shapes():
        shape = tuple(shape)
        if not shape:
            continue  # rank-0 constituents cannot form a nested tensor
        cases.append((shape, _ragged_lengths(shape[0])))
    return cases


_SHAPE_CASES = tu.selected_cases(_shape_cases(), quick=[((2, 19, 7), [1, 1])])


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("shape,lengths", _SHAPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("layout,dtype", _LAYOUT_DTYPE_CASES)
def test_to_padded_tensor_value_ranges(layout, dtype, shape, lengths, value_range):
    trailing = tuple(shape[1:])
    values = _make_values(dtype, lengths, trailing, value_range)
    nested = _make_nested(layout, values, lengths)
    ref_nested = _make_nested(layout, tu.to_reference(values), lengths)

    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD)
    values_before = nested.values().clone()
    res_out = flag_gems.to_padded_tensor(nested, _PAD)

    tu.assert_result_equal(res_out, ref_out)
    assert tuple(res_out.shape) == _expected_padded_shape(
        len(lengths), lengths, trailing
    )
    # Densifying must not write the padding fill back into the values buffer.
    tu.assert_result_equal(nested.values(), values_before)


# Padding values: zero and a small positive value in quick; zero, positive,
# negative, a large boundary and the float boundaries inf/nan by default. The
# scalar is cast to the values dtype by the kernel, so each fill is compared
# against the reference fill rather than a Python float.
_PADDING_VALUES = [0.0, 3.0, -2.5, 1e30, float("inf"), float("nan")]
# Quick keeps a nonzero and a zero fill on both layouts; the full sweep adds the
# negative, large and float-boundary fills.
_PADDING_ROWS = [
    ("jagged", torch.float32),
    ("jagged", torch.float16),
    ("strided", torch.float32),
]
_PADDING_SHAPE = (20, 320, 15)
_PADDING_LENGTHS = [3, 7, 5, 5]


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize(
    "padding", tu.selected_cases(_PADDING_VALUES, quick=[0.0, 3.0, -2.5, 1e30])
)
@pytest.mark.parametrize("layout,dtype", _PADDING_ROWS)
def test_to_padded_tensor_padding_value(layout, dtype, padding):
    trailing = _PADDING_SHAPE[1:]
    values = _make_values(dtype, _PADDING_LENGTHS, trailing, ["-1", "1"])
    nested = _make_nested(layout, values, _PADDING_LENGTHS)
    ref_nested = _make_nested(layout, tu.to_reference(values), _PADDING_LENGTHS)

    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, padding)
    res_out = flag_gems.to_padded_tensor(nested, padding)

    tu.assert_result_equal(res_out, ref_out)
    assert tuple(res_out.shape) == _expected_padded_shape(
        len(_PADDING_LENGTHS), _PADDING_LENGTHS, trailing
    )
    _assert_padding_cells(res_out, ref_out, _PADDING_LENGTHS)


_KEYWORD_ROWS = [("jagged", torch.float32), ("strided", torch.int32)]
_KEYWORD_LENGTHS = [3, 7, 5, 5]


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("layout,dtype", _KEYWORD_ROWS)
def test_to_padded_tensor_keyword_arguments(layout, dtype):
    values = _make_values(dtype, _KEYWORD_LENGTHS, (320, 15), ["-1", "1"])
    nested = _make_nested(layout, values, _KEYWORD_LENGTHS)
    ref_nested = _make_nested(layout, tu.to_reference(values), _KEYWORD_LENGTHS)

    # The public entry point must accept the schema keyword name and the
    # explicit default output_size=None. This also covers the omitted-argument
    # (None) form; the value-range tests call the op without output_size.
    ref_out = torch.ops.aten.to_padded_tensor(
        ref_nested, padding=_PAD, output_size=None
    )
    res_out = flag_gems.to_padded_tensor(nested, padding=_PAD, output_size=None)

    tu.assert_result_equal(res_out, ref_out)


# (layout, output_size, probed padded shape). The jagged layout keeps the part
# count and the trailing dims the list omits and accepts truncation, growth and a
# zero extent; the strided layout rejects a shorter list outright and only
# accepts an output_size that matches the nested dims exactly or grows them.
# Quick keeps the exact and growth forms on both layouts; the truncating and
# wrong-length forms are the negative rows below, which are kept in both modes.
_OS_LENGTHS = [3, 5, 7, 5]
_OS_TRAILING = (8, 4)
_OS_CASES = [
    ("jagged", [4, 8, 8, 4], (4, 8, 8, 4)),
    ("jagged", [4, 10, 8, 4], (4, 10, 8, 4)),
    ("jagged", [4, 7, 8, 4], (4, 7, 8, 4)),
    ("jagged", [4, 9, 8, 4], (4, 9, 8, 4)),
    ("jagged", [4, 0, 8, 4], (4, 0, 8, 4)),
    ("jagged", [4, 7], (4, 7, 8, 4)),
    ("strided", [4, 8, 8, 4], (4, 8, 8, 4)),
    ("strided", [4, 10, 8, 4], (4, 10, 8, 4)),
]


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("layout,output_size,expected_shape", _OS_CASES)
def test_to_padded_tensor_output_size(layout, output_size, expected_shape):
    values = _make_values(torch.float32, _OS_LENGTHS, _OS_TRAILING, ["-1", "1"])
    nested = _make_nested(layout, values, _OS_LENGTHS)
    ref_nested = _make_nested(layout, tu.to_reference(values), _OS_LENGTHS)

    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD, output_size)
    res_out = flag_gems.to_padded_tensor(nested, _PAD, output_size)

    tu.assert_result_equal(res_out, ref_out)
    assert tuple(res_out.shape) == tuple(expected_shape)
    _assert_padding_cells(res_out, ref_out, _OS_LENGTHS)


# These rows are valid for both layouts. Entirely empty inputs are covered
# separately: jagged accepts them while strided rejects them.
_STRUCTURE_CASES = [
    ((20, 4), [0, 20, 0, 0]),
    ((20, 4), [20]),
    ((4, 4), [1, 1, 1, 1]),
    ((64, 4, 2), [0, 32, 32]),
]


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize(
    "shape,lengths",
    _STRUCTURE_CASES,
)
@pytest.mark.parametrize("layout,dtype", _LAYOUT_DTYPE_ROWS)
def test_to_padded_tensor_ragged_structure(layout, dtype, shape, lengths):
    trailing = tuple(shape[1:])
    values = _make_values(dtype, lengths, trailing, ["-1", "1"])
    nested = _make_nested(layout, values, lengths)
    ref_nested = _make_nested(layout, tu.to_reference(values), lengths)

    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD)
    values_before = nested.values().clone()
    res_out = flag_gems.to_padded_tensor(nested, _PAD)

    tu.assert_result_equal(res_out, ref_out)
    assert tuple(res_out.shape) == _expected_padded_shape(
        len(lengths), lengths, trailing
    )
    _assert_padding_cells(res_out, ref_out, lengths)
    tu.assert_result_equal(nested.values(), values_before)


# Storage state of the densified source. nested_tensor_from_jagged keeps the
# values tensor it is given, so the ragged values can sit at a nonzero storage
# offset or be a non-contiguous view; densifying must read through those strides
# and must not write the padding fill back into the shared storage. Both states
# are kept in quick on a small operand. The list-built constructors used above
# copy their parts into fresh contiguous storage, so no such state exists for the
# strided layout.
_SOURCE_LENGTHS = [3, 5, 7, 5]
_SOURCE_ROWS = [
    ("offset", (2, 3), torch.float32),
    ("view", (2, 3), torch.float32),
    ("offset", (4,), torch.int32),
    ("view", (), torch.float16),
]


def _make_source_values(dtype, trailing, state, value_range):
    total = sum(int(length) for length in _SOURCE_LENGTHS)
    if state == "offset":
        # One lead-in row puts the values buffer at a nonzero storage offset.
        base = tu.make_input(dtype, (total + 1,) + tuple(trailing), value_range)
        return base[1:]
    # Interleaved pairs leave the last dim of the values buffer strided.
    base = tu.make_input(dtype, (total,) + tuple(trailing) + (2,), value_range)
    return base[..., 0]


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("state,trailing,dtype", _SOURCE_ROWS)
def test_to_padded_tensor_source_layout(state, trailing, dtype):
    values = _make_source_values(dtype, trailing, state, ["-1", "1"])
    offsets = torch.tensor(_offsets_from_lengths(_SOURCE_LENGTHS), device=values.device)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)
    ref_values = tu.to_reference(values)
    ref_nested = torch.nested.nested_tensor_from_jagged(
        ref_values, offsets.to(ref_values.device)
    )

    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD)
    values_before = nested.values().clone()
    res_out = flag_gems.to_padded_tensor(nested, _PAD)

    tu.assert_result_equal(res_out, ref_out)
    # The offset-built jagged path pads to offsets[-1], not to the longest row.
    assert tuple(res_out.shape) == (len(_SOURCE_LENGTHS), sum(_SOURCE_LENGTHS)) + tuple(
        trailing
    )
    _assert_padding_cells(res_out, ref_out, _SOURCE_LENGTHS)
    tu.assert_result_equal(nested.values(), values_before)


# Special values use the shared generator contract (nan for every float dtype;
# inf and mixed only where the dtype can represent them, so e4m3fn keeps nan
# only while e5m2 keeps nan/inf/mixed). Positive special cases are default-only.
_SPECIAL_LENGTHS = [1, 2, 3]
_SPECIAL_TRAILING = (4,)


def _special_rows():
    rows = [("jagged", dtype) for dtype in _FLOAT_DTYPE_OPTIONS]
    rows += [("strided", dtype) for dtype in _FP8_DTYPE_OPTIONS + [torch.float16]]
    return rows


def _special_cases():
    cases = []
    for layout, dtype in _special_rows():
        for case_dtype, scenario in tu.special_value_cases([dtype]):
            cases.append((layout, case_dtype, scenario))
    return cases


_SPECIAL_CASES = tu.selected_cases(_special_cases(), quick=[])


def _special_values(dtype, scenario, count, trailing):
    # The shared generator returns a small vector carrying the scenario payload;
    # tile it to exactly fill the (count, *trailing) values buffer.
    trailing_size = 1
    for dim in trailing:
        trailing_size *= int(dim)
    total = int(count) * trailing_size
    flat = tu.make_special_input(dtype, scenario).reshape(-1)
    repeats = -(-total // flat.numel())
    return flat.repeat(repeats)[:total].reshape((int(count),) + tuple(trailing))


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("layout,dtype,scenario", _SPECIAL_CASES)
def test_to_padded_tensor_special_values(layout, dtype, scenario):
    values = _special_values(dtype, scenario, sum(_SPECIAL_LENGTHS), _SPECIAL_TRAILING)
    nested = _make_nested(layout, values, _SPECIAL_LENGTHS)
    ref_nested = _make_nested(layout, tu.to_reference(values), _SPECIAL_LENGTHS)

    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD)
    res_out = flag_gems.to_padded_tensor(nested, _PAD)

    # NaN must land where the reference produces it and the padded cells keep
    # the finite padding value.
    tu.assert_result_equal(res_out, ref_out)
    assert tuple(res_out.shape) == _expected_padded_shape(
        len(_SPECIAL_LENGTHS), _SPECIAL_LENGTHS, _SPECIAL_TRAILING
    )
    _assert_padding_cells(res_out, ref_out, _SPECIAL_LENGTHS)


# The jagged path differentiates its values buffer; the strided path below
# differentiates a nested leaf. Both are default-only.
def _backward_cases():
    cases = [
        ((2,), [2, 4, 1, 5], torch.float32),
        ((2, 3), [3, 3], torch.float32),
        ((4,), [1, 2, 3, 4], torch.float16),
    ]
    if utils.bf16_is_supported:
        cases.append(((2,), [2, 4, 1, 5], torch.bfloat16))
    if utils.fp64_is_supported:
        cases.append(((2,), [2, 4, 1, 5], torch.float64))
    return cases


_BACKWARD_CASES = tu.selected_cases(_backward_cases(), quick=[])


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("trailing,lengths,dtype", _BACKWARD_CASES)
def test_to_padded_tensor_backward(trailing, lengths, dtype):
    # Both sides get independent leaves and independent upstream gradients that
    # hold the same values, so the gradients are comparable directly.
    base_values = _make_values(dtype, lengths, trailing, ["-1", "1"])
    values = base_values.detach().clone().requires_grad_(True)
    ref_values = tu.to_reference(base_values).detach().clone().requires_grad_(True)
    offsets = torch.tensor(_offsets_from_lengths(lengths), device=values.device)
    ref_offsets = torch.tensor(_offsets_from_lengths(lengths), device=ref_values.device)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)
    ref_nested = torch.nested.nested_tensor_from_jagged(ref_values, ref_offsets)

    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD)
    res_out = flag_gems.to_padded_tensor(nested, _PAD)

    tu.assert_result_equal(res_out, ref_out)
    # The offset-built jagged path pads to offsets[-1], not to the longest row.
    assert tuple(res_out.shape) == (len(lengths), sum(lengths)) + tuple(trailing)

    upstream = tu.make_input(dtype, tuple(res_out.shape), ["-1", "1"])
    (ref_grad,) = torch.autograd.grad(ref_out, ref_values, tu.to_reference(upstream))
    (res_grad,) = torch.autograd.grad(res_out, values, upstream)
    # Backward copies the valid row regions without reducing them.
    tu.assert_result_equal(res_grad, ref_grad)


# Negative workloads: only the candidate exception is asserted, and every row is
# collected in both modes. Native aten raises for the same invalid calls (probe
# details in the proposal summary).
_CANDIDATE_EXC = (RuntimeError, TypeError, NotImplementedError, ValueError, IndexError)
_NEGATIVE_ROWS = [("jagged", torch.float32), ("strided", torch.int8)]


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("layout,dtype", _NEGATIVE_ROWS)
def test_to_padded_tensor_rejects_negative_extent(layout, dtype):
    values = _make_values(dtype, [3, 5, 7, 5], (8, 4), ["-1", "1"])
    nested = _make_nested(layout, values, [3, 5, 7, 5])
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(nested, _PAD, [4, -1, 8, 4])


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("output_size", [[4, 6, 8, 4], [3, 7, 8, 4], [4, 0, 8, 4]])
def test_to_padded_tensor_rejects_strided_truncation(output_size):
    values = _make_values(torch.float32, [3, 5, 7, 5], (8, 4), ["-1", "1"])
    nested = _make_nested("strided", values, [3, 5, 7, 5])
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(nested, _PAD, output_size)


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("output_size", [[8, 8, 4], [4, 8]])
def test_to_padded_tensor_rejects_wrong_output_size_length(output_size):
    values = _make_values(torch.float32, [3, 5, 7, 5], (8, 4), ["-1", "1"])
    nested = _make_nested("strided", values, [3, 5, 7, 5])
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(nested, _PAD, output_size)


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("output_size", [[4, "x", 8, 4], [4, 7.5, 8, 4]])
def test_to_padded_tensor_rejects_non_integer_output_size(output_size):
    values = _make_values(torch.float32, [3, 5, 7, 5], (8, 4), ["-1", "1"])
    nested = _make_nested("jagged", values, [3, 5, 7, 5])
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(nested, _PAD, output_size)


@pytest.mark.to_padded_tensor
def test_to_padded_tensor_rejects_missing_padding():
    values = _make_values(torch.float32, [3, 5], (4,), ["-1", "1"])
    nested = _make_nested("jagged", values, [3, 5])
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(nested)


@pytest.mark.to_padded_tensor
def test_to_padded_tensor_rejects_invalid_padding_type():
    values = _make_values(torch.float32, [3, 5], (4,), ["-1", "1"])
    nested = _make_nested("jagged", values, [3, 5])
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(nested, "fill")


@pytest.mark.to_padded_tensor
def test_to_padded_tensor_rejects_dense_input():
    # A dense tensor is not a nested tensor; native aten raises
    # NotImplementedError for the CUDA backend on this call.
    dense = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(dense, _PAD)


@pytest.mark.to_padded_tensor
def test_to_padded_tensor_rejects_non_tensor_input():
    with pytest.raises(_CANDIDATE_EXC):
        flag_gems.to_padded_tensor(3.14, _PAD)


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize(
    "lengths,trailing", [([0, 0], ()), ([0, 0], (2,)), ([2, 3], (0,))]
)
@pytest.mark.parametrize("dtype", _JAGGED_DTYPES)
def test_to_padded_tensor_empty_jagged(lengths, trailing, dtype):
    values = _make_values(dtype, lengths, trailing, ["-1", "1"])
    nested = _make_nested("jagged", values, lengths)
    ref_nested = _make_nested("jagged", tu.to_reference(values), lengths)
    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD)
    res_out = flag_gems.to_padded_tensor(nested, _PAD)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize(
    "lengths,trailing", [([0, 0], ()), ([0, 0], (2,)), ([2, 3], (0,))]
)
def test_to_padded_tensor_rejects_empty_strided(lengths, trailing):
    nested = _make_nested(
        "strided", _make_values(torch.float32, lengths, trailing, ["-1", "1"]), lengths
    )
    with pytest.raises(RuntimeError, match="at least one constituent tensor"):
        flag_gems.to_padded_tensor(nested, _PAD)


# CPU nested backward lacks create_nt_buffer for FP8; the device oracle supports it.
_STRIDED_BACKWARD_DTYPES = _FLOAT_DTYPE_OPTIONS + (
    [] if cfg.TO_CPU else _FP8_DTYPE_OPTIONS
)


@pytest.mark.to_padded_tensor
@pytest.mark.parametrize("dtype", tu.selected_cases(_STRIDED_BACKWARD_DTYPES, quick=[]))
@pytest.mark.parametrize("output_size", [None, [2, 5, 3]])
def test_to_padded_tensor_strided_backward(dtype, output_size):
    parts = [tu.make_input(dtype, (length, 2), ["-1", "1"]) for length in (2, 3)]
    nested = torch.nested.nested_tensor(parts, requires_grad=True)
    ref_nested = torch.nested.nested_tensor(
        [tu.to_reference(part) for part in parts], requires_grad=True
    )
    ref_out = torch.ops.aten.to_padded_tensor(ref_nested, _PAD, output_size)
    res_out = flag_gems.to_padded_tensor(nested, _PAD, output_size)
    tu.assert_result_equal(res_out, ref_out)
    upstream = tu.make_input(dtype, tuple(res_out.shape), ["-1", "1"])
    ref_grad = torch.autograd.grad(ref_out, ref_nested, tu.to_reference(upstream))[0]
    res_grad = torch.autograd.grad(res_out, nested, upstream)[0]
    for res_part, ref_part in zip(res_grad.unbind(), ref_grad.unbind()):
        tu.assert_result_equal(res_part, ref_part)
