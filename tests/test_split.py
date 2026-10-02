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

# The four aten::split overloads share the single public candidate name:
#   split.Tensor(t, SymInt split_size, int dim=0)   -> flag_gems.split(t, size, dim)
#   split.sizes(t, SymInt[] sizes, int dim=0)       -> flag_gems.split(t, [..], dim)
#   split(t, int[] sizes, int dim=0)                -> same call as split.sizes
#   split.str(s, str? separator=None, int max=-1)   -> flag_gems.split(s, sep, max)
# The tensor forms return aliasing views, so each case also matches stride,
# storage offset, view state and storage identity against the native parts; a
# copy-based candidate keeps the values but fails those checks. The str form
# returns a Python list[str] and is compared as a list.

_SPLIT_DTYPES = (
    tu.REQUIRED_DTYPES
    + [torch.complex64, torch.bool]
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)

# split's backward is a plain scatter of the upstream parts (measured max|diff|
# == 0.0), so the gradient comparison is exact too.
_BACKWARD_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.complex64,
] + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])

# split rejects rank-0 input ("split expects at least a 1-dimensional tensor"),
# so the shared grid drops () and keeps the 1..5-dim spec shapes.
_SPLIT_SHAPES = [shape for shape in tu.selected_shapes() if shape]
_QUICK_SHAPE = (2, 19, 7)


def _chunk(extent):
    # Several parts plus a remainder at every spec extent.
    return max(1, extent // 3)


def _partition(extent, parts):
    # Even split with the remainder on the leading parts; sums to extent.
    size, rest = divmod(extent, parts)
    sizes = [size] * parts
    for i in range(rest):
        sizes[i] += 1
    return sizes


def _part_shapes(shape, dim, chunk):
    extent = shape[dim]
    counts = [chunk] * (extent // chunk)
    if extent % chunk:
        counts.append(extent % chunk)
    return [shape[:dim] + (count,) + shape[dim + 1 :] for count in counts]


def _assert_parts_equal(res_parts, ref_parts, inp):
    assert isinstance(res_parts, (list, tuple)), type(res_parts)
    assert len(res_parts) == len(ref_parts)
    storage_base = inp.untyped_storage().data_ptr()
    element_size = inp.element_size()
    for res_part, ref_part in zip(res_parts, ref_parts):
        tu.assert_result_equal(res_part, ref_part)
        assert res_part.stride() == ref_part.stride()
        assert res_part.storage_offset() == ref_part.storage_offset()
        assert res_part._is_view() == ref_part._is_view()
        # Each part aliases the input storage, empty parts included (measured:
        # the empty part of split(t, [2, 0, 4], 0) still shares the input
        # storage), so this is checked whenever the backing storage exists.
        if storage_base:
            assert res_part.untyped_storage().data_ptr() == storage_base
        # An empty tensor has an undefined data pointer (measured 0), so the
        # offset arithmetic is only checked where the pointer is defined.
        if res_part.numel():
            assert (
                res_part.data_ptr()
                == storage_base + res_part.storage_offset() * element_size
            )


def _assert_str_parts_equal(res_parts, ref_parts):
    assert isinstance(res_parts, list), type(res_parts)
    assert len(res_parts) == len(ref_parts)
    assert all(isinstance(part, str) for part in res_parts)
    assert res_parts == ref_parts


# Every family keeps its dtype list in both modes; quick trims rows, not dtypes.
_LAYOUT_DTYPES = [torch.float32, torch.int8, torch.float8_e4m3fn, torch.complex64]
_VIEW_DTYPES = [torch.float32, torch.int32]
_SIZES_DTYPES = [torch.float32, torch.int32, torch.float8_e4m3fn, torch.bool]
_DIM_DTYPES = [torch.float32, torch.int64, torch.float8_e4m3fn]
_DEFAULT_DIM_DTYPES = [torch.float32, torch.int64]

# Both partition modes (3 and 5 parts) at every shape, split on the first and
# the last dimension.
_SIZES_ROWS = [
    (shape, dim, _partition(shape[dim], parts))
    for shape in _SPLIT_SHAPES
    for dim in sorted({0, len(shape) - 1})
    for parts in (3, 5)
]
_QUICK_SIZES_ROWS = [
    (_QUICK_SHAPE, dim, _partition(_QUICK_SHAPE[dim], parts))
    for dim in sorted({0, len(_QUICK_SHAPE) - 1})
    for parts in (3, 5)
]
SIZES_CASES = tu.selected_cases(_SIZES_ROWS, quick=_QUICK_SIZES_ROWS)

# First / middle / last dimension, in both signs, on the small quick shape plus
# the spec shapes.
_DIM_ROWS = [
    ((1,), -1),
    ((256,), -1),
    ((2, 19, 7), 0),
    ((2, 19, 7), 1),
    ((2, 19, 7), 2),
    ((2, 19, 7), -3),
    ((2, 19, 7), -2),
    ((2, 19, 7), -1),
    ((1024, 1024), 1),
    ((1024, 1024), -1),
    ((1024, 1024), -2),
    ((20, 320, 15), 1),
    ((20, 320, 15), -2),
    ((20, 320, 15), 2),
    ((16, 128, 64, 60), 2),
    ((16, 128, 64, 60), -3),
    ((16, 7, 57, 32, 29), 3),
    ((16, 7, 57, 32, 29), -5),
]
_QUICK_DIM_ROWS = [row for row in _DIM_ROWS if row[0] == _QUICK_SHAPE]
DIM_CASES = tu.selected_cases(_DIM_ROWS, quick=_QUICK_DIM_ROWS)

# split_size boundaries: partial, exact, oversized and an empty split dimension
# (the only dimension where split_size 0 is legal).
_CHUNK_ROWS = [
    ((2, 19, 7), 2, 3),
    ((2, 19, 7), 2, 7),
    ((2, 19, 7), 2, 99),
    ((2, 19, 7), 2, 1),
    ((0, 3), 0, 0),
    ((256,), 0, 1),
    ((256,), 0, 256),
    ((20, 320, 15), 1, 320),
    ((16, 128, 64, 60), 2, 1),
    ((16, 128, 64, 60), 2, 129),
]
CHUNK_CASES = _CHUNK_ROWS

# Non-contiguous / offset / stride-0 inputs: the parts must keep the input's
# strides and storage offset instead of being copied to a dense layout.
_LAYOUT_ROWS = [
    ((6, 8), "transposed", None),
    ((6, 8), "column_step", None),
    ((8, 12), "both_steps", None),
    ((10, 12), "offset_window", None),
    ((1, 6), "expanded", (4, 6)),
]
LAYOUT_CASES = _LAYOUT_ROWS

_VIEW_ROWS = [
    ((8,), 0, 3),
    ((6, 4), 0, 2),
    ((6, 4), 1, 2),
    ((16, 7, 5), 2, 2),
]
VIEW_CASES = _VIEW_ROWS

_BACKWARD_ROWS = [
    ((6, 4), 0, 2),
    ((16, 32), 1, 5),
    ((7, 13, 29), 2, 3),
]
BACKWARD_CASES = tu.selected_cases(
    [
        (shape, dim, chunk, dtype)
        for shape, dim, chunk in _BACKWARD_ROWS
        for dtype in _BACKWARD_DTYPES
    ],
    quick=[],
)

# Zero-size parts are legal views; neither call form may drop them.
_ZERO_ROWS = [
    ((6, 4), 0, [2, 0, 4]),
    ((6, 4), 0, [6]),
    ((0, 3), 0, [0]),
]
ZERO_CASES = _ZERO_ROWS

# dim is omitted here, so the candidate's own signature has to carry the schema
# default; the reference is the aten packet, which takes an int or a list.
_DEFAULT_DIM_ROWS = [
    ((2, 19, 7), 6),
    ((6, 4), [2, 4]),
]
DEFAULT_DIM_CASES = _DEFAULT_DIM_ROWS

# Positive special values stay default-only.
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_SPLIT_DTYPES), quick=[])

# Native str semantics are not Python str.split: separator=None (the default)
# whitespace-tokenizes, drops the empty fields around the ends and ignores max
# completely, while an explicit separator honours max as a split bound
# (0 -> no split, negative -> unlimited) and keeps empty fields.
_STR_ROWS = [
    ("a,b,c", None, -1),
    ("a,b,c", None, 5),
    ("a b c", None, 1),
    ("  a  b ", None, -1),
    ("abc", None, -1),
    ("", None, -1),
    ("a,b,c", ",", -1),
    ("a,b,c", ",", 2),
    ("a,b,c", ",", 1),
    ("a,b,c", ",", 0),
    ("a,b,c", ",", 99),
    (",a,", ",", 1),
    (",a,", ",", 2),
    ("a,,b", ",", -1),
    ("aaa", "a", -1),
    ("aaa", "a", 1),
    ("aXbXXc", "XX", -1),
    ("abc", ",", -1),
    ("", ",", -1),
    ("中文,空格", ",", -1),
]
STR_CASES = _STR_ROWS

# separator and max are omitted, so the public candidate signature itself has to
# carry the schema defaults (separator=None, max=-1).
_STR_DEFAULT_ROWS = ["a,b,c", "  a  b ", "a b c", "", "a,,b"]
STR_DEFAULT_CASES = _STR_DEFAULT_ROWS


@pytest.mark.split
@pytest.mark.parametrize("shape", _SPLIT_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SPLIT_DTYPES)
def test_split_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    chunk = _chunk(shape[0])

    ref_parts = torch.ops.aten.split.Tensor(ref_inp, chunk, 0)
    res_parts = flag_gems.split(inp, chunk, 0)

    _assert_parts_equal(res_parts, ref_parts, inp)


@pytest.mark.split
@pytest.mark.parametrize("shape,dim,sizes", SIZES_CASES)
@pytest.mark.parametrize("dtype", _SIZES_DTYPES)
def test_split_sizes_form(shape, dim, sizes, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split.sizes(ref_inp, sizes, dim)
    res_parts = flag_gems.split(inp, sizes, dim)

    _assert_parts_equal(res_parts, ref_parts, inp)


@pytest.mark.split
@pytest.mark.parametrize("shape,dim", DIM_CASES)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_split_dim(shape, dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    chunk = _chunk(shape[dim])

    ref_parts = torch.ops.aten.split.Tensor(ref_inp, chunk, dim)
    res_parts = flag_gems.split(inp, chunk, dim)

    _assert_parts_equal(res_parts, ref_parts, inp)


@pytest.mark.split
@pytest.mark.parametrize("shape,dim,chunk", CHUNK_CASES)
def test_split_chunk_boundaries(shape, dim, chunk):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split.Tensor(ref_inp, chunk, dim)
    res_parts = flag_gems.split(inp, chunk, dim)

    _assert_parts_equal(res_parts, ref_parts, inp)


@pytest.mark.split
@pytest.mark.parametrize("shape,split_arg", DEFAULT_DIM_CASES)
@pytest.mark.parametrize("dtype", _DEFAULT_DIM_DTYPES)
def test_split_default_dim(shape, split_arg, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split(ref_inp, split_arg)
    res_parts = flag_gems.split(inp, split_arg)

    _assert_parts_equal(res_parts, ref_parts, inp)


def _apply_layout(base, layout, expand_shape=None):
    if layout == "transposed":
        return base.transpose(0, 1)
    if layout == "column_step":
        return base[:, ::2]
    if layout == "both_steps":
        return base[::3, ::4]
    if layout == "offset_window":
        return base[2:8, 1:9]
    if layout == "expanded":
        return base.expand(tuple(expand_shape))
    raise ValueError("unsupported layout " + repr(layout))


@pytest.mark.split
@pytest.mark.parametrize("storage_shape,layout,expand_shape", LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_split_strided_input(storage_shape, layout, expand_shape, dtype):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    inp = _apply_layout(base, layout, expand_shape)
    ref_inp = _apply_layout(ref_base, layout, expand_shape)
    chunk = _chunk(inp.shape[0])

    ref_parts = torch.ops.aten.split.Tensor(ref_inp, chunk, 0)
    res_parts = flag_gems.split(inp, chunk, 0)

    _assert_parts_equal(res_parts, ref_parts, inp)


@pytest.mark.split
@pytest.mark.parametrize("shape,dim,chunk", VIEW_CASES)
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
def test_split_view_writes_through(shape, dim, chunk, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    inp_before = tu.to_reference(inp.detach())

    ref_parts = torch.ops.aten.split.Tensor(ref_inp, chunk, dim)
    res_parts = flag_gems.split(inp, chunk, dim)

    _assert_parts_equal(res_parts, ref_parts, inp)
    # Reading the parts must not modify the input ...
    tu.assert_result_equal(inp, inp_before)

    # ... while writing through a part must reach the shared storage: that
    # write-through is what distinguishes a view from a copy.
    res_parts[0].fill_(3.0)
    ref_parts[0].fill_(3.0)
    tu.assert_result_equal(res_parts[0], ref_parts[0])
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split
@pytest.mark.parametrize("shape,dim,chunk,dtype", BACKWARD_CASES)
def test_split_backward(shape, dim, chunk, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    upstream = [
        tu.make_input(dtype, part_shape, ["-1", "1"])
        for part_shape in _part_shapes(shape, dim, chunk)
    ]
    ref_upstream = [tu.to_reference(grad.detach()) for grad in upstream]

    ref_parts = torch.ops.aten.split.Tensor(ref_inp, chunk, dim)
    res_parts = flag_gems.split(inp, chunk, dim)

    _assert_parts_equal(res_parts, ref_parts, inp)

    # The gradients flow through the original leaf, not through an output tensor
    # differentiated against itself.
    res_grad = torch.autograd.grad(list(res_parts), inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(list(ref_parts), ref_inp, grad_outputs=ref_upstream)[
        0
    ]

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.split
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_split_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split.Tensor(ref_inp, 2, 0)
    res_parts = flag_gems.split(inp, 2, 0)

    _assert_parts_equal(res_parts, ref_parts, inp)


@pytest.mark.split
@pytest.mark.parametrize("shape,dim,split_arg", ZERO_CASES)
def test_split_zero_size_parts(shape, dim, split_arg):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split(ref_inp, split_arg, dim)
    res_parts = flag_gems.split(inp, split_arg, dim)

    _assert_parts_equal(res_parts, ref_parts, inp)


@pytest.mark.split
@pytest.mark.parametrize("self_value,separator,max_value", STR_CASES)
def test_split_str_forms(self_value, separator, max_value):
    ref_parts = torch.ops.aten.split.str(self_value, separator, max_value)
    res_parts = flag_gems.split(self_value, separator, max_value)

    _assert_str_parts_equal(res_parts, ref_parts)


@pytest.mark.split
@pytest.mark.parametrize("self_value", STR_DEFAULT_CASES)
def test_split_str_default_arguments(self_value):
    ref_parts = torch.ops.aten.split.str(self_value)
    res_parts = flag_gems.split(self_value)

    _assert_str_parts_equal(res_parts, ref_parts)


@pytest.mark.split
@pytest.mark.parametrize("split_size", [0, -1])
def test_split_rejects_invalid_split_size(split_size):
    # 0 is only legal when the split dimension itself is empty.
    inp = tu.make_input(torch.float32, (6, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, IndexError, ValueError)):
        flag_gems.split(inp, split_size, 0)


@pytest.mark.split
@pytest.mark.parametrize("sizes", [[2, 3], [7, 1], [2, -2, 4], []])
def test_split_rejects_invalid_sizes(sizes):
    # The sizes form has to sum exactly to the split dimension extent and every
    # entry has to be non-negative.
    inp = tu.make_input(torch.float32, (6, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, IndexError, ValueError)):
        flag_gems.split(inp, sizes, 0)


@pytest.mark.split
@pytest.mark.parametrize("dim", [5, -3])
def test_split_rejects_dim_out_of_range(dim):
    inp = tu.make_input(torch.float32, (6, 4), ["-1", "1"])

    with pytest.raises((IndexError, RuntimeError, ValueError)):
        flag_gems.split(inp, 2, dim)


@pytest.mark.split
def test_split_rejects_0d_input():
    inp = tu.make_input(torch.float32, (), ["-1", "1"])

    with pytest.raises((RuntimeError, IndexError, ValueError)):
        flag_gems.split(inp, 1, 0)


@pytest.mark.split
@pytest.mark.parametrize("self_value", [3.14, "abc", [[0.0, 1.0]]])
def test_split_rejects_non_tensor_self(self_value):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.split(self_value, 1, 0)


@pytest.mark.split
@pytest.mark.parametrize(
    "self_value,separator,max_value",
    [
        ("", "", -1),
        (None, ",", -1),
        (["a"], ",", -1),
        ("a,b", 5, -1),
        ("a,b", ",", 1.5),
    ],
)
def test_split_str_rejects_invalid_arguments(self_value, separator, max_value):
    # empty separator / non-str self / non-str separator / non-int max
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.split(self_value, separator, max_value)
