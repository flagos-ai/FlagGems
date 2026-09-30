# Copyright 2026, The FlagGems Authors.
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

"""Correctness tests for _convert_weight_to_int4pack_for_cpu.

CPU-only operator: the argument is an int32 CPU weight matrix (N, K) with N a
multiple of 16 and even K, and the result is a CPU uint8 (N, K / 2) tensor in
the int4 GEMM packing layout. That layout's block width follows the host CPU
vector width (64 bytes with AVX512, 32 with AVX2), so there is no portable
Python oracle: the native operator is the oracle for every case, and reference
and candidate receive the same CPU int32 weight.
"""

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Native validation accepts int32 weights only ("expect weight to be kInt.");
# every other dtype is covered as a rejection case.
_DTYPE = torch.int32
_PARAM_SHAPE = (32, 64)

# 0/positive/negative plus the int64 boundary values of the required int
# argument; the native implementation ignores the value.
_TILE_ROWS = [0, 1, -1, 2, 4, 8, 16, 2**31 - 1, -(2**31)]


def _cpu_input(shape, value_range, dtype=_DTYPE):
    """CPU input built through the shared value-range generator.

    tu.make_input allocates on the active accelerator, so the generated values
    are moved to CPU: this operator's argument type is a CPU int32 tensor and
    both the reference and the candidate receive it unchanged.
    """
    return tu.make_input(dtype, shape, value_range).to("cpu")


def _strided_input(shape, value_range, strides, offset):
    """CPU int32 view of a larger storage with the requested strides/offset.

    The weight buffer is zero-filled and then completely written through the
    view, so no uninitialized element is ever read. The native implementation
    applies .contiguous() first, so a strided argument must pack exactly like
    its contiguous copy.
    """
    values = _cpu_input(shape, value_range)
    span = offset + 1 + sum((extent - 1) * step for extent, step in zip(shape, strides))
    backing = torch.zeros(span, dtype=values.dtype)
    view = backing.as_strided(shape, strides, offset)
    view.copy_(values)
    return view


_SPEC_SHAPES = [
    tuple(shape)
    for shape in tu.selected_shapes()
    if len(shape) == 2 and shape[0] % 16 == 0 and shape[1] % 2 == 0
]
# The operator is fixed-rank 2-D, so the spec grid keeps its single 2-D member
# and adds the N/K boundaries: N = 16/32/48 stay below the packing block width
# while N = 64/128/256 cross it, covering the partial- and full-block host
# dispatch paths for both the 64-byte (AVX512) and 32-byte (AVX2) block widths;
# K = 2 is the smallest legal row.
_LOCAL_SHAPES = [
    (0, 2),
    (16, 0),
    (0, 0),
    (16, 2),
    (16, 8),
    (16, 16),
    (16, 64),
    (16, 1024),
    (32, 4),
    (48, 32),
    (64, 8),
    (64, 64),
    (128, 64),
    (256, 128),
]
_SHAPES = list(dict.fromkeys(_SPEC_SHAPES + _LOCAL_SHAPES))

_GRID_ROWS = [
    (shape, value_range) for shape in _SHAPES for value_range in tu.selected_ranges()
]
# Quick keeps empty, smallest-K and partial/full-block boundaries.
# Every selected shape and range also occurs in the default grid.
_QUICK_GRID_ROWS = [
    (shape, value_range)
    for shape in ((0, 2), (16, 0), (0, 0), (16, 2), (16, 8), (32, 4), (48, 32), (64, 8))
    for value_range in tu.selected_ranges()
]
_GRID_CASES = tu.selected_cases(_GRID_ROWS, quick=_QUICK_GRID_ROWS)


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize("shape,value_range", _GRID_CASES)
def test__convert_weight_to_int4pack_for_cpu(shape, value_range):
    inp = _cpu_input(shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_before = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_weight_to_int4pack_for_cpu(ref_inp, 2)
    res_out = flag_gems._convert_weight_to_int4pack_for_cpu(inp, 2)

    tu.assert_result_equal(res_out, ref_out)
    # Allocation metadata the shared value assertions do not cover.
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()
    # Functional operator: the weight argument must be left untouched.
    tu.assert_result_equal(inp, ref_before)


_TILE_CASES = tu.selected_cases(
    [
        (inner_k_tiles, value_range)
        for inner_k_tiles in _TILE_ROWS
        for value_range in tu.selected_ranges()
    ],
    quick=[
        (inner_k_tiles, value_range)
        for inner_k_tiles in _TILE_ROWS
        for value_range in tu.selected_ranges()
    ],
)


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize("inner_k_tiles,value_range", _TILE_CASES)
def test__convert_weight_to_int4pack_for_cpu_inner_k_tiles(inner_k_tiles, value_range):
    inp = _cpu_input(_PARAM_SHAPE, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_weight_to_int4pack_for_cpu(ref_inp, inner_k_tiles)
    res_out = flag_gems._convert_weight_to_int4pack_for_cpu(inp, inner_k_tiles)

    tu.assert_result_equal(res_out, ref_out)


# (label, shape, strides, storage offset)
_STRIDED_LAYOUTS = [
    ("row_stride_2", (16, 8), (16, 1), 0),
    ("col_stride_2_offset_1", (16, 8), (16, 2), 1),
    ("both_strided_offset_2", (32, 8), (48, 2), 2),
    ("transposed_storage_offset_3", (16, 16), (1, 16), 3),
    ("offset_only", (16, 2), (2, 1), 5),
]
_STRIDED_CASES = tu.selected_cases(
    [
        (label, shape, strides, offset, value_range)
        for label, shape, strides, offset in _STRIDED_LAYOUTS
        for value_range in tu.selected_ranges()
    ],
    quick=[
        (label, shape, strides, offset, value_range)
        for label, shape, strides, offset in _STRIDED_LAYOUTS
        for value_range in tu.selected_ranges()
    ],
)


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize("label,shape,strides,offset,value_range", _STRIDED_CASES)
def test__convert_weight_to_int4pack_for_cpu_strided_input(
    label, shape, strides, offset, value_range
):
    del label
    inp = _strided_input(shape, value_range, strides, offset)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_weight_to_int4pack_for_cpu(ref_inp, 2)
    res_out = flag_gems._convert_weight_to_int4pack_for_cpu(inp, 2)

    tu.assert_result_equal(res_out, ref_out)


# Deterministic byte/nibble patterns built directly in int32 (no cast), so the
# stored values are exactly the intended ones. The packing takes the low nibble
# of every weight element, which random sampling only covers statistically.
_BYTE_RAMP = ("byte_ramp", lambda numel, index: index & 0xFF)
_PATTERNS = [
    ("zeros", lambda numel, index: torch.zeros(numel, dtype=_DTYPE)),
    ("all_bits_set", lambda numel, index: torch.full((numel,), -1, dtype=_DTYPE)),
    _BYTE_RAMP,
    ("nibble_ramp", lambda numel, index: index & 0xF),
    ("high_nibble_ramp", lambda numel, index: (index & 0xF) << 4),
    (
        "mixed_nibbles",
        lambda numel, index: torch.where(
            index.remainder(2) == 0,
            torch.full((numel,), 0x0A, dtype=_DTYPE),
            torch.full((numel,), 0xF5, dtype=_DTYPE),
        ),
    ),
    (
        "int32_max",
        lambda numel, index: torch.full((numel,), 2**31 - 1, dtype=_DTYPE),
    ),
    (
        "int32_min",
        lambda numel, index: torch.full((numel,), -(2**31), dtype=_DTYPE),
    ),
]
_PATTERN_SHAPES = [(16, 8), (64, 64)]
_PATTERN_CASES = tu.selected_cases(
    [
        (label, shape, pattern)
        for label, pattern in _PATTERNS
        for shape in _PATTERN_SHAPES
    ],
    quick=[(label, _PATTERN_SHAPES[0], pattern) for label, pattern in _PATTERNS],
)


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize("label,shape,pattern", _PATTERN_CASES)
def test__convert_weight_to_int4pack_for_cpu_patterns(label, shape, pattern):
    del label
    numel = math.prod(shape)
    index = torch.arange(numel, dtype=_DTYPE)
    inp = pattern(numel, index).reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._convert_weight_to_int4pack_for_cpu(ref_inp, 2)
    res_out = flag_gems._convert_weight_to_int4pack_for_cpu(inp, 2)

    tu.assert_result_equal(res_out, ref_out)


# Negative rows are collected in both default and quick mode. Native validation
# reports RuntimeError; an argument/type mismatch reports TypeError or ValueError.
_REJECT_ERRORS = (RuntimeError, TypeError, ValueError)

_REJECTED_SHAPES = [
    ("n_8_not_divisible_by_16", (8, 2)),
    ("n_24_not_divisible_by_16", (24, 8)),
    ("n_1_not_divisible_by_16", (1, 2)),
    ("k_3_odd", (16, 3)),
    ("k_1_odd", (32, 1)),
    ("k_129_odd", (64, 129)),
    ("rank_1", (16,)),
    ("rank_3", (16, 2, 2)),
    ("rank_0", ()),
]


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize("label,shape", _REJECTED_SHAPES)
def test__convert_weight_to_int4pack_for_cpu_rejects_invalid_shape(label, shape):
    del label
    inp = torch.zeros(shape, dtype=_DTYPE)
    with pytest.raises(_REJECT_ERRORS):
        flag_gems._convert_weight_to_int4pack_for_cpu(inp, 2)


_REJECTED_DTYPES = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int64,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.complex64,
]
_REJECTED_DTYPE_ROWS = [
    (str(dtype).removeprefix("torch."), dtype) for dtype in _REJECTED_DTYPES
]


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize("label,dtype", _REJECTED_DTYPE_ROWS)
def test__convert_weight_to_int4pack_for_cpu_rejects_unsupported_dtype(label, dtype):
    del label
    inp = torch.zeros((16, 2), dtype=dtype)
    with pytest.raises(_REJECT_ERRORS):
        flag_gems._convert_weight_to_int4pack_for_cpu(inp, 2)


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize(
    "label,inner_k_tiles",
    [("float_tiles", 2.5), ("string_tiles", "2"), ("none_tiles", None)],
)
def test__convert_weight_to_int4pack_for_cpu_rejects_invalid_tiles(
    label, inner_k_tiles
):
    del label
    inp = torch.zeros((16, 2), dtype=_DTYPE)
    with pytest.raises(_REJECT_ERRORS):
        flag_gems._convert_weight_to_int4pack_for_cpu(inp, inner_k_tiles)


@pytest.mark.convert_weight_to_int4pack_for_cpu
@pytest.mark.parametrize(
    "label,weight", [("weight_is_list", [0] * 32), ("weight_is_int", 16)]
)
def test__convert_weight_to_int4pack_for_cpu_rejects_non_tensor_weight(label, weight):
    del label
    with pytest.raises(_REJECT_ERRORS):
        flag_gems._convert_weight_to_int4pack_for_cpu(weight, 2)


@pytest.mark.convert_weight_to_int4pack_for_cpu
def test__convert_weight_to_int4pack_for_cpu_rejects_missing_tiles_argument():
    # innerKTiles has no schema default, so omitting it is not a valid call.
    inp = torch.zeros((16, 2), dtype=_DTYPE)
    with pytest.raises(_REJECT_ERRORS):
        flag_gems._convert_weight_to_int4pack_for_cpu(inp)
