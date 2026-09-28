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

# aten::_sparse_semi_structured_apply_dense(input, threads_masks) selects each
# element of a dense 2-D tensor by its packed mask bit: the element is kept where
# the bit is 1 and replaced by zero where it is 0. Nothing is rounded, so every
# value comparison below uses the shared exact assertion. Selection is not a
# multiply: an all-zero mask maps NaN and +-Inf to zero rather than producing NaN,
# while an all-ones mask preserves their NaN / Inf categories. The special-value
# cases compare those native outcomes.
#
# Deviations from the generic range/shape grid, all implied by the
# implementation's own checks rather than by a runtime probe:
#   * rank 2 only: stride(0)/stride(1) are read directly, so rank 0 and rank 1
#     fail the stride query, and ranks 3-5, or any 2-D input with both strides
#     > 1, fail the RowMajor/ColMajor check. The larger shape levels are kept as
#     aligned 2-D equivalents below.
#   * tile-aligned positive extents only (rows % 32 == 0, cols % 64 == 0), see
#     the geometry note in the shape table.
#   * Half/BFloat16 only: the scalar-type check rejects every other dtype.
#   * no broadcast (the mask extents are derived from the dense input) and no
#     backward or scalar-operand form.

# Aligned 2-D equivalents of the spec's larger shape levels, keeping their element
# scales: (20, 320, 15) = 96000 -> (64, 1536) = 98304; (16, 128, 64, 60) = 7864320
# -> (1024, 7680) exactly; (16, 7, 57, 32, 29) = 5924352 has no aligned 2-D
# factorisation -> (352, 16832) = 5924864.
_SHAPE_BASIC = (32, 64)
_SHAPE_PLANAR = (64, 1536)
_SHAPE_4D = (1024, 7680)
_SHAPE_5D = (352, 16832)

_SHAPES = [
    _SHAPE_BASIC,
    (32, 128),
    (64, 128),
    (64, 192),
    (256, 256),
    (1024, 1024),
    _SHAPE_PLANAR,
    _SHAPE_4D,
    _SHAPE_5D,
    (2048, 64),
    (128, 3840),
]

# Static device capability flags, never a runtime probe: float16 always exists,
# bfloat16 only where the device advertises it.
_DTYPES = [torch.float16] + ([torch.bfloat16] if utils.bf16_is_supported else [])

_ROWS_MAJOR = "rowmajor"
_COLS_MAJOR = "colmajor"
# stride(1) == 1 and stride(0) == 1 select the two kernel instantiations.
_LAYOUTS = [_ROWS_MAJOR, _COLS_MAJOR]

_SEED_PATTERN = "seed"
# Packed-bit patterns besides the seeded one: all ones (identity), all zeros, two
# isolated bits, and alternating 0x55/0xAA words so that a word and its
# complement both appear.
_EXTRA_PATTERNS = ["ones", "zeros", "single", "complement"]
_PATTERN_SHAPES = [_SHAPE_BASIC, (256, 256), (128, 3840)]

_MAIN_CASES = [
    (shape, dtype, value_range, layout, _SEED_PATTERN)
    for shape in _SHAPES
    for dtype in _DTYPES
    for value_range in tu.REQUIRED_RANGES
    for layout in _LAYOUTS
] + [
    (shape, dtype, ["-1", "1"], layout, pattern)
    for shape in _PATTERN_SHAPES
    for dtype in _DTYPES
    for layout in _LAYOUTS
    for pattern in _EXTRA_PATTERNS
]

# Quick keeps the supported dtype table, one compact valid shape, the default
# seeded mask and the default RowMajor layout. Every other positive shape,
# pattern and layout is default-only through this single selection.
_QUICK_MAIN_CASES = [
    (_SHAPE_BASIC, dtype, ["-1", "1"], _ROWS_MAJOR, _SEED_PATTERN) for dtype in _DTYPES
]

_MAIN_CASES = tu.selected_cases(_MAIN_CASES, quick=_QUICK_MAIN_CASES)

_SPECIAL_SHAPE = _SHAPE_BASIC
_SPECIAL_PATTERNS = ["ones", _SEED_PATTERN, "zeros"]

# Positive special values are default-only: the mask pattern selects per element,
# so all ones preserves the NaN / Inf categories of the payload, the seeded
# pattern keeps or zeroes them per bit, and all zeros replaces every element with
# zero, NaN and Inf included.
_SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario, pattern)
        for dtype, scenario in tu.special_value_cases(_DTYPES)
        for pattern in _SPECIAL_PATTERNS
    ],
    quick=[],
)

# Mask storage layouts that carry valid numerical coverage. Both hold exactly the
# values of their contiguous copy: `plain` is contiguous, and `offset` is a view
# of a larger buffer whose stride(0) is still packed.
_MASK_LAYOUTS = ["plain", "offset"]

# Excluded, with no case derived from it: a row-strided mask view (buffer[::2],
# stride(0) != 8 * size(1)). The mask checks read dim(), size(0), size(1),
# stride(1), size(2), stride(2) and the dtype only, so the measured build ignores
# the mask's stride(0); row-strided mask semantics remain unresolved and are not
# claimed covered. No case or assertion below references that view.
_MASK_LAYOUT_SHAPES = [_SHAPE_BASIC, (256, 256), (1024, 1024)]

_MASK_LAYOUT_CASES = tu.selected_cases(
    [
        (shape, layout, dtype)
        for shape in _MASK_LAYOUT_SHAPES
        for layout in _MASK_LAYOUTS
        for dtype in _DTYPES
    ],
    quick=[],
)

# A nonzero storage offset input view keeps stride(1) == 1 (RowMajor) or
# stride(0) == 1 (ColMajor), so it is a valid operand and holds the same values
# as its contiguous copy; it is default-only layout coverage.
_VIEW_SHAPES = [_SHAPE_BASIC, (64, 128), (256, 256)]
_VIEW_OPERANDS = ["input", "mask"]

_VIEW_CASES = tu.selected_cases(
    [
        (shape, layout, operand, dtype)
        for shape in _VIEW_SHAPES
        for layout in _LAYOUTS
        for operand in _VIEW_OPERANDS
        for dtype in _DTYPES
    ],
    quick=[],
)

_VIEW_PAD = 8


def _mask_shape(rows, cols):
    return (4 * ((rows + 31) // 32), 8 * ((cols + 63) // 64), 8)


def _rand_bytes(shape, device, seed):
    # threads_masks is a device operand, so its packed bits are produced on the
    # target device: no CPU roundtrip and no mutation of any global RNG.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randint(
        0, 256, shape, dtype=torch.uint8, device=device, generator=generator
    )


def _make_mask(rows, cols, pattern, device, seed=0):
    """Build a contract-valid ``threads_masks`` tensor on ``device``."""
    shape = _mask_shape(rows, cols)
    if pattern == "ones":
        return torch.full(shape, 0xFF, dtype=torch.uint8, device=device)
    if pattern == "zeros":
        return torch.zeros(shape, dtype=torch.uint8, device=device)
    if pattern == "single":
        # Two isolated packed bits, one in the first tile and one in the last.
        mask = torch.zeros(shape, dtype=torch.uint8, device=device)
        mask[0, 0, 0] = 0x01
        mask[-1, -1, -1] = 0x80
        return mask
    if pattern == "complement":
        # Alternating 0x55 / 0xAA words: a packed word and its complement. The
        # index/parity tensors are explicitly int32: the packed dimensions stay
        # well inside int32, and a default int64 intermediate would allocate on a
        # configuration without int64 support.
        row_index = torch.arange(shape[0], dtype=torch.int32, device=device).view(-1, 1)
        col_index = torch.arange(shape[1], dtype=torch.int32, device=device).view(1, -1)
        parity = ((row_index + col_index) % 2).to(torch.uint8).unsqueeze(-1)
        base = torch.full(shape, 0x55, dtype=torch.uint8, device=device)
        return base ^ (parity * 0xFF)
    return _rand_bytes(shape, device, seed)


def _make_offset_mask(rows, cols, device, seed=1):
    """A packed mask read from a larger buffer: nonzero storage offset only."""
    shape = _mask_shape(rows, cols)
    total = shape[0] * shape[1] * shape[2]
    return _rand_bytes((total + _VIEW_PAD,), device, seed)[_VIEW_PAD:].view(shape)


def _colmajor(inp):
    """The same values as a 2-D tensor whose stride(0) == 1."""
    return inp.t().contiguous().t()


def _offset_input(rows, cols, layout, dtype, value_range):
    """A 2-D view of a larger allocation, i.e. with a nonzero storage offset."""
    pad = _VIEW_PAD
    if layout == _ROWS_MAJOR:
        base = tu.make_input(dtype, (rows + pad, cols + 2 * pad), value_range)
        return base[pad : pad + rows, pad : pad + cols]
    base = tu.make_input(dtype, (cols, rows + 2 * pad), value_range)
    return base[:, pad : pad + rows].t()


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("shape,dtype,value_range,layout,mask_pattern", _MAIN_CASES)
def test__sparse_semi_structured_apply_dense(
    shape, dtype, value_range, layout, mask_pattern
):
    inp = tu.make_input(dtype, shape, value_range)
    if layout == _COLS_MAJOR:
        inp = _colmajor(inp)
    ref_inp = tu.to_reference(inp)

    mask = _make_mask(shape[0], shape[1], mask_pattern, flag_gems.device)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_semi_structured_apply_dense(ref_inp, ref_mask)
    res_out = flag_gems._sparse_semi_structured_apply_dense(inp, mask)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("dtype,scenario,mask_pattern", _SPECIAL_CASES)
def test__sparse_semi_structured_apply_dense_special_values(
    dtype, scenario, mask_pattern
):
    values = tu.make_special_input(dtype, scenario)
    # Tile the NaN / Inf / signed-zero payload over the whole grid so every mask
    # tile sees special values. The index buffer is 2048 positions, allocated as
    # int32 so no default int64 intermediate is created.
    index = (
        torch.arange(
            _SPECIAL_SHAPE[0] * _SPECIAL_SHAPE[1],
            dtype=torch.int32,
            device=flag_gems.device,
        )
        % values.numel()
    )
    inp = values[index].reshape(_SPECIAL_SHAPE)
    ref_inp = tu.to_reference(inp)

    mask = _make_mask(
        _SPECIAL_SHAPE[0], _SPECIAL_SHAPE[1], mask_pattern, flag_gems.device
    )
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_semi_structured_apply_dense(ref_inp, ref_mask)
    res_out = flag_gems._sparse_semi_structured_apply_dense(inp, mask)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("shape,layout,dtype", _MASK_LAYOUT_CASES)
def test__sparse_semi_structured_apply_dense_mask_layout(shape, layout, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    if layout == "plain":
        mask = _make_mask(shape[0], shape[1], _SEED_PATTERN, flag_gems.device)
    else:
        mask = _make_offset_mask(shape[0], shape[1], flag_gems.device)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_semi_structured_apply_dense(ref_inp, ref_mask)
    res_out = flag_gems._sparse_semi_structured_apply_dense(inp, mask)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("shape,layout,operand,dtype", _VIEW_CASES)
def test__sparse_semi_structured_apply_dense_offset_views(
    shape, layout, operand, dtype
):
    rows, cols = shape
    if operand == "input":
        inp = _offset_input(rows, cols, layout, dtype, ["-1", "1"])
        mask = _make_mask(rows, cols, _SEED_PATTERN, flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, ["-1", "1"])
        if layout == _COLS_MAJOR:
            inp = _colmajor(inp)
        mask = _make_offset_mask(rows, cols, flag_gems.device)
    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten._sparse_semi_structured_apply_dense(ref_inp, ref_mask)
    res_out = flag_gems._sparse_semi_structured_apply_dense(inp, mask)

    tu.assert_result_equal(res_out, ref_out)


# The negative families below assert the rejections of the operator's CUDA
# implementation, which the compile guard (USE_ROCM / _MSC_VER / CUDA_VERSION <
# 11.8) excludes from other builds and which native_functions.yaml registers under
# the CUDA dispatch key only. They are scoped to the measured vendor by the static
# tables below.
_NEGATIVE_CHECKS_VENDOR = "nvidia"
_NEGATIVE_CHECKS_ACTIVE = flag_gems.vendor_name == _NEGATIVE_CHECKS_VENDOR

# The scalar-type check runs before any data access, so these inputs are all
# zeros: a random generator would already fail for the integer, bool and float8
# dtypes and would prove nothing about the operator.
_UNSUPPORTED_DTYPE_CANDIDATES = [
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.int32,
]
if utils.fp64_is_supported:
    _UNSUPPORTED_DTYPE_CANDIDATES.append(torch.float64)
if utils.int64_is_supported:
    _UNSUPPORTED_DTYPE_CANDIDATES.append(torch.int64)
if utils.fp8_is_supported:
    _UNSUPPORTED_DTYPE_CANDIDATES.extend([torch.float8_e4m3fn, torch.float8_e5m2])

# Rank 0 raises IndexError from the stride(0) query, rank 1 raises IndexError from
# the size(1) query, and ranks 3-5 are contiguous with both strides > 1 and fail
# the RowMajor/ColMajor check. Every one of them fires before any kernel launch.
_INVALID_RANK_CANDIDATES = [(), (256,), (2, 32, 64), (2, 32, 64, 4), (1, 2, 32, 64, 4)]

# A 2-D slice with both strides > 1 is neither RowMajor nor ColMajor.
_INVALID_LAYOUT_CANDIDATES = ["split_stride"]

# One rejected mask defect per case: wrong dtype, wrong rank, and wrong packed
# tile row/column counts.
_MASK_DEFECT_CANDIDATES = ["dtype_int32", "dtype_bool", "rank", "size_x", "size_y"]

_UNSUPPORTED_DTYPES = (
    list(_UNSUPPORTED_DTYPE_CANDIDATES) if _NEGATIVE_CHECKS_ACTIVE else []
)
_INVALID_RANKS = list(_INVALID_RANK_CANDIDATES) if _NEGATIVE_CHECKS_ACTIVE else []
_INVALID_LAYOUTS = list(_INVALID_LAYOUT_CANDIDATES) if _NEGATIVE_CHECKS_ACTIVE else []
_MASK_DEFECTS = list(_MASK_DEFECT_CANDIDATES) if _NEGATIVE_CHECKS_ACTIVE else []


def _invalid_mask(defect, device):
    """Build a ``threads_masks`` tensor that violates exactly one rule."""
    good = _make_mask(32, 64, "ones", device)
    if defect == "dtype_int32":
        return good.to(torch.int32)
    if defect == "dtype_bool":
        return good.bool()
    if defect == "rank":
        return good.reshape(-1, 8)
    if defect == "size_x":
        return good[:-1]
    return good[:, :-1]


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__sparse_semi_structured_apply_dense_unsupported_dtype(dtype):
    inp = torch.zeros(_SHAPE_BASIC, dtype=dtype, device=flag_gems.device)
    mask = _make_mask(_SHAPE_BASIC[0], _SHAPE_BASIC[1], "ones", flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply_dense(inp, mask)


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("shape", _INVALID_RANKS)
def test__sparse_semi_structured_apply_dense_invalid_input_rank(shape):
    inp = torch.zeros(shape, dtype=torch.float16, device=flag_gems.device)
    mask = _make_mask(_SHAPE_BASIC[0], _SHAPE_BASIC[1], "ones", flag_gems.device)

    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._sparse_semi_structured_apply_dense(inp, mask)


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("layout_case", _INVALID_LAYOUTS)
def test__sparse_semi_structured_apply_dense_invalid_input_layout(layout_case):
    # The table carries the single case "split_stride" for vendor scoping; the
    # operand itself is a 2-D slice whose two strides are both > 1.
    del layout_case
    base = torch.zeros((32, 128), dtype=torch.float16, device=flag_gems.device)
    inp = base[:, ::2]
    mask = _make_mask(_SHAPE_BASIC[0], _SHAPE_BASIC[1], "ones", flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply_dense(inp, mask)


@pytest.mark.sparse_semi_structured_apply_dense
@pytest.mark.parametrize("defect", _MASK_DEFECTS)
def test__sparse_semi_structured_apply_dense_invalid_mask(defect):
    inp = torch.zeros(_SHAPE_BASIC, dtype=torch.float16, device=flag_gems.device)
    mask = _invalid_mask(defect, flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply_dense(inp, mask)
