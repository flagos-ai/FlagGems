# Copyright 2025, The FlagGems Authors.
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

"""Correctness tests for ``_convert_indices_from_csr_to_coo``.

Both operands are integer index tensors, so the spec's five value ranges are
used as stored index values, which leaves no nan/inf matrix
(``tu.special_value_cases()`` is empty for integer dtypes), and backward does
not apply because integer tensors cannot require grad.  The native operator
requires crow and col to have the same rank with matching leading batch
extents rather than broadcasting, so mismatched batches are covered as
rejection cases.  Every positive case keeps a valid csr structure: the offsets
are monotonic and a block's terminal count matches the stored col entries of
that block.  Negative cases keep valid index values too, so the contract named
in each row is the only invalid one.  Case data is always selected through the
shared selectors, so range bounds use the framework's symbol vocabulary instead
of hand-written literals.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_SUPPORTED_DTYPES = [torch.int8, torch.uint8, torch.int16, torch.int32]
if utils.int64_is_supported:
    _SUPPORTED_DTYPES.append(torch.int64)

_CROW_DTYPE = torch.int32
_QUICK_SHAPE = (2, 19, 7)
_SUPPLEMENT_SHAPE = (20, 320, 15)

# Malformed extents/dims surface as RuntimeError, ValueError or IndexError
# depending on which check trips first.
_STRUCTURE_ERRORS = (RuntimeError, ValueError, IndexError)
_KERNEL_ERRORS = (RuntimeError,)
_SIGNATURE_ERRORS = (TypeError, RuntimeError)
_OUT_ERRORS = (RuntimeError, TypeError)

# The schema default output is int64.  When the backend cannot produce it every
# family still makes a valid call through the explicitly supported int32 output
# form instead of having the default silently rewritten.
_DEFAULT_OUT_KWARGS = {} if utils.int64_is_supported else {"out_int32": True}


def _numel(shape):
    total = 1
    for extent in shape:
        total *= extent
    return total


def _offset_capacity(dtype):
    return int(torch.iinfo(dtype).max)


def _index_tensor(values, dtype):
    return torch.tensor(values, dtype=dtype, device=flag_gems.device)


def _csr_offsets(col_shape, dtype):
    """Monotonic csr pointers with exactly one stored entry per row.

    A block holds ``col_shape[-1]`` rows with one entry each, so its offsets
    never exceed the stored extent and stay representable whenever that extent
    is (the grid keeps only such extents).
    """
    rows = col_shape[-1] + 1
    offsets = torch.arange(rows, dtype=dtype, device=flag_gems.device)
    return offsets.repeat(_numel(col_shape[:-1])).reshape(col_shape[:-1] + (rows,))


def _strided_view(values, shape, stride, offset):
    """View of ``values`` with the requested stride and storage offset."""
    flat = values.reshape(-1)
    span = offset + flat.numel() * stride
    backing = torch.zeros(span, dtype=flat.dtype, device=flag_gems.device)
    backing[offset:span:stride] = flat
    return backing[offset:span:stride].view(shape)


# Row = (col shape, index dtype).  The boundary rows use a last extent equal to
# the dtype capacity, so their offsets reach the largest representable value.
# The spec's 0-D shape is dropped: a 0-D col has no addressable terminal extent
# and is covered by the zero-dim rejection test instead.
_SPEC_COMBOS = [
    (tuple(shape), dtype)
    for dtype in _SUPPORTED_DTYPES
    for shape in tu.selected_shapes()
    if len(shape) >= 1 and shape[-1] <= _offset_capacity(dtype)
]
_BOUNDARY_COMBOS = [
    ((127,), torch.int8),
    ((4, 127), torch.int8),
    ((255,), torch.uint8),
    ((4, 255), torch.uint8),
    ((32767,), torch.int16),
    ((4, 32767), torch.int16),
]
_GRID_ROWS = [
    (shape, dtype, value_range)
    for shape, dtype in _SPEC_COMBOS + _BOUNDARY_COMBOS
    for value_range in tu.selected_ranges()
]
# Quick keeps the spec's single small shape; the range comes from the shared
# selector so it uses the framework's bound vocabulary rather than a literal.
_QUICK_GRID_ROWS = [
    (_QUICK_SHAPE, dtype, value_range)
    for dtype in _SUPPORTED_DTYPES
    for value_range in tu.selected_ranges()
]
_GRID_CASES = tu.selected_cases(_GRID_ROWS, quick=_QUICK_GRID_ROWS)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("shape,dtype,value_range", _GRID_CASES)
def test__convert_indices_from_csr_to_coo(shape, dtype, value_range):
    crow = _csr_offsets(shape, dtype)
    col = tu.make_input(dtype, shape, value_range)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        ref_crow, ref_col, **_DEFAULT_OUT_KWARGS
    )
    res_out = flag_gems._convert_indices_from_csr_to_coo(
        crow, col, **_DEFAULT_OUT_KWARGS
    )

    tu.assert_result_equal(res_out, ref_out)


# Every flag combination, including the call that omits the argument.  The
# omitted-default row is dropped (never rewritten) when the backend cannot
# produce the default int64 output; the explicit int32 rows are kept.
_ALL_FLAG_ROWS = [
    ("flags_omitted", {}),
    ("out_int32_false", {"out_int32": False}),
    ("out_int32_true", {"out_int32": True}),
    ("transpose_true", {"transpose": True}),
    ("both_true", {"out_int32": True, "transpose": True}),
]
_FLAG_ROWS = [
    (label, flags)
    for label, flags in _ALL_FLAG_ROWS
    if flags.get("out_int32", False) or utils.int64_is_supported
]
_FLAG_CASES = tu.selected_cases(
    [
        (_SUPPLEMENT_SHAPE, label, flags, value_range)
        for label, flags in _FLAG_ROWS
        for value_range in tu.selected_ranges()
    ],
    quick=[],
)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("shape,label,flags,value_range", _FLAG_CASES)
def test__convert_indices_from_csr_to_coo_flags(shape, label, flags, value_range):
    del label
    crow = _csr_offsets(shape, _CROW_DTYPE)
    col = tu.make_input(_CROW_DTYPE, shape, value_range)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        ref_crow, ref_col, **flags
    )
    res_out = flag_gems._convert_indices_from_csr_to_coo(crow, col, **flags)

    tu.assert_result_equal(res_out, ref_out)


# Output-buffer rows keep every flag combination whose expected output dtype the
# backend can produce.  The main grid's quick subset already exercises the
# explicit int32 output on a device without int64 output support, so the
# supplement shape stays default-only here.
_OUT_FLAG_ROWS = [
    (out_int32, transpose)
    for out_int32 in (False, True)
    for transpose in (False, True)
    if out_int32 or utils.int64_is_supported
]
_OUT_CASES = tu.selected_cases(
    [
        (_SUPPLEMENT_SHAPE, dtype, out_int32, transpose, value_range)
        for dtype in _SUPPORTED_DTYPES
        for out_int32, transpose in _OUT_FLAG_ROWS
        for value_range in tu.selected_ranges()
    ],
    quick=[],
)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("shape,dtype,out_int32,transpose,value_range", _OUT_CASES)
def test__convert_indices_from_csr_to_coo_out(
    shape, dtype, out_int32, transpose, value_range
):
    crow = _csr_offsets(shape, dtype)
    col = tu.make_input(dtype, shape, value_range)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        ref_crow, ref_col, out_int32=out_int32, transpose=transpose
    )
    buf = torch.zeros(ref_out.shape, dtype=ref_out.dtype, device=flag_gems.device)
    res_out = flag_gems._convert_indices_from_csr_to_coo(
        crow, col, out_int32=out_int32, transpose=transpose, out=buf
    )

    assert res_out is buf, "the out overload must write into the provided tensor"
    tu.assert_result_equal(res_out, ref_out)


# Named csr distributions.  Batched entries use rectangular blocks whose
# terminal count matches the stored entries of every block.
_CROW_FAMILIES = [
    ("identity_rows", [0, 1, 2, 3, 4]),
    ("zero_rows_zero_nnz", [0, 0, 0, 0]),
    ("empty_middle_rows", [0, 1, 1, 2, 2]),
    ("stored_only_first_row", [0, 3, 3, 3]),
    ("stored_only_last_row", [0, 0, 0, 3]),
    ("uneven_rows", [0, 1, 3, 3, 4]),
    ("all_rows_empty", [0, 0]),
    ("single_row", [0, 2]),
    ("two_rows", [0, 1, 2]),
    ("batched_identity", [[0, 1, 2], [0, 1, 2]]),
    ("batched_uneven_rows", [[0, 1, 1, 2], [0, 0, 0, 2]]),
    ("batched_one_empty_row", [[0, 1, 1, 2], [0, 1, 2, 2]]),
    ("batched_zero_nnz", [[0, 0, 0], [0, 0, 0]]),
    ("batched_independent_offsets", [[0, 2, 2, 2], [0, 0, 1, 2]]),
]
_CROW_CASES = tu.selected_cases(
    [
        (name, crow_spec, value_range)
        for name, crow_spec in _CROW_FAMILIES
        for value_range in tu.selected_ranges()
    ],
    quick=[],
)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("name,crow_spec,value_range", _CROW_CASES)
def test__convert_indices_from_csr_to_coo_crow_distribution(
    name, crow_spec, value_range
):
    del name
    crow = _index_tensor(crow_spec, _CROW_DTYPE)
    if crow.dim() == 1:
        col_shape = (int(crow[-1]),)
    else:
        col_shape = (crow.shape[0], int(crow[0, -1]))
    col = tu.make_input(_CROW_DTYPE, col_shape, value_range)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        ref_crow, ref_col, **_DEFAULT_OUT_KWARGS
    )
    res_out = flag_gems._convert_indices_from_csr_to_coo(
        crow, col, **_DEFAULT_OUT_KWARGS
    )

    tu.assert_result_equal(res_out, ref_out)


_CROW_SPEC = [0, 1, 1, 2]
_COL_LEN = 2
# (label, crow stride, crow offset, col stride, col offset)
_STRIDED_LAYOUTS = [
    ("col_stride_4", 1, 0, 4, 0),
    ("col_stride_2_offset_1", 1, 0, 2, 1),
    ("crow_stride_2_offset_1", 2, 1, 1, 0),
    ("crow_stride_3_col_stride_3_offset_2", 3, 2, 3, 2),
    ("unit_stride_crow_offset_2_col_offset_3", 1, 2, 1, 3),
]
_STRIDED_CASES = tu.selected_cases(
    [
        (label, crow_stride, crow_offset, col_stride, col_offset, value_range)
        for label, crow_stride, crow_offset, col_stride, col_offset in _STRIDED_LAYOUTS
        for value_range in tu.selected_ranges()
    ],
    quick=[],
)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize(
    "label,crow_stride,crow_offset,col_stride,col_offset,value_range", _STRIDED_CASES
)
def test__convert_indices_from_csr_to_coo_strided_inputs(
    label, crow_stride, crow_offset, col_stride, col_offset, value_range
):
    del label
    crow = _strided_view(
        _index_tensor(_CROW_SPEC, _CROW_DTYPE),
        (len(_CROW_SPEC),),
        crow_stride,
        crow_offset,
    )
    col = _strided_view(
        tu.make_input(_CROW_DTYPE, (_COL_LEN,), value_range),
        (_COL_LEN,),
        col_stride,
        col_offset,
    )
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        ref_crow, ref_col, **_DEFAULT_OUT_KWARGS
    )
    res_out = flag_gems._convert_indices_from_csr_to_coo(
        crow, col, **_DEFAULT_OUT_KWARGS
    )

    tu.assert_result_equal(res_out, ref_out)


_ZERO_DIM_CASES = [
    ("col_0d", [0, 1, 1], 3),
    ("crow_0d", 0, [0, 1]),
    ("both_0d", 0, 3),
]


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("label,crow_spec,col_spec", _ZERO_DIM_CASES)
def test__convert_indices_from_csr_to_coo_rejects_zero_dim_operand(
    label, crow_spec, col_spec
):
    del label
    crow = _index_tensor(crow_spec, _CROW_DTYPE)
    col = _index_tensor(col_spec, _CROW_DTYPE)
    with pytest.raises(_STRUCTURE_ERRORS):
        flag_gems._convert_indices_from_csr_to_coo(crow, col)


_BATCH_CROW = [[0, 2, 2], [0, 2, 2]]
_SHAPE_MISMATCH_CASES = [
    # crow is (2, 3) and each col block stores 2 entries matching its terminal
    # count, so the leading batch extent of col is the single invalid contract:
    # native validation requires the batch shapes to match ((2,) vs (k,)).
    ("batch_extent_3_vs_2", _BATCH_CROW, (3, 2)),
    ("batch_extent_4_vs_2", _BATCH_CROW, (4, 2)),
    ("batch_extent_5_vs_2", _BATCH_CROW, (5, 2)),
    ("batch_extent_6_vs_2", _BATCH_CROW, (6, 2)),
    # These rows violate the equal-rank requirement ("crow_indices and
    # col_indices are supposed to have the same dimensionality").
    ("crow_1d_col_2d", [0, 1, 1], (2, 2)),
    ("crow_2d_col_3d", _BATCH_CROW, (2, 2, 2)),
]


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("label,crow_spec,col_shape", _SHAPE_MISMATCH_CASES)
def test__convert_indices_from_csr_to_coo_rejects_shape_mismatch(
    label, crow_spec, col_shape
):
    del label
    crow = _index_tensor(crow_spec, _CROW_DTYPE)
    # Zero is a valid column index, so the malformed extent/rank is the only
    # invalid contract and no malformed pointer structure is executed.
    col = torch.zeros(col_shape, dtype=_CROW_DTYPE, device=flag_gems.device)
    with pytest.raises(_STRUCTURE_ERRORS):
        flag_gems._convert_indices_from_csr_to_coo(crow, col)


def _rejected_dtype_pair(dtype):
    """Valid csr structure whose index dtype is then cast to a rejected type.

    The template is monotonic with a terminal count equal to the stored col
    entries, so the dtype is the only invalid contract.  bool cannot represent
    4, so it uses a single stored entry (0/1 pointers).
    """
    entries = 1 if dtype is torch.bool else 4
    crow = torch.arange(entries + 1, dtype=torch.int32, device=flag_gems.device)
    col = torch.arange(entries, dtype=torch.int32, device=flag_gems.device)
    return crow.to(dtype), col.to(dtype)


# The device kernel rejects these index dtypes with
# "convert_indices_from_csr_to_coo_cuda not implemented for '<Type>'", which is
# vendor specific, so the whole default and quick selection is gated by the
# active vendor.  float64/complex128 sit behind the fp64 capability and
# complex64 is unconditional.
_UNSUPPORTED_DTYPE_ROWS = []
if flag_gems.runtime.device.vendor_name == "nvidia":
    _UNSUPPORTED_DTYPE_ROWS = [
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("complex64", torch.complex64),
        ("bool", torch.bool),
    ]
    if utils.bf16_is_supported:
        _UNSUPPORTED_DTYPE_ROWS.append(("bfloat16", torch.bfloat16))
    if utils.fp64_is_supported:
        _UNSUPPORTED_DTYPE_ROWS.append(("float64", torch.float64))
        _UNSUPPORTED_DTYPE_ROWS.append(("complex128", torch.complex128))
    if utils.fp8_is_supported:
        _UNSUPPORTED_DTYPE_ROWS.append(("float8_e4m3fn", torch.float8_e4m3fn))
        _UNSUPPORTED_DTYPE_ROWS.append(("float8_e5m2", torch.float8_e5m2))
_UNSUPPORTED_DTYPE_CASES = tu.selected_cases(
    _UNSUPPORTED_DTYPE_ROWS,
    quick=[row for row in _UNSUPPORTED_DTYPE_ROWS if row[0] == "float32"],
)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("label,dtype", _UNSUPPORTED_DTYPE_CASES)
def test__convert_indices_from_csr_to_coo_rejects_unsupported_dtype(label, dtype):
    del label
    crow, col = _rejected_dtype_pair(dtype)
    with pytest.raises(_KERNEL_ERRORS):
        flag_gems._convert_indices_from_csr_to_coo(crow, col)


@pytest.mark.convert_indices_from_csr_to_coo
def test__convert_indices_from_csr_to_coo_rejects_positional_flags():
    crow = _csr_offsets((4,), _CROW_DTYPE)
    col = torch.zeros(4, dtype=_CROW_DTYPE, device=flag_gems.device)
    with pytest.raises(_SIGNATURE_ERRORS):
        flag_gems._convert_indices_from_csr_to_coo(crow, col, True)


@pytest.mark.convert_indices_from_csr_to_coo
def test__convert_indices_from_csr_to_coo_rejects_invalid_flag_type():
    crow = _csr_offsets((4,), _CROW_DTYPE)
    col = torch.zeros(4, dtype=_CROW_DTYPE, device=flag_gems.device)
    with pytest.raises(_SIGNATURE_ERRORS):
        flag_gems._convert_indices_from_csr_to_coo(crow, col, transpose="yes")


_OUT_MISMATCH_ROWS = [("int32_buffer_default_output", torch.int32, {})]
if utils.int64_is_supported:
    _OUT_MISMATCH_ROWS.append(
        ("int64_buffer_out_int32_output", torch.int64, {"out_int32": True})
    )


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("label,buf_dtype,flags", _OUT_MISMATCH_ROWS)
def test__convert_indices_from_csr_to_coo_rejects_out_dtype_mismatch(
    label, buf_dtype, flags
):
    del label
    crow = _csr_offsets((4,), _CROW_DTYPE)
    col = torch.zeros(4, dtype=_CROW_DTYPE, device=flag_gems.device)
    buf = torch.zeros((2, 4), dtype=buf_dtype, device=flag_gems.device)
    with pytest.raises(_OUT_ERRORS):
        flag_gems._convert_indices_from_csr_to_coo(crow, col, out=buf, **flags)
