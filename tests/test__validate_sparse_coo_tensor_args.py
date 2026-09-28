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

# Correctness tests for aten._validate_sparse_coo_tensor_args. The operator
# validates a COO descriptor, returns None and never inspects the stored values,
# so a positive workload asserts the candidate's None return plus untouched
# operands, and a negative workload asserts the candidate's exception.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Stored coordinates force int64 index storage: the validator accepts no other
# index dtype, so recasting valid coordinates to int32 would test a different
# contract. That is a required *operand* storage capability rather than a values
# payload dtype, so it is carried by its own static eligibility flag, read once
# from the device capability descriptor below and never probed. A family whose
# fixtures allocate an int64 coordinate buffer contributes no cases where that
# storage is unavailable, instead of collecting a fixture the device cannot
# build. The value-dtype gate below covers the payload only and does not
# establish that a backend can store coordinates at all.
_INT64_COORDINATES = bool(utils.int64_is_supported)


def _int64_gated(cases):
    # Collect-time eligibility for the families whose fixtures allocate an int64
    # coordinate buffer. Listing and execution read the same table, so a family
    # is either fully eligible or absent from both.
    return list(cases) if _INT64_COORDINATES else []


_FP8_TYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)

_DTYPE_FLAGS = {
    torch.bfloat16: "bf16_is_supported",
    torch.float64: "fp64_is_supported",
    torch.int64: "int64_is_supported",
}


def _value_dtype_supported(dtype):
    # Static capability flags read once at import; nothing is probed here or
    # while cases are collected.
    if dtype in _FP8_TYPES:
        return bool(utils.fp8_is_supported)
    flag = _DTYPE_FLAGS.get(dtype)
    return flag is None or bool(getattr(utils, flag))


_VALUE_DTYPES = tuple(
    dtype
    for dtype in (*tu.REQUIRED_DTYPES, torch.bool, torch.float64, torch.complex64)
    if _value_dtype_supported(dtype)
)
_FLOAT_VALUE_DTYPES = tuple(dtype for dtype in _VALUE_DTYPES if dtype.is_floating_point)

# Payload dtypes for the flag and metadata observations: an exactly
# representable float plus, when the backend reports int64 values, an integral
# type whose comparisons stay exact.
_FLAG_PAYLOAD_DTYPES = tuple(
    dtype for dtype in (torch.float32, torch.int64) if _value_dtype_supported(dtype)
)

_MAX_NNZ = 4


def _sparse_dim(size, dense_shape):
    return len(size) - len(dense_shape)


def _nnz_for(size, sparse_dim):
    # At most _MAX_NNZ stored columns, and never more than one per coordinate of
    # the sparse extent.
    if sparse_dim == 0:
        return 0
    extent = 1
    for dim in size[:sparse_dim]:
        extent *= int(dim)
    return min(_MAX_NNZ, extent)


# size is the logical tensor shape, so its rank is sparse_dim + dense_dim:
# dense_dim comes from the values payload rank and sparse_dim is the number of
# index rows. The shared shape levels supply the seven main layouts, each paired
# with trailing dense extents so dense_dim spans 0, 1 and 2 while every
# descriptor stays rank-consistent. This dedicated validator never compares
# dense_shape against the trailing size extents, so those rows are accepted
# whether or not the extents agree.
_DENSE_BY_RANK = {0: (), 1: (), 2: (3,), 3: (2,), 4: (), 5: (1, 2)}


def _main_layout(shape):
    dense_shape = _DENSE_BY_RANK[len(shape)]
    return (shape, _nnz_for(shape, _sparse_dim(shape, dense_shape)), dense_shape)


_MAIN_LAYOUTS = [_main_layout(shape) for shape in tu.selected_shapes()]

# Layouts the shared shape levels cannot express. The last two rows are genuine
# hybrids whose dense_shape equals the trailing extents of size; the earlier
# rows, whose dense_shape differs from those extents or that carry no dense axis
# at all, are unchecked-metadata workloads because the operator only derives
# dense_dim from values.dim() - 1. Rows are (size, nnz, dense_shape).
_EXTRA_LAYOUTS = [
    ((4, 4), 0, ()),
    ((4, 4), 1, ()),
    ((6,), 4, (2,)),
    ((2, 3, 4), 3, (5,)),
    ((5, 0), 0, ()),
    ((5, 6, 7, 8), 9, (8,)),
    ((4, 5, 6), 7, (5, 6)),
]

_LAYOUTS = _MAIN_LAYOUTS + _EXTRA_LAYOUTS

# Quick mode keeps the shared quick shape and drops the extra positives; the
# negative families below stay collected in both modes. Every layout row stores
# int64 coordinates, so the whole grid is INT64-eligible as a unit and the
# eligibility applies to the selected table, not to its quick subselection.
_LAYOUT_ROWS = _int64_gated(
    tu.selected_cases(
        [(layout, dtype) for layout in _LAYOUTS for dtype in _VALUE_DTYPES],
        quick=[(layout, dtype) for layout in _MAIN_LAYOUTS for dtype in _VALUE_DTYPES],
    )
)
_LAYOUT_IDS = [
    "size{}_nnz{}_dense{}_{}".format(size, nnz, dense_shape, dtype)
    for (size, nnz, dense_shape), dtype in _LAYOUT_ROWS
]


def _empty_indices(sparse_dim, nnz):
    return torch.empty((sparse_dim, nnz), dtype=torch.int64, device=flag_gems.device)


def _spread_indices(size, nnz):
    # Deterministic in-range coordinates. Each axis cycles on its own, so columns
    # may repeat a coordinate; use _sorted_unique_indices whenever distinct
    # coordinates are part of the contract.
    dims = [int(dim) for dim in size]
    if not dims or nnz == 0:
        return _empty_indices(len(dims), nnz)
    coords = []
    for dim in dims:
        if dim <= 0:
            coords.append(
                torch.zeros((nnz,), dtype=torch.int64, device=flag_gems.device)
            )
            continue
        step = max(dim // (nnz + 1), 1)
        row = torch.arange(nnz, dtype=torch.int64, device=flag_gems.device) * step
        coords.append(row % dim)
    return torch.stack(coords)


def _sorted_unique_indices(size, nnz):
    # Row-major linear positions, hence sorted and distinct: exactly what
    # is_coalesced=True promises. Requires an extent of at least nnz.
    dims = [int(dim) for dim in size]
    if not dims or nnz == 0:
        return _empty_indices(len(dims), nnz)
    capacity = 1
    for dim in dims:
        capacity *= dim
    step = max(capacity // nnz, 1)
    linear = torch.arange(nnz, dtype=torch.int64, device=flag_gems.device) * step
    strides = []
    running = 1
    for dim in reversed(dims):
        strides.append(running)
        running *= dim
    strides.reverse()
    return torch.stack([(linear // strides[i]) % dims[i] for i in range(len(dims))])


def _snapshot(*tensors):
    return [
        (
            tensor.detach().clone(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
        )
        for tensor in tensors
    ]


def _assert_operands_unchanged(snapshot, *tensors):
    # Both the stored contents and the geometry of every operand must survive the
    # call: a validator may neither rewrite coordinates nor normalize layouts.
    for (stored, shape, stride, offset), tensor in zip(snapshot, tensors):
        tu.assert_result_equal(tensor, stored)
        assert (tensor.shape, tensor.stride(), tensor.storage_offset()) == (
            shape,
            stride,
            offset,
        )


def _snapshot_size(size):
    # The schema takes ``size`` as a mutable list; copy its contents before the
    # call so a candidate that rewrites it can be caught.
    return list(size)


def _assert_size_unchanged(snapshot, size):
    assert list(size) == snapshot, f"candidate mutated size: {list(size)} != {snapshot}"


def _coalesced_indices(kind, size, nnz):
    if kind == "sparse_dim0":
        return _empty_indices(0, nnz)
    if kind == "empty_nnz0":
        return _spread_indices(size, 0)
    if kind == "single_nnz1":
        return torch.tensor([[1], [2]], dtype=torch.int64, device=flag_gems.device)
    if kind == "duplicate_sorted":
        # Sorted coordinates with one coordinate stored twice.
        col = torch.tensor(
            [0, 1, 1, 2, 5, 5], dtype=torch.int64, device=flag_gems.device
        )
        return torch.stack([torch.zeros_like(col), col])
    indices = _sorted_unique_indices(size, nnz)
    if kind == "reordered_unique":
        # The coordinate set of sorted_unique, reversed; no RNG is involved, so
        # the row is deterministically unsorted.
        return indices.flip(1)
    return indices


# (size, nnz) for the positive is_coalesced geometries; every coordinate set is
# deterministic.
_COALESCED_GEOMETRY = {
    "sorted_unique": ((8, 8), 6),
    "reordered_unique": ((8, 8), 6),
    "duplicate_sorted": ((8, 8), 6),
    "sparse_dim0": ((), 0),
    "empty_nnz0": ((8, 8), 0),
    "single_nnz1": ((8, 8), 1),
}

# is_coalesced=True may only be asserted for sorted, distinct coordinates, so
# the reordered and duplicate geometries omit it; their rejection is covered by
# the unsorted / duplicate rows below. Each geometry stores int64 coordinates,
# so the whole table is INT64-eligible as a unit.
_COALESCED_FLAGS = {
    "sorted_unique": (True, False, None, "omitted"),
    "reordered_unique": (False, None, "omitted"),
    "duplicate_sorted": (False, None, "omitted"),
    "sparse_dim0": (True, False, None, "omitted"),
    "empty_nnz0": (True, False, None, "omitted"),
    "single_nnz1": (True, False, None, "omitted"),
}
_COALESCED_CASES = _int64_gated(
    tu.selected_cases(
        [(kind, flag) for kind, flags in _COALESCED_FLAGS.items() for flag in flags],
        quick=[],
    )
)
_COALESCED_CASE_IDS = ["{}-{}".format(kind, flag) for kind, flag in _COALESCED_CASES]

# size kinds accepted for the size argument; the flag is passed positionally or
# by keyword, and omission is a distinct call from explicit None. The indices of
# every row are sorted and distinct int64 coordinates.
_FORM_CASES = _int64_gated(
    tu.selected_cases(
        [
            ("list", False, "positional"),
            ("list", True, "positional"),
            ("list", None, "positional"),
            ("tuple", False, "keyword"),
            ("tuple", True, "keyword"),
            ("torch_size", False, "keyword"),
            ("torch_size", None, "positional"),
        ],
        quick=[],
    )
)

# Rows are (dtype, scenario) with scenarios nan / inf / mixed. The shared matrix
# already omits the inf-bearing scenarios a dtype cannot represent (for example
# float8_e4m3fn). The coordinates are again int64.
_SPECIAL_CASES = _int64_gated(
    tu.selected_cases(tu.special_value_cases(_FLOAT_VALUE_DTYPES), quick=[])
)
_SPECIAL_CASE_IDS = [
    "{}_{}".format(str(dtype).replace("torch.", ""), scenario)
    for dtype, scenario in _SPECIAL_CASES
]

# The validator checks index dtype, index rank, size rank, index range and the
# coalesced promise. It derives dense_dim from values.dim() - 1 and never
# compares the values payload against nnz or against the dense size entries, so
# these descriptors are accepted. Rows are (name, nnz, size, values_shape) with
# sparse_dim derived as len(size) - (len(values_shape) - 1).
_UNCHECKED_METADATA_CASES = [
    ("values_fewer_than_nnz", 3, (3, 3), (2,)),
    ("values_more_than_nnz", 3, (3, 3), (5,)),
    ("values_empty", 3, (3, 3), (0,)),
    ("values_dense_extent_shorter", 3, (4, 4, 4, 5), (3, 3)),
    ("values_dense_extent_longer", 3, (4, 4, 4, 5), (3, 7)),
]
# The parameter table and its explicit ids come from one final selected record
# list, so the two can never disagree when the INT64 gate or quick mode empties
# the family: pytest rejects a table whose length differs from its id list. On a
# supported device in default mode the selection is all five records with their
# original ids.
_UNCHECKED_METADATA_SELECTED = _int64_gated(
    tu.selected_cases(list(_UNCHECKED_METADATA_CASES), quick=[])
)
_UNCHECKED_METADATA_IDS = [row[0] for row in _UNCHECKED_METADATA_SELECTED]
_UNCHECKED_METADATA_ROWS = [row[1:] for row in _UNCHECKED_METADATA_SELECTED]

# Measured native taxonomy on this backend: a malformed operand and a rejected
# descriptor both raise RuntimeError, while a Python-level candidate raises
# TypeError for a wrong operand type or a missing argument. No probed failure
# raised ValueError, so it is deliberately not accepted.
_SCHEMA_ERRORS = (RuntimeError, TypeError)
_VALUE_ERRORS = (RuntimeError,)
_MISSING_ERRORS = (TypeError, RuntimeError)

# (name, indices_shape, indices_dtype, indices_flat, values_shape, size,
# is_coalesced, kind, exc). kind is "tensor" except for the rows that pass a
# non-Tensor operand; indices_flat is the row-major coordinate list, or None for
# an all-zero coordinate matrix.
_INVALID_CASES = [
    # Index dtype: int64 is the only accepted coordinate type. These three rows
    # allocate a supported non-int64 buffer, so they stay collectable on every
    # device.
    (
        "indices_dtype_float32",
        (2, 2),
        torch.float32,
        [0, 1, 1, 0],
        (2,),
        [2, 2],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "indices_dtype_int32",
        (2, 2),
        torch.int32,
        [0, 1, 1, 0],
        (2,),
        [2, 2],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "indices_dtype_int16",
        (2, 2),
        torch.int16,
        [0, 1, 1, 0],
        (2,),
        [2, 2],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    # Index rank and sparse_dim, size rank and dense rank. From here on every
    # row allocates its coordinate buffer in int64.
    (
        "indices_rank_one",
        (3,),
        torch.int64,
        [0, 1, 2],
        (3,),
        [3, 3],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "indices_rank_zero",
        (),
        torch.int64,
        [0],
        (3,),
        [3, 3],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "indices_rank_three",
        (2, 3, 1),
        torch.int64,
        None,
        (3,),
        [3, 3],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "indices_rows_exceed_sparse_dim",
        (3, 3),
        torch.int64,
        None,
        (3,),
        [3, 3],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "size_rank_below_sparse_dim",
        (2, 3),
        torch.int64,
        None,
        (3,),
        [3],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "size_rank_above_sparse_dim",
        (2, 3),
        torch.int64,
        None,
        (3,),
        [3, 3, 3],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "values_rank_mismatch",
        (2, 3),
        torch.int64,
        None,
        (3, 1),
        [3, 3],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    # Non-schema arguments.
    (
        "size_not_a_list",
        (2, 2),
        torch.int64,
        [0, 1, 1, 0],
        (2,),
        2,
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "size_with_float_entries",
        (2, 2),
        torch.int64,
        [0, 1, 1, 0],
        (2,),
        [2.0, 2.0],
        None,
        "tensor",
        _SCHEMA_ERRORS,
    ),
    (
        "indices_not_a_tensor",
        (2, 2),
        torch.int64,
        [0, 1, 1, 0],
        (2,),
        [2, 2],
        None,
        "list_indices",
        _SCHEMA_ERRORS,
    ),
    (
        "values_not_a_tensor",
        (2, 2),
        torch.int64,
        [0, 1, 1, 0],
        (2,),
        [2, 2],
        None,
        "list_values",
        _SCHEMA_ERRORS,
    ),
    (
        "is_coalesced_not_a_bool",
        (2, 2),
        torch.int64,
        [0, 1, 1, 0],
        (2,),
        [2, 2],
        "yes",
        "tensor",
        _SCHEMA_ERRORS,
    ),
    # Index range: at the extent, beyond it, negative, and a single offending
    # coordinate in the first, middle and last stored column.
    (
        "index_equals_size",
        (2, 3),
        torch.int64,
        [0, 1, 3, 1, 0, 2],
        (3,),
        [3, 3],
        None,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "index_beyond_size",
        (2, 3),
        torch.int64,
        [0, 1, 5, 1, 0, 2],
        (3,),
        [3, 3],
        None,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "index_negative",
        (2, 3),
        torch.int64,
        [0, 1, -1, 1, 0, 2],
        (3,),
        [3, 3],
        None,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "index_beyond_size_first_column",
        (2, 3),
        torch.int64,
        [5, 1, 0, 0, 1, 2],
        (3,),
        [3, 3],
        None,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "index_beyond_size_middle_column",
        (2, 3),
        torch.int64,
        [0, 5, 1, 0, 1, 2],
        (3,),
        [3, 3],
        None,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "index_beyond_size_last_column",
        (2, 3),
        torch.int64,
        [0, 1, 5, 0, 1, 2],
        (3,),
        [3, 3],
        None,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "size_negative_dim",
        (2, 2),
        torch.int64,
        [0, 1, 2, 0],
        (2,),
        [3, -1],
        None,
        "tensor",
        _VALUE_ERRORS,
    ),
    # is_coalesced=True promises sorted, distinct coordinates.
    (
        "unsorted_with_coalesced_true",
        (2, 3),
        torch.int64,
        [0, 1, 0, 1, 0, 2],
        (3,),
        [2, 3],
        True,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "duplicate_with_coalesced_true",
        (2, 3),
        torch.int64,
        [0, 0, 0, 1, 1, 1],
        (3,),
        [1, 2],
        True,
        "tensor",
        _VALUE_ERRORS,
    ),
    (
        "duplicate_sorted_with_coalesced_true",
        (2, 6),
        torch.int64,
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 5, 5],
        (6,),
        [8, 8],
        True,
        "tensor",
        _VALUE_ERRORS,
    ),
]


def _collects_without_int64(row):
    # _invalid_operands allocates the coordinate buffer with the row's own
    # indices_dtype, so a row is collectable without int64 storage exactly when
    # that dtype is not int64. The rows that pass a supported wrong index dtype
    # therefore stay runnable on every device, while the int64 rows are the same
    # INT64 storage prerequisite as the positive families and are collected only
    # where the static flag permits it.
    return _INT64_COORDINATES or row[2] is not torch.int64


_INVALID_CASES_COLLECTED = [
    row for row in _INVALID_CASES if _collects_without_int64(row)
]
_INVALID_CASE_IDS = [row[0] for row in _INVALID_CASES_COLLECTED]
_INVALID_CASE_ROWS = [row[1:] for row in _INVALID_CASES_COLLECTED]

_NNZ = 8
_STRIDED_SIZE = [8, 8]

# (indices_kind, values_kind): genuine nonzero storage offsets and strided views
# of both operands, each paired with contiguous controls; the index operand is
# always an int64 buffer.
_STRIDED_CASES = [
    ("contiguous", "contiguous"),
    ("offset", "contiguous"),
    ("strided", "contiguous"),
    ("offset_strided", "contiguous"),
    ("contiguous", "offset"),
    ("contiguous", "strided"),
    ("contiguous", "offset_strided"),
]
_STRIDED_ROWS = _int64_gated(tu.selected_cases(list(_STRIDED_CASES), quick=[]))
_STRIDED_CASE_IDS = [
    "indices_{}_values_{}".format(indices_kind, values_kind)
    for indices_kind, values_kind in _STRIDED_ROWS
]

# The offending coordinate is written into an int64 coordinate buffer, so this
# family carries the same INT64 storage prerequisite as the strided grid above.
_STRIDED_OUT_OF_RANGE_AXES = [0, 1] if _INT64_COORDINATES else []

# Int64 index-range boundary. The range check is per coordinate and size is never
# allocated, so a coordinate above INT32 with an extent above INT32 is valid and
# exactly one past it is not. Every coordinate here also needs an int64 buffer,
# so both families carry the INT64 storage prerequisite of the positive grid.
_LARGE_EXTENT = 2**31 + 2
_ACCEPTED_LARGE_ROWS = _int64_gated(
    tu.selected_cases([(0, _LARGE_EXTENT - 1), (1, _LARGE_EXTENT - 1)], quick=[])
)
_ACCEPTED_LARGE_IDS = [
    "last_valid_on_axis{}".format(axis) for axis, _ in _ACCEPTED_LARGE_ROWS
]
_REJECTED_LARGE = [
    ("equal_extent_axis0", 0, _LARGE_EXTENT),
    ("equal_extent_axis1", 1, _LARGE_EXTENT),
    ("int64_min_axis0", 0, -(2**63)),
    ("int64_max_axis0", 0, 2**63 - 1),
]
_REJECTED_LARGE_ROWS = _int64_gated(_REJECTED_LARGE)
_REJECTED_LARGE_IDS = [row[0] for row in _REJECTED_LARGE_ROWS]
_REJECTED_LARGE_CASES = [row[1:] for row in _REJECTED_LARGE_ROWS]


def _strided_indices(kind):
    # Every element of the pattern is a valid coordinate for _STRIDED_SIZE, so
    # any view of this buffer is a layout variant of a valid coordinate matrix.
    pattern = (
        torch.arange(2 * (2 * _NNZ + 4), dtype=torch.int64, device=flag_gems.device) * 5
    ) % 8
    matrix = pattern.view(2, -1)
    if kind == "contiguous":
        return pattern.view(2, -1)[:, :_NNZ].contiguous()
    if kind == "offset":
        # Storage offset 7, contiguous.
        return pattern[7 : 7 + 2 * _NNZ].view(2, _NNZ)
    if kind == "strided":
        return matrix[:, : 2 * _NNZ : 2]
    # Storage offset 3 in the second dimension plus a stride of 2.
    return matrix[:, 3 : 3 + 2 * _NNZ : 2]


def _strided_values(kind, dtype, length):
    base = tu.make_input(dtype, (2 * 3 * _NNZ + 8,), ["-1", "1"])
    if kind == "contiguous":
        return base[:length].contiguous()
    if kind == "offset":
        return base[5 : 5 + length]
    return base[6::3][:length]


def _invalid_operands(row):
    (
        indices_shape,
        indices_dtype,
        indices_flat,
        values_shape,
        size,
        is_coalesced,
        kind,
        _exc,
    ) = row
    if indices_flat is None:
        indices = torch.zeros(
            indices_shape, dtype=indices_dtype, device=flag_gems.device
        )
    else:
        indices = torch.tensor(
            indices_flat, dtype=indices_dtype, device=flag_gems.device
        ).reshape(indices_shape)
    values = torch.zeros(values_shape, dtype=torch.float32, device=flag_gems.device)
    if kind == "list_indices":
        indices = indices.tolist()
    elif kind == "list_values":
        values = values.tolist()
    return indices, values, size, is_coalesced


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("layout,dtype", _LAYOUT_ROWS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__validate_sparse_coo_tensor_args_valid(layout, dtype, value_range):
    size, nnz, dense_shape = layout
    sparse_dim = _sparse_dim(size, dense_shape)
    indices = _spread_indices(size[:sparse_dim], nnz)
    values = tu.make_input(dtype, (nnz, *dense_shape), value_range)
    size_arg = list(size)
    snapshot = _snapshot(indices, values)
    size_snapshot = _snapshot_size(size_arg)

    # The reference documents that this descriptor is accepted; the candidate's
    # own contract is the None return asserted below.
    torch.ops.aten._validate_sparse_coo_tensor_args(
        tu.to_reference(indices), tu.to_reference(values), list(size)
    )
    res_out = flag_gems._validate_sparse_coo_tensor_args(indices, values, size_arg)

    assert res_out is None
    _assert_operands_unchanged(snapshot, indices, values)
    _assert_size_unchanged(size_snapshot, size_arg)


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("kind,flag", _COALESCED_CASES, ids=_COALESCED_CASE_IDS)
@pytest.mark.parametrize("dtype", _FLAG_PAYLOAD_DTYPES)
def test__validate_sparse_coo_tensor_args_coalesced_flag(kind, flag, dtype):
    size, nnz = _COALESCED_GEOMETRY[kind]
    size = list(size)
    indices = _coalesced_indices(kind, size, nnz)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    snapshot = _snapshot(indices, values)
    size_snapshot = _snapshot_size(size)

    if flag == "omitted":
        # Dropping is_coalesced must behave like the explicit None default, so
        # only an actually omitted argument checks that the public default
        # exists. Explicit None is covered by its own row.
        torch.ops.aten._validate_sparse_coo_tensor_args(
            tu.to_reference(indices), tu.to_reference(values), list(size)
        )
        res_out = flag_gems._validate_sparse_coo_tensor_args(indices, values, size)
    else:
        torch.ops.aten._validate_sparse_coo_tensor_args(
            tu.to_reference(indices), tu.to_reference(values), list(size), flag
        )
        res_out = flag_gems._validate_sparse_coo_tensor_args(
            indices, values, size, flag
        )

    assert res_out is None
    _assert_operands_unchanged(snapshot, indices, values)
    _assert_size_unchanged(size_snapshot, size)


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("size_kind,is_coalesced,call_style", _FORM_CASES)
def test__validate_sparse_coo_tensor_args_call_forms(
    size_kind, is_coalesced, call_style
):
    size = [4, 4]
    indices = _sorted_unique_indices(size, 4)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    size_arg = {
        "list": list(size),
        "tuple": tuple(size),
        "torch_size": torch.Size(size),
    }[size_kind]
    snapshot = _snapshot(indices, values)
    size_snapshot = _snapshot_size(size_arg)

    if call_style == "keyword":
        torch.ops.aten._validate_sparse_coo_tensor_args(
            tu.to_reference(indices),
            tu.to_reference(values),
            size=size_arg,
            is_coalesced=is_coalesced,
        )
        res_out = flag_gems._validate_sparse_coo_tensor_args(
            indices, values, size=size_arg, is_coalesced=is_coalesced
        )
    else:
        torch.ops.aten._validate_sparse_coo_tensor_args(
            tu.to_reference(indices), tu.to_reference(values), size_arg, is_coalesced
        )
        res_out = flag_gems._validate_sparse_coo_tensor_args(
            indices, values, size_arg, is_coalesced
        )

    assert res_out is None
    _assert_operands_unchanged(snapshot, indices, values)
    _assert_size_unchanged(size_snapshot, size_arg)


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES, ids=_SPECIAL_CASE_IDS)
def test__validate_sparse_coo_tensor_args_special_values(dtype, scenario):
    size = [4, 4]
    # The shared maker returns exactly one payload for this dtype / scenario
    # pair; its length fixes nnz so the coordinates and the stored payload agree
    # and the NaN / Inf input really reaches the validator, which never inspects
    # it.
    values = tu.make_special_input(dtype, scenario)
    indices = _spread_indices(size, values.numel())
    snapshot = _snapshot(indices, values)
    size_snapshot = _snapshot_size(size)

    torch.ops.aten._validate_sparse_coo_tensor_args(
        tu.to_reference(indices), tu.to_reference(values), list(size)
    )
    res_out = flag_gems._validate_sparse_coo_tensor_args(indices, values, size)

    assert res_out is None
    _assert_operands_unchanged(snapshot, indices, values)
    _assert_size_unchanged(size_snapshot, size)


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("row", _UNCHECKED_METADATA_ROWS, ids=_UNCHECKED_METADATA_IDS)
@pytest.mark.parametrize("dtype", _FLAG_PAYLOAD_DTYPES)
def test__validate_sparse_coo_tensor_args_accepts_unchecked_metadata(row, dtype):
    nnz, size, values_shape = row
    sparse_dim = len(size) - (len(values_shape) - 1)
    size = list(size)
    indices = _spread_indices(size[:sparse_dim], nnz)
    values = tu.make_input(dtype, values_shape, ["-1", "1"])
    snapshot = _snapshot(indices, values)
    size_snapshot = _snapshot_size(size)

    torch.ops.aten._validate_sparse_coo_tensor_args(
        tu.to_reference(indices), tu.to_reference(values), list(size)
    )
    res_out = flag_gems._validate_sparse_coo_tensor_args(indices, values, size)

    assert res_out is None
    _assert_operands_unchanged(snapshot, indices, values)
    _assert_size_unchanged(size_snapshot, size)


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize(
    "indices_kind,values_kind", _STRIDED_ROWS, ids=_STRIDED_CASE_IDS
)
def test__validate_sparse_coo_tensor_args_strided_inputs(indices_kind, values_kind):
    size = list(_STRIDED_SIZE)
    indices = _strided_indices(indices_kind)
    values = _strided_values(values_kind, torch.float32, _NNZ)
    snapshot = _snapshot(indices, values)
    size_snapshot = _snapshot_size(size)

    torch.ops.aten._validate_sparse_coo_tensor_args(
        tu.to_reference(indices), tu.to_reference(values), list(_STRIDED_SIZE)
    )
    res_out = flag_gems._validate_sparse_coo_tensor_args(indices, values, size)

    assert res_out is None
    # A read-only validator must not rewrite coordinate contents or normalize
    # either operand's layout.
    _assert_operands_unchanged(snapshot, indices, values)
    _assert_size_unchanged(size_snapshot, size)


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("axis", _STRIDED_OUT_OF_RANGE_AXES)
def test__validate_sparse_coo_tensor_args_rejects_strided_out_of_range(axis):
    # The offending coordinate sits in the last stored column of a non-adjacent
    # index matrix, so a validator that reads only the first column would miss
    # it. The coordinate buffer is int64, hence the eligibility gate above.
    base = torch.zeros((2, 2 * _NNZ), dtype=torch.int64, device=flag_gems.device)
    base[axis, 2 * _NNZ - 2] = _STRIDED_SIZE[axis]
    indices = base[:, ::2]
    values = tu.make_input(torch.float32, (indices.shape[1],), ["-1", "1"])

    with pytest.raises(_VALUE_ERRORS):
        flag_gems._validate_sparse_coo_tensor_args(indices, values, list(_STRIDED_SIZE))


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize(
    "axis,coordinate", _ACCEPTED_LARGE_ROWS, ids=_ACCEPTED_LARGE_IDS
)
def test__validate_sparse_coo_tensor_args_large_index_boundary(axis, coordinate):
    size = [_LARGE_EXTENT, _LARGE_EXTENT]
    indices = torch.zeros((2, 1), dtype=torch.int64, device=flag_gems.device)
    indices[axis, 0] = coordinate
    values = tu.make_input(torch.float32, (1,), ["-1", "1"])
    snapshot = _snapshot(indices, values)
    size_snapshot = _snapshot_size(size)

    torch.ops.aten._validate_sparse_coo_tensor_args(
        tu.to_reference(indices), tu.to_reference(values), list(size)
    )
    res_out = flag_gems._validate_sparse_coo_tensor_args(indices, values, size)

    assert res_out is None
    _assert_operands_unchanged(snapshot, indices, values)
    _assert_size_unchanged(size_snapshot, size)


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize(
    "axis,coordinate", _REJECTED_LARGE_CASES, ids=_REJECTED_LARGE_IDS
)
def test__validate_sparse_coo_tensor_args_rejects_large_index(axis, coordinate):
    size = [_LARGE_EXTENT, _LARGE_EXTENT]
    indices = torch.zeros((2, 1), dtype=torch.int64, device=flag_gems.device)
    indices[axis, 0] = coordinate
    values = tu.make_input(torch.float32, (1,), ["-1", "1"])

    with pytest.raises(_VALUE_ERRORS):
        flag_gems._validate_sparse_coo_tensor_args(indices, values, list(size))


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("row", _INVALID_CASE_ROWS, ids=_INVALID_CASE_IDS)
def test__validate_sparse_coo_tensor_args_rejects(row):
    exc = row[-1]
    indices, values, size, is_coalesced = _invalid_operands(row)

    with pytest.raises(exc):
        flag_gems._validate_sparse_coo_tensor_args(indices, values, size, is_coalesced)


# _sorted_unique_indices builds the indices argument in int64, so every
# missing-argument row needs int64 coordinate storage; without that capability
# the family contributes no rows instead of a fixture it cannot build. The size
# argument is a plain list here, so no operand allocation is implied by it.
_MISSING_ARGUMENTS = ["indices", "values", "size"] if _INT64_COORDINATES else []


@pytest.mark.validate_sparse_coo_tensor_args
@pytest.mark.parametrize("missing", _MISSING_ARGUMENTS)
def test__validate_sparse_coo_tensor_args_rejects_missing_arguments(missing):
    # Each row omits exactly one required argument while every remaining
    # argument keeps its correct name and value, so a missing-argument failure
    # cannot be confused with a shifted positional argument.
    size = [4, 4]
    kwargs = {
        "indices": _sorted_unique_indices(size, 4),
        "values": tu.make_input(torch.float32, (4,), ["-1", "1"]),
        "size": list(size),
    }
    del kwargs[missing]

    with pytest.raises(_MISSING_ERRORS):
        flag_gems._validate_sparse_coo_tensor_args(**kwargs)
