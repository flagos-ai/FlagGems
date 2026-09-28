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

from . import test_utils as tu

# The operator validates the arguments of a CSC tensor and returns nothing:
#   _validate_sparse_csc_tensor_args(Tensor ccol_indices, Tensor row_indices,
#                                    Tensor values, int[] size) -> ()
# A valid workload is observable through the call being accepted, the return
# value being None and the inspected tensors staying unmodified. The host worker
# checks the argument preconditions (ranks, strides, dtypes, sizes, devices); the
# index *content* invariants are checked by a kernel on the device as well.
# Three spec dimensions drop out for that reason:
#   * broadcast: nothing is combined elementwise, so no operand pair broadcasts;
#   * backward: the schema returns no tensor and is not differentiable;
#   * tensor vs scalar operand: the only non-tensor argument is the required
#     int[] size, which has no scalar form.
# The schema declares a single default overload (probed on the active backend:
# overloads() == ["default"]), so there is no .out call form to exercise.

_RUNTIME_DEVICE = flag_gems.runtime.device

# Static backend capability flags decide which tensors this file builds; dtype
# support is never probed during collection or execution.
_DTYPE_GATES = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e4m3fnuz: "support_fp8",
    torch.float8_e5m2: "support_fp8",
    torch.float8_e5m2fnuz: "support_fp8",
    torch.int64: "support_int64",
}


def _dtype_supported(dtype):
    # A flag that is absent is treated as available, so an unsupported dtype
    # fails at build time instead of dropping coverage silently.
    flag = _DTYPE_GATES.get(dtype)
    return flag is None or bool(getattr(_RUNTIME_DEVICE, flag, True))


# Index tensors must use a device-supported integer dtype. Where the backend has
# no int64 the same structures are built with int32 indices instead; the int64
# metadata used to compute the structure is a host temporary, so it does not
# depend on the device index dtype and is not gated.
HAS_INT64_INDEX = _dtype_supported(torch.int64)
INDEX_DTYPE_LONG = torch.int64 if HAS_INT64_INDEX else torch.int32

# The validator never dispatches on the values dtype: every spec dtype plus
# float64, bool and complex64 were accepted by the native operator on the active
# backend, so each gate-passing dtype is exercised.
VALUE_DTYPES = [
    dtype
    for dtype in list(tu.REQUIRED_DTYPES) + [torch.float64, torch.bool, torch.complex64]
    if _dtype_supported(dtype)
]

# A CSC structure is described by a size with len(size) == batch + 2 + dense, so
# the 0-dim and 1-dim entries of tu.selected_shapes() cannot describe one and are
# filtered out. Rank >= 2 is the operator's own rule.
SIZE_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]

# Rows are (size, nnz, dense_ndim, index_dtype, layout). The plain shape grid
# cannot express empty structures, full density, int32 indices, strided/offset
# operands or an index rank beyond 8, so the geometry, the index dtype and the
# storage layout of the operands are covered here. Index ranks 8 and 9 appear as
# generic rank-boundary cases; how the implementation dispatches such ranks was
# not established here.
STRUCTURE_ROWS = [
    ((1, 1), 1, 0, torch.int64, "plain"),  # smallest non-empty structure
    ((1, 1), 0, 0, torch.int64, "plain"),  # no stored entries
    ((4, 4), 16, 0, torch.int64, "plain"),  # fully dense
    ((5, 4), 4, 0, torch.int64, "plain"),  # one entry per column
    ((4, 0), 0, 0, torch.int64, "plain"),  # no columns
    ((0, 4), 0, 0, torch.int64, "plain"),  # no rows
    ((3, 4, 5), 15, 0, torch.int64, "plain"),  # batched, fully dense
    ((3, 0, 4), 0, 0, torch.int64, "plain"),  # empty plain dimension
    ((0, 3, 4), 0, 0, torch.int64, "plain"),  # empty batch dimension
    ((2, 3, 4, 5), 20, 0, torch.int64, "plain"),  # two batch dimensions
    ((1,) * 7 + (2, 3), 3, 0, torch.int64, "plain"),  # index rank 8
    ((1,) * 8 + (2, 3), 3, 0, torch.int64, "plain"),  # index rank 9
    ((2, 3, 4, 5, 7), 20, 1, torch.int64, "plain"),  # one dense dimension
    ((2, 3, 5, 0), 0, 1, torch.int64, "plain"),  # empty dense dimension
    ((4, 4), 4, 0, torch.int64, "offset_indices"),  # index storage offset
    ((4, 4), 4, 0, torch.int64, "strided_values"),  # strided values view
    ((4, 4), 16, 0, torch.int32, "plain"),  # int32 index dtype
    ((3, 4, 5), 15, 0, torch.int32, "plain"),
    ((8, 5), 5, 0, torch.int32, "plain"),
    ((2, 3, 4, 5, 7), 20, 1, torch.int32, "plain"),
    ((16, 128, 64, 60), 60, 0, torch.int64, "plain"),  # wide batch
    ((2, 19, 7), 7, 0, torch.int64, "plain"),  # regular 3-dim structure
]

# These structural positives run in the default suite only; the main grid below
# keeps the quick smoke subset. Rows whose index dtype the backend cannot build
# are dropped, which is the int32-only fallback when there is no int64 device
# dtype; the int32 rows stay either way.
STRUCTURE_CASES = [
    row
    for row in tu.selected_cases(STRUCTURE_ROWS, quick=[])
    if _dtype_supported(row[3])
]

# One broken host-side precondition per row, isolated so each failure can be
# attributed to a single argument check of
# _validate_sparse_compressed_tensor_args_worker. The index *content* invariants
# of the same native worker (sorted and distinct column entries, ccol[-1] == nnz,
# 0 <= ccol[0], per-batch contiguity of the stored column slices) are checked
# under CUDA_KERNEL_ASSERT, which aborts the CUDA context instead of raising a
# Python exception; those malformed index contents are therefore not exercised
# here and remain an unresolved part of the native contract. Only host-side
# preconditions are negative-tested. Rows are (defect, size, nnz).
INVALID_ROWS = [
    ("size_rank", (3, 4, 5), 15),
    ("ccol_rank_zero", (3, 4, 5), 15),
    ("ccol_row_rank", (3, 4, 5), 15),
    ("values_rank", (3, 4, 5), 15),
    ("row_batch_mismatch", (3, 4, 5), 15),
    ("values_batch_mismatch", (3, 4, 5), 15),
    ("ccol_length", (3, 4, 5), 15),
    ("row_length", (3, 4, 5), 15),
    ("ccol_stride", (3, 4, 5), 15),
    ("row_stride", (3, 4, 5), 15),
    *([("index_dtype_mismatch", (3, 4, 5), 15)] if HAS_INT64_INDEX else []),
    ("index_dtype_unsupported", (3, 4, 5), 15),
    ("meta_nnz", (4, 5), 3),
]

# Invalid call schemas, separate from the metadata preconditions above.
SCHEMA_DEFECTS = [
    "size_missing",
    "ccol_not_tensor",
    "size_not_int_list",
    "size_float_entry",
]

# tu.make_special_input yields five values, which fit an (8, 5) CSC structure with
# one entry per column. The shared helper already restricts the scenarios to what
# each dtype can represent (nan only for float8_e4m3fn; nan, inf and mixed for
# float8_e5m2 and the wider float types). Positive special values run in the
# default suite only.
SPECIAL_SIZE = (8, 5)

SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(VALUE_DTYPES), quick=[])

# A row coordinate above 2**31 with a huge logical row count and tiny nnz: int64
# indices describe it and no dense tensor is allocated for the structure, while
# int32 indices cannot describe the row dimension at all (see the overflow test).
LARGE_ROW_SIZE = [2**31 + 5, 1]
LARGE_ROW_COORDINATE = 2**31 + 3


def _split_size(size, dense_ndim):
    batch = tuple(size[: len(size) - 2 - dense_ndim])
    nrows = size[len(batch)]
    ncols = size[len(batch) + 1]
    dense = tuple(size[len(batch) + 2 :])
    return batch, nrows, ncols, dense


def _column_counts(batch_count, ncols, nnz):
    # A valid column-count vector (column j holds rows 0..count_j-1, so every
    # count is at most ceil(nnz / ncols) <= nrows whenever nnz <= nrows * ncols),
    # rotated by one column per batch. Rotation keeps each batch's total and
    # per-column bound while giving the batches different column distributions
    # whenever the entries do not divide evenly, so the fixture does not repeat
    # one identical index structure for every batch. That diversity alone is not
    # evidence that a validator inspected every batch.
    counts = torch.zeros(batch_count, ncols, dtype=torch.int64)
    if ncols == 0:
        return counts
    counts += nnz // ncols
    counts[:, : nnz % ncols] += 1
    if batch_count > 1 and ncols > 1:
        columns = torch.arange(ncols, dtype=torch.int64)[None, :]
        shift = (torch.arange(batch_count, dtype=torch.int64) % ncols)[:, None]
        counts = torch.gather(counts, 1, (columns - shift) % ncols)
    return counts


def _csc_indices(size, nnz, index_dtype, dense_ndim=0):
    # Valid ccol_indices/row_indices for size: ccol[..., 0] == 0,
    # ccol[..., -1] == nnz, 0 <= diff <= nrows, and each column slice is sorted
    # and distinct whenever nnz <= nrows * ncols. The int64 metadata is built on
    # the host and moved once, independently of the device index dtype.
    batch, nrows, ncols, _ = _split_size(size, dense_ndim)
    batch_count = 1
    for dim in batch:
        batch_count *= dim
    counts = _column_counts(batch_count, ncols, nnz)
    ccol = torch.zeros(batch_count, ncols + 1, dtype=torch.int64)
    if ncols:
        ccol[:, 1:] = torch.cumsum(counts, 1)
    row = torch.zeros(batch_count, nnz, dtype=torch.int64)
    if nnz:
        # Entry i belongs to the last column whose start is <= i; its row inside
        # that column is then i - ccol[column].
        offsets = (
            torch.arange(nnz, dtype=torch.int64).expand(batch_count, nnz).contiguous()
        )
        starts = ccol[:, :-1].contiguous()
        column = torch.searchsorted(starts, offsets, right=True) - 1
        row = offsets - torch.gather(starts, 1, column)
    return (
        ccol.reshape(batch + (ncols + 1,)).to(index_dtype).to(flag_gems.device),
        row.reshape(batch + (nnz,)).to(index_dtype).to(flag_gems.device),
    )


def _offset_last_dim(tensor, pad):
    # Same shape and values, reached through a wider storage: storage_offset is
    # pad and stride(-1) stays 1, which the validator accepts. The parent is
    # returned so the test can check the whole storage it allocated.
    parent = torch.zeros(
        tensor.shape[:-1] + (tensor.shape[-1] + pad,),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = parent[..., pad:]
    view.copy_(tensor)
    return view, parent


def _strided_last_dim(tensor, stride):
    # Same shape and values, but stride(-1) == stride. The validator requires the
    # last dimension of both index tensors to be dense per batch; the defective
    # operand still holds every original value, so the negative case fails for the
    # stride and nothing else.
    parent = torch.zeros(
        tensor.shape[:-1] + (tensor.shape[-1] * stride,),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = parent[..., ::stride]
    view.copy_(tensor)
    return view, parent


def _make_csc_inputs(
    size, nnz, index_dtype, value_dtype, value_range, dense_ndim=0, layout="plain"
):
    ccol, row = _csc_indices(size, nnz, index_dtype, dense_ndim)
    batch, _, _, dense = _split_size(size, dense_ndim)
    values = tu.make_input(value_dtype, batch + (nnz,) + dense, value_range)
    parents = ()
    if layout == "offset_indices":
        ccol, ccol_parent = _offset_last_dim(ccol, 3)
        row, row_parent = _offset_last_dim(row, 3)
        parents = (ccol_parent, row_parent)
    elif layout == "strided_values":
        # Only the index tensors must be dense in their last dimension; values
        # may be a strided view.
        values, values_parent = _strided_last_dim(values, 2)
        parents = (values_parent,)
    elif layout != "plain":
        raise ValueError("unknown layout " + layout)
    return ccol, row, values, list(size), parents


def _large_row_inputs(index_dtype, row_coordinate):
    # ccol = [0, 1] with one stored entry, so the structure is valid without any
    # dense allocation for the 2**31+5 logical rows. Both index tensors are
    # allocated directly in the requested dtype.
    ccol = torch.tensor([0, 1], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([row_coordinate], dtype=index_dtype, device=flag_gems.device)
    values = torch.ones(1, dtype=torch.float32, device=flag_gems.device)
    return ccol, row, values, list(LARGE_ROW_SIZE)


def _snapshot(*tensors):
    # Independent copies of the operands (and of the storage a view was taken
    # from) together with the view geometry the validator must not rewrite.
    return [
        (
            tensor,
            tu.to_reference(tensor),
            (tensor.shape, tensor.stride(), tensor.storage_offset()),
        )
        for tensor in tensors
    ]


def _assert_untouched(snapshot):
    for tensor, reference, geometry in snapshot:
        tu.assert_result_equal(tensor, reference)
        assert (tensor.shape, tensor.stride(), tensor.storage_offset()) == geometry


def _invalid_csc_inputs(defect, size, nnz):
    # Build a structurally valid CSC call and then break exactly one host-side
    # precondition, named by defect.
    ccol, row, values, call_size, _ = _make_csc_inputs(
        size, nnz, INDEX_DTYPE_LONG, torch.float32, ["-1", "1"]
    )
    if defect == "size_rank":
        return ccol, row, values, call_size + [2]
    if defect == "ccol_rank_zero":
        zero_dim = torch.tensor(0, dtype=INDEX_DTYPE_LONG, device=flag_gems.device)
        return zero_dim, row, values, call_size
    if defect == "ccol_row_rank":
        return ccol, row.flatten(), values, call_size
    if defect == "values_rank":
        return ccol, row, values.flatten(), call_size
    if defect == "row_batch_mismatch":
        return ccol, row[:-1], values, call_size
    if defect == "values_batch_mismatch":
        return ccol, row, values[:-1], call_size
    if defect == "ccol_length":
        return ccol[..., :-2], row, values, call_size
    if defect == "row_length":
        return ccol, row[..., :-1], values, call_size
    if defect == "ccol_stride":
        return _strided_last_dim(ccol, 2)[0], row, values, call_size
    if defect == "row_stride":
        return ccol, _strided_last_dim(row, 2)[0], values, call_size
    if defect == "index_dtype_mismatch":
        return ccol, row.to(torch.int32), values, call_size
    if defect == "index_dtype_unsupported":
        return ccol.to(torch.float32), row.to(torch.float32), values, call_size
    if defect == "meta_nnz":
        # The meta branch is reachable only when all three tensors are on meta,
        # and it requires no stored entries: the only meta workload the native
        # contract rejects.
        return ccol.to("meta"), row.to("meta"), values.to("meta"), call_size
    raise AssertionError("unhandled defect " + defect)


def _schema_defect_args(defect, ccol, row, values, call_size):
    if defect == "size_missing":
        return (ccol, row, values)
    if defect == "ccol_not_tensor":
        return (call_size, row, values, call_size)
    if defect == "size_not_int_list":
        return (ccol, row, values, "44")
    if defect == "size_float_entry":
        return (ccol, row, values, [4, 4.0])
    raise AssertionError("unhandled defect " + defect)


@pytest.mark.validate_sparse_csc_tensor_args
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("size", SIZE_SHAPES)
def test__validate_sparse_csc_tensor_args(size, value_range, dtype):
    # One stored entry per column of every matrix in the batch, always within the
    # nrows * ncols capacity.
    nnz = size[-1]
    ccol, row, values, call_size, parents = _make_csc_inputs(
        size, nnz, INDEX_DTYPE_LONG, dtype, value_range
    )
    snapshot = _snapshot(ccol, row, values, *parents)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], call_size
    )
    res_out = flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, call_size)

    # The schema returns (), so the only comparable value is the returned None.
    assert res_out is None
    # A validator must not modify the tensors it inspects.
    _assert_untouched(snapshot)


@pytest.mark.validate_sparse_csc_tensor_args
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("size,nnz,dense_ndim,index_dtype,layout", STRUCTURE_CASES)
def test__validate_sparse_csc_tensor_args_structure(
    size, nnz, dense_ndim, index_dtype, layout, value_range
):
    ccol, row, values, call_size, parents = _make_csc_inputs(
        size, nnz, index_dtype, torch.float32, value_range, dense_ndim, layout
    )
    snapshot = _snapshot(ccol, row, values, *parents)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], call_size
    )
    res_out = flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, call_size)

    assert res_out is None
    _assert_untouched(snapshot)


@pytest.mark.validate_sparse_csc_tensor_args
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test__validate_sparse_csc_tensor_args_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ccol, row = _csc_indices(SPECIAL_SIZE, values.numel(), INDEX_DTYPE_LONG)
    snapshot = _snapshot(ccol, row, values)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], list(SPECIAL_SIZE)
    )
    res_out = flag_gems._validate_sparse_csc_tensor_args(
        ccol, row, values, list(SPECIAL_SIZE)
    )

    assert res_out is None
    _assert_untouched(snapshot)


@pytest.mark.validate_sparse_csc_tensor_args
@pytest.mark.parametrize("defect,size,nnz", INVALID_ROWS)
def test__validate_sparse_csc_tensor_args_invalid(defect, size, nnz):
    ccol, row, values, call_size = _invalid_csc_inputs(defect, size, nnz)
    # Only the candidate is called: the native result is the reference contract,
    # not a second assertion.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, call_size)


@pytest.mark.validate_sparse_csc_tensor_args
@pytest.mark.parametrize("defect", SCHEMA_DEFECTS)
def test__validate_sparse_csc_tensor_args_schema_defects(defect):
    ccol, row, values, call_size, _ = _make_csc_inputs(
        (4, 4), 4, INDEX_DTYPE_LONG, torch.float32, ["-1", "1"]
    )
    args = _schema_defect_args(defect, ccol, row, values, call_size)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_sparse_csc_tensor_args(*args)


@pytest.mark.validate_sparse_csc_tensor_args
@pytest.mark.parametrize(
    "index_dtype", tu.selected_cases([torch.int64] if HAS_INT64_INDEX else [], quick=[])
)
def test__validate_sparse_csc_tensor_args_large_row_coordinate(index_dtype):
    ccol, row, values, call_size = _large_row_inputs(index_dtype, LARGE_ROW_COORDINATE)
    snapshot = _snapshot(ccol, row, values)

    torch.ops.aten._validate_sparse_csc_tensor_args(
        snapshot[0][1], snapshot[1][1], snapshot[2][1], call_size
    )
    res_out = flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, call_size)

    assert res_out is None
    _assert_untouched(snapshot)


@pytest.mark.validate_sparse_csc_tensor_args
def test__validate_sparse_csc_tensor_args_row_dimension_overflow():
    # int32 indices cannot describe the 2**31+5 row dimension, so the native
    # operator rejects the call on the host, before any kernel is launched.
    ccol, row, values, call_size = _large_row_inputs(torch.int32, 3)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_sparse_csc_tensor_args(ccol, row, values, call_size)
