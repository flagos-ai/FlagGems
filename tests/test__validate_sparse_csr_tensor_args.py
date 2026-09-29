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

from . import accuracy_utils as utils
from . import test_utils as tu

# `aten::_validate_sparse_csr_tensor_args(Tensor, Tensor, Tensor, int[]) -> ()`
# is a host-side structural validator. Its schema returns `()`, so there is no
# elementwise result to compare, no broadcast workload and no backward, and its
# only non-tensor argument is the required `int[] size`, which has no schema
# default. `values` is inspected for dtype, device and shape only, which is why
# every dtype the native validator accepts is exercised through it.
#
# Every dtype list below is built once at import time from the static backend
# capability flags that accuracy_utils re-exports; nothing here probes a device
# and nothing is filtered, skipped or xfailed at runtime.
VALUES_DTYPES = [
    torch.float32,
    torch.float16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.bool,
]
if utils.fp8_is_supported:
    VALUES_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.bf16_is_supported:
    VALUES_DTYPES.append(torch.bfloat16)
if utils.int64_is_supported:
    VALUES_DTYPES.append(torch.int64)
if utils.fp64_is_supported:
    VALUES_DTYPES.append(torch.float64)

FLOAT_VALUES_DTYPES = [dtype for dtype in VALUES_DTYPES if dtype.is_floating_point]

# crow_indices/col_indices are dispatched with AT_DISPATCH_INDEX_TYPES. The list
# is static, so a backend that does not advertise int64 never allocates an int64
# index tensor.
INDEX_DTYPES = [torch.int32] + ([torch.int64] if utils.int64_is_supported else [])

# Index dtype of the fixed-shape cases, picked statically. Every tensor below is
# allocated straight in the dtype that is used, never as int64 and downcast.
PRIMARY_INDEX_DTYPE = torch.int64 if utils.int64_is_supported else torch.int32

# crow_indices and col_indices must be Int or Long.
NON_INDEX_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
]
if utils.fp8_is_supported:
    NON_INDEX_DTYPES.append(torch.float8_e4m3fn)
if utils.fp64_is_supported:
    NON_INDEX_DTYPES.append(torch.float64)


def _make_csr(
    case, index_dtype=PRIMARY_INDEX_DTYPE, dtype=torch.float32, value_range=("-1", "1")
):
    """Assemble a CSR description from an explicit per-row entry count vector.

    A case is ``(size, row_entries, dense_dims)`` where ``size`` is the CSR
    tensor shape ``batch + (rows, cols) + dense_dims`` and ``row_entries[i]`` is
    the number of stored entries of row ``i``. Row ``i`` stores columns
    ``0 .. row_entries[i] - 1``, so the columns are sorted and distinct inside
    the row while the same column may appear again in another row, which CSR
    allows. Counts, crow_indices and the cumulative sum are all allocated in
    ``index_dtype``.
    """
    size, row_entries, dense_dims = case
    size = tuple(size)
    batch = size[: len(size) - 2 - len(dense_dims)]
    rows = size[len(batch)]
    batches = math.prod(batch) if batch else 1
    device = flag_gems.device

    counts = torch.zeros(rows, dtype=index_dtype, device=device)
    if row_entries:
        prefix = torch.tensor(row_entries[:rows], dtype=index_dtype, device=device)
        counts[: prefix.numel()] = prefix
    crow = torch.zeros(rows + 1, dtype=index_dtype, device=device)
    crow[1:] = counts.cumsum(0, dtype=index_dtype)

    if row_entries:
        col = torch.cat(
            [
                torch.arange(count, dtype=index_dtype, device=device)
                for count in row_entries
            ]
        )
    else:
        col = torch.empty(0, dtype=index_dtype, device=device)
    nnz = col.numel()

    crow_indices = crow.repeat(batches).reshape(batch + (rows + 1,))
    col_indices = col.repeat(batches).reshape(batch + (nnz,))
    values = tu.make_input(dtype, batch + (nnz,) + tuple(dense_dims), list(value_range))
    return crow_indices, col_indices, values, list(size)


def _tensor_metadata(tensor):
    """Shape, strides, storage offset and storage identity of an operand.

    ``tu.assert_result_equal`` compares dtype and stored values, so it cannot
    notice a candidate that swaps an operand for an equally valued clone. A
    validator may only read its inputs, so this snapshot must survive the call.
    """
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(),
    )


def _snapshot_metadata(tensors):
    return [_tensor_metadata(tensor) for tensor in tensors]


def _assert_metadata_unchanged(tensors, snapshots):
    for tensor, snapshot in zip(tensors, snapshots):
        assert _tensor_metadata(tensor) == snapshot


# `size[-2:]` are the sparse rows and columns, so a CSR description needs rank
# >= 2: the rank-0 and rank-1 shapes of the shared grid cannot describe one and
# are covered as invalid-size negative cases below.
SPEC_SHAPE_CASES = [
    (shape, [1] * min(shape[-2], 4), ())
    for shape in tu.selected_shapes()
    if len(shape) >= 2
]

# CSR boundary structures: nnz == 0, a single stored entry, one row, one column,
# empty rows between stored rows, ragged rows, dense dimensions and batching.
STRUCTURE_CASES = [
    ((3, 4), [2, 1], ()),
    ((2, 2), [], ()),
    ((1, 1), [1], ()),
    ((1, 5), [3], ()),
    ((5, 1), [1, 0, 0, 0], ()),
    ((4, 4), [2, 0, 1, 0], ()),
    ((2, 3, 6), [1, 3, 0], ()),
    ((3, 4, 2), [2, 1], (2,)),
    ((2, 3, 4, 2, 3), [2, 1], (2, 3)),
]

# validate_compressed_sparse_indices_kernel selects its static-shape kernel with
# `idx_max_ndims = 8` on col_indices.dim(), so both sides of that threshold are
# exercised (col_indices ranks 7, 8 and 9).
BATCH_RANK_CASES = [
    ((2, 2, 2, 2, 2, 2, 3, 4), [2, 1], ()),
    ((2, 2, 2, 2, 2, 2, 2, 3, 4), [2, 1], ()),
    ((2, 2, 2, 2, 2, 2, 2, 2, 3, 4), [2, 1], ()),
]

SIZE_CASES = tu.selected_cases(
    SPEC_SHAPE_CASES + STRUCTURE_CASES + BATCH_RANK_CASES,
    quick=SPEC_SHAPE_CASES,
)


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", VALUES_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("case", SIZE_CASES)
def test_validate_sparse_csr_tensor_args(case, value_range, dtype, index_dtype):
    crow_indices, col_indices, values, size = _make_csr(
        case, index_dtype=index_dtype, dtype=dtype, value_range=value_range
    )
    inputs = (crow_indices, col_indices, values)
    snapshots = _snapshot_metadata(inputs)
    # the independent references double as the pre-call value snapshot
    ref_crow_indices = tu.to_reference(crow_indices)
    ref_col_indices = tu.to_reference(col_indices)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_csr_tensor_args(
        ref_crow_indices, ref_col_indices, ref_values, list(size)
    )
    res_out = flag_gems._validate_sparse_csr_tensor_args(
        crow_indices, col_indices, values, list(size)
    )

    # the schema returns `()`, so the only documented result is None
    assert res_out is None
    tu.assert_result_equal(crow_indices, ref_crow_indices)
    tu.assert_result_equal(col_indices, ref_col_indices)
    tu.assert_result_equal(values, ref_values)
    _assert_metadata_unchanged(inputs, snapshots)


# Ragged row structures the one-entry rows of SPEC_SHAPE_CASES do not reach: two
# entries in one row, a row with none and mixed widths, plus zero-extent rows,
# columns and batches. Default-only: a zero-extent geometry is not a smoke case.
UNEVEN_GEOMETRIES = tu.selected_cases(
    [(0, 4), (3, 0), (2, 3, 0), (0, 3, 4), (2, 0, 4), (2, 3, 4)],
    quick=[],
)


def _uneven_row_entries(size):
    rows, cols = size[-2], size[-1]
    widths = (min(2, cols), 0, min(3, cols))
    return [widths[row] if row < len(widths) else min(1, cols) for row in range(rows)]


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("size", UNEVEN_GEOMETRIES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_validate_sparse_csr_tensor_args_uneven_rows(size, index_dtype):
    case = (tuple(size), _uneven_row_entries(size), ())
    crow_indices, col_indices, values, size = _make_csr(case, index_dtype=index_dtype)
    inputs = (crow_indices, col_indices, values)
    snapshots = _snapshot_metadata(inputs)

    ref_crow_indices = tu.to_reference(crow_indices)
    ref_col_indices = tu.to_reference(col_indices)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_csr_tensor_args(
        ref_crow_indices, ref_col_indices, ref_values, list(size)
    )
    res_out = flag_gems._validate_sparse_csr_tensor_args(
        crow_indices, col_indices, values, list(size)
    )

    assert res_out is None
    tu.assert_result_equal(crow_indices, ref_crow_indices)
    tu.assert_result_equal(col_indices, ref_col_indices)
    tu.assert_result_equal(values, ref_values)
    _assert_metadata_unchanged(inputs, snapshots)


# Only the index tensors must be contiguous per batch, so a values tensor that
# carries a storage offset or a stride is still a valid workload. Both index
# operands are views into a retained parent that holds a padding entry in front
# (non-zero storage offset), and the reference operands are sliced out of the
# transferred parents so whole parents and logical operands can both be checked.
VALUES_LAYOUT_KINDS = tu.selected_cases(["offset", "strided"], quick=[])


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("kind", VALUES_LAYOUT_KINDS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_validate_sparse_csr_tensor_args_offsets_and_strides(kind, index_dtype):
    rows, cols = 3, 4
    device = flag_gems.device
    crow_parent = torch.tensor([9, 0, 1, 2, 3], dtype=index_dtype, device=device)
    col_parent = torch.tensor([7, 0, 1, 2], dtype=index_dtype, device=device)
    if kind == "offset":
        values_parent = torch.randn(rows + 2, dtype=torch.float32, device=device)
        values_slice = slice(1, rows + 1)
    else:
        values_parent = torch.randn(2 * rows + 1, dtype=torch.float32, device=device)
        values_slice = slice(0, 2 * rows, 2)

    crow_indices = crow_parent[1:]
    col_indices = col_parent[1:]
    values = values_parent[values_slice]

    inputs = (
        crow_parent,
        col_parent,
        values_parent,
        crow_indices,
        col_indices,
        values,
    )
    snapshots = _snapshot_metadata(inputs)

    ref_crow_parent = tu.to_reference(crow_parent)
    ref_col_parent = tu.to_reference(col_parent)
    ref_values_parent = tu.to_reference(values_parent)
    ref_crow_indices = ref_crow_parent[1:]
    ref_col_indices = ref_col_parent[1:]
    ref_values = ref_values_parent[values_slice]

    torch.ops.aten._validate_sparse_csr_tensor_args(
        ref_crow_indices, ref_col_indices, ref_values, [rows, cols]
    )
    res_out = flag_gems._validate_sparse_csr_tensor_args(
        crow_indices, col_indices, values, [rows, cols]
    )

    assert res_out is None
    # the whole backing parents and the logical operands must both be untouched
    tu.assert_result_equal(crow_parent, ref_crow_parent)
    tu.assert_result_equal(col_parent, ref_col_parent)
    tu.assert_result_equal(values_parent, ref_values_parent)
    tu.assert_result_equal(crow_indices, ref_crow_indices)
    tu.assert_result_equal(col_indices, ref_col_indices)
    tu.assert_result_equal(values, ref_values)
    _assert_metadata_unchanged(inputs, snapshots)


# per-batch index views with a non-zero storage offset, plus a strided values
# view. Each row of each batch keeps sorted, distinct columns; the row layout
# differs between the two batches, which a valid description is allowed to do.
BATCHED_OFFSET_VIEW_SIZES = tu.selected_cases([(2, 3, 4)], quick=[])


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("size", BATCHED_OFFSET_VIEW_SIZES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_validate_sparse_csr_tensor_args_batched_offset_views(size, index_dtype):
    device = flag_gems.device
    crow_parent = torch.tensor(
        [[9, 0, 2, 3, 3], [9, 0, 1, 3, 3]], dtype=index_dtype, device=device
    )
    col_parent = torch.tensor(
        [[7, 0, 1, 2], [7, 3, 0, 1]], dtype=index_dtype, device=device
    )
    values_parent = torch.randn(2, 6, dtype=torch.float32, device=device)
    values_slice = slice(0, 6, 2)

    crow_indices = crow_parent[:, 1:]
    col_indices = col_parent[:, 1:]
    values = values_parent[:, values_slice]

    inputs = (
        crow_parent,
        col_parent,
        values_parent,
        crow_indices,
        col_indices,
        values,
    )
    snapshots = _snapshot_metadata(inputs)

    ref_crow_parent = tu.to_reference(crow_parent)
    ref_col_parent = tu.to_reference(col_parent)
    ref_values_parent = tu.to_reference(values_parent)
    ref_crow_indices = ref_crow_parent[:, 1:]
    ref_col_indices = ref_col_parent[:, 1:]
    ref_values = ref_values_parent[:, values_slice]

    torch.ops.aten._validate_sparse_csr_tensor_args(
        ref_crow_indices, ref_col_indices, ref_values, list(size)
    )
    res_out = flag_gems._validate_sparse_csr_tensor_args(
        crow_indices, col_indices, values, list(size)
    )

    assert res_out is None
    tu.assert_result_equal(crow_parent, ref_crow_parent)
    tu.assert_result_equal(col_parent, ref_col_parent)
    tu.assert_result_equal(values_parent, ref_values_parent)
    tu.assert_result_equal(crow_indices, ref_crow_indices)
    tu.assert_result_equal(col_indices, ref_col_indices)
    tu.assert_result_equal(values, ref_values)
    _assert_metadata_unchanged(inputs, snapshots)


# A single stored entry whose column index exceeds the 32-bit range. The dense
# size is never materialized, so only the sparse index carries the extreme
# value; the int64 index dtype is required here, which is why the case is gated
# on the advertised int64 capability (an int32 column tensor for the same entry
# would instead hit the native 32-bit overflow check).
LARGE_COLUMN_CASES = tu.selected_cases(
    [2**31] if utils.int64_is_supported else [], quick=[]
)


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("column", LARGE_COLUMN_CASES)
def test_validate_sparse_csr_tensor_args_column_index_beyond_int32(column):
    device = flag_gems.device
    size = [1, column + 1]
    crow_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
    col_indices = torch.tensor([column], dtype=torch.int64, device=device)
    values = torch.zeros(1, dtype=torch.float32, device=device)
    inputs = (crow_indices, col_indices, values)
    snapshots = _snapshot_metadata(inputs)

    ref_crow_indices = tu.to_reference(crow_indices)
    ref_col_indices = tu.to_reference(col_indices)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_csr_tensor_args(
        ref_crow_indices, ref_col_indices, ref_values, list(size)
    )
    res_out = flag_gems._validate_sparse_csr_tensor_args(
        crow_indices, col_indices, values, list(size)
    )

    assert res_out is None
    tu.assert_result_equal(crow_indices, ref_crow_indices)
    tu.assert_result_equal(col_indices, ref_col_indices)
    tu.assert_result_equal(values, ref_values)
    _assert_metadata_unchanged(inputs, snapshots)


# `values` never takes part in the checks, so a description holding NaN/Inf is
# accepted rather than rejected: this test keeps the whole shared payload (the
# nan, inf, -inf, +0 and -0 positions) and only asks that it still validates.
# e4m3fn cannot represent infinity, so tu.special_value_cases gives it the nan
# payload alone; every other float dtype also gets the inf and mixed payloads.
#
# There is deliberately no NaN/Inf negative case for this operator: values are
# never read, and the index tensors must be Int or Long, so an index tensor
# cannot carry a NaN in the first place.
SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(FLOAT_VALUES_DTYPES), quick=[]
)


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("dtype,scenario", SPECIAL_VALUE_CASES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_validate_sparse_csr_tensor_args_special_values(dtype, scenario, index_dtype):
    device = flag_gems.device
    values = tu.make_special_input(dtype, scenario).reshape(-1)
    nnz = values.numel()
    # one row holding every payload entry keeps the description valid:
    # crow_indices ends at nnz and col_indices ascends inside the column bound.
    crow_indices = torch.tensor([0, nnz], dtype=index_dtype, device=device)
    col_indices = torch.arange(nnz, dtype=index_dtype, device=device)
    size = [1, max(nnz, 1)]
    inputs = (crow_indices, col_indices, values)
    snapshots = _snapshot_metadata(inputs)

    ref_crow_indices = tu.to_reference(crow_indices)
    ref_col_indices = tu.to_reference(col_indices)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_csr_tensor_args(
        ref_crow_indices, ref_col_indices, ref_values, list(size)
    )
    res_out = flag_gems._validate_sparse_csr_tensor_args(
        crow_indices, col_indices, values, list(size)
    )

    assert res_out is None
    tu.assert_result_equal(crow_indices, ref_crow_indices)
    tu.assert_result_equal(col_indices, ref_col_indices)
    tu.assert_result_equal(values, ref_values)
    _assert_metadata_unchanged(inputs, snapshots)


# --- negative cases ----------------------------------------------------------
#
# Every case below is rejected by a host-side check, so the candidate raises a
# Python exception instead of aborting the process. The lists are plain, not
# tu.selected_cases, so the whole family also runs in --quick mode.
#
# UNRESOLVED COVERAGE: the index *value* invariants of a CSR description
# (crow_indices[0] == 0, non-decreasing crow_indices, crow_indices[-1] == nnz,
# 0 <= col_indices[i] < cols, and columns sorted and distinct within each row)
# are enforced on the device by at::_validate_compressed_sparse_indices, whose
# `_assert` is a CUDA_KERNEL_ASSERT. Feeding it a violating description aborts
# the process with a fatal device-side assert and leaves the CUDA context
# poisoned for every later test, so those descriptions are NOT delivered as
# pytest cases here and this file makes no full-coverage claim for them; they
# need a process-isolated harness.

VALID_CASE = ((3, 4), [2, 1], ())


def _valid_csr_args(index_dtype=PRIMARY_INDEX_DTYPE, dtype=torch.float32):
    return _make_csr(VALID_CASE, index_dtype=index_dtype, dtype=dtype)


def _index_tensor(values, index_dtype=PRIMARY_INDEX_DTYPE):
    return torch.tensor(values, dtype=index_dtype, device=flag_gems.device)


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize(
    "crow_len,col_len,match",
    [
        (3, 3, "must be equal to the number of rows"),
        (4, 2, "must be equal to nnz"),
    ],
)
def test_validate_sparse_csr_tensor_args_bad_index_length(crow_len, col_len, match):
    _, _, values, size = _valid_csr_args()
    crow_indices = torch.arange(
        crow_len, dtype=PRIMARY_INDEX_DTYPE, device=flag_gems.device
    )
    col_indices = torch.arange(
        col_len, dtype=PRIMARY_INDEX_DTYPE, device=flag_gems.device
    )

    with pytest.raises(RuntimeError, match=match):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, size
        )


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("bad_size", [[], [3], [256], [3, 4, 5]])
def test_validate_sparse_csr_tensor_args_bad_size_rank(bad_size):
    crow_indices, col_indices, values, _ = _valid_csr_args()

    with pytest.raises(
        RuntimeError,
        match="tensor dimensionality must be sum of batch, base, and dense dimensionalities",
    ):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, list(bad_size)
        )


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize(
    "match",
    [
        "crow_indices must have dimensionality >= 1",
        "values must have dimensionality > sum of batch and block dimensionalities",
    ],
)
def test_validate_sparse_csr_tensor_args_zero_rank_argument(match):
    crow_indices, col_indices, values, size = _valid_csr_args()
    if match.startswith("crow_indices"):
        crow_indices = torch.tensor(
            0, dtype=PRIMARY_INDEX_DTYPE, device=flag_gems.device
        )
    else:
        values = torch.tensor(1.0, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match=match):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, size
        )


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize(
    "crow_shape,col_shape,values_shape,size",
    [
        # crow_indices and col_indices must have the same number of dimensions
        ((1, 4), (3,), (1, 3), [1, 3, 4]),
        # the batch dimensions of crow_indices drive the size's batch dimensions
        ((2, 4), (2, 3), (2, 3), [3, 3, 4]),
        # crow_indices, col_indices and values must share their batch dimensions
        ((2, 4), (3, 3), (2, 3), [2, 3, 4]),
        ((3, 4), (2, 3), (2, 3), [2, 3, 4]),
        ((2, 4), (2, 3), (3, 3), [2, 3, 4]),
    ],
)
def test_validate_sparse_csr_tensor_args_index_shape_mismatch(
    crow_shape, col_shape, values_shape, size
):
    device = flag_gems.device
    crow_indices = torch.zeros(crow_shape, dtype=PRIMARY_INDEX_DTYPE, device=device)
    col_indices = torch.zeros(col_shape, dtype=PRIMARY_INDEX_DTYPE, device=device)
    values = torch.zeros(values_shape, dtype=torch.float32, device=device)

    with pytest.raises(RuntimeError, match="dimensionalit|batch dimensions"):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, list(size)
        )


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize(
    "size,values_shape,match",
    [
        # crow_indices[-1] is 3, so values must hold three entries
        ([3, 4, 2], (2, 2), "must be equal to nnz"),
        # a dense dimension of rank 2 makes the tensor rank one too high
        ([3, 4, 2], (3, 2, 2), "tensor dimensionality must be sum"),
    ],
)
def test_validate_sparse_csr_tensor_args_values_shape_mismatch(
    size, values_shape, match
):
    device = flag_gems.device
    crow_indices = _index_tensor([0, 2, 3, 3])
    col_indices = _index_tensor([0, 1, 2])
    values = torch.zeros(values_shape, dtype=torch.float32, device=device)

    with pytest.raises(RuntimeError, match=match):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, list(size)
        )


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("dtype", NON_INDEX_DTYPES)
def test_validate_sparse_csr_tensor_args_non_integer_index_rejected(dtype):
    device = flag_gems.device
    crow_indices = torch.zeros(4, dtype=dtype, device=device)
    col_indices = torch.zeros(3, dtype=dtype, device=device)
    values = torch.zeros(3, dtype=torch.float32, device=device)

    with pytest.raises(RuntimeError, match="must be Int or Long"):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, [3, 4]
        )


MIXED_INDEX_DTYPES = [(torch.int32, torch.int64)] if utils.int64_is_supported else []


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("crow_dtype,col_dtype", MIXED_INDEX_DTYPES)
def test_validate_sparse_csr_tensor_args_mixed_index_dtype_rejected(
    crow_dtype, col_dtype
):
    crow_indices = _index_tensor([0, 1, 2], crow_dtype)
    col_indices = _index_tensor([0, 1], col_dtype)
    values = torch.zeros(2, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="must have the same dtype"):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, [2, 4]
        )


# repeat_interleave followed by a step-2 slice keeps the index values valid
# ([0, 1, 2, 3] and [0, 1, 2]) while making the layout non-contiguous, so only
# the per-batch contiguity check can reject the description.
NON_CONTIGUOUS_TARGETS = ["crow_indices", "col_indices"]


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("target", NON_CONTIGUOUS_TARGETS)
def test_validate_sparse_csr_tensor_args_non_contiguous_index_rejected(target):
    crow_indices, col_indices, values, size = _valid_csr_args()
    sparse = torch.arange(4, dtype=PRIMARY_INDEX_DTYPE, device=flag_gems.device)
    sparse = sparse.repeat_interleave(2)[::2]
    if target == "crow_indices":
        crow_indices = sparse
    else:
        col_indices = sparse[:3]

    with pytest.raises(RuntimeError, match="to be a contiguous tensor per batch"):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, size
        )


# `cpu` and `meta` are deliberately wrong devices for `values` (the index
# tensors stay on flag_gems.device). `cpu` only differs when the target is not
# CPU, so it is selected statically from flag_gems.device; `meta` always
# differs, which keeps the case collected in every mode.
OTHER_DEVICES = (["cpu"] if str(flag_gems.device) != "cpu" else []) + ["meta"]


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("other_device", OTHER_DEVICES)
def test_validate_sparse_csr_tensor_args_values_device_mismatch_rejected(other_device):
    crow_indices, col_indices, _, size = _valid_csr_args()
    values = torch.zeros(3, dtype=torch.float32, device=other_device)

    with pytest.raises(RuntimeError, match="device"):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, size
        )


@pytest.mark.validate_sparse_csr_tensor_args
@pytest.mark.parametrize("bad_size", [[-3, 4]], ids=["negative_num_rows"])
def test_validate_sparse_csr_tensor_args_negative_num_rows(bad_size):
    crow_indices, col_indices, values, _ = _valid_csr_args()

    with pytest.raises(RuntimeError, match="must be equal to the number of rows"):
        flag_gems._validate_sparse_csr_tensor_args(
            crow_indices, col_indices, values, list(bad_size)
        )


@pytest.mark.validate_sparse_csr_tensor_args
def test_validate_sparse_csr_tensor_args_non_tensor_crow_indices():
    _, col_indices, values, size = _valid_csr_args()
    # the native dispatcher rejects a non-Tensor argument with TypeError; a
    # candidate that validates before dispatching may reject it with
    # RuntimeError. Both are a rejection of this invalid description.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._validate_sparse_csr_tensor_args([0, 2, 3], col_indices, values, size)
