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

# _validate_sparse_bsr_tensor_args(crow_indices, col_indices, values, size) returns
# None. It checks the metadata of a blocked sparse-row (BSR) layout - the ranks, the
# extents, the index dtype and the agreement between 'size', 'crow_indices',
# 'col_indices' and the payload shape - together with the index invariants of that
# layout (the row offsets and the column block indices). It reads its inputs and
# returns no tensor, so nothing is numerically comparable: every positive test below
# asserts the candidate's None return and compares the three input tensors that the
# candidate was given.
#
# Layout formulas used throughout:
#   * logical 'size' has len(batch) + 2 sparse (block row/column) extents + len(dense)
#     dense extents, i.e. rank = batch + 2 + dense;
#   * 'values' has rank = batch + 1 NNZ + 2 block (block row/column) + dense, where
#     the NNZ axis counts stored blocks and is the axis directly after the batch axes;
#   * crow_indices.shape[-1] == row_blocks + 1;
#   * col_indices.shape[-1] == nnz == values.shape[len(batch)].
#
# Coverage notes:
#   * No broadcast dimension: the four operands describe one compressed structure and
#     must agree exactly, so no broadcastable operand pair has a meaning here.
#   * No backward dimension: the operator returns None, so there is no output to
#     differentiate.
#   * No tensor/scalar operand pair: the only non-tensor argument is 'size', an int
#     list, covered positionally, as a tuple container and as keyword arguments.
#   * The rank-8 and rank-9 index cases below are extra coverage of the index-rank
#     boundary at high batch rank; they are not evidence that a higher rank is
#     rejected.
#   * Unresolved coverage, recorded rather than claimed as a settled limitation:
#     malformed index domains (a first row offset other than 0, a last row offset
#     other than nnz, unsorted or duplicated column blocks, column blocks outside the
#     column extent) are not covered. The native validator checks those inside its
#     compute kernel, and no archived evidence of a context-safe rejection was
#     readable from the checkout this task may inspect, so no unsafe malformed
#     indices are run and CPU is not used as a substitute oracle.

_ALL_VALUES_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.float16,
    torch.int32,
    torch.bool,
]
if utils.bf16_is_supported:
    _ALL_VALUES_DTYPES.append(torch.bfloat16)
if utils.fp8_is_supported:
    _ALL_VALUES_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.int64_is_supported:
    _ALL_VALUES_DTYPES.append(torch.int64)
if utils.fp64_is_supported:
    _ALL_VALUES_DTYPES.append(torch.float64)

# Index dtypes the device offers, read from the suite's static capability flags, so no
# case requests an unavailable dtype and nothing is probed at collection or run time.
_INDEX_DTYPES = [torch.int32] + ([torch.int64] if utils.int64_is_supported else [])
_DEFAULT_INDEX_DTYPE = _INDEX_DTYPES[-1]
_INDEX_DTYPE_MISMATCHES = [
    (first, second)
    for first in _INDEX_DTYPES
    for second in _INDEX_DTYPES
    if first != second
]
_NON_INDEX_DTYPES = [torch.int16, torch.float32, torch.bool]

# Both modes retain supported dtypes; quick reduces only the shape/range grid.
# Structural, special-value and supplemental call forms remain default-only.
_VALUES_DTYPES = _ALL_VALUES_DTYPES
_SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_ALL_VALUES_DTYPES), quick=[]
)

# A BSR 'size' has rank >= 2: its last two entries are the sparse block extents and the
# leading entries are batch extents, so the spec's 0-dim and 1-dim shapes have no BSR
# form and are dropped. Every remaining spec shape is used as 'size' itself with a
# chosen small 1x1 block, so the block extents multiply their own block counts by one.
_SHAPES = tu.selected_cases(
    [tuple(shape) for shape in tu.selected_shapes() if len(shape) >= 2],
    quick=[(2, 19, 7)],
)

# Structural variants of the compressed layout: block sizes and densities, zero
# extents on the row axis, the column axis and a dense axis, batched structures,
# independent per-batch row offsets, both index dtypes and the index-rank boundary.
# The payload never influences the validator, so each deck runs on one value range
# while the dtype x range x shape grid above covers the five ranges.
_STRUCTURE_DECKS = tu.selected_cases(
    [
        pytest.param(
            {"row_blocks": 4, "col_blocks": 4, "blocks_per_row": 1}, id="square_1x1"
        ),
        pytest.param(
            {"row_blocks": 3, "col_blocks": 4, "blocks_per_row": 1, "block": (2, 3)},
            id="square_2x3",
        ),
        pytest.param(
            {"row_blocks": 3, "col_blocks": 12, "blocks_per_row": 3}, id="dense_3_of_12"
        ),
        pytest.param(
            {"row_blocks": 0, "col_blocks": 4, "blocks_per_row": 0},
            id="zero_row_extent",
        ),
        pytest.param(
            {"row_blocks": 4, "col_blocks": 0, "blocks_per_row": 0},
            id="zero_column_extent",
        ),
        pytest.param(
            {"row_blocks": 0, "col_blocks": 0, "blocks_per_row": 0},
            id="zero_both_extents",
        ),
        pytest.param(
            {"row_blocks": 0, "col_blocks": 4, "blocks_per_row": 0, "batch": (2,)},
            id="zero_row_extent_batched",
        ),
        pytest.param(
            {
                "row_blocks": 3,
                "col_blocks": 0,
                "blocks_per_row": 0,
                "block": (2, 4),
                "batch": (2,),
            },
            id="zero_column_extent_batched",
        ),
        pytest.param(
            {"row_blocks": 2, "col_blocks": 2, "blocks_per_row": 1, "dense": (0,)},
            id="zero_dense_extent",
        ),
        pytest.param(
            {"row_blocks": 4, "col_blocks": 6, "blocks_per_row": 2, "batch": (2,)},
            id="batched_2x3",
        ),
        pytest.param(
            {
                "row_blocks": 4,
                "col_blocks": 4,
                "blocks_per_row": 1,
                "batch": (2,),
                "dense": (2, 3),
            },
            id="batched_dense",
        ),
        pytest.param(
            {"row_blocks": 3, "col_blocks": 5, "blocks_per_row": 2, "dense": (2, 3, 4)},
            id="trailing_dense_3d",
        ),
        pytest.param(
            {
                "row_blocks": 4,
                "col_blocks": 4,
                "batch": (2,),
                "crow_per_batch": ([0, 1, 2, 3, 4], [0, 2, 4, 4, 4]),
            },
            id="independent_batch_structure",
        ),
        pytest.param(
            {
                "row_blocks": 4,
                "col_blocks": 4,
                "blocks_per_row": 1,
                "index_dtype": torch.int32,
            },
            id="int32_indices",
        ),
        pytest.param(
            {
                "row_blocks": 3,
                "col_blocks": 5,
                "blocks_per_row": 2,
                "batch": (2,),
                "index_dtype": torch.int32,
            },
            id="int32_indices_batched",
        ),
        pytest.param(
            {"row_blocks": 2, "col_blocks": 2, "blocks_per_row": 1, "batch": (1,) * 7},
            id="high_rank_crow_8",
        ),
        pytest.param(
            {"row_blocks": 2, "col_blocks": 2, "blocks_per_row": 1, "batch": (1,) * 8},
            id="high_rank_crow_9",
        ),
    ],
    quick=[],
)

# Supplemental default-only families: an operand layout with a storage-offset view and
# a strided payload slice, the tuple container of 'size', and a keyword-argument call.
_LAYOUT_CASES = tu.selected_cases(
    [
        pytest.param(
            {"row_blocks": 4, "blocks_per_row": 2, "col_blocks": 8, "block": (2, 4)},
            id="offset_crow_strided_values",
        )
    ],
    quick=[],
)
_SIZE_CONTAINERS = tu.selected_cases([pytest.param(tuple, id="tuple_size")], quick=[])
_KEYWORD_NAMES = tu.selected_cases(
    [
        pytest.param(
            ("crow_indices", "col_indices", "values", "size"), id="keyword_call"
        )
    ],
    quick=[],
)

# Host-side metadata checks raise RuntimeError; the tuple also accepts the TypeError a
# Python-level argument check would raise.
_NEGATIVE_EXC = (RuntimeError, TypeError)


def _logical_size(descriptor):
    """size = batch + [block row extent, block column extent] + dense."""
    block = tuple(descriptor.get("block", (1, 1)))
    if len(block) != 2:
        raise ValueError("a BSR block has exactly two dimensions")
    batch = tuple(descriptor.get("batch", ()))
    dense = tuple(descriptor.get("dense", ()))
    extents = (
        list(batch)
        + list(dense)
        + list(block)
        + [
            descriptor["row_blocks"],
            descriptor["col_blocks"],
            descriptor.get("blocks_per_row", 1),
        ]
    )
    for value in extents:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"a BSR extent must be an integer, got {value!r}")
        if value < 0:
            raise ValueError(f"a BSR extent must be non-negative, got {value}")
    size = (
        list(batch)
        + [descriptor["row_blocks"] * block[0], descriptor["col_blocks"] * block[1]]
        + list(dense)
    )
    if len(size) < 2:
        raise ValueError("a BSR size needs at least the two block extents")
    return size


def _batch_rows(batch):
    rows = 1
    for extent in batch:
        rows *= extent
    return rows


def _check_offsets(offsets, row_blocks):
    """Row offsets are non-negative integers starting at 0 and non-decreasing."""
    for value in offsets:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"row offsets must be non-negative integers, got {value!r}"
            )
    if len(offsets) != row_blocks + 1:
        raise ValueError("crow_indices needs exactly row_blocks + 1 offsets")
    if offsets[0] != 0:
        raise ValueError("the first row offset must be 0")
    for previous, current in zip(offsets, offsets[1:]):
        if current < previous:
            raise ValueError("crow offsets must be non-decreasing")


def _uniform_offsets(descriptor):
    """One structure for every batch row: blocks_per_row stored blocks per row."""
    col_blocks = descriptor["col_blocks"]
    blocks_per_row = descriptor.get("blocks_per_row", 1)
    if blocks_per_row > col_blocks:
        raise ValueError(
            f"blocks_per_row ({blocks_per_row}) cannot exceed col_blocks ({col_blocks})"
        )
    return [row * blocks_per_row for row in range(descriptor["row_blocks"] + 1)]


def _column_blocks(offsets, col_blocks):
    """Sorted, distinct, in-range column blocks for every row of one crow row."""
    columns = []
    for row, (start, end) in enumerate(zip(offsets, offsets[1:])):
        count = end - start
        if count > col_blocks:
            raise ValueError(
                f"row {row} stores {count} blocks but the column extent is {col_blocks}"
            )
        if count == 0:
            continue
        span = col_blocks - count + 1
        base = (row * count) % span
        columns.extend(base + step for step in range(count))
    return columns


def _build_bsr(descriptor, dtype, value_range):
    """Build the (crow_indices, col_indices, values, size) BSR operands.

    Column blocks stay sorted, distinct and inside the column extent, and every batch
    row ends at the same stored-block count (values is dense over NNZ), so the
    produced structure satisfies the layout invariants. Requested extents are used
    exactly as given: an inconsistent request raises instead of being adjusted.
    """
    device = flag_gems.device
    block = tuple(descriptor.get("block", (1, 1)))
    batch = tuple(descriptor.get("batch", ()))
    dense = tuple(descriptor.get("dense", ()))
    row_blocks = descriptor["row_blocks"]
    col_blocks = descriptor["col_blocks"]
    index_dtype = descriptor.get("index_dtype", _DEFAULT_INDEX_DTYPE)
    if index_dtype not in _INDEX_DTYPES:
        raise ValueError(f"index dtype {index_dtype} is unavailable on this device")
    size = _logical_size(descriptor)

    crow_per_batch = descriptor.get("crow_per_batch")
    if crow_per_batch is None:
        offsets = _uniform_offsets(descriptor)
        nnz = offsets[-1]
        crow = (
            torch.tensor(offsets, dtype=index_dtype, device=device)
            .expand(batch + (row_blocks + 1,))
            .contiguous()
        )
        col = (
            torch.tensor(
                _column_blocks(offsets, col_blocks), dtype=index_dtype, device=device
            )
            .expand(batch + (nnz,))
            .contiguous()
        )
    else:
        offsets_per_batch = [list(offsets) for offsets in crow_per_batch]
        for offsets in offsets_per_batch:
            _check_offsets(offsets, row_blocks)
        counts = {offsets[-1] for offsets in offsets_per_batch}
        if len(counts) != 1:
            raise ValueError("every batch row must store the same number of blocks")
        nnz = counts.pop()
        if len(offsets_per_batch) != _batch_rows(batch):
            raise ValueError(
                "crow_per_batch must describe exactly one structure per batch row"
            )
        crow = torch.tensor(
            offsets_per_batch, dtype=index_dtype, device=device
        ).reshape(batch + (row_blocks + 1,))
        col = torch.tensor(
            [_column_blocks(offsets, col_blocks) for offsets in offsets_per_batch],
            dtype=index_dtype,
            device=device,
        ).reshape(batch + (nnz,))
    values = tu.make_input(dtype, batch + (nnz,) + block + dense, value_range)
    return crow, col, values, size


def _base_operands():
    """A valid 4 x 4 BSR structure with one stored block per row."""
    return _build_bsr(
        {"row_blocks": 4, "col_blocks": 4, "blocks_per_row": 1},
        torch.float32,
        ["-1", "1"],
    )


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__validate_sparse_bsr_tensor_args_shapes_and_ranges(shape, value_range, dtype):
    descriptor = {
        "row_blocks": shape[-2],
        "col_blocks": shape[-1],
        "blocks_per_row": 1,
        "batch": tuple(shape[:-2]),
    }
    crow, col, values, size = _build_bsr(descriptor, dtype, value_range)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsr_tensor_args(ref_crow, ref_col, ref_values, size)
    res_out = flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, size)

    assert res_out is None
    tu.assert_result_equal(crow, ref_crow)
    tu.assert_result_equal(col, ref_col)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("descriptor", _STRUCTURE_DECKS)
def test__validate_sparse_bsr_tensor_args_structures(descriptor):
    crow, col, values, size = _build_bsr(descriptor, torch.float32, ["-1", "1"])
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsr_tensor_args(ref_crow, ref_col, ref_values, size)
    res_out = flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, size)

    assert res_out is None
    tu.assert_result_equal(crow, ref_crow)
    tu.assert_result_equal(col, ref_col)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test__validate_sparse_bsr_tensor_args_special_values(dtype, scenario):
    # The validator never reads the payload, so NaN and Inf blocks must be accepted
    # exactly like finite ones.
    descriptor = {"row_blocks": 5, "col_blocks": 5, "blocks_per_row": 1}
    crow, col, _, size = _build_bsr(descriptor, dtype, ["-1", "1"])
    values = tu.make_special_input(dtype, scenario).reshape(5, 1, 1)
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsr_tensor_args(ref_crow, ref_col, ref_values, size)
    res_out = flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, size)

    assert res_out is None
    tu.assert_result_equal(crow, ref_crow)
    tu.assert_result_equal(col, ref_col)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("layout", _LAYOUT_CASES)
def test__validate_sparse_bsr_tensor_args_accepts_offset_and_strided_inputs(layout):
    # crow_indices may be a contiguous view with a leading storage offset, and the
    # payload may be a strided slice of a wider parent; col_indices must stay
    # contiguous per batch.
    device = flag_gems.device
    row_blocks = layout["row_blocks"]
    blocks_per_row = layout["blocks_per_row"]
    block_rows, block_cols = layout["block"]
    nnz = row_blocks * blocks_per_row

    crow_storage = torch.zeros(
        row_blocks + 2, dtype=_DEFAULT_INDEX_DTYPE, device=device
    )
    crow_storage[1:] = (
        torch.arange(row_blocks + 1, dtype=_DEFAULT_INDEX_DTYPE, device=device)
        * blocks_per_row
    )
    crow = crow_storage[1:]
    col = torch.arange(nnz, dtype=_DEFAULT_INDEX_DTYPE, device=device)
    values_storage = tu.make_input(
        torch.float32, (nnz, block_rows, block_cols), ["-1", "1"]
    )
    values = values_storage[:, :, :2]
    size = [row_blocks * block_rows, layout["col_blocks"] * 2]

    # Both views are compared through their parents, so a write into the untouched crow
    # sentinel or the payload padding cannot hide behind an equal view. Snapshot the
    # view semantics first: a candidate returning a rebound storage keeps the values.
    crow_view = (crow.storage_offset(), crow.stride())
    values_view = (values.storage_offset(), values.stride())

    ref_crow_storage = tu.to_reference(crow_storage)
    ref_values_storage = tu.to_reference(values_storage)
    ref_crow = ref_crow_storage[1:]
    ref_values = ref_values_storage[:, :, :2]
    ref_col = tu.to_reference(col)

    torch.ops.aten._validate_sparse_bsr_tensor_args(ref_crow, ref_col, ref_values, size)
    res_out = flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, size)

    assert res_out is None
    tu.assert_result_equal(crow_storage, ref_crow_storage)
    tu.assert_result_equal(values_storage, ref_values_storage)
    tu.assert_result_equal(crow, ref_crow)
    tu.assert_result_equal(col, ref_col)
    tu.assert_result_equal(values, ref_values)
    assert (crow.storage_offset(), crow.stride()) == crow_view
    assert (values.storage_offset(), values.stride()) == values_view


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("container", _SIZE_CONTAINERS)
def test__validate_sparse_bsr_tensor_args_accepts_size_container(container):
    crow, col, values, size = _base_operands()
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsr_tensor_args(
        ref_crow, ref_col, ref_values, container(size)
    )
    res_out = flag_gems._validate_sparse_bsr_tensor_args(
        crow, col, values, container(size)
    )

    assert res_out is None
    tu.assert_result_equal(crow, ref_crow)
    tu.assert_result_equal(col, ref_col)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("names", _KEYWORD_NAMES)
def test__validate_sparse_bsr_tensor_args_accepts_keyword_arguments(names):
    crow, col, values, size = _base_operands()
    ref_crow = tu.to_reference(crow)
    ref_col = tu.to_reference(col)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsr_tensor_args(
        **dict(zip(names, (ref_crow, ref_col, ref_values, size)))
    )
    res_out = flag_gems._validate_sparse_bsr_tensor_args(
        **dict(zip(names, (crow, col, values, size)))
    )

    assert res_out is None
    tu.assert_result_equal(crow, ref_crow)
    tu.assert_result_equal(col, ref_col)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_short_crow_indices():
    crow, col, values, _ = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow[:-1], col, values, [4, 4])


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_short_col_indices():
    crow, col, values, _ = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, col[:-1], values, [4, 4])


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("index_pair", _INDEX_DTYPE_MISMATCHES)
def test__validate_sparse_bsr_tensor_args_rejects_index_dtype_mismatch(index_pair):
    crow, col, values, size = _base_operands()
    crow_dtype, col_dtype = index_pair
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(
            crow.to(crow_dtype), col.to(col_dtype), values, size
        )


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize("index_dtype", _NON_INDEX_DTYPES)
def test__validate_sparse_bsr_tensor_args_rejects_non_index_dtype(index_dtype):
    crow, col, values, size = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(
            crow.to(index_dtype), col.to(index_dtype), values, size
        )


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_non_contiguous_crow_indices():
    crow, col, values, size = _base_operands()
    storage = torch.zeros(2 * crow.numel(), dtype=crow.dtype, device=flag_gems.device)
    storage[::2] = crow.reshape(-1)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(storage[::2], col, values, size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_non_contiguous_col_indices():
    crow, col, values, size = _base_operands()
    storage = torch.zeros(2 * col.numel(), dtype=col.dtype, device=flag_gems.device)
    storage[::2] = col.reshape(-1)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, storage[::2], values, size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_index_batch_mismatch():
    crow, col, values, size = _base_operands()
    batched_crow = crow.expand(2, crow.numel()).contiguous()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(batched_crow, col, values, size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_index_rank_mismatch():
    crow, col, values, size = _base_operands()
    batched_col = col.expand(2, col.numel()).contiguous()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, batched_col, values, size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_values_batch_mismatch():
    # The indices describe a batch extent that 'values' does not have.
    crow, col, _, size = _build_bsr(
        {"row_blocks": 3, "col_blocks": 4, "blocks_per_row": 1, "batch": (2,)},
        torch.float32,
        ["-1", "1"],
    )
    unbatched_values = tu.make_input(torch.float32, (3, 1, 1), ["-1", "1"])
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, col, unbatched_values, size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_scalar_crow_indices():
    crow, col, values, size = _base_operands()
    scalar = torch.tensor(0, dtype=crow.dtype, device=flag_gems.device)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(scalar, col, values, size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_size_rank_below_two():
    crow, col, values, size = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, size[:1])


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_values_rank_too_low():
    crow, col, values, size = _base_operands()
    flat_values = torch.zeros(values.numel(), dtype=values.dtype, device=values.device)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, col, flat_values, size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_size_dense_rank_mismatch():
    # 'size' declares a trailing dense extent that 'values' does not have.
    crow, col, values, size = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, size + [2])


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_size_not_divisible_by_block():
    crow, col, values, size = _build_bsr(
        {"row_blocks": 3, "col_blocks": 4, "blocks_per_row": 1, "block": (2, 3)},
        torch.float32,
        ["-1", "1"],
    )
    with pytest.raises(_NEGATIVE_EXC):
        # The built row extent 6 is a multiple of the block row extent 2, while the
        # requested 7 is not.
        flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, [7, size[1]])


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_negative_extent():
    crow, col, values, size = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(
            crow, col, values, [-size[0], size[1]]
        )


@pytest.mark.validate_sparse_bsr_tensor_args
@pytest.mark.parametrize(
    "bad_size",
    [
        pytest.param([8.0, 8], id="float_extent"),
        pytest.param(8, id="scalar"),
        pytest.param(None, id="none"),
    ],
)
def test__validate_sparse_bsr_tensor_args_rejects_invalid_size(bad_size):
    crow, col, values, _ = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, col, values, bad_size)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_missing_size():
    crow, col, values, _ = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow, col, values)


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_unexpected_keyword():
    crow, col, values, size = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(
            crow_indices=crow, col_indices=col, values=values, dims=size
        )


@pytest.mark.validate_sparse_bsr_tensor_args
def test__validate_sparse_bsr_tensor_args_rejects_non_tensor_indices():
    crow, col, values, size = _base_operands()
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._validate_sparse_bsr_tensor_args(crow.tolist(), col, values, size)
