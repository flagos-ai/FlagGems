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

from . import accuracy_utils as au
from . import test_utils as tu

# aten::_validate_sparse_compressed_tensor_args(compressed_indices, plain_indices,
#     values, int[] size, Layout layout) -> ()
#
# The op is a pure structural validator: it returns nothing and only reads the two
# index tensors, the values tensor, the logical shape and the layout. Every case
# therefore passes the same description to the native reference and to the
# candidate and checks that the candidate returns None without touching its inputs.
#
# A description is written as
#     (layout, batch extents, (rows, cols), dense extents, block shape, nnz)
# and the logical size handed to the op is batch + (rows, cols) + dense, which is
# what the operator's `size` argument means. The spec's pointwise shapes appear as
# logical sizes: (1024, 1024) and (20, 320) as the 2-D base, and the rank 3..5
# shapes as base plus batch/dense extents, e.g. (16, 128, 64, 60) as batch (16,),
# base (128, 64), dense (60,) and (16, 7, 57, 32, 29) as batch (16, 7), base
# (57, 32), dense (29,). Ranks 0 and 1 cannot describe a compressed layout (the op
# needs batch + 2 base + dense entries) and are covered as negative cases.
#
# `layout` is the operator's explicit non-tensor parameter and is swept over all
# four valid compressed layouts in the grid, with the invalid layouts covered as
# negatives. Other parameter boundaries are exercised too: both accepted index
# widths, block shapes from 1x1 up to 2x3 (square, non-square and 1-wide), nnz from
# 0 to full density, and zero batch/dense/base extents. Dimensions that do not
# apply, with the reason:
#  * Broadcast: the op compares each operand's extent against `size`; it never
#    broadcasts operands against one another.
#  * Backward: the schema returns (), so there is no tensor output and no autograd
#    formula.
#  * Scalar operand: there is no scalar argument.
#  * Index *content* violations (out-of-range, unsorted or duplicated plain
#    indices) are not delivered: the native op validates content with a device-side
#    assertion kernel that fails asynchronously and poisons the accelerator context
#    instead of raising, so such a case cannot be exercised in-process. This stays
#    uncovered here.

_ROW_MAJOR_LAYOUTS = (torch.sparse_csr, torch.sparse_bsr)
_BLOCKED_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)

# Static capability flags of the active backend; the values component of this op
# accepts any strided dtype, so the required dtype list is filtered only where the
# backend cannot materialise the dtype at all.
_DTYPE_GATES = {
    torch.bfloat16: au.bf16_is_supported,
    torch.float64: au.fp64_is_supported,
    torch.int64: au.int64_is_supported,
    torch.float8_e4m3fn: au.fp8_is_supported,
    torch.float8_e5m2: au.fp8_is_supported,
}


def _supported(dtypes):
    return [dtype for dtype in dtypes if _DTYPE_GATES.get(dtype, True)]


_INDEX_DTYPES = tu.selected_cases(
    _supported([torch.int32, torch.int64]), quick=_supported([torch.int32])
)
_VALUES_DTYPES = _supported(tu.REQUIRED_DTYPES)

# Six rows make up the quick subset: both index orientations, both block layouts
# (each a square 2x2 block shape), an empty description and a zero compressed extent.
_CSR_QUICK = (torch.sparse_csr, (), (3, 4), (), (1, 1), 5)
_CSC_QUICK = (torch.sparse_csc, (), (3, 4), (), (1, 1), 5)
_BSR_QUICK = (torch.sparse_bsr, (), (6, 6), (), (2, 2), 2)
_BSC_QUICK = (torch.sparse_bsc, (), (6, 6), (), (2, 2), 2)
_EMPTY_QUICK = (torch.sparse_csr, (), (7, 13), (), (1, 1), 0)
_ZERO_QUICK = (torch.sparse_csr, (), (0, 4), (), (1, 1), 0)

_DESCRIPTORS = tu.selected_cases(
    [
        _CSR_QUICK,
        _CSC_QUICK,
        _BSR_QUICK,
        _BSC_QUICK,
        _EMPTY_QUICK,
        _ZERO_QUICK,
        # Single stored entry.
        (torch.sparse_csr, (), (1, 1), (), (1, 1), 1),
        (torch.sparse_csc, (), (1, 1), (), (1, 1), 1),
        # Fewer stored entries than compressed rows.
        (torch.sparse_csr, (), (6, 6), (), (1, 1), 2),
        (torch.sparse_csc, (), (6, 6), (), (1, 1), 2),
        # Empty descriptions (nnz == 0), plain and blocked.
        (torch.sparse_csc, (), (7, 13), (), (1, 1), 0),
        (torch.sparse_bsr, (), (8, 12), (), (2, 2), 0),
        (torch.sparse_bsc, (), (8, 12), (), (2, 2), 0),
        # Full density: nnz == compressed_dim * plain_dim.
        (torch.sparse_csr, (), (7, 13), (), (1, 1), 91),
        # Large logical extents.
        (torch.sparse_csr, (), (1024, 1024), (), (1, 1), 256),
        (torch.sparse_csc, (), (1024, 1024), (), (1, 1), 256),
        (torch.sparse_bsr, (), (1024, 1024), (), (2, 2), 256),
        (torch.sparse_bsc, (), (1024, 1024), (), (2, 2), 256),
        (torch.sparse_csr, (), (20, 320), (), (1, 1), 120),
        (torch.sparse_csc, (), (20, 320), (), (1, 1), 120),
        (torch.sparse_bsr, (), (20, 320), (), (2, 2), 120),
        (torch.sparse_bsc, (), (20, 320), (), (2, 2), 120),
        # One batch extent.
        (torch.sparse_csr, (4,), (16, 128), (), (1, 1), 64),
        (torch.sparse_csc, (4,), (16, 128), (), (1, 1), 64),
        (torch.sparse_bsr, (4,), (16, 128), (), (2, 2), 64),
        (torch.sparse_bsc, (4,), (16, 128), (), (2, 2), 64),
        # One dense extent.
        (torch.sparse_csr, (), (16, 128), (2,), (1, 1), 64),
        (torch.sparse_csc, (), (16, 128), (2,), (1, 1), 64),
        (torch.sparse_bsr, (), (16, 128), (2,), (2, 2), 64),
        (torch.sparse_bsc, (), (16, 128), (2,), (2, 2), 64),
        # Zero extents: empty matrix, empty batch axis, empty dense axis and an
        # unrepresentable-nnz compressed dimension (nnz must stay 0 there).
        (torch.sparse_csc, (), (0, 0), (), (1, 1), 0),
        (torch.sparse_bsr, (), (0, 0), (), (2, 2), 0),
        (torch.sparse_bsc, (), (0, 0), (), (2, 2), 0),
        (torch.sparse_csc, (), (4, 0), (), (1, 1), 0),
        (torch.sparse_csr, (0,), (3, 4), (), (1, 1), 0),
        (torch.sparse_csr, (), (3, 4), (0,), (1, 1), 0),
        (torch.sparse_bsr, (), (4, 4), (0,), (2, 2), 0),
        # Minimal single-element batch and dense axes.
        (torch.sparse_csr, (1,), (1, 3), (1,), (1, 1), 1),
        # Multiple batch axes: every batch gets its own valid index pattern.
        (torch.sparse_csr, (2, 3), (8, 10), (), (1, 1), 9),
        (torch.sparse_csc, (2, 3), (8, 10), (), (1, 1), 9),
        # Multiple dense axes.
        (torch.sparse_csr, (), (6, 6), (2, 3), (1, 1), 4),
        # Block-shape boundaries: non-square, 1-wide and blocked-odd rows.
        (torch.sparse_bsr, (), (6, 9), (), (2, 3), 4),
        (torch.sparse_bsr, (), (5, 6), (), (1, 2), 3),
        (torch.sparse_bsc, (), (1, 3), (), (1, 1), 1),
        (torch.sparse_bsc, (), (10, 6), (), (2, 3), 4),
        (torch.sparse_bsr, (), (16, 128), (), (1, 1), 64),
        # Required canonical logical sizes as batch + base + dense.
        (torch.sparse_csr, (16,), (128, 64), (60,), (1, 1), 512),
        (torch.sparse_csc, (16,), (128, 64), (60,), (1, 1), 512),
        (torch.sparse_bsr, (16,), (128, 64), (60,), (2, 2), 512),
        (torch.sparse_bsc, (16,), (128, 64), (60,), (2, 2), 512),
        (torch.sparse_csr, (16, 7), (57, 32), (29,), (1, 1), 100),
        (torch.sparse_csc, (16, 7), (57, 32), (29,), (1, 1), 100),
        # 57 == 3 * 19 and 32 == 2 ** 5, so this base also admits unequal block
        # shapes such as 1x2 and 3x1; 1x1 keeps the stored entry count small.
        (torch.sparse_bsr, (16, 7), (57, 32), (29,), (1, 1), 100),
        (torch.sparse_bsc, (16, 7), (57, 32), (29,), (1, 1), 100),
        (torch.sparse_csr, (20,), (320, 15), (), (1, 1), 120),
        (torch.sparse_csc, (20,), (320, 15), (), (1, 1), 120),
        (torch.sparse_bsr, (20,), (320, 15), (), (1, 1), 120),
        (torch.sparse_csr, (), (20, 320), (15,), (1, 1), 120),
        (torch.sparse_csc, (), (20, 320), (15,), (1, 1), 120),
        (torch.sparse_bsr, (), (20, 320), (15,), (2, 2), 60),
        (torch.sparse_bsc, (), (20, 320), (15,), (2, 2), 60),
        (torch.sparse_bsr, (4,), (16, 32), (64,), (2, 2), 32),
        # Tiny nnz with very large plain/compressed coordinates.
        (torch.sparse_csr, (), (4, 2**30), (), (1, 1), 1),
        (torch.sparse_csc, (), (2**30, 4), (), (1, 1), 1),
        (torch.sparse_bsr, (), (4, 2**30), (), (2, 2), 1),
    ],
    quick=[
        _CSR_QUICK,
        _CSC_QUICK,
        _BSR_QUICK,
        _BSC_QUICK,
        _EMPTY_QUICK,
        _ZERO_QUICK,
    ],
)


# Zero-size blocks are excluded on purpose: the native op asserts blocksize > 0
# with TORCH_INTERNAL_ASSERT, which aborts the process instead of raising.
def _check_description(layout, batch, base, dense, blocks, nnz):
    for extent in (*batch, *base, *dense, nnz):
        assert extent >= 0, extent
    assert blocks[0] >= 1 and blocks[1] >= 1, blocks
    assert base[0] % blocks[0] == 0 and base[1] % blocks[1] == 0, (base, blocks)
    compressed_dim, plain_dim = _sparse_extents(layout, base, blocks)
    if compressed_dim == 0:
        # A zero compressed extent stores nothing; nnz is never divided by it.
        assert nnz == 0, nnz
    else:
        quotient, remainder = divmod(nnz, compressed_dim)
        assert quotient + (1 if remainder else 0) <= plain_dim, (
            nnz,
            compressed_dim,
            plain_dim,
        )


def _sparse_extents(layout, base, blocks):
    block_rows, block_cols = blocks
    rows, cols = base
    if layout in _ROW_MAJOR_LAYOUTS:
        return rows // block_rows, cols // block_cols
    return cols // block_cols, rows // block_rows


def _num_batches(batch):
    total = 1
    for extent in batch:
        total *= extent
    return total


def _values_shape(layout, batch, nnz, blocks, dense):
    block_shape = blocks if layout in _BLOCKED_LAYOUTS else ()
    return (*batch, nnz, *block_shape, *dense)


def _build_indices(layout, batch, base, dense, blocks, nnz, index_dtype):
    """Build (compressed_indices, plain_indices, size) for one description.

    The counts are spread over the compressed dimension and each batch gets its own
    rotation, so a batched description holds independent valid index patterns. Every
    row's plain indices are a sorted run of in-range positions, which is what the
    native op verifies with its device-side assertion before it checks anything
    else.
    """
    _check_description(layout, batch, base, dense, blocks, nnz)
    compressed_dim, plain_dim = _sparse_extents(layout, base, blocks)
    quotient, remainder = (0, 0)
    if compressed_dim:
        quotient, remainder = divmod(nnz, compressed_dim)

    compressed_flat = []
    plain_flat = []
    for batch_index in range(_num_batches(batch)):
        compressed_row = [0]
        for row in range(compressed_dim):
            count = quotient + (
                1 if ((row + batch_index) % compressed_dim) < remainder else 0
            )
            compressed_row.append(compressed_row[-1] + count)
            if count:
                start = ((batch_index + 1) * 7 + row * 3) % (plain_dim - count + 1)
                plain_flat.extend(range(start, start + count))
        compressed_flat.extend(compressed_row)

    compressed = torch.tensor(
        compressed_flat, dtype=index_dtype, device=flag_gems.device
    ).reshape(*batch, compressed_dim + 1)
    plain = torch.tensor(
        plain_flat, dtype=index_dtype, device=flag_gems.device
    ).reshape(*batch, nnz)
    size = [*batch, base[0], base[1], *dense]
    return compressed, plain, size


def _description(
    layout, batch, base, dense, blocks, nnz, index_dtype, values_dtype, value_range
):
    compressed, plain, size = _build_indices(
        layout, batch, base, dense, blocks, nnz, index_dtype
    )
    values = tu.make_input(
        values_dtype, _values_shape(layout, batch, nnz, blocks, dense), value_range
    )
    return compressed, plain, values, size


@pytest.mark.validate_sparse_compressed_tensor_args
@pytest.mark.parametrize("layout,batch,base,dense,blocks,nnz", _DESCRIPTORS)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("values_dtype", _VALUES_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_validate_sparse_compressed_tensor_args(
    layout, batch, base, dense, blocks, nnz, index_dtype, values_dtype, value_range
):
    compressed, plain, values, size = _description(
        layout, batch, base, dense, blocks, nnz, index_dtype, values_dtype, value_range
    )
    # Independent oracle in separate storage; it doubles as the pre-call snapshot
    # used to check that the candidate only reads its inputs.
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_compressed_tensor_args(
        ref_compressed, ref_plain, ref_values, list(size), layout
    )
    res_out = flag_gems._validate_sparse_compressed_tensor_args(
        compressed, plain, values, size, layout
    )

    assert res_out is None
    tu.assert_result_equal(compressed, ref_compressed)
    tu.assert_result_equal(plain, ref_plain)
    tu.assert_result_equal(values, ref_values)


_SPECIAL_VALUES_CASES = tu.selected_cases(
    tu.special_value_cases(
        _supported(
            [
                torch.float32,
                torch.bfloat16,
                torch.float16,
                torch.float64,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]
        )
    ),
    quick=[],
)
# NaN/Inf can only live in the values component, and a blocked layout stores its
# block extents there too, so a plain and a blocked layout are both covered.
_SPECIAL_LAYOUTS = tu.selected_cases([torch.sparse_csr, torch.sparse_bsr], quick=[])
_SPECIAL_BLOCKS = {torch.sparse_csr: (1, 1), torch.sparse_bsr: (2, 2)}


def _special_values(values_dtype, scenario, layout):
    """Special-value tensor shaped like the layout's values component.

    A blocked layout stores the block extents inside `values`, so its rank must
    exceed batch + block extents; the flat generator output is therefore padded with
    its own leading elements up to a whole number of blocks before reshaping. The
    special values it carries are untouched, and padding never removes an element.
    """
    blocks = _SPECIAL_BLOCKS[layout]
    block_elems = blocks[0] * blocks[1]
    flat = tu.make_special_input(values_dtype, scenario).reshape(-1)
    nnz = -(-flat.numel() // block_elems)
    stored = nnz * block_elems
    if stored != flat.numel():
        flat = torch.cat([flat, flat[: stored - flat.numel()]])
    block_shape = blocks if layout in _BLOCKED_LAYOUTS else ()
    values = flat.reshape(nnz, *block_shape).to(flag_gems.device)
    return values, nnz, blocks


def _capacity_base(layout, nnz, blocks):
    """Smallest base extents that can hold ``nnz`` entries for the given layout."""
    compressed_dim = min(4, max(1, nnz))
    plain_dim = max(1, -(-nnz // compressed_dim))
    rows, cols = compressed_dim * blocks[0], plain_dim * blocks[1]
    return (rows, cols) if layout in _ROW_MAJOR_LAYOUTS else (cols, rows)


@pytest.mark.validate_sparse_compressed_tensor_args
@pytest.mark.parametrize("values_dtype,scenario", _SPECIAL_VALUES_CASES)
@pytest.mark.parametrize("layout", _SPECIAL_LAYOUTS)
def test_validate_sparse_compressed_tensor_args_special_values(
    values_dtype, scenario, layout
):
    # tu.special_value_cases drops the inf and mixed scenarios for float8_e4m3fn
    # (the format has no infinity) and keeps them for float8_e5m2, so the case list
    # matches each dtype's representable set. The logical size is derived from the
    # helper's element count so the description stays valid for every scenario.
    values, nnz, blocks = _special_values(values_dtype, scenario, layout)
    base = _capacity_base(layout, nnz, blocks)
    compressed, plain, size = _build_indices(
        layout, (), base, (), blocks, nnz, torch.int32
    )
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_compressed_tensor_args(
        ref_compressed, ref_plain, ref_values, list(size), layout
    )
    res_out = flag_gems._validate_sparse_compressed_tensor_args(
        compressed, plain, values, size, layout
    )

    assert res_out is None
    tu.assert_result_equal(compressed, ref_compressed)
    tu.assert_result_equal(plain, ref_plain)
    tu.assert_result_equal(values, ref_values)


# Supplementary values-dtype family; selected default-only so that quick keeps the
# required dtype coverage of the main grid and these extra dtypes stay in the default
# suite.
_EXTRA_VALUES_DTYPES = tu.selected_cases(
    _supported([torch.float64, torch.bool, torch.complex64]), quick=[]
)
_EXTRA_DTYPE_LAYOUTS = tu.selected_cases(
    [
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    ],
    quick=[],
)


@pytest.mark.validate_sparse_compressed_tensor_args
@pytest.mark.parametrize("layout", _EXTRA_DTYPE_LAYOUTS)
@pytest.mark.parametrize("values_dtype", _EXTRA_VALUES_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_validate_sparse_compressed_tensor_args_extra_values_dtypes(
    layout, values_dtype, value_range
):
    # The values component only has to be a strided tensor of the right shape, so
    # the required nine dtypes are covered by the grid and float64/bool/complex are
    # covered here for every layout and value range.
    blocks = (2, 2) if layout in _BLOCKED_LAYOUTS else (1, 1)
    compressed, plain, values, size = _description(
        layout, (), (6, 6), (), blocks, 2, torch.int32, values_dtype, value_range
    )
    # Independent oracle in separate storage; it doubles as the pre-call snapshot
    # used to check that the candidate only reads its inputs.
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    torch.ops.aten._validate_sparse_compressed_tensor_args(
        ref_compressed, ref_plain, ref_values, list(size), layout
    )
    res_out = flag_gems._validate_sparse_compressed_tensor_args(
        compressed, plain, values, size, layout
    )

    assert res_out is None
    tu.assert_result_equal(compressed, ref_compressed)
    tu.assert_result_equal(plain, ref_plain)
    tu.assert_result_equal(values, ref_values)


# Valid descriptions whose operands are non-contiguous or carry a storage offset:
# values contiguity is not required by the op, the index tensors only require
# stride(-1) == 1 per batch, and neither aliasing nor offsetting changes the result.
_OFFSET_CASES = tu.selected_cases(
    [
        "values_stride_two",
        "values_storage_offset",
        "compressed_storage_offset",
        "plain_storage_offset",
    ],
    quick=[],
)


def _offset_args(case):
    """(parents, slices, size, layout) for one offset fixture.

    crow_indices must hold rows + 1 == 4 entries and col_indices one entry per
    stored value, so the offset variants keep the payload behind a small prefix in
    storage and slice the prefix away; the tensor handed to the op still has the
    declared lengths while carrying a non-zero storage offset. The parent tensor and
    the slice that produces its operand are both returned, so the same slice rebuilds
    the operand from an independent reference parent and the whole storage, hidden
    prefix and stride gaps included, can be compared after the call.
    """
    device = flag_gems.device
    if case == "values_stride_two":
        # stride(-1) == 1 is required for the indices only; values may be strided.
        # float32 keeps this values-only fixture independent of int64 support.
        parents = (
            torch.tensor([0, 2, 3, 5], dtype=torch.int32, device=device),
            torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32, device=device),
            torch.arange(10, dtype=torch.float32, device=device),
        )
        slices = (slice(None), slice(None), slice(None, None, 2))
    elif case == "values_storage_offset":
        parents = (
            torch.tensor([0, 2, 3, 5], dtype=torch.int32, device=device),
            torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32, device=device),
            torch.arange(1.0, 7.0, dtype=torch.float32, device=device),
        )
        slices = (slice(None), slice(None), slice(1, None))
    elif case == "compressed_storage_offset":
        parents = (
            torch.tensor([9, 9, 0, 2, 3, 5], dtype=torch.int32, device=device),
            torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32, device=device),
            torch.zeros(5, dtype=torch.float32, device=device),
        )
        slices = (slice(2, None), slice(None), slice(None))
    elif case == "plain_storage_offset":
        parents = (
            torch.tensor([0, 2, 3, 5], dtype=torch.int32, device=device),
            torch.tensor([9, 0, 1, 2, 0, 1], dtype=torch.int32, device=device),
            torch.zeros(5, dtype=torch.float32, device=device),
        )
        slices = (slice(None), slice(1, None), slice(None))
    else:
        raise ValueError(f"unknown offset case: {case!r}")
    return parents, slices, [3, 4], torch.sparse_csr


@pytest.mark.validate_sparse_compressed_tensor_args
@pytest.mark.parametrize("case", _OFFSET_CASES)
def test_validate_sparse_compressed_tensor_args_offsets_and_strides(case):
    parents, slices, size, layout = _offset_args(case)
    compressed, plain, values = (parent[item] for parent, item in zip(parents, slices))
    # Slicing independent reference parents with the same slices reproduces the exact
    # strides and storage offsets on the reference side, so a plain clone that compacted
    # them would not hide a candidate that consumed a flattened operand.
    ref_parents = tuple(tu.to_reference(parent) for parent in parents)
    ref_compressed, ref_plain, ref_values = (
        parent[item] for parent, item in zip(ref_parents, slices)
    )

    torch.ops.aten._validate_sparse_compressed_tensor_args(
        ref_compressed, ref_plain, ref_values, list(size), layout
    )
    res_out = flag_gems._validate_sparse_compressed_tensor_args(
        compressed, plain, values, size, layout
    )

    assert res_out is None
    # The stride/offset geometry of the operands, then the full storage of every
    # parent: comparing the parents covers the operands themselves and also catches a
    # candidate that rewrote only the hidden index prefix or the gap between two
    # strided values.
    assert [tensor.stride() for tensor in (compressed, plain, values)] == [
        tensor.stride() for tensor in (ref_compressed, ref_plain, ref_values)
    ]
    assert [tensor.storage_offset() for tensor in (compressed, plain, values)] == [
        tensor.storage_offset() for tensor in (ref_compressed, ref_plain, ref_values)
    ]
    # The operands' own values, then the whole storage of every parent: the operand
    # checks catch a candidate that re-pointed an operand's storage with set_ (which
    # leaves the parent untouched and the geometry identical), and the parent checks
    # also catch a rewritten hidden index prefix or values stride gap.
    tu.assert_result_equal(compressed, ref_compressed)
    tu.assert_result_equal(plain, ref_plain)
    tu.assert_result_equal(values, ref_values)
    for parent, ref_parent in zip(parents, ref_parents):
        tu.assert_result_equal(parent, ref_parent)


# Structural violations, one argument family each. Every entry raises a clean
# host-side RuntimeError (or the pybind TypeError for the string layout) natively;
# index-content violations are absent, see the module comment.
_NEGATIVE_CASES = [
    "compressed_indices_ndim_low",
    "compressed_indices_ndim_high",
    "compressed_indices_length",
    "compressed_indices_stride",
    "plain_indices_length",
    "plain_indices_ndim",
    "plain_indices_stride",
    "values_ndim_low",
    "values_ndim_high",
    "size_rank_zero",
    "size_rank_low",
    "size_rank_high",
    "size_negative",
    "block_size_indivisible",
    "batch_size_missing",
    "values_batch_mismatch",
    "index_dtype_float",
    "index_dtype_int8",
    "index_dtype_bool",
    "layout_strided",
    "layout_sparse_coo",
    "layout_string",
]
# The two remaining cases need a capability the active backend may not have, so
# they are selected from static capability flags rather than probed or skipped at
# run time.
if torch.device(flag_gems.device).type != "cpu":
    # The values component must live on the same device as the indices.
    _NEGATIVE_CASES.append("values_device_mismatch")
if au.int64_is_supported:
    # The mismatch widens one index tensor to int64, so the case needs int64.
    _NEGATIVE_CASES.append("index_dtype_mismatch")


def _csr_components(index_dtype=torch.int32):
    """(compressed_indices, plain_indices, values) of the 3x4 CSR example."""
    device = flag_gems.device
    compressed = torch.tensor([0, 2, 3, 5], dtype=index_dtype, device=device)
    plain = torch.tensor([0, 1, 2, 0, 1], dtype=index_dtype, device=device)
    return compressed, plain, torch.zeros(5, device=device)


def _invalid_args(case):
    """Malformed arguments for one structural negative case."""
    device = flag_gems.device
    compressed, plain, values = _csr_components()
    if case == "compressed_indices_ndim_low":
        return compressed[0], plain, values, [3, 4], torch.sparse_csr
    if case == "compressed_indices_ndim_high":
        return compressed.unsqueeze(0), plain, values, [3, 4], torch.sparse_csr
    if case == "compressed_indices_length":
        return compressed[:3], plain, values, [3, 4], torch.sparse_csr
    if case == "compressed_indices_stride":
        # stride(-1) != 1: the op requires a contiguous tensor per batch.
        return (
            compressed.repeat_interleave(2)[::2],
            plain,
            values,
            [3, 4],
            torch.sparse_csr,
        )
    if case == "plain_indices_length":
        return compressed, plain[:4], values, [3, 4], torch.sparse_csr
    if case == "plain_indices_ndim":
        return compressed, plain.unsqueeze(0), values, [3, 4], torch.sparse_csr
    if case == "plain_indices_stride":
        return (
            compressed,
            plain.repeat_interleave(2)[::2],
            values,
            [3, 4],
            torch.sparse_csr,
        )
    if case == "values_ndim_low":
        return (
            compressed,
            plain,
            torch.zeros((), device=device),
            [3, 4],
            torch.sparse_csr,
        )
    if case == "values_ndim_high":
        return (
            compressed,
            plain,
            torch.zeros(5, 2, device=device),
            [3, 4],
            torch.sparse_csr,
        )
    if case == "size_rank_zero":
        return compressed, plain, values, [], torch.sparse_csr
    if case == "size_rank_low":
        return compressed, plain, values, [3], torch.sparse_csr
    if case == "size_rank_high":
        return compressed, plain, values, [3, 4, 5], torch.sparse_csr
    if case == "size_negative":
        return compressed, plain, values, [-3, 4], torch.sparse_csr
    if case == "block_size_indivisible":
        # 5x6 with 2x2 blocks: crow_indices holds 5 // 2 + 1 == 3 offsets, col_indices
        # holds one entry per stored value, and the two rows store the in-range sorted
        # pairs [0, 1] and [1, 2], so the only violated precondition is the row extent
        # 5 not being divisible by the block extent 2.
        block_compressed = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
        block_plain = torch.tensor([0, 1, 1, 2], dtype=torch.int32, device=device)
        block_values = torch.zeros(4, 2, 2, device=device)
        return block_compressed, block_plain, block_values, [5, 6], torch.sparse_bsr
    if case == "batch_size_missing":
        # Batched index tensors but a non-batched `size`.
        return (
            compressed.unsqueeze(0).expand(2, -1).contiguous(),
            plain.unsqueeze(0).expand(2, -1).contiguous(),
            values.unsqueeze(0).expand(2, -1).contiguous(),
            [3, 4],
            torch.sparse_csr,
        )
    if case == "values_batch_mismatch":
        # `size` carries a batch extent of 2 while values has 3.
        return (
            compressed.unsqueeze(0).expand(2, -1).contiguous(),
            plain.unsqueeze(0).expand(2, -1).contiguous(),
            torch.zeros(3, 5, device=device),
            [2, 3, 4],
            torch.sparse_csr,
        )
    if case == "index_dtype_float":
        return compressed.float(), plain.float(), values, [3, 4], torch.sparse_csr
    if case == "index_dtype_int8":
        return (
            compressed.to(torch.int8),
            plain.to(torch.int8),
            values,
            [3, 4],
            torch.sparse_csr,
        )
    if case == "index_dtype_bool":
        return compressed.bool(), plain.bool(), values, [3, 4], torch.sparse_csr
    if case == "index_dtype_mismatch":
        return compressed, plain.to(torch.int64), values, [3, 4], torch.sparse_csr
    if case == "layout_strided":
        return compressed, plain, values, [3, 4], torch.strided
    if case == "layout_sparse_coo":
        return compressed, plain, values, [3, 4], torch.sparse_coo
    if case == "layout_string":
        return compressed, plain, values, [3, 4], "csr"
    if case == "values_device_mismatch":
        # The values component must live on the same device as the indices.
        return compressed, plain, torch.zeros(5), [3, 4], torch.sparse_csr
    raise ValueError(f"unknown invalid case: {case!r}")


@pytest.mark.validate_sparse_compressed_tensor_args
@pytest.mark.parametrize("case", _NEGATIVE_CASES)
def test_validate_sparse_compressed_tensor_args_rejects_invalid(case):
    compressed, plain, values, size, layout = _invalid_args(case)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._validate_sparse_compressed_tensor_args(
            compressed, plain, values, size, layout
        )
