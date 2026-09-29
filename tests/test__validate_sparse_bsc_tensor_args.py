# Copyright 2024, The FlagGems Authors.
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

# `_validate_sparse_bsc_tensor_args` inspects BSC metadata and the encoded index
# contents, then returns `None`. `.default` is its only overload: there is no
# tensor output, no `.out` form, no broadcasting and no autograd, so those spec
# dimensions are inapplicable. The coverage below uses the accepted descriptor
# space (logical rank, block shape, dense tail, index dtype, values dtype, value
# range), the input-preservation contract and the rejected-input space instead.
#
# Exempt shape levels: `size` must hold `batch + 2 base + dense` entries, so the
# 0-dim and 1-dim entries of the shared shape grid cannot describe a BSC tensor.
# They are dropped rather than replaced by an unrelated shape; the four remaining
# default ranks ((1024,1024), (20,320,15), (16,128,64,60), (16,7,57,32,29)) and
# the quick rank (2,19,7) are all exercised.
_BSC_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]


def _values_dtype_available(dtype):
    """Static capability gate for a `values` payload dtype (no probe allocation).

    The payload dtype is never inspected by the validator, so its accepted space
    is the widest the backend supports; widening it must not depend on a probe
    that allocates on the accelerator.
    """
    if dtype == torch.bfloat16:
        return utils.bf16_is_supported
    if dtype == torch.float64:
        return utils.fp64_is_supported
    if dtype == torch.int64:
        return utils.int64_is_supported
    if dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ):
        return utils.fp8_is_supported
    return True


# The nine required dtypes plus the further payload dtypes the native worker
# accepts: bool, int16, float64 and complex64 payloads are native-valid and left
# unchanged, because the worker performs no values-dtype check.
_EXTRA_VALUE_DTYPES = [torch.bool, torch.int16, torch.float64, torch.complex64]
_VALUE_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES + _EXTRA_VALUE_DTYPES
    if _values_dtype_available(dtype)
]

# `ccol_indices` and `row_indices` must share one int32/int64 dtype. Which widths
# exist here is a static backend capability, so every index list is selected at
# collection time and never probed or skipped at run time.
_INDEX_DTYPES = (
    [torch.int64, torch.int32] if utils.int64_is_supported else [torch.int32]
)
# Indices are assembled in int64 with torch.arange (host metadata staging, then a
# cast to the descriptor dtype), which is independent of accelerator int64
# support, so the widest available width is the natural main-grid choice.
_MAIN_INDEX_DTYPE = _INDEX_DTYPES[0]


def _bsc_layout(batch, nrows, ncols, dense, blocksize, fill):
    """Build accepted BSC index metadata plus the matching `values` shape and `size`.

    Block column j receives `fill` nonzeros placed on its first `fill` block
    rows, so `row_indices` stays sorted and duplicate-free inside every block
    column and the content invariants hold. Indices are assembled in int64 and
    cast by the caller.
    """
    rb, cb = blocksize
    counts = torch.full((ncols // cb,), fill, dtype=torch.int64)
    offsets = torch.cumsum(counts, 0)
    nnz = int(counts.sum())
    ccol = torch.cat((torch.zeros(1, dtype=torch.int64), offsets))
    row = torch.arange(nnz) - (offsets - counts).repeat_interleave(counts)
    if batch:
        ccol = ccol.expand(tuple(batch) + (-1,)).contiguous()
        row = row.expand(tuple(batch) + (-1,)).contiguous()
    values_shape = tuple(batch) + (nnz, rb, cb) + tuple(dense)
    size = list(batch) + [nrows, ncols] + list(dense)
    return ccol, row, values_shape, size


def _bsc_inputs(batch, nrows, ncols, dense, blocksize, fill, dtype, index_dtype=None):
    """Deterministic accepted descriptor, shared by the layout and negative tests."""
    index_dtype = _MAIN_INDEX_DTYPE if index_dtype is None else index_dtype
    ccol, row, values_shape, size = _bsc_layout(
        batch, nrows, ncols, dense, blocksize, fill
    )
    values = (
        torch.arange(torch.Size(values_shape).numel(), dtype=torch.float32)
        .reshape(values_shape)
        .to(dtype)
    )
    return (
        ccol.to(index_dtype).to(flag_gems.device),
        row.to(index_dtype).to(flag_gems.device),
        values.to(flag_gems.device),
        size,
    )


# --- accepted descriptor space: logical shape x value range x values dtype ----


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("shape", _BSC_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__validate_sparse_bsc_tensor_args_values(shape, value_range, dtype):
    # 1x1 blocks keep the main-grid descriptor shape (nnz == number of columns);
    # the structural rows below supply the nontrivial and dense block shapes.
    batch = tuple(shape[:-2])
    ccol, row, values_shape, size = _bsc_layout(
        batch, shape[-2], shape[-1], (), (1, 1), 1
    )
    ccol = ccol.to(_MAIN_INDEX_DTYPE).to(flag_gems.device)
    row = row.to(_MAIN_INDEX_DTYPE).to(flag_gems.device)
    values = tu.make_input(dtype, values_shape, value_range)
    ccol_ref = tu.to_reference(ccol)
    row_ref = tu.to_reference(row)
    values_ref = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsc_tensor_args(ccol_ref, row_ref, values_ref, size)
    result = flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)

    assert result is None
    # The operator only inspects descriptors: it must leave the index tensors and
    # the values payload exactly as it received them.
    tu.assert_result_equal(ccol, ccol_ref)
    tu.assert_result_equal(row, row_ref)
    tu.assert_result_equal(values, values_ref)


# --- structural descriptors --------------------------------------------------
# (id, batch dims, logical rows, logical cols, block shape, nonzeros per block
#  column, dense tail), every row native-accepted. Supplementary positives are
#  default-only (empty quick subset): the quick level keeps the main grid and the
#  negative boundaries, while the full default set keeps every block, batch,
#  dense and index-rank combination below.
_LAYOUT_ROWS = [
    ("1x1-fill1", (), 4, 4, (1, 1), 1, ()),
    ("1x1-matrix", (), 1, 1, (1, 1), 1, ()),
    ("2x2-fill1", (), 6, 4, (2, 2), 1, ()),
    ("4x4-fill2", (), 8, 8, (4, 4), 2, ()),
    ("4x3-fill2", (), 8, 6, (4, 3), 2, ()),
    ("4x4-fill1", (), 16, 16, (4, 4), 1, ()),
    ("nontrivial-2x4", (), 6, 12, (2, 4), 2, ()),
    ("3x5-fill2", (), 9, 15, (3, 5), 2, ()),
    ("singleton-block", (), 2, 2, (2, 2), 1, ()),
    ("block-1x1", (), 8, 12, (1, 1), 2, ()),
    ("fill0-zero-nnz", (), 4, 4, (1, 1), 0, ()),
    ("fill4-full-columns", (), 4, 4, (1, 1), 4, ()),
    ("zero-rows", (), 0, 8, (2, 2), 0, ()),
    ("zero-cols", (), 8, 0, (2, 2), 0, ()),
    ("dense-1", (), 4, 4, (1, 1), 1, (3,)),
    ("dense-2", (), 4, 4, (2, 2), 1, (2, 3)),
    ("dense-2x2-fill2", (), 8, 8, (2, 2), 2, (5,)),
    ("dense-zero-extent", (), 4, 4, (1, 1), 1, (0,)),
    ("batch-2", (2,), 4, 4, (1, 1), 1, ()),
    ("batch-2x3", (2, 3), 8, 6, (4, 3), 1, ()),
    ("batch-2-4x4", (2,), 8, 8, (4, 4), 1, ()),
    ("batch-zero", (0,), 4, 4, (1, 1), 1, ()),
    ("batch-2x2-fill2", (2, 2), 8, 8, (2, 2), 2, ()),
    ("batch-dense", (2,), 8, 8, (4, 4), 1, (2,)),
    ("index-rank-8", (2,) * 7, 4, 4, (1, 1), 1, ()),
    ("index-rank-9", (2,) * 8, 4, 4, (1, 1), 1, ()),
]


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize(
    "layout",
    tu.selected_cases(_LAYOUT_ROWS, quick=[]),
    ids=lambda layout: layout[0],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test__validate_sparse_bsc_tensor_args_layout(layout, dtype, index_dtype):
    _, batch, nrows, ncols, blocksize, fill, dense = layout
    ccol, row, values, size = _bsc_inputs(
        batch, nrows, ncols, dense, blocksize, fill, dtype, index_dtype
    )
    ccol_ref = tu.to_reference(ccol)
    row_ref = tu.to_reference(row)
    values_ref = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsc_tensor_args(ccol_ref, row_ref, values_ref, size)
    result = flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)

    assert result is None
    tu.assert_result_equal(ccol, ccol_ref)
    tu.assert_result_equal(row, row_ref)
    tu.assert_result_equal(values, values_ref)


# Every batch element has the same nnz here but distributes it over the block
# columns differently; a validator that only checks one batch pattern would pass
# the uniform layouts above and still accept this descriptor.
@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize(
    "dtype", tu.selected_cases([torch.float32, torch.float16], quick=[])
)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test__validate_sparse_bsc_tensor_args_batch_distributions(dtype, index_dtype):
    counts = torch.tensor([[1, 0, 1, 1], [0, 3, 0, 0]], dtype=torch.int64)
    ccol = torch.cat((torch.zeros(2, 1, dtype=torch.int64), counts.cumsum(1)), 1)
    row = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1).to(dtype)
    ccol = ccol.to(index_dtype).to(flag_gems.device)
    row = row.to(index_dtype).to(flag_gems.device)
    values = values.to(flag_gems.device)
    size = [2, 4, 4]
    ccol_ref = tu.to_reference(ccol)
    row_ref = tu.to_reference(row)
    values_ref = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsc_tensor_args(ccol_ref, row_ref, values_ref, size)
    result = flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)

    assert result is None
    tu.assert_result_equal(ccol, ccol_ref)
    tu.assert_result_equal(row, row_ref)
    tu.assert_result_equal(values, values_ref)


# Accepted descriptors whose tensors are not canonical: index tensors may be
# stride-1 views at a nonzero storage offset (the native check is
# `stride(-1) == 1` per batch, not `is_contiguous()`), and the `values` block
# extent may be a non-contiguous slice, because payload elements are never
# indexed. The real parent buffers stay resident so the accepted view and the
# untouched padding are both asserted; the reference views are sliced from
# independently transferred parents, so the native oracle runs on its own device
# with the same offset/stride geometry.
_OFFSET_CASES = tu.selected_cases(
    ["offset-indices", "offsets-and-strided-values"], quick=[]
)


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("case", _OFFSET_CASES)
@pytest.mark.parametrize(
    "dtype", tu.selected_cases([torch.float32, torch.float16], quick=[])
)
def test__validate_sparse_bsc_tensor_args_offset_and_strided_inputs(case, dtype):
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, dtype)
    pad = 3
    ccol_parent = torch.zeros(
        ccol.numel() + pad, dtype=ccol.dtype, device=flag_gems.device
    )
    ccol_parent[pad:] = ccol
    row_parent = torch.zeros(
        row.numel() + pad, dtype=row.dtype, device=flag_gems.device
    )
    row_parent[pad:] = row
    ccol_arg, row_arg = ccol_parent[pad:], row_parent[pad:]
    if case == "offsets-and-strided-values":
        values_parent = torch.zeros(
            values.shape[0],
            2 * values.shape[1],
            values.shape[2],
            dtype=values.dtype,
            device=flag_gems.device,
        )
        values_parent[:, : values.shape[1], :] = values
        values_cols = values.shape[1]
        values_arg = values_parent[:, :values_cols, :]
    else:
        values_parent = values
        values_cols = None
        values_arg = values
    ccol_parent_ref = tu.to_reference(ccol_parent)
    row_parent_ref = tu.to_reference(row_parent)
    values_parent_ref = tu.to_reference(values_parent)
    ccol_ref, row_ref = ccol_parent_ref[pad:], row_parent_ref[pad:]
    values_ref = (
        values_parent_ref[:, :values_cols, :]
        if values_cols is not None
        else values_parent_ref
    )

    torch.ops.aten._validate_sparse_bsc_tensor_args(ccol_ref, row_ref, values_ref, size)
    result = flag_gems._validate_sparse_bsc_tensor_args(
        ccol_arg, row_arg, values_arg, size
    )

    assert result is None
    tu.assert_result_equal(ccol_parent, ccol_parent_ref)
    tu.assert_result_equal(row_parent, row_parent_ref)
    tu.assert_result_equal(values_parent, values_parent_ref)


# The native row bound is the full logical extent and the block-row index is
# checked against the index dtype range rather than against the block count, so
# a tiny-nnz descriptor whose single block row sits above 2**31 is accepted
# without allocating a dense matrix. int32 indices cannot express it (native
# probe: 32-bit integer overflow in row dimension), so this family is selected
# only where int64 indices exist.
_HUGE_ROW_CASES = tu.selected_cases(
    [torch.float32, torch.float16] if utils.int64_is_supported else [], quick=[]
)


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("dtype", _HUGE_ROW_CASES)
def test__validate_sparse_bsc_tensor_args_int64_row_range(dtype):
    size = [2**32 + 4, 1]
    ccol = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([2**32 + 3], dtype=torch.int64, device=flag_gems.device)
    values = torch.zeros(1, 1, 1, dtype=dtype, device=flag_gems.device)
    ccol_ref = tu.to_reference(ccol)
    row_ref = tu.to_reference(row)
    values_ref = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsc_tensor_args(ccol_ref, row_ref, values_ref, size)
    result = flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)

    assert result is None
    tu.assert_result_equal(ccol, ccol_ref)
    tu.assert_result_equal(row, row_ref)
    tu.assert_result_equal(values, values_ref)


# The native worker clamps a zero block extent to 1 (`std::max<int64_t>(1,
# values.size(...))` in aten/src/ATen/native/sparse/SparseCsrTensor.cpp), so this
# payload is read as 1x1 blocks and accepted; a zero block extent cannot be used
# as a negative case. Supplementary positive, so it is default-only.
_ZERO_BLOCK_CASES = tu.selected_cases([torch.float32], quick=[])


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("dtype", _ZERO_BLOCK_CASES)
def test__validate_sparse_bsc_tensor_args_zero_block_extents(dtype):
    ccol = torch.tensor(
        [0, 1, 1, 1, 1, 1, 1, 1, 1],
        dtype=_MAIN_INDEX_DTYPE,
        device=flag_gems.device,
    )
    row = torch.tensor([0], dtype=_MAIN_INDEX_DTYPE, device=flag_gems.device)
    values = torch.zeros(1, 0, 0, dtype=dtype, device=flag_gems.device)
    size = [8, 8]
    ccol_ref = tu.to_reference(ccol)
    row_ref = tu.to_reference(row)
    values_ref = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsc_tensor_args(ccol_ref, row_ref, values_ref, size)
    result = flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)

    assert result is None
    tu.assert_result_equal(ccol, ccol_ref)
    tu.assert_result_equal(row, row_ref)
    tu.assert_result_equal(values, values_ref)


# --- accepted `size` call forms ---------------------------------------------


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize(
    "form", tu.selected_cases(["list", "tuple", "torch.Size", "keyword"], quick=[])
)
def test__validate_sparse_bsc_tensor_args_size_forms(form):
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    if form == "tuple":
        size = tuple(size)
    elif form == "torch.Size":
        size = torch.Size(size)
    ccol_ref = tu.to_reference(ccol)
    row_ref = tu.to_reference(row)
    values_ref = tu.to_reference(values)

    if form == "keyword":
        torch.ops.aten._validate_sparse_bsc_tensor_args(
            ccol_ref, row_ref, values_ref, size=size
        )
        result = flag_gems._validate_sparse_bsc_tensor_args(
            ccol, row, values, size=size
        )
    else:
        torch.ops.aten._validate_sparse_bsc_tensor_args(
            ccol_ref, row_ref, values_ref, size
        )
        result = flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)

    assert result is None
    tu.assert_result_equal(ccol, ccol_ref)
    tu.assert_result_equal(row, row_ref)
    tu.assert_result_equal(values, values_ref)


# --- special values ----------------------------------------------------------
# The payload is never interpreted, so NaN/Inf must be accepted and returned
# bit-identically; only the descriptor metadata decides.

_FLOAT_VALUE_DTYPES = [dtype for dtype in _VALUE_DTYPES if dtype.is_floating_point]
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_FLOAT_VALUE_DTYPES), quick=[]
)


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__validate_sparse_bsc_tensor_args_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    nnz = int(payload.numel())
    ccol, row, _, size = _bsc_layout((), 4, nnz, (), (1, 1), 1)
    ccol = ccol.to(_MAIN_INDEX_DTYPE).to(flag_gems.device)
    row = row.to(_MAIN_INDEX_DTYPE).to(flag_gems.device)
    values = payload.reshape(nnz, 1, 1)
    ccol_ref = tu.to_reference(ccol)
    row_ref = tu.to_reference(row)
    values_ref = tu.to_reference(values)

    torch.ops.aten._validate_sparse_bsc_tensor_args(ccol_ref, row_ref, values_ref, size)
    result = flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)

    assert result is None
    tu.assert_result_equal(ccol, ccol_ref)
    tu.assert_result_equal(row, row_ref)
    tu.assert_result_equal(values, values_ref)


# --- rejected input space ----------------------------------------------------
# Descriptor violations (rank, index dtype, extents, lengths, stride, batch
# mismatch, dense rank and block divisibility) are host checks in
# aten/src/ATen/native/sparse/SparseCsrTensor.cpp and raise a clean
# RuntimeError; every case below was confirmed by a native probe on the
# reference device.
#
# UNRESOLVED COVERAGE: the content invariants of
# aten/src/ATen/native/sparse/ValidateCompressedIndicesCommon.h (a row index
# outside its block column's range, an unsorted or duplicated row index,
# `ccol_indices[0] != 0`, `ccol_indices[-1] != nnz`, a decreasing
# `ccol_indices`) are enforced through an assertion that aborts the process on
# this backend, so they cannot be expressed as `pytest.raises` and stay
# unresolved until safe native or protocol support exists. They are not re-run
# and are not emulated with a CPU oracle.


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("broken_shape", [(16,), (2, 8), (2, 2, 2, 2)])
def test__validate_sparse_bsc_tensor_args_rejects_values_rank(broken_shape):
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(
            ccol, row, values.reshape(broken_shape), size
        )


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("bad_size", [[8], [8, 8, 1]])
def test__validate_sparse_bsc_tensor_args_rejects_size_rank(bad_size):
    ccol, row, values, _ = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, bad_size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_dense_rank_mismatch():
    # A dense tail makes `size` one entry longer than a dense-free descriptor:
    # the worker derives the dense rank from `values` and requires
    # size.size() == batch + 2 + dense, so a shorter size is rejected.
    ccol, row, values, _ = _bsc_inputs((), 4, 4, (3,), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, [4, 4])


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize(
    "nrows,ncols,blocksize", [(4, 5, (2, 2)), (5, 4, (2, 2)), (8, 6, (3, 3))]
)
def test__validate_sparse_bsc_tensor_args_rejects_indivisible_blocksize(
    nrows, ncols, blocksize
):
    ccol, row, values, size = _bsc_inputs(
        (), nrows, ncols, (), blocksize, 1, torch.float32
    )
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_short_ccol():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol[:-1], row, values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_long_ccol():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(
            torch.cat((ccol, ccol[-1:])), row, values, size
        )


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_short_row():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row[:-1], values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_long_row():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(
            ccol, torch.cat((row, row[:1])), values, size
        )


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_values_nnz_mismatch():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    broken = torch.ones(
        values.shape[0] + 1, 2, 2, dtype=values.dtype, device=flag_gems.device
    )
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, broken, size)


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize("bad_dtype", [torch.int8, torch.int16, torch.float32])
def test__validate_sparse_bsc_tensor_args_rejects_index_dtype(bad_dtype):
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(
            ccol.to(bad_dtype), row.to(bad_dtype), values, size
        )


# An int32 `ccol_indices` against the main int64 `row_indices` is only an invalid
# pair when both widths exist, so the case is selected on that static capability
# instead of being skipped at run time; where int64 is unavailable the list is
# empty and no case is collected.
_MIXED_INDEX_CASES = [torch.int32] if utils.int64_is_supported else []


@pytest.mark.validate_sparse_bsc_tensor_args
@pytest.mark.parametrize(
    "bad_ccol_dtype", tu.selected_cases(_MIXED_INDEX_CASES, quick=_MIXED_INDEX_CASES)
)
def test__validate_sparse_bsc_tensor_args_rejects_mixed_index_dtype(bad_ccol_dtype):
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(
            ccol.to(bad_ccol_dtype), row, values, size
        )


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_noncontiguous_ccol():
    # The strided view keeps the original logical contents; only the layout is
    # invalid (`stride(-1) == 1` is required per batch).
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    buffer = torch.zeros(2 * ccol.numel(), dtype=ccol.dtype, device=flag_gems.device)
    buffer[::2] = ccol
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(buffer[::2], row, values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_noncontiguous_row():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    buffer = torch.zeros(2 * row.numel(), dtype=row.dtype, device=flag_gems.device)
    buffer[::2] = row
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, buffer[::2], values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_batch_size_mismatch():
    ccol, row, values, _ = _bsc_inputs((2,), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, [3, 8, 8])


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_index_rank_mismatch():
    ccol, row, values, size = _bsc_inputs((2,), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row[0], values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_rank_zero_indices():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol[0], row, values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_negative_extent():
    # A negative logical extent is rejected by the derived column-block count
    # (native probe: ccol_indices.shape[-1] must be equal to the number of
    # column blocks + 1 (=-3)).
    ccol, row, values, _ = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, [8, -8])


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_int32_index_overflow():
    ccol, row, values, size = _bsc_inputs(
        (), 2**31 + 4, 1, (), (1, 1), 1, torch.float32, torch.int32
    )
    with pytest.raises(RuntimeError):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values, size)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_missing_size():
    ccol, row, values, _ = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._validate_sparse_bsc_tensor_args(ccol, row, values)


@pytest.mark.validate_sparse_bsc_tensor_args
def test__validate_sparse_bsc_tensor_args_rejects_non_tensor_inputs():
    ccol, row, values, size = _bsc_inputs((), 8, 8, (), (2, 2), 1, torch.float32)
    lists = (ccol.tolist(), row.tolist(), values.tolist())
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._validate_sparse_bsc_tensor_args(*lists, size)
