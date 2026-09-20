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

# Build compressed sparse storage from index and value tensors.
# Pass dtype explicitly: these ATen factories default to float32.
# Accelerator inputs also need an explicit device.
_BSC_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)
_INDEX_DTYPES = [torch.int32, torch.int64]

# (matrix_shape, block_shape, nnz).
_BSC_CASES = [
    ((4, 4), (2, 2), 3),  # 2x2 row/col blocks, partial fill
    ((8, 8), (2, 2), 16),  # full 4x4 block grid
    ((16, 16), (4, 4), 8),  # larger blocks
    ((6, 8), (2, 2), 6),  # non-square matrix
    ((6, 6), (3, 3), 4),  # 3x3 blocks
    ((2, 6), (2, 3), 2),  # single row block
    ((4, 4), (2, 2), 0),  # empty (nnz == 0)
]

_BSC_VALUE_CASES = [
    ((4, 4), (2, 2), 3),
    ((6, 8), (2, 2), 6),
]

# Legacy 1-D values cannot be densified; compare the stored structure.
_LEGACY_CASES = [
    ((4, 5), 3),
    ((4, 5), 0),
]


def _bsc_shape_level_cases():
    # Use trailing matrix dimensions and blocks that divide each extent.
    cases = []
    for shape in tu.selected_shapes():
        shape = tuple(shape)
        if len(shape) < 2:
            continue
        nrows, ncols = shape[-2], shape[-1]
        block = (2 if nrows % 2 == 0 else 1, 2 if ncols % 2 == 0 else 1)
        n_blocks = (nrows // block[0]) * (ncols // block[1])
        nnz = min(6, max(1, n_blocks))
        cases.append(((nrows, ncols), block, nnz))
    if not cases:
        cases = [((4, 4), (2, 2), 4)]
    return cases


def _make_bsc_structure(shape, block, nnz, index_dtype=torch.int64):
    M, N = shape
    Br, Bc = block
    n_row_blocks, n_col_blocks = M // Br, N // Bc
    assert M % Br == N % Bc == 0
    assert 0 <= nnz <= n_row_blocks * n_col_blocks
    counts = torch.full((n_col_blocks,), nnz // n_col_blocks, dtype=torch.long)
    counts[: nnz % n_col_blocks] += 1
    ccol = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    row = torch.arange(nnz) - torch.repeat_interleave(ccol[:-1], counts)
    return ccol.to(flag_gems.device, index_dtype), row.to(flag_gems.device, index_dtype)


def _make_bsc_inputs(shape, block, nnz, dtype, value_range, index_dtype=torch.int64):
    ccol, row = _make_bsc_structure(shape, block, nnz, index_dtype=index_dtype)
    values = tu.make_input(dtype, (nnz,) + tuple(block), value_range).to(
        flag_gems.device
    )
    return ccol, row, values


def _call_reference(ccol, row, values, size, dtype):
    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    return torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
        ref_ccol,
        ref_row,
        ref_values,
        size=list(size),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )


def _call_candidate(ccol, row, values, size, dtype):
    return flag_gems.sparse_bsc_tensor(
        ccol,
        row,
        values,
        size=list(size),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=flag_gems.device,
    )


def _assert_result(res_out, ref_out, dtype):
    assert res_out.layout == torch.sparse_bsc
    assert res_out.dtype == dtype
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    utils.gems_assert_equal(res_out.row_indices(), ref_out.row_indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor(case, dtype, index_dtype):
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape, block, nnz, dtype, ["-1", "1"], index_dtype=index_dtype
    )

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _bsc_shape_level_cases())
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_shape_levels(case, dtype, index_dtype):
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape, block, nnz, dtype, ["-1", "1"], index_dtype=index_dtype
    )

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    assert tuple(res_out.shape) == tuple(shape)
    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_value_ranges(case, value_range, dtype):
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(shape, block, nnz, dtype, value_range)

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_BSC_DTYPES))
)
def test_sparse_bsc_tensor_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(4)[:18].reshape(2, 3, 3)
    ccol = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)

    ref_out = _call_reference(ccol, row, values, [6, 6], dtype)
    res_out = _call_candidate(ccol, row, values, [6, 6], dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _LEGACY_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_unchecked_legacy(case, dtype, index_dtype):
    shape, nnz = case
    ccol, row = _make_bsc_structure(shape, (1, 1), nnz, index_dtype=index_dtype)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"]).to(flag_gems.device)

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_unchecked_uncoalesced(dtype, index_dtype):
    ccol = torch.tensor([0, 2, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 0, 1], dtype=index_dtype, device=flag_gems.device)
    values = tu.make_input(dtype, (3, 2, 2), ["-1", "1"]).to(flag_gems.device)

    ref_out = _call_reference(ccol, row, values, [4, 4], dtype)
    res_out = _call_candidate(ccol, row, values, [4, 4], dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_unchecked_unsorted_rows(dtype, index_dtype):
    ccol = torch.tensor([0, 3, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([1, 0, 1], dtype=index_dtype, device=flag_gems.device)
    values = tu.make_input(dtype, (3, 2, 2), ["-1", "1"]).to(flag_gems.device)

    ref_out = _call_reference(ccol, row, values, [4, 4], dtype)
    res_out = _call_candidate(ccol, row, values, [4, 4], dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_dtype_mismatch():
    ccol = torch.tensor([0, 2, 3], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float64, (3, 2, 2), ["-1", "1"])

    with pytest.raises(RuntimeError):
        _call_reference(ccol, row, values, [4, 4], torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(ccol, row, values, [4, 4], torch.float32)


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_layout():
    ccol = torch.tensor([0, 2, 3], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (3, 2, 2), ["-1", "1"])
    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            ref_ccol,
            ref_row,
            ref_values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=ref_ccol.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_bsc_tensor(
            ccol,
            row,
            values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_size():
    ccol = torch.tensor([0, 2, 3], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (3, 2, 2), ["-1", "1"])
    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            ref_ccol,
            ref_row,
            ref_values,
            size=[-4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=ref_ccol.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_bsc_tensor(
            ccol,
            row,
            values,
            size=[-4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_non_tensor():
    row = torch.tensor([0, 1, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (3, 2, 2), ["-1", "1"])

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            3.14,
            tu.to_reference(row),
            tu.to_reference(values),
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_bsc_tensor(
            3.14,
            row,
            values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
