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

# ccol_indices returns a view of the compressed column pointers in CSC/BSC storage.
_CSC_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# CSC: (shape, nnz), including multiple batch dimensions.
_CSC_CASES = tu.selected_cases(
    [
        ((5, 4), 7),
        ((3, 8), 16),
        ((8, 3), 12),
        ((4, 4), 16),
        ((1, 6), 4),
        ((3, 5, 4), 7),
        ((2, 4, 6), 12),
        ((2, 3, 4, 5), 8),
        ((12, 9, 3, 6), 9),
        ((3, 6, 4, 4, 6, 5), 11),
        ((7, 3, 12, 4, 2, 15), 10),
        ((3, 4, 2, 5, 3, 4, 2), 8),
    ],
    quick=[((2, 19, 7), 8), ((4, 5), 6)],
)

# Value-range layouts: plain and batched CSC.
_CSC_RANGE_CASES = tu.selected_cases(
    [
        ((5, 4), 7),
        ((3, 5, 4), 7),
        ((3, 6, 4, 4, 6, 5), 11),
    ],
    quick=[((2, 19, 7), 8), ((4, 5), 6)],
)

# CSC needs two matrix dimensions; retain leading batch dimensions.
_CSC_SHAPE_CASES = [
    (tuple(shape), min(4, shape[-2] * shape[-1]))
    for shape in tu.selected_shapes()
    if len(shape) >= 2
]

# (layout, matrix_shape, values_shape, compressed_indices, plain_indices).
_INDEX_LAYOUT_CASES = [
    (torch.sparse_csc, (6, 4), (4,), [0, 2, 2, 3, 4], [0, 4, 1, 5]),
    (torch.sparse_bsc, (6, 4), (3, 3, 2), [0, 2, 3], [0, 1, 1]),
    (torch.sparse_csc, (6, 4), (0,), [0, 0, 0, 0, 0], []),
    (torch.sparse_bsc, (6, 4), (0, 3, 2), [0, 0, 0], []),
]
_INDEX_CASES = tu.selected_cases(
    [
        (case, batch_shape, dense_shape)
        for case in _INDEX_LAYOUT_CASES
        for batch_shape in [(), (2,), (2, 3)]
        for dense_shape in [(), (2,)]
    ],
    quick=[
        (_INDEX_LAYOUT_CASES[0], (), ()),
        (_INDEX_LAYOUT_CASES[1], (2,), (2,)),
        (_INDEX_LAYOUT_CASES[2], (), ()),
        (_INDEX_LAYOUT_CASES[3], (2,), ()),
    ],
)


def _make_input(shape, nnz, dtype, value_range):
    # Unique sorted coordinates, spread over the logical matrix.
    nrows, ncols = shape[-2], shape[-1]
    batch = tuple(shape[:-2])
    entries_shape = batch + (nnz,)
    assert 0 <= nnz <= nrows * ncols
    positions = torch.arange(nnz, dtype=torch.long) * (nrows * ncols) // max(nnz, 1)
    rows = (positions % nrows).expand(entries_shape).contiguous()
    cols = (positions // nrows).expand(entries_shape).contiguous()
    batch_numel = 1
    for dim in batch:
        batch_numel *= dim
    offset = (torch.arange(batch_numel, dtype=torch.long) * ncols).view(batch_numel, 1)
    flat = (cols.reshape(batch_numel, nnz) + offset).reshape(-1)
    counts = torch.bincount(flat, minlength=batch_numel * ncols).view(batch + (ncols,))
    ccol = torch.zeros(batch + (ncols + 1,), dtype=torch.long)
    ccol[..., 1:] = torch.cumsum(counts, -1)
    values = tu.make_input(dtype, entries_shape, value_range)
    return torch.sparse_csc_tensor(
        ccol.to(flag_gems.device),
        rows.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # Check exact output, storage aliasing and unchanged input metadata/values.
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == torch.ops.aten.ccol_indices(inp).data_ptr()
    utils.gems_assert_equal(inp.ccol_indices(), ref_inp.ccol_indices())
    utils.gems_assert_equal(inp.row_indices(), ref_inp.row_indices())
    if inp.dtype.is_floating_point:
        utils.gems_assert_equal(inp.values(), ref_inp.values(), equal_nan=True)
    else:
        utils.gems_assert_equal(inp.values(), ref_inp.values())


@pytest.mark.ccol_indices
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_index_layouts(case, batch_shape, dense_shape, dtype, index_dtype):
    layout, matrix_shape, values_shape, compressed, plain = case
    compressed = torch.tensor(compressed, dtype=index_dtype, device=flag_gems.device)
    plain = torch.tensor(plain, dtype=index_dtype, device=flag_gems.device)
    compressed = compressed.expand(batch_shape + compressed.shape).contiguous()
    plain = plain.expand(batch_shape + plain.shape).contiguous()
    values = tu.make_input(dtype, batch_shape + values_shape + dense_shape, ["-1", "1"])
    inp = torch.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        size=batch_shape + matrix_shape + dense_shape,
        layout=layout,
        check_invariants=True,
    )
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)
    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("case", _CSC_CASES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_layouts(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("case", _CSC_SHAPE_CASES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_shape_levels(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("case", _CSC_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_value_ranges(case, value_range, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_empty(dtype):
    shape = (4, 5)
    ccol = torch.zeros(6, dtype=torch.long, device=flag_gems.device)
    rows = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_empty_batched(dtype):
    shape = (2, 4, 5)
    ccol = torch.zeros(2, 6, dtype=torch.long, device=flag_gems.device)
    rows = torch.empty(2, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(2, 0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_single_column(dtype):
    shape, nnz = (7, 1), 5
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_unchecked_uncoalesced(dtype):
    shape = (4, 3)
    ccol = torch.tensor([0, 3, 3, 5], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert rows[0].item() == rows[1].item()
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(ccol, rows, values.to(flag_gems.device), shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_full_storage(dtype):
    shape = (2, 3)
    ccol = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=flag_gems.device)
    rows = torch.arange(2).repeat(3).to(flag_gems.device)  # [0, 1, 0, 1, 0, 1]
    values = tu.make_input(dtype, (6,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(ccol, rows, values.to(flag_gems.device), shape)
    assert inp._nnz() == 6
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_CSC_DTYPES))
)
def test_ccol_indices_nan_inf_values_ignored(dtype, scenario):
    shape = (3, 4)
    ccol = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_special_input(dtype, scenario).repeat(2)[:7]
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
def test_ccol_indices_dense_raises():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices(inp)


@pytest.mark.ccol_indices
def test_ccol_indices_csr_raises():
    crow_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    col_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values.to(flag_gems.device), (2, 4)
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices(inp)


@pytest.mark.ccol_indices
def test_ccol_indices_coo_raises():
    inp = torch.randn(3, 4, device=flag_gems.device).to_sparse_coo()
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ccol_indices(inp)


@pytest.mark.ccol_indices
def test_ccol_indices_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.ccol_indices(3.14)
