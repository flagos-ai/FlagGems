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

# crow_indices returns a view of the compressed row pointers in CSR/BSR storage.
_CSR_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool, torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)

# CSR: (shape, nnz), including multiple batch dimensions.
_CSR_CASES = tu.selected_cases(
    [
        ((1, 1), 1),
        ((1, 6), 4),
        ((5, 4), 7),
        ((256, 256), 512),
        ((1024, 1024), 4096),
        ((3, 5, 4), 7),
        ((20, 320, 15), 100),
        ((16, 128, 64, 60), 50),
        ((16, 7, 57, 32, 29), 5),
        ((3, 8), 16),
        ((8, 3), 12),
        ((4, 4), 16),
        ((2, 4, 6), 12),
        ((2, 3, 4, 5), 8),
        ((12, 9, 3, 6), 9),
        ((3, 6, 4, 4, 6, 5), 11),
        ((7, 3, 12, 4, 2, 15), 10),
        ((3, 4, 2, 5, 3, 4, 2), 8),
    ],
    quick=[((2, 19, 7), 8)],
)

# Value-range layouts: plain and batched CSR.
_CSR_RANGE_CASES = tu.selected_cases(
    [
        ((5, 4), 7),
        ((3, 5, 4), 7),
        ((3, 6, 4, 4, 6, 5), 11),
    ],
    quick=[((2, 19, 7), 8)],
)

# (layout, matrix_shape, values_shape, compressed_indices, plain_indices).
_INDEX_LAYOUT_CASES = [
    (torch.sparse_csr, (4, 6), (4,), [0, 2, 2, 3, 4], [0, 4, 1, 5]),
    (torch.sparse_bsr, (4, 6), (3, 2, 3), [0, 2, 3], [0, 1, 1]),
    (torch.sparse_csr, (4, 6), (0,), [0, 0, 0, 0, 0], []),
    (torch.sparse_bsr, (4, 6), (0, 2, 3), [0, 0, 0], []),
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
    batch = shape[:-2]
    entries_shape = batch + (nnz,)
    assert 0 <= nnz <= nrows * ncols
    positions = torch.arange(nnz, dtype=torch.long) * (nrows * ncols) // max(nnz, 1)
    rows = (positions // ncols).expand(entries_shape).contiguous()
    cols = (positions % ncols).expand(entries_shape).contiguous()
    batch_numel = 1
    for dim in batch:
        batch_numel *= dim
    offset = (torch.arange(batch_numel, dtype=torch.long) * nrows).view(batch_numel, 1)
    flat = (rows.reshape(batch_numel, nnz) + offset).reshape(-1)
    counts = torch.bincount(flat, minlength=batch_numel * nrows).view(
        batch_numel, nrows
    )
    crow = torch.zeros(batch_numel, nrows + 1, dtype=torch.long)
    crow[:, 1:] = torch.cumsum(counts, -1)
    crow = crow.view(batch + (nrows + 1,))
    values = tu.make_input(dtype, entries_shape, value_range)
    return torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # Check exact output, storage aliasing and unchanged input metadata/values.
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == torch.ops.aten.crow_indices(inp).data_ptr()
    utils.gems_assert_equal(inp.crow_indices(), ref_inp.crow_indices())
    utils.gems_assert_equal(inp.col_indices(), ref_inp.col_indices())
    if inp.dtype.is_floating_point:
        utils.gems_assert_equal(inp.values(), ref_inp.values(), equal_nan=True)
    else:
        utils.gems_assert_equal(inp.values(), ref_inp.values())


@pytest.mark.crow_indices
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_index_layouts(case, batch_shape, dense_shape, dtype, index_dtype):
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
    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)
    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("case", _CSR_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_layouts(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("case", _CSR_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_value_ranges(case, value_range, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_empty(dtype):
    shape = (4, 5)
    crow = torch.zeros(5, dtype=torch.long, device=flag_gems.device)
    cols = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_empty_batched(dtype):
    shape = (2, 4, 5)
    crow = torch.zeros(2, 5, dtype=torch.long, device=flag_gems.device)
    cols = torch.empty(2, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(2, 0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_single_row(dtype):
    shape, nnz = (1, 7), 5
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_unchecked_uncoalesced(dtype):
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert cols[0].item() == cols[1].item()
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_full_storage(dtype):
    shape = (2, 3)
    crow = torch.tensor([0, 3, 6], dtype=torch.long, device=flag_gems.device)
    cols = torch.arange(3).repeat(2).to(flag_gems.device)  # [0, 1, 2, 0, 1, 2]
    values = tu.make_input(dtype, (6,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    assert inp._nnz() == 6
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_CSR_DTYPES))
)
def test_crow_indices_nan_inf_values_ignored(dtype, scenario):
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor(
        [0, 1, 0, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_special_input(dtype, scenario).repeat(2)[:7]
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
def test_crow_indices_dense_raises():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.crow_indices(inp)


@pytest.mark.crow_indices
def test_crow_indices_csc_raises():
    ccol = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor(
        [0, 1, 0, 1, 0, 1], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_input(torch.float32, (6,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol, row_indices, values.to(flag_gems.device), (2, 3)
    )
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.crow_indices(inp)


@pytest.mark.crow_indices
def test_crow_indices_coo_raises():
    inp = torch.randn(3, 4, device=flag_gems.device).to_sparse_coo()
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.crow_indices(inp)


@pytest.mark.crow_indices
def test_crow_indices_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.crow_indices(3.14)
