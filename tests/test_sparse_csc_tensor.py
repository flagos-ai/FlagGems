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

# Store CSC indices and values verbatim, including repeated/unsorted rows.
# Pass dtype and device explicitly to the ATen factory.
_EXACT_CSC_DTYPES = [torch.int8, torch.uint8] + utils.ALL_INT_DTYPES + [torch.bool]
_FLOAT_STORAGE_DTYPES = utils.ALL_FLOAT_DTYPES + [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
_CSC_DTYPES = _FLOAT_STORAGE_DTYPES + _EXACT_CSC_DTYPES
_INDEX_DTYPES = [torch.int32, torch.int64]

# (matrix_shape, nnz), then batched variants.
_CSC_CASES = [
    ((1, 8), 4),
    ((8, 1), 4),
    ((4, 4), 16),
    ((5, 7), 13),
    ((7, 5), 13),
    ((4, 4), 0),
]
_CSC_BATCHED_CASES = [
    ((2, 4, 4), 4),
    ((3, 6, 5), 6),
]

# (ccol_indices, row_indices); size is inferred by the factory.
_CSC_NO_SIZE_CASES = [
    ([0, 2, 3, 5], [0, 2, 1, 2, 3]),
    ([0, 3, 3, 4], [0, 1, 2, 3]),
    ([0, 0, 0], []),
    ([0, 2, 4], [0, 1, 0, 1]),
]

# Skip scalars; map 1-D shapes to square matrices.
_CSC_SHAPE_CASES = [
    (
        (shape + shape, shape[0])
        if len(shape) == 1
        else (shape, min(shape[-2] * shape[-1], 8))
    )
    for shape in tu.selected_shapes()
    if shape
]
_BOUNDARY_RANGES = [
    ["min", "min"],
    ["max", "max"],
    ["0", "0"],
    ["1", "1"],
    ["-1", "-1"],
]


def _make_values(nnz, dtype, shape=None, value_range=("-1", "1")):
    # Clamp uint8 bounds and fill ranges that collapse to a constant.
    shape = (nnz,) if shape is None else shape
    if dtype == torch.uint8:
        low = max(int(tu.resolve_bound(value_range[0], dtype)), 0)
        high = max(int(tu.resolve_bound(value_range[1], dtype)), 0)
        if low > high:
            low = high
        if low == high:
            return torch.full(shape, low, device=flag_gems.device, dtype=dtype)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )
    return tu.make_input(dtype, shape, value_range).to(flag_gems.device)


def _make_csc_inputs(
    shape, nnz, dtype, index_dtype=torch.int64, value_range=("-1", "1")
):
    device = flag_gems.device
    batch, (M, N) = shape[:-2], shape[-2:]
    assert 0 <= nnz <= M * N
    counts = torch.full((N,), nnz // N, dtype=torch.long)
    counts[: nnz % N] += 1
    ccol = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    row = torch.arange(nnz) - torch.repeat_interleave(ccol[:-1], counts)
    ccol = ccol.expand(batch + (N + 1,)).contiguous()
    row = row.expand(batch + (nnz,)).contiguous()
    values = _make_values(nnz, dtype, shape=batch + (nnz,), value_range=value_range)
    return (
        ccol.to(device=device, dtype=index_dtype),
        row.to(device=device, dtype=index_dtype),
        values,
    )


def _assert_result(res_out, ref_out, dtype, index_dtype):
    assert res_out.layout == torch.sparse_csc
    assert res_out.dtype == dtype
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.sparse_dim() == 2
    assert res_out.dense_dim() == 0
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)
    assert res_out.ccol_indices().dtype == index_dtype
    assert res_out.row_indices().dtype == index_dtype
    utils.gems_assert_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    utils.gems_assert_equal(res_out.row_indices(), ref_out.row_indices())
    # Metadata and index arrays above, stored values here: no densification
    # is needed to validate the constructor.
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _CSC_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor(shape, nnz, dtype, index_dtype, value_range):
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, index_dtype=index_dtype, value_range=value_range
    )

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _CSC_BATCHED_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor_batched(shape, nnz, dtype, index_dtype, value_range):
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, index_dtype=index_dtype, value_range=value_range
    )

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("ccol_list, row_list", _CSC_NO_SIZE_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor_no_size(
    ccol_list, row_list, dtype, index_dtype, value_range
):
    ccol = torch.tensor(ccol_list, dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor(row_list, dtype=index_dtype, device=flag_gems.device)
    values = _make_values(len(row_list), dtype, value_range=value_range)

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)
    assert tuple(res_out.shape) == tuple(ref_out.shape)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_unchecked_uncoalesced(dtype, index_dtype):
    ccol = torch.tensor([0, 1, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 0, 0], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype, value_range=["-1", "1"])

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)
    assert torch.ops.aten._nnz(res_out) == 3


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_unchecked_unsorted_rows(dtype, index_dtype):
    ccol = torch.tensor([0, 3, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([1, 0, 2], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype, value_range=["-1", "1"])

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [3, 1],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        [3, 1],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _CSC_SHAPE_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _FLOAT_STORAGE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor_shape_levels(shape, nnz, dtype, index_dtype, value_range):
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, index_dtype=index_dtype, value_range=value_range
    )

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)
    assert tuple(res_out.shape) == tuple(shape)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("value_range", _BOUNDARY_RANGES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_boundary_values(dtype, value_range):
    ccol, row, values = _make_csc_inputs(
        (4, 4), 4, dtype, index_dtype=torch.int64, value_range=value_range
    )

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [4, 4],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        [4, 4],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, torch.int64)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_CSC_DTYPES))
)
def test_sparse_csc_tensor_nan_inf_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ccol = torch.tensor([0, 2, 5], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int64, device=flag_gems.device)

    ref_ccol = tu.to_reference(ccol)
    ref_row = tu.to_reference(row)
    ref_values = tu.to_reference(values)
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [3, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = flag_gems.sparse_csc_tensor(
        ccol,
        row,
        values,
        [3, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, torch.int64)
    tu.assert_result_equal(ccol, ref_ccol)
    tu.assert_result_equal(row, ref_row)
    tu.assert_result_equal(values, ref_values)


def _build_default_inputs(dtype=torch.float32, index_dtype=torch.int64):
    values = _make_values(2, dtype, value_range=["0", "1"])
    ccol = torch.tensor([0, 1, 2], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 1], dtype=index_dtype, device=flag_gems.device)
    return ccol, row, values


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_dtype_mismatch():
    ccol, row, values = _build_default_inputs(dtype=torch.float16)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_missing_dtype():
    ccol, row, values = _build_default_inputs(dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_wrong_layout():
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_negative_size():
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [-2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [-2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_csc_tensor_rejects_device_mismatch():
    ccol, row, values = _build_default_inputs()
    values = values.to("cpu")
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_csc_tensor_rejects_missing_device():
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )
