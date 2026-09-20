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
_FP8_CSR_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_CSR_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + _FP8_CSR_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)
_SHAPE_LEVEL_DTYPES = [
    torch.float32,
    torch.float8_e4m3fn,
    torch.int32,
]
_BOUNDARY_CSR_DTYPES = utils.ALL_FLOAT_DTYPES + _FP8_CSR_DTYPES

# (size, crow_indices, col_indices).
_CSR_2D_CASES = [
    ((4, 4), [0, 2, 4, 4, 4], [0, 1, 0, 1]),
    ((5, 4), [0, 2, 3, 3, 5, 5], [0, 1, 2, 0, 3]),
    ((3, 6), [0, 1, 3, 5], [0, 2, 4, 0, 5]),
    ((1, 4), [0, 2], [0, 3]),
    ((6, 1), [0, 1, 1, 2, 2, 3, 3], [0, 0, 0]),
    ((7, 5), [0, 1, 3, 4, 4, 6, 6, 6], [0, 2, 4, 1, 2, 3]),
    ((3, 3), [0, 3, 4, 6], [0, 1, 2, 0, 1, 2]),
]

# Batched matrices with a separate index grid per batch.
_CSR_3D_CASES = [
    ((2, 3, 4), [[0, 2, 3, 4], [0, 1, 3, 4]], [[0, 1, 0, 2], [1, 0, 2, 3]]),
    (
        (3, 4, 5),
        [[0, 1, 3, 4, 5], [0, 2, 2, 3, 5], [0, 1, 2, 4, 5]],
        [[0, 1, 3, 2, 4], [0, 4, 2, 1, 3], [1, 3, 0, 4, 2]],
    ),
    (
        (2, 5, 4),
        [[0, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 6]],
        [[0, 1, 3, 0, 2, 1], [3, 2, 0, 1, 0, 3]],
    ),
]

# (size, batch); None denotes an unbatched matrix.
_CSR_EMPTY_CASES = [
    ((4, 5), None),
    ((2, 4, 5), 2),
]

# (crow_indices, col_indices), with size inferred from indices.
_CSR_2D_INFERRED_CASES = [
    ([0, 2, 4], [0, 1, 0, 1]),
    ([0, 2, 2, 4], [0, 1, 0, 1]),
    ([0, 1, 3, 5], [0, 2, 4, 0, 5]),
    ([0, 1, 2], [0, 0]),
]

# Reject negative matrix dimensions.
_CSR_INVALID_SIZES = [[-1, 4], [4, -1], [-2, -2]]


def _range_for_dtype(dtype, value_range):
    # Clamp a negative lower bound to zero for unsigned storage.
    low_symbol, high_symbol = value_range
    if dtype != torch.bool and not dtype.is_floating_point:
        if torch.iinfo(dtype).min == 0 and low_symbol.startswith("-"):
            low_symbol = "0"
    return [low_symbol, high_symbol]


def _shape_level_cases():
    # Skip scalars, make 1-D shapes square, and generate seeded ragged rows.
    cases = []
    for shape in tu.selected_shapes():
        if len(shape) == 0:
            continue
        if len(shape) == 1:
            batch, rows, cols = (), shape[0], shape[0]
        else:
            batch, rows, cols = shape[:-2], shape[-2], shape[-1]
        gen = torch.Generator("cpu").manual_seed(len(shape))
        crow = [0]
        col = []
        for r in range(rows):
            k = min(1 + (r % 2), cols)
            chosen = torch.randperm(cols, generator=gen)[:k].sort().values
            col.extend(chosen.tolist())
            crow.append(crow[-1] + k)
        cases.append((batch + (rows, cols), crow, col))
    return cases


def _make_csr_values(nnz, dtype, batch=None, value_range=("-1", "1")):
    if batch is None:
        shape = (nnz,)
    elif isinstance(batch, tuple):
        shape = batch + (nnz,)
    else:
        shape = (batch,) + (nnz,)
    return tu.make_input(dtype, shape, _range_for_dtype(dtype, value_range))


def _assert_csr_structure(out, size, nnz, dtype, batch=None):
    assert out.layout == torch.sparse_csr
    assert tuple(out.shape) == tuple(size)
    assert out.dtype == dtype
    assert out.sparse_dim() == 2
    assert out.dense_dim() == 0
    assert out._nnz() == nnz
    expected_crow_len = size[-2] + 1
    if batch is None:
        assert tuple(out.values().shape) == (nnz,)
        assert len(out.crow_indices()) == expected_crow_len
        assert len(out.col_indices()) == nnz
    elif isinstance(batch, tuple):
        assert tuple(out.values().shape) == batch + (nnz,)
        assert tuple(out.crow_indices().shape) == batch + (expected_crow_len,)
        assert tuple(out.col_indices().shape) == batch + (nnz,)
    else:
        assert tuple(out.values().shape) == (batch, nnz)
        assert tuple(out.crow_indices().shape) == (batch, expected_crow_len)
        assert tuple(out.col_indices().shape) == (batch, nnz)
    assert (out.col_indices() >= 0).all()
    assert (out.col_indices() < size[-1]).all()


def _assert_csr_equal(res_out, ref_out):
    assert res_out.layout == ref_out.layout
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_crow_col_value_size(case, dtype, value_range):
    size, crow, col = case
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    # Values are stored verbatim, so the float comparison is exact within
    # tolerance; the index arrays must match bit-for-bit.
    _assert_csr_equal(res_out, ref_out)
    # The constructor reads its inputs; it must not mutate them.
    utils.gems_assert_equal(crow_t, ref_crow)
    utils.gems_assert_equal(col_t, ref_col)
    utils.gems_assert_equal(values, ref_values)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_3D_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_crow_col_value_size_batched(case, dtype, value_range):
    size, crow, col = case
    batch = size[0]
    nnz = len(col[0])
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype, batch=batch, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype, batch=batch)
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_empty(case, dtype):
    size, batch = case
    n_rows = size[-2]
    if batch is None:
        crow_t = torch.zeros(n_rows + 1, dtype=torch.long, device=flag_gems.device)
        col_t = torch.empty(0, dtype=torch.long, device=flag_gems.device)
        values = _make_csr_values(0, dtype)
    else:
        crow_t = torch.zeros(
            batch, n_rows + 1, dtype=torch.long, device=flag_gems.device
        )
        col_t = torch.empty(batch, 0, dtype=torch.long, device=flag_gems.device)
        values = _make_csr_values(0, dtype, batch=batch)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, 0, dtype, batch=batch)
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_CASES[:3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_index_dtypes(case, index_dtype, dtype):
    size, crow, col = case
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=index_dtype, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=index_dtype, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    assert res_out.crow_indices().dtype == index_dtype
    assert res_out.col_indices().dtype == index_dtype
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_trailing_empty_rows(dtype):
    # The final three rows contain no stored entries.
    size = (5, 2)
    crow = [0, 2, 4, 4, 4, 4]
    col = [0, 1, 0, 1]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_crow_col_value(case, dtype, value_range):
    crow, col = case
    nnz = len(col)
    size = (len(crow) - 1, max(col) + 1)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _shape_level_cases())
@pytest.mark.parametrize("dtype", _SHAPE_LEVEL_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_shape_levels(case, dtype, value_range):
    size, crow, col = case
    batch = size[:-2] or None
    nnz = len(col)
    if batch is None:
        crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
        col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    else:
        crow_t = (
            torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
            .expand(batch + (len(crow),))
            .contiguous()
        )
        col_t = (
            torch.tensor(col, dtype=torch.long, device=flag_gems.device)
            .expand(batch + (nnz,))
            .contiguous()
        )
    values = _make_csr_values(nnz, dtype, batch=batch, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype, batch=batch)
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_CSR_DTYPES))
)
def test_sparse_csr_tensor_nan_inf_values(dtype, scenario):
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    values = tu.make_special_input(dtype, scenario)[:nnz]
    ref_values = tu.to_reference(values)

    crow_t = torch.tensor([0, 2, 4, 4, 4], dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_values.device,
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=values.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", _BOUNDARY_CSR_DTYPES)
def test_sparse_csr_tensor_boundary_values(dtype):
    # Include exact finfo endpoints that random range sampling may miss.
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    finfo = torch.finfo(dtype)
    specials = torch.tensor(
        [finfo.min, finfo.max, 0.0, -0.0, 1.0, -1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    values = specials.repeat((nnz + specials.numel() - 1) // specials.numel())[:nnz]
    ref_values = tu.to_reference(values)

    crow_t = torch.tensor([0, 2, 4, 4, 4], dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_values.device
    )
    res_out = flag_gems.sparse_csr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=values.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    _assert_csr_equal(res_out, ref_out)


@pytest.mark.sparse_csr_tensor
def test_sparse_csr_tensor_rejects_dtype_mismatch():
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float16)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            ref_crow,
            ref_col,
            ref_values,
            list(size),
            dtype=torch.float32,
            device=ref_crow.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32, device=crow_t.device
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int32]
)
def test_sparse_csr_tensor_rejects_missing_dtype(dtype):
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            ref_crow, ref_col, ref_values, list(size), device=ref_crow.device
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csr_tensor(
            crow_t, col_t, values, list(size), device=crow_t.device
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("invalid_size", _CSR_INVALID_SIZES)
def test_sparse_csr_tensor_rejects_invalid_size(invalid_size):
    crow = [0, 2, 4, 4, 4]
    col = [0, 1, 0, 1]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float32)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            ref_crow,
            ref_col,
            ref_values,
            list(invalid_size),
            dtype=torch.float32,
            device=ref_crow.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csr_tensor(
            crow_t,
            col_t,
            values,
            list(invalid_size),
            dtype=torch.float32,
            device=crow_t.device,
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_csr_tensor_rejects_device_mismatch():
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float32)
    cpu_values = values.cpu()

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.skipif(
    flag_gems.device != "cuda",
    reason="the missing-device quirk only exists on CUDA",
)
def test_sparse_csr_tensor_rejects_missing_device():
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float32)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
