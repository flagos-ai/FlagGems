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
_VALUE_DTYPES = tu.REQUIRED_DTYPES + [torch.bool, torch.int16, torch.float64]
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FLOAT_VALUE_DTYPES = [
    dtype
    for dtype in _VALUE_DTYPES
    if dtype.is_floating_point and dtype not in _FP8_DTYPES
]

# (size, block_shape, crow_indices, col_indices).
_BSR_2D_CASES = [
    ((4, 4), (2, 2), [0, 2, 4], [0, 1, 0, 1]),
    ((6, 6), (2, 2), [0, 2, 3, 3], [0, 2, 1]),
    ((4, 6), (2, 3), [0, 1, 2], [0, 1]),
    ((8, 8), (4, 2), [0, 2, 3], [1, 3, 0]),
    ((6, 4), (3, 2), [0, 1, 2], [0, 1]),
    ((10, 12), (2, 4), [0, 3, 5, 6, 6, 6], [0, 1, 2, 0, 2, 1]),
]

# Leading dimensions are batch dimensions; batches share the index grid.
_BSR_BATCHED_CASES = [
    ((2, 6, 6), (2, 3), [0, 2, 4, 4], [0, 1, 0, 1]),
    ((3, 4, 8), (2, 4), [0, 2, 3], [0, 1, 0]),
    ((2, 4, 4), (2, 2), [0, 2, 2], [0, 1]),
    ((2, 8, 12), (4, 3), [0, 1, 3], [0, 1, 3]),
    ((2, 3, 4, 4), (2, 2), [0, 2, 4], [0, 1, 0, 1]),
]

# (size, block_shape, batch); None denotes an unbatched matrix.
_BSR_EMPTY_CASES = [
    ((4, 4), (2, 2), None),
    ((2, 4, 4), (2, 2), 2),
]

# (block_shape, crow_indices, col_indices), with size inferred from indices.
_BSR_2D_INFERRED_CASES = [
    ((2, 2), [0, 2, 4], [0, 1, 0, 1]),
    ((2, 2), [0, 2, 3, 3], [0, 2, 1]),
    ((2, 3), [0, 1, 2], [0, 1]),
]

# Block shape used for the shared shape sweep.
_BLOCK = (2, 2)


def _shape_level_cases():
    # Skip scalars, make 1-D shapes square, and round matrix extents to full blocks.
    cases = []
    for shape in tu.selected_shapes():
        if len(shape) == 0:
            continue
        if len(shape) == 1:
            batch, rows, cols = (), shape[0], shape[0]
        else:
            batch, rows, cols = shape[:-2], shape[-2], shape[-1]
        br, bc = _BLOCK
        rows = max(br, rows // br * br)
        cols = max(bc, cols // bc * bc)
        n_row_blocks = rows // br
        n_col_blocks = cols // bc
        gen = torch.Generator("cpu").manual_seed(len(shape))
        crow = [0]
        col = []
        for r in range(n_row_blocks):
            k = min(1 + (r % 2), n_col_blocks)
            chosen = torch.randperm(n_col_blocks, generator=gen)[:k].sort().values
            col.extend(chosen.tolist())
            crow.append(crow[-1] + k)
        cases.append((batch + (rows, cols), _BLOCK, crow, col))
    return cases


def _make_values(shape, dtype, value_range):
    # Clamp bounds per dtype; fill collapsed intervals with a constant.
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)

    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        low = max(low, finfo.min)
        high = min(high, finfo.max)
        if not low < high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
    else:
        low, high = int(low), int(high)
        dmin, dmax = tu.dtype_bounds(dtype)
        low = max(low, int(dmin))
        high = min(max(high, low), int(dmax))
        if low >= high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def _make_bsr_values(nnz, block, dtype, batch=None, value_range=("-1", "1")):
    if batch is None:
        shape = (nnz, block[0], block[1])
    elif isinstance(batch, tuple):
        shape = batch + (nnz, block[0], block[1])
    else:
        shape = (batch,) + (nnz, block[0], block[1])
    return _make_values(shape, dtype, value_range)


def _assert_bsr_structure(out, size, block, nnz, dtype, batch=None):
    assert out.layout == torch.sparse_bsr
    assert tuple(out.shape) == tuple(size)
    assert out.dtype == dtype
    assert out.sparse_dim() == 2
    assert out.dense_dim() == 0
    assert out._nnz() == nnz
    if batch is None:
        assert tuple(out.values().shape) == (nnz, block[0], block[1])
    elif isinstance(batch, tuple):
        assert tuple(out.values().shape) == batch + (nnz, block[0], block[1])
    else:
        assert tuple(out.values().shape) == (batch,) + (nnz, block[0], block[1])
    n_row_blocks = size[-2] // block[0]
    n_col_blocks = size[-1] // block[1]
    assert out.crow_indices().shape == size[:-2] + (n_row_blocks + 1,)
    assert (out.col_indices() < n_col_blocks).all()
    assert (out.col_indices() >= 0).all()


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_CASES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_size(case, dtype, value_range):
    size, block, crow, col = case
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, dtype, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_bsr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    # The constructor reads its inputs; it must not mutate them.
    utils.gems_assert_equal(crow_t, ref_crow)
    utils.gems_assert_equal(col_t, ref_col)
    utils.gems_assert_equal(values, ref_values)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_BATCHED_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_size_batched(case, dtype, value_range):
    size, block, crow, col = case
    batch = size[:-2]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    crow_t = crow_t.expand(batch + (len(crow),)).contiguous()
    col_t = col_t.expand(batch + (nnz,)).contiguous()
    values = _make_bsr_values(nnz, block, dtype, batch=batch, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_bsr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype, batch=batch)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test_sparse_bsr_tensor_crow_col_value_size_empty(case, dtype):
    size, block, batch = case
    n_row_blocks = size[-2] // block[0]
    crow_t = torch.zeros(
        size[:-2] + (n_row_blocks + 1,), dtype=torch.long, device=flag_gems.device
    )
    col_t = torch.empty(size[:-2] + (0,), dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(0, block, dtype, batch=batch)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_bsr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, 0, dtype, batch=batch)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value(case, dtype, value_range):
    block, crow, col = case
    nnz = len(col)
    size = ((len(crow) - 1) * block[0], (max(col) + 1) * block[1])
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, dtype, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_bsr_tensor(
        crow_t, col_t, values, dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _shape_level_cases())
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_shape_levels(case, dtype, value_range):
    size, block, crow, col = case
    batch = size[:-2]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    crow_t = crow_t.expand(batch + (len(crow),)).contiguous()
    col_t = col_t.expand(batch + (nnz,)).contiguous()
    values = _make_bsr_values(nnz, block, dtype, batch=batch, value_range=value_range)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = flag_gems.sparse_bsr_tensor(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype, batch=batch)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_VALUE_DTYPES))
)
def test_sparse_bsr_tensor_nan_inf_values(dtype, scenario):
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    values = tu.make_special_input(dtype, scenario).repeat(4)[:16].reshape(nnz, *block)
    ref_values = tu.to_reference(values)

    ref_crow = torch.tensor([0, 2, 4], dtype=torch.long, device=ref_values.device)
    ref_col = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=ref_values.device)
    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_values.device,
    )
    res_out = flag_gems.sparse_bsr_tensor(
        torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device),
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device),
        values,
        list(size),
        dtype=dtype,
        device=values.device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    tu.assert_result_equal(res_out.crow_indices(), ref_out.crow_indices())
    tu.assert_result_equal(res_out.col_indices(), ref_out.col_indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test_sparse_bsr_tensor_boundary_values(dtype):
    # Include exact finfo endpoints that random range sampling may miss.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    finfo = torch.finfo(dtype)
    specials = torch.tensor(
        [finfo.min, finfo.max, 0.0, -0.0, 1.0, -1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    n_elems = nnz * block[0] * block[1]
    values = specials.repeat((n_elems + specials.numel() - 1) // specials.numel())[
        :n_elems
    ]
    values = values.reshape(nnz, block[0], block[1])
    ref_values = tu.to_reference(values)

    ref_crow = torch.tensor([0, 2, 4], dtype=torch.long, device=ref_values.device)
    ref_col = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=ref_values.device)
    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_values.device,
    )
    res_out = flag_gems.sparse_bsr_tensor(
        torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device),
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device),
        values,
        list(size),
        dtype=dtype,
        device=values.device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_bsr_tensor
def test_sparse_bsr_tensor_rejects_dtype_mismatch():
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, torch.float16)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            ref_crow,
            ref_col,
            ref_values,
            list(size),
            dtype=torch.float32,
            device=ref_crow.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32, device=crow_t.device
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32])
def test_sparse_bsr_tensor_rejects_missing_dtype(dtype):
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, dtype)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            ref_crow, ref_col, ref_values, list(size), device=ref_crow.device
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), device=crow_t.device
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_bsr_tensor_rejects_device_mismatch():
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, torch.float32)
    cpu_values = values.cpu()

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="device inference from component tensors only applies to accelerators",
)
def test_sparse_bsr_tensor_rejects_missing_device():
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, torch.float32)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
