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

# sparse_dim counts sparse dimensions: zero for strided tensors, the
# leading sparse rank for COO, and two for the CSR layouts below.
_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# Dense shapes, including scalars and higher ranks.
_DENSE_SHAPES = tu.selected_cases(
    [
        (),
        (5,),
        (3, 4),
        (8, 8, 8),
        (3, 4, 2, 5),
        (3, 4, 5, 4, 5),
        (3, 6, 4, 4, 6, 5, 4),
        (7, 3, 12, 4, 2, 15, 2, 2),
    ],
    quick=[(2, 19, 7)],
)

# COO: (sparse_shape, dense_shape, nnz).
_COO_CASES = tu.selected_cases(
    [
        ((4, 4), (), 8),
        ((8, 8, 8), (), 64),
        ((4, 4), (3,), 8),
        ((2, 3, 4), (5,), 12),
        ((16, 16), (7, 13), 40),
        ((2, 3, 4), (5, 6), 12),
        ((3,), (4, 5, 6), 2),
        ((12, 9, 3, 6), (4,), 9),
        ((3, 4, 2, 5, 3), (4, 2), 11),
    ],
    quick=[((2, 19, 7), (), 8)],
)

# COO value ranges: all-sparse and hybrid layouts.
_COO_RANGE_CASES = tu.selected_cases(
    [
        ((3, 4), (), 7),
        ((3, 4), (3,), 8),
        ((12, 9, 3, 6), (4,), 9),
    ],
    quick=[((2, 19, 7), (), 8)],
)

# CSR: (shape, nnz), including batched tensors.
_CSR_CASES = tu.selected_cases(
    [
        ((4, 4), 3),
        ((2, 4, 4), 5),
        ((3, 5, 7), 3),
        ((3, 4, 4), 4),
    ],
    quick=[((2, 19, 7), 3)],
)

_SPEC_NNZ = 6  # Keep duplicates possible even in small index spaces.


def _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range, seed=0):
    # Seeded CPU indices allow duplicates; values are created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = tu.make_input(dtype, (nnz,) + tuple(dense_shape), value_range)
    size = tuple(sparse_shape) + tuple(dense_shape)
    return torch.sparse_coo_tensor(indices, values, size, device=flag_gems.device)


def _make_csr(shape, nnz, dtype, value_range):
    if len(shape) == 2:
        rows, cols = shape
    else:
        _, rows, cols = shape
    assert 0 <= nnz <= rows * cols
    counts = torch.full((rows,), nnz // rows, dtype=torch.long)
    counts[: nnz % rows] += 1
    crow_indices = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    col_indices = torch.arange(nnz) - torch.repeat_interleave(crow_indices[:-1], counts)
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so the layout stays 2-D sparse for every batch.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = tu.make_input(dtype, (shape[0], nnz), value_range)
    else:
        values = tu.make_input(dtype, (nnz,), value_range)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _make_csr_with_dense_dims(dtype, value_range):
    """CSR layout carrying a dense dim: shape (rows, cols, dense)."""
    rows, cols, dense, nnz = 4, 4, 3, 5
    # crow segments: row0 -> 1, row1 -> 1, row2 -> 2, row3 -> 1 stored block.
    crow = torch.tensor([0, 1, 2, 4, 5], dtype=torch.long, device=flag_gems.device)
    col = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(dtype, (nnz, dense), value_range)
    return torch.sparse_csr_tensor(
        crow, col, values, (rows, cols, dense), device=flag_gems.device
    )


def _make_empty_csr(shape, dtype):
    """Build an empty plain or batched CSR tensor."""
    if len(shape) == 2:
        rows, _ = shape
        crow_indices = torch.zeros(rows + 1, dtype=torch.long, device=flag_gems.device)
        col_indices = torch.empty(0, dtype=torch.long, device=flag_gems.device)
        values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    else:
        _, rows, _ = shape
        crow_indices = torch.zeros(
            shape[0], rows + 1, dtype=torch.long, device=flag_gems.device
        )
        col_indices = torch.empty(
            shape[0], 0, dtype=torch.long, device=flag_gems.device
        )
        values = torch.empty(shape[0], 0, dtype=dtype, device=flag_gems.device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _assert_result(res_out, ref_out):
    assert type(res_out) is int
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", _DENSE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_dense_layouts(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", [(0,), (0, 5), (2, 0, 3)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_dense(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    assert inp.numel() == 0
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_dense_spec_shapes_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _COO_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_coo_layouts(case, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == len(sparse_shape)
    assert inp.dense_dim() == len(dense_shape)
    assert inp._nnz() == nnz


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if len(shape) >= 1]
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_coo_spec_shapes_value_ranges(shape, value_range, dtype):
    inp = _make_coo(tuple(shape), (), _SPEC_NNZ, dtype, value_range, seed=0)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _COO_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_coo_value_ranges(case, value_range, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_coo(dtype):
    sparse_shape, dense_shape = (3, 4), (5, 6)
    indices = torch.empty(
        len(sparse_shape), 0, dtype=torch.long, device=flag_gems.device
    )
    values = torch.empty(
        (0,) + tuple(dense_shape), dtype=dtype, device=flag_gems.device
    )
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_uncoalesced_coo(dtype):
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    values = tu.make_input(dtype, (5,) + tuple(dense_shape), ["-1", "1"])
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _CSR_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_layouts(case, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == 2
    assert inp.dense_dim() == 0


@pytest.mark.sparse_dim
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_value_ranges(value_range, dtype):
    shape, nnz = (4, 4), 5
    inp = _make_csr(shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_dense_dims(value_range, dtype):
    inp = _make_csr_with_dense_dims(dtype, value_range)
    assert inp.sparse_dim() == 2
    assert inp.dense_dim() == 1
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", [(4, 4), (3, 4, 4)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_csr(shape, dtype):
    inp = _make_empty_csr(shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if 2 <= len(shape) <= 3]
)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_spec_shapes(shape, dtype):
    nnz = 5 if len(shape) == 2 else 3
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_sparse_dim_nan_inf_dense(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).repeat(2)[:6].reshape(2, 3)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_sparse_dim_nan_inf_coo(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_sparse_dim_nan_inf_csr(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    crow_indices = torch.tensor([0, 3, 4, 6], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (3, 4), device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = flag_gems.sparse_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.sparse_dim
def test_sparse_dim_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_dim(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.sparse_dim(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.sparse_dim(None)
