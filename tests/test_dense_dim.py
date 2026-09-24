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

# dense_dim counts the trailing dense dimensions of a sparse tensor,
# or the full rank of a strided tensor.
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

# CSR value ranges: plain and batched layouts.
_CSR_RANGE_CASES = tu.selected_cases(
    [
        ((4, 4), 3),
        ((2, 4, 4), 5),
    ],
    quick=[((2, 19, 7), 3)],
)


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
    values = tu.make_input(dtype, (nnz,), value_range)
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so the logical rank is 3 and dense_dim stays 0.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = values.expand(shape[0], -1).contiguous()
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _make_empty_csr(shape, dtype):
    """Build an empty plain or batched CSR tensor."""
    if len(shape) == 2:
        rows = shape[0]
        crow_indices = torch.zeros(rows + 1, dtype=torch.long, device=flag_gems.device)
        col_indices = torch.empty(0, dtype=torch.long, device=flag_gems.device)
        values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    else:
        rows = shape[1]
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


@pytest.mark.dense_dim
@pytest.mark.parametrize("shape", _DENSE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_dense(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("shape", [(0,), (0, 5), (2, 0, 3)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_empty_dense(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    assert inp.numel() == 0
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_dense_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_noncontiguous_dense(dtype):
    base = tu.make_input(dtype, (4, 5, 6), ["-1", "1"])
    inp = base.transpose(0, 2)
    assert not inp.is_contiguous()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("case", _COO_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_sparse_coo(case, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == len(dense_shape)
    assert inp.sparse_dim() == len(sparse_shape)
    assert inp._nnz() == nnz


@pytest.mark.dense_dim
@pytest.mark.parametrize("case", _COO_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_sparse_coo_value_ranges(case, value_range, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("case", _CSR_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_sparse_csr(case, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == 0
    assert inp.sparse_dim() == 2


@pytest.mark.dense_dim
@pytest.mark.parametrize("case", _CSR_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_sparse_csr_value_ranges(case, value_range, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_empty_coo(dtype):
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

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("shape", [(4, 4), (3, 4, 4)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_empty_csr(shape, dtype):
    inp = _make_empty_csr(shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_dim_uncoalesced_coo(dtype):
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    values = tu.make_input(dtype, (5,) + tuple(dense_shape), ["-1", "1"])
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_dense_dim_nan_inf_dense(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).repeat(2)[:6].reshape(2, 3)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_dense_dim_nan_inf_coo(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_dense_dim_nan_inf_csr(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    crow_indices = torch.tensor([0, 3, 4, 6], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (3, 4), device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.dense_dim(ref_inp)
    res_out = flag_gems.dense_dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dense_dim
def test_dense_dim_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.dense_dim(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.dense_dim(3.14)
