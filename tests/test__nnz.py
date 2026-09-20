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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register the underscore-prefixed pytest marker explicitly.
setattr(
    pytest.mark,
    "_nnz",
    MarkDecorator(Mark("_nnz", (), {}, _ispytest=True), _ispytest=True),
)

# _nnz counts stored sparse entries, including explicit zeros and duplicate
# coordinates. Dense inputs are rejected.
_NNZ_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# COO: (shape, sparse_dim, nnz).
_NNZ_COO_CASES = tu.selected_cases(
    [
        ((5,), 1, 4),
        ((3, 4), 2, 7),
        ((3, 4), 1, 16),
        ((8, 8, 8), 3, 32),
        ((3, 4, 2), 2, 12),
        ((4, 3, 4, 5), 1, 24),
        ((3, 4, 5, 4, 5), 3, 40),
        ((12, 9, 3, 6), 4, 9),
        ((3, 6, 4, 4, 6, 5), 4, 11),
        ((7, 3, 12, 4, 2, 15), 5, 10),
        ((3, 4, 2, 5, 3, 4, 2), 3, 13),
    ],
    quick=[((2, 19, 7), 2, 8)],
)

_NNZ_SPEC_NNZ = 6  # Keep duplicates possible even in small index spaces.


def _make_coo_input(shape, sparse_dim, nnz, dtype, value_range, seed=0):
    # Seeded CPU indices allow duplicates; values are created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = tu.make_input(dtype, (nnz,) + dense_shape, value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_csr_input(shape, nnz, dtype, value_range):
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
        # crow/col pattern), so ``_nnz`` reports the per-batch stored count.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = tu.make_input(dtype, (shape[0], nnz), value_range)
    else:
        values = tu.make_input(dtype, (nnz,), value_range)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _assert_result(res_out, ref_out):
    assert type(res_out) is int
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("case", _NNZ_COO_CASES)
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_coo_layouts(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)
    assert inp.sparse_dim() == sparse_dim


@pytest.mark._nnz
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if len(shape) >= 1]
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_spec_shapes_value_ranges(shape, value_range, dtype):
    nnz = _NNZ_SPEC_NNZ
    inp = _make_coo_input(shape, len(shape), nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_empty_hybrid(dtype):
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_explicit_zeros(dtype):
    shape = (3, 3)
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    if dtype == torch.bool:
        values = torch.tensor([False, True, False], dtype=dtype)
    else:
        values = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_full_storage(dtype):
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")
    )
    indices = indices.reshape(2, nnz)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_NNZ_DTYPES))
)
def test__nnz_nan_inf_values_ignored(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("case", [(4, 4), (2, 4, 4), (3, 5, 7)])
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_csr(case, dtype):
    shape = case
    nnz = 5 if len(shape) == 2 else 3
    inp = _make_csr_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_csr_value_ranges(value_range, dtype):
    shape, nnz = (4, 4), 5
    inp = _make_csr_input(shape, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if 2 <= len(shape) <= 3]
)
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_spec_shapes_csr(shape, dtype):
    nnz = 5 if len(shape) == 2 else 3
    inp = _make_csr_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_csr_dense_dims(dtype):
    rows, cols, dense, nnz = 4, 4, 3, 5
    # crow segments: row0 -> 1, row1 -> 1, row2 -> 2, row3 -> 1 stored block.
    crow = torch.tensor([0, 1, 2, 4, 5])
    col = torch.tensor([0, 1, 0, 1, 2])
    values = tu.make_input(dtype, (nnz, dense), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow, col, values, (rows, cols, dense), device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = flag_gems._nnz(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._nnz
def test__nnz_dense_raises():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._nnz(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        flag_gems._nnz(inp)


@pytest.mark._nnz
def test__nnz_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._nnz(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems._nnz(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems._nnz(None)
