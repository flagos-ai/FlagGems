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
    "_indices",
    MarkDecorator(Mark("_indices", (), {}, _ispytest=True), _ispytest=True),
)

# _indices returns a view of the COO index storage, including duplicate entries.
# Dense and compressed sparse inputs are rejected.
_INDICES_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# COO: (shape, sparse_dim, nnz).
_INDICES_COO_CASES = tu.selected_cases(
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

# Value-range layouts: all-sparse and hybrid COO.
_INDICES_RANGE_CASES = tu.selected_cases(
    [
        ((3, 4), 2, 7),
        ((3, 4, 2), 2, 12),
        ((12, 9, 3, 6), 4, 9),
    ],
    quick=[((2, 19, 7), 2, 8)],
)

_INDICES_SPEC_NNZ = 6  # Allow repeated coordinates in small index spaces.


def _make_coo_input(shape, sparse_dim, nnz, dtype, value_range, seed=0):
    # Seeded CPU indices may repeat; values are created on the test device.
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


def _assert_result(res_out, ref_out, inp, ref_inp):
    # Check exact output, storage aliasing and unchanged input metadata/values.
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp._indices().data_ptr()
    utils.gems_assert_equal(inp._indices(), ref_inp._indices())
    if inp.dtype.is_floating_point:
        utils.gems_assert_equal(inp._values(), ref_inp._values(), equal_nan=True)
    else:
        utils.gems_assert_equal(inp._values(), ref_inp._values())


@pytest.mark._indices
@pytest.mark.parametrize("case", _INDICES_COO_CASES)
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_layouts(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if len(shape) >= 1]
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_spec_shapes_value_ranges(shape, value_range, dtype):
    nnz = _INDICES_SPEC_NNZ
    inp = _make_coo_input(shape, len(shape), nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("case", _INDICES_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_value_ranges(case, value_range, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_empty_hybrid(dtype):
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_explicit_zeros(dtype):
    shape = (3, 3)
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    if dtype == torch.bool:
        values = torch.tensor([False, False, False], dtype=dtype)
    else:
        values = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)
    assert res_out.shape == (2, 3)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_full_storage(dtype):
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    ).reshape(2, nnz)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == nnz
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_INDICES_DTYPES))
)
def test__indices_nan_inf_values_ignored(dtype, scenario):
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = flag_gems._indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
def test__indices_dense_raises():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._indices(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        flag_gems._indices(inp)


@pytest.mark._indices
def test__indices_csr_raises():
    crow_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    col_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 4), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._indices(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        flag_gems._indices(inp)


@pytest.mark._indices
def test__indices_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems._indices(3.14)
