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
    "_values",
    MarkDecorator(Mark("_values", (), {}, _ispytest=True), _ispytest=True),
)

# _values returns a non-differentiable view of the COO value storage.
# Explicit zeros and duplicate entries are preserved; dense and CSR inputs fail.
_VALUES_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

_VALUES_QUICK_CASES = [((2, 19, 7), 2, 8), ((3, 4), 2, 7), ((3, 4, 2), 2, 12)]

# COO: (shape, sparse_dim, nnz).
_VALUES_COO_CASES = tu.selected_cases(
    [
        ((5,), 1, 4),
        ((3, 4), 2, 7),
        ((3, 4), 1, 16),
        ((8, 8, 8), 3, 32),
        ((3, 4, 2), 2, 12),
        ((4, 3, 4, 5), 1, 24),
        ((3, 4, 5, 4, 5), 3, 40),
        ((12, 9, 3, 6), 4, 48),
        ((3, 6, 4, 4, 6, 5), 4, 64),
        ((7, 3, 12, 4, 2, 15), 5, 80),
        ((3, 4, 2, 5, 3, 4, 2), 3, 96),
    ],
    quick=_VALUES_QUICK_CASES,
)

# Value-range layouts: all-sparse and hybrid COO.
_VALUES_RANGE_CASES = tu.selected_cases(
    [
        ((3, 4), 2, 7),
        ((3, 4, 2), 2, 12),
        ((12, 9, 3, 6), 4, 48),
    ],
    quick=_VALUES_QUICK_CASES,
)


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
    # Check exact output, storage aliasing and unchanged input values.
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp._values().data_ptr()
    utils.gems_assert_equal(inp._values(), ref_inp._values(), equal_nan=True)


@pytest.mark._values
@pytest.mark.parametrize("case", _VALUES_COO_CASES)
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_layouts(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("case", _VALUES_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_value_ranges(case, value_range, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == 0
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_empty_hybrid(dtype):
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == 0
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_full_storage(dtype):
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    ).reshape(2, nnz)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == nnz
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_VALUES_DTYPES))
)
def test__values_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
def test__values_dense_raises():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._values(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._values(inp)


@pytest.mark._values
def test__values_csr_raises():
    crow_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    col_indices = torch.tensor([0, 1], dtype=torch.long)
    values = tu.make_input(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._values(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._values(inp)


@pytest.mark._values
def test__values_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._values(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._values(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten._values("not-a-tensor")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._values("not-a-tensor")
