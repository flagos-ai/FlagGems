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
    "_dimI",
    MarkDecorator(Mark("_dimI", (), {}, _ispytest=True), _ispytest=True),
)

# _dimI reports the sparse dimension count of a COO tensor. Dense and
# compressed sparse inputs are rejected.
_DIMI_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)

# COO: (shape, sparse_dim); small layouts also check the sparse/dense split.
_DIMI_SMALL_CASES = [
    ((5,), 1),
    ((3, 4), 2),
    ((3, 4), 1),
    ((8, 8, 8), 3),
    ((3, 4, 2), 2),
    ((4, 3, 4, 5), 1),
    ((3, 4, 5, 4, 5), 3),
]
_DIMI_COO_CASES = tu.selected_cases(
    _DIMI_SMALL_CASES
    + [
        ((12, 9, 3, 6), 4),
        ((3, 6, 4, 4, 6, 5), 4),
        ((7, 3, 12, 4, 2, 15), 5),
        ((3, 4, 2, 5, 3, 4, 2), 3),
        ((2, 4, 2, 4, 2, 4), 2),
    ],
    quick=[((2, 19, 7), 2), ((2, 19, 7), 3), ((2, 19, 7, 5), 2)],
)

# Value ranges: (shape, sparse_dim) for hybrid COO layouts.
_DIMI_HYBRID_CASES = tu.selected_cases(
    [
        ((3, 4), 1),
        ((3, 4, 2), 2),
        ((4, 3, 4, 5), 1),
        ((3, 4, 5, 4, 5), 3),
    ],
    quick=[((2, 19, 7), 2)],
)


def _make_coo_input(shape, sparse_dim, dtype, value_range, nnz=8, seed=0):
    # Seeded CPU indices allow duplicates; values are created on the test device.
    shape = tuple(shape)
    if sparse_dim < 1:
        raise ValueError("sparse COO tensors need at least one sparse dimension")
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = tu.make_input(dtype, (nnz,) + tuple(dense_shape), value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_empty_coo(shape, sparse_dim, dtype):
    shape = tuple(shape)
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(
        (0,) + shape[sparse_dim:], dtype=dtype, device=flag_gems.device
    )
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _assert_result(res_out, ref_out):
    assert isinstance(res_out, int) and not isinstance(res_out, bool)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._dimI
@pytest.mark.parametrize("case", _DIMI_COO_CASES)
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_coo(case, dtype):
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == sparse_dim
    assert inp.dense_dim() == len(shape) - sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if len(shape) >= 1]
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_shape_value_range_grid(shape, value_range, dtype):
    inp = _make_coo_input(shape, len(shape), dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)
    assert inp.dense_dim() == 0


@pytest.mark._dimI
@pytest.mark.parametrize("case", _DIMI_HYBRID_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_hybrid_value_ranges(case, value_range, dtype):
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimI
@pytest.mark.parametrize("case", _DIMI_SMALL_CASES)
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_hybrid_dense_dim_zero(case, dtype):
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    inp = _make_empty_coo(shape, sparse_dim, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._dimI
@pytest.mark.parametrize("shape, sparse_dim", [((4, 5, 6), 2), ((4, 5, 6), 1)])
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_empty_hybrid(shape, sparse_dim, dtype):
    inp = _make_empty_coo(shape, sparse_dim, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)
    assert inp.dense_dim() == len(shape) - sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_single_entry(dtype):
    shape, sparse_dim = (3, 4, 5), 2
    inp = _make_coo_input(shape, sparse_dim, dtype, ["-1", "1"], nnz=1)
    assert inp._nnz() == 1
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._dimI
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DIMI_DTYPES))
)
def test__dimI_nan_inf_values_ignored(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = flag_gems._dimI(inp)

    _assert_result(res_out, ref_out)


_NEGATIVE_EXC = (
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
)


@pytest.mark._dimI
def test__dimI_dense_raises():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimI(tu.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._dimI(inp)


@pytest.mark._dimI
def test__dimI_csr_raises():
    crow_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([0, 1])
    values = tu.make_input(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimI(tu.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._dimI(inp)


@pytest.mark._dimI
def test__dimI_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._dimI(3.14)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._dimI(3.14)
