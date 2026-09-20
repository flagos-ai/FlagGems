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

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Copy COO entries and metadata into the destination, preserving storage order.
_SUPPORTED_DTYPES = tuple(tu.REQUIRED_DTYPES + [torch.float64, torch.int16, torch.bool])

# (shape, sparse_dim, nnz), including hybrid and empty storage.
_SPARSE_LAYOUTS = [
    ((6,), 1, 4),  # 1-D all-sparse
    ((4, 5), 2, 3),  # 2-D COO
    ((8, 8), 2, 16),  # 2-D COO, more stored entries
    ((16, 32), 2, 64),  # 2-D COO, many stored entries
    ((4, 4), 1, 3),  # hybrid, sparse_dim == 1
    ((2, 4, 5), 2, 6),  # 3-D hybrid (dense_dim == 1)
    ((3, 4, 5, 6), 2, 12),  # 4-D hybrid (dense_dim == 2)
    ((4, 5, 6), 3, 7),  # 3-D all-sparse
    ((2, 3, 4, 5), 4, 9),  # 4-D all-sparse
    ((4, 5), 2, 0),  # empty (nnz == 0) boundary
]

_VALUE_RANGE_LAYOUTS = [
    ((6,), 1, 4),
    ((4, 5), 2, 3),
    ((2, 4, 5), 2, 6),
    ((2, 3, 4, 5), 4, 9),
    ((4, 5), 2, 0),
]

_NAN_INF_LAYOUTS = [
    ((4, 5), 2, 3),
    ((2, 4, 5), 2, 6),
    ((4, 5, 6), 3, 7),
    ((2, 3, 4, 5), 4, 9),
]


def _make_indices(shape, sparse_dim, nnz, seed):
    # Generate seeded, unique indices in lexicographic order.
    gen = torch.Generator("cpu").manual_seed(seed)
    num_sparse = math.prod(shape[:sparse_dim])
    if nnz == 0:
        return torch.empty((sparse_dim, 0), dtype=torch.long)
    linear = torch.sort(torch.randperm(num_sparse, generator=gen)[:nnz]).values
    return torch.stack(torch.unravel_index(linear, shape[:sparse_dim]), dim=0)


def _make_values(values_shape, dtype, seed=0, value_range=None):
    # Use seeded random values, or the requested per-dtype range.
    if value_range is None:
        gen = torch.Generator("cpu").manual_seed(seed)
        if dtype.is_floating_point:
            return torch.randn(values_shape, dtype=torch.float32, generator=gen).to(
                dtype
            )
        if dtype == torch.bool:
            return torch.randint(0, 2, values_shape, dtype=torch.bool, generator=gen)
        low = 0 if dtype == torch.uint8 else -5
        return torch.randint(low, 6, values_shape, dtype=dtype, generator=gen)

    # Value-range framework. Seed the global RNG so make_tensor is reproducible.
    torch.manual_seed(seed)
    return tu.make_input(dtype, values_shape, list(value_range))


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, value_range=None):
    indices = _make_indices(shape, sparse_dim, nnz, seed)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    values = _make_values(values_shape, dtype, seed=seed, value_range=value_range)
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        values.to(flag_gems.device),
        tuple(shape),
        device=flag_gems.device,
    )


def _make_special_values(values_shape, dtype, scenario):
    base = tu.make_special_input(dtype, scenario)
    numel = math.prod(values_shape)
    return base.repeat((numel + base.numel() - 1) // base.numel())[:numel].view(
        values_shape
    )


def _assert_sparse_equal(res, ref):
    assert res.layout == ref.layout
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res.is_coalesced() == ref.is_coalesced()
    tu.assert_result_equal(res._indices(), ref._indices())
    tu.assert_result_equal(res._values(), ref._values())


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("layout", _SPARSE_LAYOUTS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_copy_sparse_to_sparse_(layout, dtype, non_blocking):
    shape, sparse_dim, nnz = layout
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    dst = torch.zeros_like(src)
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, non_blocking)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, non_blocking)

    # In-place semantics: the op returns self and mutates dst in place.
    assert res_out is dst
    # Validate the copied structure and entries, and that src is unchanged.
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("layout", _VALUE_RANGE_LAYOUTS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_copy_sparse_to_sparse_value_ranges(layout, dtype, value_range):
    shape, sparse_dim, nnz = layout
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype, value_range=value_range)
    dst = torch.zeros_like(src)
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert dst._nnz() == src._nnz()
    # A verbatim copy transfers the entries exactly for every value range.
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("layout", _NAN_INF_LAYOUTS)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES))
)
def test_copy_sparse_to_sparse_nan_inf(layout, dtype, scenario):
    shape, sparse_dim, nnz = layout
    base = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    src = torch.sparse_coo_tensor(
        base._indices().clone(),
        _make_special_values(values_shape, dtype, scenario),
        tuple(shape),
        device=flag_gems.device,
    )
    dst = torch.zeros_like(src)
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert dst._nnz() == src._nnz()
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_resizes_self(dtype):
    src = _make_sparse_input((6, 5), 2, 8, dtype)
    dst = _make_sparse_input((4, 5), 2, 5, dtype, seed=1)
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == 5
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert tuple(dst.shape) == tuple(src.shape) == (6, 5)
    assert dst._nnz() == src._nnz() == 8
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_resizes_nnz(dtype):
    src = _make_sparse_input((4, 5), 2, 3, dtype)
    dst = _make_sparse_input((4, 5), 2, 7, dtype, seed=1)
    assert dst._nnz() == 7
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == src._nnz() == 3
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_grows_dense_dims(dtype):
    src = _make_sparse_input((4, 5, 3), 2, 3, dtype)
    dst = _make_sparse_input((4, 5, 2), 2, 3, dtype, seed=1)
    assert tuple(dst.shape) == (4, 5, 2)
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert tuple(dst.shape) == tuple(src.shape) == (4, 5, 3)
    assert dst.dense_dim() == src.dense_dim() == 1
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_empty_dst_adopts_sparse_dim(dtype):
    shape = (2, 4, 5)
    src = _make_sparse_input(shape, 3, 4, dtype)
    dense_shape = tuple(shape[2:])
    indices = torch.empty((2, 0), dtype=torch.long, device=flag_gems.device)
    values = torch.empty((0,) + dense_shape, dtype=dtype, device=flag_gems.device)
    dst = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert dst.sparse_dim() == 2
    assert dst._nnz() == 0
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert dst.sparse_dim() == src.sparse_dim() == 3
    assert dst.dense_dim() == src.dense_dim() == 0
    assert dst._nnz() == src._nnz() == 4
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_empty_src(dtype):
    src = _make_sparse_input((4, 5), 2, 0, dtype)
    dst = _make_sparse_input((4, 5), 2, 3, dtype, seed=1)
    assert dst._nnz() == 3
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == src._nnz() == 0
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_uncoalesced(dtype):
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values((4,), dtype, seed=0).to(flag_gems.device)
    src = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    assert not src.is_coalesced()
    dst = torch.zeros_like(src)
    ref_src = tu.to_reference(src)
    ref_dst = tu.to_reference(dst)

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = flag_gems.copy_sparse_to_sparse_(dst, src, False)

    assert res_out is dst
    assert dst._nnz() == src._nnz() == 4
    # Entry order is preserved too, so the indices can be compared directly.
    _assert_sparse_equal(res_out, ref_out)
    _assert_sparse_equal(src, ref_src)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_dense_self():
    src = _make_sparse_input((4, 5), 2, 3, torch.float32)
    self_dense = torch.zeros((4, 5), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(self_dense, src, False)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.copy_sparse_to_sparse_(self_dense, src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_dense_src():
    src_dense = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    ref_dst = _make_sparse_input((4, 5), 2, 3, torch.float32)
    res_dst = _make_sparse_input((4, 5), 2, 3, torch.float32, seed=1)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, src_dense, False)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.copy_sparse_to_sparse_(res_dst, src_dense, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_csr():
    csr = torch.randn(
        (4, 5), dtype=torch.float32, device=flag_gems.device
    ).to_sparse_csr()
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(csr.clone(), csr, False)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.copy_sparse_to_sparse_(csr.clone(), csr, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_sparse_dim_change():
    ref_src = _make_sparse_input((2, 4, 5), 3, 3, torch.float32)
    ref_dst = _make_sparse_input((2, 4, 5), 2, 3, torch.float32)
    res_src = _make_sparse_input((2, 4, 5), 3, 3, torch.float32, seed=1)
    res_dst = _make_sparse_input((2, 4, 5), 2, 3, torch.float32, seed=2)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.copy_sparse_to_sparse_(res_dst, res_src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_shrinking_sparse_dims():
    ref_src = _make_sparse_input((4, 5), 2, 3, torch.float32)
    ref_dst = _make_sparse_input((6, 5), 2, 3, torch.float32)
    res_src = _make_sparse_input((4, 5), 2, 3, torch.float32, seed=1)
    res_dst = _make_sparse_input((6, 5), 2, 3, torch.float32, seed=2)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.copy_sparse_to_sparse_(res_dst, res_src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_shrinking_dense_dims():
    ref_src = _make_sparse_input((4, 5, 2), 2, 3, torch.float32)
    ref_dst = _make_sparse_input((4, 5, 3), 2, 3, torch.float32)
    res_src = _make_sparse_input((4, 5, 2), 2, 3, torch.float32, seed=1)
    res_dst = _make_sparse_input((4, 5, 3), 2, 3, torch.float32, seed=2)
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.copy_sparse_to_sparse_(res_dst, res_src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_backward():
    src = _make_sparse_input((4, 5), 2, 3, torch.float32)
    src.requires_grad_(True)
    dst = torch.zeros_like(src)
    with pytest.raises((RuntimeError, TypeError)):
        out = torch.ops.aten.copy_sparse_to_sparse_(dst.clone(), src, False)
        torch.autograd.grad(out.to_dense().sum(), [src], allow_unused=True)
    with pytest.raises((RuntimeError, TypeError)):
        out = flag_gems.copy_sparse_to_sparse_(torch.zeros_like(src), src, False)
        torch.autograd.grad(out.to_dense().sum(), [src], allow_unused=True)
