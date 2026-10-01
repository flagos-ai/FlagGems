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

"""Correctness tests for the sparse COO accessor aten::indices.

Dimensions this operator cannot express, and the mechanism:

* The stored values never reach the result, so the value grid varies the operand,
  not the answer.  A single operand has no broadcast dimension, and the schema
  has no scalar operand and no optional parameter, so the broadcast,
  scalar/tensor and parameter-sweep dimensions have nothing to vary.
* The result is an int64 index tensor, which autograd cannot track, so there is
  no backward to exercise; the non-differentiable contract is checked instead.
* nnz is clamped to the logical element count, so even the large spec shapes
  store few coordinates and never allocate their dense shape.
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# The nine required dtypes plus the extra storage types this backend accepts,
# probed with the real call torch.ops.aten.indices on a coalesced COO operand.
_INDICES_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.bool,
    torch.complex64,
] + ([torch.float64] if utils.fp64_is_supported else [])

_INDICES_DTYPES = [dtype for dtype in _INDICES_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

_NNZ = 6
# The layout and mutation rows only need representative values, so reuse the
# first value range of the level rather than naming its representation here.
_VALUE_RANGE = tu.selected_ranges()[0]

# (sparse_shape, dense_shape, nnz, index_layout).  A non-empty dense_shape is a
# hybrid COO operand; index_layout selects the backing of the coordinate storage
# (see _index_view) so the accessor is exercised against non-normalized indices.
_LAYOUT_ROWS = [
    ((5,), (), 4, "contiguous"),
    ((3, 4), (), 7, "contiguous"),
    ((3, 4, 2), (), 12, "contiguous"),
    ((4, 3, 4, 5), (), 24, "contiguous"),
    ((12, 9, 3, 6), (), 9, "contiguous"),
    ((3, 6, 4, 4, 6, 5), (), 11, "contiguous"),
    ((7, 3, 12, 4, 2, 15), (), 10, "contiguous"),
    ((2, 3, 4, 5, 6), (), 40, "contiguous"),
    ((4, 5, 6), (7,), 9, "contiguous"),
    ((3, 4, 2), (5, 3), 11, "contiguous"),
    ((1,), (), 6, "contiguous"),
    ((), (), 1, "contiguous"),
    ((2, 19, 7), (), 8, "contiguous"),
    # nnz == 0 operands store no coordinates: only shape and metadata remain.
    ((5,), (), 0, "contiguous"),
    ((3, 4), (), 0, "contiguous"),
    ((4, 5, 6), (7,), 0, "contiguous"),
    ((2, 19, 7), (), 0, "contiguous"),
    # Strided and nonzero-offset coordinate storage: the accessor must hand back
    # the operand stored index tensor, not a normalized contiguous copy.
    ((3, 4), (), 7, "strided_cols"),
    ((4, 5, 6), (7,), 9, "offset_window"),
    ((3, 4, 2), (), 11, "transposed"),
    ((2, 3, 4, 5, 6), (3,), 13, "strided_cols"),
]
# Every row is tiny (nnz <= 40 and no dense block), so quick keeps the whole
# rank / hybrid / scalar-shape / empty / alias coverage.
_LAYOUT_CASES = tu.selected_cases(_LAYOUT_ROWS, quick=_LAYOUT_ROWS)
_LAYOUT_DTYPES = [torch.float32, torch.bool, torch.float8_e4m3fn]

_LAYOUT_DTYPES = [dtype for dtype in _LAYOUT_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

_MUTATION_ROWS = [
    ((3, 4), (), 7, "contiguous"),
    ((4, 5, 6), (7,), 9, "contiguous"),
    ((3, 4), (), 7, "strided_cols"),
    ((4, 5, 6), (7,), 9, "offset_window"),
    ((3, 4, 2), (), 11, "transposed"),
    ((2, 3, 4, 5, 6), (3,), 13, "contiguous"),
    ((2, 19, 7), (), 8, "offset_window"),
]
_MUTATION_CASES = tu.selected_cases(_MUTATION_ROWS, quick=_MUTATION_ROWS)

# Positive NaN/Inf operands stay in the default suite.  The dtype list is the
# tested set, so e4m3fn keeps its nan-only case and e5m2 its nan/inf/mixed cases.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_INDICES_DTYPES), quick=[])

# The state check runs in both modes, so it keeps float64 where supported.
_GRAD_DTYPES = [
    dtype for dtype in _INDICES_DTYPES if dtype.is_floating_point or dtype.is_complex
]


def _distinct_flat(numel, nnz, device, seed=0):
    """Exactly min(nnz, numel) distinct flat row-major coordinates.

    A step coprime with numel makes index * step % numel injective, so the
    realized nnz always matches the request instead of shrinking when random
    draws collide; sorting restores the row-major order of a coalesced operand.
    Only nnz values are allocated, never the logical shape.
    """
    count = min(nnz, numel)
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    step = max(1, numel // count)
    while math.gcd(step, numel) != 1:
        step += 1
    index = torch.arange(count, dtype=torch.long, device=device)
    return torch.sort((index * step + seed) % numel).values


def _unravel(flat, extents):
    """Row-major unflatten of distinct flat coordinates into extents."""
    coords = torch.empty(
        (len(extents), flat.numel()), dtype=torch.long, device=flat.device
    )
    rest = flat
    for dim in reversed(range(len(extents))):
        coords[dim] = rest % extents[dim]
        rest = rest // extents[dim]
    return coords


def _index_view(coords, layout):
    """Back coords with layout storage and return that view."""
    sparse_dim, nnz = coords.shape
    if layout == "contiguous":
        return coords.contiguous()
    if layout == "strided_cols":
        storage = coords.new_zeros(sparse_dim, 2 * nnz)
        storage[:, ::2] = coords
        return storage[:, ::2]
    if layout == "offset_window":
        storage = coords.new_zeros(sparse_dim, 2 * nnz + 2)
        storage[:, 1 : nnz + 1] = coords
        return storage[:, 1 : nnz + 1]
    if layout == "transposed":
        return coords.t().contiguous().t()
    raise ValueError(f"invalid index layout {layout!r}")


def _wrap_coo(sparse_shape, src, values):
    """Wrap an index view into a coalesced COO operand that keeps its storage.

    is_coalesced=True marks the already row-major coordinates coalesced and
    keeps the source view strides, offset and storage.  Calling coalesce()
    runs the coalesce kernel, which has no FP8 instantiation and normalizes the
    index layout.  src is built on flag_gems.device and the operand is created on
    the same device, because a device transfer would copy the coordinates into
    fresh contiguous index storage.
    """
    return torch.sparse_coo_tensor(
        src,
        values,
        tuple(sparse_shape) + tuple(values.shape[1:]),
        device=flag_gems.device,
        is_coalesced=True,
    )


def _make_coo(
    sparse_shape, dense_shape, nnz, dtype, value_range, index_layout="contiguous"
):
    """Return an operand and the index view that backs its coordinates."""
    coords = _unravel(
        _distinct_flat(math.prod(sparse_shape), nnz, flag_gems.device), sparse_shape
    )
    src = _index_view(coords, index_layout)
    values = tu.make_input(dtype, (coords.shape[1],) + tuple(dense_shape), value_range)
    return _wrap_coo(sparse_shape, src, values), src


def _assert_indices_result(res_out, ref_out, inp, src):
    """Exact coordinates, then the metadata and storage aliasing of the accessor."""
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.shape == (inp.sparse_dim(), inp._nnz())
    assert res_out.dtype == torch.int64
    assert res_out.layout == torch.strided
    assert res_out.device == inp.device
    # The operand kept the source view as its index storage: same strides, same
    # offset, and the same storage when there are coordinates to alias.
    assert inp._indices().stride() == src.stride()
    assert inp._indices().storage_offset() == src.storage_offset()
    if src.numel():
        assert inp._indices().data_ptr() == src.data_ptr()
    # Tensor(a) -> Tensor(a): the result is that stored index tensor itself, so
    # it inherits the source strides and offset instead of being normalized.
    assert res_out.stride() == src.stride()
    assert res_out.storage_offset() == src.storage_offset()
    assert res_out.data_ptr() == inp._indices().data_ptr()


@pytest.mark.indices
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test_indices_value_ranges(shape, value_range, dtype):
    inp, src = _make_coo(shape, (), _NNZ, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.indices(ref_inp)
    res_out = flag_gems.indices(inp)

    _assert_indices_result(res_out, ref_out, inp, src)


@pytest.mark.indices
@pytest.mark.parametrize("case", _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_indices_coo_layouts(case, dtype):
    sparse_shape, dense_shape, nnz, index_layout = case
    inp, src = _make_coo(
        sparse_shape, dense_shape, nnz, dtype, _VALUE_RANGE, index_layout
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.indices(ref_inp)
    res_out = flag_gems.indices(inp)

    _assert_indices_result(res_out, ref_out, inp, src)


@pytest.mark.indices
@pytest.mark.parametrize("case", _MUTATION_CASES)
def test_indices_result_aliases_index_storage(case):
    sparse_shape, dense_shape, nnz, index_layout = case
    inp, src = _make_coo(
        sparse_shape, dense_shape, nnz, torch.float32, _VALUE_RANGE, index_layout
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.indices(ref_inp)
    res_out = flag_gems.indices(inp)

    _assert_indices_result(res_out, ref_out, inp, src)
    values_before = tu.to_reference(inp._values())

    # The result is not a copy, so a write through it must land in the operand
    # index storage.  Every mutation row has first extent >= 2, so the
    # incremented coordinate always differs from the stored one.
    last = inp._nnz() - 1
    new_value = (int(src[0, last]) + 1) % sparse_shape[0]
    res_out[0, last] = new_value
    ref_out[0, last] = new_value
    tu.assert_result_equal(inp._indices(), ref_inp._indices())

    # A direct write to the operand index storage is visible to a later call.
    inp._indices()[0, 0] = new_value
    ref_inp._indices()[0, 0] = new_value
    tu.assert_result_equal(flag_gems.indices(inp), torch.ops.aten.indices(ref_inp))

    # The coordinates are never taken from the operand values.
    tu.assert_result_equal(inp._values(), values_before)


@pytest.mark.indices
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_indices_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    # One coordinate per stored value: NaN/Inf live only in the values, so they
    # must never leak into the returned coordinates.
    src = torch.arange(
        values.numel(), dtype=torch.long, device=flag_gems.device
    ).reshape(1, -1)
    inp = _wrap_coo((values.numel(),), src, values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.indices(ref_inp)
    res_out = flag_gems.indices(inp)

    _assert_indices_result(res_out, ref_out, inp, src)


@pytest.mark.indices
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test_indices_result_is_not_differentiable(dtype):
    inp, src = _make_coo((3, 4), (), 5, dtype, _VALUE_RANGE)
    inp.requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.indices(ref_inp)
    res_out = flag_gems.indices(inp)

    _assert_indices_result(res_out, ref_out, inp, src)
    # The accessor returns int64 coordinates, which autograd cannot track, so the
    # operator has no backward to exercise.
    assert not res_out.requires_grad
    assert res_out.grad_fn is None


@pytest.mark.indices
def test_indices_rejects_uncoalesced():
    # torch.sparse_coo_tensor leaves the coalesced flag unset even for unique
    # coordinates, so this operand is uncoalesced and must be refused.
    coords = torch.tensor(
        [[0, 1, 1], [2, 0, 2]], dtype=torch.long, device=flag_gems.device
    )
    values = torch.ones(3, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(coords, values, (2, 3), device=flag_gems.device)

    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems.indices(inp)


@pytest.mark.indices
def test_indices_rejects_strided_dense():
    inp = tu.make_input(torch.float32, (4, 4), _VALUE_RANGE)

    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems.indices(inp)


@pytest.mark.indices
def test_indices_rejects_sparse_csr():
    crow = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    col = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), _VALUE_RANGE)
    inp = torch.sparse_csr_tensor(crow, col, values, (2, 4), device=flag_gems.device)

    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems.indices(inp)


@pytest.mark.indices
def test_indices_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.indices(3.14)


@pytest.mark.indices
def test_indices_requires_operand():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.indices()
