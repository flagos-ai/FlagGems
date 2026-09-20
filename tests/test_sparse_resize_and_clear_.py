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

from . import accuracy_utils as utils
from . import test_utils as tu

# Resize COO metadata in place and discard all stored entries.
_RESIZE_DTYPES = (
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)

# (src_shape, src_sparse_dim, src_dense_dim, dst_shape, dst_sparse_dim, dst_dense_dim, nnz).
_RESIZE_CASES = [
    ((4, 5), 2, 0, (4, 5), 2, 0, 5),
    ((4, 5), 2, 0, (4, 5), 1, 1, 5),
    ((2, 3), 2, 0, (4, 5), 2, 0, 4),
    ((5, 5), 2, 0, (2, 3), 2, 0, 7),
    ((4, 5, 6), 2, 1, (4, 5, 6), 2, 1, 5),
    ((4, 5, 6), 1, 2, (3, 6, 7), 1, 2, 4),
    ((4, 5, 6), 2, 1, (4, 5, 6), 1, 2, 5),
    ((5,), 1, 0, (7,), 1, 0, 3),
    ((2, 3, 4, 5), 2, 2, (2, 3, 4, 5), 2, 2, 6),
    ((2, 2, 2, 2, 2), 3, 2, (3, 3, 3, 2, 2), 3, 2, 8),
    ((4, 5), 2, 0, (4, 5), 0, 2, 5),
    ((2, 3, 4), 3, 0, (2, 3, 4), 2, 1, 6),
    ((4, 5), 2, 0, (4, 5), 2, 0, 0),
]

# (dst_shape, sparse_dim, dense_dim) for an empty source.
_EMPTY_SOURCE_TARGETS = [
    ((7,), 1, 0),
    ((2, 3), 2, 0),
    ((4, 5, 6), 2, 1),
    ((4, 5), 0, 2),
    ((3, 3, 3, 3), 3, 1),
]

_NEGATIVE_DTYPES = [torch.float32, torch.int8]

_INVALID_CALLS = [
    pytest.param([4, 5], 1, 0, id="split_too_small"),
    pytest.param([4, 5], 2, 1, id="split_too_large"),
    pytest.param([4, 5], 3, 0, id="split_too_large_sparse"),
    pytest.param([4, 5], -1, 2, id="negative_sparse_dim"),
    pytest.param([4, 5], 2, -1, id="negative_dense_dim"),
    pytest.param([4, -5], 2, 0, id="negative_size"),
]

_SELECTED_RANGES = tu.selected_ranges()

_VALUE_RANGE_PAIRS = [
    (dtype, value_range) for dtype in _RESIZE_DTYPES for value_range in _SELECTED_RANGES
]

_VALUE_RANGE_IDS = [
    f"{str(dtype).replace('torch.', '')}-{'_'.join(value_range)}"
    for dtype, value_range in _VALUE_RANGE_PAIRS
]


def _default_values(dtype, values_shape, gen):
    # Generate on CPU before transferring; FP8 is cast from float32.
    if dtype.is_floating_point:
        base = torch.randn(values_shape, dtype=torch.float32, generator=gen)
        return base.to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    if dtype == torch.uint8:
        return torch.randint(0, 6, values_shape, dtype=dtype, generator=gen)
    return torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, values=None):
    # Use seeded unique coordinates; supplied values replace the default payload.
    gen = torch.Generator("cpu").manual_seed(seed)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    num_sparse = math.prod(shape[:sparse_dim])
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(num_sparse, generator=gen, device="cpu")[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if values is None:
        values = _default_values(dtype, values_shape, gen)
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
        device=flag_gems.device,
    )


def _split_for_shape(shape):
    # Use half the target dimensions as sparse dimensions, rounded up.
    ndim = len(shape)
    if ndim == 0:
        return 0, 0
    sparse_dim = (ndim + 1) // 2
    return sparse_dim, ndim - sparse_dim


def _assert_empty_resized(t, shape, sparse_dim, dense_dim, dtype):
    # Check the requested split and empty storage, including its coalesced flag.
    assert t.layout == torch.sparse_coo
    assert tuple(t.shape) == tuple(shape)
    assert t.dtype == dtype
    assert t.sparse_dim() == sparse_dim
    assert t.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(t) == 0
    assert tuple(torch.ops.aten._indices(t).shape) == (sparse_dim, 0)
    assert tuple(torch.ops.aten._values(t).shape) == (0,) + tuple(shape[sparse_dim:])
    assert t.is_coalesced()


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_(case, dtype):
    src_shape, src_spd, src_dnd, dst_shape, dst_spd, dst_dnd, src_nnz = case
    inp = _make_sparse_input(src_shape, src_spd, src_nnz, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(dst_shape), dst_spd, dst_dnd
    )
    res_out = flag_gems.sparse_resize_and_clear_(inp, list(dst_shape), dst_spd, dst_dnd)

    # In-place semantics: the op returns self and mutates the input in place.
    assert res_out is inp
    # The mutated input (not only the return value) carries the new structure.
    _assert_empty_resized(inp, dst_shape, dst_spd, dst_dnd, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dst_shape,dst_spd,dst_dnd", _EMPTY_SOURCE_TARGETS)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_empty_source(dst_shape, dst_spd, dst_dnd, dtype):
    inp = _make_sparse_input((4, 5), 2, 0, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(dst_shape), dst_spd, dst_dnd
    )
    res_out = flag_gems.sparse_resize_and_clear_(inp, list(dst_shape), dst_spd, dst_dnd)

    assert res_out is inp
    _assert_empty_resized(inp, dst_shape, dst_spd, dst_dnd, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_uncoalesced(dtype):
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_input(dtype, (4,), _SELECTED_RANGES[0]).to(flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = flag_gems.sparse_resize_and_clear_(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype,value_range", _VALUE_RANGE_PAIRS, ids=_VALUE_RANGE_IDS)
def test_sparse_resize_and_clear_value_ranges(dtype, value_range):
    values = tu.make_input(dtype, (5,), value_range)
    inp = _make_sparse_input((4, 5), 2, 5, dtype, values=values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = flag_gems.sparse_resize_and_clear_(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_RESIZE_DTYPES))
)
def test_sparse_resize_and_clear_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario).repeat(2)[:6]
    indices = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 4, 4]],
        dtype=torch.long,
        device=flag_gems.device,
    )
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = flag_gems.sparse_resize_and_clear_(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_shape_levels(shape, dtype):
    sparse_dim, dense_dim = _split_for_shape(shape)
    inp = _make_sparse_input((4, 5), 2, 3, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(shape), sparse_dim, dense_dim
    )
    res_out = flag_gems.sparse_resize_and_clear_(
        inp, list(shape), sparse_dim, dense_dim
    )

    assert res_out is inp
    _assert_empty_resized(inp, shape, sparse_dim, dense_dim, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("size,sparse_dim,dense_dim", _INVALID_CALLS)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_and_clear_invalid_params(size, sparse_dim, dense_dim, dtype):
    inp = _make_sparse_input((4, 5), 2, 3, dtype)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(
            tu.to_reference(inp),
            size,
            sparse_dim,
            dense_dim,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_resize_and_clear_(inp, size, sparse_dim, dense_dim)


@pytest.mark.sparse_resize_and_clear_
def test_sparse_resize_and_clear_non_sparse_input():
    inp = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(tu.to_reference(inp), [4, 5], 2, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_resize_and_clear_(inp, [4, 5], 2, 0)
