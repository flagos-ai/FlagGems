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

# Resize COO storage in place; non-empty inputs retain their sparse/dense split.
# Empty inputs may change that split.
_RESIZE_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.BOOL_TYPES
)
# (src_shape, sparse_dim, nnz, dst_size, dst_sparse_dim, dst_dense_dim).
_RESIZE_CASES = [
    ((4, 5), 2, 3, [4, 5], 2, 0),
    ((4, 5, 6), 2, 3, [4, 5, 6], 2, 1),
    ((5,), 1, 2, [8], 1, 0),
    ((4, 5), 2, 3, [6, 5], 2, 0),
    ((4, 5), 2, 3, [8, 8], 2, 0),
    ((3, 4, 5), 3, 7, [4, 4, 5], 3, 0),
    ((4, 5, 6), 2, 3, [6, 5, 6], 2, 1),
    ((4, 5), 2, 0, [2, 4, 5], 2, 1),
    ((4, 5), 2, 0, [4, 5, 6], 3, 0),
    ((4, 5), 2, 0, [3, 5], 2, 0),
    ((5,), 1, 0, [2, 3], 2, 0),
]

_VALUE_RANGE_CASES = [
    ((4, 5), 2, 3, [6, 5], 2, 0),
    ((4, 5, 6), 2, 3, [6, 5, 6], 2, 1),
]

_SPARSE_SHAPE_LEVELS = tuple(s for s in tu.selected_shapes() if len(s) > 0)

_NEGATIVE_DTYPES = [torch.float32, torch.int32]

_INVALID_RESIZE_CASES = [
    pytest.param(
        ((4, 5), 2, 3, [4, 5, 6], 2, 0),
        id="invalid_split",
    ),
    pytest.param(
        ((4, 5), 2, 3, [6, 5], 1, 1),
        id="sparse_dim_change_non_empty",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, [6, 5, 6], 1, 2),
        id="dense_dim_change_non_empty",
    ),
    pytest.param(
        ((4, 5), 2, 3, [3, 5], 2, 0),
        id="shrink_sparse_dim_non_empty",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, [4, 5, 4], 2, 1),
        id="shrink_dense_dim_non_empty",
    ),
    pytest.param(
        ((4, 5), 2, 3, [4, -5], 2, 0),
        id="negative_size",
    ),
]


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, values=None):
    # Use seeded unique coordinates; supplied values replace the default payload.
    gen = torch.Generator("cpu").manual_seed(seed)
    dense_shape = tuple(shape[sparse_dim:])
    values_shape = (nnz,) + dense_shape
    num_sparse = math.prod(shape[:sparse_dim])
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(num_sparse, generator=gen, device="cpu")[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if values is None:
        # Generate in a wide dtype then cast: randn/randint do not accept every
        # storage dtype (notably fp8), and random_ cannot sample unsigned
        # ranges that include negative values.
        if dtype == torch.bool:
            values = torch.randint(
                0, 2, values_shape, generator=gen, device="cpu"
            ).bool()
        elif dtype.is_floating_point:
            values = torch.randn(values_shape, generator=gen, device="cpu").to(dtype)
        else:
            # Keep the magnitude small so the values stay valid for every
            # integer storage dtype (int8/int16 included).
            values = torch.randint(-5, 6, values_shape, generator=gen, device="cpu").to(
                dtype
            )
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
        device=flag_gems.device,
    )


def _assert_resized(res, ref):
    assert res.layout == ref.layout
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res.is_coalesced() == ref.is_coalesced()
    tu.assert_result_equal(res._indices(), ref._indices())
    tu.assert_result_equal(res._values(), ref._values())


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_(case, dtype):
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = case
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(
        ref_inp, size, new_sparse_dim, new_dense_dim
    )
    res_out = flag_gems.sparse_resize_(inp, size, new_sparse_dim, new_dense_dim)

    # In-place semantics: the op returns self and mutates self in place.
    assert res_out is inp
    # The mutated input (not only the return value) carries the new structure.
    _assert_resized(res_out, ref_out)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_resize_value_ranges(case, dtype, value_range):
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = case
    dense_shape = tuple(src_shape[sparse_dim:])
    values = tu.make_input(dtype, (nnz,) + dense_shape, value_range)
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype, values=values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(
        ref_inp, size, new_sparse_dim, new_dense_dim
    )
    res_out = flag_gems.sparse_resize_(inp, size, new_sparse_dim, new_dense_dim)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("shape", _SPARSE_SHAPE_LEVELS)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_shape_levels(shape, dtype):
    sparse_dim = 1
    dense_dim = len(shape) - 1
    nnz = min(2, shape[0])
    inp = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, list(shape), sparse_dim, dense_dim)
    res_out = flag_gems.sparse_resize_(inp, list(shape), sparse_dim, dense_dim)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_RESIZE_DTYPES))
)
def test_sparse_resize_nan_inf(dtype, scenario):
    src_shape, sparse_dim, nnz = (4, 5, 6), 2, 4
    values = tu.make_special_input(dtype, scenario).repeat(5)[:24].reshape(nnz, 6)
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype, values=values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, [6, 5, 6], 2, 1)
    res_out = flag_gems.sparse_resize_(inp, [6, 5, 6], 2, 1)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _INVALID_RESIZE_CASES)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_invalid_raises(case, dtype):
    src_shape, sparse_dim, nnz, size, bad_sparse_dim, bad_dense_dim = case
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype)
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_(ref_inp, size, bad_sparse_dim, bad_dense_dim)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_resize_(inp, size, bad_sparse_dim, bad_dense_dim)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_dense_input_rejected(dtype):
    if dtype.is_floating_point:
        inp = torch.randn((4, 5), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 5, (4, 5), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_(tu.to_reference(inp), [6, 5], 2, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_resize_(inp, [6, 5], 2, 0)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_uncoalesced(dtype):
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_input(dtype, (4,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, [6, 5], 2, 0)
    res_out = flag_gems.sparse_resize_(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_resized(res_out, ref_out)
