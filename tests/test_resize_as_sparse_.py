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

# aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
# resizes sparse COO ``self`` in place to the template's shape and sparse/dense
# split and returns ``self``; the template contributes sizes only. Native-probed
# exemptions:
# * backward -- the operator registers no derivative ("derivative for
#   aten::resize_as_sparse_ is not implemented").
# * broadcast -- sparse COO has no broadcasting; the template must keep the
#   rank, both split counts and every non-empty extent of ``self``.
# * parameter sweeps -- the schema takes two tensors and has no bool/int/float
#   parameter with a default or an interesting boundary.
# A native probe of this overload accepted every dtype listed here, so there is
# no unsupported-dtype negative case to add.
_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
] + utils.BOOL_TYPES

_DEFAULT_RANGE = ["-1", "1"]

# Columns: self_shape, self_sparse_dim, nnz, coalesced, template_kind,
# template_shape, template_sparse_dim, retains_storage, grows_dense.
_ALL_CASES = [
    pytest.param(
        ((4, 5), 2, 3, True, "sparse", (4, 5), 2, True, False), id="same_size"
    ),
    pytest.param(
        ((4, 5), 2, 3, True, "sparse", (6, 5), 2, True, False), id="grow_sparse"
    ),
    pytest.param(
        ((4, 5), 2, 4, False, "sparse", (6, 5), 2, False, False), id="uncoalesced"
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, True, "sparse", (6, 5, 6), 2, False, False),
        id="grow_sparse_with_dense",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, True, "sparse", (4, 5, 9), 2, False, True),
        id="grow_dense",
    ),
    pytest.param(
        ((4, 5), 2, 0, True, "dense", (6, 7), 0, False, False), id="empty_to_dense"
    ),
    pytest.param(((), 0, 0, True, "sparse", (), 0, False, False), id="rank0_empty"),
    pytest.param(
        ((3, 4, 5), 3, 7, True, "sparse", (4, 4, 5), 3, False, False),
        id="three_sparse_dims",
    ),
    pytest.param(
        ((2, 3, 4, 5), 2, 4, True, "sparse", (2, 5, 4, 5), 2, False, False),
        id="two_dense_dims",
    ),
]

# Columns: self_shape, self_sparse_dim, nnz, template_kind, template_shape,
# template_sparse_dim. Every row is a valid native call for its ``self`` that
# the reference rejects because the template would shrink a non-empty tensor or
# change the sparse/dense split; the candidate must reject it too.
_NEGATIVE_CASES = [
    pytest.param(((4, 5), 2, 3, "sparse", (3, 5), 2), id="shrink_sparse"),
    pytest.param(((4, 5, 6), 2, 3, "sparse", (4, 5, 4), 2), id="shrink_dense"),
    pytest.param(((4, 5), 2, 3, "sparse", (6, 5), 1), id="sparse_dim_count"),
    pytest.param(((4, 5, 6), 2, 3, "sparse", (6, 6), 2), id="dense_dim_count"),
    pytest.param(((4, 5), 2, 3, "dense", (6, 7), 0), id="dense_template"),
]

_NEGATIVE_DTYPES = [torch.float32, torch.int32]


def _sparse_coo(
    shape, sparse_dim, nnz, dtype, value_range, *, coalesced=True, payload=None
):
    """Sparse COO tensor holding ``nnz`` entries over ``shape``.

    Coordinates come from a seeded CPU permutation of the sparse coordinate
    space, so the tensor is well formed without calling ``coalesce()`` -- that
    kernel has no fp8 CUDA implementation and the values here may be fp8.
    """
    dense_shape = tuple(shape[sparse_dim:])
    if nnz == 0:
        indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
        payload = torch.empty((0,) + dense_shape, dtype=dtype, device=flag_gems.device)
    else:
        generator = torch.Generator("cpu").manual_seed(0)
        unique = nnz if coalesced else nnz - 1
        flat = torch.sort(
            torch.randperm(math.prod(shape[:sparse_dim]), generator=generator)[:unique]
        ).values
        if not coalesced:
            # Repeating the first coordinate keeps the tensor non-coalesced.
            flat = torch.cat([flat[:1], flat])
        indices = torch.stack(torch.unravel_index(flat, shape[:sparse_dim]), dim=0).to(
            flag_gems.device
        )
        payload = (
            tu.make_input(dtype, (nnz,) + dense_shape, value_range)
            if payload is None
            else payload
        )
    return torch.sparse_coo_tensor(
        indices, payload, shape, device=flag_gems.device, is_coalesced=coalesced
    )


def _make_template(kind, shape, sparse_dim, dtype):
    """Template tensor; only its shape and sparse/dense split are read."""
    if kind == "dense":
        return torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    return _sparse_coo(shape, sparse_dim, 0, dtype, _DEFAULT_RANGE)


def _assert_resized(res, ref, *, original_values=None):
    assert res.layout == ref.layout
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res.is_coalesced() == ref.is_coalesced()
    assert res._indices().shape == ref._indices().shape
    assert res._values().shape == ref._values().shape
    tu.assert_result_equal(res._indices(), ref._indices())
    if original_values is None:
        tu.assert_result_equal(res._values(), ref._values())
    else:
        # Growing a dense extent reallocates the values block; the reference's
        # own new region is unspecified, so only the retained prefix is defined.
        n = original_values.numel()
        tu.assert_result_equal(res._values().flatten()[:n], ref._values().flatten()[:n])
        tu.assert_result_equal(res._values().flatten()[:n], original_values)


# Every row above is a cheap small-tensor branch (same size, sparse/dense
# growth, duplicate coordinates, empty, rank 0, multi-sparse and multi-dense
# splits), so the quick level keeps all of them; only the larger spec shapes and
# the positive special values stay default-only.
_RESIZE_CASES = tu.selected_cases(_ALL_CASES, quick=_ALL_CASES)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_resize_as_sparse_(case, dtype):
    (
        self_shape,
        self_sparse_dim,
        nnz,
        coalesced,
        kind,
        template_shape,
        template_sparse_dim,
        retains_storage,
        grows_dense,
    ) = case
    inp = _sparse_coo(
        self_shape, self_sparse_dim, nnz, dtype, _DEFAULT_RANGE, coalesced=coalesced
    )
    ref_inp = tu.to_reference(inp)
    template = _make_template(kind, template_shape, template_sparse_dim, dtype)
    ref_template = tu.to_reference(template)
    old_indices_ptr = inp._indices().data_ptr()
    old_values_ptr = inp._values().data_ptr()
    original_values = inp._values().flatten().clone() if grows_dense else None

    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = flag_gems.resize_as_sparse_(inp, template)

    # In-place: the operator returns self and rewrites self's own metadata.
    assert res_out is inp
    assert inp.shape == ref_inp.shape
    if retains_storage:
        # A resize that keeps both extents and the split only rewrites metadata,
        # so the allocator keeps the original index and value blocks.
        assert res_out._indices().data_ptr() == old_indices_ptr
        assert res_out._values().data_ptr() == old_values_ptr
    _assert_resized(res_out, ref_out, original_values=original_values)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_resize_as_sparse_value_ranges(case, dtype, value_range):
    (
        self_shape,
        self_sparse_dim,
        nnz,
        coalesced,
        kind,
        template_shape,
        template_sparse_dim,
        _retains_storage,
        grows_dense,
    ) = case
    inp = _sparse_coo(
        self_shape, self_sparse_dim, nnz, dtype, value_range, coalesced=coalesced
    )
    ref_inp = tu.to_reference(inp)
    template = _make_template(kind, template_shape, template_sparse_dim, dtype)
    ref_template = tu.to_reference(template)
    original_values = inp._values().flatten().clone() if grows_dense else None

    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = flag_gems.resize_as_sparse_(inp, template)

    assert res_out is inp
    _assert_resized(res_out, ref_out, original_values=original_values)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_resize_as_sparse_shape_levels(shape, dtype, value_range):
    # A sparse COO tensor keeps sparse_dim + dense_dim == rank, so one sparse
    # dimension (none for a rank-0 input) keeps the dense tail; the template
    # grows only the outer sparse extent.
    sparse_dim = 1 if shape else 0
    nnz = min(2, shape[0]) if sparse_dim else 0
    template_shape = (shape[0] + 1,) + tuple(shape[1:]) if shape else ()
    inp = _sparse_coo(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)
    template = _make_template("sparse", template_shape, sparse_dim, dtype)
    ref_template = tu.to_reference(template)

    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = flag_gems.resize_as_sparse_(inp, template)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])
)
def test_resize_as_sparse_special_values(dtype, scenario):
    # nnz == 5 with no dense extent keeps the values block 1-D, so the shared
    # special-value payload feeds it directly.
    payload = tu.make_special_input(dtype, scenario)
    inp = _sparse_coo((4, 5), 2, 5, dtype, _DEFAULT_RANGE, payload=payload)
    ref_inp = tu.to_reference(inp)
    template = _make_template("sparse", (6, 5), 2, dtype)
    ref_template = tu.to_reference(template)

    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = flag_gems.resize_as_sparse_(inp, template)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("case", _NEGATIVE_CASES)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_resize_as_sparse_invalid_template(case, dtype):
    self_shape, self_sparse_dim, nnz, kind, template_shape, template_sparse_dim = case
    inp = _sparse_coo(self_shape, self_sparse_dim, nnz, dtype, _DEFAULT_RANGE)
    template = _make_template(kind, template_shape, template_sparse_dim, dtype)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.resize_as_sparse_(inp, template)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_resize_as_sparse_non_tensor_template(dtype):
    # The schema requires a Tensor template; a sequence must not be accepted.
    inp = _sparse_coo((4, 5), 2, 3, dtype, _DEFAULT_RANGE)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        flag_gems.resize_as_sparse_(inp, [6, 5])
