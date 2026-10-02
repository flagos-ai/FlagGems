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

"""Correctness tests for ``aten::_sparse_csr_tensor_unsafe``.

Schema::

    _sparse_csr_tensor_unsafe(Tensor crow_indices, Tensor col_indices, Tensor values,
                              int[] size, *, ScalarType? dtype=None, Layout? layout=None,
                              Device? device=None, bool? pin_memory=None) -> Tensor

The factory stores its three components by reference and validates no metadata, so it
also accepts descriptors that ``sparse_csr_tensor`` rejects: short or over-long row
pointers, out-of-range column indices, more stored entries than the last row pointer
reports, unsorted columns, or a rank-0/rank-1 ``size``. The contract checked here is
therefore which descriptor is accepted, the reported size/layout/dtype/nnz, and that
the stored components are the caller's own storage. Densifying any of these tensors
would dereference unchecked indices, so no test calls ``to_dense()``.

Coverage notes, each verified against the native oracle on the active backend:
  * the value grid keeps only shapes of rank >= 2 because a CSR row block needs the
    two trailing matrix extents; rank-0/rank-1 ``size`` values remain covered by the
    ``size`` rows of ``_UNCHECKED_CASES``;
  * no broadcast workload: the operator has no broadcasting operands;
  * no backward workload: the result neither requires grad nor has a grad_fn
    (``autograd.grad`` over the stored values reports "element 0 of tensors does not
    require grad and does not have a grad_fn");
  * no tensor/scalar operand pair: every operand is a tensor or the int list ``size``;
  * ``dtype`` defaults to float32, so the value grid requests the values dtype
    explicitly while ``..._optional_keywords`` keeps the omitted-argument call form.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]

# Values dtypes the factory accepts once `dtype=` matches. Everything except fp64
# was probed directly; fp64 uses the shared support flag.
_VALUE_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
    torch.bool,
]
if utils.fp64_is_supported:
    _VALUE_DTYPES.insert(3, torch.float64)
if utils.fp8_is_supported:
    _VALUE_DTYPES += _FP8_DTYPES

# The crow_indices / col_indices dtype is not restricted either: the caller's dtype
# is kept verbatim, integer or not (all four probed as preserved).
_INDEX_DTYPES = [torch.int8, torch.int32, torch.int64, torch.float32]
_STRUCTURE_DTYPES = [torch.float32, torch.int32, torch.uint8]

_SPECIAL_DTYPES = list(utils.ALL_FLOAT_DTYPES)
if utils.fp8_is_supported:
    _SPECIAL_DTYPES += _FP8_DTYPES

# Accelerator-only negatives are defined behind this import-time gate, so a CPU-only
# backend neither collects them nor touches a device to decide.
_ACCELERATOR = flag_gems.device != "cpu"

# A CSR size carries the two trailing matrix extents; the 0-dim and 1-dim spec shapes
# cannot express a row block, so those ranks are covered through the rank-0/rank-1
# `size` rows of _UNCHECKED_CASES instead.
_CSR_SHAPES = [size for size in tu.selected_shapes() if len(size) >= 2]

# Each row is (size, crow, col, values): genuine CSR descriptors plus the empty and
# degenerate ones. Values are stored verbatim, so they only have to be finite.
_STRUCTURE_CASES = [
    ((4, 4), [0, 2, 4, 4, 4], [0, 1, 0, 1], [0.5, -1.5, 2.0, -2.0], ()),
    ((5, 4), [0, 2, 3, 3, 5, 5], [0, 1, 2, 0, 3], [1.0, -2.0, 3.0, -4.0, 5.0], ()),
    ((3, 3), [0, 3, 4, 6], [0, 1, 2, 0, 1, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ()),
    ((6, 1), [0, 1, 1, 2, 2, 3, 3], [0, 0, 0], [1.0, 2.0, 3.0], ()),
    ((1, 3), [0, 3], [0, 1, 2], [1.0, 2.0, 3.0], ()),
    ((4, 5), [0, 0, 0, 0, 0], [], [], ()),
    ((3, 0), [0, 0, 0, 0], [], [], ()),
    ((0, 4), [0], [], [], ()),
    ((0, 0), [0], [], [], ()),
    (
        (4, 4, 2),
        [0, 2, 3, 3, 5],
        [0, 1, 0, 1, 0],
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0],
        (2,),
    ),
    (
        (3, 5, 3),
        [0, 1, 2, 3, 4, 4],
        [0, 2, 1, 0],
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0],
        (3,),
    ),
]
# Quick keeps one row per cheap branch on small inputs: a plain 2-D block, the
# empty-values, zero-width, zero-row and dense-tail shapes.
_STRUCTURE_QUICK = _STRUCTURE_CASES

# Batched row blocks: each batch entry owns one row-pointer block of length rows + 1.
_BATCHED_CASES = [
    (
        (2, 3, 4),
        [[0, 2, 3, 4], [0, 1, 3, 4]],
        [[0, 1, 0, 2], [1, 0, 2, 3]],
        [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]],
        (),
    ),
    (
        (2, 5, 4),
        [[0, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 6]],
        [[0, 1, 3, 0, 2, 1], [3, 2, 0, 1, 0, 3]],
        [[1.0, -1.0, 2.0, -2.0, 3.0, -3.0], [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0]],
        (),
    ),
]
_BATCHED_QUICK = _BATCHED_CASES

# Descriptors whose metadata is unchecked: accepted (and then reported) verbatim.
# `size` is the exact container handed to both the reference and the candidate.
_UNCHECKED_CASES = [
    {
        "label": "short_crow",
        "size": (4, 4),
        "crow": [0, 2, 4, 4],
        "col": [0, 1, 0, 1],
        "values": [1.0, 2.0, 3.0, 4.0],
    },
    {
        "label": "long_crow",
        "size": (4, 4),
        "crow": [0, 2, 4, 4, 4, 4],
        "col": [0, 1, 0, 1],
        "values": [1.0, 2.0, 3.0, 4.0],
    },
    {
        "label": "col_out_of_range",
        "size": (3, 4),
        "crow": [0, 2, 4, 4],
        "col": [0, 5, 7, 1],
        "values": [1.0, 2.0, 3.0, 4.0],
    },
    {
        "label": "nnz_beyond_last_row_pointer",
        "size": (3, 4),
        "crow": [0, 1, 2, 3],
        "col": [0, 1, 2, 3],
        "values": [1.0, 2.0, 3.0, 4.0],
    },
    {
        "label": "unsorted_col",
        "size": (3, 4),
        "crow": [0, 2, 4, 4],
        "col": [2, 0, 3, 1],
        "values": [1.0, 2.0, 3.0, 4.0],
    },
    {
        "label": "float_crow",
        "size": (3, 4),
        "crow": [0.0, 2.0, 4.0, 4.0],
        "col": [0, 1, 2, 3],
        "values": [1.0, 2.0, 3.0, 4.0],
        "index_dtype": torch.float32,
    },
    {"label": "rank0_size", "size": [], "crow": [0], "col": [], "values": []},
    {"label": "rank1_size", "size": [256], "crow": [0], "col": [], "values": []},
    {
        "label": "tuple_size",
        "size": (4, 4),
        "crow": [0, 2, 4, 4, 4],
        "col": [0, 1, 0, 1],
        "values": [1.0, 2.0, 3.0, 4.0],
        "size_is_tuple": True,
    },
]
_UNCHECKED_QUICK = _UNCHECKED_CASES

# Optional keywords: the omitted form is what proves the schema default, the others
# are accepted no-ops next to it.
_OPTIONAL_KEYWORD_CASES = [
    {},
    {"dtype": torch.float32},
    {"layout": torch.sparse_csr},
    {"pin_memory": False},
]


def _components(crow, col, payload, dtype, *, index_dtype=torch.int32, dense_shape=()):
    """Build the three component tensors of one descriptor on the test device."""
    crow_indices = torch.tensor(crow, dtype=index_dtype, device=flag_gems.device)
    col_indices = torch.tensor(col, dtype=index_dtype, device=flag_gems.device)
    # Built at float32 and cast: a negative literal is out of range for the unsigned
    # dtypes and torch.tensor would refuse it, while the cast defines the wrap. Both
    # the oracle and the candidate receive this same stored tensor.
    values = torch.tensor(payload, dtype=torch.float32, device=flag_gems.device).to(
        dtype
    )
    if dense_shape:
        values = values.reshape((-1,) + tuple(dense_shape))
    return crow_indices, col_indices, values


def _reference(size, crow_indices, col_indices, values, *, dtype):
    """Native oracle over independently cloned components.

    ``size`` is passed through unchanged: the container type is part of the call
    form under test. ``to_reference`` keeps offsets and strides, so a component
    view arrives in the oracle as the same view.
    """
    ref_crow = tu.to_reference(crow_indices)
    ref_col = tu.to_reference(col_indices)
    ref_values = tu.to_reference(values)
    return torch.ops.aten._sparse_csr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        size,
        dtype=dtype,
        device=ref_values.device,
    )


def _assert_stored_components(out, crow_indices, col_indices, values):
    """Each component is the caller's buffer, reached through a new wrapper.

    The three properties checked are storage aliasing (same storage pointer), the
    wrapper being a distinct object, and the metadata of the stored view - offset,
    stride, size and dtype - surviving unchanged.
    """
    for stored, original in (
        (out.crow_indices(), crow_indices),
        (out.col_indices(), col_indices),
        (out.values(), values),
    ):
        assert stored is not original
        assert (
            stored.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        )
        assert stored.storage_offset() == original.storage_offset()
        assert tuple(stored.stride()) == tuple(original.stride())
        assert tuple(stored.shape) == tuple(original.shape)
        assert stored.dtype == original.dtype
        assert stored.device == original.device


def _assert_matches_reference(res, ref):
    """Metadata and stored components against the native result."""
    assert res.layout == ref.layout
    assert res.dtype == ref.dtype
    assert tuple(res.shape) == tuple(ref.shape)
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res.numel() == ref.numel()
    assert res._nnz() == ref._nnz()
    tu.assert_result_equal(res.crow_indices(), ref.crow_indices())
    tu.assert_result_equal(res.col_indices(), ref.col_indices())
    tu.assert_result_equal(res.values(), ref.values())


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize("size", _CSR_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__sparse_csr_tensor_unsafe_values(size, value_range, dtype):
    dense = tu.make_input(dtype, size, value_range)
    batch, rows = tuple(size[:-2]), size[-2]
    crow_indices = torch.arange(rows + 1, dtype=torch.int32, device=flag_gems.device)
    col_indices = torch.zeros(rows, dtype=torch.int32, device=flag_gems.device)
    if batch:
        crow_indices = crow_indices.expand(batch + (rows + 1,)).contiguous()
        col_indices = col_indices.expand(batch + (rows,)).contiguous()
    # One stored entry per row, addressed in every row's first column: a genuine
    # CSR layout whose values are the first column of the range-filled input.
    values = dense[..., :, 0]
    size_arg = list(size)

    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=dtype)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=dtype,
        device=flag_gems.device,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize(
    "case", tu.selected_cases(_STRUCTURE_CASES, quick=_STRUCTURE_QUICK)
)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _STRUCTURE_DTYPES)
def test__sparse_csr_tensor_unsafe_structure(case, index_dtype, dtype):
    size, crow, col, payload, dense_shape = case
    crow_indices, col_indices, values = _components(
        crow, col, payload, dtype, index_dtype=index_dtype, dense_shape=dense_shape
    )

    size_arg = list(size)
    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=dtype)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=dtype,
        device=flag_gems.device,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)
    assert res.dtype == dtype
    assert res.crow_indices().dtype == index_dtype
    assert res.col_indices().dtype == index_dtype


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize(
    "case", tu.selected_cases(_BATCHED_CASES, quick=_BATCHED_QUICK)
)
@pytest.mark.parametrize("dtype", _STRUCTURE_DTYPES)
def test__sparse_csr_tensor_unsafe_batched(case, dtype):
    size, crow, col, payload, dense_shape = case
    crow_indices, col_indices, values = _components(
        crow, col, payload, dtype, dense_shape=dense_shape
    )
    size_arg = list(size)
    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=dtype)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=dtype,
        device=flag_gems.device,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)
    # A batched descriptor keeps its leading batch extent in `sparse_dim()`.
    assert res.sparse_dim() == ref.sparse_dim()


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize(
    "case", tu.selected_cases(_UNCHECKED_CASES, quick=_UNCHECKED_QUICK)
)
def test__sparse_csr_tensor_unsafe_unchecked_metadata(case):
    size = tuple(case["size"])
    size_arg = tuple(size) if case.get("size_is_tuple") else list(size)
    index_dtype = case.get("index_dtype", torch.int32)
    crow_indices, col_indices, values = _components(
        case["crow"],
        case["col"],
        case["values"],
        torch.float32,
        index_dtype=index_dtype,
    )

    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=torch.float32)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=torch.float32,
        device=flag_gems.device,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test__sparse_csr_tensor_unsafe_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    rows = values.shape[0]
    crow_indices = torch.arange(rows + 1, dtype=torch.int32, device=flag_gems.device)
    col_indices = torch.zeros(rows, dtype=torch.int32, device=flag_gems.device)
    size = (rows, rows)

    size_arg = list(size)
    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=dtype)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=dtype,
        device=flag_gems.device,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize("kwargs", _OPTIONAL_KEYWORD_CASES)
def test__sparse_csr_tensor_unsafe_optional_keywords(kwargs):
    crow_indices, col_indices, values = _components(
        [0, 2, 4, 4, 4], [0, 1, 0, 1], [1.0, 2.0, 3.0, 4.0], torch.float32
    )
    size = (4, 4)

    size_arg = list(size)
    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=torch.float32)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        device=flag_gems.device,
        **kwargs,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)
    # The omitted form is the schema default: float32 regardless of the caller.
    assert res.dtype == torch.float32


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test__sparse_csr_tensor_unsafe_keeps_component_dtypes(index_dtype):
    crow_indices, col_indices, values = _components(
        [0, 2, 4, 4, 4],
        [0, 1, 0, 1],
        [1.0, 2.0, 3.0, 4.0],
        torch.float32,
        index_dtype=index_dtype,
    )
    size = (4, 4)

    size_arg = list(size)
    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=torch.float32)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=torch.float32,
        device=flag_gems.device,
    )

    # The stored-component contract includes the index dtype.
    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)


@pytest.mark.sparse_csr_tensor_unsafe
def test__sparse_csr_tensor_unsafe_stores_component_views():
    # Components that are views of a larger buffer keep their offset and strides:
    # the factory compacts nothing and copies nothing.
    buffer = torch.arange(2 * 4 + 2, dtype=torch.float32, device=flag_gems.device)
    values = buffer[3::2]
    crow_buffer = torch.zeros(2 * 4 + 1, dtype=torch.int32, device=flag_gems.device)
    crow_buffer[::2] = torch.arange(5, dtype=torch.int32, device=flag_gems.device)
    crow_indices = crow_buffer[::2]
    col_buffer = torch.zeros(2 * 4, dtype=torch.int32, device=flag_gems.device)
    col_indices = col_buffer[::2]
    size = (4, 4)

    size_arg = list(size)
    ref = _reference(size_arg, crow_indices, col_indices, values, dtype=torch.float32)
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=torch.float32,
        device=flag_gems.device,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)


@pytest.mark.sparse_csr_tensor_unsafe
def test__sparse_csr_tensor_unsafe_aliases_component_storage():
    crow_indices, col_indices, values = _components(
        [0, 2, 4, 4, 4], [0, 1, 0, 1], [1.0, 2.0, 3.0, 4.0], torch.float32
    )
    size = (4, 4)
    ref_crow = tu.to_reference(crow_indices)
    ref_col = tu.to_reference(col_indices)
    ref_values = tu.to_reference(values)

    size_arg = list(size)
    ref = torch.ops.aten._sparse_csr_tensor_unsafe(
        ref_crow,
        ref_col,
        ref_values,
        size_arg,
        dtype=torch.float32,
        device=ref_values.device,
    )
    res = flag_gems._sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        values,
        size_arg,
        dtype=torch.float32,
        device=flag_gems.device,
    )

    _assert_stored_components(res, crow_indices, col_indices, values)
    _assert_matches_reference(res, ref)
    # A write through the caller's buffer is visible in the stored component, on the
    # candidate and on the native result alike (neither one copied).
    values.fill_(7.0)
    ref_values.fill_(7.0)
    tu.assert_result_equal(res.values(), ref.values())
    assert bool((res.values() == 7.0).all())
    # The three components stay three independent buffers.
    pointers = {
        res.crow_indices().untyped_storage().data_ptr(),
        res.col_indices().untyped_storage().data_ptr(),
        res.values().untyped_storage().data_ptr(),
    }
    assert len(pointers) == 3


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize("layout", [torch.sparse_bsr, torch.sparse_csc, torch.strided])
def test__sparse_csr_tensor_unsafe_rejects_non_csr_layout(layout):
    # Only the CSR compressed layout is accepted (bsr/csc/strided are not).
    crow_indices, col_indices, values = _components(
        [0, 1, 2], [0, 1], [1.0, 2.0], torch.float32
    )
    with pytest.raises(RuntimeError):
        flag_gems._sparse_csr_tensor_unsafe(
            crow_indices,
            col_indices,
            values,
            [2, 2],
            layout=layout,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.uint8,
        torch.bool,
    ],
)
def test__sparse_csr_tensor_unsafe_rejects_mismatched_dtype(dtype):
    # `dtype` names the sparse tensor's dtype and has to equal the values dtype; the
    # unsafe factory checks this even though it validates no metadata.
    crow_indices, col_indices, values = _components(
        [0, 1, 2], [0, 1], [1.0, 2.0], torch.float32
    )
    with pytest.raises(RuntimeError):
        flag_gems._sparse_csr_tensor_unsafe(
            crow_indices,
            col_indices,
            values,
            [2, 2],
            dtype=dtype,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize(
    "values_dtype",
    [torch.float16, torch.bfloat16, torch.int16, torch.int64, torch.int8, torch.bool],
)
def test__sparse_csr_tensor_unsafe_rejects_values_without_matching_dtype(values_dtype):
    # Omitting `dtype` leaves the sparse tensor at float32, so any other values dtype
    # has to be requested explicitly - and then it has to match.
    crow_indices, col_indices, values = _components(
        [0, 1, 2], [0, 1], [1, 2], values_dtype
    )
    with pytest.raises(RuntimeError):
        flag_gems._sparse_csr_tensor_unsafe(
            crow_indices, col_indices, values, [2, 2], device=flag_gems.device
        )


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize("size", [[2, -1], [-1, 2], [2, -4]])
def test__sparse_csr_tensor_unsafe_rejects_negative_size(size):
    crow_indices, col_indices, values = _components(
        [0, 1, 2], [0, 1], [1.0, 2.0], torch.float32
    )
    with pytest.raises(RuntimeError):
        flag_gems._sparse_csr_tensor_unsafe(
            crow_indices, col_indices, values, size, device=flag_gems.device
        )


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize("position", [0, 1, 2])
def test__sparse_csr_tensor_unsafe_rejects_non_tensor_component(position):
    components = [
        torch.tensor([0, 1, 2], dtype=torch.int32, device=flag_gems.device),
        torch.tensor([0, 1], dtype=torch.int32, device=flag_gems.device),
        torch.tensor([1.0, 2.0], dtype=torch.float32, device=flag_gems.device),
    ]
    components[position] = [0, 1]
    with pytest.raises(RuntimeError):
        flag_gems._sparse_csr_tensor_unsafe(
            components[0],
            components[1],
            components[2],
            [2, 2],
            device=flag_gems.device,
        )


if _ACCELERATOR:

    @pytest.mark.sparse_csr_tensor_unsafe
    def test__sparse_csr_tensor_unsafe_requires_matching_component_device():
        # With accelerator components and no `device=`, the factory would have to
        # place the tensor on a device it cannot derive, so it refuses instead.
        crow_indices, col_indices, values = _components(
            [0, 1, 2], [0, 1], [1.0, 2.0], torch.float32
        )
        with pytest.raises(RuntimeError):
            flag_gems._sparse_csr_tensor_unsafe(
                crow_indices, col_indices, values, [2, 2]
            )

    @pytest.mark.sparse_csr_tensor_unsafe
    @pytest.mark.parametrize("position", [0, 1, 2])
    def test__sparse_csr_tensor_unsafe_rejects_cross_device_components(position):
        crow_indices, col_indices, values = _components(
            [0, 1, 2], [0, 1], [1.0, 2.0], torch.float32
        )
        # A list, because one component is replaced by its CPU copy here.
        components = [crow_indices, col_indices, values]
        components[position] = components[position].cpu()
        with pytest.raises(RuntimeError):
            flag_gems._sparse_csr_tensor_unsafe(
                components[0],
                components[1],
                components[2],
                [2, 2],
                device=flag_gems.device,
            )

    @pytest.mark.sparse_csr_tensor_unsafe
    def test__sparse_csr_tensor_unsafe_rejects_pinned_accelerator_values():
        # pin_memory only applies to dense CPU tensors.
        crow_indices, col_indices, values = _components(
            [0, 1, 2], [0, 1], [1.0, 2.0], torch.float32
        )
        with pytest.raises(RuntimeError):
            flag_gems._sparse_csr_tensor_unsafe(
                crow_indices,
                col_indices,
                values,
                [2, 2],
                device=flag_gems.device,
                pin_memory=True,
            )


@pytest.mark.sparse_csr_tensor_unsafe
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in _VALUE_DTYPES if dtype.is_floating_point or dtype.is_complex],
)
def test__sparse_csr_tensor_unsafe_requires_grad_input(dtype):
    # Unsafe factories alias the payload but do not create an autograd edge.
    compressed = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    plain = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    values_shape = (2,)
    size = [2, 2]
    values = tu.make_input(dtype, values_shape, ["-1", "1"]).requires_grad_()
    ref_out = torch.ops.aten._sparse_csr_tensor_unsafe(
        compressed,
        plain,
        values,
        size,
        dtype=dtype,
        layout=torch.sparse_csr,
        device=values.device,
    )
    res_out = flag_gems._sparse_csr_tensor_unsafe(
        compressed,
        plain,
        values,
        size,
        dtype=dtype,
        layout=torch.sparse_csr,
        device=values.device,
    )

    assert not res_out.requires_grad
    assert res_out.requires_grad == ref_out.requires_grad
    assert res_out.grad_fn is ref_out.grad_fn is None
    assert res_out.values().data_ptr() == values.data_ptr()
    tu.assert_result_equal(res_out.values(), ref_out.values())
