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

"""Correctness tests for the sparse ``values_copy`` operator.

``values_copy`` is the non-aliasing counterpart of ``torch.values``: the result
is a freshly allocated, contiguous strided tensor holding a copy of a sparse
tensor's stored values, so every value assertion here is exact. Compressed
layouts (CSR, CSC, BSR, BSC and their batched forms) are accepted. Dense strided
inputs are rejected everywhere; the COO and nested rejections were measured on the
NVIDIA dispatcher and their rows exist only there. COO additionally needs int64,
since torch always stores its indices as int64.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# complex128 shares the fp64 capability of the backend; int64 index tensors are
# only built when the backend declares int64 support.
_INDEX_DTYPE = torch.int64 if utils.int64_is_supported else torch.int32


def _capability_gated(dtypes):
    """Keep the dtypes this backend declares it supports (static device flags)."""
    gated = []
    for dtype in dtypes:
        if dtype in (torch.float64, torch.complex128) and not utils.fp64_is_supported:
            continue
        if dtype == torch.bfloat16 and not utils.bf16_is_supported:
            continue
        if dtype == torch.int64 and not utils.int64_is_supported:
            continue
        if dtype in _FP8_DTYPES and not utils.fp8_is_supported:
            continue
        gated.append(dtype)
    return gated


_VALUE_DTYPES = _capability_gated(
    tu.REQUIRED_DTYPES
    + [torch.float64, torch.int16, torch.complex64, torch.complex128, torch.bool]
)
# tu.make_input fills bool from 0/1 and ignores the requested value range, so a
# bool row per range would repeat one workload five times; bool keeps its dtype
# coverage in the storage rows of the combined case list instead.
_RANGE_DTYPES = [dtype for dtype in _VALUE_DTYPES if dtype != torch.bool]
_SPECIAL_DTYPES = [dtype for dtype in _VALUE_DTYPES if dtype.is_floating_point]
# Complex gradients are natively unsupported for compressed sparse outputs
# (RuntimeError: "sparse_compressed_tensor does not support automatic
# differentiation for outputs with complex dtype"), so backward is float-only.
_BACKWARD_DTYPES = _capability_gated(
    [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
_OUT_BUFFER_DTYPES = [torch.float32, torch.float16]
# COO (SparseCUDA) and nested (NestedTensorCUDA) have no values_copy kernel on the
# measured NVIDIA dispatcher: NotImplementedError "Could not run
# 'aten::values_copy' with arguments from the 'SparseCUDA' backend" and
# "aten.values_copy.default". Another vendor may register them, so those rows are
# selected at import time from static capability flags instead of being skipped.
# The two are scoped separately: nested is rejected on the measured vendor alone,
# while COO is additionally gated on int64 support because torch stores COO indices
# as int64 no matter what dtype the constructor is given, and selecting that row
# would demand an int64 allocation this backend cannot make.
_REJECTION_ROWS = (
    ([("coo", torch.float32)] if utils.int64_is_supported else [])
    + [("nested", torch.float32)]
    if flag_gems.vendor_name == "nvidia"
    else []
)
# The out dtype must match the stored-values dtype (RuntimeError: "Expected out
# tensor to have dtype ..., but got ... instead").
_MISMATCHED_OUT_DTYPES = [torch.int32] + (
    [torch.float64] if utils.fp64_is_supported else []
)


# Compressed-sparse input construction. A case row is
# (layout, size, nnz, blocks, nbatch, unsorted):
#   layout    "csr"/"csc"/"bsr"/"bsc", the compressed layout;
#   size      (*batch, M, N, *dense) in the compressed layout's own row/column
#             sense, with the batch axes given explicitly by nbatch;
#   nnz       stored values per batch (0 builds the empty-storage cases);
#   blocks    BSR/BSC block shape, None for the element layouts;
#   nbatch    number of leading batch axes (0 for a single sparse matrix);
#   unsorted  reverse the within-group index order, which makes the storage
#             unsorted and must be copied verbatim.
_LAYOUTS = {
    "csr": torch.sparse_csr_tensor,
    "csc": torch.sparse_csc_tensor,
    "bsr": torch.sparse_bsr_tensor,
    "bsc": torch.sparse_bsc_tensor,
}


def _sparse_layout(layout, size, nnz, blocks, nbatch):
    """Validate a case row and return (batch, groups, bound, dense, values shape).

    ``groups`` is the number of compressed groups (rows for CSR/BSR, columns for
    CSC/BSC) and ``bound`` the largest index a group's plain index may take, so
    ``groups * bound`` is the capacity of the layout. The values shape keeps the
    requested rank: batch axes, the nnz axis, the BSR/BSC block dims and the dense
    trailing dims in that order.
    """
    assert layout in _LAYOUTS, f"unknown sparse layout: {layout!r}"
    assert size and all(isinstance(extent, int) for extent in size), size
    assert nbatch >= 0 and len(size) >= nbatch + 2, (size, nbatch)
    batch = tuple(size[:nbatch])
    m, n = size[nbatch], size[nbatch + 1]
    dense = tuple(size[nbatch + 2 :])
    if layout in ("bsr", "bsc"):
        block_rows, block_cols = blocks
        assert m % block_rows == 0 and n % block_cols == 0, (layout, size, blocks)
        if layout == "bsr":
            groups, bound = m // block_rows, n // block_cols
        else:
            groups, bound = n // block_cols, m // block_rows
    elif layout == "csc":
        groups, bound = n, m
    else:
        groups, bound = m, n
    assert 0 <= nnz <= groups * bound, f"{layout} {size} cannot store {nnz} values"
    if groups == 0 or bound == 0:
        assert nnz == 0, f"{layout} {size} has no room for {nnz} values"
    values_shape = batch + (nnz,) + tuple(blocks or ()) + dense
    return batch, groups, bound, dense, values_shape


def _sparse_indices(layout, size, nnz, blocks, nbatch, unsorted):
    """Build the compressed and plain index tensors of one case row on the device."""
    batch, groups, bound, dense, _ = _sparse_layout(layout, size, nnz, blocks, nbatch)
    del bound, dense
    device = flag_gems.device
    index_dtype = _INDEX_DTYPE
    if groups == 0:
        # Zero-extent storage: one empty compressed-offset vector and no plain
        # index, which is the only well-formed empty structure of the layout.
        compressed = torch.zeros(1, dtype=index_dtype, device=device)
        plain = torch.empty(0, dtype=index_dtype, device=device)
    else:
        # Evenly spread the stored values over the groups, so the plain indices
        # stay distinct and inside ``bound`` even for a full-density row.
        counts = torch.full((groups,), nnz // groups, dtype=index_dtype, device=device)
        counts[: nnz % groups] += 1
        # cumsum promotes int32 to int64 by default, which would make every index
        # tensor wider than the dtype this backend declares; pin it.
        compressed = torch.cat(
            [
                torch.zeros(1, dtype=index_dtype, device=device),
                counts.cumsum(0, dtype=index_dtype),
            ]
        )
        plain = torch.arange(nnz, device=device, dtype=index_dtype) - (
            torch.repeat_interleave(compressed[:-1], counts)
        )
        if unsorted:
            plain = torch.repeat_interleave(counts, counts) - 1 - plain
    if batch:
        compressed = compressed.expand(*batch, -1).contiguous()
        plain = plain.expand(*batch, -1).contiguous()
    return compressed, plain


def _stored_shape(layout, size, nnz, blocks, nbatch, unsorted):
    del unsorted
    return _sparse_layout(layout, size, nnz, blocks, nbatch)[4]


def _make_sparse(*case, dtype, value_range=("-1", "1"), values=None):
    """Build the sparse input of one case row on the device of its values."""
    layout, size, nnz, blocks, nbatch, unsorted = case
    values_shape = _stored_shape(*case)
    if values is None:
        values = tu.make_input(dtype, values_shape, list(value_range))
    compressed, plain = _sparse_indices(layout, size, nnz, blocks, nbatch, unsorted)
    return _LAYOUTS[layout](
        compressed, plain, values, size=tuple(size), device=values.device
    )


def _assert_copy_result(res_out, ref_out, inp):
    """A fresh contiguous copy, never a view sharing the input values' storage."""
    if res_out.numel():
        assert res_out.untyped_storage().data_ptr() != (
            inp.values().untyped_storage().data_ptr()
        )
    assert res_out.is_contiguous()
    tu.assert_result_equal(res_out, ref_out)


def _rejection_input(kind, dtype):
    """Build one measured-rejection input.

    A COO tensor always holds its indices as int64, whatever dtype the constructor
    receives, so the structural int64 requirement of this row is not something
    ``_INDEX_DTYPE`` can avoid; the row is selected only where int64 exists.
    """
    if kind == "coo":
        values = tu.make_input(dtype, (2,), ["-1", "1"])
        # The required int64 metadata is allocated directly on the target device
        # rather than built on the host and transferred.
        indices = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.int64, device=flag_gems.device
        )
        return torch.sparse_coo_tensor(
            indices, values, size=(2, 2), device=flag_gems.device
        ).coalesce()
    return torch.nested.nested_tensor(
        [
            torch.zeros((2, 3), dtype=dtype, device=flag_gems.device),
            torch.zeros((2, 4), dtype=dtype, device=flag_gems.device),
        ]
    )


# Case rows shared by the copy, out and range grids.
def _shape_to_row(shape):
    """Map one spec shape level onto a compressed-sparse case row.

    Stored values of a compressed layout are never 0-dim, so the 0-dim level
    cannot carry its own stored rank and uses a single stored value instead. Every
    other level keeps the requested scale: the compressed groups of one matrix
    store a full row (or block row), the dense trailing dims keep the requested
    rank, and rank >= 5 uses two batch axes so the batch axes are exercised too.
    The stored element count of each row equals the logical element count of the
    requested shape.
    """
    if len(shape) == 0:
        return ("csr", (1, 1), 1, None, 0, False)
    if len(shape) == 1:
        return ("csr", (shape[0], 1), shape[0], None, 0, False)
    if len(shape) == 2:
        return ("csr", tuple(shape), shape[0] * shape[1], None, 0, False)
    nbatch = 2 if len(shape) >= 5 else 1
    return (
        "csr",
        tuple(shape),
        shape[nbatch] * shape[nbatch + 1],
        None,
        nbatch,
        False,
    )


# Layout/rank/emptiness rows: stored values of rank 1..5, dense trailing dims,
# zero-nnz and zero-extent storage, BSR/BSC blocks, CSC, multiple batch axes and
# one row whose stored indices are unsorted.
_STORAGE_CASES = tu.selected_cases(
    [
        ("csr", (5, 4), 6, None, 0, False),
        ("csr", (4, 1), 3, None, 0, False),
        ("csr", (1, 5), 2, None, 0, False),
        ("csr", (8, 8), 16, None, 0, False),
        ("csr", (256, 1), 256, None, 0, False),
        ("csr", (3, 3), 0, None, 0, False),
        ("csr", (16, 16, 64), 256, None, 1, False),
        ("csr", (32, 32, 4, 4), 128, None, 1, False),
        ("csr", (3, 3), 4, None, 0, True),
        ("csr", (2, 6, 8), 12, None, 1, False),
        ("csr", (2, 19, 1, 7), 19, None, 1, False),
        ("csr", (2, 3, 4, 6, 7), 24, None, 2, False),
        ("csc", (5, 4), 6, None, 0, False),
        ("csc", (2, 6, 8), 12, None, 1, False),
        ("bsr", (4, 6), 4, (2, 2), 0, False),
        ("bsr", (6, 6), 6, (3, 2), 0, False),
        ("bsr", (4, 6), 0, (2, 2), 0, False),
        ("bsr", (12, 12), 12, (3, 4), 0, False),
        ("bsr", (2, 8, 12), 6, (4, 4), 1, False),
        ("bsr", (4, 20, 10, 3), 5, (5, 5), 1, False),
        ("bsc", (6, 4), 6, (2, 2), 0, False),
        ("bsc", (6, 4), 6, (2, 2), 0, True),
        ("bsc", (2, 6, 4), 6, (2, 2), 1, False),
        ("bsc", (4, 20, 10, 3), 8, (5, 5), 1, False),
        ("bsc", (2, 3, 6, 4), 6, (2, 2), 2, False),
        ("csr", (0, 5), 0, None, 0, False),
        ("csr", (0, 5, 4), 0, None, 0, False),
        ("csr", (2, 0, 5), 0, None, 1, False),
        ("csc", (5, 0), 0, None, 0, False),
        ("csc", (0, 5), 0, None, 0, False),
        ("bsr", (0, 6), 0, (2, 2), 0, False),
        ("bsr", (4, 0), 0, (2, 2), 0, False),
        ("bsc", (6, 0), 0, (2, 2), 0, False),
        ("bsc", (4, 0), 0, (2, 2), 0, False),
    ],
    quick=[
        ("csr", (8, 8), 16, None, 0, False),
        ("csr", (3, 3), 0, None, 0, False),
        ("csr", (3, 3), 4, None, 0, True),
        ("bsr", (2, 8, 12), 6, (4, 4), 1, False),
    ],
)

# One workload per row: the shape levels x the five value ranges over the dtypes
# that carry a range, then the storage rows over every supported dtype.
_COPY_ROWS = [
    (row, dtype, list(value_range))
    for row in [_shape_to_row(shape) for shape in tu.selected_shapes()]
    for dtype in _RANGE_DTYPES
    for value_range in tu.selected_ranges()
] + [(row, dtype, ["-1", "1"]) for row in _STORAGE_CASES for dtype in _VALUE_DTYPES]

# Positive NaN/Inf workloads are default-only.
_SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])

# The copy is the identity on the stored values, so the gradient must be exactly
# the upstream gradient; both an element and a block layout are covered.
_BACKWARD_CASES = tu.selected_cases(
    [("csr", (8, 8), 16, None, 0, False), ("bsr", (4, 6), 4, (2, 2), 0, False)],
    quick=[("csr", (8, 8), 16, None, 0, False)],
)

# Stored values may live in a strided or offset view of a larger buffer. These
# and the out-buffer families are positive supplements, so they stay default-only
# and the quick subset keeps its original baseline.
_VIEW_CASES = tu.selected_cases(
    [("strided", dtype) for dtype in _VALUE_DTYPES]
    + [("offset", dtype) for dtype in _VALUE_DTYPES],
    quick=[],
)

# Out buffers at a nonzero storage offset (contiguous slice, and a stride-2 view),
# each with sentinel neighbours and an empty-storage row.
_OFFSET_OUT_CASES = tu.selected_cases(
    [
        ("csr", (2, 3), 4, None, 0, False),
        ("csr", (2, 3), 0, None, 0, False),
        ("bsr", (4, 6), 4, (2, 2), 0, False),
        ("bsr", (4, 6), 0, (2, 2), 0, False),
    ],
    quick=[],
)
_STRIDED_OUT_CASES = tu.selected_cases(
    [("csr", (2, 3), 4, None, 0, False), ("bsr", (4, 6), 4, (2, 2), 0, False)],
    quick=[],
)


@pytest.mark.values_copy
@pytest.mark.parametrize("case,dtype,value_range", _COPY_ROWS)
def test_values_copy(case, dtype, value_range):
    inp = _make_sparse(*case, dtype=dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("case,dtype,value_range", _COPY_ROWS)
def test_values_copy_out(case, dtype, value_range):
    inp = _make_sparse(*case, dtype=dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    shape = _stored_shape(*case)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    tu.assert_result_equal(out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("kind,dtype", _VIEW_CASES)
def test_values_copy_values_view(kind, dtype):
    """A strided or offset stored-values view is copied faithfully and not modified."""
    nnz = 4
    base = tu.make_input(dtype, (4 * nnz + 8,), ["-1", "1"])
    values = base[::2][:nnz] if kind == "strided" else base[3 : 3 + nnz]
    inp = _make_sparse("csr", (nnz, 1), nnz, None, 0, False, dtype=dtype, values=values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype", _OUT_BUFFER_DTYPES)
@pytest.mark.parametrize("case", _OFFSET_OUT_CASES)
def test_values_copy_out_into_offset_buffer(case, dtype):
    """A contiguous out slice at a nonzero storage offset keeps its neighbours."""
    inp = _make_sparse(*case, dtype=dtype)
    ref_inp = tu.to_reference(inp)

    sentinel = 3.5
    values_shape = inp.values().shape
    numel = inp.values().numel()
    buffer = torch.full((numel + 8,), sentinel, dtype=dtype, device=flag_gems.device)
    ref_buffer = torch.full_like(buffer, sentinel, device=ref_inp.device)
    # The slice is a contiguous view at offset 3; reshaping it to the native values
    # shape keeps that offset and states the real out contract, so the write cannot
    # be accepted through the deprecated implicit resize of the out tensor.
    out = buffer[3 : 3 + numel].reshape(values_shape)
    ref_out = ref_buffer[3 : 3 + numel].reshape(values_shape)
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    assert out.storage_offset() == 3 and out.is_contiguous()
    assert out.shape == values_shape
    # The whole buffer, so the sentinel neighbours must survive the write.
    tu.assert_result_equal(buffer, ref_buffer)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype", _OUT_BUFFER_DTYPES)
@pytest.mark.parametrize("case", _STRIDED_OUT_CASES)
def test_values_copy_out_into_strided_view(case, dtype):
    """A genuinely strided out view is filled in place and never touches neighbours."""
    inp = _make_sparse(*case, dtype=dtype)
    ref_inp = tu.to_reference(inp)

    sentinel = 3.5
    shape = _stored_shape(*case)
    parent = torch.full(
        (2 * shape[0],) + tuple(shape[1:]),
        sentinel,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_parent = torch.full_like(parent, sentinel, device=ref_inp.device)
    out, ref_out = parent[1::2], ref_parent[1::2]
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    # Same parent storage and the native stored-values shape: identical data alone
    # cannot reject a return that was reshaped instead of filled in place.
    assert tuple(res_ret.shape) == tuple(ref_out.shape) == tuple(inp.values().shape)
    assert out.stride() == ref_out.stride()
    # Exact whole-parent comparison: written slice and sentinel neighbours.
    tu.assert_result_equal(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_values_copy_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    inp = _make_sparse(
        "csr", (2, 3), values.numel(), None, 0, False, dtype=dtype, values=values
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_values_copy_out_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    inp = _make_sparse(
        "csr", (2, 3), values.numel(), None, 0, False, dtype=dtype, values=values
    )
    ref_inp = tu.to_reference(inp)

    out = torch.empty_like(values)
    ref_out = torch.empty_like(out, device=ref_inp.device)
    torch.ops.aten.values_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.values_copy(inp, out=out)

    assert res_ret is out
    tu.assert_result_equal(out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("case", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_values_copy_backward(case, dtype):
    """The copy is the identity, so the stored values receive the exact upstream."""
    values = tu.make_input(dtype, _stored_shape(*case), ["-1", "1"])
    values = values.requires_grad_(True)
    inp = _make_sparse(*case, dtype=dtype, values=values)

    ref_values = tu.to_reference(values).detach().requires_grad_(True)
    ref_inp = _make_sparse(*case, dtype=dtype, values=ref_values)

    ref_out = torch.ops.aten.values_copy(ref_inp)
    res_out = flag_gems.values_copy(inp)

    _assert_copy_result(res_out, ref_out, inp)

    # Nonuniform upstream, so a swapped or partially reduced gradient cannot pass.
    upstream = tu.make_input(dtype, res_out.shape, ["-1", "1"]) * 3.0 + 1.0
    ref_upstream = tu.to_reference(upstream).detach()
    ref_grad = torch.autograd.grad(ref_out, ref_values, grad_outputs=ref_upstream)[0]
    res_grad = torch.autograd.grad(res_out, values, grad_outputs=upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.values_copy
def test_values_copy_rejects_dense_input():
    # native: RuntimeError: values expected sparse tensor layout but got Strided
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(inp)


@pytest.mark.values_copy
@pytest.mark.parametrize("kind,dtype", _REJECTION_ROWS)
def test_values_copy_rejects_vendor_unsupported_layout(kind, dtype):
    inp = _rejection_input(kind, dtype)

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(inp)


@pytest.mark.values_copy
def test_values_copy_rejects_non_tensor_input():
    # native: RuntimeError: failed to match any schema with overloads None
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(3.14)


@pytest.mark.values_copy
@pytest.mark.parametrize("mismatched", _MISMATCHED_OUT_DTYPES)
def test_values_copy_out_rejects_mismatched_dtype(mismatched):
    # native: RuntimeError: Expected out tensor to have dtype float, but got ...
    inp = _make_sparse("csr", (4, 5), 4, None, 0, False, dtype=torch.float32)
    out = torch.empty(
        _stored_shape("csr", (4, 5), 4, None, 0, False),
        dtype=mismatched,
        device=flag_gems.device,
    )

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values_copy(inp, out=out)
