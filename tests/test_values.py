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

"""Correctness tests for the sparse ``values`` accessor.

``aten::values`` returns an aliasing view of a sparse tensor's stored values,
so every case checks the view contract (leading native shape/stride/storage
offset, shared buffer, write-through both ways, untouched index buffers) plus
the stored values themselves. COO inputs are built coalesced from truthful
sorted, unique indices; the compressed layouts (CSR/CSC/BSR/BSC, batched and
hybrid forms, block sizes in the values shape) have no precondition.

No sparse layout stores a rank-0 values buffer, so the 0-dim spec shape is
realized as a documented singleton matrix (see ``_shape_to_row``).
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_LAYOUTS = {
    "csr": torch.sparse_csr_tensor,
    "csc": torch.sparse_csc_tensor,
    "bsr": torch.sparse_bsr_tensor,
    "bsc": torch.sparse_bsc_tensor,
}
_POINTER_ATTRS = {
    torch.sparse_csr: ("crow_indices", "col_indices"),
    torch.sparse_csc: ("ccol_indices", "row_indices"),
    torch.sparse_bsr: ("crow_indices", "col_indices"),
    torch.sparse_bsc: ("ccol_indices", "row_indices"),
}


def _capability_gated(dtypes):
    """Keep the dtypes whose kernels this backend declares it supports."""
    gated = []
    for dtype in dtypes:
        if dtype == torch.bfloat16 and not utils.bf16_is_supported:
            continue
        if dtype == torch.int64 and not utils.int64_is_supported:
            continue
        if dtype in _FP8_DTYPES and not utils.fp8_is_supported:
            continue
        if (dtype in (torch.float64, torch.complex128)) and not utils.fp64_is_supported:
            continue
        gated.append(dtype)
    return gated


# int8/uint8/fp8 are required where the backend supports them; float64 and the
# complex types are extra. No separate flag reports complex support;
# complex128 follows fp64; complex64 remains available.
_VALUE_DTYPES = _capability_gated(
    tu.REQUIRED_DTYPES + [torch.float64, torch.complex64, torch.complex128, torch.bool]
)
_SPECIAL_DTYPES = [dtype for dtype in _VALUE_DTYPES if dtype.is_floating_point]
# COO stores its coordinates as int64.
_COO_DTYPES = _VALUE_DTYPES if utils.int64_is_supported else []
_INDEX_DTYPE = torch.int64 if utils.int64_is_supported else torch.int32
# The accessor's gradient is the identity on the stored values. Measured
# support: the compressed layouts reject complex backward
# ('sparse_compressed_tensor does not support automatic differentiation for
# outputs with complex dtype'), and fp8 gradients need an fp8 index_add, so
# neither becomes fake coverage.
_FLOAT_BACKWARD_DTYPES = _capability_gated(
    [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
_COMPLEX_BACKWARD_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)


def _compressed_meta(layout, size, nnz, blocks, nbatch):
    """(batch, groups, bound, values_shape) of a compressed case row.

    ``groups`` counts the compressed groups (rows for CSR/BSR, columns for
    CSC/BSC) and ``bound`` is the exclusive bound of a plain index inside one
    group, so ``groups * bound`` is the storage capacity of the layout.
    """
    assert layout in _LAYOUTS, layout
    assert size and all(isinstance(extent, int) for extent in size), size
    assert 0 <= nbatch <= len(size) - 2, (size, nbatch)
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
    assert 0 <= nnz <= groups * bound, (layout, size, nnz)
    if groups == 0 or bound == 0:
        assert nnz == 0, (layout, size, nnz)
    return batch, groups, bound, batch + (nnz,) + tuple(blocks or ()) + dense


def _stored_shape(*case):
    """Stored-values shape of a compressed case row."""
    layout, size, nnz, blocks, nbatch, unsorted = case
    del unsorted
    return _compressed_meta(layout, size, nnz, blocks, nbatch)[3]


def _compressed_indices(layout, size, nnz, blocks, nbatch, unsorted, device):
    """Compressed offsets and plain indices of a case row, built on ``device``."""
    _, groups, _, _ = _compressed_meta(layout, size, nnz, blocks, nbatch)
    index_dtype = _INDEX_DTYPE
    if groups == 0:
        # A zero-extent matrix has an empty offset vector and no plain index.
        compressed = torch.zeros(1, dtype=index_dtype, device=device)
        plain = torch.empty(0, dtype=index_dtype, device=device)
    else:
        counts = torch.full((groups,), nnz // groups, dtype=index_dtype, device=device)
        counts[: nnz % groups] += 1
        # cumsum promotes int32 to int64 by default, which would widen the index
        # tensors past the dtype this backend declares; pin it.
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
            # Reverse the order inside every group, so the storage is genuinely
            # unsorted and must be read back verbatim.
            plain = torch.repeat_interleave(counts, counts) - 1 - plain
    if nbatch:
        compressed = compressed.expand(*size[:nbatch], -1).contiguous()
        plain = plain.expand(*size[:nbatch], -1).contiguous()
    return compressed, plain


def _coo_indices(sparse_shape, nnz, device):
    """Distinct int64 COO indices: the first ``nnz`` flat positions, unranked.

    Unranking is injective and increasing, so the coordinates are already sorted
    and unique, which is what the ``is_coalesced=True`` construction relies on.
    """
    assert 0 <= nnz <= math.prod(sparse_shape), (sparse_shape, nnz)
    if nnz == 0 or not sparse_shape:
        return torch.empty((len(sparse_shape), nnz), dtype=torch.int64, device=device)
    flat = torch.arange(nnz, dtype=torch.int64, device=device)
    rows = []
    for extent in reversed(tuple(sparse_shape)):
        rows.append(flat % extent)
        flat = flat // extent
    return torch.stack(rows[::-1])


def _make_compressed(layout, size, nnz, blocks, nbatch, unsorted, dtype, values):
    """Build the compressed sparse input of a case row from its values buffer."""
    compressed, plain = _compressed_indices(
        layout, size, nnz, blocks, nbatch, unsorted, values.device
    )
    return _LAYOUTS[layout](
        compressed, plain, values, size=tuple(size), device=values.device
    )


def _make_coo(sparse_shape, dense_shape, nnz, dtype, values, is_coalesced=True):
    """Build a COO input from truthful sorted, unique indices."""
    del dtype
    return torch.sparse_coo_tensor(
        _coo_indices(sparse_shape, nnz, values.device),
        values,
        tuple(sparse_shape) + tuple(dense_shape),
        device=values.device,
        is_coalesced=is_coalesced,
    )


def _strided_strides(shape):
    """Row-major strides with an inner step of 2, so the buffer is not dense."""
    strides, step = [], 2
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return tuple(reversed(strides))


def _strided_values(shape, dtype, value_range=("-1", "1")):
    """Stored-values buffer with inner stride 2 and storage offset 3.

    Slicing a longer buffer keeps the tensor genuinely noncontiguous and offset,
    so a compacting copy cannot satisfy the layout assertions.
    """
    numel = math.prod(shape) if shape else 1
    storage = tu.make_input(dtype, (2 * numel + 4,), list(value_range))
    return storage[3:].as_strided(shape, _strided_strides(shape))


def _filled_like(tensor, value):
    """Constant tensor of ``tensor``'s shape/dtype (7 and 5 are exact in fp8)."""
    return torch.full_like(tensor, value)


def _index_buffers(inp):
    """The input's index buffers, in a layout-independent order."""
    if inp.layout == torch.sparse_coo:
        return [inp.indices()]
    compressed, plain = _POINTER_ATTRS[inp.layout]
    return [getattr(inp, compressed)(), getattr(inp, plain)()]


def _assert_values_view(res_out, ref_out, inp, ref_inp):
    """Stored values plus the aliasing contract of a stored-values view."""
    native = torch.ops.aten.values(inp)
    indices = _index_buffers(ref_inp)

    # Exact native metadata and forward values, before anything is mutated.
    assert res_out.shape == native.shape == ref_out.shape
    assert res_out.stride() == native.stride() == ref_out.stride()
    assert (
        res_out.storage_offset() == native.storage_offset() == ref_out.storage_offset()
    )
    assert res_out.dtype == native.dtype
    tu.assert_result_equal(res_out, ref_out)

    # A view, not a copy: `res is inp.values()` does not hold on this backend, so
    # aliasing is shown by the shared buffer plus the write-through below.
    assert res_out.data_ptr() == native.data_ptr()

    marker = _filled_like(native, 7)
    res_out.copy_(marker)
    tu.assert_result_equal(native, tu.to_reference(marker))
    marker = _filled_like(native, 5)
    inp.values().copy_(marker)
    tu.assert_result_equal(res_out, tu.to_reference(marker))

    for expected, actual in zip(indices, _index_buffers(inp)):
        tu.assert_result_equal(actual.detach(), expected)


def _shape_to_row(shape):
    """Map one spec shape level onto a compressed case row of the same scale.

    No sparse layout stores a rank-0 values buffer, so the 0-dim level uses a
    singleton matrix. Batched levels keep the requested rank: trailing dims stay
    dense, and rank >= 5 uses two batch axes.
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


# Layout and emptiness rows: hybrid dense trailing dims, zero nnz, zero-extent
# storage, BSR/BSC blocks, CSC, one and two batch axes, and unsorted storage.
# Every row is a small boundary case, so quick keeps the whole list.
_LAYOUT_ROWS = [
    ("csr", (5, 4), 6, None, 0, False),
    ("csr", (8, 8), 16, None, 0, False),
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
    ("bsr", (2, 8, 12), 6, (4, 4), 1, False),
    ("bsr", (4, 20, 10, 3), 5, (5, 5), 1, False),
    ("bsc", (6, 4), 6, (2, 2), 0, False),
    ("bsc", (6, 4), 6, (2, 2), 0, True),
    ("bsc", (2, 6, 4), 6, (2, 2), 1, False),
    ("csr", (0, 5), 0, None, 0, False),
    ("csr", (2, 0, 5), 0, None, 1, False),
    ("csc", (5, 0), 0, None, 0, False),
    ("bsr", (4, 0), 0, (2, 2), 0, False),
    ("bsc", (6, 0), 0, (2, 2), 0, False),
]
_STORAGE_CASES = tu.selected_cases(_LAYOUT_ROWS, quick=_LAYOUT_ROWS)

# COO boundaries: element, hybrid (dense trailing dims), multi-sparse-dim,
# rank-0 scalar and empty forms. The coalesced precondition applies to every row
# here and is also exercised as a negative below.
_COO_BOUNDARY_ROWS = [
    ((2, 2), (), 2),
    ((8, 8), (4,), 16),
    ((6, 6), (), 12),
    ((3, 5, 7), (), 20),
    ((4, 6), (2, 3), 12),
    ((5, 5), (), 0),
    ((64, 64), (), 256),
    ((), (), 1),
]
_COO_CASES = tu.selected_cases(_COO_BOUNDARY_ROWS, quick=_COO_BOUNDARY_ROWS)

# Stored values in a noncontiguous, offset buffer; small rows, kept in quick too.
_STRIDED_ROWS = [
    ("compressed", ("csr", (5, 4), 6, None, 0, False)),
    ("compressed", ("csc", (5, 4), 6, None, 0, False)),
    ("compressed", ("bsr", (6, 6), 6, (3, 2), 0, False)),
    ("compressed", ("csr", (2, 6, 8), 12, None, 1, False)),
    ("compressed", ("bsc", (2, 6, 4), 6, (2, 2), 1, False)),
    ("coo", ((8, 8), (4,), 16)),
    ("coo", ((4, 6), (2, 3), 12)),
]


def _case_values_shape(kind, case):
    if kind == "coo":
        sparse_shape, dense_shape, nnz = case
        del sparse_shape
        return (nnz,) + tuple(dense_shape)
    return _stored_shape(*case)


def _make_case(kind, case, dtype, values, is_coalesced=True):
    if kind == "coo":
        return _make_coo(*case, dtype, values, is_coalesced=is_coalesced)
    return _make_compressed(*case, dtype, values)


# One workload per row: the shape levels x the five value ranges over every
# supported dtype, then the layout rows over every supported dtype.
_VALUE_ROWS = [
    (row, dtype, list(value_range))
    for row in [_shape_to_row(shape) for shape in tu.selected_shapes()]
    for dtype in _VALUE_DTYPES
    for value_range in tu.selected_ranges()
] + [(row, dtype, ["-1", "1"]) for row in _STORAGE_CASES for dtype in _VALUE_DTYPES]
_COO_ROWS = [(case, dtype) for case in _COO_CASES for dtype in _COO_DTYPES]
_STRIDED_CASES = tu.selected_cases(
    [(kind, case, dtype) for kind, case in _STRIDED_ROWS for dtype in _VALUE_DTYPES],
    quick=[
        (kind, case, dtype) for kind, case in _STRIDED_ROWS for dtype in _VALUE_DTYPES
    ],
)

# Positive NaN/Inf workloads are default-only.
_SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])

_BACKWARD_ROWS = [
    ("compressed", ("csr", (8, 8), 16, None, 0, False)),
    ("compressed", ("csr", (5, 4), 6, None, 0, False)),
    ("coo", ((6, 6), (), 12)),
    ("coo", ((4, 6), (2, 3), 12)),
]
_BACKWARD_DEFAULT = [
    (kind, case, dtype)
    for kind, case in _BACKWARD_ROWS
    for dtype in (
        _FLOAT_BACKWARD_DTYPES
        if kind == "compressed"
        else _FLOAT_BACKWARD_DTYPES + _COMPLEX_BACKWARD_DTYPES
    )
]
_BACKWARD_CASES = tu.selected_cases(_BACKWARD_DEFAULT, quick=[])


@pytest.mark.values
@pytest.mark.parametrize("case,dtype,value_range", _VALUE_ROWS)
def test_values(case, dtype, value_range):
    values = tu.make_input(dtype, _stored_shape(*case), value_range)
    ref_values = tu.to_reference(values)

    inp = _make_compressed(*case, dtype, values)
    ref_inp = _make_compressed(*case, dtype, ref_values)

    ref_out = torch.ops.aten.values(ref_inp)
    res_out = flag_gems.values(inp)

    _assert_values_view(res_out, ref_out, inp, ref_inp)


@pytest.mark.values
@pytest.mark.parametrize("case,dtype", _COO_ROWS)
def test_values_coo(case, dtype):
    values = tu.make_input(dtype, _case_values_shape("coo", case), ["-1", "1"])
    ref_values = tu.to_reference(values)

    inp = _make_coo(*case, dtype, values)
    ref_inp = _make_coo(*case, dtype, ref_values)

    ref_out = torch.ops.aten.values(ref_inp)
    res_out = flag_gems.values(inp)

    _assert_values_view(res_out, ref_out, inp, ref_inp)


@pytest.mark.values
@pytest.mark.parametrize("kind,case,dtype", _STRIDED_CASES)
def test_values_strided_values(kind, case, dtype):
    values = _strided_values(_case_values_shape(kind, case), dtype)
    ref_values = tu.to_reference(values)

    inp = _make_case(kind, case, dtype, values)
    ref_inp = _make_case(kind, case, dtype, ref_values)

    ref_out = torch.ops.aten.values(ref_inp)
    res_out = flag_gems.values(inp)

    # The stored buffer is noncontiguous and offset, so the accessor must hand
    # back exactly that layout rather than a compacted copy.
    assert res_out.stride() == values.stride()
    assert res_out.storage_offset() == values.storage_offset()
    _assert_values_view(res_out, ref_out, inp, ref_inp)


@pytest.mark.values
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_values_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_values = tu.to_reference(values)

    inp = _make_compressed("csr", (2, 3), values.numel(), None, 0, False, dtype, values)
    ref_inp = _make_compressed(
        "csr", (2, 3), values.numel(), None, 0, False, dtype, ref_values
    )

    ref_out = torch.ops.aten.values(ref_inp)
    res_out = flag_gems.values(inp)

    _assert_values_view(res_out, ref_out, inp, ref_inp)


@pytest.mark.values
@pytest.mark.parametrize("kind,case,dtype", _BACKWARD_CASES)
def test_values_backward(kind, case, dtype):
    values = tu.make_input(dtype, _case_values_shape(kind, case), ["-1", "1"])
    values = values.requires_grad_(True)
    ref_values = tu.to_reference(values).detach().requires_grad_(True)

    inp = _make_case(kind, case, dtype, values)
    ref_inp = _make_case(kind, case, dtype, ref_values)

    ref_out = torch.ops.aten.values(ref_inp)
    res_out = flag_gems.values(inp)

    assert res_out.shape == ref_out.shape
    tu.assert_result_equal(res_out, ref_out)

    # Nonuniform upstream, so a reordered or partially reduced gradient cannot
    # pass. The operator is differentiated through the original leaf values
    # tensor, not through the returned view.
    upstream = tu.make_input(dtype, res_out.shape, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream).detach()
    ref_grad = torch.autograd.grad(ref_out, ref_values, grad_outputs=ref_upstream)[0]
    res_grad = torch.autograd.grad(res_out, values, grad_outputs=upstream)[0]

    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.values
def test_values_rejects_dense_input():
    # native: "values expected sparse tensor layout but got Strided"
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values(inp)


@pytest.mark.values
def test_values_rejects_uncoalesced_coo():
    # native: "Cannot get values on an uncoalesced tensor, please call
    # .coalesce() first"
    values = tu.make_input(torch.float32, (8,), ["-1", "1"])
    inp = _make_coo((4, 4), (), 8, torch.float32, values, is_coalesced=False)

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values(inp)


@pytest.mark.values
def test_values_rejects_non_tensor_input():
    # native: no overload of aten::values accepts a bare Python scalar
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.values(3.14)
