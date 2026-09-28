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

"""Correctness tests for ``aten::to_dense``.

Native behaviour pinned on the active backend:

* a strided input without ``dtype`` returns the *same* tensor, keeping its
  strides, storage offset and lazy conj/neg flags;
* a strided input with ``dtype`` behaves like ``Tensor.to`` and may compact a
  stepped or column-sliced stride, so the result geometry is compared with the
  native result instead of with the input;
* a sparse input is densified: COO (empty, a zero dense tail, uncoalesced
  unique coordinates, a coalesced set, duplicate coordinates that are summed,
  a dense-tail hybrid and the ``sparse_dim == 0`` scalar form, which may store
  more than one entry), CSR/CSC and BSR/BSC (rectangular, skewed rows, zero
  extents, batched with independent patterns, hybrid dense tails, zero stored
  blocks, int32 pointers);
* ``dtype`` combined with a sparse input is rejected by the native op;
* ``masked_grad`` only changes the gradient of a sparse leaf.

Stored COO coordinates are int64: ``torch.sparse_coo_tensor`` accepts a
correctly shaped int32 coordinate tensor but normalizes the *stored* indices to
int64, and ``Tensor.to_sparse()`` emits int64 coordinates (it takes no
index-width argument).  A COO fixture here therefore always materializes int64
index storage, so every COO workload is selected out where the static
capability flags report no int64 support.  The compressed layouts are built at
the supported pointer width and stay available there.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _supported(dtypes):
    """Keep the dtypes this backend statically reports as supported."""
    flags = {
        torch.float64: flag_gems.runtime.device.support_fp64,
        torch.bfloat16: flag_gems.runtime.device.support_bf16,
        torch.int64: flag_gems.runtime.device.support_int64,
        torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
        torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
    }
    return [dtype for dtype in dtypes if flags.get(dtype, True)]


# The spec's nine dtypes plus the two extra types this op moves or casts.
_ALL_DTYPES = _supported(
    [
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
        torch.bool,
    ]
)
_STRIDED_DTYPES = _ALL_DTYPES
_FLOAT_DTYPES = [dtype for dtype in _ALL_DTYPES if dtype.is_floating_point]
# Measured on this backend: fp8 has no sparse path.  A hand-built fp8 COO, CSR
# or BSR input raises "'index_add' not implemented for 'Float8_e4m3fn'" (and
# 'Float8_e5m2') inside the native to_dense, to_sparse() raises
# "'nonzero_cuda' not implemented" and COO->CSR raises
# "'coalesce_sparse_cuda' not implemented", so there is no native sparse
# reference value to compare a candidate against.  bool builds and densifies
# correctly, so it stays in the sparse set.
_SPARSE_DTYPES = [dtype for dtype in _ALL_DTYPES if dtype not in _FP8_DTYPES]

# Structural index width, read from the static capability flags.  Stored COO
# coordinates are int64 on this backend: torch.sparse_coo_tensor accepts a
# correctly shaped int32 coordinate tensor but normalizes the stored indices to
# int64 (probed), and Tensor.to_sparse() emits int64 coordinates with no
# user-selectable width.  The COO fixtures below therefore store int64
# coordinates, so every COO workload is selected out where the static flags
# report no int64 support.  The compressed constructors accept the narrower
# width (measured for every CSR/CSC/BSR/BSC fixture below, including the
# batched and hybrid ones), so those families and the dense strided coverage
# stay available there.
_INT64_STRUCTURAL = flag_gems.runtime.device.support_int64


def _pointer_dtype():
    """Structural width for the compressed (CSR/CSC/BSR/BSC) pointers."""
    return torch.int64 if _INT64_STRUCTURAL else torch.int32


def _coo_cases(cases):
    """Select COO workloads only where int64 structural indices exist."""
    return list(cases) if _INT64_STRUCTURAL else []


def _index(values, dtype=torch.int64):
    """A structural index tensor on the candidate device.

    ``values`` is a flat or nested list of ints; nested lists build the 2-D
    pointer tensors that batched compressed layouts take.
    """
    return torch.tensor(values, dtype=dtype, device=flag_gems.device)


def _pointers(values):
    """Compressed crow/col pointers at the supported structural width.

    Native requires both pointers to share one dtype ("crow_indices and
    col_indices must have the same dtype"), so every pointer tensor of a
    fixture is built through this helper.
    """
    return _index(values, _pointer_dtype())


@pytest.mark.to_dense
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _STRIDED_DTYPES)
def test_to_dense_strided_is_identity(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    # Native contract: the input tensor itself is returned, so its strides,
    # storage offset and lazy flags are the input's by construction; the shared
    # value comparison exercises them.
    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


_VIEW_DTYPES = _supported([torch.float32, torch.float16, torch.bfloat16, torch.float64])


def _view_input(geometry, dtype):
    """Strided inputs that carry a non-contiguous or lazy view state.

    Returns ``(view, parent)``.  ``parent`` is the real tensor the view is
    carved from -- for a lazy neg/conj view the un-flagged base tensor -- so a
    test can compare the whole backing allocation, view padding included.
    """
    if geometry == "transpose":
        parent = tu.make_input(dtype, (8, 16), ["-1", "1"])
        return parent.t(), parent
    if geometry == "step":
        parent = tu.make_input(dtype, (96,), ["-1", "1"])
        return parent[::3], parent
    if geometry == "offset":
        parent = tu.make_input(dtype, (12, 16), ["-1", "1"])
        return parent[3:], parent
    if geometry == "neg":
        parent = tu.make_input(dtype, (8, 16), ["-1", "1"])
        return torch._neg_view(parent), parent
    if geometry == "conj":
        parent = tu.make_input(dtype, (8, 16), ["-1", "1"])
        return parent.conj(), parent
    raise ValueError(geometry)


_VIEW_CASES = tu.selected_cases(
    [
        (geometry, dtype)
        for geometry in ("transpose", "step", "offset", "neg")
        for dtype in _VIEW_DTYPES
    ]
    + [("conj", torch.complex64)],
    quick=[],
)


@pytest.mark.to_dense
@pytest.mark.parametrize("geometry,dtype", _VIEW_CASES)
def test_to_dense_strided_view_identity(geometry, dtype):
    inp, parent = _view_input(geometry, dtype)
    # Whole-allocation snapshot: the view shares its parent's storage, so a
    # write outside the view (a lazy flag resolved in place, or a scribble into
    # the padding) is only visible on the parent.
    parent_before = parent.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    # Aliasing is the contract: returning the identical object keeps the view
    # geometry and the lazy flags, so only the aliasing is asserted here and the
    # shared comparison then exercises the view's values.
    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)
    # Resolving the view must not write into the parent allocation.
    tu.assert_result_equal(parent, parent_before)


# Both endpoints must be statically supported to exercise the cast.
_CAST_PAIRS = [
    pair
    for pair in (
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float64),
        (torch.int32, torch.float32),
        (torch.int64, torch.float64),
        (torch.uint8, torch.float32),
        (torch.int64, torch.int32),
        (torch.float8_e4m3fn, torch.float32),
        (torch.float8_e5m2, torch.float32),
        (torch.float32, torch.float8_e4m3fn),
        (torch.float32, torch.float8_e5m2),
    )
    if _supported(list(pair)) == list(pair)
]
# Measured on this backend: the lazy neg flag cannot be resolved for fp8, so
# torch.ops.aten.to_dense(torch._neg_view(float8_e4m3fn_tensor), dtype=...) raises
# RuntimeError: "neg_cuda" not implemented for 'Float8_e4m3fn' (and 'Float8_e5m2').
# The neg family therefore keeps the dtypes whose neg kernel exists; fp8 casts
# stay covered by the contiguous, transpose, offset and step families.
_NEG_CAST_PAIRS = [
    pair for pair in _CAST_PAIRS if not any(dtype in _FP8_DTYPES for dtype in pair)
]
# The small symmetric range keeps the cast itself the subject of the test; the
# extreme ranges exercise the narrowing targets.  Measured on this backend: the
# two fp8 formats differ, float8_e4m3fn's largest magnitude is 448 and an
# overflow produces nan while float8_e5m2's is 57344 and an overflow produces
# inf (e.g. 1e5 -> nan vs inf, 65504.0 -> nan vs inf), which the shared
# assertion compares with equal_nan.
_CAST_RANGES = tu.selected_cases(
    [["-1", "1"], ["0", "max"], ["min", "0"]], quick=[["-1", "1"]]
)


@pytest.mark.to_dense
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype_pair", tu.selected_cases(_CAST_PAIRS, quick=[_CAST_PAIRS[0]])
)
@pytest.mark.parametrize("value_range", _CAST_RANGES)
def test_to_dense_strided_dtype_cast(shape, dtype_pair, value_range):
    in_dtype, out_dtype = dtype_pair
    inp = tu.make_input(in_dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp, dtype=out_dtype)
    res_out = flag_gems.to_dense(inp, dtype=out_dtype)

    assert res_out.device == inp.device
    # Casting may compact a stepped/column-sliced stride or resolve a lazy flag:
    # compare the result geometry with the native result, not with the input.
    # The output dtype is compared by the shared assertion.
    assert res_out.stride() == ref_out.stride()
    assert res_out.is_neg() == ref_out.is_neg()
    assert res_out.is_conj() == ref_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)


_VIEW_CAST_CASES = tu.selected_cases(
    [
        (geometry, pair)
        for geometry in ("transpose", "offset", "step")
        for pair in _CAST_PAIRS
    ]
    + [("neg", pair) for pair in _NEG_CAST_PAIRS]
    + [("conj", (torch.complex64, torch.complex64))],
    quick=[],
)


@pytest.mark.to_dense
@pytest.mark.parametrize("geometry,dtype_pair", _VIEW_CAST_CASES)
def test_to_dense_strided_view_cast(geometry, dtype_pair):
    in_dtype, out_dtype = dtype_pair
    inp, parent = _view_input(geometry, in_dtype)
    parent_before = parent.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp, dtype=out_dtype)
    res_out = flag_gems.to_dense(inp, dtype=out_dtype)

    assert res_out.device == inp.device
    assert res_out.stride() == ref_out.stride()
    assert res_out.is_neg() == ref_out.is_neg()
    assert res_out.is_conj() == ref_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)
    # Resolving the view (and any lazy flag) must not write into its parent.
    tu.assert_result_equal(parent, parent_before)


# A COO workload needs int64 structural indices, so this grid is empty on a
# backend without int64 support (the compressed families below still run).
_COO_SHAPES = _coo_cases(tu.selected_shapes())


@pytest.mark.to_dense
@pytest.mark.parametrize("shape", _COO_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_to_dense_sparse_coo(shape, value_range, dtype):
    # Covers every spec shape, including the 0-dim tensor, whose COO form has
    # sparse_dim == 0 (the native op supports it).
    inp = tu.make_input(dtype, shape, value_range).to_sparse()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)


_LAYOUT_DTYPES = _supported([torch.float32, torch.int32])
_COMPRESSED_LAYOUT_NAMES = (
    "csr_last",
    "csr_skew",
    "csr_zero_rows",
    "csr_zero_cols",
    "csr_batched",
    "csr_batched_independent",
    "csr_hybrid_tail",
    "csr_int32_pointers",
    "csc_rect",
    "bsr_rect",
    "bsc_rect",
    "bsr_zero_block",
    "bsr_zero_blocks",
    "bsc_zero_blocks",
)
_COO_LAYOUT_NAMES = (
    "coo_scalar",
    "coo_scalar_duplicates",
    "coo_scalar_tail",
    "coo_hybrid",
    "coo_coalesced",
    "coo_empty",
    "coo_zero_tail",
)
# The compressed layouts are built at the supported pointer width, so they are
# available everywhere; the COO layouts store int64 coordinates and are
# selected out statically on a backend without int64 support.
_LAYOUT_NAMES = _COMPRESSED_LAYOUT_NAMES + tuple(_coo_cases(_COO_LAYOUT_NAMES))


def _sparse_layout_input(layout, dtype, value_range):
    """Sparse inputs for the hybrid, compressed and edge-case layouts.

    The fixtures pin the layout conversion (block size, zero extents, empty
    rows/columns, entries at the last row/column, batch dimensions, compressed
    dense tails, pointer width) with valid shapes; values come from the shared
    value ranges wherever the layout stores more than one entry.
    """
    device = flag_gems.device

    if layout == "coo_scalar":
        return torch.sparse_coo_tensor(
            torch.empty(0, 1, dtype=torch.int64, device=device),
            tu.make_input(dtype, (1,), value_range),
            (),
            device=device,
        )
    if layout == "coo_scalar_duplicates":
        # A scalar COO may store more than one entry; the coordinate is the
        # single scalar position, so the entries are summed like duplicates.
        return torch.sparse_coo_tensor(
            torch.empty(0, 2, dtype=torch.int64, device=device),
            tu.make_input(dtype, (2,), value_range),
            (),
            device=device,
        )
    if layout == "coo_scalar_tail":
        # sparse_dim 0 with a dense tail: the logical shape is the tail shape.
        return torch.sparse_coo_tensor(
            torch.empty(0, 1, dtype=torch.int64, device=device),
            tu.make_input(dtype, (1, 2, 3), value_range),
            (2, 3),
            device=device,
        )
    if layout == "coo_hybrid":
        return torch.sparse_coo_tensor(
            _index([[0, 2, 4]]),
            tu.make_input(dtype, (3, 3, 4), value_range),
            (5, 3, 4),
            device=device,
        )
    if layout == "coo_coalesced":
        return torch.sparse_coo_tensor(
            _index([[0, 1, 3], [2, 0, 1]]),
            tu.make_input(dtype, (3,), value_range),
            (4, 5),
            device=device,
        ).coalesce()
    if layout == "coo_empty":
        return torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=dtype, device=device),
            (8, 16),
            device=device,
        )
    if layout == "coo_zero_tail":
        # sparse_dim 1 with a zero DENSE tail: the stored values are (2, 0) and
        # the native op densifies to (3, 0) (measured on this backend).
        return torch.sparse_coo_tensor(
            _index([[0, 2]]),
            tu.make_input(dtype, (2, 0), value_range),
            (3, 0),
            device=device,
        )
    if layout == "csr_last":
        return torch.sparse_csr_tensor(
            _pointers([0, 0, 0, 0, 0, 1]),
            _pointers([2]),
            tu.make_input(dtype, (1,), value_range),
            (5, 3),
            device=device,
        )
    if layout == "csr_skew":
        return torch.sparse_csr_tensor(
            _pointers([0, 1, 1, 3]),
            _pointers([3, 0, 3]),
            tu.make_input(dtype, (3,), value_range),
            (3, 4),
            device=device,
        )
    if layout == "csr_zero_rows":
        return torch.sparse_csr_tensor(
            _pointers([0]),
            torch.empty(0, dtype=_pointer_dtype(), device=device),
            torch.empty(0, dtype=dtype, device=device),
            (0, 4),
            device=device,
        )
    if layout == "csr_zero_cols":
        return torch.sparse_csr_tensor(
            _pointers([0, 0, 0, 0, 0]),
            torch.empty(0, dtype=_pointer_dtype(), device=device),
            torch.empty(0, dtype=dtype, device=device),
            (4, 0),
            device=device,
        )
    if layout == "csr_batched":
        return torch.sparse_csr_tensor(
            _pointers([[0, 1, 2], [0, 1, 2]]),
            _pointers([[0, 1], [0, 1]]),
            tu.make_input(dtype, (2, 2), value_range),
            (2, 2, 2),
            device=device,
        )
    if layout == "csr_batched_independent":
        # Two batches with independent pointer/column patterns.  Native requires
        # col to be as many-dimensional as crow ("crow_indices and col_indices
        # dimensionalities must be equal"), so both are 2-D here: batch 0 stores
        # columns {0, 3} and batch 1 columns {2, 3}, and a flat col is rejected.
        return torch.sparse_csr_tensor(
            _pointers([0, 2, 2, 2, 0, 1, 2, 2]).reshape(2, 4),
            _pointers([0, 3, 2, 3]).reshape(2, 2),
            tu.make_input(dtype, (2, 2), value_range),
            (2, 3, 4),
            device=device,
        )
    if layout == "csr_hybrid_tail":
        # Compressed rows with a dense tail of shape[2:] per stored entry
        # (measured to densify to (3, 4, 2)).
        return torch.sparse_csr_tensor(
            _pointers([0, 1, 2, 3]),
            _pointers([0, 3, 1]),
            tu.make_input(dtype, (3, 2), value_range),
            (3, 4, 2),
            device=device,
        )
    if layout == "csr_int32_pointers":
        # Measured: the compressed constructor and the native densify accept
        # int32 crow/col pointers, so this fixture pins the narrower structural
        # width explicitly instead of always building int64 pointers.
        return torch.sparse_csr_tensor(
            _index([0, 1, 2, 3], torch.int32),
            _index([0, 3, 1], torch.int32),
            tu.make_input(dtype, (3, 2), value_range),
            (3, 4, 2),
            device=device,
        )
    if layout == "csc_rect":
        return torch.sparse_csc_tensor(
            _pointers([0, 0, 1, 2, 2, 2, 2]),
            _pointers([1, 0]),
            tu.make_input(dtype, (2,), value_range),
            (3, 6),
            device=device,
        )
    if layout == "bsr_rect":
        # The block size is inferred from the values shape (1, 2, 3).
        return torch.sparse_bsr_tensor(
            _pointers([0, 1, 1, 1, 1]),
            _pointers([3]),
            tu.make_input(dtype, (1, 2, 3), value_range),
            (8, 12),
            device=device,
        )
    if layout == "bsc_rect":
        return torch.sparse_bsc_tensor(
            _pointers([0, 1, 1, 1, 1]),
            _pointers([1]),
            tu.make_input(dtype, (1, 3, 2), value_range),
            (12, 8),
            device=device,
        )
    if layout == "bsr_zero_block":
        return torch.sparse_bsr_tensor(
            _pointers([0, 1, 1, 1, 1]),
            _pointers([1]),
            torch.zeros(1, 2, 2, dtype=dtype, device=device),
            (8, 8),
            device=device,
        )
    if layout == "bsr_zero_blocks":
        # Stores no block at all (nnz == 0); measured to densify to a zero (4, 4).
        return torch.sparse_bsr_tensor(
            _pointers([0, 0, 0]),
            torch.empty(0, dtype=_pointer_dtype(), device=device),
            torch.empty(0, 2, 2, dtype=dtype, device=device),
            (4, 4),
            device=device,
        )
    if layout == "bsc_zero_blocks":
        return torch.sparse_bsc_tensor(
            _pointers([0, 0, 0]),
            torch.empty(0, dtype=_pointer_dtype(), device=device),
            torch.empty(0, 2, 2, dtype=dtype, device=device),
            (4, 4),
            device=device,
        )
    raise ValueError(layout)


def _sparse_components(inp):
    """The stored index/value tensors of a sparse input, per layout."""
    if inp.layout == torch.sparse_coo:
        return (inp._indices(), inp._values())
    if inp.layout in (torch.sparse_csr, torch.sparse_bsr):
        return (inp.crow_indices(), inp.col_indices(), inp.values())
    return (inp.ccol_indices(), inp.row_indices(), inp.values())


@pytest.mark.to_dense
@pytest.mark.parametrize("layout", tu.selected_cases(_LAYOUT_NAMES, quick=[]))
@pytest.mark.parametrize(
    "value_range", tu.selected_cases(tu.selected_ranges(), quick=[])
)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_dense_sparse_layouts(layout, value_range, dtype):
    inp = _sparse_layout_input(layout, dtype, value_range)
    storage_before = [component.clone() for component in _sparse_components(inp)]
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)
    # Densifying must not rewrite or reorder the storage it reads.
    for component, snapshot in zip(_sparse_components(inp), storage_before):
        tu.assert_result_equal(component, snapshot)


_ACCUMULATION_DTYPES = _supported(
    [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
_ACCUMULATION_PATTERNS = (
    "coalesced",
    "unique_uncoalesced",
    "duplicates",
    "duplicates_coalesced",
    "cancellation",
    "cancellation_extreme",
    "large_range",
)
# Two stored entries per coordinate, i.e. six stored entries over three
# coordinates in a (4, 4) matrix.
_DUPLICATE_ROWS = (0, 0, 1, 1, 3, 3)
_DUPLICATE_COLS = (1, 1, 0, 0, 2, 2)


def _accumulation_input(pattern, dtype):
    """COO fixtures that isolate a specific addition pattern.

    ``cancellation`` and ``cancellation_extreme`` store ``+v`` and ``-v`` at the
    *same* coordinate by interleaving the negated values, so each coordinate's
    two entries really cancel and the reference result is exactly zero.  The
    dtypes here are all floating point, so the negation is a plain sign flip.
    """
    device = flag_gems.device
    rows = _index(_DUPLICATE_ROWS)
    cols = _index(_DUPLICATE_COLS)
    if pattern in ("coalesced", "unique_uncoalesced"):
        # Distinct coordinates: a pure scattering copy, no summation.
        inp = torch.sparse_coo_tensor(
            torch.stack([rows[0::2], cols[0::2]]),
            tu.make_input(dtype, (3,), ["-1", "1"]),
            (4, 4),
            device=device,
        )
        return inp.coalesce() if pattern == "coalesced" else inp
    if pattern in ("duplicates", "duplicates_coalesced"):
        values = tu.make_input(dtype, (6,), ["-1", "1"])
    elif pattern in ("cancellation", "cancellation_extreme"):
        magnitude = ["-1", "1"] if pattern == "cancellation" else ["0", "max"]
        positive = tu.make_input(dtype, (3,), magnitude)
        values = torch.stack([positive, -positive], dim=1).reshape(-1)
    elif pattern == "large_range":
        # Largest-magnitude values: duplicates may saturate to inf, and the
        # candidate must reproduce whatever classification the reference has.
        values = tu.make_input(dtype, (6,), ["0", "max"])
    else:
        raise ValueError(pattern)
    inp = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, (4, 4), device=device
    )
    return inp.coalesce() if pattern == "duplicates_coalesced" else inp


@pytest.mark.to_dense
@pytest.mark.parametrize(
    "pattern", tu.selected_cases(_coo_cases(_ACCUMULATION_PATTERNS), quick=[])
)
@pytest.mark.parametrize("dtype", _ACCUMULATION_DTYPES)
def test_to_dense_sparse_accumulation(pattern, dtype):
    inp = _accumulation_input(pattern, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    if pattern in ("coalesced", "unique_uncoalesced"):
        tu.assert_result_equal(res_out, ref_out)
    elif pattern in ("cancellation", "cancellation_extreme"):
        # Every coordinate stores +v and -v, so the native result is exactly
        # zero; atol=0 pins that cancellation instead of letting rtol excuse a
        # mispaired fixture.
        tu.assert_result_close(res_out, ref_out, atol=0)
    else:
        # Duplicate coordinates are summed on both sides: arithmetic result.
        tu.assert_result_close(res_out, ref_out)


_STORAGE_VIEW_DTYPES = _supported(
    [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
_STORAGE_VIEW_KINDS = ("values", "indices")
_STORAGE_VIEW_COORDS = ((0, 0), (1, 2), (2, 1), (3, 3), (4, 0), (5, 4))


def _storage_view_input(kind, dtype):
    """COO whose stored values or indices are views of a larger buffer.

    The values leg stores a strided view (measured to keep its strides through
    the sparse constructor) and the index leg an offset view of a larger
    buffer, so densifying reads the stored components through their storage
    layout instead of a freshly packed (nnz,) / (2, nnz) tensor.  Each input is
    returned together with the real parent allocation it is carved from, so a
    test can compare the whole buffer and not only the logical view.
    """
    device = flag_gems.device
    rows = _index([row for row, _ in _STORAGE_VIEW_COORDS])
    cols = _index([col for _, col in _STORAGE_VIEW_COORDS])
    indices = torch.stack([rows, cols])
    values = tu.make_input(dtype, (len(_STORAGE_VIEW_COORDS),), ["-1", "1"])
    if kind == "values":
        padded = torch.zeros(2 * values.numel(), dtype=dtype, device=device)
        padded[::2] = values
        inp = torch.sparse_coo_tensor(indices, padded[::2], (6, 5), device=device)
        return inp, (padded,)
    buffer = torch.zeros(4, indices.shape[1], dtype=torch.int64, device=device)
    buffer[1:3] = indices
    inp = torch.sparse_coo_tensor(buffer[1:3], values, (6, 5), device=device)
    return inp, (buffer,)


_STORAGE_VIEW_CASES = tu.selected_cases(
    _coo_cases(
        [
            (kind, dtype)
            for kind in _STORAGE_VIEW_KINDS
            for dtype in _STORAGE_VIEW_DTYPES
        ]
    ),
    quick=[],
)


@pytest.mark.to_dense
@pytest.mark.parametrize("kind,dtype", _STORAGE_VIEW_CASES)
def test_to_dense_sparse_storage_views(kind, dtype):
    inp, parents = _storage_view_input(kind, dtype)
    index_stride = inp._indices().stride()
    value_stride = inp._values().stride()
    indices_before = inp._indices().clone()
    values_before = inp._values().clone()
    # Each stored component is a view carved from a parent buffer, so a clone of
    # the view alone cannot see a write into the unused padding of that buffer.
    # The parents are the fixture's own allocations, so snapshot them whole.
    parents_before = [parent.clone() for parent in parents]
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    tu.assert_result_equal(res_out, ref_out)
    # A densify must not re-stride or rewrite the stored components it reads,
    # nor write outside the logical view into the backing allocation.
    assert inp._indices().stride() == index_stride
    assert inp._values().stride() == value_stride
    tu.assert_result_equal(inp._indices(), indices_before)
    tu.assert_result_equal(inp._values(), values_before)
    for parent, snapshot in zip(parents, parents_before):
        tu.assert_result_equal(parent, snapshot)


@pytest.mark.to_dense
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[]),
)
def test_to_dense_strided_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    assert res_out is inp
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_dense
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(
        _coo_cases(
            [
                case
                for case in tu.special_value_cases(_FLOAT_DTYPES)
                if case[0] not in _FP8_DTYPES
            ]
        ),
        quick=[],
    ),
)
def test_to_dense_sparse_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).to_sparse()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    res_out = flag_gems.to_dense(inp)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)


_BACKWARD_DTYPES = _supported(
    [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
_BACKWARD_SHAPES = tu.selected_cases([(256,), (1024, 1024), (20, 320, 15)], quick=[])


@pytest.mark.to_dense
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_to_dense_backward_strided(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp)
    # Nonuniform upstream: an all-ones gradient could pass by accident.
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )[0]

    res_out = flag_gems.to_dense(inp)
    tu.assert_result_equal(res_out, ref_out)

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    # The strided gradient is the upstream tensor itself: exact.
    tu.assert_result_equal(res_grad, ref_grad)


_VALUES_BACKWARD_FORMS = ("unique", "duplicates")


def _values_backward_case(form, dtype):
    if form == "unique":
        indices = _index([[0, 1, 3], [2, 0, 1]])
    else:
        # Duplicate coordinates: the forward sums them, but each stored value
        # still receives exactly its own upstream element.
        indices = _index([[0, 0, 1], [1, 1, 2]])
    return indices, tu.make_input(dtype, (indices.shape[1],), ["-1", "1"])


@pytest.mark.to_dense
@pytest.mark.parametrize(
    "form", tu.selected_cases(_coo_cases(_VALUES_BACKWARD_FORMS), quick=[])
)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_to_dense_backward_sparse_values(form, dtype):
    indices, values = _values_backward_case(form, dtype)
    ref_values = tu.to_reference(values).requires_grad_(True)
    # The reference graph stays on the reference device, so --ref cpu keeps the
    # native side on CPU instead of being pulled back to the candidate device.
    ref_inp = torch.sparse_coo_tensor(
        tu.to_reference(indices), ref_values, (4, 5), device=ref_values.device
    )
    ref_out = torch.ops.aten.to_dense(ref_inp)
    upstream = tu.make_input(dtype, (4, 5), ["-1", "1"])
    ref_grad = torch.autograd.grad(
        ref_out, ref_values, grad_outputs=tu.to_reference(upstream)
    )[0]

    values = values.detach().requires_grad_(True)
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    res_out = flag_gems.to_dense(inp)
    tu.assert_result_equal(res_out, ref_out)

    res_grad = torch.autograd.grad(res_out, values, grad_outputs=upstream)[0]
    # Densifying gathers whole upstream rows: every stored value keeps its own
    # gradient element, so the comparison is exact even for the duplicate form.
    tu.assert_result_equal(res_grad, ref_grad)


_SPARSE_LEAF_FORMS = ("coalesced", "uncoalesced", "duplicates", "hybrid")
# The gradient of a sparse leaf is itself sparse, and masked_grad chooses how
# many sparse coordinates it carries.  "default" leaves the argument out.
_MASKED_GRAD_FORMS = tu.selected_cases(
    [
        ("default", {}),
        ("true", {"masked_grad": True}),
        ("false", {"masked_grad": False}),
    ],
    quick=[],
)


def _sparse_leaf(form, dtype):
    device = flag_gems.device
    if form in ("coalesced", "uncoalesced"):
        # The same distinct coordinates, once explicitly flagged coalesced and
        # once left as stored.
        indices = _index([[0, 1, 3], [2, 0, 1]])
        inp = torch.sparse_coo_tensor(
            indices, tu.make_input(dtype, (3,), ["-1", "1"]), (4, 5), device=device
        )
        return inp.coalesce() if form == "coalesced" else inp
    if form == "duplicates":
        indices = _index([[0, 0, 1], [1, 1, 2]])
        return torch.sparse_coo_tensor(
            indices, tu.make_input(dtype, (3,), ["-1", "1"]), (4, 5), device=device
        )
    if form == "hybrid":
        indices = _index([[0, 2]])
        values = tu.make_input(dtype, (2, 3, 2), ["-1", "1"])
        return torch.sparse_coo_tensor(indices, values, (4, 3, 2), device=device)
    raise ValueError(form)


@pytest.mark.to_dense
@pytest.mark.parametrize(
    "form", tu.selected_cases(_coo_cases(_SPARSE_LEAF_FORMS), quick=[])
)
@pytest.mark.parametrize("masked_form", _MASKED_GRAD_FORMS)
@pytest.mark.parametrize(
    "dtype", _supported([torch.float32, torch.float16, torch.bfloat16])
)
def test_to_dense_backward_sparse_leaf(form, masked_form, dtype):
    _, kwargs = masked_form
    inp = _sparse_leaf(form, dtype).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_dense(ref_inp, **kwargs)
    # One candidate-device upstream, shared by both autograd paths.
    upstream = tu.make_input(dtype, tuple(ref_out.shape), ["-1", "1"])
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )[0]

    res_out = flag_gems.to_dense(inp, **kwargs)
    tu.assert_result_equal(res_out, ref_out)

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    # masked_grad changes which coordinates the sparse gradient keeps, so the
    # sparse metadata, the coordinate set and the stored values are compared.
    assert res_grad.layout == torch.sparse_coo
    assert res_grad.sparse_dim() == ref_grad.sparse_dim()
    assert res_grad.dense_dim() == ref_grad.dense_dim()
    assert res_grad._nnz() == ref_grad._nnz()
    tu.assert_result_equal(
        res_grad.coalesce()._indices(), ref_grad.coalesce()._indices()
    )
    if form == "duplicates":
        # Summed input coordinates make the stored gradient values a sum.
        tu.assert_result_close(
            res_grad.coalesce()._values(), ref_grad.coalesce()._values()
        )
    else:
        # Distinct coordinates make the gradient a pure gather: exact.
        tu.assert_result_equal(
            res_grad.coalesce()._values(), ref_grad.coalesce()._values()
        )


@pytest.mark.to_dense
@pytest.mark.parametrize("layout", _coo_cases(["coo"]) + ["csr"])
def test_to_dense_sparse_dtype_argument_is_rejected(layout):
    # ATen rejects a dtype argument for every sparse layout, so the candidate
    # must not silently cast (or silently ignore) it.  The COO fixture needs
    # int64 structural indices, so it is selected out with the COO family.
    inp = _sparse_layout_input(
        "coo_hybrid" if layout == "coo" else "csr_last", torch.float32, ["-1", "1"]
    )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.to_dense(inp, dtype=torch.float32)


@pytest.mark.to_dense
@pytest.mark.parametrize("masked_grad", ["invalid", "not_a_bool"])
def test_to_dense_invalid_masked_grad_is_rejected(masked_grad):
    # The schema types masked_grad as Optional[bool]; a str is neither a bool
    # nor coercible to one, unlike a float, which the dispatcher accepts.
    inp = tu.make_input(torch.float32, (6,), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.to_dense(inp, masked_grad=masked_grad)
