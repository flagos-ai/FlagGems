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

import flag_gems

from . import test_utils as tu

# Native to_sparse_csc needs a matrix, so the spec shape grid keeps only the rank
# >= 2 shapes; rank 0/1 inputs are the negative cases at the end of the file.
MATRIX_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]

# Static capability flags of the active runtime, matching the policy
# tests/accuracy_utils.py applies to its own dtype tables: an optional dtype whose
# kernel is missing is dropped at collection time instead of being skipped at run
# time. complex64 has its own kernels; complex128 additionally needs FP64.
_OPTIONAL_DTYPES = {
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
}

# Sparse COO/CSR/CSC storage and blocked-sparse (BSR) storage both carry int64 index
# tensors, so the families that build them are collected only on a runtime with int64
# support. This is a static flag, never a runtime probe; the dense input grid is
# unaffected.
_INT64_SUPPORTED = flag_gems.runtime.device.support_int64

# One gated table: every dtype appears exactly once and an optional dtype is collected
# only when its static flag is set. On the probed backend (cuda / nvidia) all of them -
# both FP8 dtypes, complex64 and complex128 included - are accepted by a valid native
# call; the flags scope that set to the backends that carry the kernels.
DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.float16,
    torch.int32,
    torch.bool,
    torch.complex64,
] + [dtype for dtype, supported in _OPTIONAL_DTYPES.items() if supported]

# The special-value matrix follows the supported floating dtypes, so FP64 keeps its
# nan/inf coverage on the backends that provide it.
SPECIAL_DTYPES = [dtype for dtype in DTYPES if dtype.is_floating_point]

# Backward table: the real floating dtypes. Integer and bool inputs are not
# differentiable; a complex gradient follows the Wirtinger conjugate convention that the
# exact-equality oracle below does not model, and FP8 dtypes are not part of the
# autograd table this conversion exercises.
BACKWARD_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    if _OPTIONAL_DTYPES.get(dtype, True)
]

# Supplementary families (view layouts, sparse sources, empty shapes, ...).
SUPPLEMENTAL_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.int32,
        torch.bfloat16,
        torch.float64,
    )
    if _OPTIONAL_DTYPES.get(dtype, True)
]

EMPTY_DTYPES = [torch.float32, torch.int32, torch.float16]
PROFILE_DTYPES = [torch.float32, torch.int32, torch.float16]


def _prod(extents):
    size = 1
    for extent in extents:
        size *= extent
    return size


def _blocks_mask(batch_shape, rows, cols, blocks, device):
    """Bool mask marking `blocks` stored (row, col) cells in every batch entry.

    Batch entry b stores the cyclic window starting at b, modulo rows * cols: the
    stored-cell pattern of the original build. Every entry stores the same number of
    cells - the rule native to_sparse_csc enforces - while the patterns differ per
    entry. The positional arithmetic stays in int32 while the larger of the batch
    count and the cell count is representable there and moves to int64 above that, so
    the window index never wraps.
    """
    batches = _prod(batch_shape) if batch_shape else 1
    total = rows * cols
    index_dtype = (
        torch.int32
        if max(batches, total) <= torch.iinfo(torch.int32).max
        else torch.int64
    )
    position = torch.arange(total, dtype=index_dtype, device=device)[None, :]
    start = torch.arange(batches, dtype=index_dtype, device=device)[:, None]
    keep = ((position - start) % total) < blocks
    return keep.reshape(*batch_shape, rows, cols)


def _nonzero_value(dtype, value_range):
    """A representable nonzero value inside `value_range` for `dtype`.

    None means the requested range cannot hold a nonzero value (the degenerate range an
    unsigned dtype gets for [-1, 0]): the fixture then keeps the plain input, whose
    all-zero columns store nothing anyway, so an unsigned all-zero input keeps its
    all-zero result.
    """
    if dtype == torch.bool:
        # bool has a single nonzero value and tu.make_input ignores the range.
        return True
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)
    dtype_min, dtype_max = tu.dtype_bounds(dtype)
    low, high = max(low, dtype_min), min(high, dtype_max)
    if dtype.is_complex:
        if high > 0:
            return complex(0.5, 0.5)
        if low < 0:
            return complex(-0.5, -0.5)
        return None
    if dtype.is_floating_point:
        if high > 0:
            return 0.5
        if low < 0:
            return -0.5
        return None
    if high > 0:
        return 1
    if low < 0:
        return -1
    return None


def _csc_input(dtype, shape, value_range, dense_dim=None, density=4):
    """Dense input whose CSC conversion stores the same number of blocks per entry.

    dense_dim = k splits `shape` into the batch extents shape[:rank-2-k], the matrix
    shape[rank-2-k:rank-k] and the dense tail shape[rank-k:]. Native to_sparse_csc
    rejects inputs whose batch entries do not all store the same number of (row, col)
    blocks, so the keep mask marks a `density` fraction of the matrix in every entry,
    shifted cyclically so the blocks differ per entry. A block counts as stored when
    any of its dense-tail values is nonzero, so kept blocks are pinned to a
    representable nonzero of the requested range; where that range holds no nonzero the
    input is left exactly as generated. A shape with a zero-sized matrix, batch or
    dense extent stores nothing and is returned as the plain input; those shapes are
    pinned by EMPTY_ROWS.
    """
    base = tu.make_input(dtype, shape, value_range)
    dense = 0 if dense_dim is None else dense_dim
    rank = len(shape)
    rows = shape[rank - 2 - dense]
    cols = shape[rank - 1 - dense]
    batch_shape = shape[: rank - 2 - dense]
    dense_shape = shape[rank - dense :]
    if rows * cols == 0 or _prod(batch_shape) == 0 or _prod(dense_shape) == 0:
        return base
    blocks = max(1, (rows * cols) // density)
    keep = _blocks_mask(batch_shape, rows, cols, blocks, base.device)
    if dense_shape:
        keep = keep.reshape(*batch_shape, rows, cols, *([1] * len(dense_shape))).expand(
            shape
        )
    zeros = torch.zeros((), dtype=dtype, device=base.device)
    inp = torch.where(keep, base, zeros)
    fill_value = _nonzero_value(dtype, value_range)
    if fill_value is None:
        return inp
    fill = torch.full((), fill_value, dtype=dtype, device=base.device)
    # A kept block whose values are all zero would not be counted as stored.
    return torch.where(keep & inp.eq(0), fill, inp)


def _assert_csc_equal(res, ref, source):
    """Check the CSC-specific result properties, then the shared value contract.

    Only the layout and the result device are operator specific: the shared sparse-aware
    comparator covers the extents, the dtype and the ccol/row/values payload. `source` is
    the tensor handed to the candidate, so the result must land on the same device as its
    input; the reference tensor's device does not establish that.
    """
    assert res.layout == torch.sparse_csc
    assert res.ccol_indices().device == source.device
    assert res.row_indices().device == source.device
    assert res.values().device == source.device
    tu.assert_result_equal(res, ref)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DTYPES)
def test_to_sparse_csc_dense_input(shape, value_range, dtype):
    inp = _csc_input(dtype, shape, value_range)
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp)
    res_out = flag_gems.to_sparse_csc(inp)

    _assert_csc_equal(res_out, ref_out, inp)
    # `inp` is the tensor the candidate received; the conversion must not write it.
    tu.assert_result_equal(inp, before)


# Rank-3, rank-4 and rank-5 shapes whose valid dense_dim range is wider; they
# supplement the original (2, 2, 2, 2) and (4, 3, 2, 2) products instead of replacing
# them.
_DENSE_DIM_SHAPES = [(2, 2, 2, 2), (4, 3, 2, 2)]
_DENSE_DIM_RANK5_SHAPE = (2, 2, 3, 4, 5)
_DENSE_DIM_SUPPLEMENTS = [
    ((4, 6, 3), 0, torch.float32),
    ((4, 6, 3), 1, torch.float32),
    ((4, 6, 3), None, torch.float32),
    ((4, 6, 3, 2), 0, torch.float32),
    ((4, 6, 3, 2), 1, torch.float32),
    ((4, 6, 3, 2), 2, torch.float32),
    ((4, 6, 3, 2), None, torch.float32),
] + [
    (_DENSE_DIM_RANK5_SHAPE, dense_dim, torch.float32)
    for dense_dim in range(len(_DENSE_DIM_RANK5_SHAPE) - 1)
]
DENSE_DIM_ROWS = (
    [
        (shape, dense_dim, dtype)
        for shape in _DENSE_DIM_SHAPES
        for dense_dim in range(len(shape) - 1)
        for dtype in (torch.float32, torch.int32, torch.float16)
    ]
    + [(shape, None, torch.float32) for shape in _DENSE_DIM_SHAPES]
    + _DENSE_DIM_SUPPLEMENTS
)
DENSE_DIM_CASES = tu.selected_cases(DENSE_DIM_ROWS, quick=[])


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,dtype", DENSE_DIM_CASES)
def test_to_sparse_csc_dense_dim(shape, dense_dim, dtype):
    # None is the schema default; the omitted-argument call form is checked by
    # test_to_sparse_csc_default_argument.
    inp = _csc_input(dtype, shape, ["-1", "1"], dense_dim=dense_dim)
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp, dense_dim)
    res_out = flag_gems.to_sparse_csc(inp, dense_dim)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)


# Measured native results for empty extents (ccol shape, values shape).
EMPTY_ROWS = [
    ((0, 3), None, (4,), (0,)),
    ((3, 0), None, (1,), (0,)),
    ((0, 0), None, (1,), (0,)),
    ((2, 0, 3), None, (2, 4), (2, 0)),
    ((2, 3, 0), None, (2, 1), (2, 0)),
    ((2, 3, 4, 0), None, (2, 3, 1), (2, 3, 0)),
    ((2, 3, 0, 4), None, (2, 3, 5), (2, 3, 0)),
    ((2, 3, 0, 4), 1, (2, 1), (2, 0, 4)),
    ((2, 3, 4, 0), 1, (2, 5), (2, 0, 0)),
    ((2, 0, 3, 4), 1, (2, 4), (2, 0, 4)),
    ((0, 3, 4), 1, (4,), (0, 4)),
    ((0, 2, 3), 1, (3,), (0, 3)),
    ((0, 0, 4), 1, (1,), (0, 4)),
]
EMPTY_ROWS_FULL = [
    (shape, dense_dim, ccol_shape, values_shape, dtype)
    for shape, dense_dim, ccol_shape, values_shape in EMPTY_ROWS
    for dtype in EMPTY_DTYPES
]
EMPTY_CASES = tu.selected_cases(EMPTY_ROWS_FULL, quick=[])


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,ccol_shape,values_shape,dtype", EMPTY_CASES)
def test_to_sparse_csc_empty(shape, dense_dim, ccol_shape, values_shape, dtype):
    inp = _csc_input(dtype, shape, ["-1", "1"], dense_dim=dense_dim)
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp, dense_dim)
    res_out = flag_gems.to_sparse_csc(inp, dense_dim)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)
    assert tuple(res_out.ccol_indices().shape) == ccol_shape
    assert tuple(res_out.values().shape) == values_shape


PROFILE_ROWS = [
    ("zero_nnz", (6, 4), [0, 0, 0, 0], [0, 0, 0, 0, 0]),
    ("fully_dense", (6, 4), [6, 6, 6, 6], [0, 6, 12, 18, 24]),
    ("interior_empty", (6, 4), [6, 6, 0, 6], [0, 6, 12, 12, 18]),
    ("nonuniform", (6, 4), [1, 0, 3, 6], [0, 1, 1, 4, 10]),
    ("batched", (2, 6, 4), [2, 3, 1, 0], [[0, 2, 5, 6, 6], [0, 2, 5, 6, 6]]),
]
PROFILE_ROWS_FULL = [
    (label, shape, counts, expected, dtype)
    for label, shape, counts, expected in PROFILE_ROWS
    for dtype in PROFILE_DTYPES
]
PROFILE_CASES = tu.selected_cases(PROFILE_ROWS_FULL, quick=[])


def _column_counts_input(shape, counts, dtype):
    """Matrix whose column c holds counts[c] leading nonzero rows, repeated per batch.

    Only used where the last two extents are the matrix (dense_dim None/0), so a
    batched shape repeats the same column profile in every batch entry.
    """
    rows, cols = shape[-2], shape[-1]
    matrix = torch.zeros((rows, cols), dtype=dtype, device=flag_gems.device)
    for col, count in enumerate(counts):
        if count:
            matrix[:count, col] = 1
    if len(shape) > 2:
        return matrix.expand(*shape).contiguous()
    return matrix


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("label,shape,counts,expected,dtype", PROFILE_CASES)
def test_to_sparse_csc_column_profile(label, shape, counts, expected, dtype):
    inp = _column_counts_input(shape, counts, dtype)
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp)
    res_out = flag_gems.to_sparse_csc(inp)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)
    # Per-column stored counts are native CSC semantics, not only a reference match. The
    # expectation is already the independent structural list above, so the comparison
    # needs no auxiliary device-side index buffer.
    assert res_out.ccol_indices().tolist() == expected


LAYOUT_ROWS = [
    ("contiguous", None),
    ("transpose", None),
    ("storage_offset", None),
    ("column_slice", None),
    ("expanded", None),
    ("channels_last", 1),
]
LAYOUT_ROWS_FULL = [
    (label, dense_dim, dtype)
    for label, dense_dim in LAYOUT_ROWS
    for dtype in SUPPLEMENTAL_DTYPES
]
LAYOUT_CASES = tu.selected_cases(LAYOUT_ROWS_FULL, quick=[])


def _layout_input(label, dtype):
    """Return (tensor handed to the operator, whole parent storage tensor).

    The parent is snapshotted too: for the offset, stepped and expanded views a check
    of the view alone would not show a write outside its own index set.
    """
    if label == "transpose":
        parent = _csc_input(dtype, (4, 6), ["-1", "1"])
        return parent.t(), parent
    if label == "storage_offset":
        parent = _csc_input(dtype, (8, 9), ["-1", "1"])
        return parent[1:7, 2:6], parent
    if label == "column_slice":
        parent = _csc_input(dtype, (6, 8), ["-1", "1"])
        return parent[:, ::2], parent
    if label == "expanded":
        parent = _csc_input(dtype, (1, 4), ["-1", "1"])
        return parent.expand(6, 4), parent
    if label == "channels_last":
        view = _csc_input(dtype, (2, 5, 6, 4), ["-1", "1"], dense_dim=1).to(
            memory_format=torch.channels_last
        )
        return view, view
    parent = _csc_input(dtype, (6, 4), ["-1", "1"])
    return parent, parent


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("label,dense_dim,dtype", LAYOUT_CASES)
def test_to_sparse_csc_input_layout(label, dense_dim, dtype):
    inp, parent = _layout_input(label, dtype)
    saved_view = tu.to_reference(inp)
    saved_parent = tu.to_reference(parent)

    ref_out = torch.ops.aten.to_sparse_csc(tu.to_reference(inp), dense_dim)
    res_out = flag_gems.to_sparse_csc(inp, dense_dim)

    _assert_csc_equal(res_out, ref_out, inp)
    # The conversion is a read-only observation of its operand. The whole parent storage
    # is checked as well, so a write outside the view's own index set is still caught;
    # the contiguous and channels-last rows hand the parent itself to the operator, and
    # the two checks then agree.
    tu.assert_result_equal(inp, saved_view)
    tu.assert_result_equal(parent, saved_parent)


SPARSE_SOURCE_LABELS = ["coo", "csr", "csc", "csr_batched", "csc_batched"]
SPARSE_SOURCE_ROWS = [
    (label, dtype) for label in SPARSE_SOURCE_LABELS for dtype in SUPPLEMENTAL_DTYPES
]
# Every source builds int64 index storage, so this family is gated on the static int64
# capability flag; the dense input grid above covers the same dtypes regardless.
SPARSE_SOURCE_CASES = tu.selected_cases(
    SPARSE_SOURCE_ROWS if _INT64_SUPPORTED else [], quick=[]
)


def _sparse_source(label, dtype):
    """Sparse input in one of the layouts native to_sparse_csc accepts."""
    if label.endswith("_batched"):
        dense = _csc_input(dtype, (2, 5, 4), ["-1", "1"])
    else:
        dense = _csc_input(dtype, (5, 4), ["-1", "1"])
    if label.startswith("coo"):
        return dense.to_sparse()
    if label.startswith("csr"):
        return dense.to_sparse_csr()
    return dense.to_sparse_csc()


def _sparse_state(tensor):
    """Observable state of a sparse source: layout, extents and its own components.

    The components a conversion must not touch are snapshotted directly. Comparing a
    coalesced or densified view instead would hide storage, order and coalesced-flag
    mutation. Each snapshot goes through `tu.to_reference`, which copies the component
    storage without densifying it and puts the expected copy on the configured
    reference device, so the comparison also holds for a CPU reference.
    """
    state = {
        "layout": tensor.layout,
        "shape": tuple(tensor.shape),
        "sparse_dim": tensor.sparse_dim(),
        "dense_dim": tensor.dense_dim(),
    }
    if tensor.layout == torch.sparse_coo:
        state["coalesced"] = tensor.is_coalesced()
        state["indices"] = tu.to_reference(tensor._indices())
        state["values"] = tu.to_reference(tensor._values())
    elif tensor.layout == torch.sparse_csr:
        state["crow"] = tu.to_reference(tensor.crow_indices())
        state["col"] = tu.to_reference(tensor.col_indices())
        state["values"] = tu.to_reference(tensor.values())
    else:
        state["ccol"] = tu.to_reference(tensor.ccol_indices())
        state["row"] = tu.to_reference(tensor.row_indices())
        state["values"] = tu.to_reference(tensor.values())
    return state


def _assert_sparse_state(tensor, state):
    current = _sparse_state(tensor)
    for key, expected in state.items():
        actual = current[key]
        if torch.is_tensor(expected):
            tu.assert_result_equal(actual, expected)
        else:
            assert actual == expected


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("label,dtype", SPARSE_SOURCE_CASES)
def test_to_sparse_csc_sparse_source(label, dtype):
    # The candidate operand is the original object, snapshotted before the oracle is
    # derived from it, so the state check below covers the tensor the candidate received.
    res_src = _sparse_source(label, dtype)
    before = _sparse_state(res_src)
    ref_src = tu.to_reference(res_src)

    ref_out = torch.ops.aten.to_sparse_csc(ref_src)
    res_out = flag_gems.to_sparse_csc(res_src)

    _assert_csc_equal(res_out, ref_out, res_src)
    # `res_src` is the tensor the candidate received; its own components and sparse
    # metadata must be unchanged.
    _assert_sparse_state(res_src, before)


DEFAULT_ARGUMENT_ROWS = [
    (shape, dtype)
    for shape in [(12, 7), (16, 7, 57, 32, 29)]
    for dtype in SUPPLEMENTAL_DTYPES
]
DEFAULT_ARGUMENT_CASES = tu.selected_cases(DEFAULT_ARGUMENT_ROWS, quick=[])


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dtype", DEFAULT_ARGUMENT_CASES)
def test_to_sparse_csc_default_argument(shape, dtype):
    # Call form that omits the optional dense_dim, so the schema default is used.
    inp = _csc_input(dtype, shape, ["-1", "1"])
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp)
    res_out = flag_gems.to_sparse_csc(inp)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)


# The upper bound of the valid dense_dim range is rank - 2, so rank 3 stops at 1 and
# rank 4 at 2; True behaves as the int 1 in the native schema.
DENSE_DIM_VALUE_ROWS = [
    ((3, 4, 5), 0),
    ((3, 4, 5), 1),
    ((3, 4, 5), True),
    ((2, 3, 4, 5), 2),
]
DENSE_DIM_VALUE_CASES = tu.selected_cases(DENSE_DIM_VALUE_ROWS, quick=[])


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim", DENSE_DIM_VALUE_CASES)
def test_to_sparse_csc_dense_dim_values(shape, dense_dim):
    inp = _csc_input(torch.float32, shape, ["-1", "1"], dense_dim=dense_dim)
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp, dense_dim)
    res_out = flag_gems.to_sparse_csc(inp, dense_dim)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)


SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SPECIAL_DTYPES), quick=[])


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_to_sparse_csc_special_values(dtype, scenario):
    # The payload sits in a single column: nan and inf are stored (they are not zero),
    # -0.0 is not.
    inp = tu.make_special_input(dtype, scenario).reshape(5, 1)
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp)
    res_out = flag_gems.to_sparse_csc(inp)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_to_sparse_csc_special_values_hybrid(dtype, scenario):
    # dense_dim=1 on this rank-3 shape splits () / (2, 5) / (4): one 2x5 matrix with a
    # 4-wide dense tail, so the payload fills both matrix rows across the tail and the
    # stored count is the number of nonzero payload values.
    payload = tu.make_special_input(dtype, scenario)
    inp = torch.zeros((2, 5, 4), dtype=dtype, device=flag_gems.device)
    inp[0, :, 0] = payload
    inp[1, :, 0] = payload
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp, 1)
    res_out = flag_gems.to_sparse_csc(inp, 1)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_to_sparse_csc_special_values_batched(dtype, scenario):
    # Genuinely batched: dense_dim=1 on this rank-4 shape splits (2) / (3, 5) / (4), so
    # both batch entries hold a full matrix and must store the same block count.
    payload = tu.make_special_input(dtype, scenario)
    inp = torch.zeros((2, 3, 5, 4), dtype=dtype, device=flag_gems.device)
    inp[:, 0, :, 0] = payload
    before = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp, 1)
    res_out = flag_gems.to_sparse_csc(inp, 1)

    _assert_csc_equal(res_out, ref_out, inp)
    tu.assert_result_equal(inp, before)


BACKWARD_LAYOUTS = [
    ((8, 16, 12), None),
    ((8, 16, 12), 1),
    ((6, 4, 5, 3), 2),
    ((4, 5), None),
    ((2, 3, 4, 5, 6), 3),
]
BACKWARD_ROWS = [
    (shape, dense_dim, dtype)
    for shape, dense_dim in BACKWARD_LAYOUTS
    for dtype in BACKWARD_DTYPES
]
BACKWARD_CASES = tu.selected_cases(BACKWARD_ROWS, quick=[])


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,dtype", BACKWARD_CASES)
def test_to_sparse_csc_backward(shape, dense_dim, dtype):
    inp = _csc_input(dtype, shape, ["-1", "1"], dense_dim=dense_dim)
    inp = inp.detach().requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    before = tu.to_reference(inp.detach())

    ref_out = torch.ops.aten.to_sparse_csc(ref_inp, dense_dim)
    res_out = flag_gems.to_sparse_csc(inp, dense_dim)
    _assert_csc_equal(res_out, ref_out, inp)

    # `inp` is the tensor the candidate received, so this checks the conversion did not
    # modify its operand.
    tu.assert_result_equal(inp.detach(), before)

    # Bounded, nonuniform upstream values built on the candidate device, and the
    # structure taken from the native result only, so a wrong candidate layout can
    # never feed the oracle. Each side gets its own copy moved to its own device, so a
    # reference on another device works too.
    shape_tuple = tuple(ref_out.shape)
    grad_values = torch.linspace(
        0.25, 0.75, ref_out.values().numel(), dtype=dtype, device=flag_gems.device
    ).reshape(ref_out.values().shape)
    ref_upstream = torch.sparse_csc_tensor(
        tu.to_reference(ref_out.ccol_indices()),
        tu.to_reference(ref_out.row_indices()),
        tu.to_reference(grad_values),
        size=shape_tuple,
    )
    res_upstream = torch.sparse_csc_tensor(
        ref_out.ccol_indices().to(flag_gems.device).clone(),
        ref_out.row_indices().to(flag_gems.device).clone(),
        grad_values.clone(),
        size=shape_tuple,
    )

    # The conversion only relocates stored values, so its gradient is exactly the
    # upstream written back at the stored coordinates: no reduction, no tolerance.
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=res_upstream)

    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(res_grad, ref_upstream.to_dense())


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape", [(), (1,), (256,)])
def test_to_sparse_csc_rank_too_low(shape):
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    # Measured native category for a rank < 2 input: IndexError, raised by the shape
    # arithmetic before a kernel is selected ('Dimension out of range').
    with pytest.raises(IndexError):
        flag_gems.to_sparse_csc(inp)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("dense_dim", [1.5, "0", [0]])
def test_to_sparse_csc_dense_dim_type(dense_dim):
    inp = torch.zeros((4, 4), dtype=torch.float32, device=flag_gems.device)
    # Measured native category: RuntimeError from the schema/argument parser; TypeError
    # is the Python-level class a signature check raises for the same argument.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.to_sparse_csc(inp, dense_dim)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize(
    "shape,dense_dim",
    [((4, 4), -1), ((4, 4), 1), ((2, 3, 4), -1), ((2, 3, 4), 2)],
)
def test_to_sparse_csc_dense_dim_out_of_range(shape, dense_dim):
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    # Measured native category: RuntimeError for every dense_dim outside [0, rank - 2].
    with pytest.raises(RuntimeError):
        flag_gems.to_sparse_csc(inp, dense_dim)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("value", [1.0, [1.0, 2.0]])
def test_to_sparse_csc_requires_tensor(value):
    # Measured native category: RuntimeError from the dispatcher; TypeError is the
    # Python-level class a signature check raises for the same argument.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.to_sparse_csc(value)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize(
    "shape,dense_dim",
    [((0, 3, 4), None), ((0, 2, 3), None), ((2, 0, 3, 4), 0)],
)
def test_to_sparse_csc_empty_batch(shape, dense_dim):
    # A zero batch extent makes the batch product zero, which native rejects with a
    # RuntimeError; an empty matrix extent is legal and covered by
    # test_to_sparse_csc_empty.
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.to_sparse_csc(inp, dense_dim)


def _unsupported_input(kind):
    """Input without a native CSC kernel: a quantized tensor or a BSR source.

    Both fixtures are built on the active device outside any pytest.raises block; this
    build constructs both successfully (the quantized tensor as a quantized strided
    tensor, the blocked-sparse source as a torch.sparse_bsr tensor), so the expected
    errors come from the operator, not from the construction.
    """
    if kind == "bsr":
        return torch.randn(4, 4, device=flag_gems.device).to_sparse_bsr(2)
    return torch.quantize_per_tensor(
        torch.randn(4, 4, device=flag_gems.device), 0.1, 0, torch.qint8
    )


# The BSR source carries int64 index tensors, so it joins the family only where int64
# storage exists (static flag); the quantized tensor needs no such gate.
_UNSUPPORTED_KINDS = [("quantized", RuntimeError)] + (
    [("bsr", RuntimeError)] if _INT64_SUPPORTED else []
)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("kind,error", _UNSUPPORTED_KINDS)
def test_to_sparse_csc_unsupported_input(kind, error):
    # Measured native errors: a quantized input has no Quantized* CSC kernel
    # (NotImplementedError, a RuntimeError) and a blocked-sparse source is rejected
    # with a RuntimeError because only SparseCsr/SparseCsc sources are accepted.
    inp = _unsupported_input(kind)
    with pytest.raises(error):
        flag_gems.to_sparse_csc(inp)
