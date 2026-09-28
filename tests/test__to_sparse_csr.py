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

from . import accuracy_utils as utils
from . import test_utils as tu

_DEVICE = flag_gems.device

# Dtype tables come from the checkout's static capability flags, so nothing here
# allocates a tensor or calls the operator at import/collection time.
_DTYPES = [torch.int8, torch.uint8, torch.float16, torch.float32, torch.int32]
if utils.fp8_is_supported:
    _DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.int64_is_supported:
    _DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _DTYPES.append(torch.float64)
_DTYPES.append(torch.bool)

_FLOAT_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _FLOAT_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _FLOAT_DTYPES.append(torch.float64)
if utils.fp8_is_supported:
    _FLOAT_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]

_BACKWARD_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _BACKWARD_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _BACKWARD_DTYPES.append(torch.float64)

# Auxiliary index tensors (CSR crow/col_indices buffers) are materialised with the
# widest index dtype the backend supports; no fixture needs one wider than that.
_INDEX_DTYPE = torch.int64 if utils.int64_is_supported else torch.int32

# Only the out-buffer dtype negative needs a second dtype; float64 keeps the
# mismatch unmistakable wherever the backend can hold it.
_MISMATCH_DTYPE = torch.float64 if utils.fp64_is_supported else torch.float16

# Rank >= 2 is required (rank 0/1 are native rejects, see the negative tests), and
# dense_dim = rank - 2 keeps the conversion non-batched, so this plain value-range
# grid carries no per-batch 'equal number of specified elements' constraint.
_GRID_ROWS = tu.selected_cases(
    [(shape, len(shape) - 2) for shape in tu.selected_shapes() if len(shape) >= 2],
    quick=[((2, 19, 7), 1)],
)

# Batched CSR (dense_dim < rank - 2) requires the same number of specified
# elements in every batch, so these rows use the patterned builder below.
_BATCHED_ROWS = tu.selected_cases([((4, 8, 16), 0), ((16, 8, 16, 32), 1)], quick=[])

# dense_dim call forms: 'omitted' passes no argument (schema default), 'none'
# passes None explicitly, 'int' passes the value. These rows are parameter
# coverage, so the whole table is default-only.
_PARAM_ROWS = tu.selected_cases(
    [
        ((1024, 1024), None, "omitted"),
        ((1024, 1024), None, "none"),
        ((1024, 1024), 0, "int"),
        ((20, 320, 15), None, "none"),
        ((20, 320, 15), 0, "int"),
        ((20, 320, 15), 1, "int"),
        ((16, 128, 64, 60), 0, "int"),
        ((16, 128, 64, 60), 1, "int"),
        ((16, 128, 64, 60), 2, "int"),
        ((16, 7, 57, 32, 29), 0, "int"),
        ((16, 7, 57, 32, 29), 1, "int"),
        ((16, 7, 57, 32, 29), 2, "int"),
        ((16, 7, 57, 32, 29), 3, "int"),
    ],
    quick=[],
)

# Real .out overload rows: the candidate must return the caller's buffer and must
# rewrite it, so each buffer is a valid CSR holding a different structure with
# exactly the required number of specified elements. The (4, 8, 16) row also
# covers a batched buffer whose per-row pattern is expanded over the batches.
_OUT_ROWS = tu.selected_cases(
    [
        ((1024, 1024), 0, "int"),
        ((1024, 1024), None, "omitted"),
        ((20, 320, 15), 1, "int"),
        ((4, 8, 16), 0, "int"),
    ],
    quick=[((2, 19, 7), 1, "int")],
)

# Zero-extent operands: an empty sparse extent or an empty dense tail converts
# natively and the CSR result must keep the empty dimension instead of dropping
# it. An empty BATCH extent is a native error and is covered by the zero-batch
# negative test below, so every row here keeps a non-zero batch product.
_EMPTY_ROWS = tu.selected_cases(
    [((4, 0), 0), ((0, 4), 0), ((0, 4, 8), 1), ((4, 0, 8), 0), ((4, 8, 0), 1)],
    quick=[],
)

_LAYOUTS = tu.selected_cases(["strided", "offset", "transposed"], quick=[])

_BACKWARD_ROWS = tu.selected_cases(
    [((20, 320, 15), 1), ((16, 128, 64, 60), 2)], quick=[]
)

# nan / inf / mixed for every supported floating dtype. tu.special_value_cases
# scopes float8_e4m3fn to nan (that dtype cannot represent infinity) and keeps
# nan/inf/mixed for float8_e5m2.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])

# Measured: the valid dense_dim range is [0, rank - 2] and the native error names
# that range, so the rank-4 shape also rejects 5 and the rank-5 shape rejects 4.
# Negative coverage is required in both modes, so the quick subset keeps every row.
_DENSE_DIM_REJECT_ROWS = tu.selected_cases(
    [
        ((20, 320, 15), -1),
        ((20, 320, 15), 2),
        ((16, 128, 64, 60), 5),
        ((16, 7, 57, 32, 29), 4),
    ],
    quick=[
        ((20, 320, 15), -1),
        ((20, 320, 15), 2),
        ((16, 128, 64, 60), 5),
        ((16, 7, 57, 32, 29), 4),
    ],
)

# Rank 0 and rank 1 leave the valid dense_dim range [0, rank - 2] empty, so every
# dense_dim value is rejected with the same measured RuntimeError. Kept in full in
# the quick subset for the same reason.
_RANK_REJECT_ROWS = tu.selected_cases([(), (1,), (256,)], quick=[(), (1,), (256,)])

# Measured: the batch axes (everything before the two sparse matrix axes) must
# have a non-zero product, so an empty batch extent is rejected while an empty
# sparse extent or dense tail converts. Kept in full in the quick subset because
# negative coverage is required in both modes.
_BATCH_REJECT_ROWS = tu.selected_cases(
    [((0, 4, 8), 0), ((2, 0, 4, 8), 0)],
    quick=[((0, 4, 8), 0), ((2, 0, 4, 8), 0)],
)

_OUT_REJECTS = tu.selected_cases(
    ["nnz", "dtype", "layout"], quick=["nnz", "dtype", "layout"]
)


def _row_counts(rows, cols):
    """Specified elements per row of the patterned operand.

    Empty rows appear at the leading, middle and trailing position (only for
    shapes with at least four rows, so small shapes stay non-empty), one row is
    completely full for narrow matrices, and the rest hold one or two elements.
    """
    counts = [0] * max(rows, 0)
    if rows <= 0 or cols <= 0:
        return counts
    for row in range(rows):
        if rows >= 4 and (row == 0 or row == rows - 1 or row == rows // 2):
            continue
        if row % 4 == 3 and cols <= 64:
            counts[row] = cols
        elif row % 4 == 1:
            counts[row] = 1
        else:
            counts[row] = min(2, cols)
    return counts


def _geometry(shape, dense_dim):
    """(rows, columns, batch count, dense block extent) of a dense -> CSR call."""
    shape = tuple(shape)
    rank = len(shape)
    rows = shape[rank - dense_dim - 2]
    cols = shape[rank - dense_dim - 1]
    batches = 1
    for size in shape[: rank - dense_dim - 2]:
        batches *= size
    block = 1
    for size in shape[rank - dense_dim :] if dense_dim else ():
        block *= size
    return rows, cols, batches, block


def _pattern_host(shape, dense_dim):
    """Dense operand with its own zero pattern, built entirely on the host.

    Index arithmetic stays on host int64 and the payload is float32, so the
    fixture needs neither device int64 support nor a device scatter kernel; the
    caller moves the result to the test device and casts it to the test dtype.

    Batched CSR requires an equal number of specified elements per batch, so the
    row pattern repeats over every batch while the column offsets and payload
    values differ. Magnitudes stay in 1..9 so int8/float8 elements cannot
    saturate to zero and silently change the specified-element count.
    """
    shape = tuple(shape)
    dense = torch.zeros(shape, dtype=torch.float32)
    if len(shape) < 2:
        return dense
    rows, cols, batches, block = _geometry(shape, dense_dim)
    counts = torch.tensor(_row_counts(rows, cols), dtype=torch.int64)
    total = int(counts.sum().item())
    if total == 0 or batches == 0 or cols == 0:
        return dense
    crow = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    row_of = torch.repeat_interleave(torch.arange(rows), counts)
    within = torch.arange(total) - torch.repeat_interleave(crow[:-1], counts)
    span = (cols - counts).clamp(min=1)
    offsets = (
        torch.arange(rows)[:, None] * 3 + torch.arange(batches)[None, :] * 5 + 1
    ) % span[:, None]
    col = offsets[row_of] + within[:, None]
    payload = torch.arange(block)
    batch = torch.arange(batches)
    mag = (
        (
            within[:, None, None] * 3
            + row_of[:, None, None] * 5
            + col[:, :, None]
            + batch[None, :, None] * 7
            + payload[None, None, :]
        )
        % 9
    ) + 1
    sign = torch.where(
        ((within[:, None, None] + row_of[:, None, None] + batch[None, :, None]) % 2)
        == 0,
        -1.0,
        1.0,
    )
    values = mag.to(torch.float32) * sign
    flat = (
        (
            batch[None, :, None] * rows * cols
            + row_of[:, None, None] * cols
            + col[:, :, None]
        )
        * block
        + payload[None, None, :]
    ).reshape(-1)
    dense.view(-1).index_put_((flat,), values.reshape(-1))
    return dense


def _make_dense_pattern(shape, dense_dim, dtype, device):
    dense = _pattern_host(shape, dense_dim)
    if dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
        # These dtypes are exercised with positive fixture values; the
        # magnitudes are already nonzero, so the specified-element pattern is
        # unchanged.
        dense = dense.abs()
    return dense.to(device=device, dtype=dtype)


def _specified_counts(inp, dense_dim):
    """Per-row specified-element counts that the patterned operand implies.

    A batched CSR conversion requires the same number of specified elements in
    every batch, and the fixture gives every batch the same per-row counts, so
    one whole batch matrix describes the counts of the entire buffer; collapsing
    the batches with any() would instead union their deliberately different
    column coordinates and overstate the count. The hybrid dense tail is reduced
    away first so one count remains per sparse row, and a zero extent yields a
    zero count for every row instead of an empty vector. The reductions run on
    the CPU copy, so the fixture needs no device int64 kernel.
    """
    nonzero = inp != 0
    if dense_dim:
        nonzero = nonzero.any(
            dim=tuple(range(nonzero.dim() - dense_dim, nonzero.dim()))
        )
    nonzero = nonzero.cpu()
    rows = int(nonzero.shape[-2])
    if nonzero.numel() == 0:
        return [0] * rows
    if nonzero.dim() == 2:
        return nonzero.sum(-1).tolist()
    cols = int(nonzero.shape[-1])
    return nonzero.reshape(-1, rows, cols)[0].sum(-1).tolist()


def _csr_buffer(shape, dense_dim, counts, dtype, device):
    """Valid CSR tensor with the given per-row counts and sorted distinct columns.

    Columns are drawn inside each row's own range, so the buffer satisfies the
    CSR invariants for any shape, including zero extents and a single column, and
    a batched call expands the same per-row pattern over every batch.
    """
    shape = tuple(shape)
    rank = len(shape)
    rows = len(counts)
    cols = shape[rank - dense_dim - 1]
    lead = shape[: rank - dense_dim - 2]
    batches = 1
    for size in lead:
        batches *= size
    crow_row = [0]
    for count in counts:
        crow_row.append(crow_row[-1] + count)
    nnz = crow_row[-1]
    col_row = []
    for row, count in enumerate(counts):
        start = (row * 5 + 2) % max(cols - count, 1)
        col_row.extend(range(start, start + count))
    crow = torch.tensor(crow_row * batches, dtype=_INDEX_DTYPE, device=device).reshape(
        lead + (rows + 1,)
    )
    col = torch.tensor(col_row * batches, dtype=_INDEX_DTYPE, device=device).reshape(
        lead + (nnz,)
    )
    if dtype.is_floating_point:
        sentinel = 7.5
    elif dtype is torch.bool:
        sentinel = True
    else:
        sentinel = 7
    vshape = lead + (nnz,) + (shape[rank - dense_dim :] if dense_dim else ())
    values = torch.full(vshape, sentinel, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(
        crow, col, values, size=shape, dtype=dtype, device=device
    )


def _make_wrong_structure_buffer(inp, dense_dim, dtype):
    """CSR buffer with the required element count but a different structure."""
    return _csr_buffer(
        tuple(inp.shape),
        dense_dim,
        _specified_counts(inp, dense_dim),
        dtype,
        inp.device,
    )


def _make_nnz_mismatch_buffer(inp, dense_dim, dtype):
    """Valid CSR buffer whose element count differs from the conversion's."""
    shape = tuple(inp.shape)
    counts = list(_specified_counts(inp, dense_dim))
    cols = shape[len(shape) - dense_dim - 1]
    for index in range(len(counts) - 1, -1, -1):
        if counts[index] > 0:
            counts[index] -= 1
            break
    else:
        counts[-1] = min(1, cols)
    return _csr_buffer(shape, dense_dim, counts, dtype, inp.device)


def _build_input(shape, dense_dim, dtype):
    return _make_dense_pattern(shape, dense_dim, dtype, _DEVICE)


def _dense_dim_for(fixture_dim, form):
    # An omitted argument and an explicit None both make the operator use the
    # schema default; only the invocation differs.
    return fixture_dim if form == "int" else 0


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("case", _GRID_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test__to_sparse_csr_shape_value_range(case, value_range, dtype):
    shape, dense_dim = case
    inp = tu.make_input(dtype, shape, value_range)
    snapshot = tu.to_reference(inp)
    ref_out = torch.ops.aten._to_sparse_csr(tu.to_reference(inp), dense_dim)
    res_out = flag_gems._to_sparse_csr(inp, dense_dim)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("case", _BATCHED_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__to_sparse_csr_batched(case, dtype):
    shape, dense_dim = case
    inp = _build_input(shape, dense_dim, dtype)
    snapshot = tu.to_reference(inp)
    ref_out = torch.ops.aten._to_sparse_csr(tu.to_reference(inp), dense_dim)
    res_out = flag_gems._to_sparse_csr(inp, dense_dim)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("case", _PARAM_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__to_sparse_csr_dense_dim_forms(case, dtype):
    shape, dense_dim, form = case
    inp = _build_input(shape, _dense_dim_for(dense_dim, form), dtype)
    snapshot = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)
    if form == "omitted":
        ref_out = torch.ops.aten._to_sparse_csr(ref_inp)
        res_out = flag_gems._to_sparse_csr(inp)
    else:
        ref_out = torch.ops.aten._to_sparse_csr(ref_inp, dense_dim)
        res_out = flag_gems._to_sparse_csr(inp, dense_dim)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("case", _OUT_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__to_sparse_csr_out(case, dtype):
    shape, dense_dim, form = case
    fixture_dim = _dense_dim_for(dense_dim, form)
    inp = _build_input(shape, fixture_dim, dtype)
    snapshot = tu.to_reference(inp)
    ref_inp = tu.to_reference(inp)
    # Both buffers come from the operand's own structure, i.e. independently of
    # the reference result, and are valid CSR holding exactly the required number
    # of specified elements per block.
    ref_buf = _make_wrong_structure_buffer(ref_inp, fixture_dim, dtype)
    res_buf = _make_wrong_structure_buffer(inp, fixture_dim, dtype)
    if form == "omitted":
        torch.ops.aten._to_sparse_csr.out(ref_inp, out=ref_buf)
        res_out = flag_gems._to_sparse_csr(inp, out=res_buf)
    else:
        torch.ops.aten._to_sparse_csr.out(ref_inp, dense_dim, out=ref_buf)
        res_out = flag_gems._to_sparse_csr(inp, dense_dim, out=res_buf)
    # The candidate must hand back the caller's buffer, rewritten with the real
    # conversion, so its previous contents cannot satisfy the comparison.
    assert res_out is res_buf
    assert res_buf.device == inp.device
    tu.assert_result_equal(res_buf, ref_buf)
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("case", _EMPTY_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__to_sparse_csr_empty(case, dtype):
    shape, dense_dim = case
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    snapshot = tu.to_reference(inp)
    ref_out = torch.ops.aten._to_sparse_csr(tu.to_reference(inp), dense_dim)
    res_out = flag_gems._to_sparse_csr(inp, dense_dim)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__to_sparse_csr_non_contiguous(layout, dtype):
    # Views cut from the patterned operand keep its zero pattern, and the offset
    # view additionally carries a nonzero storage offset. The whole parent is
    # snapshotted so a write outside the view cannot stay invisible.
    if layout == "strided":
        parent = _build_input((16, 32), 0, dtype)
        inp = parent[:, ::2]
    elif layout == "offset":
        parent = _build_input((20, 20), 0, dtype)
        inp = parent[2:12, 3:13]
    else:
        parent = _build_input((24, 16), 0, dtype)
        inp = parent.t()
    snapshot = tu.to_reference(parent)
    ref_out = torch.ops.aten._to_sparse_csr(tu.to_reference(inp), 0)
    res_out = flag_gems._to_sparse_csr(inp, 0)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(parent, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("case", _BACKWARD_ROWS)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test__to_sparse_csr_backward(case, dtype):
    shape, dense_dim = case
    values = _build_input(shape, dense_dim, dtype)
    snapshot = tu.to_reference(values)
    # An independent patterned operand supplies the upstream CSR gradient, whose
    # payload varies over every column, batch and dense-tail position, so the
    # gradients are nonuniform and a reordered dense tail cannot stay invisible.
    upstream = torch.ops.aten._to_sparse_csr(
        _build_input(shape, dense_dim, dtype), dense_dim
    )
    # The reference path must not consume a candidate-device tensor, so the
    # upstream gradient is moved to the configured reference like any other input.
    ref_upstream = tu.to_reference(upstream)
    ref_inp = tu.to_reference(values).requires_grad_(True)
    res_inp = values.requires_grad_(True)
    ref_out = torch.ops.aten._to_sparse_csr(ref_inp, dense_dim)
    res_out = flag_gems._to_sparse_csr(res_inp, dense_dim)
    tu.assert_result_equal(res_out, ref_out)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_outputs=upstream)
    # The gradient relocates stored values without any arithmetic on them.
    tu.assert_result_equal(res_grad, ref_grad)
    assert res_out.device == res_inp.device
    assert res_grad.device == res_inp.device
    tu.assert_result_equal(values, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__to_sparse_csr_nan_inf(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(5, 1)
    snapshot = tu.to_reference(inp)
    ref_out = torch.ops.aten._to_sparse_csr(tu.to_reference(inp), 0)
    res_out = flag_gems._to_sparse_csr(inp, 0)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, snapshot)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape", _RANK_REJECT_ROWS)
def test__to_sparse_csr_rejects_rank_below_two(shape):
    # Measured: dense_to_sparse_csr requires rank >= 2 because the valid dense_dim
    # range [0, rank - 2] is empty for rank 0 and rank 1.
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_csr(inp, 0)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape,dense_dim", _BATCH_REJECT_ROWS)
def test__to_sparse_csr_rejects_zero_batch(shape, dense_dim):
    # Measured: 'to_sparse_csr: Expected product of batch dimensions to be
    # non-zero.' Only the batch axes are constrained; the zero-extent rows above
    # (empty sparse extent, empty dense tail) convert natively.
    inp = torch.zeros(shape, dtype=torch.float32, device=_DEVICE)
    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_csr(inp, dense_dim)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("shape,dense_dim", _DENSE_DIM_REJECT_ROWS)
def test__to_sparse_csr_rejects_out_of_range_dense_dim(shape, dense_dim):
    # Measured: the native error names the [0, rank - 2] range.
    inp = _build_input(shape, len(shape) - 2, torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_csr(inp, dense_dim)


@pytest.mark.to_sparse_csr
def test__to_sparse_csr_rejects_non_integer_dense_dim():
    # Measured: the native schema rejects a fractional dense_dim with RuntimeError;
    # a TypeError is equally valid for a Python implementation.
    inp = _build_input((10, 10), 0, torch.float32)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._to_sparse_csr(inp, 1.5)


@pytest.mark.to_sparse_csr
def test__to_sparse_csr_rejects_non_tensor_input():
    # Measured: the native schema rejects a non-tensor first argument.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._to_sparse_csr([1.0, 2.0], 0)


@pytest.mark.to_sparse_csr
@pytest.mark.parametrize("invalid", _OUT_REJECTS)
def test__to_sparse_csr_out_rejects_invalid_buffer(invalid):
    # Measured native early checks: a buffer with a different specified-element
    # count fails in copy_, a buffer of another dtype fails the dtype check, and a
    # strided buffer fails the sparse-compressed layout check. The operand and the
    # buffers are valid for this operator; only the buffer is wrong for the call.
    shape = (8, 8)
    inp = _build_input(shape, 0, torch.float32)
    if invalid == "nnz":
        buf = _make_nnz_mismatch_buffer(inp, 0, torch.float32)
    elif invalid == "dtype":
        buf = _make_wrong_structure_buffer(inp, 0, _MISMATCH_DTYPE)
    else:
        buf = torch.zeros(shape, dtype=torch.float32, device=_DEVICE)
    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_csr(inp, 0, out=buf)
