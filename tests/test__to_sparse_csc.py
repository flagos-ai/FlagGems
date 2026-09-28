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

"""Correctness tests for ``torch.ops.aten._to_sparse_csc``.

The conversion drops exact zeros only, so no rounding is introduced and every
positive case is compared exactly with ``tu.assert_result_equal``. ``dense_dim``
selects the trailing payload axes: those stay dense, the two axes before them
form the sparse matrix and the remaining leading axes are batch axes. The
operator needs rank >= 2, so only the rank >= 2 spec shapes enter the positive
grid; rank errors are covered by the negative table.
"""

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Sentinel for a call that omits ``dense_dim`` entirely, as opposed to passing
# an explicit ``None``. It is a plain string so serialized metadata (shape
# files, case ids) compares equal to the in-process value; comparisons use
# equality, never identity.
_OMITTED = "omitted"

# Static backend capability flags, read once from the runtime device
# description. Dtype support is never probed at import, collection or run time.
_CAPABILITY = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}


def _supported(dtype):
    return _CAPABILITY.get(dtype, True)


# The nine required dtypes plus float64. Rows whose static capability flag
# reports no backend support are dropped before parametrization, never skipped
# at run time.
DTYPES = [
    dtype for dtype in list(tu.REQUIRED_DTYPES) + [torch.float64] if _supported(dtype)
]

# ``_to_sparse_csc`` builds two sparse axes and therefore needs rank >= 2; the
# 0-D and 1-D spec shapes cannot express a positive CSC conversion.
SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]


def _dense_dim_args(dense_dim):
    return () if dense_dim == _OMITTED else (dense_dim,)


def _axes(shape, dense_dim):
    # (batch, rows, cols, payload) for this ``dense_dim``.
    dd = 0 if dense_dim is None or dense_dim == _OMITTED else dense_dim
    return (
        tuple(shape[: len(shape) - dd - 2]),
        shape[len(shape) - dd - 2],
        shape[len(shape) - dd - 1],
        tuple(shape[len(shape) - dd :]),
    )


def _scalar_for(dtype, magnitude):
    if dtype.is_floating_point:
        return float(magnitude)
    limits = torch.iinfo(dtype)
    return int(min(max(int(magnitude), limits.min), limits.max))


def _dense_input(dtype, shape, value_range, dense_dim=_OMITTED):
    batch, rows, cols, payload = _axes(shape, dense_dim)
    nb = math.prod(batch) if batch else 1
    if nb == 1 or rows == 0:
        return tu.make_input(dtype, shape, value_range)
    # The native batched conversion needs one common count of specified
    # elements, so every slice is a cyclic row shift of a single prototype: the
    # per-column counts (column pointers) stay identical while row coordinates
    # and stored values differ between slices, which still catches batch-index
    # errors. The rotation index stays in the device int32 type throughout,
    # including the ``index_select`` argument: it is never widened to int64, and
    # its values never exceed the row count.
    prototype = tu.make_input(dtype, (rows, cols, *payload), value_range)
    batch_index = torch.arange(nb, dtype=torch.int32, device=prototype.device)
    row_index = torch.arange(rows, dtype=torch.int32, device=prototype.device)
    rotation = (batch_index.reshape(nb, 1) + row_index.reshape(1, rows)) % rows
    index = rotation.reshape(-1)
    return prototype.reshape(rows, -1).index_select(0, index).reshape(shape)


def _coordinates(pattern, rows, cols, batch_index, batch_size):
    total = rows * cols
    if pattern == "zero" or total == 0:
        return []
    if pattern == "single":
        return [(batch_index % rows, 0)]
    if pattern == "full":
        return [(row, col) for col in range(cols) for row in range(rows)]
    if pattern == "empty_edges":
        # Leading and trailing columns stay empty; interior columns specify one
        # block each, at a different row per slice.
        return [((batch_index + col) % rows, col) for col in range(1, max(1, cols - 1))]
    if pattern == "batch_distinct":
        # Equal-size windows over the column-major cell list, started at a
        # different offset per slice: the specified-element count is shared,
        # but each slice uses other columns and other rows inside them, so the
        # column pointers differ as well.
        count = max(1, total // 2)
        start = batch_index * max(1, total // batch_size)
        cells = []
        for index in range(count):
            flat = (start + index) % total
            cells.append((flat % rows, flat // rows))
        return cells
    return [(batch_index % rows, col) for col in range(cols)]


def _pattern_input(dtype, shape, dense_dim, pattern):
    batch, rows, cols, payload = _axes(shape, dense_dim)
    nb = math.prod(batch) if batch else 1
    payload_size = math.prod(payload) if payload else 1
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    blocks = inp.reshape(nb, rows, cols, *payload)
    for batch_index in range(nb):
        for offset, (row, col) in enumerate(
            _coordinates(pattern, rows, cols, batch_index, nb)
        ):
            value = _scalar_for(dtype, 1 + batch_index + offset % 7)
            if payload:
                # Distinct payload components expose dense-tail permutations.
                components = torch.arange(
                    1, payload_size + 1, dtype=torch.float32, device=inp.device
                )
                blocks[batch_index, row, col] = (
                    (value + components).reshape(*payload).to(dtype)
                )
            else:
                blocks[batch_index, row, col] = value
    return inp


def _out_columns(cols):
    return list(range(0, cols, 2))


def _out_input(dtype, shape, dense_dim, columns):
    # One specified block per selected column, rotated per batch slice, so every
    # slice carries the same number of specified elements.
    batch, rows, cols, payload = _axes(shape, dense_dim)
    nb = math.prod(batch) if batch else 1
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    blocks = inp.reshape(nb, rows, cols, *payload)
    for batch_index in range(nb):
        for offset, col in enumerate(columns):
            blocks[batch_index, (col + batch_index) % rows, col] = _scalar_for(
                dtype, 1 + offset
            )
    return inp


def _out_buffer(
    dtype, shape, dense_dim, columns, device, row_shift=1, factor=1.0 / 64.0
):
    # A valid CSC tensor on ``device`` with the same shape and the same number
    # of specified elements as ``_out_input`` for the same ``columns``, but with
    # shifted row coordinates and unrelated negative values: a candidate that
    # returns its ``out`` argument unwritten cannot match the comparison below.
    batch, rows, cols, payload = _axes(shape, dense_dim)
    nb = math.prod(batch) if batch else 1
    block_count = len(columns)
    payload_size = math.prod(payload) if payload else 1
    ccol_indices, row_indices, values = [], [], []
    for batch_index in range(nb):
        counts = torch.zeros(cols + 1, dtype=torch.int32, device=device)
        for col in columns:
            counts[col + 1] = 1
        # The CSC index buffers are int32 by contract, so the accumulation is
        # requested in int32 instead of relying on the default int64 promotion.
        ccol_indices.append(torch.cumsum(counts, 0, dtype=torch.int32))
        row_indices.append(
            torch.tensor(
                [(col + row_shift + batch_index) % rows for col in columns],
                dtype=torch.int32,
                device=device,
            )
        )
        magnitudes = -(
            torch.arange(
                1, block_count * payload_size + 1, dtype=torch.float32, device=device
            )
            * factor
        )
        values.append(magnitudes.reshape(block_count, *payload).to(dtype))
    return torch.sparse_csc_tensor(
        torch.stack(ccol_indices).reshape(*batch, cols + 1),
        torch.stack(row_indices).reshape(*batch, block_count),
        torch.stack(values).reshape(*batch, block_count, *payload),
        size=shape,
    )


# dense_dim coverage: omitted, explicit None and every valid integer 0..rank-2
# (the rank-5 shape reaches dense_dim 3). The supplemental positive families
# are default only; quick mode keeps the main grid and the negative table.
_DENSE_DIM_CASES = tu.selected_cases(
    [
        (shape, dense_dim)
        for shape in SHAPES
        for dense_dim in (_OMITTED, None, *range(len(shape) - 1))
    ],
    quick=[],
)

# Sparse structures: empty, single nonzero, fully dense, sparse columns, empty
# leading/interior/trailing columns, padded payload and per-slice distinct
# coordinates with dense-tail components.
_STRUCTURE_CASES = tu.selected_cases(
    [
        ((4, 6), _OMITTED, "zero"),
        ((4, 6), _OMITTED, "single"),
        ((4, 6), _OMITTED, "full"),
        ((4, 6), _OMITTED, "sparse"),
        ((5, 3), _OMITTED, "empty_edges"),
        ((3, 4, 6), 0, "batch_distinct"),
        ((2, 4, 6), 0, "empty_edges"),
        ((2, 2, 3, 5), 1, "batch_distinct"),
        ((3, 2, 4, 6), 0, "batch_distinct"),
        ((2, 2, 3, 4), 1, "full"),
    ],
    quick=[],
)

# Zero extents, asymmetric and single-element matrices.
_EDGE_CASES = tu.selected_cases(
    [
        ((0, 5), _OMITTED, torch.float32),
        ((5, 0), _OMITTED, torch.float32),
        ((0, 0), _OMITTED, torch.float32),
        ((2, 0, 3), 1, torch.float32),
        ((2, 3, 0), 1, torch.float32),
        ((1, 1), _OMITTED, torch.int32),
        ((2, 19, 7), _OMITTED, torch.float32),
    ],
    quick=[],
)

# Non-contiguous slice, storage-offset view and transposed view of a larger
# storage buffer. ``flip``/``[::-1]`` are copies or rejected, so no
# negative-stride view is claimed here.
_VIEW_CASES = tu.selected_cases(
    [
        ("slice", torch.float32),
        ("offset", torch.float32),
        ("transpose", torch.float32),
        ("transpose", torch.int32),
    ],
    quick=[],
)

# NaN / Inf matrix derived from the supported dtype list itself (e4m3fn cannot
# represent infinity, so only its nan-only scenario is generated).
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(DTYPES), quick=[])

# out= workloads: the omitted call form, an explicit None plus nondefault
# dense_dim on the spec's large shapes.
_OUT_CASES = [
    (shape, dense_dim, dtype)
    for shape, dense_dim, dtype in tu.selected_cases(
        [
            ((1024, 1024), _OMITTED, torch.float32),
            ((1024, 1024), None, torch.float32),
            ((20, 320, 15), 1, torch.float16),
            ((20, 320, 15), 0, torch.float32),
            ((16, 128, 64, 60), 0, torch.float32),
            ((16, 7, 57, 32, 29), 2, torch.bfloat16),
            ((16, 7, 57, 32, 29), 3, torch.float32),
        ],
        quick=[],
    )
    if _supported(dtype)
]

# Backward: nonuniform upstream gradients, dense_dim variants and explicit
# sparse zero patterns in addition to the large rows.
_BACKWARD_CASES = [
    (shape, dense_dim, dtype, pattern)
    for shape, dense_dim, dtype, pattern in tu.selected_cases(
        [
            ((1024, 1024), _OMITTED, torch.float32, None),
            ((20, 320, 15), 1, torch.float16, None),
            ((16, 128, 64, 60), 0, torch.bfloat16, None),
            ((16, 7, 57, 32, 29), 2, torch.float32, None),
            ((4, 6), None, torch.float32, "empty_edges"),
            ((2, 2, 3, 5), 1, torch.float32, "batch_distinct"),
        ],
        quick=[],
    )
    if _supported(dtype)
]

_NEGATIVE_ROWS = [
    # Rank-deficient calls: an omitted dense_dim raises IndexError, an
    # explicit one raises RuntimeError.
    ((256,), _OMITTED, IndexError),
    ((), _OMITTED, IndexError),
    ((256,), 0, RuntimeError),
    ((), 0, RuntimeError),
    # dense_dim outside the valid [0, rank - 2] window.
    ((1024, 1024), 1, RuntimeError),
    ((20, 320, 15), 2, RuntimeError),
    ((1024, 1024), -3, RuntimeError),
    # The schema casts dense_dim to Optional[int]; a float is rejected.
    ((1024, 1024), 1.5, RuntimeError),
]

# Every row is a distinct, cheap error path, so the negative table stays
# complete in quick mode as well.
_NEGATIVE_CASES = tu.selected_cases(_NEGATIVE_ROWS, quick=_NEGATIVE_ROWS)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DTYPES)
def test__to_sparse_csc(shape, value_range, dtype):
    inp = _dense_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_csc(ref_inp)
    res_out = flag_gems._to_sparse_csc(inp)

    tu.assert_result_equal(res_out, ref_out)
    # A conversion must leave its input untouched; ``tu.to_reference`` copied the
    # storage up front, so ``ref_inp`` still holds the pre-call contents.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim", _DENSE_DIM_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__to_sparse_csc_dense_dim(shape, dense_dim, dtype):
    # dense_dim selects which axes are sparse, so every dense_dim > 0 yields
    # different column pointers and row indices: a candidate that ignores it
    # cannot match the reference.
    inp = _dense_input(dtype, shape, ["-1", "1"], dense_dim)
    ref_inp = tu.to_reference(inp)
    args = _dense_dim_args(dense_dim)

    ref_out = torch.ops.aten._to_sparse_csc(ref_inp, *args)
    res_out = flag_gems._to_sparse_csc(inp, *args)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,pattern", _STRUCTURE_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__to_sparse_csc_sparse_structure(shape, dense_dim, pattern, dtype):
    inp = _pattern_input(dtype, shape, dense_dim, pattern)
    ref_inp = tu.to_reference(inp)
    args = _dense_dim_args(dense_dim)

    ref_out = torch.ops.aten._to_sparse_csc(ref_inp, *args)
    res_out = flag_gems._to_sparse_csc(inp, *args)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,dtype", _EDGE_CASES)
def test__to_sparse_csc_edge_shape(shape, dense_dim, dtype):
    inp = _dense_input(dtype, shape, ["-1", "1"], dense_dim)
    ref_inp = tu.to_reference(inp)
    args = _dense_dim_args(dense_dim)

    ref_out = torch.ops.aten._to_sparse_csc(ref_inp, *args)
    res_out = flag_gems._to_sparse_csc(inp, *args)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("variant,dtype", _VIEW_CASES)
def test__to_sparse_csc_view_input(variant, dtype):
    storage = tu.make_input(dtype, (17, 25), ["-1", "1"])
    # The view is rebuilt from an independent copy of the parent, so comparing
    # the whole parent below also catches a candidate that writes outside the
    # view; applying the same indexing to both keeps the real stride and storage
    # offset under test.
    ref_storage = tu.to_reference(storage)
    if variant == "slice":
        # [1::2, 2::3] of a 17x25 tensor keeps 8 rows and 8 columns, with stride
        # (50, 3) and storage offset 27.
        inp = storage[1::2, 2::3]
        ref_inp = ref_storage[1::2, 2::3]
    elif variant == "offset":
        inp = storage[3:, 2:]
        ref_inp = ref_storage[3:, 2:]
    else:
        inp = storage.t()
        ref_inp = ref_storage.t()

    ref_out = torch.ops.aten._to_sparse_csc(ref_inp)
    res_out = flag_gems._to_sparse_csc(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(storage, ref_storage)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__to_sparse_csc_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_sparse_csc(ref_inp)
    res_out = flag_gems._to_sparse_csc(inp)

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,dtype", _OUT_CASES)
def test__to_sparse_csc_out(shape, dense_dim, dtype):
    columns = _out_columns(_axes(shape, dense_dim)[2])
    inp = _out_input(dtype, shape, dense_dim, columns)
    ref_inp = tu.to_reference(inp)
    args = _dense_dim_args(dense_dim)

    native_buffer = _out_buffer(dtype, shape, dense_dim, columns, ref_inp.device)
    buffer = _out_buffer(dtype, shape, dense_dim, columns, inp.device)

    ref_out = torch.ops.aten._to_sparse_csc.out(ref_inp, *args, out=native_buffer)
    res_out = flag_gems._to_sparse_csc(inp, *args, out=buffer)

    assert res_out is buffer
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_csc
def test__to_sparse_csc_out_rejects_dense_buffer():
    inp = tu.make_input(torch.float32, (16, 16), ["-1", "1"])
    buffer = torch.zeros(16, 16, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_csc(inp, out=buffer)


@pytest.mark.to_sparse_csc
def test__to_sparse_csc_out_rejects_wrong_nnz():
    inp = _out_input(torch.float32, (32, 8), _OMITTED, _out_columns(8))
    # Same shape and layout, twice as many specified elements.
    buffer = _out_buffer(
        torch.float32, (32, 8), _OMITTED, list(range(8)), flag_gems.device
    )

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_csc(inp, out=buffer)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,exception", _NEGATIVE_CASES)
def test__to_sparse_csc_invalid_dense_dim(shape, dense_dim, exception):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    args = _dense_dim_args(dense_dim)

    with pytest.raises(exception):
        flag_gems._to_sparse_csc(inp, *args)


@pytest.mark.to_sparse_csc
@pytest.mark.parametrize("shape,dense_dim,dtype,pattern", _BACKWARD_CASES)
def test__to_sparse_csc_backward(shape, dense_dim, dtype, pattern):
    args = _dense_dim_args(dense_dim)
    base_inp = (
        _pattern_input(dtype, shape, dense_dim, pattern)
        if pattern is not None
        else _dense_input(dtype, shape, ["-1", "1"], dense_dim)
    )
    inp = base_inp.clone().requires_grad_(True)
    ref_inp = tu.to_reference(base_inp).requires_grad_(True)

    ref_out = torch.ops.aten._to_sparse_csc(ref_inp, *args)
    res_out = flag_gems._to_sparse_csc(inp, *args)
    tu.assert_result_equal(res_out, ref_out)

    # A nonuniform upstream gradient gives every specified element its own
    # coefficient instead of an all-ones sum, and the sparse patterns above
    # check that zeroed positions receive no gradient. Each side gets its own
    # upstream storage; the operation is a pure gather, so its gradient is
    # compared exactly.
    upstream = tu.make_input(dtype, tuple(ref_out.values().shape), ["-1", "1"])
    ref_grad = torch.autograd.grad(
        ref_out.values(), ref_inp, grad_outputs=upstream.to(ref_inp.device).clone()
    )[0]
    res_grad = torch.autograd.grad(
        res_out.values(), inp, grad_outputs=upstream.to(inp.device).clone()
    )[0]

    tu.assert_result_equal(res_grad, ref_grad)
    # The differentiable input must come back unchanged from the gradient
    # calls as well.
    tu.assert_result_equal(inp, ref_inp)
