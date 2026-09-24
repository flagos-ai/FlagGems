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

"""Correctness tests for ``aten::_sparse_mm``.

ATen exposes the default overload ``_sparse_mm(Tensor sparse, Tensor dense)``
plus a ``_sparse_mm.reduce(Tensor sparse, Tensor dense, str)`` variant; there is
no ``.out`` overload and no scalar operand, so every workload reaches the single
public candidate name ``flag_gems._sparse_mm``. The spec's rank ladder and five
value ranges are expressed as 2-D multiplications, and the accepted sparse
layouts, storage states, the native-valid dense left operand, dtypes and invalid
calls are covered explicitly. The ``.reduce`` variant has no kernel on this
backend and is covered as a vendor-scoped negative.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ``_sparse_mm`` starts with an underscore and ``pytest.mark`` refuses attribute
# access for such names, so register the marker on the MarkGenerator itself:
# ``@pytest.mark._sparse_mm`` and ``-m _sparse_mm`` then both work.
setattr(
    pytest.mark,
    "_sparse_mm",
    MarkDecorator(Mark("_sparse_mm", (), {}, _ispytest=True), _ispytest=True),
)

# Value range used by every non-grid workload (layouts, storage states, special
# values, backward, negatives).
_UNIT_RANGE = ["-1", "1"]

# 64-bit real and complex arithmetic does not exist on every backend; the shared
# helper exposes the static capability flag for it.
_WIDE_DTYPES = [torch.float64, torch.complex128] if utils.fp64_is_supported else []

# COO @ dense dispatches to addmm_sparse_cuda, which covers the
# float32/float64/complex64/complex128 family only.
COO_DENSE_DTYPES = [torch.float32, torch.complex64] + _WIDE_DTYPES

# CSR @ dense and the sparse @ sparse forms dispatch to the cuSPARSE paths, which
# additionally accept the reduced-precision types. A dense left operand dispatches
# to addmm_cuda and accepts the same set.
_REDUCED_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    _REDUCED_DTYPES.append(torch.bfloat16)
SPARSE_FORM_DTYPES = COO_DENSE_DTYPES + _REDUCED_DTYPES

# One stored value per four columns keeps the grid sparse without making the
# large spec shapes trivial.
_NNZ_PER_ROW_DIVISOR = 4

# Geometry below typical block sizes (both dimensions smaller than a block, inner
# not a multiple of it) plus a wide RHS with more columns than inner, in addition
# to the spec shapes.
_EXTRA_MM_SHAPES = [(7, 33, 5), (300, 7, 19)]


def _as_mm_shape(shape):
    """(rows, inner, cols) for one entry of ``tu.selected_shapes()``.

    ``_sparse_mm`` has a single 2-D form, so the spec's rank ladder is expressed
    as 2-D multiplications. A rank-0/1 entry of length ``n`` becomes the square
    ``n x n x n`` problem, which keeps the size of the ``(256,)`` level instead
    of collapsing every low-rank entry onto the trivial 1x1 problem. A plain 2-D
    entry keeps its row count and is squared on the matmul dimensions, and a
    higher rank folds its leading dimensions into the sparse row count.
    """
    if len(shape) < 2:
        size = shape[0] if shape else 1
        return (size, size, size)
    if len(shape) == 2:
        return (shape[0], shape[1], shape[1])
    rows = 1
    for dim in shape[:-2]:
        rows *= dim
    return (rows, shape[-2], shape[-1])


MM_SHAPES = list(
    dict.fromkeys(
        [_as_mm_shape(shape) for shape in tu.selected_shapes()]
        + tu.selected_cases(_EXTRA_MM_SHAPES, quick=[])
    )
)


def _default_nnz_per_row(inner):
    return max(1, inner // _NNZ_PER_ROW_DIVISOR)


def _unique_pattern(rows, inner, nnz_per_row):
    """Sorted, duplicate-free (row, column) indices for the first
    ``nnz_per_row`` columns of every row."""
    row_idx = torch.arange(rows, device=flag_gems.device).repeat_interleave(nnz_per_row)
    col_idx = torch.arange(nnz_per_row, device=flag_gems.device) % inner
    return torch.stack([row_idx, col_idx.repeat(rows)])


def _coo_from(indices, values, shape, is_coalesced=True, device=None):
    return torch.sparse_coo_tensor(
        indices,
        values,
        shape,
        device=flag_gems.device if device is None else device,
        is_coalesced=is_coalesced,
    )


def _empty_sparse(dtype, rows, inner):
    indices = torch.empty((2, 0), dtype=torch.int64, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    return _coo_from(indices, values, (rows, inner))


def _make_sparse_operand(dtype, rows, inner, value_range, *, nnz_per_row=None):
    """Coalesced COO operand of shape ``(rows, inner)`` with a sorted, unique
    support whose stored values follow ``value_range``. An operand with no
    storable entry (a zero extent, or no column to store into) is built empty
    instead of being dropped."""
    if nnz_per_row is None:
        nnz_per_row = _default_nnz_per_row(inner)
    nnz_per_row = min(nnz_per_row, inner)
    if nnz_per_row == 0 or rows == 0 or inner == 0:
        return _empty_sparse(dtype, rows, inner)
    indices = _unique_pattern(rows, inner, nnz_per_row)
    values = tu.make_input(dtype, (indices.shape[1],), value_range)
    return _coo_from(indices, values, (rows, inner))


@pytest.mark._sparse_mm
@pytest.mark.parametrize("dtype", COO_DENSE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", MM_SHAPES)
def test__sparse_mm(shape, value_range, dtype):
    rows, inner, cols = shape
    sparse = _make_sparse_operand(dtype, rows, inner, value_range)
    dense = tu.make_input(dtype, (inner, cols), value_range)

    ref_out = torch.ops.aten._sparse_mm(tu.to_reference(sparse), tu.to_reference(dense))
    res_out = flag_gems._sparse_mm(sparse, dense)

    # The shared comparison moves values to CPU before checking dtype and values,
    # so the candidate's output device is asserted here against its own input.
    assert res_out.device == sparse.device
    tu.assert_result_close(res_out, ref_out)


_FORMS = ("csr_dense", "coo_coo", "csr_csr")

# Reduced precision covers the cuSPARSE-backed forms, so the five-range sweep of
# those forms uses the float32/reduced-precision subset. ``(6, 5, 4)`` is small
# enough for all five ranges to stay in range for the reduced dtypes.
_FORM_RANGE_DTYPES = [torch.float32] + _REDUCED_DTYPES

# Two homogeneous coverage sets, both default-only: the wide mid-size geometries
# where the reduced dtypes are valid, and the dense-range sweep of the small
# geometry. Rows carry their own value range so both sets share one test body.
_FORM_ROWS = [
    (form, shape, dtype, _UNIT_RANGE)
    for form in _FORMS
    for shape in [(64, 48, 32), (33, 17, 9)]
    for dtype in SPARSE_FORM_DTYPES
] + [
    (form, (6, 5, 4), dtype, value_range)
    for form in _FORMS
    for dtype in _FORM_RANGE_DTYPES
    for value_range in tu.selected_ranges()
]
_FORM_CASES = tu.selected_cases(_FORM_ROWS, quick=[])


def _form_operands(form, shape, dtype, value_range):
    """Second operand of ``form``, rebuilt from the same COO pattern so both
    operands carry the same value range."""
    rows, inner, cols = shape
    coo = _make_sparse_operand(dtype, rows, inner, value_range)
    if form == "csr_dense":
        return coo.to_sparse_csr(), tu.make_input(dtype, (inner, cols), value_range)
    other = _make_sparse_operand(dtype, inner, cols, value_range)
    if form == "coo_coo":
        return coo, other
    return coo.to_sparse_csr(), other.to_sparse_csr()


@pytest.mark._sparse_mm
@pytest.mark.parametrize("form,shape,dtype,value_range", _FORM_CASES)
def test__sparse_mm_layout_form(form, shape, dtype, value_range):
    sparse, dense = _form_operands(form, shape, dtype, value_range)

    ref_out = torch.ops.aten._sparse_mm(tu.to_reference(sparse), tu.to_reference(dense))
    res_out = flag_gems._sparse_mm(sparse, dense)

    tu.assert_result_close(res_out, ref_out)


# A dense left operand is native-valid too: the operator falls back to addmm_cuda
# for it, which covers the float/complex family plus the reduced-precision types.
# Probed on this backend as valid for these geometries and dtypes, so it is a
# positive workload rather than only the mismatched negative below.
_DENSE_PAIR_SHAPES = [(5, 3, 4), (7, 33, 5)]
_DENSE_PAIR_CASES = tu.selected_cases(
    [(shape, dtype) for shape in _DENSE_PAIR_SHAPES for dtype in SPARSE_FORM_DTYPES],
    quick=[],
)


@pytest.mark._sparse_mm
@pytest.mark.parametrize("shape,dtype", _DENSE_PAIR_CASES)
def test__sparse_mm_dense_pair(shape, dtype):
    rows, inner, cols = shape
    lhs = tu.make_input(dtype, (rows, inner), _UNIT_RANGE)
    rhs = tu.make_input(dtype, (inner, cols), _UNIT_RANGE)

    ref_out = torch.ops.aten._sparse_mm(tu.to_reference(lhs), tu.to_reference(rhs))
    res_out = flag_gems._sparse_mm(lhs, rhs)

    tu.assert_result_close(res_out, ref_out)


# Sparse storage states a COO kernel has to normalise itself: unsorted and
# duplicate coordinates, explicit stored zeros, uneven row occupancy, stored
# values at a nonzero storage offset, and empty supports. The dense-layout states
# transpose, stride or offset the dense operand. The rows with an empty support
# keep the operand valid instead of being dropped, and duplicate coordinates are
# never marked as coalesced.
#
# Non-contiguous stored *values* are absent on purpose. Measured on this backend,
# the native spmm path reads the stored-value buffer as if it were contiguous: a
# repeatable stride-2 value view whose logical entries are [1, 3, 5, 7] returns
# the product of that backing buffer's first nnz slots rather than the product of
# the logical entries, while ``tu.to_reference`` compacts sparse value storage
# (stride (2,) -> (1,)). The reference side therefore cannot be produced through
# the shared helper's input contract for that layout without handing both sides
# the same tensor, so the workload stays archived as a pending helper-layout
# conversion gap in the task report instead of being claimed as validated. The
# shared helper is left untouched. Contiguous stored values at a nonzero storage
# offset (``values_offset``) are supported and covered.
_SPARSE_STATES = (
    "uncoalesced_duplicates",
    "unsorted_rows",
    "stored_zeros",
    "nonuniform_rows",
    "values_offset",
    "empty_values",
    "zero_rows",
    "zero_inner",
    "zero_cols",
    "coalesced",
    "single_entry",
    "full",
    "one_per_row",
    "wide_rhs",
    "tall_lhs",
)
_DENSE_LAYOUT_STATES = ("dense_col_major", "dense_row_strided", "dense_offset")

_STATE_GEOMETRY = {
    "uncoalesced_duplicates": (6, 5, 4),
    "unsorted_rows": (4, 5, 3),
    "stored_zeros": (4, 5, 3),
    "nonuniform_rows": (4, 5, 3),
    "values_offset": (4, 5, 3),
    "empty_values": (4, 5, 3),
    "zero_rows": (0, 5, 3),
    "zero_inner": (4, 0, 3),
    "zero_cols": (4, 5, 0),
    "coalesced": (4, 5, 3),
    "single_entry": (4, 5, 3),
    "full": (4, 5, 3),
    "one_per_row": (4, 5, 3),
    "wide_rhs": (4, 3, 11),
    "tall_lhs": (13, 3, 5),
    "dense_col_major": (6, 4, 3),
    "dense_row_strided": (6, 4, 3),
    "dense_offset": (6, 4, 3),
}

_STATE_CASES = tu.selected_cases(
    [
        (state, dtype)
        for state in _SPARSE_STATES + _DENSE_LAYOUT_STATES
        for dtype in COO_DENSE_DTYPES
    ],
    quick=[],
)


def _state_operands(state, dtype):
    rows, inner, cols = _STATE_GEOMETRY[state]

    if state == "dense_col_major":
        dense = tu.make_input(dtype, (cols, inner), _UNIT_RANGE).t()
    elif state == "dense_row_strided":
        dense = tu.make_input(dtype, (inner, 2 * cols), _UNIT_RANGE)[:, :cols]
    elif state == "dense_offset":
        dense = tu.make_input(dtype, (inner + 2, cols), _UNIT_RANGE)[1:-1]
    else:
        dense = tu.make_input(dtype, (inner, cols), _UNIT_RANGE)

    if state in ("empty_values", "zero_rows", "zero_inner"):
        return _empty_sparse(dtype, rows, inner), dense

    if state == "uncoalesced_duplicates":
        # Row 3 stores column 1 twice with different values and the rows are not
        # sorted, so the implicit coalesce has to sum the duplicates.
        indices = torch.tensor(
            [[0, 3, 3, 2, 0], [4, 1, 1, 0, 2]], device=flag_gems.device
        )
        values = tu.make_input(dtype, (5,), _UNIT_RANGE)
        return _coo_from(indices, values, (rows, inner), is_coalesced=False), dense

    if state == "unsorted_rows":
        indices = torch.tensor([[3, 0, 2, 1], [4, 1, 0, 2]], device=flag_gems.device)
        values = tu.make_input(dtype, (4,), _UNIT_RANGE)
        return _coo_from(indices, values, (rows, inner), is_coalesced=False), dense

    if state == "nonuniform_rows":
        # Row 0 is dense, row 1 is empty and the last row stores a single entry.
        indices = torch.tensor(
            [[0, 0, 0, 0, 3], [0, 1, 2, 4, 4]], device=flag_gems.device
        )
        values = tu.make_input(dtype, (5,), _UNIT_RANGE)
        return _coo_from(indices, values, (rows, inner)), dense

    if state == "single_entry":
        indices = torch.tensor([[0], [0]], device=flag_gems.device)
        return (
            _coo_from(indices, tu.make_input(dtype, (1,), _UNIT_RANGE), (rows, inner)),
            dense,
        )

    per_row = {
        "one_per_row": 1,
        "full": inner,
        "wide_rhs": 3,
        "tall_lhs": 2,
    }.get(state, _default_nnz_per_row(inner))
    indices = _unique_pattern(rows, inner, min(per_row, inner))
    nnz = indices.shape[1]
    if state == "stored_zeros":
        values = torch.zeros(nnz, dtype=dtype, device=flag_gems.device)
    elif state == "values_offset":
        # Stored values are a contiguous slice at a nonzero storage offset.
        values = tu.make_input(dtype, (nnz + 2,), _UNIT_RANGE)[1:-1]
    else:
        values = tu.make_input(dtype, (nnz,), _UNIT_RANGE)
    return _coo_from(indices, values, (rows, inner)), dense


@pytest.mark._sparse_mm
@pytest.mark.parametrize("state,dtype", _STATE_CASES)
def test__sparse_mm_sparse_state(state, dtype):
    sparse, dense = _state_operands(state, dtype)

    ref_out = torch.ops.aten._sparse_mm(tu.to_reference(sparse), tu.to_reference(dense))
    res_out = flag_gems._sparse_mm(sparse, dense)

    tu.assert_result_close(res_out, ref_out)


# nan/inf coverage for each supported floating dtype. ``tu.make_special_input``
# returns the same payload (nan, inf, -inf, 0.0, -0.0) for every scenario name, so
# the three scenarios below are not three distinct value classes; the scenarios
# are kept because the payload contract is the shared helper's, and each placement
# moves that payload to a different operand.
_SPECIAL_DTYPES = [torch.float32, torch.complex64] + _WIDE_DTYPES
_SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario, placement)
        for placement in ("sparse", "dense", "both")
        for dtype in _SPECIAL_DTYPES
        for scenario in ("nan", "inf", "mixed")
    ],
    quick=[],
)


def _special_operands(dtype, scenario, placement):
    """Diagonal sparse x dense pair whose special values sit in one or both
    operands, so nan/inf propagation is covered for each side separately."""
    payload = tu.make_special_input(dtype, scenario)
    size = payload.numel()
    normal = tu.make_input(dtype, (size,), _UNIT_RANGE)
    stored = payload if placement in ("sparse", "both") else normal
    row = payload if placement in ("dense", "both") else normal
    diag = torch.arange(size, device=flag_gems.device)
    sparse = _coo_from(torch.stack([diag, diag]), stored, (size, size))
    # Row i of the product is stored[i] * row, which spreads the special
    # classifications over the whole output for either placement.
    return sparse, row.repeat(size).reshape(size, size)


@pytest.mark._sparse_mm
@pytest.mark.parametrize("dtype,scenario,placement", _SPECIAL_CASES)
def test__sparse_mm_special_values(dtype, scenario, placement):
    sparse, dense = _special_operands(dtype, scenario, placement)

    ref_out = torch.ops.aten._sparse_mm(tu.to_reference(sparse), tu.to_reference(dense))
    res_out = flag_gems._sparse_mm(sparse, dense)

    tu.assert_result_close(res_out, ref_out)


# Native autograd for this operator exists for the COO x dense form and the four
# float/complex dtypes. The candidate upstream gradient is built on the candidate
# device and converted with ``tu.to_reference`` for the reference side, and the
# reference operands live on the reference device so the native backward runs
# where the comparison values are.
_BACKWARD_CASES = tu.selected_cases(
    [
        (shape, dtype, upstream)
        for shape in [(4, 5, 3), (33, 17, 9)]
        for dtype in COO_DENSE_DTYPES
        for upstream in ("constant", "varying")
    ],
    quick=[],
)


@pytest.mark._sparse_mm
@pytest.mark.parametrize("shape,dtype,upstream", _BACKWARD_CASES)
def test__sparse_mm_backward(shape, dtype, upstream):
    rows, inner, cols = shape
    indices = _unique_pattern(rows, inner, _default_nnz_per_row(inner))
    values = tu.make_input(dtype, (indices.shape[1],), _UNIT_RANGE).requires_grad_(True)
    dense = tu.make_input(dtype, (inner, cols), _UNIT_RANGE).requires_grad_(True)
    sparse = _coo_from(indices, values, (rows, inner))

    ref_values = tu.to_reference(values.detach()).requires_grad_(True)
    ref_dense = tu.to_reference(dense.detach()).requires_grad_(True)
    ref_sparse = _coo_from(
        indices.to(ref_values.device),
        ref_values,
        (rows, inner),
        device=ref_values.device,
    )

    out_shape = (rows, cols)
    if upstream == "constant":
        grad_upstream = torch.ones(out_shape, dtype=dtype, device=flag_gems.device)
    else:
        grad_upstream = tu.make_input(dtype, out_shape, _UNIT_RANGE)
    ref_upstream = tu.to_reference(grad_upstream)

    ref_out = torch.ops.aten._sparse_mm(ref_sparse, ref_dense)
    res_out = flag_gems._sparse_mm(sparse, dense)
    tu.assert_result_close(res_out, ref_out)

    ref_values_grad, ref_dense_grad = torch.autograd.grad(
        ref_out, (ref_values, ref_dense), grad_outputs=ref_upstream
    )
    values_grad, dense_grad = torch.autograd.grad(
        res_out, (values, dense), grad_outputs=grad_upstream
    )
    tu.assert_result_close(values_grad, ref_values_grad)
    tu.assert_result_close(dense_grad, ref_dense_grad)


# Invalid calls. The expected exception type is the one measured for each form:
# an operand with fewer than two dimensions fails inside the operator's view
# logic with IndexError, while a rank-3 operand and mismatched matrix dimensions
# reach the addmm argument checks and raise RuntimeError.
_INVALID_OPERAND_CASES = [
    ("dense_1d", IndexError),
    ("dense_0d", IndexError),
    ("sparse_1d", IndexError),
    ("sparse_0d", IndexError),
    ("sparse_3d", RuntimeError),
    ("dense_3d", RuntimeError),
    ("inner_mismatch", RuntimeError),
    # A shape-compatible dense-first call ((5, 3) @ (3, 4)) succeeds natively and
    # is covered positively above, so this negative uses two (5, 3) operands:
    # their inner dimensions 3 and 5 disagree and addmm reports that the shapes
    # cannot be multiplied.
    ("dense_shape_mismatch", (RuntimeError, ValueError)),
    ("non_tensor_dense", (RuntimeError, TypeError)),
    ("dtype_mismatch", (RuntimeError, TypeError)),
]


def _invalid_operands(kind):
    if kind == "sparse_1d":
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0, 2, 4]], device=flag_gems.device),
            torch.ones(3, device=flag_gems.device),
            (5,),
            device=flag_gems.device,
        )
        return sparse, torch.ones(5, 3, device=flag_gems.device)
    if kind == "sparse_0d":
        sparse = torch.sparse_coo_tensor(
            torch.zeros((0, 0), dtype=torch.int64, device=flag_gems.device),
            torch.zeros(0, device=flag_gems.device),
            (),
            device=flag_gems.device,
        )
        return sparse, torch.ones((), device=flag_gems.device)
    if kind == "sparse_3d":
        # One stored entry in a 3-D COO tensor: the index tensor needs one row
        # per sparse dimension.
        indices = torch.tensor(
            [[0], [0], [0]], dtype=torch.int64, device=flag_gems.device
        )
        sparse = torch.sparse_coo_tensor(
            indices,
            torch.ones(1, device=flag_gems.device),
            (1, 2, 4),
            device=flag_gems.device,
            is_coalesced=True,
        )
        return sparse, torch.ones(4, 5, device=flag_gems.device)

    sparse = _make_sparse_operand(torch.float32, 2, 4, _UNIT_RANGE)
    if kind == "dense_1d":
        return sparse, torch.ones(5, device=flag_gems.device)
    if kind == "dense_0d":
        return sparse, torch.ones((), device=flag_gems.device)
    if kind == "dense_3d":
        return sparse, torch.ones(1, 4, 5, device=flag_gems.device)
    if kind == "inner_mismatch":
        return sparse, torch.ones(7, 5, device=flag_gems.device)
    if kind == "dense_shape_mismatch":
        lhs = torch.ones(5, 3, dtype=torch.float32, device=flag_gems.device)
        return lhs, lhs.clone()
    if kind == "non_tensor_dense":
        return sparse, [[1.0, 2.0, 3.0, 4.0]]
    if kind == "dtype_mismatch":
        return sparse, tu.make_input(torch.float16, (4, 5), _UNIT_RANGE)
    raise AssertionError(f"unhandled invalid operand: {kind}")


@pytest.mark._sparse_mm
@pytest.mark.parametrize("kind,expected", _INVALID_OPERAND_CASES)
def test__sparse_mm_invalid_operand(kind, expected):
    sparse, dense = _invalid_operands(kind)

    with pytest.raises(expected):
        flag_gems._sparse_mm(sparse, dense)


# The native CUDA sparse kernel rejects these dtypes for COO @ dense with
# "addmm_sparse_cuda" not implemented for '<type>'. That rejection is a property
# of the measured backend rather than of the operator, so the negative workload
# is scoped to the measured vendor and is not asserted anywhere else. Reduced
# precision is valid for the CSR, dense-left and sparse @ sparse forms and is
# covered positively above. The dtype gates are the shared static capability
# flags.
_UNSUPPORTED_COO_DTYPES = []
if flag_gems.runtime.device.vendor_name == "nvidia":
    _UNSUPPORTED_COO_DTYPES = [torch.float16, torch.int8, torch.uint8, torch.int32]
    if utils.bf16_is_supported:
        _UNSUPPORTED_COO_DTYPES.append(torch.bfloat16)
    if utils.int64_is_supported:
        _UNSUPPORTED_COO_DTYPES.append(torch.int64)
    _UNSUPPORTED_COO_DTYPES.append(torch.bool)
    if utils.fp8_is_supported:
        _UNSUPPORTED_COO_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]


@pytest.mark._sparse_mm
@pytest.mark.parametrize("dtype", _UNSUPPORTED_COO_DTYPES)
def test__sparse_mm_unsupported_dtype(dtype):
    sparse = _make_sparse_operand(dtype, 4, 4, _UNIT_RANGE)
    dense = tu.make_input(dtype, (4, 4), _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_mm(sparse, dense)


# ``_sparse_mm.reduce`` needs aten::_sparse_mm_reduce_impl, which has no
# SparseCUDA kernel on the measured vendor: every schema-valid reduce string
# raises NotImplementedError there. Scoped to that measured vendor for the same
# reason as the dtype list above, and only the schema's real semantic values are
# used.
_REDUCE_CASES = (
    ["sum", "mean", "amax", "amin"]
    if flag_gems.runtime.device.vendor_name == "nvidia"
    else []
)


@pytest.mark._sparse_mm
@pytest.mark.parametrize("reduce", _REDUCE_CASES)
def test__sparse_mm_reduce_unsupported(reduce):
    sparse = _make_sparse_operand(torch.float32, 4, 5, _UNIT_RANGE)
    dense = tu.make_input(torch.float32, (5, 3), _UNIT_RANGE)

    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_mm(sparse, dense, reduce)
