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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import test_utils as tu

# ``_sparse_csr_sum`` starts with an underscore, so ``pytest.mark`` cannot be
# extended by attribute access; register the marker on the MarkGenerator.
setattr(
    pytest.mark,
    "_sparse_csr_sum",
    MarkDecorator(Mark("_sparse_csr_sum", (), {}, _ispytest=True), _ispytest=True),
)

pytestmark = pytest.mark._sparse_csr_sum

_DEVICE_FLAGS = flag_gems.runtime.device

# aten::_sparse_csr_sum reduces the axes named by ``dim`` on a 2-D sparse CSR
# operand: one axis keeps the other one (dim=0 gives (1, cols), dim=1 gives
# (rows, 1)), while the two-axis and empty-axis forms reduce the whole operand.
# There is no broadcast dimension (the operator takes a single operand) and no
# backward dimension: the native operator registers no autograd formula, so
# autograd.grad over its output fails with "derivative for aten::_sparse_csr_sum
# is not implemented".
_SUM_SHAPES = tu.selected_cases(
    [(1024, 1024), (20, 320), (320, 20), (16, 128), (1, 1), (0, 8), (8, 0)],
    quick=[(2, 19)],
)

_SUM_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.float16,
    torch.float32,
    torch.complex64,
]
# int64/bfloat16/float64 follow the static device capability flags.
if _DEVICE_FLAGS.support_int64:
    _SUM_DTYPES.append(torch.int64)
if _DEVICE_FLAGS.support_bf16:
    _SUM_DTYPES.append(torch.bfloat16)
if _DEVICE_FLAGS.support_fp64:
    _SUM_DTYPES.append(torch.float64)

_FLOAT_DTYPES = [dtype for dtype in _SUM_DTYPES if dtype.is_floating_point]

# bool and the FP8 formats have no kernel for this operator on the NVIDIA CUDA
# backend (native: "_sparse_csr_sum_cuda" not implemented for 'Bool',
# 'Float8_e4m3fn' and 'Float8_e5m2'). The FP8 entries are additionally gated on
# the static FP8 capability flag, because those dtypes are only constructible
# where the backend provides them; the list needs no runtime probe and no
# authored skip. Both conditions nest under the vendor check, so a backend that
# does not report the NVIDIA vendor collects an empty list instead of constants
# that were never probed there; nothing is probed at import time.
_REJECTED_INPUT_DTYPES = []
if _DEVICE_FLAGS.vendor_name == "nvidia":
    _REJECTED_INPUT_DTYPES.append(torch.bool)
    if _DEVICE_FLAGS.support_fp8:
        _REJECTED_INPUT_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]

_KEEPDIM = True

_DIMS = tu.selected_cases([[0], [1]], quick=[[0]])

# Start value of a preallocated ``out`` buffer; representable in every buffer
# dtype. Correctness comes from comparing the rewritten result with the
# reference, not from observing this value.
_SENTINEL = 7.0


def _csr_from_values(values):
    """One stored element per row, so every payload entry survives.

    ``values`` already comes from the shared generator on ``flag_gems.device``,
    so the operand is built on that device without a further transfer.
    """
    n = values.numel()
    crow = torch.arange(n + 1, device=values.device)
    col = torch.zeros(n, dtype=torch.int64, device=values.device)
    return torch.sparse_csr_tensor(crow, col, values, size=(n, 1), device=values.device)


def _small_csr(dtype, size=(2, 3)):
    """Fixed-index operand used by the invalid-input cases."""
    crow = torch.tensor([0, 2, 3], device=flag_gems.device)
    col = torch.tensor([0, 2, 2], device=flag_gems.device)
    values = torch.tensor([1, 0, 1], dtype=dtype, device=flag_gems.device)
    return torch.sparse_csr_tensor(
        crow, col, values, size=size, device=flag_gems.device
    )


def _structural_csr(dtype, size, index_dtype, crow, col, values):
    """Valid CSR operand from explicit indices, stored values and index dtype.

    Every caller passes a ``crow`` that is monotone, ``len(crow) == size[0] + 1``
    and ends at ``len(values)``, with ``col`` sorted and duplicate-free inside
    each row and below ``size[1]``.
    """
    device = flag_gems.device
    return torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=index_dtype, device=device),
        torch.tensor(col, dtype=index_dtype, device=device),
        torch.tensor(values, dtype=dtype, device=device),
        size=size,
        device=device,
    )


def _out_buffer(ref_out, device, index_dtype, shift=0):
    """Independent, *valid* CSR ``out`` buffer for the result shape of ``ref_out``.

    The kernel keeps the buffer object and rewrites its support, but every
    fixture must already be a valid CSR structure of the result's shape, stored
    count and dtype: ``crow`` is monotone and ends at ``nnz``, ``col`` is sorted
    and duplicate-free inside each row and in bounds. ``shift`` rotates a valid
    support of the single-row (``dim=[0]``) result shape -- ``nnz <= columns``
    there -- so the preloaded support genuinely differs from the result's own;
    a row-per-entry layout keeps the multi-row (``dim=[1]``) result valid too.
    """
    shape = tuple(ref_out.shape)
    nnz = ref_out.values().numel()
    if shape[0] == 1:
        crow = torch.tensor([0, nnz], dtype=index_dtype, device=device)
        col = torch.arange(nnz, dtype=index_dtype, device=device)
        if shift:
            col = torch.sort((col + shift) % shape[-1]).values
    else:
        crow = torch.clamp(torch.arange(shape[0] + 1, device=device), max=nnz).to(
            index_dtype
        )
        col = torch.zeros(nnz, dtype=index_dtype, device=device)
    values = torch.full((nnz,), _SENTINEL, dtype=ref_out.dtype, device=device)
    return torch.sparse_csr_tensor(crow, col, values, size=shape, device=device)


def _csr_with_stored_entry(dtype, shape):
    """CSR operand with at least one stored element.

    The multi-axis dim forms below must not depend on a dense draw happening to
    contain nonzero data -- over ``[-1, 1]`` a (1, 1) draw can be exactly zero,
    and a zero-stored operand reaches the unresolved native gap documented
    above the structural cases. Pinning one in-range element keeps those forms
    on a well-defined operand; it adds no zero-stored-element coverage of its
    own.
    """
    dense = tu.make_input(dtype, shape, ["-1", "1"])
    dense[0, 0] = 1
    return dense.to_sparse_csr()


def _assert_index_dtype(result, index_dtype):
    """The candidate keeps the operand's index dtype.

    Stored and index values are compared by the shared assertion, which also
    compares index values exactly.
    """
    assert result.crow_indices().dtype == index_dtype
    assert result.col_indices().dtype == index_dtype


@pytest.mark.parametrize("shape", _SUM_SHAPES)
@pytest.mark.parametrize("dtype", _SUM_DTYPES)
@pytest.mark.parametrize("dim", _DIMS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__sparse_csr_sum(shape, dtype, dim, value_range):
    inp = tu.make_input(dtype, shape, value_range).to_sparse_csr()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_inp, dim, _KEEPDIM)
    res_out = flag_gems._sparse_csr_sum(inp, dim, _KEEPDIM)

    tu.assert_result_close(res_out, ref_out)


# Every native-supported way of spelling the reduced axes: both orders of the
# two-element list, negative axes, and the empty list.
_DIM_FORMS = [[0, 1], [1, 0], [-1], [-2], [-1, -2], [-2, -1], []]
_DIM_FORM_DTYPES = [torch.int32, torch.float16, torch.float32]
_DIM_FORM_SHAPES = tu.selected_cases(
    [(1024, 1024), (20, 320), (320, 20), (16, 128), (1, 1)], quick=[]
)


@pytest.mark.parametrize("shape", _DIM_FORM_SHAPES)
@pytest.mark.parametrize("dtype", _DIM_FORM_DTYPES)
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=str)
def test__sparse_csr_sum_dim_forms(shape, dtype, dim):
    inp = _csr_with_stored_entry(dtype, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_inp, dim, _KEEPDIM)
    res_out = flag_gems._sparse_csr_sum(inp, dim, _KEEPDIM)

    tu.assert_result_close(res_out, ref_out)


# Operands a dense draw cannot express: explicitly stored zeros, empty rows,
# a column that holds no stored entry, non-default index dtypes, and an operand
# with no stored element at all. Every row below is a valid CSR fixture
# (monotone crow ending at len(values), sorted in-bounds columns).
#
# Unresolved native gap, documented instead of worked around: a full reduction
# (dim=[] or dim=[0, 1]) over a zero-stored-element operand returns a
# structurally invalid CSR on this build. On shape (2, 3) with crow [0, 0, 0]
# and no stored element, dim=[] and dim=[0, 1] give shape (1, 1) with crow
# [0, 0] and no col_indices, but a 0-dim (scalar) values tensor; re-constructing
# that output raises "values must have dimensionality > sum of batch and block
# dimensionalities". This is a deterministic output-contract gap rather than
# nondeterminism: float32 and float64 produced identical output in 8/8 repeats
# per dtype and dim. Single-axis reductions over the same operand return a valid
# empty CSR in every repeat and stay covered here, by the "zero-stored" row
# below and by the (0, 8) / (8, 0) shapes of the main grid. No case
# reconstructs, densifies or pads the malformed native output, and no
# zero-stored full reduction is claimed.
_STRUCTURAL_CASES = [
    (
        "stored-zeros",
        torch.float32,
        (2, 3),
        torch.int64,
        [0, 2, 3],
        [0, 2, 2],
        [0.0, -0.0, 5.0],
        [1],
    ),
    (
        "empty-row",
        torch.float32,
        (3, 4),
        torch.int64,
        [0, 0, 2, 2],
        [1, 3],
        [1.5, -2.5],
        [0],
    ),
    (
        "empty-column",
        torch.float32,
        (2, 4),
        torch.int64,
        [0, 1, 2],
        [3, 3],
        [4.0, 2.0],
        [1],
    ),
    (
        "int32-indices",
        torch.float32,
        (2, 3),
        torch.int32,
        [0, 2, 3],
        [0, 2, 2],
        [3.0, 5.0, 7.0],
        [0],
    ),
    (
        "int64-indices",
        torch.float16,
        (2, 3),
        torch.int64,
        [0, 1, 2],
        [1, 0],
        [-1.0, 2.0],
        [1],
    ),
    ("zero-stored", torch.float32, (2, 4), torch.int64, [0, 0, 0], [], [], [0]),
]
_STRUCTURE = tu.selected_cases(_STRUCTURAL_CASES, quick=[])
_STRUCTURE_IDS = [row[0] for row in _STRUCTURE]
_STRUCTURE_ARGS = "case,dtype,size,index_dtype,crow,col,values,dim"


@pytest.mark.parametrize(_STRUCTURE_ARGS, _STRUCTURE, ids=_STRUCTURE_IDS)
def test__sparse_csr_sum_structure(
    case, dtype, size, index_dtype, crow, col, values, dim
):
    inp = _structural_csr(dtype, size, index_dtype, crow, col, values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_inp, dim, _KEEPDIM)
    res_out = flag_gems._sparse_csr_sum(inp, dim, _KEEPDIM)

    tu.assert_result_close(res_out, ref_out)
    _assert_index_dtype(res_out, index_dtype)


@pytest.mark.parametrize(_STRUCTURE_ARGS, _STRUCTURE, ids=_STRUCTURE_IDS)
def test__sparse_csr_sum_structure_out(
    case, dtype, size, index_dtype, crow, col, values, dim
):
    inp = _structural_csr(dtype, size, index_dtype, crow, col, values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_inp, dim, _KEEPDIM)
    out = _out_buffer(ref_out, inp.device, inp.crow_indices().dtype)
    ref_buffer = _out_buffer(ref_out, ref_inp.device, ref_inp.crow_indices().dtype)

    torch.ops.aten._sparse_csr_sum.dim_dtype_out(ref_inp, dim, _KEEPDIM, out=ref_buffer)
    res_out = flag_gems._sparse_csr_sum(inp, dim, _KEEPDIM, out=out)

    assert res_out is out
    tu.assert_result_close(res_out, ref_buffer)
    _assert_index_dtype(res_out, index_dtype)


# The out buffer must match the result's stored count, dtype, shape and be a
# valid CSR structure itself, but not its index values: the kernel keeps the
# buffer object and rewrites crow/col with the result's own support. Every case
# reduces along dim=[0], whose result keeps a single row, so a rotated support
# of that row is valid for the buffer while genuinely differing from the
# result's columns (each case's preloaded support differs from the result's).
_REBUILD_CASES = [
    (
        "int64-preload",
        (4, 6),
        torch.int64,
        [0, 1, 1, 2, 2],
        [1, 4],
        [2.5, -1.5],
        [0],
        2,
    ),
    (
        "int32-index-preload",
        (3, 5),
        torch.int32,
        [0, 2, 3, 3],
        [0, 4, 2],
        [1.0, 2.0, 3.0],
        [0],
        1,
    ),
    (
        "int64-wider-columns",
        (2, 7),
        torch.int64,
        [0, 2, 3],
        [0, 6, 3],
        [1.0, 2.0, 3.0],
        [0],
        3,
    ),
]
_REBUILD = tu.selected_cases(_REBUILD_CASES, quick=[])
_REBUILD_IDS = [row[0] for row in _REBUILD]
_REBUILD_ARGS = "case,size,index_dtype,crow,col,values,dim,shift"


@pytest.mark.parametrize(_REBUILD_ARGS, _REBUILD, ids=_REBUILD_IDS)
def test__sparse_csr_sum_out_rebuild_support(
    case, size, index_dtype, crow, col, values, dim, shift
):
    inp = _structural_csr(torch.float32, size, index_dtype, crow, col, values)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_inp, dim, _KEEPDIM)
    out = _out_buffer(ref_out, inp.device, index_dtype, shift)
    ref_buffer = _out_buffer(ref_out, ref_inp.device, index_dtype, shift)

    torch.ops.aten._sparse_csr_sum.dim_dtype_out(ref_inp, dim, _KEEPDIM, out=ref_buffer)
    res_out = flag_gems._sparse_csr_sum(inp, dim, _KEEPDIM, out=out)

    assert res_out is out
    tu.assert_result_close(res_out, ref_buffer)
    _assert_index_dtype(res_out, index_dtype)


_OUT_SHAPES = tu.selected_cases(_SUM_SHAPES, quick=[])


@pytest.mark.parametrize("shape", _OUT_SHAPES)
@pytest.mark.parametrize("dtype", _SUM_DTYPES)
@pytest.mark.parametrize("dim", _DIMS)
def test__sparse_csr_sum_out(shape, dtype, dim):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).to_sparse_csr()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_inp, dim, _KEEPDIM)
    # The buffer dtype follows the result, not the operand: integer operands
    # promote their accumulation (an int8 operand yields a long result).
    out = _out_buffer(ref_out, inp.device, inp.crow_indices().dtype)
    ref_buffer = _out_buffer(ref_out, ref_inp.device, ref_inp.crow_indices().dtype)

    torch.ops.aten._sparse_csr_sum.dim_dtype_out(ref_inp, dim, _KEEPDIM, out=ref_buffer)
    res_out = flag_gems._sparse_csr_sum(inp, dim, _KEEPDIM, out=out)

    assert res_out is out
    tu.assert_result_close(res_out, ref_buffer)


_OVERRIDE_DTYPES = [torch.float32]
if _DEVICE_FLAGS.support_int64:
    _OVERRIDE_DTYPES.append(torch.int64)
if _DEVICE_FLAGS.support_fp64:
    _OVERRIDE_DTYPES.append(torch.float64)

_OVERRIDE_SHAPES = tu.selected_cases(_SUM_SHAPES, quick=[])


@pytest.mark.parametrize("dtype", _OVERRIDE_DTYPES)
@pytest.mark.parametrize("shape", _OVERRIDE_SHAPES)
def test__sparse_csr_sum_dtype_override(dtype, shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"]).to_sparse_csr()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(
        ref_inp, [1], _KEEPDIM, dtype=dtype
    )
    res_out = flag_gems._sparse_csr_sum(inp, [1], _KEEPDIM, dtype=dtype)

    tu.assert_result_close(res_out, ref_out)


# tu.special_value_cases enumerates real floating dtypes only, but complex64 is
# a supported operand dtype. tu.make_special_input builds a float32 payload and
# casts it with .to(dtype), so the complex64 rows carry exactly zero imaginary
# parts and the reduction preserves the real NaN/Inf pattern.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_FLOAT_DTYPES)
    + [(torch.complex64, scenario) for scenario in ("nan", "inf", "mixed")],
    quick=[],
)


@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES, ids=str)
def test__sparse_csr_sum_special_values(dtype, scenario):
    inp = _csr_from_values(tu.make_special_input(dtype, scenario))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_inp, [1], _KEEPDIM)
    res_out = flag_gems._sparse_csr_sum(inp, [1], _KEEPDIM)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("dtype", _REJECTED_INPUT_DTYPES)
def test__sparse_csr_sum_rejects_unsupported_dtype(dtype):
    inp = _small_csr(dtype)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_csr_sum(inp, [1], _KEEPDIM)


@pytest.mark.parametrize(
    "dim",
    [[2], [-3], [0, 0], ["0"], [0.0]],
    ids=["above-rank", "below-rank", "duplicate", "str-element", "float-element"],
)
def test__sparse_csr_sum_rejects_invalid_dim(dim):
    inp = _small_csr(torch.float32)
    # An out-of-range axis is rejected by IndexError before the operator body
    # runs, so the expected set spans the index, runtime and type errors.
    with pytest.raises((RuntimeError, TypeError, IndexError)):
        flag_gems._sparse_csr_sum(inp, dim, _KEEPDIM)


def test__sparse_csr_sum_rejects_keepdim_false():
    # keepdim defaults to False and the native operator rejects it, so the
    # schema default itself is the invalid case here.
    inp = _small_csr(torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_csr_sum(inp, [1])


@pytest.mark.parametrize("kind", ["dense-2d", "coo", "dense-1d"])
def test__sparse_csr_sum_rejects_non_csr_input(kind):
    if kind == "dense-2d":
        inp = torch.ones(4, 4, device=flag_gems.device)
    elif kind == "coo":
        inp = torch.ones(4, 4, device=flag_gems.device).to_sparse()
    else:
        inp = torch.ones(4, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csr_sum(inp, [1], _KEEPDIM)
