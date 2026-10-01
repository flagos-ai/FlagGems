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

# aten::_sparse_mm_reduce_impl has exactly one registered kernel, SparseCsrCPU
# (no SparseCsrCUDA in this build) and no .out overload, so `reduce` is the only
# parameter and the default call form is the only callable overload. self is a
# non-hybrid 2-D sparse-CSR tensor, other a 2-D strided dense tensor with the
# dtype of self.values() and other.size(0) == self.size(1); the reference and the
# injected candidate receive the same CPU operands unchanged.
_CPU = torch.device("cpu")
_FINITE_RANGE = ["-1", "1"]
_REDUCES = ["sum", "mean", "amax", "amin"]

# The CPU kernel accepts these value dtypes and rejects every other one with
# RuntimeError: "spmm_reduce_kernel" not implemented for "<dtype>". bfloat16 and
# float64 are additionally gated by the shared static device flags only because
# tu.make_input allocates on flag_gems.device before the operand is moved to CPU;
# those flags are not a kernel restriction.
_DTYPES = [torch.float32, torch.float16]
if flag_gems.runtime.device.support_bf16:
    _DTYPES.append(torch.bfloat16)
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)

# Value dtypes the kernel itself rejects, probed with real CPU operands. The list
# is independent of _DTYPES above so these rows stay negative at both levels.
_UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

# self is (M, K) and other is (K, N), so the spec shape levels map to (M, K, N)
# triples of the same scale; the trailing triples are this signature's zero-extent
# boundaries, which the structure grid below re-exercises at both levels.
_MNK_SHAPES = [
    (1, 1, 1),
    (256, 256, 256),
    (1024, 1024, 1024),
    (20, 320, 15),
    (16, 128, 64),
    (16, 7, 57),
    (64, 1, 32),
    (0, 4, 3),
    (4, 0, 3),
    (4, 3, 0),
]
_MNK_ROWS = tu.selected_cases(
    _MNK_SHAPES + [(2, 19, 7)],
    quick=[(2, 19, 7), (1, 1, 1), (0, 4, 3), (4, 0, 3), (4, 3, 0)],
)

# reduce selects the kernel that runs, so all four modes are cases at both levels.
_REDUCE_MNK = [(6, 8, 5), (20, 320, 15), (256, 256, 256)]
_REDUCE_ROWS = tu.selected_cases(
    [(mnk, reduce) for mnk in _REDUCE_MNK for reduce in _REDUCES],
    quick=[((2, 19, 7), reduce) for reduce in _REDUCES],
)

# Storage patterns and operand states are cheap, so quick keeps every reduce.
_STRUCTURE_SHAPES = [
    (6, 8, 5, "sparse"),
    (6, 8, 5, "empty_rows"),
    (6, 8, 5, "zero_nnz"),
    (6, 8, 5, "fully_stored"),
    (6, 8, 5, "single_nnz"),
    (6, 8, 5, "strided_other"),
    (6, 8, 5, "int32_index"),
    (0, 4, 3, "sparse"),
    (4, 0, 3, "sparse"),
    (4, 3, 0, "sparse"),
]
_STRUCTURE_ROWS = tu.selected_cases(
    [
        (m, k, n, label, reduce)
        for (m, k, n, label) in _STRUCTURE_SHAPES
        for reduce in _REDUCES
    ],
    quick=[
        (m, k, n, label, reduce)
        for (m, k, n, label) in _STRUCTURE_SHAPES
        for reduce in _REDUCES
    ],
)

# Independent differentiable leaves over two storage patterns, one of them with
# rows that store nothing; the extra backward shapes are default-only.
_BACKWARD_ROWS = tu.selected_cases(
    [
        (reduce, mnk, label, dtype)
        for reduce in _REDUCES
        for (mnk, label) in (((20, 320, 15), "sparse"), ((16, 128, 64), "empty_rows"))
        for dtype in _DTYPES
    ],
    quick=[],
)

# arg_out is filled only in grad mode for amax/amin, and its dtype follows the CSR
# index dtype, so both index types and both differentiable operands are covered.
# Both index widths and differentiable operands are cheap enough for quick.
_GRAD_ARG_ROWS = tu.selected_cases(
    [
        (reduce, index_dtype, grad_operand)
        for reduce in _REDUCES
        for index_dtype in (torch.int64, torch.int32)
        for grad_operand in ("self", "other")
    ],
    quick=[
        (reduce, index_dtype, grad_operand)
        for reduce in _REDUCES
        for index_dtype in (torch.int64, torch.int32)
        for grad_operand in ("self", "other")
    ],
)

# nan / inf / mixed payloads in the stored values and in the dense operand, for
# every supported dtype; positive special values are default-only.
_SPECIAL_ROWS = tu.selected_cases(
    [
        (dtype, scenario, placement, reduce)
        for (dtype, scenario) in tu.special_value_cases(_DTYPES)
        for placement in ("values", "other")
        for reduce in _REDUCES
    ],
    quick=[],
)

_SPECIAL_MNK = (5, 4, 5)
_NEGATIVE_LABELS = [
    "prod",
    "unknown-reduce",
    "empty-reduce",
    "upper-case-reduce",
    "dense-self",
    "coo-self",
    "hybrid-self",
    "other-3d",
    "other-dtype-mismatch",
    "other-row-mismatch",
]


def _csr_index(rows, m, index_dtype=torch.int64):
    """(crow_indices, col_indices) for explicit per-row column lists."""
    columns = [torch.tensor(row, dtype=torch.int64) for row in rows]
    col = torch.cat(columns) if columns else torch.empty(0, dtype=torch.int64)
    counts = torch.tensor([c.numel() for c in columns], dtype=torch.int64)
    crow = torch.zeros(m + 1, dtype=torch.int64)
    crow[1:] = counts.cumsum(0)
    return crow.to(index_dtype), col.to(index_dtype)


def _row_pattern(m, k, label):
    """Per-row stored columns for the structural rows and the value grid."""
    if label == "empty_rows":
        return [list(range(k)) if i % 2 else [] for i in range(m)]
    if label == "zero_nnz":
        return [[] for _ in range(m)]
    if label == "fully_stored":
        return [list(range(k)) for _ in range(m)]
    if label == "single_nnz":
        return [[i % k] if k else [] for i in range(m)]
    # default: every third column, with a per-row offset so the rows differ
    step = max(1, min(3, k))
    return [list(range(i % step, k, step)) for i in range(m)]


def _csr(dtype, rows, shape, value_range=_FINITE_RANGE, index_dtype=torch.int64):
    row_count, col_count = shape
    crow, col = _csr_index(rows, row_count, index_dtype)
    values = tu.make_input(dtype, (col.numel(),), value_range).to(_CPU)
    return torch.sparse_csr_tensor(crow, col, values, size=(row_count, col_count))


def _dense(dtype, shape, value_range=_FINITE_RANGE):
    return tu.make_input(dtype, shape, value_range).to(_CPU)


def _grid_operands(mnk, dtype, value_range):
    m, k, n = mnk
    self_t = _csr(dtype, _row_pattern(m, k, "sparse"), (m, k), value_range)
    return self_t, _dense(dtype, (k, n), value_range)


def _structure_operands(m, k, n, label):
    """self/other pair for one structural label; ``int32_index`` only changes the
    CSR index dtype and ``strided_other`` only the dense strides."""
    if label == "strided_other":
        # The kernel checks extent and dtype only, and this build accepts a
        # non-contiguous dense operand, so a candidate must not demand contiguity
        # here: the values match the compact (K, N) tensor.
        other = _dense(torch.float32, (k, n + 3))[:, :n]
    else:
        other = _dense(torch.float32, (k, n))
    index_dtype = torch.int32 if label == "int32_index" else torch.int64
    pattern = "sparse" if label == "int32_index" else label
    self_t = _csr(
        torch.float32,
        _row_pattern(m, k, pattern),
        (m, k),
        index_dtype=index_dtype,
    )
    return self_t, other


def _special_operands(dtype, scenario, placement):
    """Operands where every output entry reduces exactly one payload term, so the
    NaN / Inf classification cannot depend on the accumulation order."""
    m, k, n = _SPECIAL_MNK
    payload = tu.make_special_input(dtype, scenario).to(_CPU)
    if placement == "values":
        crow, col = _csr_index([[i % k] for i in range(m)], m)
        values = torch.zeros(col.numel(), dtype=dtype)
        values[: payload.numel()] = payload
        self_t = torch.sparse_csr_tensor(crow, col, values, size=(m, k))
        return self_t, _dense(dtype, (k, n))
    # Payload in the first dense row, one stored column per row on the CSR side.
    other = _dense(dtype, (k, n))
    other[0, : payload.numel()] = payload
    return _csr(dtype, [[0] for _ in range(m)], (m, k)), other


def _grad_operands(dtype, index_dtype, grad_operand):
    """Operands with exactly one differentiable leaf, the one named by
    ``grad_operand``; the other operand is a constant."""
    m, k, n = 5, 6, 4
    crow, col = _csr_index(_row_pattern(m, k, "empty_rows"), m, index_dtype)
    values = _dense(dtype, (col.numel(),))
    other = _dense(dtype, (k, n))
    if grad_operand == "self":
        values.requires_grad_(True)
    else:
        other.requires_grad_(True)
    return torch.sparse_csr_tensor(crow, col, values, size=(m, k)), other


def _invalid_operands(label):
    """A native-invalid call (self, other, reduce) for one negative row."""
    m, k, n = 4, 5, 3
    self_t = _csr(torch.float32, _row_pattern(m, k, "sparse"), (m, k))
    other = _dense(torch.float32, (k, n))
    if label == "prod":
        # Valid operands, but this reduce type is not enabled by the operator.
        return self_t, other, "prod"
    if label == "unknown-reduce":
        return self_t, other, "bogus"
    if label == "empty-reduce":
        return self_t, other, ""
    if label == "upper-case-reduce":
        return self_t, other, "SUM"
    if label == "dense-self":
        return _dense(torch.float32, (m, k)), other, "sum"
    if label == "coo-self":
        return self_t.to_sparse_coo(), other, "sum"
    if label == "hybrid-self":
        # dense_dim() > 0 is rejected by the native kernel.
        hybrid = torch.sparse_csr_tensor(
            self_t.crow_indices(),
            self_t.col_indices(),
            _dense(torch.float32, (self_t._nnz(), 2)),
            size=(m, k),
        )
        return hybrid, other, "sum"
    if label == "other-3d":
        return self_t, _dense(torch.float32, (k, n, 2)), "sum"
    if label == "other-dtype-mismatch":
        return self_t, _dense(torch.float64, (k, n)), "sum"
    # other.size(0) must equal self.size(1)
    return self_t, _dense(torch.float32, (k + 2, n)), "sum"


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("mnk", _MNK_ROWS)
def test__sparse_mm_reduce_impl_value_grid(mnk, value_range, dtype):
    """Value ranges x shape descriptors x supported dtypes for the sum mode; the
    other three reduce modes have their own case list."""
    self_t, other = _grid_operands(mnk, dtype, value_range)

    ref_out, ref_arg = torch.ops.aten._sparse_mm_reduce_impl(
        tu.to_reference(self_t), tu.to_reference(other), "sum"
    )
    res_out, res_arg = flag_gems._sparse_mm_reduce_impl(self_t, other, "sum")

    tu.assert_result_close(res_out, ref_out)
    # Outside grad mode arg_out is an empty tensor in the crow_indices dtype.
    tu.assert_result_equal(res_arg, ref_arg)


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("mnk,reduce", _REDUCE_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__sparse_mm_reduce_impl_reduce_modes(mnk, reduce, dtype):
    m, k, n = mnk
    self_t = _csr(dtype, _row_pattern(m, k, "sparse"), (m, k))
    other = _dense(dtype, (k, n))

    ref_out, _ = torch.ops.aten._sparse_mm_reduce_impl(
        tu.to_reference(self_t), tu.to_reference(other), reduce
    )
    res_out, _ = flag_gems._sparse_mm_reduce_impl(self_t, other, reduce)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("m,k,n,label,reduce", _STRUCTURE_ROWS)
def test__sparse_mm_reduce_impl_structure(m, k, n, label, reduce):
    self_t, other = _structure_operands(m, k, n, label)

    ref_out, ref_arg = torch.ops.aten._sparse_mm_reduce_impl(
        tu.to_reference(self_t), tu.to_reference(other), reduce
    )
    res_out, res_arg = flag_gems._sparse_mm_reduce_impl(self_t, other, reduce)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(res_arg, ref_arg)
    if label == "zero_nnz":
        # No stored value: every output entry is zero for every reduce mode.
        assert not bool(res_out.any())


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("reduce", _REDUCES)
def test__sparse_mm_reduce_impl_arg_out_inference(reduce):
    """Outside grad mode the second result is empty, in the crow_indices dtype."""
    m, k, n = 5, 6, 4
    self_t = _csr(torch.float32, _row_pattern(m, k, "empty_rows"), (m, k))
    other = _dense(torch.float32, (k, n))

    ref_out, ref_arg = torch.ops.aten._sparse_mm_reduce_impl(
        tu.to_reference(self_t), tu.to_reference(other), reduce
    )
    res_out, res_arg = flag_gems._sparse_mm_reduce_impl(self_t, other, reduce)

    tu.assert_result_equal(res_arg, ref_arg)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("reduce,index_dtype,grad_operand", _GRAD_ARG_ROWS)
def test__sparse_mm_reduce_impl_arg_out_in_grad_mode(reduce, index_dtype, grad_operand):
    """amax/amin fill arg_out with the winning row index per output entry when a
    differentiable operand is present; sum/mean keep it empty."""
    self_t, other = _grad_operands(torch.float32, index_dtype, grad_operand)

    ref_out, ref_arg = torch.ops.aten._sparse_mm_reduce_impl(
        tu.to_reference(self_t), tu.to_reference(other), reduce
    )
    res_out, res_arg = flag_gems._sparse_mm_reduce_impl(self_t, other, reduce)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(res_arg, ref_arg)


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("reduce,mnk,label,dtype", _BACKWARD_ROWS)
def test__sparse_mm_reduce_impl_backward(reduce, mnk, label, dtype):
    m, k, n = mnk
    crow, col = _csr_index(_row_pattern(m, k, label), m)
    values = _dense(dtype, (col.numel(),)).requires_grad_(True)
    other = _dense(dtype, (k, n)).requires_grad_(True)
    self_t = torch.sparse_csr_tensor(crow, col, values, size=(m, k))

    # Independent leaves for the reference graph: same values, no shared storage.
    ref_values = values.detach().clone().requires_grad_(True)
    ref_other = other.detach().clone().requires_grad_(True)
    ref_self = torch.sparse_csr_tensor(crow, col, ref_values, size=(m, k))

    res_out, _ = flag_gems._sparse_mm_reduce_impl(self_t, other, reduce)
    ref_out, _ = torch.ops.aten._sparse_mm_reduce_impl(ref_self, ref_other, reduce)
    tu.assert_result_close(res_out, ref_out)

    upstream = _dense(dtype, (m, n))
    res_dvalues, res_dother = torch.autograd.grad(
        res_out, [values, other], grad_outputs=upstream
    )
    ref_dvalues, ref_dother = torch.autograd.grad(
        ref_out, [ref_values, ref_other], grad_outputs=tu.to_reference(upstream)
    )

    tu.assert_result_close(res_dvalues, ref_dvalues)
    tu.assert_result_close(res_dother, ref_dother)


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("dtype,scenario,placement,reduce", _SPECIAL_ROWS)
def test__sparse_mm_reduce_impl_special_values(dtype, scenario, placement, reduce):
    self_t, other = _special_operands(dtype, scenario, placement)

    ref_out, ref_arg = torch.ops.aten._sparse_mm_reduce_impl(
        tu.to_reference(self_t), tu.to_reference(other), reduce
    )
    res_out, res_arg = flag_gems._sparse_mm_reduce_impl(self_t, other, reduce)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(res_arg, ref_arg)


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__sparse_mm_reduce_impl_rejects_unsupported_dtype(dtype):
    """int8 / uint8 / int32 / int64 / bool / float8 have no spmm_reduce_kernel."""
    m, k, n = 4, 5, 3
    self_t = _csr(dtype, _row_pattern(m, k, "sparse"), (m, k))
    other = _dense(dtype, (k, n))

    with pytest.raises(RuntimeError):
        flag_gems._sparse_mm_reduce_impl(self_t, other, "sum")


@pytest.mark.sparse_mm_reduce_impl
@pytest.mark.parametrize("label", _NEGATIVE_LABELS)
def test__sparse_mm_reduce_impl_rejects_invalid_arguments(label):
    """Native-invalid operand forms and reduce strings, kept in every mode."""
    self_t, other, reduce = _invalid_operands(label)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_mm_reduce_impl(self_t, other, reduce)
