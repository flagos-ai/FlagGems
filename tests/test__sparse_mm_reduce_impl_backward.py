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

"""Correctness for the native SparseCsrCPU reduction backward primitive.

Both implementations receive CPU CSR/dense operands and the workspace produced
by the matching native forward. The primitive has no out overload.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# (m, k, n, stored entries per row).
_SHAPES = [
    (1, 1, 1, 1),
    (2, 19, 7, 3),
    (4, 8, 4, 3),
    (256, 256, 256, 1),
    (320, 480, 15, 8),
    (1024, 1024, 1024, 4),
]
_QUICK_SHAPES = [(1, 1, 1, 1), (2, 19, 7, 3)]

_REDUCES = ["sum", "mean", "amax", "amin"]

# Probed on the target: only these four dtypes have SparseCsrCPU kernels. The
# six below build a valid CSR but raise `..._kernel not implemented for ...`,
# so they are pinned by the negative tests rather than dropped.
_SUPPORTED_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
_UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
]

_MASK_CASES = [(True, True), (True, False), (False, True), (False, False)]
_NEGATIVE_SHAPE = (4, 8, 4, 3)

_GRID_SHAPES = tu.selected_cases(_SHAPES, quick=_QUICK_SHAPES)
_GRID_RANGES = [tuple(value_range) for value_range in tu.selected_ranges()]
_GRID_REDUCES = _REDUCES
# Positive special values and backward are default-suite dimensions.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES), quick=[])
_BACKWARD_CASES = tu.selected_cases(
    [(reduce, dtype) for reduce in _REDUCES for dtype in _SUPPORTED_DTYPES],
    quick=[],
)


def _dense(shape, dtype, value_range):
    # tu.make_input allocates on flag_gems.device; the SparseCsrCPU kernels need
    # CPU operands, so the generated values are moved over unchanged.
    return tu.make_input(dtype, shape, value_range).cpu()


def _special_dense(shape, dtype, scenario):
    # Repeat the shared special-value payload so nan/inf/-inf/0 reach every
    # position without a second value conversion.
    payload = tu.make_special_input(dtype, scenario).cpu()
    count = 1
    for size in shape:
        count *= size
    return payload[torch.arange(count) % payload.numel()].reshape(shape)


def _csr(shape, values):
    m, k, _, nnz = shape
    cols = torch.arange(nnz) + torch.arange(m).unsqueeze(1) * 2
    if nnz:
        cols = cols % k
    cols = cols.sort(dim=1).values
    crow = torch.arange(m + 1, dtype=torch.int64) * nnz
    return torch.sparse_csr_tensor(
        crow, cols.reshape(-1), values.reshape(-1), size=(m, k)
    )


def _arg_out(self_csr, weight, reduce):
    # Forward emits absolute CSR storage indices for amax/amin. Requiring a
    # gradient requests the real workspace; sum/mean return an empty workspace.
    _, arg_out = torch.ops.aten._sparse_mm_reduce_impl(
        self_csr.detach().requires_grad_(True), weight, reduce
    )
    return arg_out


def _backward_inputs(shape, values, grad_out, weight, reduce):
    self_csr = _csr(shape, values)
    return self_csr, grad_out, weight, _arg_out(self_csr, weight, reduce)


def _make_inputs(shape, dtype, value_range, reduce):
    m, k, n, nnz = shape
    return _backward_inputs(
        shape,
        _dense((m, nnz), dtype, value_range),
        _dense((m, n), dtype, value_range),
        _dense((k, n), dtype, value_range),
        reduce,
    )


def _make_special_inputs(shape, dtype, scenario, reduce):
    m, k, n, nnz = shape
    return _backward_inputs(
        shape,
        _special_dense((m, nnz), dtype, scenario),
        _special_dense((m, n), dtype, scenario),
        _special_dense((k, n), dtype, scenario),
        reduce,
    )


def _assert_backward_result(res, ref, mask):
    """Compare every tuple component, including the sparse pattern of grad_self.

    The shared value assertions compare stored values, so the CSR structure and
    layout are checked here as well.
    """
    assert len(res) == len(ref) == len(mask)
    for res_part, ref_part, wanted in zip(res, ref, mask):
        assert (res_part is None) == (ref_part is None) == (not wanted)
        if res_part is None:
            continue
        if ref_part.layout == torch.sparse_csr:
            assert res_part.layout == torch.sparse_csr
            tu.assert_result_equal(res_part.crow_indices(), ref_part.crow_indices())
            tu.assert_result_equal(res_part.col_indices(), ref_part.col_indices())
            tu.assert_result_close(res_part.values(), ref_part.values())
        else:
            tu.assert_result_close(res_part, ref_part)


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("reduce", _GRID_REDUCES)
@pytest.mark.parametrize("value_range", _GRID_RANGES)
@pytest.mark.parametrize("shape", _GRID_SHAPES)
def test__sparse_mm_reduce_impl_backward(shape, value_range, reduce, dtype):
    mask = (True, True)
    self_csr, grad_out, weight, arg_out = _make_inputs(
        shape, dtype, value_range, reduce
    )

    ref_self_out, ref_weight_out = torch.ops.aten._sparse_mm_reduce_impl_backward(
        tu.to_reference(self_csr),
        tu.to_reference(grad_out),
        tu.to_reference(weight),
        reduce,
        tu.to_reference(arg_out),
        list(mask),
    )
    res_self_out, res_weight_out = flag_gems._sparse_mm_reduce_impl_backward(
        self_csr, grad_out, weight, reduce, arg_out, list(mask)
    )

    _assert_backward_result(
        (res_self_out, res_weight_out), (ref_self_out, ref_weight_out), mask
    )


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("mask", _MASK_CASES)
@pytest.mark.parametrize("reduce", _REDUCES)
def test__sparse_mm_reduce_impl_backward_output_mask(mask, dtype, reduce):
    self_csr, grad_out, weight, arg_out = _make_inputs(
        _NEGATIVE_SHAPE, dtype, ("-1", "1"), reduce
    )

    ref = torch.ops.aten._sparse_mm_reduce_impl_backward(
        tu.to_reference(self_csr),
        tu.to_reference(grad_out),
        tu.to_reference(weight),
        reduce,
        tu.to_reference(arg_out),
        list(mask),
    )
    res = flag_gems._sparse_mm_reduce_impl_backward(
        self_csr, grad_out, weight, reduce, arg_out, list(mask)
    )

    _assert_backward_result(res, ref, mask)


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("reduce", _REDUCES)
@pytest.mark.parametrize("case", _SPECIAL_CASES)
def test__sparse_mm_reduce_impl_backward_special_values(case, reduce):
    dtype, scenario = case
    shape = _NEGATIVE_SHAPE
    self_csr, grad_out, weight, arg_out = _make_special_inputs(
        shape, dtype, scenario, reduce
    )
    mask = (True, True)

    ref_self_out, ref_weight_out = torch.ops.aten._sparse_mm_reduce_impl_backward(
        tu.to_reference(self_csr),
        tu.to_reference(grad_out),
        tu.to_reference(weight),
        reduce,
        tu.to_reference(arg_out),
        list(mask),
    )
    res_self_out, res_weight_out = flag_gems._sparse_mm_reduce_impl_backward(
        self_csr, grad_out, weight, reduce, arg_out, list(mask)
    )

    _assert_backward_result(
        (res_self_out, res_weight_out), (ref_self_out, ref_weight_out), mask
    )


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("case", _BACKWARD_CASES)
def test__sparse_mm_reduce_impl_backward_autograd(case):
    reduce, dtype = case
    shape = _NEGATIVE_SHAPE
    self_csr, grad_out, weight, arg_out = _make_inputs(
        shape, dtype, ("-1", "1"), reduce
    )

    # Differentiate independent leaves through the native sparse matmul, whose
    # autograd formula is this operator, and compare its two gradients.
    self_leaf = self_csr.detach().clone().requires_grad_(True)
    weight_leaf = weight.detach().clone().requires_grad_(True)
    out = torch.sparse.mm(self_leaf, weight_leaf, reduce)
    grad_self, grad_weight = torch.autograd.grad(
        out, [self_leaf, weight_leaf], grad_out
    )

    res_self, res_weight = flag_gems._sparse_mm_reduce_impl_backward(
        self_leaf.detach(),
        grad_out,
        weight_leaf.detach(),
        reduce,
        arg_out,
        [True, True],
    )

    assert res_self is not None and res_weight is not None
    tu.assert_result_equal(res_self.crow_indices(), grad_self.crow_indices())
    tu.assert_result_equal(res_self.col_indices(), grad_self.col_indices())
    tu.assert_result_close(res_self.values(), grad_self.values())
    tu.assert_result_close(res_weight, grad_weight)


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__sparse_mm_reduce_impl_backward_unsupported_dtype(dtype):
    m, k, n, nnz = _NEGATIVE_SHAPE
    self_csr = _csr(_NEGATIVE_SHAPE, torch.ones(m, nnz).to(dtype))
    grad_out = torch.ones(m, n).to(dtype)
    weight = torch.ones(k, n).to(dtype)
    arg_out = torch.zeros(m, n, dtype=torch.int64)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_mm_reduce_impl_backward(
            self_csr, grad_out, weight, "sum", arg_out, [True, True]
        )


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("reduce", ["prod", "bogus"])
def test__sparse_mm_reduce_impl_backward_invalid_reduce(reduce):
    self_csr, grad_out, weight, arg_out = _make_inputs(
        _NEGATIVE_SHAPE, torch.float32, ("-1", "1"), "sum"
    )

    with pytest.raises(RuntimeError):
        flag_gems._sparse_mm_reduce_impl_backward(
            self_csr, grad_out, weight, reduce, arg_out, [True, True]
        )


@pytest.mark.sparse_mm_reduce_impl_backward
def test__sparse_mm_reduce_impl_backward_invalid_output_mask():
    self_csr, grad_out, weight, arg_out = _make_inputs(
        _NEGATIVE_SHAPE, torch.float32, ("-1", "1"), "sum"
    )

    # output_mask is a bool[2]; a one-element list cannot satisfy it.
    with pytest.raises(RuntimeError):
        flag_gems._sparse_mm_reduce_impl_backward(
            self_csr, grad_out, weight, "sum", arg_out, [True]
        )


@pytest.mark.sparse_mm_reduce_impl_backward
def test__sparse_mm_reduce_impl_backward_dense_self():
    m, k, n, _ = _NEGATIVE_SHAPE
    dense_self = _dense((m, k), torch.float32, ("-1", "1"))
    grad_out = _dense((m, n), torch.float32, ("-1", "1"))
    weight = _dense((k, n), torch.float32, ("-1", "1"))
    arg_out = torch.zeros(m, n, dtype=torch.int64)

    # self must be sparse-CSR: no strided kernel is registered for this operator.
    with pytest.raises(RuntimeError):
        flag_gems._sparse_mm_reduce_impl_backward(
            dense_self, grad_out, weight, "sum", arg_out, [True, True]
        )


@pytest.mark.sparse_mm_reduce_impl_backward
def test__sparse_mm_reduce_impl_backward_weight_shape_mismatch():
    shape = _NEGATIVE_SHAPE
    self_csr, grad_out, _, arg_out = _make_inputs(
        shape, torch.float32, ("-1", "1"), "sum"
    )
    bad_weight = _dense((shape[1] + 3, shape[2]), torch.float32, ("-1", "1"))

    with pytest.raises(RuntimeError):
        flag_gems._sparse_mm_reduce_impl_backward(
            self_csr, grad_out, bad_weight, "sum", arg_out, [True, True]
        )


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("reduce", _REDUCES)
@pytest.mark.parametrize(
    "shape", [(0, 4, 3, 0), (4, 0, 3, 0), (4, 3, 0, 1), (4, 3, 2, 0)]
)
def test__sparse_mm_reduce_impl_backward_empty(shape, reduce, dtype):
    self_csr, grad_out, weight, arg_out = _make_inputs(
        shape, dtype, ("-1", "1"), reduce
    )
    ref = torch.ops.aten._sparse_mm_reduce_impl_backward(
        tu.to_reference(self_csr),
        tu.to_reference(grad_out),
        tu.to_reference(weight),
        reduce,
        tu.to_reference(arg_out),
        [True, True],
    )
    res = flag_gems._sparse_mm_reduce_impl_backward(
        self_csr, grad_out, weight, reduce, arg_out, [True, True]
    )
    _assert_backward_result(res, ref, (True, True))


@pytest.mark.sparse_mm_reduce_impl_backward
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("reduce", _REDUCES)
@pytest.mark.parametrize("layout", ["transpose", "strided_offset"])
def test__sparse_mm_reduce_impl_backward_layout(layout, reduce, dtype):
    shape = _NEGATIVE_SHAPE
    self_csr, grad_out, weight, _ = _make_inputs(shape, dtype, ("-1", "1"), reduce)
    if layout == "transpose":
        grad_out = grad_out.t().contiguous().t()
        weight = weight.t().contiguous().t()
    else:
        grad_storage = torch.empty(shape[0], shape[2] * 2 + 1, dtype=dtype)
        weight_storage = torch.empty(shape[1], shape[2] * 2 + 1, dtype=dtype)
        grad_storage[:, 1::2].copy_(grad_out)
        weight_storage[:, 1::2].copy_(weight)
        grad_out, weight = grad_storage[:, 1::2], weight_storage[:, 1::2]
    arg_out = _arg_out(self_csr, weight, reduce)
    ref = torch.ops.aten._sparse_mm_reduce_impl_backward(
        tu.to_reference(self_csr),
        tu.to_reference(grad_out),
        tu.to_reference(weight),
        reduce,
        tu.to_reference(arg_out),
        [True, True],
    )
    res = flag_gems._sparse_mm_reduce_impl_backward(
        self_csr, grad_out, weight, reduce, arg_out, [True, True]
    )
    _assert_backward_result(res, ref, (True, True))
