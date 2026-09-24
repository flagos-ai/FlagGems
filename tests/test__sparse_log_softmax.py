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

"""Correctness tests for aten::_sparse_log_softmax on sparse COO input.

The native kernel normalizes the stored values: for ``dim >= sparse_dim`` each
value row is normalized over its dense block, for ``dim < sparse_dim`` the
entries sharing the other sparse coordinates are pooled.  Both paths are
covered, and the coordinate fixtures place several entries on the pooled axis so
the sparse path sees real normalization groups instead of singletons.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# pytest blocks attribute access for underscore-prefixed marker names.
setattr(
    pytest.mark,
    "_sparse_log_softmax",
    MarkDecorator(Mark("_sparse_log_softmax", (), {}, _ispytest=True), _ispytest=True),
)

# Static construction capability, taken from the runtime detector's existing
# flags: a dtype whose flag is False is never used to build a fixture, so no
# workload claims to test a failure of an input this backend cannot even
# construct.  Dtypes without a flag (fp16, int8, uint8, int32, bool) are
# baseline and always kept.
_DTYPE_CONSTRUCTIBLE = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}


def _constructible_dtype(dtype):
    return _DTYPE_CONSTRUCTIBLE.get(dtype, True)


SUPPORTED_DTYPES = [torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)

# Only float32/float64 have a kernel for the .default form
# (AT_DISPATCH_FLOATING_TYPES).  The rejection wording of the negative cases
# below was measured on the nvidia backend, so those expectations are collected
# only for that measured vendor; construction, rank and dim checks stay vendor
# independent.
_MEASURED_VENDOR = "nvidia"
_VENDOR_MEASURED = (
    getattr(flag_gems.runtime.device, "vendor_name", "") == _MEASURED_VENDOR
)

# Each default-form rejection measured on the measured vendor: RuntimeError
# "log_softmax" not implemented for 'Bool', 'Char', 'Byte', 'Float8_e4m3fn',
# 'Float8_e5m2', 'BFloat16', 'Half', 'Int', 'Long'.
_UNSUPPORTED_DTYPE_CANDIDATES = [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]
UNSUPPORTED_DTYPES = [
    dtype for dtype in _UNSUPPORTED_DTYPE_CANDIDATES if _constructible_dtype(dtype)
]

_NNZ_CAP = 64
_QUICK_SHAPE = (2, 19, 7)
_PARAM_SHAPE = (16, 128, 64, 60)
_PARAM_SPARSE_DIM = 3
_PARAM_DIM = 0

# (sparse_dim, dim) pairs per spec shape; each shape contributes a pooled axis
# (dim < sparse_dim) and a dense block axis (dim >= sparse_dim).  The scalar
# shape () has no valid dim at all, so it is covered by
# test__sparse_log_softmax_rejects_scalar_input rather than a positive row.
ROW_SPECS = {
    (): [],
    (1,): [(1, 0)],
    (256,): [(1, 0), (0, 0)],
    (1024, 1024): [(1, 0), (1, 1), (2, 0), (0, 1)],
    (20, 320, 15): [(1, 2), (2, 0), (2, 2), (3, 1), (0, 0)],
    (16, 128, 64, 60): [(2, 1), (2, 3), (3, 2), (4, 3)],
    (16, 7, 57, 32, 29): [(3, 0), (3, 4), (4, 0), (4, 4)],
    (2, 19, 7): [(2, 0), (2, 1), (2, 2)],
}


def _shape_rows(shapes):
    return [
        (shape, sparse_dim, dim)
        for shape in shapes
        for sparse_dim, dim in ROW_SPECS[shape]
    ]


# Full spec-shape grid plus the extra sparse/dense axes of the quick smoke
# shape.  Quick mode keeps a single representative positive row per supported
# dtype.
POSITIVE_ROWS = tu.selected_cases(
    _shape_rows(tu.REQUIRED_SHAPES) + _shape_rows([_QUICK_SHAPE]),
    quick=[(_QUICK_SHAPE, 2, 0)],
)

# Native-valid dtype= conversions through the .int overload, measured on
# _PARAM_SHAPE with sparse_dim 3 at dim 0.  float16 -> float32 is the one
# measured rejection ('log_softmax: with half to float conversion is not
# supported on cuda:0'), while float16 -> float64 is supported and kept; every
# other required input dtype converts to both float32 and float64.  Both the
# input and the output dtype are gated on construction capability, and this
# additional positive family is default-only so quick runs no large conversion
# grid.
_CONVERSION_INPUTS = [
    (torch.int8, (torch.float32, torch.float64)),
    (torch.uint8, (torch.float32, torch.float64)),
    (torch.int32, (torch.float32, torch.float64)),
    (torch.int64, (torch.float32, torch.float64)),
    (torch.bfloat16, (torch.float32, torch.float64)),
    (torch.float16, (torch.float64,)),
    (torch.float8_e4m3fn, (torch.float32, torch.float64)),
    (torch.float8_e5m2, (torch.float32, torch.float64)),
    (torch.float32, (torch.float32, torch.float64)),
    (torch.float64, (torch.float32, torch.float64)),
]

CONVERSION_ROWS = tu.selected_cases(
    [
        (in_dtype, out_dtype)
        for in_dtype, out_dtypes in _CONVERSION_INPUTS
        if _constructible_dtype(in_dtype)
        for out_dtype in out_dtypes
        if _constructible_dtype(out_dtype)
    ],
    quick=[],
)

# dtype= omitted entirely, which checks the candidate's own schema default.
DEFAULT_DTYPE_DTYPES = tu.selected_cases(SUPPORTED_DTYPES, quick=[])

# Special-value workloads for the conversion form, restricted to the floating
# inputs whose conversion the native operator supports.  Scenario sets come from
# the shared generator, which already limits float8_e4m3fn to nan only; the
# input and output dtypes are gated on construction capability as well.
_SPECIAL_CONVERSION_INPUTS = [
    (torch.float16, (torch.float64,)),
    (torch.bfloat16, (torch.float32, torch.float64)),
    (torch.float32, (torch.float32, torch.float64)),
    (torch.float8_e4m3fn, (torch.float32, torch.float64)),
    (torch.float8_e5m2, (torch.float32, torch.float64)),
    (torch.float64, (torch.float32, torch.float64)),
]


def _scenarios_for(dtype):
    return [
        scenario
        for candidate, scenario in tu.special_value_cases([dtype])
        if candidate == dtype
    ]


SPECIAL_CONVERSION_CASES = tu.selected_cases(
    [
        (in_dtype, out_dtype, scenario)
        for in_dtype, out_dtypes in _SPECIAL_CONVERSION_INPUTS
        if _constructible_dtype(in_dtype)
        for out_dtype in out_dtypes
        if _constructible_dtype(out_dtype)
        for scenario in _scenarios_for(in_dtype)
    ],
    quick=[],
)

SPECIAL_ROWS = tu.selected_cases([((256,), 1, 0), ((20, 320, 15), 2, 2)], quick=[])
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[])
BACKWARD_ROWS = tu.selected_cases(
    [((256,), 1, 0), ((20, 320, 15), 2, 2), ((16, 128, 64, 60), 3, 1)], quick=[]
)
EMPTY_ROWS = tu.selected_cases(
    [
        ((256,), 1, 0),
        ((1024, 1024), 1, 1),
        ((1024, 1024), 2, 0),
        ((16, 128, 64, 60), 3, 1),
    ],
    quick=[],
)
UNCOALESCED_ROWS = tu.selected_cases(
    [((2, 19, 7), 2, 2), ((20, 320, 15), 1, 2), ((16, 128, 64, 60), 3, 0)], quick=[]
)
OUT_ROWS = tu.selected_cases(
    [((256,), 1, 0), ((20, 320, 15), 2, 2), ((1024, 1024), 1, 1)], quick=[]
)

# Measured rejection for every floating input dtype on the measured vendor:
# 'log_softmax: with half to float conversion is not supported on cuda:0'.
# Fixtures of a dtype this backend cannot construct are never collected.
HALF_TO_FLOAT_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if _constructible_dtype(dtype)
]

# The fixture here is float32, so no dtype-construction gate applies to the
# parametrized ScalarType values themselves.
INVALID_DTYPE_PARAMS = [torch.int64, torch.bfloat16, torch.float16]


def _rank_fixture(rank):
    """Metadata-equivalent shape containing ``rank`` sparse dimensions."""
    return (2,) * rank, max(1, rank - 1)


# Out-of-range dims on small metadata-equivalent fixtures, so the rank boundary
# cases survive quick mode instead of depending on selected_shapes.  Native
# rejects both ends with IndexError (measured for ranks 1..5).
INVALID_DIM_ROWS = [
    (shape, sparse_dim, dim)
    for rank in range(1, 6)
    for shape, sparse_dim in [_rank_fixture(rank)]
    for dim in (rank, -rank - 1)
]

# A rank-0 sparse COO tensor (sparse_dim 0) is constructible, but every dim is
# out of range: dim 0 and dim -1 both raise for nnz 0 and nnz 1 (measured).
SCALAR_NEG_ROWS = [(nnz, dim) for nnz in (0, 1) for dim in (0, -1)]


def _sparse_extent(shape, sparse_dim):
    """Number of distinct coordinates the sparse dimensions can address."""
    extent = 1
    for size in shape[:sparse_dim]:
        extent *= int(size)
    return extent


def _nnz(shape, sparse_dim):
    """Stored entries, capped by the sparse extent so no coordinate repeats."""
    if 0 in shape:
        return 0
    if sparse_dim == 0:
        return 1  # hybrid tensor: a single dense value block
    return max(1, min(_sparse_extent(shape, sparse_dim), _NNZ_CAP))


def _reduced_axis(shape, sparse_dim, dim):
    """Sparse axis the kernel pools over for ``dim`` (last one if dim is dense)."""
    axis = dim + len(shape) if dim < 0 else dim
    return axis if 0 <= axis < sparse_dim else sparse_dim - 1


def _strides(sizes):
    strides, run = [], 1
    for size in reversed(sizes):
        strides.append(run)
        run *= size
    return strides[::-1]


def _coo_indices(shape, sparse_dim, nnz, group_axis=None):
    """Unique sorted COO coordinates of shape (sparse_dim, nnz) on the device.

    Coordinates are built in O(nnz) with ``torch.arange`` on the configured
    device: entries are emitted in groups along ``group_axis`` so the pooled axis
    always holds ties to normalize, then the row-major rank is sorted, which is
    exactly the lexicographic order coalescing requires.
    """
    if nnz == 0:
        return torch.zeros((sparse_dim, 0), dtype=torch.long, device=flag_gems.device)
    if sparse_dim == 0:
        return torch.zeros((0, nnz), dtype=torch.long, device=flag_gems.device)
    sizes = [int(size) for size in shape[:sparse_dim]]
    axis = sparse_dim - 1 if group_axis is None else group_axis
    others = [index for index in range(sparse_dim) if index != axis]
    other_extent = 1
    for index in others:
        other_extent *= sizes[index]
    group = min(sizes[axis], max(2, -(-nnz // other_extent)))
    order = torch.arange(nnz, device=flag_gems.device, dtype=torch.long)
    base = order // group
    coords = [None] * sparse_dim
    coords[axis] = order % group
    for index in reversed(others):
        coords[index] = base % sizes[index]
        base = base // sizes[index]
    strides = _strides(sizes)
    rank = torch.zeros(nnz, device=flag_gems.device, dtype=torch.long)
    for index in range(sparse_dim):
        rank = rank + coords[index] * strides[index]
    rank = rank.sort().values
    return torch.stack(
        [(rank // strides[index]) % sizes[index] for index in range(sparse_dim)]
    ).contiguous()


def _value_shape(shape, sparse_dim, nnz):
    return (nnz,) + tuple(shape[sparse_dim:])


def _make_coo(indices, values, shape, dtype, is_coalesced=True):
    return torch.sparse_coo_tensor(
        indices,
        values,
        shape,
        device=flag_gems.device,
        dtype=dtype,
        is_coalesced=is_coalesced,
        check_invariants=True,
    )


def _sparse_input(
    shape,
    sparse_dim,
    dtype,
    value_range=("-1", "1"),
    nnz=None,
    dim=0,
    requires_grad=False,
):
    """Valid COO fixture built on the configured device.

    ``tu.make_input`` already returns a tensor on ``flag_gems.device``, so no
    host roundtrip is involved.
    """
    stored = _nnz(shape, sparse_dim) if nnz is None else nnz
    indices = _coo_indices(
        shape, sparse_dim, stored, _reduced_axis(shape, sparse_dim, dim)
    )
    values = tu.make_input(dtype, _value_shape(shape, sparse_dim, stored), value_range)
    inp = _make_coo(indices, values, shape, dtype)
    if requires_grad:
        inp.requires_grad_(True)
    return inp


def _tile_payload(payload, dtype, shape, sparse_dim, stored):
    """Repeat the shared payload to fill exactly the stored value block."""
    tail = 1
    for size in shape[sparse_dim:]:
        tail *= int(size)
    needed = stored * tail
    block = payload.repeat(-(-needed // payload.numel()))[:needed]
    return block.to(dtype).reshape(_value_shape(shape, sparse_dim, stored))


def _sparse_special_input(shape, sparse_dim, dtype, scenario, nnz=None, dim=0):
    """nan/inf fixture whose payload is tiled over the stored value block."""
    stored = _nnz(shape, sparse_dim) if nnz is None else nnz
    indices = _coo_indices(
        shape, sparse_dim, stored, _reduced_axis(shape, sparse_dim, dim)
    )
    payload = tu.make_special_input(dtype, scenario)
    values = _tile_payload(payload, dtype, shape, sparse_dim, stored)
    return _make_coo(indices, values, shape, dtype)


def _empty_coo_like(inp):
    """Empty COO buffer in inp's shape/device/dtype for the .out form."""
    indices = torch.zeros((inp.sparse_dim(), 0), dtype=torch.long, device=inp.device)
    values = torch.zeros(
        _value_shape(inp.shape, inp.sparse_dim(), 0), dtype=inp.dtype, device=inp.device
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        inp.shape,
        device=inp.device,
        dtype=inp.dtype,
        is_coalesced=True,
    )


def _scalar_coo(nnz, dtype):
    """Rank-0 sparse COO tensor: sparse_dim 0 with ``nnz`` stored values."""
    indices = torch.zeros((0, nnz), dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(dtype, (nnz,), ("-1", "1"))
    return torch.sparse_coo_tensor(
        indices,
        values,
        (),
        device=flag_gems.device,
        dtype=dtype,
        is_coalesced=True,
        check_invariants=True,
    )


def _sparse_upstream(out):
    """Nonconstant sparse grad_output on out's own coordinates.

    Constant upstream values cannot reveal a permuted or collapsed gradient, so
    each stored entry gets its own value.
    """
    values = torch.arange(
        1, out.values().numel() + 1, dtype=out.dtype, device=out.device
    )
    values = values.reshape(out.values().shape)
    return torch.sparse_coo_tensor(
        out._indices(),
        values,
        out.shape,
        device=out.device,
        dtype=out.dtype,
        is_coalesced=True,
    )


def _uncoalesced_input(shape, sparse_dim, dtype, dim):
    """Explicitly uncoalesced fixture: stored coordinates repeat."""
    unique = _coo_indices(
        shape,
        sparse_dim,
        _nnz(shape, sparse_dim),
        _reduced_axis(shape, sparse_dim, dim),
    )
    duplicates = unique[:, : min(3, unique.shape[1])]
    indices = torch.cat([unique, duplicates], dim=1)
    values = tu.make_input(
        dtype, _value_shape(shape, sparse_dim, indices.shape[1]), ("-1", "1")
    )
    return _make_coo(indices, values, shape, dtype, is_coalesced=False)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("shape,sparse_dim,dim", POSITIVE_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_log_softmax(shape, sparse_dim, dim, value_range, dtype):
    inp = _sparse_input(shape, sparse_dim, dtype, value_range, dim=dim)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_log_softmax(inp, dim, half_to_float=False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("in_dtype,out_dtype", CONVERSION_ROWS)
def test__sparse_log_softmax_int_conversion(in_dtype, out_dtype):
    inp = _sparse_input(_PARAM_SHAPE, _PARAM_SPARSE_DIM, in_dtype, dim=_PARAM_DIM)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.int(ref_inp, _PARAM_DIM, out_dtype)
    res_out = flag_gems._sparse_log_softmax(inp, _PARAM_DIM, dtype=out_dtype)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("dtype", DEFAULT_DTYPE_DTYPES)
def test__sparse_log_softmax_int_default_dtype(dtype):
    inp = _sparse_input(_PARAM_SHAPE, _PARAM_SPARSE_DIM, dtype, dim=_PARAM_DIM)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.int(ref_inp, _PARAM_DIM)
    res_out = flag_gems._sparse_log_softmax(inp, _PARAM_DIM)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("shape,sparse_dim,dim", SPECIAL_ROWS)
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test__sparse_log_softmax_special_values(shape, sparse_dim, dim, dtype, scenario):
    inp = _sparse_special_input(shape, sparse_dim, dtype, scenario, dim=dim)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_log_softmax(inp, dim, half_to_float=False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("in_dtype,out_dtype,scenario", SPECIAL_CONVERSION_CASES)
def test__sparse_log_softmax_int_special_values(in_dtype, out_dtype, scenario):
    inp = _sparse_special_input(
        _PARAM_SHAPE, _PARAM_SPARSE_DIM, in_dtype, scenario, dim=_PARAM_DIM
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.int(ref_inp, _PARAM_DIM, out_dtype)
    res_out = flag_gems._sparse_log_softmax(inp, _PARAM_DIM, dtype=out_dtype)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("shape,sparse_dim,dim", BACKWARD_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_log_softmax_backward(shape, sparse_dim, dim, dtype):
    inp = _sparse_input(shape, sparse_dim, dtype, dim=dim, requires_grad=True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_log_softmax(inp, dim, half_to_float=False)
    tu.assert_result_close(res_out, ref_out)

    # Both sides use the same value sequence on their own output coordinates, so
    # a wrong gradient support or a wrong value ordering is detected.
    res_grad = torch.autograd.grad(
        res_out, inp, grad_outputs=_sparse_upstream(res_out)
    )[0]
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=_sparse_upstream(ref_out)
    )[0]

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("shape,sparse_dim,dim", EMPTY_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_log_softmax_empty_input(shape, sparse_dim, dim, dtype):
    inp = _sparse_input(shape, sparse_dim, dtype, nnz=0, dim=dim)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_log_softmax(inp, dim, half_to_float=False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("shape,sparse_dim,dim", UNCOALESCED_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_log_softmax_uncoalesced_input(shape, sparse_dim, dim, dtype):
    inp = _uncoalesced_input(shape, sparse_dim, dtype, dim)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_log_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_log_softmax(inp, dim, half_to_float=False)

    # The native kernel coalesces internally, so the output holds one entry per
    # distinct coordinate; the candidate must agree.
    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("shape,sparse_dim,dim", OUT_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_log_softmax_out(shape, sparse_dim, dim, dtype):
    inp = _sparse_input(shape, sparse_dim, dtype, dim=dim)
    ref_inp = tu.to_reference(inp)
    ref_out = _empty_coo_like(ref_inp)
    res_out = _empty_coo_like(inp)

    ref_ret = torch.ops.aten._sparse_log_softmax.out(
        ref_inp, dim, half_to_float=False, out=ref_out
    )
    res_ret = flag_gems._sparse_log_softmax(inp, dim, half_to_float=False, out=res_out)

    assert res_ret is res_out
    tu.assert_result_close(res_ret, ref_ret)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("shape,sparse_dim,dim", INVALID_DIM_ROWS)
def test__sparse_log_softmax_rejects_invalid_dim(shape, sparse_dim, dim):
    inp = _sparse_input(shape, sparse_dim, torch.float32, dim=dim)

    with pytest.raises(IndexError):
        flag_gems._sparse_log_softmax(inp, dim, half_to_float=False)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("nnz,dim", SCALAR_NEG_ROWS)
def test__sparse_log_softmax_rejects_scalar_input(nnz, dim):
    inp = _scalar_coo(nnz, torch.float32)

    with pytest.raises(IndexError):
        flag_gems._sparse_log_softmax(inp, dim, half_to_float=False)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES if _VENDOR_MEASURED else [])
def test__sparse_log_softmax_rejects_unsupported_dtype(dtype):
    inp = _sparse_input((256,), 1, dtype, dim=0)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_log_softmax(inp, 0, half_to_float=False)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("dtype", INVALID_DTYPE_PARAMS if _VENDOR_MEASURED else [])
def test__sparse_log_softmax_rejects_invalid_dtype_param(dtype):
    inp = _sparse_input((256,), 1, torch.float32, dim=0)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_log_softmax(inp, 0, dtype=dtype)


@pytest.mark._sparse_log_softmax
@pytest.mark.parametrize("dtype", HALF_TO_FLOAT_DTYPES if _VENDOR_MEASURED else [])
def test__sparse_log_softmax_rejects_half_to_float(dtype):
    inp = _sparse_input((256,), 1, dtype, dim=0)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_log_softmax(inp, 0, half_to_float=True)


@pytest.mark._sparse_log_softmax
def test__sparse_log_softmax_rejects_dense_input():
    # Native raises NotImplementedError (a RuntimeError subclass) for a strided
    # input, since only the SparseCPU/SparseCUDA kernels exist.
    inp = tu.make_input(torch.float32, (256,), ("-1", "1"))

    with pytest.raises(RuntimeError):
        flag_gems._sparse_log_softmax(inp, 0, half_to_float=False)
