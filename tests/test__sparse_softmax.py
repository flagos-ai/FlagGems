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

# The operator name starts with an underscore, which pytest's MarkGenerator cannot
# synthesise, so the marker is registered explicitly.
setattr(
    pytest.mark,
    "_sparse_softmax",
    MarkDecorator(Mark("_sparse_softmax", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_sparse_softmax(x, dim, False) binds the positional bool to the ScalarType
# parameter of .int, so every family names its overload: .default for the
# half_to_float form, .int for dtype= (and for an omitted dtype, which keeps the
# input dtype), .out for the out form. half_to_float has no default in .default.
# The candidate is always the public entry point flag_gems._sparse_softmax called
# with the same arguments.

_DEVICE = flag_gems.runtime.device

# Measured on this NVIDIA target: the default half_to_float=False form has native
# kernels for float32/float64 only, the .int dtype= form converts
# int8/uint8/int32/int64/bfloat16/fp8 inputs to float32/float64, and float16 fails
# on both. Those kernel limits are scoped to the vendor they were measured on;
# schema/shape rejections (dense input, invalid dim, 0-dim input) are generic.
# The capability flags only keep out constructs this device cannot represent.
_IS_NVIDIA = _DEVICE.vendor_name == "nvidia"

_DEFAULT_FORM_DTYPES = [torch.float32] + (
    [torch.float64] if _DEVICE.support_fp64 else []
)
_CONVERT_DTYPES = (
    [torch.int8, torch.uint8]
    + ([torch.float8_e4m3fn, torch.float8_e5m2] if _DEVICE.support_fp8 else [])
    + [torch.int32]
    + ([torch.bfloat16] if _DEVICE.support_bf16 else [])
    + ([torch.int64] if _DEVICE.support_int64 else [])
)
_CONVERT_FLOAT_DTYPES = (
    [torch.float8_e4m3fn, torch.float8_e5m2] if _DEVICE.support_fp8 else []
) + ([torch.bfloat16] if _DEVICE.support_bf16 else [])
_FP64_TARGET_DTYPES = list(_CONVERT_DTYPES) if _DEVICE.support_fp64 else []

# Default-form rejections on NVIDIA: no native softmax kernel for these dtypes.
_NVIDIA_REJECTED_DTYPES = (
    [torch.int8, torch.uint8, torch.float16, torch.int32]
    + ([torch.bfloat16] if _DEVICE.support_bf16 else [])
    + ([torch.int64] if _DEVICE.support_int64 else [])
)
_REJECTED_DTYPES = _NVIDIA_REJECTED_DTYPES if _IS_NVIDIA else []
# FP8 inputs reach the default form's internal coalesce, which FP8 lacks on NVIDIA.
_REJECTED_FP8_DTYPES = (
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if _IS_NVIDIA and _DEVICE.support_fp8
    else []
)
_HALF_TO_FLOAT_REJECTED_DTYPES = list(_DEFAULT_FORM_DTYPES) if _IS_NVIDIA else []


def _linear_coords(shape, nnz):
    # Distinct ascending row-major columns: a lattice of stride max(size // nnz, 1)
    # spreads the entries over the index space without permuting it, which for the
    # large spec shapes would cost far more than the stored entries it selects.
    if nnz == 0:
        # No stored entries; every axis keeps its requested extent, so zero-size
        # axes are expressed here too.
        return torch.empty((len(shape), 0), dtype=torch.long, device=flag_gems.device)
    if not shape:
        return torch.empty((0, nnz), dtype=torch.long, device=flag_gems.device)
    size = 1
    for extent in shape:
        size *= extent
    step = max(size // nnz, 1)
    flat = torch.arange(nnz, dtype=torch.long, device=flag_gems.device) * step
    remainder = flat
    columns = []
    for extent in reversed(shape):
        columns.append(remainder % extent)
        remainder = remainder // extent
    return torch.stack(list(reversed(columns)))


def _group_coords(shape, dim, groups, per_group):
    # per_group entries per normalization group; the group's other coordinates are
    # derived over every axis except dim, so groups stay distinct when the
    # normalized axis is a middle dimension. dim is normalized first so negative
    # dims exclude and unravel the correct axis.
    dim = dim % len(shape)
    axes = [extent for axis, extent in enumerate(shape) if axis != dim]
    total = 1
    for extent in axes:
        total *= extent
    group = torch.arange(groups, dtype=torch.long, device=flag_gems.device) % total
    remainder = group
    parts = []
    for extent in reversed(axes):
        parts.append(remainder % extent)
        remainder = remainder // extent
    per_axis = list(reversed(parts))
    stride = max(shape[dim] // per_group, 1)
    positions = (
        torch.arange(per_group, dtype=torch.long, device=flag_gems.device) * stride
    )
    columns = []
    axis_index = 0
    for axis in range(len(shape)):
        if axis == dim:
            columns.append(positions.repeat(groups))
        else:
            columns.append(per_axis[axis_index].repeat_interleave(per_group))
            axis_index += 1
    return torch.stack(columns)


def _support_coords(shape, dim, support):
    kind = support[0]
    if kind in ("coords", "dup"):
        indices = _linear_coords(shape, support[1])
        if kind == "dup":
            indices = torch.cat([indices, indices], dim=1)
        return indices
    return _group_coords(shape, dim, support[1], support[2])


def _stored_values(count, dtype, value_range=None, scenario=None):
    if scenario is None:
        return tu.make_input(dtype, (count,), value_range)
    payload = tu.make_special_input(dtype, scenario)
    repeats = (count + payload.numel() - 1) // payload.numel()
    return payload.repeat(repeats)[:count]


def _make_sparse(shape, dim, support, dtype, value_range=None, scenario=None):
    indices = _support_coords(shape, dim, support)
    values = _stored_values(indices.shape[1], dtype, value_range, scenario)
    inp = torch.sparse_coo_tensor(
        indices, values, shape, dtype=dtype, device=flag_gems.device
    )
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 has no sparse coalesce kernel; the lattice columns are already
        # distinct, so the stored order is coalesced by content.
        return inp
    if support[0] == "dup":
        # Left uncoalesced on purpose: the operator must merge duplicate columns.
        return inp
    return inp.coalesce()


def _make_hybrid(shape, sparse_dim, nnz, dtype, value_range):
    # One dense value block shape[sparse_dim:] per stored entry; with sparse_dim 0
    # the whole shape is the block, so a single entry covers it.
    dense = tuple(shape[sparse_dim:])
    indices = _linear_coords(shape[:sparse_dim], nnz)
    values = tu.make_input(dtype, (indices.shape[1],) + dense, value_range)
    return torch.sparse_coo_tensor(
        indices, values, shape, dtype=dtype, device=flag_gems.device
    ).coalesce()


def _empty_out(shape, dtype, device):
    indices = torch.empty((len(shape), 0), dtype=torch.long, device=device)
    values = torch.empty((0,), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)


def _rejection_input(shape, dim, nnz, dtype):
    # Coalesced input on purpose: the rejection under test is the missing reduction
    # kernel, not a missing sparse coalesce kernel (fp8 has none, so its lattice
    # columns stay distinct and its own rejection test documents that limit).
    indices = _linear_coords(shape, nnz)
    values = torch.zeros(indices.shape[1], dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(
        indices, values, shape, dtype=dtype, device=flag_gems.device
    )
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return inp
    return inp.coalesce()


def _upstream_like(inp, shape):
    # Non-constant upstream over groups of several entries: an all-ones upstream
    # only checks zero gradients, since the entries of one group sum to one.
    values = tu.make_input(inp.dtype, (inp._nnz(),), ["-1", "1"])
    return torch.sparse_coo_tensor(
        inp._indices(), values, shape, dtype=inp.dtype, device=inp.device
    )


# No broadcast family: the operator takes a single tensor operand.
#
# Default half_to_float=False rows: (shape, dim, support). The spec's 0-dim shape is
# a rejection workload (test__sparse_softmax_rejects_zero_dim_input) because a 0-dim
# COO tensor has no axis to normalize; every other spec shape appears here with
# positive/negative dims, multi-entry groups, uncoalesced duplicate columns and
# empty-nnz reductions, including zero-size shape axes.
_SPARSE_ROWS = tu.selected_cases(
    [
        ((1,), 0, ("coords", 1)),
        ((256,), 0, ("coords", 64)),
        ((1024, 1024), 1, ("coords", 4096)),
        ((20, 320, 15), 2, ("coords", 2048)),
        ((16, 128, 64, 60), 3, ("coords", 4096)),
        ((16, 7, 57, 32, 29), 4, ("coords", 8192)),
        ((16, 7, 57, 32, 29), 2, ("coords", 8192)),
        ((8, 9), 0, ("coords", 30)),
        ((8, 9), 1, ("coords", 30)),
        ((8, 9), -1, ("coords", 30)),
        ((20, 320, 15), 1, ("coords", 512)),
        ((20, 320, 15), -3, ("coords", 512)),
        ((64, 512), 1, ("group", 64, 16)),
        ((64, 512), -1, ("group", 64, 16)),
        ((20, 320, 15), 2, ("group", 300, 4)),
        ((20, 320, 15), -2, ("group", 300, 4)),
        ((2, 19, 7), 1, ("group", 6, 18)),
        ((256,), 0, ("group", 1, 64)),
        ((8, 9), 1, ("dup", 20)),
        ((1024, 1024), 1, ("dup", 512)),
        ((8, 9), 1, ("coords", 0)),
        ((0, 9), 1, ("coords", 0)),
        ((8, 0), 1, ("coords", 0)),
        ((16, 7, 57, 32, 29), 4, ("coords", 0)),
    ],
    quick=[((2, 19, 7), 2, ("coords", 128))],
)

# .int dtype= rows: the spec shapes without duplicate columns, because fp8 has no
# sparse coalesce kernel to merge them, plus an empty-nnz reduction.
_CONVERT_ROWS = tu.selected_cases(
    [
        ((1,), 0, ("coords", 1)),
        ((256,), 0, ("coords", 64)),
        ((1024, 1024), 1, ("coords", 4096)),
        ((20, 320, 15), 2, ("coords", 2048)),
        ((16, 128, 64, 60), 3, ("coords", 4096)),
        ((16, 7, 57, 32, 29), 4, ("coords", 8192)),
        ((2, 19, 7), 1, ("group", 6, 18)),
        ((8, 9), -1, ("coords", 30)),
        ((8, 9), 1, ("coords", 0)),
    ],
    quick=[],
)

# The other ScalarType target of the .int dtype= overload.
_CONVERT64_ROWS = tu.selected_cases(
    [((256,), 0, ("coords", 64)), ((8, 9), -1, ("coords", 30))], quick=[]
)

# Omitting the dtype keeps the input dtype: the .int overload with dtype=None.
_DTYPE_NONE_ROWS = tu.selected_cases(
    [((256,), 0, ("coords", 64)), ((1024, 1024), 1, ("coords", 4096))], quick=[]
)

# Hybrid COO rows: (shape, sparse_dim, nnz, dim). Reductions run over sparse axes and
# over dense value axes, including the fully dense sparse_dim 0 form.
_HYBRID_ROWS = tu.selected_cases(
    [
        ((8, 9, 4, 5), 2, 64, 2),
        ((8, 9, 4, 5), 2, 64, 3),
        ((16, 7, 57, 32, 4), 3, 128, 4),
        ((16, 7, 57, 32, 4), 3, 128, 2),
        ((4,), 0, 1, 0),
        ((4, 5), 0, 1, 0),
        ((4, 5), 0, 1, 1),
    ],
    quick=[],
)

_OUT_ROWS = tu.selected_cases(
    [((256,), 0, ("coords", 64)), ((1024, 1024), 1, ("coords", 4096))], quick=[]
)

_BACKWARD_ROWS = tu.selected_cases(
    [
        ((256,), 0, ("group", 1, 64)),
        ((8, 9), 1, ("group", 3, 3)),
        ((2, 19, 7), 1, ("group", 6, 18)),
        ((64, 512), -1, ("group", 64, 16)),
        ((20, 320, 15), 2, ("group", 300, 4)),
    ],
    quick=[],
)

# Several entries per group, so a NaN/Inf payload reaches more than one reduction.
_SPECIAL_ROWS = tu.selected_cases(
    [
        ((256,), 0, ("group", 1, 64)),
        ((64, 128), 1, ("group", 32, 8)),
        ((1024, 1024), 1, ("group", 128, 4)),
    ],
    quick=[],
)

# Scenarios follow the operator's supported dtypes: the default form has no native
# kernel for bfloat16 or fp8 here, so positive special values are covered for the
# converted inputs through the dtype= overload instead (see below).
_SPECIAL_SCENARIOS = tu.selected_cases(
    tu.special_value_cases(_DEFAULT_FORM_DTYPES), quick=[]
)

_SPECIAL_CONVERT_ROWS = tu.selected_cases(
    [((256,), 0, ("group", 1, 64)), ((64, 128), 1, ("group", 32, 8))], quick=[]
)
_SPECIAL_CONVERT_SCENARIOS = tu.selected_cases(
    tu.special_value_cases(_CONVERT_FLOAT_DTYPES), quick=[]
)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _SPARSE_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DEFAULT_FORM_DTYPES)
def test__sparse_softmax_default_form(case, value_range, dtype):
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_softmax(inp, dim, half_to_float=False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _CONVERT_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CONVERT_DTYPES)
def test__sparse_softmax_dtype_conversion(case, value_range, dtype):
    # The requested dtype is what makes int8/uint8/bf16/int32/int64/fp8 inputs
    # runnable at all.
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_softmax.int(ref_inp, dim, dtype=torch.float32)
    res_out = flag_gems._sparse_softmax(inp, dim, dtype=torch.float32)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _CONVERT64_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FP64_TARGET_DTYPES)
def test__sparse_softmax_dtype_conversion_float64(case, value_range, dtype):
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_softmax.int(ref_inp, dim, dtype=torch.float64)
    res_out = flag_gems._sparse_softmax(inp, dim, dtype=torch.float64)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _DTYPE_NONE_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DEFAULT_FORM_DTYPES)
def test__sparse_softmax_int_overload_keeps_input_dtype(case, value_range, dtype):
    # The dtype argument is omitted on both sides, so this also checks that the
    # candidate's own default matches the schema instead of a hardcoded float32.
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_softmax.int(ref_inp, dim)
    res_out = flag_gems._sparse_softmax(inp, dim)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _HYBRID_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DEFAULT_FORM_DTYPES)
def test__sparse_softmax_hybrid(case, value_range, dtype):
    shape, sparse_dim, nnz, dim = case
    inp = _make_hybrid(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_softmax(inp, dim, half_to_float=False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _OUT_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DEFAULT_FORM_DTYPES)
def test__sparse_softmax_out(case, value_range, dtype):
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, value_range=value_range)
    ref_inp = tu.to_reference(inp)
    out = _empty_out(shape, inp.dtype, inp.device)
    ref_out = _empty_out(shape, ref_inp.dtype, ref_inp.device)

    ref_ret = torch.ops.aten._sparse_softmax.out(
        ref_inp, dim, half_to_float=False, out=ref_out
    )
    res_ret = flag_gems._sparse_softmax(inp, dim, half_to_float=False, out=out)

    # Native .out returns the supplied buffer, so the candidate must both write it
    # and return it, with the values the native call returned.
    assert res_ret is out
    tu.assert_result_close(res_ret, ref_ret)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _BACKWARD_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DEFAULT_FORM_DTYPES)
def test__sparse_softmax_backward(case, value_range, dtype):
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, value_range=value_range)
    inp.requires_grad_(True)
    upstream = _upstream_like(inp, shape)
    ref_inp = tu.to_reference(inp)
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten._sparse_softmax.default(ref_inp, dim, False)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    res_out = flag_gems._sparse_softmax(inp, dim, half_to_float=False)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _SPECIAL_ROWS)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_SCENARIOS)
def test__sparse_softmax_special_values(case, dtype, scenario):
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, scenario=scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_softmax.default(ref_inp, dim, False)
    res_out = flag_gems._sparse_softmax(inp, dim, half_to_float=False)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("case", _SPECIAL_CONVERT_ROWS)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CONVERT_SCENARIOS)
def test__sparse_softmax_special_values_dtype_conversion(case, dtype, scenario):
    # The default form rejects bfloat16/fp8 inputs here, but the dtype= overload
    # converts them, so their NaN/Inf payloads still have a native result to match.
    shape, dim, support = case
    inp = _make_sparse(shape, dim, support, dtype, scenario=scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_softmax.int(ref_inp, dim, dtype=torch.float32)
    res_out = flag_gems._sparse_softmax(inp, dim, dtype=torch.float32)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test__sparse_softmax_rejects_dtype_in_default_form(dtype):
    # No native softmax kernel for the default half_to_float=False form on this
    # vendor; the convertible subset still runs through the dtype= overload above.
    inp = _rejection_input((8, 9), 1, 20, dtype)

    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._sparse_softmax(inp, 1, half_to_float=False)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("dtype", _REJECTED_FP8_DTYPES)
def test__sparse_softmax_rejects_fp8_in_default_form(dtype):
    # The default form coalesces the input first, and this vendor has no sparse
    # coalesce kernel for fp8, so fp8 inputs are rejected before the reduction.
    inp = _rejection_input((8, 9), 1, 20, dtype)

    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._sparse_softmax(inp, 1, half_to_float=False)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("dtype", _HALF_TO_FLOAT_REJECTED_DTYPES)
def test__sparse_softmax_rejects_half_to_float(dtype):
    # half_to_float=True has no native kernel on this vendor. The argument is passed
    # by name: a positional True would bind the ScalarType parameter of .int and the
    # test could pass for the wrong reason.
    inp = _make_sparse((256,), 0, ("coords", 64), dtype, value_range=["-1", "1"])

    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._sparse_softmax(inp, 0, half_to_float=True)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("shape,invalid_dim", [((8, 9), 2), ((20, 320, 15), -4)])
def test__sparse_softmax_rejects_out_of_range_dim(shape, invalid_dim):
    inp = _make_sparse(shape, 1, ("coords", 32), torch.float32, value_range=["-1", "1"])

    with pytest.raises((IndexError, RuntimeError, TypeError)):
        flag_gems._sparse_softmax(inp, invalid_dim, half_to_float=False)


@pytest.mark._sparse_softmax
@pytest.mark.parametrize("nnz", [0, 1, 3])
@pytest.mark.parametrize("dim", [0, -1])
def test__sparse_softmax_rejects_zero_dim_input(nnz, dim):
    # A 0-dim COO tensor has no axis to normalize: native raises IndexError for both
    # of its candidate dims (0 and -1), for empty and nonempty nnz alike, so the
    # spec's () shape is a rejection workload rather than a positive one.
    inp = torch.sparse_coo_tensor(
        torch.empty((0, nnz), dtype=torch.long, device=flag_gems.device),
        torch.empty(nnz, dtype=torch.float32, device=flag_gems.device),
        (),
        dtype=torch.float32,
        device=flag_gems.device,
    )

    with pytest.raises((IndexError, RuntimeError, TypeError)):
        flag_gems._sparse_softmax(inp, dim, half_to_float=False)


@pytest.mark._sparse_softmax
def test__sparse_softmax_rejects_dense_input():
    inp = tu.make_input(torch.float32, (8, 9), ["-1", "1"])

    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._sparse_softmax(inp, 1, half_to_float=False)
