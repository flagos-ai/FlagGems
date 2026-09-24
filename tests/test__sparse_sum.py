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

from . import accuracy_utils as utils
from . import test_utils as tu

# '_sparse_sum' starts with an underscore, so pytest cannot create the marker by
# attribute access; register it explicitly (this also enables -m _sparse_sum).
setattr(
    pytest.mark,
    "_sparse_sum",
    MarkDecorator(Mark("_sparse_sum", (), {}, _ispytest=True), _ispytest=True),
)

# Value-grid operand dtypes. A sparse COO tensor of an fp8 dtype constructs fine,
# but on the measured NVIDIA CUDA backend the native .default / .dim / .dtype forms
# have no fp8 kernel: they raise
# RuntimeError: 'coalesce_sparse_cuda' not implemented for 'Float8_e4m3fn' (the
# same for 'Float8_e5m2'), so fp8 is an invalid operand for those forms (covered by
# test_sparse_sum_rejects_fp8_operand, whose case list is vendor-scoped) and its
# special-value scenarios are
# inapplicable there for the same reason. The failure happens while coalescing,
# before any stored value is read, so it is a kernel-capability gap and not a
# value-domain one. The .dim_dtype form casts the stored values before reducing and
# does work for fp8 (probed for both formats), so that overload keeps fp8 positives,
# including the nan/inf matrix of test_sparse_sum_dim_dtype_special_values.
_SUM_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.float16,
    torch.int32,
]
# bf16 / fp64 / int64 join from collection-time capability flags. This concerns the
# stored value dtype only: a COO index buffer is int64 everywhere.
if utils.int64_is_supported:
    _SUM_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    _SUM_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _SUM_DTYPES.append(torch.float64)

_FLOAT_DTYPES = [dtype for dtype in _SUM_DTYPES if dtype.is_floating_point]

# ScalarType targets of the dtype overload; all probed on a float32 operand.
_DTYPE_TARGETS = [
    torch.float32,
    torch.float16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.bool,
]
if utils.int64_is_supported:
    _DTYPE_TARGETS.append(torch.int64)
if utils.bf16_is_supported:
    _DTYPE_TARGETS.append(torch.bfloat16)
if utils.fp64_is_supported:
    _DTYPE_TARGETS.append(torch.float64)

_DIM_DTYPE_TARGETS = [torch.float32]
if utils.int64_is_supported:
    _DIM_DTYPE_TARGETS.append(torch.int64)
if utils.fp64_is_supported:
    _DIM_DTYPE_TARGETS.append(torch.float64)

# Cast targets used for the fp8 .dim_dtype positives. The cast happens before the
# reduction, so these are the dtypes the stored fp8 values are widened into.
_FP8_CAST_TARGETS = [torch.float32, torch.float16]
if utils.int64_is_supported:
    _FP8_CAST_TARGETS.append(torch.int64)
if utils.bf16_is_supported:
    _FP8_CAST_TARGETS.append(torch.bfloat16)

# Positive fp8 .dim_dtype special values. Because that overload casts the stored
# fp8 values into the target dtype before reducing, a stored nan (e4m3fn, e5m2) or
# inf (e5m2 only) reaches the result and is meaningful; the shared generator already
# yields nan-only for e4m3fn and nan / inf / mixed for e5m2. Cast targets stay
# floating: with an integer target, casting a nan/inf operand is not a meaningful
# comparison, and the normal fp8 rows above keep the integer targets covered.
_FP8_SPECIAL_CASES = (
    tu.special_value_cases([torch.float8_e4m3fn, torch.float8_e5m2])
    if utils.fp8_is_supported
    else []
)
_FP8_SPECIAL_TARGETS = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    _FP8_SPECIAL_TARGETS.append(torch.bfloat16)

# The fp8 special-value operand is a (2, 3) COO tensor; the whole-dim form gives a
# dense 0-dim result, the partial form keeps a sparse result.
_FP8_SPECIAL_SHAPE = (2, 3)
_FP8_SPECIAL_DIMS = [[0, 1], [0]]

# The NVIDIA fp8 rejection below is vendor-specific, so its dtype list is chosen
# statically at collection time: on a backend whose native sparse fp8 reduction is
# supported the list is empty and the workload is not collected at all, instead of
# being skipped inside the test body.
_FP8_REJECT_DTYPES = (
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if utils.fp8_is_supported and flag_gems.vendor_name == "nvidia"
    else []
)

# Parameter coverage for the dim argument: positive, negative and empty dim lists
# over the spec shapes. A rank-2 operand accepts dims in [-2, 1] and a rank-1
# operand only [0] / [-1].
_DIM_ROWS = [
    ((256,), [0]),
    ((256,), [-1]),
    ((1024, 1024), [0]),
    ((1024, 1024), [1]),
    ((1024, 1024), [0, 1]),
    ((1024, 1024), []),
    ((20, 320, 15), [0]),
    ((20, 320, 15), [0, 2]),
    ((16, 128, 64, 60), [1]),
    ((16, 7, 57, 32, 29), [0]),
    ((16, 7, 57, 32, 29), [-1]),
    # A 0-dim operand has no dim to reduce: native rejects dim [0] with
    # 'Trying to create tensor with negative dimension -1' (covered by the
    # negative test), while the empty dim list is the valid form and returns a
    # dense 0-dim tensor.
    ((), []),
]

_DIM_DTYPE_ROWS = [
    ((1024, 1024), [0]),
    ((20, 320, 15), [0, 2]),
    ((256,), [0]),
    ((), []),
]

# .dim_dtype cases: (operand dtype, shape, dims, cast target). fp8 operands are
# default-only positives gated on the static backend fp8 capability flag.
_DIM_DTYPE_CASES = [
    (torch.float32, shape, dims, target)
    for shape, dims in _DIM_DTYPE_ROWS
    for target in _DIM_DTYPE_TARGETS
]
if utils.fp8_is_supported:
    _DIM_DTYPE_CASES += [
        (operand, (2, 3, 4), dims, target)
        for operand in (torch.float8_e4m3fn, torch.float8_e5m2)
        for dims in ([0], [0, 1], [])
        for target in _FP8_CAST_TARGETS
    ]

# The .out workloads reduce dim 0 or dim 1 of this shape with distinct
# coordinates, so no merge is required and the sparse out buffer's nnz is known.
_OUT_SHAPE = (256, 128)

# Coordinate structures. Each comment states the actual pattern of its row and
# the sparse-dim count of its coordinate list.
_STRUCTURE_ROWS = [
    # five unique coordinates inside a (3, 3) shape: nothing to merge.
    ((3, 3), [[0, 0, 1, 1, 2], [0, 1, 2, 0, 1]], [1.0, 2.0, 3.0, 4.0, 5.0]),
    # all five entries share coordinate (0, 0): merging must sum them.
    ((2, 3), [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [1.0, 2.0, 3.0, 4.0, 5.0]),
    # exactly one duplicated pair, (0, 0), plus three unique coordinates.
    ((2, 3), [[0, 0, 0, 1, 1], [0, 0, 1, 0, 1]], [1.0, 2.0, 3.0, 4.0, 5.0]),
    # four unique diagonal coordinates.
    ((6, 4), [[0, 1, 2, 3], [0, 1, 2, 3]], [1.0, 2.0, 3.0, 4.0]),
    # hybrid COO (two sparse dims, one dense tail dim) with three unique coords.
    (
        (2, 3, 3),
        [[0, 0, 1], [0, 2, 1]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    ),
    # hybrid COO whose three entries all share coordinate (0, 0): merging must
    # sum the dense tails position-wise.
    (
        (2, 3, 3),
        [[0, 0, 0], [0, 0, 0]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    ),
]

# Backward is default-only, and each row carries the dtypes that natively support
# that form on the active backend. Rows whose native result is a sparse tensor go
# through _sparse_sum_backward_cuda, whose CUDA dispatch covers half/float/double
# but NOT bfloat16: backward of a bf16 partial reduction raises RuntimeError
# '"_sparse_sum_backward_cuda" not implemented for BFloat16' (the dense-result
# rows differentiate fine in bf16). bf16 is therefore only exercised on the rows
# whose native result is dense.
_SPARSE_BACKWARD_DTYPES = [torch.float32, torch.float16]
if utils.fp64_is_supported:
    _SPARSE_BACKWARD_DTYPES.append(torch.float64)

_BACKWARD_ROWS = [
    # ((), None) is the all-dim .default form; (256,) dim [0] and
    # (1024, 1024) dims [0, 1] both reduce to a dense result.
    (shape, dims, dtype)
    for shape, dims in (((), None), ((256,), [0]), ((1024, 1024), [0, 1]))
    for dtype in _FLOAT_DTYPES
] + [
    # This partial reduction keeps a sparse result, so bf16 is absent here.
    ((1024, 1024), [0], dtype)
    for dtype in _SPARSE_BACKWARD_DTYPES
]

_NEGATIVE_DIM_CASES = [
    ((1024, 1024), [2]),
    ((1024, 1024), [-3]),
    ((20, 320, 15), [3]),
    ((20, 320, 15), [-4]),
]

# Stored values per spec shape. A sparse reduction's meaningful cost is nnz, so
# the largest spec shapes carry _MAX_NNZ coordinates rather than their full numel;
# this keeps the index buffer a few tens of KB instead of hundreds of MB while
# still giving duplicate coordinates for the smaller shapes.
_MAX_NNZ = 4096


def _nnz_for(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return max(1, min(numel, _MAX_NNZ))


def _random_coords(shape, nnz):
    # Random coordinates drawn with replacement: the operand is uncoalesced and
    # contains duplicates, which is the workload native _sparse_sum coalesces
    # internally, so merging is exercised as well.
    if len(shape) == 0:
        # A 0-dim sparse COO tensor stores an (0, nnz) index buffer; this is the
        # only valid construction for shape () (probed with torch.sparse_coo_tensor
        # on the active backend).
        return torch.empty((0, nnz), dtype=torch.long, device=flag_gems.device)
    return torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=flag_gems.device)
            for dim in shape
        ]
    )


def _distinct_coords(shape, nnz):
    # One coordinate row per sparse dim, each a ramp i % dim: distinct along every
    # axis as long as nnz <= min(shape), which is the precondition of the .out
    # buffers below. Callers whose nnz exceeds min(shape) use _unravel_coords.
    if len(shape) == 0:
        return torch.empty((0, nnz), dtype=torch.long, device=flag_gems.device)
    return torch.stack(
        [
            torch.arange(nnz, dtype=torch.long, device=flag_gems.device) % dim
            for dim in shape
        ]
    )


def _unravel_coords(shape, nnz):
    # The first nnz linear indices unravelled into shape. The coordinate vectors are
    # genuinely distinct even when nnz exceeds min(shape), because _nnz_for caps nnz
    # at prod(shape); a per-axis ramp would repeat coordinates in that case.
    if len(shape) == 0:
        return torch.empty((0, nnz), dtype=torch.long, device=flag_gems.device)
    linear = torch.arange(nnz, dtype=torch.long, device=flag_gems.device)
    rows = []
    stride = 1
    for dim in reversed(shape):
        rows.append((linear // stride) % dim)
        stride *= dim
    return torch.stack(list(reversed(rows)))


def _sparse_operand(dtype, shape, value_range, nnz=None, indices=None, values=None):
    if nnz is None:
        nnz = _nnz_for(shape)
    if values is None:
        values = tu.make_input(dtype, (nnz,), value_range)
    if indices is None:
        indices = _random_coords(shape, nnz)
    # The operand lives wherever its stored values live: the candidate side sits on
    # flag_gems.device, the reference side on the configured reference device.
    return torch.sparse_coo_tensor(
        indices.to(values.device), values, shape, device=values.device
    )


def _explicit_operand(dtype, shape, coords, payload):
    # coords carries one row per sparse dim, so converting it directly keeps the
    # (sparse_dim, nnz) index shape for plain (rank == sparse_dim) and hybrid COO
    # operands alike; a rank-based reshape would truncate the hybrid index matrix.
    # The payload literals are constructible in the target dtype directly, so no
    # float32 intermediate (and no extra rounding) is introduced.
    indices = torch.tensor(coords, dtype=torch.long, device=flag_gems.device)
    values = torch.tensor(payload, dtype=dtype, device=flag_gems.device)
    # A nested payload describes the dense tail of a hybrid COO tensor (one row per
    # stored entry); a flat payload is the stored-value vector of a plain tensor.
    if values.dim() > 1:
        values = values.reshape(indices.shape[1], -1)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _empty_operand(shape):
    indices = torch.empty((len(shape), 0), dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _observation(out):
    # Nonconstant upstream weight: the reduced observation depends on the position
    # of each stored value, so a gradient that drops or permutes positions is
    # visible. Both sides use the same coordinates, so the weights line up.
    dense = out.to_dense() if out.is_sparse else out
    weight = torch.linspace(
        1.0, 2.0, dense.numel(), dtype=torch.float32, device=dense.device
    ).to(dense.dtype)
    return (dense * weight).sum()


@pytest.mark._sparse_sum
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUM_DTYPES)
def test_sparse_sum_default(shape, value_range, dtype):
    inp = _sparse_operand(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum(ref_inp)
    res_out = flag_gems._sparse_sum(inp)

    # assert_result_close also pins the result dtype, so the int/bool -> int64
    # promotion of this overload is checked against the native reference.
    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("shape,dims", tu.selected_cases(_DIM_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _SUM_DTYPES)
def test_sparse_sum_dim(shape, dims, dtype):
    inp = _sparse_operand(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum.dim(ref_inp, dims)
    res_out = flag_gems._sparse_sum(inp, dims)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("target", tu.selected_cases(_DTYPE_TARGETS, quick=[]))
def test_sparse_sum_dtype(target):
    # Parameter-coverage workload: spec shape (1024, 1024), range [-1, 1], fp32
    # operand so every target ScalarType is a representable cast.
    inp = _sparse_operand(torch.float32, (1024, 1024), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum.dtype(ref_inp, dtype=target)
    res_out = flag_gems._sparse_sum(inp, dtype=target)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize(
    "operand_dtype,shape,dims,target",
    tu.selected_cases(_DIM_DTYPE_CASES, quick=[]),
)
def test_sparse_sum_dim_dtype(operand_dtype, shape, dims, target):
    inp = _sparse_operand(operand_dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum.dim_dtype(ref_inp, dims, dtype=target)
    res_out = flag_gems._sparse_sum(inp, dims, dtype=target)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("dims", _FP8_SPECIAL_DIMS)
@pytest.mark.parametrize("target", _FP8_SPECIAL_TARGETS)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(_FP8_SPECIAL_CASES, quick=[])
)
def test_sparse_sum_dim_dtype_special_values(dtype, scenario, target, dims):
    # Positive nan/inf cases are default-only (the quick selection is empty). The
    # shared generator gives e4m3fn a nan-only case and e5m2 the nan / inf / mixed
    # cases, matching each format's representable values. Coordinates are distinct,
    # so no merging changes the special-value reduction.
    values = tu.make_special_input(dtype, scenario)
    nnz = values.numel()
    inp = _sparse_operand(
        dtype,
        _FP8_SPECIAL_SHAPE,
        ["-1", "1"],
        nnz=nnz,
        indices=_unravel_coords(_FP8_SPECIAL_SHAPE, nnz),
        values=values,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum.dim_dtype(ref_inp, dims, dtype=target)
    res_out = flag_gems._sparse_sum(inp, dims, dtype=target)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("dims", [[0], [1]])
@pytest.mark.parametrize("dtype", tu.selected_cases(_SUM_DTYPES, quick=[]))
def test_sparse_sum_dim_out(dims, dtype):
    nnz = min(_OUT_SHAPE)
    indices = _distinct_coords(_OUT_SHAPE, nnz)
    inp = _sparse_operand(dtype, _OUT_SHAPE, ["-1", "1"], nnz=nnz, indices=indices)
    ref_inp = tu.to_reference(inp)

    # A partial reduction keeps the operand's dtype, so the sparse out buffer uses
    # the same dtype and device as its own input side.
    out_shape = (_OUT_SHAPE[1],) if dims == [0] else (_OUT_SHAPE[0],)
    out = torch.sparse_coo_tensor(
        torch.zeros((1, nnz), dtype=torch.long, device=flag_gems.device),
        torch.zeros(nnz, dtype=dtype, device=flag_gems.device),
        out_shape,
        device=flag_gems.device,
    )
    ref_out = torch.sparse_coo_tensor(
        torch.zeros((1, nnz), dtype=torch.long, device=ref_inp.device),
        torch.zeros(nnz, dtype=dtype, device=ref_inp.device),
        out_shape,
        device=ref_inp.device,
    )

    res_ret = flag_gems._sparse_sum(inp, dims, out=out)
    ref_ret = torch.ops.aten._sparse_sum.dim_out(ref_inp, dims, out=ref_out)

    # The .out form writes into and returns the provided buffer.
    assert res_ret is out
    tu.assert_result_close(res_ret, ref_ret)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("dtype", tu.selected_cases(_SUM_DTYPES, quick=[]))
def test_sparse_sum_dim_out_dense(dtype):
    # Reducing every dim yields a dense 0-dim result. Integer operands promote to
    # int64 (as the .default overload shows), so their out buffer is int64 while
    # floating operands write into an out buffer of their own dtype.
    nnz = min(_OUT_SHAPE)
    indices = _distinct_coords(_OUT_SHAPE, nnz)
    inp = _sparse_operand(dtype, _OUT_SHAPE, ["-1", "1"], nnz=nnz, indices=indices)
    ref_inp = tu.to_reference(inp)

    result_dtype = dtype if dtype.is_floating_point else torch.int64
    out = torch.zeros((), dtype=result_dtype, device=flag_gems.device)
    ref_out = torch.zeros((), dtype=result_dtype, device=ref_inp.device)

    res_ret = flag_gems._sparse_sum(inp, [0, 1], out=out)
    ref_ret = torch.ops.aten._sparse_sum.dim_out(ref_inp, [0, 1], out=ref_out)

    assert res_ret is out
    tu.assert_result_close(res_ret, ref_ret)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("shape", [(256,), (1024, 1024)])
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[]),
)
def test_sparse_sum_special_values(shape, dtype, scenario):
    # Positive nan/inf cases are default-only (the quick selection is empty).
    # Coordinates stay distinct so no merging changes the special-value sum.
    values = tu.make_special_input(dtype, scenario)
    nnz = values.numel()
    inp = _sparse_operand(
        dtype,
        shape,
        ["-1", "1"],
        nnz=nnz,
        indices=_unravel_coords(shape, nnz),
        values=values,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum(ref_inp)
    res_out = flag_gems._sparse_sum(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize(
    "shape,dims,dtype", tu.selected_cases(_BACKWARD_ROWS, quick=[])
)
def test_sparse_sum_backward(shape, dims, dtype):
    # Backward is default-only (the quick selection is empty). Each side owns an
    # independent differentiable values tensor built from the same values, and
    # both sides use the same coordinates so the nonconstant upstream weights of
    # _observation line up. The coordinates are distinct even where nnz exceeds
    # min(shape) (the (1024, 1024) partial-reduction row), so no merging is part of
    # the compared gradient; duplicate merging is covered by the structure rows.
    nnz = _nnz_for(shape)
    indices = _unravel_coords(shape, nnz)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    res_values = values.detach().clone().requires_grad_(True)
    ref_values = tu.to_reference(values).detach().clone().requires_grad_(True)

    res_inp = _sparse_operand(
        dtype, shape, ["-1", "1"], nnz=nnz, indices=indices, values=res_values
    )
    ref_inp = _sparse_operand(
        dtype, shape, ["-1", "1"], nnz=nnz, indices=indices, values=ref_values
    )

    if dims is None:
        ref_out = torch.ops.aten._sparse_sum(ref_inp)
        res_out = flag_gems._sparse_sum(res_inp)
    else:
        ref_out = torch.ops.aten._sparse_sum.dim(ref_inp, dims)
        res_out = flag_gems._sparse_sum(res_inp, dims)

    tu.assert_result_close(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(_observation(ref_out), ref_values)
    (res_grad,) = torch.autograd.grad(_observation(res_out), res_values)

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark._sparse_sum
@pytest.mark.parametrize(
    "shape,coords,payload", tu.selected_cases(_STRUCTURE_ROWS, quick=[])
)
def test_sparse_sum_structure_default(shape, coords, payload):
    inp = _explicit_operand(torch.float32, shape, coords, payload)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum(ref_inp)
    res_out = flag_gems._sparse_sum(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("dims", [[0], [1], [0, 1]])
@pytest.mark.parametrize(
    "shape,coords,payload", tu.selected_cases(_STRUCTURE_ROWS, quick=[])
)
def test_sparse_sum_structure_dim(shape, coords, payload, dims):
    inp = _explicit_operand(torch.float32, shape, coords, payload)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum.dim(ref_inp, dims)
    res_out = flag_gems._sparse_sum(inp, dims)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize(
    "shape", tu.selected_cases([(0,), (0, 8), (8, 0), (0, 0)], quick=[])
)
def test_sparse_sum_empty_default(shape):
    inp = _empty_operand(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum(ref_inp)
    res_out = flag_gems._sparse_sum(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("dims", [[], [0]])
@pytest.mark.parametrize(
    "shape", tu.selected_cases([(0,), (0, 8), (8, 0), (0, 0)], quick=[])
)
def test_sparse_sum_empty_dim(shape, dims):
    inp = _empty_operand(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_sum.dim(ref_inp, dims)
    res_out = flag_gems._sparse_sum(inp, dims)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum
@pytest.mark.parametrize(
    "shape,dims", tu.selected_cases(_NEGATIVE_DIM_CASES, quick=_NEGATIVE_DIM_CASES)
)
def test_sparse_sum_rejects_out_of_range_dim(shape, dims):
    # A rank-2 operand accepts dims in [-2, 1] and a rank-3 operand in [-3, 2],
    # so these positive and negative values are outside the operand's own rank.
    inp = _sparse_operand(torch.float32, shape, ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems._sparse_sum(inp, dims)


@pytest.mark._sparse_sum
def test_sparse_sum_rejects_duplicate_dim():
    inp = _sparse_operand(torch.float32, (1024, 1024), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._sparse_sum(inp, [0, 0])


@pytest.mark._sparse_sum
def test_sparse_sum_rejects_dim_on_scalar():
    # The 0-dim operand has no dim 0: native raises 'Trying to create tensor with
    # negative dimension -1: [-1, nnz]' for dim [0].
    inp = _sparse_operand(torch.float32, (), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._sparse_sum(inp, [0])


@pytest.mark._sparse_sum
def test_sparse_sum_rejects_dense_input():
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        flag_gems._sparse_sum(inp)


@pytest.mark._sparse_sum
@pytest.mark.parametrize("dtype", _FP8_REJECT_DTYPES)
def test_sparse_sum_rejects_fp8_operand(dtype):
    # The fp8 .default form has no kernel on the NVIDIA CUDA backend this rejection
    # was measured on (see _FP8_REJECT_DTYPES): it raises 'coalesce_sparse_cuda' not
    # implemented for 'Float8_e4m3fn' / 'Float8_e5m2' while coalescing, independently
    # of the stored values, so a sparse fp8 operand is an invalid input for this
    # overload rather than a value-grid row.
    indices = torch.zeros((1, 3), dtype=torch.long, device=flag_gems.device)
    values = torch.zeros(3, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4,), device=flag_gems.device)
    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems._sparse_sum(inp)
