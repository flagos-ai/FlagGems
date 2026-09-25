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

"""Correctness tests for aten::_sparse_sum_backward.

_sparse_sum_backward(grad, self, dim) is the autograd formula of aten::_sparse_sum:
self is SparseCOO and grad is dense when dim covers every sparse dim, sparse
otherwise - keeping the retained sparse dims plus every dense dim dim did not sum.

Fixtures hand self and grad over either an already coalesced SparseCOO (is_coalesced=True
on a family whose row-major keys ascend) or through .coalesce() where the coordinates can
repeat. Which path a descriptor takes is decided from its metadata by
_projected_sorted_unique, never by reading a tensor back: _ring_permutation sorts the ring
into ascending row-major order, so a self - and a grad that keeps every sparse axis -
reproduces that order exactly and is the family the constructor path accepts, which is
also the only path fp8 has since this backend has no fp8 coalesce kernel. A grad projected
onto a proper subset of the sparse axes can reorder and repeat the ring, so that family
coalesces explicitly; that is the branch where the native kernel regathers and is
float-only. is_coalesced=True declares the family, it does not verify it, so the coordinate
families are validated in the generation-time probe instead of being re-checked here. The
result layout is the one the native operator produces, so every family compares the
candidate result against the native reference rather than against a hand-written
expectation about the result's shape or sparse_dim.
"""

import math

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import test_utils as tu

setattr(
    pytest.mark,
    "_sparse_sum_backward",
    MarkDecorator(Mark("_sparse_sum_backward", (), {}, _ispytest=True), _ispytest=True),
)

# Existing static capability flags (tests/accuracy_utils.py re-exports the same values
# as bf16_is_supported / fp64_is_supported / int64_is_supported / fp8_is_supported).
_SUPPORTS_BF16 = flag_gems.runtime.device.support_bf16
_SUPPORTS_FP64 = flag_gems.runtime.device.support_fp64
_SUPPORTS_INT64 = flag_gems.runtime.device.support_int64
_SUPPORTS_FP8 = flag_gems.runtime.device.support_fp8

# Native dispatch, probed on this backend with the real overload:
#   * a dense grad (dim covers every sparse dim), and a sparse grad that keeps every
#     sparse dim - empty dense tail included - only gather indices and copy values, and
#     return a result for every dtype, float8_e4m3fn and float8_e5m2 included. Probed
#     with a dense grad on ((256,), 1, (0,), 32) and with an already-coalesced sparse
#     grad on ((4, 5), 2, (), 3), ((4, 5, 6), 1, (), 3) and ((20, 320, 15), 2, (), 8);
#   * any other sparse grad regathers values and dispatches floating types only:
#     RuntimeError: "_sparse_sum_backward_cuda" not implemented for 'Char'
#     on ((20, 320, 15), 2, (0,)), with 'Byte', 'Int', 'Long', 'Bool', 'BFloat16' and
#     both float8 types failing alike (see _needs_float_only).
# fp8 is therefore supported on the copy path: torch.sparse_coo_tensor accepts an fp8
# SparseCOO handed coordinates whose row-major keys ascend, with is_coalesced=True, and
# the copy path returns a result for it. The measured fp8
# limit is the coalesce kernel, which the operator reaches only for an already
# uncoalesced self:
#   RuntimeError: "coalesce_sparse_cuda" not implemented for 'Float8_e4m3fn'
# fp8 values come from tu.make_input(dtype, ...) directly: torch.testing.make_tensor
# resolves the range symbols against the dtype it is handed, so a ("min", "0") range is
# the fp8 interval (float8_e4m3fn: -448..0) and no fp32 extrema are reinterpreted after
# the fact. fp8 keeps every dtype and every required range here.
_FP8_DTYPES = []
if _SUPPORTS_FP8:
    _FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]

# Differentiability, probed on this backend: this operator registers no derivative.
# out.requires_grad follows grad.requires_grad, but
#   torch.autograd.grad(out, grad)
#   RuntimeError: derivative for aten::_sparse_sum_backward is not implemented
# A backward op is not exempt from that, so no higher-order workload exists for the
# native op and none is added; the forward result is the whole contract here.
_COPY_PATH_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.float16,
    torch.int32,
    torch.bool,
]
if _SUPPORTS_BF16:
    _COPY_PATH_DTYPES.append(torch.bfloat16)
if _SUPPORTS_INT64:
    _COPY_PATH_DTYPES.append(torch.int64)
if _SUPPORTS_FP64:
    _COPY_PATH_DTYPES.append(torch.float64)
_COPY_PATH_DTYPES.extend(_FP8_DTYPES)

_FLOAT_DTYPES = [torch.float32, torch.float16]
if _SUPPORTS_FP64:
    _FLOAT_DTYPES.append(torch.float64)

# ATen reports an out-of-range dim as IndexError, an unsupported layout or grad form
# as RuntimeError (NotImplementedError derives from it) and a wrong argument type as
# TypeError.
_INVALID_INPUT_ERRORS = (IndexError, RuntimeError, TypeError)

# Copy-form descriptors (shape, sparse_dim, dims, nnz) on tiny shapes plus prescribed
# large ones. dims either covers every sparse dim (dense grad) or keeps every sparse
# dim; the operator gathers an index set and has no broadcast semantics, so the spec's
# broadcast dimension has no workload here, and dim has no schema default.
_COPY_ROWS = [
    ((), 0, (), 1),
    ((1,), 1, (0,), 1),
    ((4, 5), 2, (0, 1), 3),
    ((4, 5), 2, (1, 0), 3),
    ((4, 5), 2, (-1, -2), 3),
    ((4, 5, 6), 1, (), 3),
    ((4, 5, 6), 2, (), 3),
    ((1024, 1024), 1, (0,), 64),
    ((1024, 1024), 1, (1,), 1024),
    ((1024, 1024), 1, (-2,), 64),
    ((16, 128, 64, 60), 2, (2, 3), 16),
    ((16, 7, 57, 32, 29), 3, (3, 4), 16),
    ((16, 7, 57, 32, 29), 3, (-1, -2), 16),
]

# Quick smoke subset on the spec's quick shape, so a smoke run still spans the full
# supported dtype set; the unparametrized negatives below always run.
_QUICK_ROWS = [
    ((2, 19, 7), 2, (0, 1), 4),
]

# Prescribed required shapes (256 / 1024x1024 / 3-dim / 4-dim / 5-dim) on the float
# family, which every native branch accepts: dense grads, a full dense tail with 1024
# stored coordinates, and summed dense dims.
_SHAPE_FLOAT_ROWS = [
    ((256,), 1, (0,), 32),
    ((1024, 1024), 1, (), 1024),
    ((20, 320, 15), 2, (), 8),
    ((20, 320, 15), 2, (0,), 16),
    ((16, 128, 64, 60), 2, (), 16),
    ((16, 128, 64, 60), 2, (2, 3), 16),
    ((16, 7, 57, 32, 29), 3, (), 16),
    ((16, 7, 57, 32, 29), 3, (0, 1, 2), 16),
]

# Dim forms on tiny shapes: negative dims, the empty dim list (identity gradient),
# summed dense dims and dim order.
_DIM_ROWS = [
    ((4, 5), 2, (), 3),
    ((4, 5), 2, (0,), 3),
    ((4, 5), 2, (1,), 3),
    ((4, 5), 2, (-1,), 3),
    ((4, 5), 2, (-2,), 3),
    ((4, 5, 6), 3, (), 3),
    ((4, 5, 6), 3, (0, 1), 3),
    ((4, 5, 6), 2, (2,), 3),
    ((4, 5, 6), 2, (0, 2), 3),
    ((4, 5, 6), 1, (0,), 3),
    ((4, 5, 6), 1, (1,), 3),
]

# The same dim forms on the prescribed 1024x1024 / 4-dim / 5-dim shapes: negative,
# reordered and reduced dense axes, including cases where summing dense dims removes
# several tail axes at once and cases that sum a sparse dim next to dense ones.
_DIM_LARGE_ROWS = [
    ((1024, 1024), 1, (0,), 64),
    ((1024, 1024), 1, (-2,), 64),
    ((16, 128, 64, 60), 2, (1,), 16),
    ((16, 128, 64, 60), 2, (0, 2), 16),
    ((16, 128, 64, 60), 2, (1, 3), 16),
    ((16, 7, 57, 32, 29), 3, (2, 4), 16),
    ((16, 7, 57, 32, 29), 3, (-1, -5), 16),
    ((16, 7, 57, 32, 29), 3, (1,), 16),
    ((16, 7, 57, 32, 29), 3, (4, 3, 2), 16),
]

# Zero NNZ and zero-size sparse axes. A zero-size sparse axis can only hold zero
# coordinates, so those rows request nnz == 0 and the fixture keeps the coordinate set
# empty instead of generating out-of-range indices. ((4, 0), 2, 3) with nnz > 0 was
# measured to be rejected at construction (size is inconsistent with indices).
_ZERO_ROWS = [
    ((8,), 1, (0,), 0),
    ((4, 0), 2, (), 0),
    ((0, 5), 2, (), 0),
]

# Non-contiguous dense grad and non-contiguous dense tail, which the native kernel
# reads as strided input.
_NONCONTIG_ROWS = [
    ((4, 5, 6), 1, (0,), 3),
    ((4, 5, 6), 1, (), 3),
    ((4, 5, 6, 7), 1, (), 3),
]

# Stored offsets: the stored values are a suffix of a larger buffer, so the fixture
# hands over a sliced values tensor instead of a front-aligned one.
_OFFSET_ROWS = [
    ((4, 5, 6), 1, (), 3),
    ((4, 5, 6), 2, (0, 1), 3),
]

# out= overload: a self-shaped sparse buffer is accepted and returned.
_OUT_ROWS = [
    ((4, 5, 6), 1, (), 3),
    ((4, 5, 6), 1, (1,), 3),
    ((4, 5, 6), 2, (), 3),
    ((4, 5, 6), 2, (0, 1), 3),
]

# Uncoalesced self whose coordinates are unique: the coalesced flag is unset but no
# duplicate merging happens on either side.
_UNCOALESCED_ROWS = [
    ((4, 5, 6), 1, (), 3),
    ((4, 5, 6), 2, (1,), 3),
]

# Uncoalesced self with real duplicates. The ring is emitted twice, so 2 * nnz
# pre-coalesce coordinates collapse onto nnz distinct coordinates and the stored values
# become sums of the two colliding draws; the pre-coalesce count is preserved.
_DUPLICATE_ROWS = [
    ((4, 5, 6), 1, (), 3),
    ((16, 128, 64, 60), 2, (1,), 4),
]

# The nan / inf / mixed payload is the grad value vector itself, so both rows keep
# nnz == len(payload) == 5 and no dense tail.
_SPECIAL_ROWS = [
    ((32, 64), 2, (1,), 5),
    ((64, 32), 2, (0,), 5),
]
_SPECIAL_CASES = tu.special_value_cases(_FLOAT_DTYPES)

# FP8 specials. tu.special_value_cases is the one place that knows which specials a
# floating format can represent: it yields a 'nan' case for every floating dtype and
# adds 'inf' / 'mixed' only where the format has an infinity. float8_e4m3fn has none
# (measured: float('inf') cast to float8_e4m3fn is nan, so the format's inf payload
# collapses to nan), float8_e5m2 keeps both. No infinity is claimed for e4m3fn. The
# rows below are copy-form (dims=() keeps every sparse dim), the only branch this
# backend supports for fp8.
_FP8_SPECIAL_ROWS = [
    ((20, 320, 15), 2, (), 8),
    ((32, 64), 2, (), 5),
]
_FP8_SPECIAL_CASES = tu.special_value_cases(_FP8_DTYPES)

# Regather descriptor for the dtype-rejection grid. dims=(0,) keeps sparse axis 1 only,
# so the grad regathers; the projection onto that one axis holds 8 distinct, ascending
# positions (nnz 8 under the axis extent 320), so the negative fixtures are built
# through the constructor path and need no coalesce kernel - fp8 included - while the
# operator still takes its float-only regather branch.
_REGATHER_ROWS = [
    ((20, 320, 15), 2, (0,), 8),
]

# Regather dispatch. The fixture is built for every dtype listed here and the very same
# descriptor returns a result for float16, float32 and float64 (probed), so a failure
# raised below comes from the operator and not from fixture construction. Which dtypes
# that branch dispatches is this backend's own measurement, so the list is only built on
# the NVIDIA vendor and the assertion never binds to the wording this backend happens to
# use. int64 / bfloat16 / fp8 entries are gated on the static capability flags, so a
# device without the dtype never builds the fixture.
_REGATHER_REJECTED_DTYPES = []
if flag_gems.vendor_name == "nvidia":
    _REGATHER_REJECTED_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.bool]
    if _SUPPORTS_INT64:
        _REGATHER_REJECTED_DTYPES.append(torch.int64)
    if _SUPPORTS_BF16:
        _REGATHER_REJECTED_DTYPES.append(torch.bfloat16)
    if _SUPPORTS_FP8:
        _REGATHER_REJECTED_DTYPES.extend(_FP8_DTYPES)


def _summed_dims(shape, dims):
    rank = len(shape)
    return {dim % rank for dim in dims} if rank else set()


def _grad_shape(shape, dims):
    summed = _summed_dims(shape, dims)
    return tuple(shape[d] for d in range(len(shape)) if d not in summed)


def _kept_sparse_dims(shape, sparse_dim, dims):
    summed = _summed_dims(shape, dims)
    return [d for d in range(sparse_dim) if d not in summed]


def _needs_float_only(shape, sparse_dim, dims):
    """True when the native kernel needs a float dtype (dispatch note above)."""
    kept = _kept_sparse_dims(shape, sparse_dim, dims)
    return bool(kept) and len(kept) != sparse_dim


def _coordinate_period(shape, sparse_dim):
    period = 1
    for size in shape[:sparse_dim]:
        period = math.lcm(period, max(size, 1))
    return period


def _projection_period(shape, kept):
    """Collision period of a ring projection onto the kept axes.

    Two ring positions land on the same kept coordinate when they differ by a multiple
    of this, so a family that keeps every sparse axis only stores as many distinct
    coordinates as its own period. This is a uniqueness fact only; sortedness is a
    separate property and is checked by _projected_sorted_unique.
    """
    period = 1
    for dim in kept:
        period = math.lcm(period, max(shape[dim], 1))
    return period


def _projected_sorted_unique(shape, kept, nnz):
    """Whether a ring projection onto the kept axes is strictly ascending, hence unique.

    Plain Python from the descriptor: the row-major key of ring position p over the kept
    axes is sum((p % shape[d]) * stride_d) with the same strides the sparse tensor itself
    uses, so the keys are re-derived here rather than read back from a tensor. The family
    takes the constructor's is_coalesced=True path only when that key rises with p, which
    is the order _ring_permutation sorts the ring into. Keeping every sparse axis satisfies
    it by construction - sorting the ring by exactly these keys leaves the kept projection
    in that order - while a proper subset can reorder the projection and fail, as on
    ((16, 7, 57, 32, 29), 3, (-1, -5), 16), where the projection onto axes 1 and 2 steps
    back at position 7. Those families are coalesced instead, which is the float-only
    branch anyway.
    """
    if nnz <= 1:
        return True
    if nnz > _projection_period(shape, kept):
        return False
    strides = []
    stride = 1
    for dim in reversed(kept):
        strides.append(stride)
        stride *= max(shape[dim], 1)
    strides.reverse()
    previous = None
    for position in range(nnz):
        key = 0
        for dim, step in zip(kept, strides):
            key += (position % max(shape[dim], 1)) * step
        if previous is not None and key <= previous:
            return False
        previous = key
    return True


def _validate_dims(shape, dims):
    """Plain Python: dims address real axes and no axis is named twice.

    Range is checked first and the repeat check runs on the normalized axes, because
    [0, -rank] names axis 0 twice. Measured on rank 2: [0, -2] and [1, -1] both raise
    RuntimeError: dim N appears multiple times in the list of dims, while reordered
    and negative forms such as [1, 0] and [-1, -2] stay legal.
    """
    rank = len(shape)
    for dim in dims:
        if rank == 0 or dim < -rank or dim >= rank:
            raise ValueError(f"dim {dim} out of range for rank {rank} in {shape}")
    normalized = [dim % rank for dim in dims] if rank else []
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"dim {list(dims)} names an axis twice in {shape}")


def _validate(shape, sparse_dim, nnz):
    """Reject a descriptor the ring fixture would otherwise build unsafely.

    Plain Python: a zero-size sparse axis holds no coordinates and more coordinates
    than the ring period would repeat them, so both are refused here instead of
    silently changing the requested workload.
    """
    if not 0 <= sparse_dim <= len(shape):
        raise ValueError(f"sparse_dim {sparse_dim} out of range for rank {len(shape)}")
    if nnz < 0:
        raise ValueError(f"nnz must be non-negative, got {nnz}")
    if nnz and any(size == 0 for size in shape[:sparse_dim]):
        raise ValueError(f"a zero-size sparse axis cannot hold {nnz} coordinates")
    period = _coordinate_period(shape, sparse_dim)
    if nnz > period:
        raise ValueError(
            f"{shape} with sparse_dim={sparse_dim} holds only {period} unique "
            f"ring coordinates, not {nnz}"
        )


def _sort_keys(indices, size):
    """Row-major linear keys of the given coordinates inside size."""
    keys = torch.zeros(indices.shape[1], dtype=torch.long, device=indices.device)
    stride = 1
    for dim in range(indices.shape[0] - 1, -1, -1):
        keys = keys + indices[dim] * stride
        stride *= size[dim]
    return keys


def _ring_permutation(shape, sparse_dim, nnz):
    """Ring coordinates f % shape[d] for f = 0..nnz-1, in lexicographic order.

    Offsets repeat with period lcm(shape[:sparse_dim]), so nnz <= period yields unique
    coordinates; the returned order maps ring position to sorted position, which makes
    the whole sparse axis set a strictly ascending unique column family.
    """
    device = flag_gems.device
    if sparse_dim == 0 or nnz == 0:
        return torch.empty((sparse_dim, nnz), dtype=torch.long, device=device), None
    flat = torch.arange(nnz, dtype=torch.long, device=device)
    raw = torch.stack([flat % shape[d] for d in range(sparse_dim)])
    order = torch.argsort(_sort_keys(raw, shape))
    return raw[:, order], order


def _sparse_coo(indices, values, size, sorted_unique):
    """SparseCOO built through the path the coordinate family already guarantees.

    sorted_unique is a generation-time fact about the family handed in, computed from
    the descriptor by _projected_sorted_unique and validated over the fixed families by
    the generation-time probe - never read back from a tensor. The is_coalesced=True flag
    states that the columns ascend; it does not verify them, so the projection has to be
    right before the call. The ring satisfies it when it keeps every sparse axis and a
    projection can break it. A family the check rejects is coalesced instead; coalescing
    is the only sort/merge step either side runs, and this backend has no fp8 coalesce
    kernel, so the dtype-wide grids keep to families the check accepts (guarded below).
    """
    if sorted_unique:
        return torch.sparse_coo_tensor(
            indices, values, size, device=flag_gems.device, is_coalesced=True
        )
    return torch.sparse_coo_tensor(
        indices, values, size, device=flag_gems.device
    ).coalesce()


def _make_values(dtype, shape, value_range):
    """Values for the requested dtype, straight from the shared range helper.

    tu.make_input resolves the range symbols against the dtype it is given, so an fp8
    workload gets the fp8 interval and no fp32 extrema are reinterpreted afterwards.
    """
    return tu.make_input(dtype, shape, value_range)


def _strided(tensor):
    """The same logical values with a non-contiguous last-axis stride.

    The logical shape is kept: the native op sizes the grad from the reduced shape and
    rejects a transposed one (measured on ((4, 5, 6), 1, (0,), 3),
    RuntimeError: The expanded size of the tensor (6) must match the existing size (5)
    at non-singleton dimension 2). Only the strides change, by viewing every other
    entry of a zero-padded buffer.
    """
    if tensor.dim() < 1 or tensor.shape[-1] == 0:
        return tensor
    padded = tensor.new_zeros(tensor.shape[:-1] + (tensor.shape[-1] * 2,))
    out = padded[..., ::2]
    out.copy_(tensor)
    return out


def _special_values(payload, shape):
    """The shared special payload cycled to a required (count,) + tail shape.

    The payload is the one tests/test_utils.py builds for this dtype and scenario, already
    on the target device and already cast to that dtype, so it is exactly the nan / inf /
    mixed pattern the format can represent and is used as handed over: no cast is applied
    here, and in particular none that would rebuild the pattern through fp32 or move it
    back to the host. Only the positions repeat: the cycle is a device-side index select
    of the payload itself, so the pattern is kept without a host round-trip and without
    resolving the bounds again as fp32. Every dtype these rows run, float8_e4m3fn and
    float8_e5m2 included, has a native index_select kernel on this target (probed), so no
    dtype-specific path is needed.
    """
    count = 1
    for size in shape:
        count *= size
    flat = payload.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("empty special payload")
    index = torch.arange(count, device=flat.device) % flat.numel()
    return flat.index_select(0, index).reshape(shape)


def _sparse_self(
    shape,
    sparse_dim,
    dtype,
    value_range,
    nnz,
    coalesced=True,
    duplicate=False,
    storage_offset=False,
):
    """SparseCOO self with nnz stored coordinates, coalesced where the coordinates allow.

    The ring coordinates are distinct while nnz <= period (_validate) and are emitted
    in ascending row-major order, so a plain self is handed over through the constructor
    path with is_coalesced=True; coalesced=False leaves that flag unset on the same
    unique coordinates; duplicate=True emits the ring twice, so the tensor carries
    2 * nnz pre-coalesce coordinates that collide pairwise and the caller's .coalesce()
    sums the two draws stored at each of the nnz distinct coordinates;
    storage_offset=True takes the stored values from the back half of a larger buffer,
    so the fixture hands over a sliced tensor.
    """
    _validate(shape, sparse_dim, nnz)
    tail = tuple(shape[sparse_dim:])
    indices, order = _ring_permutation(shape, sparse_dim, nnz)
    copies = 2 if duplicate else 1
    values = _make_values(dtype, (copies * nnz,) + tail, value_range)
    if order is not None:
        # The permutation is applied inside every copy, so the row order of each copy
        # matches the (already sorted) index columns it is stored with.
        parts = [values[i * nnz : (i + 1) * nnz][order] for i in range(copies)]
        values = torch.cat(parts, dim=0)
    if duplicate:
        indices = torch.cat([indices] * 2, dim=1)
    if storage_offset:
        count = values.shape[0]
        buffer = _make_values(dtype, (2 * count,) + tail, value_range)
        buffer[count:] = values
        values = buffer[count:]
    if duplicate or not coalesced:
        return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    return _sparse_coo(indices, values, shape, True)


def _make_grad(
    inp,
    shape,
    sparse_dim,
    dims,
    dtype,
    value_range,
    values=None,
    noncontig=False,
    storage_offset=False,
):
    """Upstream gradient for dims: dense iff every sparse dim is summed.

    A sparse grad keeps the retained sparse dims and carries one value per stored
    coordinate plus the unsummed dense dims as its tail. Its construction path follows
    from the descriptor: the projection of the ring onto the kept axes is handed to the
    constructor when _projected_sorted_unique accepts it - keeping every sparse axis
    always does - and is coalesced otherwise, since the projection can reorder as well as
    repeat the ring. noncontig gives the dense grad (or the sparse grad's dense tail) a
    non-contiguous last-axis stride; storage_offset takes the stored values from the back
    half of a larger buffer (the dense branch has no stored values to offset).
    """
    grad_shape = _grad_shape(shape, dims)
    kept = _kept_sparse_dims(shape, sparse_dim, dims)
    if not kept:
        grad = _make_values(dtype, grad_shape, value_range)
        return _strided(grad) if noncontig else grad
    indices = inp._indices()[kept]
    tail = grad_shape[len(kept) :]
    count = indices.shape[1]
    if values is None:
        if storage_offset:
            buffer = _make_values(dtype, (2 * count,) + tail, value_range)
            values = buffer[count:]
        else:
            values = _make_values(dtype, (count,) + tail, value_range)
    else:
        # The shared payload already carries this dtype and device.
        values = _special_values(values, (count,) + tail)
    if noncontig:
        values = _strided(values)
    return _sparse_coo(
        indices, values, grad_shape, _projected_sorted_unique(shape, kept, count)
    )


def _empty_out(inp, dtype):
    """Empty SparseCOO buffer for the out= overload.

    The native result has self's shape and sparse_dim, not the reduced grad shape, and
    .out resizes into the supplied buffer, so a grad-shaped or dense buffer raises
    NotImplementedError: Could not run 'aten::resize_' with arguments from the
    'SparseCUDA' backend. A SparseCOO buffer of inp.shape / inp.sparse_dim with
    matching dtype and device is accepted and returned as the same object.
    """
    indices = torch.empty((inp.sparse_dim(), 0), dtype=torch.long, device=inp.device)
    values = torch.empty(
        (0,) + tuple(inp.shape[inp.sparse_dim() :]), dtype=dtype, device=inp.device
    )
    return torch.sparse_coo_tensor(indices, values, inp.shape, device=inp.device)


# Descriptor guards, plain Python and tensor-free, so collection touches no device:
# every row names real axes without naming one twice, the grids that span every dtype in
# _COPY_PATH_DTYPES hold copy-form descriptors, and the regather grid really regathers.
# All of them build their sparse grads from families _projected_sorted_unique accepts,
# so no fp8 case ever needs the absent coalesce kernel.
for _row in (
    _COPY_ROWS
    + _QUICK_ROWS
    + _SHAPE_FLOAT_ROWS
    + _DIM_ROWS
    + _DIM_LARGE_ROWS
    + _ZERO_ROWS
    + _NONCONTIG_ROWS
    + _OFFSET_ROWS
    + _OUT_ROWS
    + _UNCOALESCED_ROWS
    + _DUPLICATE_ROWS
    + _SPECIAL_ROWS
    + _FP8_SPECIAL_ROWS
    + _REGATHER_ROWS
):
    _validate_dims(_row[0], _row[2])

for _row in _COPY_ROWS + _QUICK_ROWS + _ZERO_ROWS:
    _shape, _sparse_dim, _dims, _nnz = _row
    if _needs_float_only(_shape, _sparse_dim, _dims):
        raise ValueError(f"copy-form descriptor needs a float dtype: {_row!r}")
    _kept = _kept_sparse_dims(_shape, _sparse_dim, _dims)
    if _kept and not _projected_sorted_unique(_shape, _kept, _nnz):
        raise ValueError(f"copy-form descriptor {_row!r} would need a coalesce kernel")

for _row in _REGATHER_ROWS:
    _shape, _sparse_dim, _dims, _nnz = _row
    if not _needs_float_only(_shape, _sparse_dim, _dims):
        raise ValueError(f"regather descriptor {_row!r} does not regather")
    _kept = _kept_sparse_dims(_shape, _sparse_dim, _dims)
    if not _projected_sorted_unique(_shape, _kept, _nnz):
        raise ValueError(f"regather descriptor {_row!r} would need a coalesce kernel")


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_COPY_ROWS, quick=_QUICK_ROWS))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COPY_PATH_DTYPES)
def test__sparse_sum_backward(dtype, value_range, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, value_range, nnz)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, value_range)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    # Index remapping plus a value copy: no rounding is introduced.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_SHAPE_FLOAT_ROWS, quick=[]))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_shapes(dtype, value_range, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, value_range, nnz)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, value_range)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_DIM_ROWS, quick=[]))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_dim_forms(dtype, value_range, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, value_range, nnz)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, value_range)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_DIM_LARGE_ROWS, quick=[]))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_dim_forms_large(dtype, value_range, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, value_range, nnz)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, value_range)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    # The result layout is whatever the native operator defines - it is part of what is
    # compared here - so the candidate is checked against the native result rather than
    # against a hand-written assumption about the result's shape or sparse_dim.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_ZERO_ROWS, quick=[]))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COPY_PATH_DTYPES)
def test__sparse_sum_backward_zero_nnz_and_zero_axis(dtype, value_range, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, value_range, nnz)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, value_range)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    # Empty coordinates and zero-size axes stay a pure copy, so the check is exact.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_NONCONTIG_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_noncontiguous(dtype, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, ["-1", "1"], nnz)
    # The strided operand is the dense grad, or the dense tail of a sparse grad.
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, ["-1", "1"], noncontig=True)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_OFFSET_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_stored_offsets(dtype, row):
    shape, sparse_dim, dims, nnz = row
    # The fixture hands over a values tensor sliced out of a larger buffer; whether a COO
    # constructor later copies those values is framework behaviour, so nothing is
    # probed on the constructed tensor here.
    inp = _sparse_self(shape, sparse_dim, dtype, ["-1", "1"], nnz, storage_offset=True)
    grad = _make_grad(
        inp, shape, sparse_dim, dims, dtype, ["-1", "1"], storage_offset=True
    )

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_OUT_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_out(dtype, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, ["-1", "1"], nnz)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_buf = _empty_out(ref_inp, dtype)
    res_buf = _empty_out(inp, dtype)
    torch.ops.aten._sparse_sum_backward.out(ref_grad, ref_inp, dims, out=ref_buf)
    res_ret = flag_gems._sparse_sum_backward(grad, inp, dims, out=res_buf)

    # The candidate must return the supplied buffer, and that buffer must hold the
    # result; no reference-buffer identity is asserted.
    assert res_ret is res_buf
    tu.assert_result_equal(res_buf, ref_buf)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_UNCOALESCED_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_uncoalesced_self(dtype, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, ["-1", "1"], nnz, coalesced=False)
    grad = _make_grad(inp.coalesce(), shape, sparse_dim, dims, dtype, ["-1", "1"])

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    # Unique coordinates, so no merging arithmetic happens and the check is exact.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_DUPLICATE_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test__sparse_sum_backward_duplicate_coordinates(dtype, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(
        shape, sparse_dim, dtype, ["-1", "1"], nnz, coalesced=False, duplicate=True
    )
    # The fixture stores 2 * nnz pre-coalesce coordinates that collide pairwise; the
    # caller's .coalesce() leaves nnz distinct coordinates, and the coalesced self is
    # what the grad is projected onto.
    grad = _make_grad(inp.coalesce(), shape, sparse_dim, dims, dtype, ["-1", "1"])

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    # Coalescing adds the colliding values, so this path does carry merging
    # arithmetic: compared with the shared tolerance instead of the exact one, and
    # both sides merge the same duplicates.
    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_SPECIAL_ROWS, quick=[]))
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__sparse_sum_backward_special_values(dtype, scenario, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, ["-1", "1"], nnz)
    payload = tu.make_special_input(dtype, scenario)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, ["-1", "1"], values=payload)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    # Matching nan / inf positions are part of the reference semantics.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("row", tu.selected_cases(_FP8_SPECIAL_ROWS, quick=[]))
@pytest.mark.parametrize("dtype,scenario", _FP8_SPECIAL_CASES)
def test__sparse_sum_backward_fp8_special_values(dtype, scenario, row):
    shape, sparse_dim, dims, nnz = row
    inp = _sparse_self(shape, sparse_dim, dtype, ["-1", "1"], nnz)
    payload = tu.make_special_input(dtype, scenario)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, ["-1", "1"], values=payload)

    ref_out = torch.ops.aten._sparse_sum_backward(
        tu.to_reference(grad), tu.to_reference(inp), dims
    )
    res_out = flag_gems._sparse_sum_backward(grad, inp, dims)

    # A copy path, so the special pattern carries through unchanged; the scenario list
    # only contains specials this dtype can represent.
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._sparse_sum_backward
@pytest.mark.parametrize("dtype", _REGATHER_REJECTED_DTYPES)
def test__sparse_sum_backward_regather_rejects_non_float_dtypes(dtype):
    # dims=(0,) keeps sparse axis 1 only, so the grad regathers values. The two fixtures
    # above this call build successfully for every dtype in the list, and the identical
    # descriptor returns a result for float16 / float32 / float64, so what is asserted is
    # the operator's own dtype dispatch and not a fixture failure. The message is
    # deliberately not asserted: the wording is this backend's, not the contract. The
    # projected coordinates ascend, so the fixture is built without coalescing and fp8
    # needs no coalesce kernel to reach the operator.
    shape, sparse_dim, dims, nnz = _REGATHER_ROWS[0]
    inp = _sparse_self(shape, sparse_dim, dtype, ["-1", "1"], nnz)
    grad = _make_grad(inp, shape, sparse_dim, dims, dtype, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_sum_backward(grad, inp, dims)


@pytest.mark._sparse_sum_backward
def test__sparse_sum_backward_negative_dense_self():
    # self must be SparseCOO; a dense self has no native kernel.
    inp = torch.randn((4, 5), device=flag_gems.device)
    grad = tu.make_input(torch.float32, (5,), ["-1", "1"])
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems._sparse_sum_backward(grad, inp, [0])


@pytest.mark._sparse_sum_backward
def test__sparse_sum_backward_negative_out_of_range_dim():
    inp = _sparse_self((4, 5), 2, torch.float32, ["-1", "1"], 3)
    grad = _make_grad(inp, (4, 5), 2, (), torch.float32, ["-1", "1"])
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems._sparse_sum_backward(grad, inp, [5])


@pytest.mark._sparse_sum_backward
def test__sparse_sum_backward_negative_aliased_dim():
    # [0, -2] names axis 0 twice on rank 2, which the native op rejects (measured:
    # RuntimeError: dim 0 appears multiple times in the list of dims). The exception is
    # asserted, not that wording. A dense grad is handed over so the failure cannot come
    # from the grad form.
    inp = _sparse_self((4, 5), 2, torch.float32, ["-1", "1"], 3)
    grad = tu.make_input(torch.float32, _grad_shape((4, 5), [0, -2]), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._sparse_sum_backward(grad, inp, [0, -2])


@pytest.mark._sparse_sum_backward
def test__sparse_sum_backward_negative_sparse_grad_for_summed_dims():
    # Summing every sparse dim requires a dense grad, so the sparse grad built for the
    # empty dim list is rejected for dims=[0, 1].
    inp = _sparse_self((4, 5), 2, torch.float32, ["-1", "1"], 3)
    grad = _make_grad(inp, (4, 5), 2, (), torch.float32, ["-1", "1"])
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems._sparse_sum_backward(grad, inp, [0, 1])
