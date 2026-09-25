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

"""Benchmark for aten::_sparse_sum_backward.

A descriptor is (shape, sparse_dim, dim, nnz). The self handed to the operator is a
SparseCOO tensor of that shape and sparse_dim whose stored coordinates are drawn per
sparse axis with one seeded torch.randint per axis (generator manual_seed(_SEED)), then
resolved by the real .coalesce() kernel.

nnz is a REQUESTED PRE-COALESCE count: random coordinates repeat, so the support actually
stored after coalescing holds fewer than nnz coordinates and its size is not known while
the case list is written. The metadata publishes it as nnz_requested and claims no actual
stored count.

The grad follows the same random construction: dense when dim covers every sparse dim,
otherwise sparse, carrying the projected self coordinates - .coalesce() merges the
projected duplicates - plus every dense dim dim did not sum. The grad's stored support
after coalescing is likewise smaller than its requested pre-coalesce count and is not
published as an actual figure.

A descriptor tagged grad_layout='strided' or values_offset=True asks for a stored-values
layout that .coalesce() would otherwise discard, because coalescing merges duplicates and
rewrites the values. The layout is therefore re-applied to the merged values, and that is
the tensor the operator receives.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

setattr(
    pytest.mark,
    "_sparse_sum_backward",
    MarkDecorator(Mark("_sparse_sum_backward", (), {}, _ispytest=True), _ispytest=True),
)

_SUPPORTS_BF16 = flag_gems.runtime.device.support_bf16
_SUPPORTS_FP64 = flag_gems.runtime.device.support_fp64

_BENCH_DTYPES = [
    dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16 or _SUPPORTS_BF16
]
if _SUPPORTS_FP64:
    _BENCH_DTYPES.append(torch.float64)

# The regather branch of the native kernel dispatches half and floating types only
# (..._FLOATING_AND_COMPLEX_TYPES_AND1(kHalf)), so a descriptor that takes that branch
# is skipped for the remaining dtypes in _case_fn - after descriptor validation.
# bfloat16 is not in the dispatched set (measured: the same regather fixture returns a
# result for float16 / float32 / float64 and raises for bfloat16).
_FLOAT_ONLY_DTYPES = (torch.float16, torch.float32, torch.float64)

# Workload descriptors (shape, sparse_dim, dim, nnz), with the prescribed 1024x1024 /
# 4-dim / 5-dim shapes, their dense-tail / dense-grad / dense-dim forms, and the smaller
# and 1M-draw variants of the same families. nnz is the requested pre-coalesce count and
# the values tensor is (nnz, dense-tail): 1048576 x 1024 for ((1024, 1024), 1, (1,),
# 1048576), which is the size requested here rather than a reduced stand-in.
_BENCH_ROWS = [
    ((), 0, (), 1),
    ((256,), 1, (), 65536),
    ((1024, 1024), 1, (1,), 1048576),
    ((1024, 1024), 1, (0,), 1048576),
    ((20, 320, 15), 2, (0, 1), 262144),
    ((16, 128, 64, 60), 2, (0, 1), 2048),
    ((16, 7, 57, 32, 29), 3, (0, 1, 2), 2048),
    ((256,), 1, (0,), 65536),
    ((1024, 1024), 1, (), 1024),
    ((20, 320, 15), 2, (), 8),
    ((20, 320, 15), 2, (0, 1), 8),
    ((20, 320, 15), 2, (0,), 1048576),
    ((16, 128, 64, 60), 2, (), 16),
    ((16, 128, 64, 60), 2, (2, 3), 16),
    ((16, 7, 57, 32, 29), 3, (), 16),
]

# Zero NNZ and zero-size sparse axes (a zero-size sparse axis can only hold zero
# coordinates, so nnz stays 0 instead of generating out-of-range indices).
_BENCH_ZERO_ROWS = [
    ((8,), 1, (0,), 0),
    ((4, 0), 2, (), 0),
    ((0, 5), 2, (), 0),
]

# Non-contiguous dense grad and non-contiguous dense tail. The logical shape is kept
# (the native op sizes the grad from the reduced shape and rejects a transposed one)
# and only the last-axis stride changes.
_BENCH_NONCONTIG_ROWS = [
    ((4, 5, 6), 1, (0,), 3),
    ((4, 5, 6), 1, (), 3),
    ((4, 5, 6, 7), 1, (), 3),
]

# Stored offsets: the stored values are a suffix of a larger buffer.
_BENCH_OFFSET_ROWS = [
    ((4, 5, 6), 1, (), 3),
    ((4, 5, 6), 2, (0, 1), 3),
]


def _unique_rows(rows):
    """Rows in first-seen order with exact duplicates kept once.

    One descriptor in two layout tables is one workload, not two: the layout flags are
    read from set membership of the same key, so a duplicate would emit two identical
    combined offset+strided plans. Only exact duplicates are dropped, and both source
    tables and their frozensets stay as they are, so the surviving row still carries both
    flags. Every unique shape / sparse_dim / dim / nnz descriptor is kept, the 1M-draw
    rows included.
    """
    seen = set()
    unique = []
    for row in rows:
        shape, sparse_dim, dims, nnz = row
        key = (tuple(shape), sparse_dim, tuple(dims), nnz)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


_ALL_ROWS = _unique_rows(
    _BENCH_ROWS + _BENCH_ZERO_ROWS + _BENCH_NONCONTIG_ROWS + _BENCH_OFFSET_ROWS
)
_NONCONTIG_SET = frozenset(_BENCH_NONCONTIG_ROWS)
_OFFSET_SET = frozenset(_BENCH_OFFSET_ROWS)

_SEED = 20260924


def _summed_dims(shape, dims):
    rank = len(shape)
    return {dim % rank for dim in dims} if rank else set()


def _grad_shape(shape, dims):
    summed = _summed_dims(shape, dims)
    return tuple(shape[d] for d in range(len(shape)) if d not in summed)


def _kept_sparse_dims(shape, sparse_dim, dims):
    summed = _summed_dims(shape, dims)
    return [d for d in range(sparse_dim) if d not in summed]


def _needs_float_dispatch(shape, sparse_dim, dims):
    """True when the native kernel regathers values and needs a float dtype."""
    kept = _kept_sparse_dims(shape, sparse_dim, dims)
    return bool(kept) and len(kept) != sparse_dim


def _validate_descriptor(row):
    """Checked in plain Python, before any dtype filter and before any tensor exists.

    Extents are non-negative integers, sparse_dim addresses the sparse prefix, dim is
    interpreted literally (no modulo mapping of an out-of-range axis) and without
    repeats after normalization, and nnz is a non-negative integer that a zero-size
    sparse axis cannot hold. These are the native shape and parameter rules, not a
    fixture budget, so no descriptor is shrunk or dropped here.
    """
    if not isinstance(row, (tuple, list)) or len(row) != 4:
        raise ValueError(f"descriptor {row!r} must be (shape, sparse_dim, dim, nnz)")
    shape, sparse_dim, dims, nnz = row
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"shape in {row!r} must be a tuple of extents")
    shape = tuple(shape)
    for size in shape:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"extent {size!r} in {row!r} must be a non-negative int")
    if isinstance(sparse_dim, bool) or not isinstance(sparse_dim, int):
        raise ValueError(f"sparse_dim in {row!r} must be an int")
    if not 0 <= sparse_dim <= len(shape):
        raise ValueError(f"sparse_dim {sparse_dim} out of range for rank {len(shape)}")
    if not isinstance(dims, (tuple, list)) or not all(
        not isinstance(dim, bool) and isinstance(dim, int) for dim in dims
    ):
        raise ValueError(f"dim in {row!r} must be a tuple of ints")
    rank = len(shape)
    for dim in dims:
        if rank == 0 or dim < -rank or dim >= rank:
            raise ValueError(f"dim {dim} out of range for rank {rank} in {row!r}")
    # Range first, then normalize: [0, -rank] names axis 0 twice, and the native op
    # rejects the aliased pair (measured on rank 2, RuntimeError: dim N appears
    # multiple times in the list of dims), while reordered negative dims stay legal.
    normalized = [dim % rank for dim in dims] if rank else []
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"dim {list(dims)} in {row!r} repeats an axis")
    if isinstance(nnz, bool) or not isinstance(nnz, int) or nnz < 0:
        raise ValueError(f"nnz in {row!r} must be a non-negative int")
    if nnz and any(size == 0 for size in shape[:sparse_dim]):
        raise ValueError(
            f"{row!r}: a zero-size sparse axis cannot hold {nnz} coordinates"
        )
    return shape, sparse_dim, tuple(dims), nnz


def _strided(tensor):
    """The same logical values with a non-contiguous last-axis stride.

    The logical shape is kept - the native op sizes the grad from the reduced shape -
    and only the strides change, by viewing every other entry of a zero-padded buffer.
    """
    if tensor.dim() < 1 or tensor.shape[-1] == 0:
        return tensor
    padded = tensor.new_zeros(tensor.shape[:-1] + (tensor.shape[-1] * 2,))
    out = padded[..., ::2]
    out.copy_(tensor)
    return out


def _with_layout(source, storage_offset=False, strided=False):
    """The stored values with the requested layout, taken from one enclosing buffer.

    storage_offset leaves the logical extent alone and starts the handed-over tensor
    further inside a larger buffer; strided leaves it alone too and views every other
    entry of a buffer twice as long. Both compose in the same buffer, so a descriptor that
    asks for both hands over a strided view that also starts inside a larger buffer
    instead of the second layout replacing the first. A rank-1 value tensor has no second
    axis, so both layouts act on its only axis there: the one axis carries both the offset
    and the doubled span, so the buffer is sized from the span both flags consume and the
    view is taken from the offset by that span's step. The logical count and values are
    unchanged in every combination.
    """
    if source.dim() < 1:
        return source
    if source.dim() == 1:
        count = source.shape[0]
        span = count * (2 if strided else 1)
        start = span if storage_offset else 0
        buffer = source.new_empty((start + span,))
        view = buffer[start::2] if strided else buffer[start:]
        view.copy_(source)
        return view
    rows = source.shape[0] * (2 if storage_offset else 1)
    last = source.shape[-1] * (2 if strided else 1)
    buffer = source.new_empty((rows,) + tuple(source.shape[1:-1]) + (last,))
    view = buffer[source.shape[0] :] if storage_offset else buffer
    if strided:
        view = view[..., ::2]
    view.copy_(source)
    return view


def _values_tensor(shape, dtype, device, storage_offset=False, strided=False):
    # Every dtype this benchmark runs (float16 / bfloat16 / float32 / float64) has a
    # native randn kernel on this target, so the draw is made in the requested dtype.
    values = torch.randn(shape, dtype=dtype, device=device)
    if storage_offset or strided:
        values = _with_layout(values, storage_offset=storage_offset, strided=strided)
    return values


def _relayout(sparse, storage_offset=False, strided=False):
    """The coalesced values with the requested layout re-applied after .coalesce().

    .coalesce() merges duplicate coordinates and rewrites the values, so a layout applied
    before it would not survive onto the tensor handed to the operator. The merged values
    are re-laid-out in one buffer and handed back together with the coalesced indices.
    is_coalesced=True is accepted for such a values view, and the operator reads the
    layout from it (both probed).
    """
    values = _with_layout(
        sparse._values(), storage_offset=storage_offset, strided=strided
    )
    return torch.sparse_coo_tensor(
        sparse._indices(),
        values,
        sparse.shape,
        device=sparse.device,
        is_coalesced=True,
    )


def _indices(shape, sparse_dim, nnz, device):
    """One seeded torch.randint per sparse axis, so the coordinates are reproducible."""
    if sparse_dim == 0:
        # A rank-0 SparseCOO tensor stores no coordinates at all.
        return torch.empty((0, nnz), dtype=torch.long, device=device)
    generator = torch.Generator("cpu").manual_seed(_SEED)
    return torch.stack(
        [
            torch.randint(
                0, max(shape[d], 1), (nnz,), dtype=torch.long, generator=generator
            )
            for d in range(sparse_dim)
        ]
    ).to(device)


def _sparse_self(shape, sparse_dim, dtype, device, nnz, storage_offset=False):
    """SparseCOO self for a descriptor, built by coalescing nnz random coordinates.

    Random coordinates repeat, so the support stored afterwards is smaller than nnz; a
    requested storage_offset is re-applied to the merged values afterwards.
    """
    indices = _indices(shape, sparse_dim, nnz, device)
    values = _values_tensor(
        (nnz,) + tuple(shape[sparse_dim:]),
        dtype,
        device,
    )
    self_ = torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()
    if storage_offset:
        self_ = _relayout(self_, storage_offset=True)
    return self_


def _case_fn(row, dtype):
    if isinstance(row, dict):
        row = row.get("input", row)
    shape, sparse_dim, dims, nnz = _validate_descriptor(row)
    key = (shape, sparse_dim, tuple(dims), nnz)
    kept = _kept_sparse_dims(shape, sparse_dim, dims)
    # Descriptor facts only: nnz is the requested pre-coalesce coordinate count, and the
    # support really stored after coalescing random coordinates is smaller and not known
    # while the case list is written, so no actual stored count is published here.
    params = {
        "sparse_dim": sparse_dim,
        "dim": list(dims),
        "nnz_requested": nnz,
        "grad": "sparse" if kept else "dense",
        "dense_tail": list(_grad_shape(shape, dims)[len(kept) :]),
        "grad_layout": "strided" if key in _NONCONTIG_SET else "contiguous",
        "values_offset": key in _OFFSET_SET,
    }
    # Validation above runs for every dtype, so an invalid custom descriptor is never
    # quietly skipped by this filter.
    if (
        _needs_float_dispatch(shape, sparse_dim, dims)
        and dtype not in _FLOAT_ONLY_DTYPES
    ):
        return
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params=params,
        builder_args=(shape, sparse_dim, tuple(dims), nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    """Builds (grad, self, dim) for one descriptor; the harness unpacks them in order.

    The strided / offset flags are derived from the same descriptor key the case list
    carries, so listing and execution cannot disagree.
    """
    shape, sparse_dim, dims, nnz = plan.builder_args
    shape = tuple(shape)
    dims = tuple(dims)
    key = (shape, sparse_dim, dims, nnz)
    offset = key in _OFFSET_SET
    strided = key in _NONCONTIG_SET
    inp = _sparse_self(shape, sparse_dim, dtype, device, nnz, storage_offset=offset)
    grad_shape = _grad_shape(shape, dims)
    kept = _kept_sparse_dims(shape, sparse_dim, dims)
    if not kept:
        # A dense grad: the native op reads the reduced shape, and only the last-axis
        # stride can change without breaking it.
        grad = torch.randn(grad_shape, dtype=dtype, device=device)
        if strided:
            grad = _strided(grad)
    else:
        # A sparse grad keeps the retained sparse dims and every dense tail dim; the
        # projected coordinates may repeat, so coalescing is done explicitly and a
        # requested layout is re-applied to the merged values afterwards.
        indices = inp._indices()[kept]
        tail = grad_shape[len(kept) :]
        values = _values_tensor(
            (indices.shape[1],) + tail,
            dtype,
            device,
        )
        grad = torch.sparse_coo_tensor(
            indices, values, grad_shape, device=device
        ).coalesce()
        if offset or strided:
            grad = _relayout(grad, storage_offset=offset, strided=strided)
    # Flat sequence: unpack_to_args_kwargs maps non-dict items to positional args and
    # only a dict item to kwargs, so dim is the third positional argument exactly as in
    # the reference schema.
    return grad, inp, list(dims)


class SparseSumBackwardBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with SparseCOO workload descriptors.

    The default shapes come through the shared OperatorBenchmark hook, so a custom
    shape file still overrides them (and a missing or malformed shape file still
    raises) exactly as for other operators. The base set_more_shapes already returns
    [], so no override is needed.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_ALL_ROWS)


@pytest.mark._sparse_sum_backward
def test__sparse_sum_backward():
    bench = SparseSumBackwardBenchmark(
        op_name="_sparse_sum_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_sum_backward,
        gems_op=getattr(flag_gems, "_sparse_sum_backward", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
