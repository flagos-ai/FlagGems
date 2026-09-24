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

"""Benchmark for aten::_sparse_log_softmax on sparse COO input.

The candidate is timed against the native sparse kernel with identical sparse
COO descriptors.  Both the sparse pooling path (``dim < sparse_dim``) and the
dense block path (``dim >= sparse_dim``) are represented, including the dense
tail axis of the 4-dim shape.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# pytest blocks attribute access for underscore-prefixed marker names.
setattr(
    pytest.mark,
    "_sparse_log_softmax",
    MarkDecorator(Mark("_sparse_log_softmax", (), {}, _ispytest=True), _ispytest=True),
)

# float64 rows follow the static capability flag exposed by the runtime
# detector; no parsed device name and no probe at import time.
SPARSE_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)

# Stored-entry cap: large enough to be representative, small enough to stay a
# per-shape nnz that the sparse extent can always supply exactly once.
_NNZ_CAP = 8192

SPARSE_SHAPES = [(1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]

# Per shape: one sparse pooling axis (dim < sparse_dim) and the dense block axes
# (dim >= sparse_dim).  For (16, 128, 64, 60) with sparse_dim 3 the dense tail is
# axis 3, so (3, 3) is timed alongside the two pooled axes.
_PAIRS = {
    (1024, 1024): [(1, 0), (1, 1)],
    (20, 320, 15): [(2, 0), (2, 2)],
    (16, 128, 64, 60): [(3, 0), (3, 2), (3, 3)],
    (16, 7, 57, 32, 29): [(4, 0), (4, 4)],
}


def _sparse_extent(shape, sparse_dim):
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


def _coo_indices(shape, sparse_dim, nnz, group_axis, device):
    """Unique sorted COO coordinates of shape (sparse_dim, nnz) on ``device``.

    Built in O(nnz) with ``torch.arange``: entries are emitted in groups along
    ``group_axis`` so the pooled axis carries ties, then the row-major rank is
    sorted, which is exactly the lexicographic order coalescing requires, so
    ``is_coalesced=True`` stays truthful.  Every allocation uses the device the
    caller supplies alongside the values, never a hardcoded one.
    """
    if nnz == 0:
        return torch.zeros((sparse_dim, 0), dtype=torch.long, device=device)
    if sparse_dim == 0:
        return torch.zeros((0, nnz), dtype=torch.long, device=device)
    sizes = [int(size) for size in shape[:sparse_dim]]
    axis = sparse_dim - 1 if group_axis is None else group_axis
    others = [index for index in range(sparse_dim) if index != axis]
    other_extent = 1
    for index in others:
        other_extent *= sizes[index]
    group = min(sizes[axis], max(2, -(-nnz // other_extent)))
    order = torch.arange(nnz, device=device, dtype=torch.long)
    base = order // group
    coords = [None] * sparse_dim
    coords[axis] = order % group
    for index in reversed(others):
        coords[index] = base % sizes[index]
        base = base // sizes[index]
    strides = _strides(sizes)
    rank = torch.zeros(nnz, device=device, dtype=torch.long)
    for index in range(sparse_dim):
        rank = rank + coords[index] * strides[index]
    rank = rank.sort().values
    return torch.stack(
        [(rank // strides[index]) % sizes[index] for index in range(sparse_dim)]
    ).contiguous()


def _sparse_input(shape, sparse_dim, dtype, device, dim):
    """Sparse COO input on the benchmark's configured device."""
    stored = _nnz(shape, sparse_dim)
    indices = _coo_indices(
        shape, sparse_dim, stored, _reduced_axis(shape, sparse_dim, dim), device
    )
    values = torch.randn(
        (stored,) + tuple(shape[sparse_dim:]), dtype=dtype, device=device
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        shape,
        device=device,
        dtype=dtype,
        is_coalesced=True,
        check_invariants=True,
    )


def _specs_for(shape):
    """(sparse_dim, dim) pairs to time for ``shape``.

    A rank the native operator cannot represent is a planning error rather than
    a silently empty case list: rank-0 sparse COO raises IndexError for every
    dim (the kernel maps the dense axis to index 1 of a rank-1 value block), so
    shape () is reported and never dropped.  Every other rank uses sparse_dim
    rank-1 with valid dims.
    """
    if len(shape) == 0:
        raise ValueError(
            "_sparse_log_softmax requires rank >= 1: a rank-0 sparse COO tensor "
            "has no valid dim, so shape () cannot be benchmarked"
        )
    if shape in _PAIRS:
        return _PAIRS[shape]
    rank = len(shape)
    sparse_dim = max(1, rank - 1)
    dims = [0, rank - 1] if rank > 1 else [0]
    return [(sparse_dim, dim) for dim in dims]


def _case_fn(shape, dtype):
    del dtype
    for sparse_dim, dim in _specs_for(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape), "sparse_dim": sparse_dim},
            params={"dim": dim, "half_to_float": False},
            builder_args=(shape, sparse_dim, dim),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, sparse_dim, dim = plan.builder_args
    inp = _sparse_input(shape, sparse_dim, dtype, device, dim)
    # half_to_float must travel as a keyword: a positional False would bind the
    # ScalarType parameter of the .int overload.  dim stays positional, matching
    # the candidate's public call signature.
    return inp, dim, {"half_to_float": False}


class SparseLogSoftmaxBenchmark(OperatorBenchmark):
    """Timing driver whose shapes come from the generated plans."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SPARSE_SHAPES)


@pytest.mark._sparse_log_softmax
def test__sparse_log_softmax():
    bench = SparseLogSoftmaxBenchmark(
        op_name="_sparse_log_softmax",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_log_softmax.default,
        gems_op=getattr(flag_gems, "_sparse_log_softmax", None),
        dtypes=SPARSE_DTYPES,
    )
    bench.run()
