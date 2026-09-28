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

"""Benchmark for ``torch.ops.aten._sparse_broadcast_to``.

``torch_op`` is the timing reference and ``gems_op`` the candidate; both are
called as ``op(operand, size)`` on the same COO operand, so the comparison has
identical call semantics.

A workload descriptor is ``(sparse_shape, sparse_dim, nnz, size)``: the shapes
and the entry count are tensor-free, JSON-compatible case metadata, and the same
descriptor is kept in ``builder_args`` for execution. One descriptor list drives
both ``--list-cases`` and normal runs, so a ``--case-id`` replay benchmarks
exactly what a full run would.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# bfloat16 is the only entry of consts.FLOAT_DTYPES whose availability is a
# static capability flag rather than a given.
_BENCH_DTYPE_GATES = {torch.bfloat16: flag_gems.runtime.device.support_bf16}
BENCH_DTYPES = [
    dtype for dtype in consts.FLOAT_DTYPES if _BENCH_DTYPE_GATES.get(dtype, True)
]

# (sparse shape, sparse dim, nnz, target size).
#   * the first nine are the performance descriptors: an identity target, new
#     leading sparse dims of several ranks, a large entry count, and the zero
#     cases (zero nnz, zero sparse extent, rank-zero operand);
#   * the next four are supported singleton-source broadcasts, where a size-1
#     sparse dim or a size-1 dense dim expands instead of a new leading dim
#     being prepended;
#   * the last one is a zero *dense* tail: stored entries with an empty payload.
# nnz may exceed the cell count of the sparse part: duplicate coordinates are a
# legitimate COO state and, together with the unique strictly increasing rows,
# keep the operand uncoalesced.
_BENCH_CASES = [
    ((1024, 1024), 2, 65536, (1024, 1024)),
    ((1024, 1024), 2, 65536, (16, 1024, 1024)),
    ((4096, 4096), 2, 262144, (2, 4096, 4096)),
    ((2048, 2048), 2, 262144, (8, 2048, 2048)),
    ((256, 256, 256), 3, 262144, (4, 256, 256, 256)),
    ((1024, 1024), 2, 1048576, (1024, 1024)),
    ((1024, 1024), 2, 0, (1024, 1024)),
    ((1024, 0), 2, 0, (1024, 0)),
    ((), 0, 1, ()),
    ((1, 64), 1, 128, (256, 64)),
    ((1024, 1), 1, 16384, (1024, 128)),
    ((1, 4), 1, 128, (3, 4)),
    ((20, 1, 15), 1, 3, (20, 320, 15)),
    ((1024, 0), 1, 4096, (1024, 0)),
]


def _validate_descriptor(descriptor):
    """Validate a ``(sparse_shape, sparse_dim, nnz, size)`` descriptor.

    Runs for listing and for execution alike and performs no tensor work.
    Extents are used as given (no int coercion); only combinations the native
    operator cannot accept are rejected.
    """
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 4:
        raise TypeError(
            "a benchmark case must be (sparse_shape, sparse_dim, nnz, size), "
            f"got {descriptor!r}"
        )
    sparse_shape, sparse_dim, nnz, size = descriptor
    for name, value in (("sparse_dim", sparse_dim), ("nnz", nnz)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a plain int, got {value!r}")
    sparse_shape = tuple(sparse_shape)
    size = tuple(size)
    for extent in sparse_shape + size:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError(f"extents must be plain ints, got {extent!r}")
        if extent < 0:
            raise ValueError(f"extents must be non-negative, got {extent!r}")
    if sparse_dim < 0 or sparse_dim > len(sparse_shape):
        raise ValueError(f"sparse_dim {sparse_dim} is invalid for {sparse_shape}")
    if nnz < 0:
        raise ValueError(f"nnz {nnz} must not be negative")
    if len(size) < len(sparse_shape):
        raise ValueError(f"target {size} must not drop dims of {sparse_shape}")
    prefix = len(size) - len(sparse_shape)
    # A zero *leading* target extent is an unresolved native hazard: the earlier
    # probe of this operator aborted the process on this backend. Such a
    # descriptor is never built and never handed to the operator. Zero extents
    # elsewhere are valid and stay in the workload list.
    if any(extent == 0 for extent in size[:prefix]):
        raise ValueError(f"target {size} has a zero leading extent")
    # Real broadcast compatibility: a source dim is kept as is or expanded from
    # a size-1 dim (both sparse and dense dims may be singletons).
    for source_extent, target_extent in zip(sparse_shape, size[prefix:]):
        if source_extent != target_extent and source_extent != 1:
            raise ValueError(
                f"target {size} cannot broadcast source {sparse_shape}: "
                f"{source_extent} -> {target_extent}"
            )
    if nnz:
        cells = 1
        for extent in sparse_shape[:sparse_dim]:
            cells *= extent
        if cells == 0:
            raise ValueError(
                f"nnz {nnz} needs a non-zero sparse extent, got "
                f"{sparse_shape[:sparse_dim]}"
            )
    return sparse_shape, sparse_dim, nnz, size


def _sparse_operand(sparse_shape, sparse_dim, nnz, dtype, device):
    """Build the COO operand of a descriptor on ``device``."""
    extents = sparse_shape[:sparse_dim]
    if nnz and extents:
        cells = 1
        for extent in extents:
            cells *= extent
        # Coordinates repeat when nnz exceeds the cell count, which keeps the
        # operand uncoalesced exactly like a real broadcast workload.
        flat = torch.arange(nnz, dtype=torch.long, device=device) % cells
        rows = []
        for extent in reversed(extents):
            rows.append(flat % extent)
            flat = flat // extent
        indices = torch.stack(list(reversed(rows)))
    else:
        indices = torch.zeros((sparse_dim, nnz), dtype=torch.long, device=device)
    values = utils.generate_tensor_input(
        (nnz,) + sparse_shape[sparse_dim:], dtype, device
    )
    return torch.sparse_coo_tensor(indices, values, sparse_shape)


def _case_fn(shape, dtype):
    # Two-phase GenericBenchmark: one BenchmarkCasePlan per descriptor, with the
    # tensor construction deferred to _build_inputs_fn.
    del dtype
    sparse_shape, sparse_dim, nnz, size = _validate_descriptor(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape, "target": size},
        params={"sparse_dim": sparse_dim, "nnz": nnz},
        builder_args=(sparse_shape, sparse_dim, nnz, size),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, sparse_dim, nnz, size = _validate_descriptor(plan.builder_args)
    inp = _sparse_operand(sparse_shape, sparse_dim, nnz, dtype, device)
    return inp, {"size": list(size)}


class SparseBroadcastToBenchmark(OperatorBenchmark):
    # core_shapes.yaml holds dense shapes, which cannot describe a COO operand,
    # so these descriptors are the defaults; a caller-supplied shape file still
    # takes precedence through the base lookup.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.sparse_broadcast_to
def test__sparse_broadcast_to():
    bench = SparseBroadcastToBenchmark(
        op_name="_sparse_broadcast_to",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_broadcast_to,
        gems_op=getattr(flag_gems, "_sparse_broadcast_to", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
