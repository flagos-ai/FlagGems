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

"""Benchmark suite for ``torch.ops.aten._to_sparse_csc``.

Every workload is a ``(dense shape, dense_dim)`` descriptor; a bare shape uses
the schema default. Descriptors are validated before anything is built, so
``--list-cases`` allocates no tensor and runs no operator, and ``--case-id``
replay uses exactly the plans of normal execution.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Sentinel for a workload that calls the operator without ``dense_dim``. It is a
# plain string so a serialized shape file (``[[2, 3], "omitted"]``) validates
# like the in-process value; comparisons use equality, never identity, and the
# omitted form stays distinct from an explicit ``None``.
_OMITTED = "omitted"

# Static backend capability flags, read once from the runtime device
# description; the benchmark never probes dtype support itself.
_CAPABILITY = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}


def _supported(dtype):
    return _CAPABILITY.get(dtype, True)


# (dense shape, dense_dim) descriptors. All of them are served at both levels,
# so listing and replay use the same plans; the rank-5 shape reaches the largest
# valid dense_dim (rank - 2 == 3) and the rank-4 shape exercises a wide dense
# payload with dense_dim 2.
DEFAULT_SHAPES = [
    ((1024, 1024), _OMITTED),
    ((20, 320, 15), 1),
    ((16, 128, 64, 60), 0),
    ((16, 7, 57, 32, 29), 2),
    ((1024, 1024), None),
    ((20, 320, 15), 0),
    ((16, 128, 64, 60), 1),
    ((16, 128, 64, 60), 2),
    ((16, 7, 57, 32, 29), 3),
]


def _descriptor(entry):
    # Canonical entry is (shape, dense_dim); a bare shape sequence uses the
    # operator default (omitted dense_dim), which keeps custom shape files
    # usable without rewriting them. The serialized sentinel string is accepted
    # as-is and only ever compared with equality.
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[0], (tuple, list))
    ):
        return tuple(entry[0]), entry[1]
    if isinstance(entry, (tuple, list)):
        return tuple(entry), _OMITTED
    return entry, _OMITTED


def _validate_workload(shape, dense_dim):
    # Validate metadata before any tensor is built or listed: real integer
    # (non-bool, non-negative) dimensions, rank >= 2 and a canonical optional
    # dense_dim inside [0, rank - 2].
    if not isinstance(shape, (tuple, list)) or len(shape) < 2:
        raise ValueError(f"_to_sparse_csc needs a rank >= 2 dense shape, got {shape!r}")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"invalid dense dimension {dim!r} in shape {shape!r}")
    if dense_dim is None or dense_dim == _OMITTED:
        return
    if isinstance(dense_dim, bool) or not isinstance(dense_dim, int):
        raise ValueError(f"dense_dim must be None or an int, got {dense_dim!r}")
    if not 0 <= dense_dim <= len(shape) - 2:
        raise ValueError(f"dense_dim {dense_dim} out of range for shape {shape!r}")


def _axes(shape, dense_dim):
    dd = 0 if dense_dim is None or dense_dim == _OMITTED else dense_dim
    return (
        tuple(shape[: len(shape) - dd - 2]),
        shape[len(shape) - dd - 2],
        shape[len(shape) - dd - 1],
        tuple(shape[len(shape) - dd :]),
    )


def _case_fn(shape, dtype):
    # Phase one: metadata only, every tensor is deferred to the builder.
    del dtype
    dense_shape, dense_dim = _descriptor(shape)
    _validate_workload(dense_shape, dense_dim)
    params = {} if dense_dim == _OMITTED else {"dense_dim": dense_dim}
    yield base.BenchmarkCasePlan(
        shape={"input": [int(dim) for dim in dense_shape]},
        params=params,
        builder_args=(tuple(int(dim) for dim in dense_shape), dense_dim),
    )


def _benchmark_input(shape, dense_dim, dtype, device):
    """Dense input whose batch slices share one nnz but not one zero pattern.

    Rank >= 3 needs the same number of specified elements in every batch slice.
    Each slice therefore clears exactly one row and two columns - so the counts,
    and hence nnz, match - but at slice-specific coordinates (the row and the
    first column shift by the slice index), so the column pointers and row
    indices differ between slices. Nonzero blocks then take a bounded,
    non-constant offset (1..4) per dense-payload component, so a kernel that
    permutes the payload axes cannot pass by broadcasting one stored element.
    Offsets are never zero, so the ramp changes stored values only and leaves
    the support (and the workload geometry) exactly as built.
    """
    batch, rows, cols, payload = _axes(shape, dense_dim)
    nb = math.prod(batch) if batch else 1
    payload_size = math.prod(payload) if payload else 1

    inp = torch.ones(shape, dtype=dtype, device=device)
    if rows and cols:
        blocks = inp.reshape(nb, rows, cols, *payload)
        for batch_index in range(nb):
            blocks[batch_index, batch_index % rows, :] = 0
            blocks[batch_index, :, batch_index % cols] = 0
            blocks[batch_index, :, (batch_index + cols // 2) % cols] = 0
            blocks[batch_index, batch_index % rows, batch_index % cols] = (
                batch_index + 2
            )
    if payload:
        tail = torch.arange(payload_size, dtype=torch.int32, device=device)
        tail = (tail % 4 + 1).to(dtype).reshape(payload)
        inp = torch.where(inp == 0, inp, inp + tail)
    return inp


def _build_inputs_fn(plan, dtype, device):
    dense_shape, dense_dim = plan.builder_args
    _validate_workload(dense_shape, dense_dim)
    inp = _benchmark_input(dense_shape, dense_dim, dtype, device)
    # _OMITTED means the schema default call, so the keyword is not passed at
    # all; an explicit None is forwarded as None and stays distinct.
    return inp, dict(plan.params)


class ToSparseCscBenchmark(OperatorBenchmark):
    # A sparse layout conversion has no core_shapes.yaml entry, so the curated
    # descriptors above are the defaults. A caller-supplied shape file still
    # goes through the shared loader, which honors its operator or class entry;
    # no extra shape expansion and no local size cap are layered on top.
    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        if shape_file_path is None:
            # The shared loader opens the shape file directly, so it needs a
            # path; fall back to the curated descriptors when none is given.
            self.shapes = list(default_shapes or DEFAULT_SHAPES)
            return
        super().set_shapes(
            shape_file_path, default_shapes=default_shapes or DEFAULT_SHAPES
        )


@pytest.mark.to_sparse_csc
def test__to_sparse_csc():
    bench = ToSparseCscBenchmark(
        op_name="_to_sparse_csc",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_sparse_csc,
        gems_op=getattr(flag_gems, "_to_sparse_csc", None),
        dtypes=[dtype for dtype in consts.FLOAT_DTYPES if _supported(dtype)],
    )
    bench.run()
