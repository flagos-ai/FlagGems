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

"""Benchmark for ``aten::_sparse_csr_tensor_unsafe``.

The factory stores the three component tensors and copies nothing, so the measured
work is the native call plus building the components (component allocation happens
in the builder, outside the timing window). Each case is a native-valid ``(size,
nnz)`` descriptor whose row pointers, in-range column indices and values are built
in genuine CSR form; the shared shape file may contribute bare size tuples, which
are valid CSR sizes and take one stored entry per row.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts

OP_NAME = "_sparse_csr_tensor_unsafe"

# (size, nnz): the sparse-CSR benchmark corpus with the stored-entry totals of the
# sibling operator benchmark. `nnz` is the total across the batch and is split
# evenly over the rows of every batch entry.
DEFAULT_CASES = [
    ((512, 512), 65536),
    ((1024, 1024), 262144),
    ((2048, 2048), 1048576),
    ((4096, 4096), 2097152),
    ((64, 512, 512), 65536),
    ((16, 1024, 1024), 262144),
]

# Merged by the shared API at the comprehensive level only.
MORE_CASES = [
    ((8192, 8192), 4194304),
    ((32, 1024, 1024), 524288),
]


def _descriptor(row):
    """Normalize a benchmark shape row into ``(size, nnz)``.

    This file's rows carry an explicit ``(size, nnz)`` pair; the shared shape file
    (and the base-class fallback) contributes bare dense shape tuples, which are
    valid CSR sizes. ``nnz`` is ``None`` when the row does not ask for a count.
    """
    row = tuple(row)
    if row and isinstance(row[0], (tuple, list)):
        return tuple(int(dim) for dim in row[0]), int(row[1])
    return tuple(int(dim) for dim in row), None


def _extents(size, nnz):
    """Return ``(rows, batch, total_nnz, nnz_per_batch)`` for one descriptor."""
    rows = size[-2] if len(size) >= 2 else 0
    batch = tuple(size[:-2])
    batch_size = math.prod(batch) if batch else 1
    if nnz is None:
        # One stored entry per row of every batch entry.
        nnz = batch_size * rows
    if not batch_size:
        if nnz:
            raise ValueError(f"{size!r} has an empty batch but asks for {nnz} entries")
        return rows, batch, 0, 0
    if nnz % batch_size or (rows == 0 and nnz):
        raise ValueError(f"{nnz} stored entries do not fit the row block of {size!r}")
    return rows, batch, nnz, nnz // batch_size


def _case_fn(shape, dtype):
    del dtype
    size, nnz = _descriptor(shape)
    rows, _batch, total, per_batch = _extents(size, nnz)
    yield base.BenchmarkCasePlan(
        shape={
            "crow_indices": [rows + 1],
            "col_indices": [per_batch],
            "values": [per_batch],
        },
        params={"size": list(size), "nnz": total},
        builder_args=(size, total),
    )


def _build_inputs_fn(plan, dtype, device):
    size = tuple(plan.builder_args[0])
    rows, batch, _total, per_batch = _extents(size, int(plan.builder_args[1]))
    cols = size[-1] if size else 0

    crow_indices = torch.zeros(rows + 1, dtype=torch.int32)
    if rows:
        # Front-loaded row counts: the first `remainder` rows hold one extra entry.
        counts = torch.full((rows,), per_batch // rows, dtype=torch.int32)
        remainder = per_batch % rows
        if remainder:
            counts[:remainder] += 1
        crow_indices[1:] = torch.cumsum(counts, 0)
    col_indices = torch.arange(per_batch, dtype=torch.int32) % max(cols, 1)
    values = torch.rand(per_batch, dtype=torch.float32)
    if batch:
        crow_indices = crow_indices.expand(batch + (rows + 1,)).contiguous()
        col_indices = col_indices.expand(batch + (per_batch,)).contiguous()
        values = values.expand(batch + (per_batch,)).contiguous()

    # Flat positional arguments plus one trailing kwargs dict: the shared iterator
    # passes each Tensor and the int list positionally.
    return (
        crow_indices.to(device),
        col_indices.to(device),
        values.to(dtype).to(device),
        list(size),
        {"dtype": dtype, "device": device},
    )


class SparseCsrTensorUnsafeBenchmark(base.GenericBenchmark):
    DEFAULT_METRICS = ["latency"]

    def set_shapes(self, shape_file_path=None):
        # Shared API first, so a configured shape file keeps its own rows.
        super().set_shapes(shape_file_path)
        # Union, so neither the file's descriptors nor this operator's are dropped.
        self.shapes = list(
            dict.fromkeys(self.shapes + [tuple(row) for row in DEFAULT_CASES])
        )

    def set_more_shapes(self):
        return list(super().set_more_shapes()) + [tuple(row) for row in MORE_CASES]


@pytest.mark.sparse_csr_tensor_unsafe
def test__sparse_csr_tensor_unsafe():
    bench = SparseCsrTensorUnsafeBenchmark(
        op_name=OP_NAME,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_csr_tensor_unsafe,
        gems_op=getattr(flag_gems, OP_NAME, None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
