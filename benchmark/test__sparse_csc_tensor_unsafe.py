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

import flag_gems

from . import base, consts, utils

# _sparse_csc_tensor_unsafe takes three caller-supplied buffers plus `size`, so
# a workload is a (matrix size, nnz) descriptor: the builder derives the column
# pointers and row indices that store exactly the requested nnz.
_CSC_DEF_SHAPES = [
    ((1024, 1024), 4096),
    ((2048, 2048), 8192),
    ((4096, 4096), 8192),
    ((8192, 8192), 8192),
    ((1024, 2048), 4096),
    ((2048, 1024), 4096),
    ((128, 128, 512), 2048),
]

# A bare shape row denotes that logical size with the smaller of this many
# stored entries and the matrix capacity; a rank-0/rank-1 size has no
# compressed axis and keeps empty components.
_DEFAULT_NNZ = 4096


def _is_descriptor(entry):
    return (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[0], (tuple, list))
        and len(entry[0]) >= 2
    )


def _descriptor(entry):
    if not _is_descriptor(entry):
        raise ValueError(f"not a (matrix size, nnz) CSC descriptor: {entry!r}")
    matrix_shape = [int(dim) for dim in entry[0]]
    nnz = int(entry[1])
    if nnz < 0:
        raise ValueError(f"nnz must be non-negative: {entry!r}")
    return matrix_shape, nnz


def _entry_key(entry):
    # Hashable identity used to union shape rows without duplicates.
    if _is_descriptor(entry):
        matrix_shape, nnz = _descriptor(entry)
        return ("descriptor", tuple(matrix_shape), nnz)
    return ("shape", tuple(entry))


def _entry_plan(entry):
    if _is_descriptor(entry):
        return _descriptor(entry)
    extents = [int(dim) for dim in entry]
    if any(dim < 0 for dim in extents):
        raise ValueError(f"negative extent in shape {entry!r}")
    if len(extents) < 2:
        return extents, 0
    return extents, min(_DEFAULT_NNZ, extents[-2] * extents[-1])


def _hashable_entry(entry):
    if isinstance(entry, (tuple, list)):
        return tuple(_hashable_entry(item) for item in entry)
    return entry


def _csc_components(matrix_shape, entries, index_dtype, device):
    # Column pointers and row indices for exactly `entries` stored values: the
    # per-column counts come from the quotient and remainder of entries / cols,
    # so the requested nnz is preserved (no floor division, no max/min clamp).
    entries = int(entries)
    if entries < 0:
        raise ValueError(f"nnz must be non-negative, got {entries}")
    empty = torch.empty(0, dtype=index_dtype, device=device)
    if len(matrix_shape) < 2:
        # A rank-0/rank-1 `size` has no compressed axis; native accepts it with
        # empty components and keeps the requested size.
        return empty.clone(), empty.clone(), [0]
    batch = [int(dim) for dim in matrix_shape[:-2]]
    rows, cols = int(matrix_shape[-2]), int(matrix_shape[-1])
    if rows <= 0 or cols <= 0:
        if entries:
            raise ValueError(f"nnz={entries} cannot be stored in {matrix_shape}")
        return empty.clone(), empty.clone(), batch + [0]
    per_column, remainder = divmod(entries, cols)
    if per_column + (1 if remainder else 0) > rows:
        raise ValueError(f"nnz={entries} cannot be stored in {matrix_shape}")
    counts = torch.full((cols,), per_column, dtype=index_dtype, device=device)
    if remainder:
        counts[:remainder] += 1
    ccol = torch.zeros(cols + 1, dtype=index_dtype, device=device)
    ccol[1:] = torch.cumsum(counts, dim=0)
    row = torch.cat(
        (
            torch.arange(per_column + 1, dtype=index_dtype, device=device).repeat(
                remainder
            ),
            torch.arange(per_column, dtype=index_dtype, device=device).repeat(
                cols - remainder
            ),
        )
    )
    assert int(row.numel()) == entries
    nnz = int(row.numel())
    if batch:
        ccol = ccol.expand(*batch, cols + 1).contiguous()
        row = row.expand(*batch, nnz).contiguous()
    return ccol, row, batch + [nnz]


def _case_fn(shape, dtype):
    del dtype
    matrix_shape, entries = _entry_plan(shape)
    yield base.BenchmarkCasePlan(
        shape={"matrix": list(matrix_shape), "nnz": entries},
        params={"nnz": entries},
        builder_args=(matrix_shape, entries),
    )


def _build_inputs_fn(plan, dtype, device):
    matrix_shape, entries = plan.builder_args
    ccol, row, values_shape = _csc_components(
        matrix_shape, entries, torch.int64, device
    )
    values = utils.generate_tensor_input(values_shape, dtype, device)
    return (
        ccol,
        row,
        values,
        list(matrix_shape),
        {
            "dtype": dtype,
            "layout": torch.sparse_csc,
            "device": device,
        },
    )


class SparseCscTensorUnsafeBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Shared loader first: an operator-specific entry from the caller's
        # shape file, otherwise DEFAULT_SHAPES.
        super().set_shapes(shape_file_path)
        # Union the curated CSC descriptors so a caller file adds workloads
        # instead of replacing them and no shape row is dropped.
        merged = {}
        for entry in list(self.shapes) + list(_CSC_DEF_SHAPES):
            merged.setdefault(_entry_key(entry), entry)
        self.shapes = list(merged.values())

    def set_more_shapes(self):
        # The loader calls this after self.shapes is resolved but before it
        # merges through dict.fromkeys, so nested descriptor rows are made
        # hashable here; the extras stay CSC-valid instead of dense shapes.
        self.shapes = [_hashable_entry(entry) for entry in self.shapes]
        return list(super().set_more_shapes()) + list(_CSC_DEF_SHAPES)


@pytest.mark.sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe():
    bench = SparseCscTensorUnsafeBenchmark(
        op_name="_sparse_csc_tensor_unsafe",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_csc_tensor_unsafe,
        gems_op=getattr(flag_gems, "_sparse_csc_tensor_unsafe", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
