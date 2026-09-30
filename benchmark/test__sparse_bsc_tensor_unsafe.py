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

"""Benchmark for ``aten::_sparse_bsc_tensor_unsafe``.

A shape entry is the sparse tensor ``size`` exactly as the shared shape files
provide it; the ``((nrows, ncols), block, k)`` form additionally asks for a
non-trivial BSC block and ``k`` stored blocks per column block, because a block
is not part of ``size``. ``torch_op`` and ``gems_op`` share the same call
semantics ``op(ccol_indices, row_indices, values, size, *, dtype, layout,
device)``.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# BSC descriptors the shared default shapes cannot express, including non-1x1
# block forms; they are unioned with whatever the shape file provides.
BSC_SHAPES = [
    (256, 256),
    (1024, 1024),
    (20, 320),
    ((1024, 1024), (4, 4), 1),
    ((20, 320), (2, 5), 2),
]

# The stored payload covers floating and integer values.
BSC_DTYPES = consts.FLOAT_DTYPES + [torch.int64]


def _descriptor(entry):
    """Read one entry as ``(size, block, stored blocks per column block)``.

    Shared shape-file entries are plain ``size`` tuples and use 1x1 blocks; the
    explicit ``((nrows, ncols), block, k)`` form carries a larger BSC block.
    """
    if entry and isinstance(entry[0], (tuple, list)):
        size, block, per_column_block = entry
        return list(size), (int(block[0]), int(block[1])), int(per_column_block)
    return list(entry), (1, 1), 1


def _structure(size, block, per_column_block, device):
    """Column-block pointers and block-row indices matching ``size``."""
    block_rows, block_cols = block
    nrows = size[0] if size else 1
    ncols = size[1] if len(size) >= 2 else 0
    # A rank < 2 ``size`` has no column extent, so one column block is emitted
    # while the requested size is still stored verbatim by the factory.
    counts = torch.full(
        (max(ncols // block_cols, 1),), per_column_block, dtype=torch.int64
    )
    offsets = counts.cumsum(0)
    nnz = int(counts.sum())
    ccol = torch.cat([torch.zeros(1, dtype=torch.int64), offsets])
    rows = (torch.arange(nnz) - (offsets - counts).repeat_interleave(counts)) % max(
        nrows // block_rows, 1
    )
    return ccol.to(device), rows.to(device), nnz


def _payload(shape, dtype, device):
    """Construct stored values directly for each declared benchmark dtype."""
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(-4, 5, shape, dtype=dtype, device=device)


def _case_fn(shape, dtype):
    del dtype
    size, block, per_column_block = _descriptor(shape)
    yield base.BenchmarkCasePlan(
        shape={"size": list(size)},
        params={
            "block": [block[0], block[1]],
            "entries_per_column_block": per_column_block,
        },
        builder_args=(size, block, per_column_block),
    )


def _build_inputs_fn(plan, dtype, device):
    size, block, per_column_block = plan.builder_args
    ccol, rows, nnz = _structure(size, block, per_column_block, device)
    values = _payload((nnz,) + tuple(block) + tuple(size[2:]), dtype, device)
    # Flat positional arguments plus the trailing kwargs dict, which is how the
    # shared runner unpacks a builder result.
    return (
        ccol,
        rows,
        values,
        list(size),
        {"dtype": dtype, "layout": torch.sparse_bsc, "device": device},
    )


class SparseBscTensorUnsafeBenchmark(base.GenericBenchmark):
    """Two-phase generic benchmark over BSC descriptors."""

    def set_more_shapes(self):
        # Keep the extra shapes the generic family contributes, then add the
        # block-forming BSC descriptors the shared defaults lack.
        return list(super().set_more_shapes()) + [tuple(entry) for entry in BSC_SHAPES]

    def set_shapes(self, shape_file_path=None):
        # Keep the shared resolution (shape file or DEFAULT_SHAPES) and union the
        # BSC descriptors on top, so no valid requested shape is dropped.
        super().set_shapes(shape_file_path=shape_file_path)
        merged = list(self.shapes)
        seen = {repr(entry) for entry in merged}
        for entry in BSC_SHAPES:
            if repr(entry) not in seen:
                merged.append(entry)
                seen.add(repr(entry))
        self.shapes = merged


@pytest.mark.sparse_bsc_tensor_unsafe
def test__sparse_bsc_tensor_unsafe():
    bench = SparseBscTensorUnsafeBenchmark(
        op_name="_sparse_bsc_tensor_unsafe",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_bsc_tensor_unsafe,
        gems_op=getattr(flag_gems, "_sparse_bsc_tensor_unsafe", None),
        dtypes=BSC_DTYPES,
    )
    bench.run()
