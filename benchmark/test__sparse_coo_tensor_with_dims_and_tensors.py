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
from .generated_operator_utils import OperatorBenchmark

# One descriptor row per workload: (size, sparse_dim, dense_dim, nnz). The
# constructor takes no tensor-shape argument, so a row carries the whole call and
# the extents of the COO tensor it builds. A row of plain extents is accepted as
# well and becomes the COO `size` with every axis sparse; the sparse domain is
# never materialized, so large extents stay cheap to time.
_BENCH_CASES = [
    ((256,), 1, 0, 64),
    ((1024, 1024), 2, 0, 1024),
    ((4096, 4096), 2, 0, 8192),
    ((65536, 65536), 2, 0, 32768),
    ((20, 320, 15), 3, 0, 320),
    ((16, 128, 64, 60), 2, 2, 16),
    ((16, 7, 57, 32, 29), 5, 0, 128),
]

# An extents-only row carries no nnz field; it stores this many entries. Duplicate
# coordinates are legal, so the count is independent of the shape's element count.
_DEFAULT_NNZ = 64


def _row_key(row):
    """Hashable form of one row, used to keep configured rows and defaults unique."""
    if isinstance(row, int) and not isinstance(row, bool):
        row = (row,)
    if not isinstance(row, (list, tuple)):
        raise ValueError(f"a sparse COO row must be a sequence, got {row!r}")
    return tuple(
        tuple(item) if isinstance(item, (list, tuple)) else item for item in row
    )


def _case_descriptor(row):
    """Normalize one row into (size, sparse_dim, dense_dim, nnz)."""
    row = _row_key(row)
    if len(row) == 4 and isinstance(row[0], (list, tuple)):
        size, sparse_dim, dense_dim, nnz = row
        size = tuple(int(extent) for extent in size)
        sparse_dim, dense_dim, nnz = int(sparse_dim), int(dense_dim), int(nnz)
        if sparse_dim < 0 or dense_dim < 0 or nnz < 0:
            raise ValueError(f"sparse_dim/dense_dim/nnz must be >= 0, got {row!r}")
        if sparse_dim + dense_dim != len(size):
            raise ValueError(
                f"sparse_dim + dense_dim must equal len(size), got {row!r}"
            )
        if any(extent < 0 for extent in size):
            raise ValueError(f"size extents must be >= 0, got {row!r}")
        if nnz > 0 and any(extent == 0 for extent in size[:sparse_dim]):
            raise ValueError(f"nnz={nnz} needs a non-empty sparse domain, got {row!r}")
        return size, sparse_dim, dense_dim, nnz

    size = tuple(int(extent) for extent in row)
    if any(extent < 0 for extent in size):
        raise ValueError(f"size extents must be >= 0, got {row!r}")
    if not size or 0 in size:
        return size, len(size), 0, 0
    return size, len(size), 0, _DEFAULT_NNZ


def _case_fn(shape, dtype):
    del dtype
    size, sparse_dim, dense_dim, nnz = _case_descriptor(shape)
    yield base.BenchmarkCasePlan(
        shape={"size": list(size)},
        params={"sparse_dim": sparse_dim, "dense_dim": dense_dim, "nnz": nnz},
        builder_args=(size, sparse_dim, dense_dim, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    size, sparse_dim, dense_dim, nnz = plan.builder_args
    if nnz == 0 or sparse_dim == 0:
        indices = torch.zeros((sparse_dim, nnz), dtype=torch.int64, device=device)
    else:
        gen = torch.Generator(device="cpu").manual_seed(0)
        indices = torch.stack(
            [
                torch.randint(0, extent, (nnz,), generator=gen, dtype=torch.int64)
                for extent in size[:sparse_dim]
            ]
        ).to(device)
    values = utils.generate_tensor_input(
        (nnz,) + tuple(size[sparse_dim:]), dtype, device
    )
    return (
        sparse_dim,
        dense_dim,
        list(size),
        indices,
        values,
        {"layout": torch.sparse_coo, "device": device, "dtype": dtype},
    )


class SparseCooTensorWithDimsAndTensorsBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Configured rows and the descriptors above are unioned instead of one
        # replacing the other, so no requested extent is dropped.
        super().set_shapes(shape_file_path)
        seen, merged = set(), []
        for row in list(self.shapes) + _BENCH_CASES:
            key = _row_key(row)
            if key not in seen:
                seen.add(key)
                merged.append(row)
        self.shapes = merged


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test__sparse_coo_tensor_with_dims_and_tensors():
    bench = SparseCooTensorWithDimsAndTensorsBenchmark(
        op_name="_sparse_coo_tensor_with_dims_and_tensors",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors,
        gems_op=getattr(flag_gems, "_sparse_coo_tensor_with_dims_and_tensors", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
