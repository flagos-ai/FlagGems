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

"""Benchmark for the sparse ``values`` accessor.

``aten::values`` is sparse-only, so the dense ``core_shapes.yaml`` shapes are
realized as sparse tensors of the same declared size: a sparse size is
independent of the stored nnz, so no dense allocation is needed and the rank-0
shape becomes a scalar COO holding one stored value. All custom sparse rows are
kept as well. COO inputs are built coalesced from sorted, unique indices, which
the operator requires; the compressed layouts CSR/CSC/BSR/BSC cover their
batched and hybrid forms.

Case listing is metadata-only: ``case_fn`` allocates nothing, and the input
builder is the only place that touches the device.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_LAYOUTS = {
    "csr": torch.sparse_csr_tensor,
    "csc": torch.sparse_csc_tensor,
    "bsr": torch.sparse_bsr_tensor,
    "bsc": torch.sparse_bsc_tensor,
}


_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# The operator is a dtype-agnostic storage relocation, so every supported dtype
# family is timed: float, complex, int and bool.
_BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.COMPLEX_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
    if _dtype_supported(dtype)
]


_INDEX_DTYPE = torch.int64 if flag_gems.runtime.device.support_int64 else torch.int32

# Shared logical shapes realized as sparse tensors of the same declared size. The
# stored counts are fixed, not caps: a sparse size does not reserve nnz slots.
_SHARED_NNZ = 65536
_SHARED_ROWS = [
    ("coo", (1024**3,), (), 4096),
    ("coo", (64, 64), (), 4096),
    ("coo", (4096, 4096), (), _SHARED_NNZ),
    ("coo", (64, 512, 512), (), _SHARED_NNZ),
    ("coo", (1024, 1024, 1024), (2,), _SHARED_NNZ),
    ("coo", (), (), 1),
    ("csr", (64, 512, 512), 32768, None, 1),
    ("bsr", (2048, 2048), 16384, (16, 16), 0),
    ("csr", (4096, 4096), _SHARED_NNZ, None, 0),
]

# Custom rows from the original workload list: hybrid COO, multi-sparse-dim COO,
# 3-D/4-D batches and the blocked and CSC layouts.
_CUSTOM_ROWS = [
    ("coo", (1024, 1024), (16,), 262144),
    ("coo", (256, 256, 256), (), 1048576),
    ("coo", (128, 128, 128, 128), (8,), 1048576),
    ("coo", (4096, 4096), (4,), 1048576),
    ("csr", (4096, 4096), 1048576, None, 0),
    ("csc", (4096, 4096), 1048576, None, 0),
    ("bsr", (4096, 4096), 65536, (16, 16), 0),
    ("bsc", (4096, 4096), 65536, (16, 16), 0),
    ("csr", (16, 1024, 1024), 262144, None, 1),
    ("csr", (256, 1024, 64), 16384, None, 1),
    ("bsr", (8, 1024, 1024, 8), 8192, (8, 8), 1),
]

_BENCH_CASES = _SHARED_ROWS + _CUSTOM_ROWS


def _compressed_plan(layout, size, nbatch, blocks):
    """(batch, groups, bound) of a compressed row: the layout's capacity."""
    assert layout in _LAYOUTS, layout
    batch = tuple(size[:nbatch])
    m, n = size[nbatch], size[nbatch + 1]
    if layout in ("bsr", "bsc"):
        block_rows, block_cols = blocks
        assert m % block_rows == 0 and n % block_cols == 0, (layout, size, blocks)
        if layout == "bsr":
            groups, bound = m // block_rows, n // block_cols
        else:
            groups, bound = n // block_cols, m // block_rows
    elif layout == "csc":
        groups, bound = n, m
    else:
        groups, bound = m, n
    return batch, groups, bound


def _stored_shape(row):
    """Stored-values shape of one row; listing stays on the host."""
    if row[0] == "coo":
        _, sparse_shape, dense_shape, nnz = row
        assert 0 <= nnz <= math.prod(sparse_shape), row
        return (nnz,) + tuple(dense_shape)
    layout, size, nnz, blocks, nbatch = row
    _, groups, bound = _compressed_plan(layout, size, nbatch, blocks)
    assert 0 <= nnz <= groups * bound, row
    return (
        tuple(size[:nbatch]) + (nnz,) + tuple(blocks or ()) + tuple(size[nbatch + 2 :])
    )


def _case_fn(row, dtype):
    """One plan per sparse row (``row`` is the descriptor itself)."""
    del dtype
    if row[0] == "coo":
        _, sparse_shape, dense_shape, nnz = row
        yield base.BenchmarkCasePlan(
            shape={"sparse": list(sparse_shape), "dense": list(dense_shape)},
            params={"layout": "coo", "nnz": nnz},
            builder_args=row,
        )
        return
    layout, size, nnz, blocks, nbatch = row
    yield base.BenchmarkCasePlan(
        shape={"size": list(size)},
        params={
            "layout": layout,
            "nnz": nnz,
            "blocks": list(blocks) if blocks else None,
            "nbatch": nbatch,
        },
        builder_args=row,
    )


def _stored_values(values_shape, dtype, device):
    # The accessor returns stored payload without reading it during reference-only execution.
    return torch.empty(values_shape, dtype=dtype, device=device)


def _coo_indices(sparse_shape, nnz, device):
    """Sorted, unique int64 indices: the first ``nnz`` flat positions, unranked."""
    if nnz == 0 or not sparse_shape:
        return torch.empty((len(sparse_shape), nnz), dtype=torch.int64, device=device)
    flat = torch.arange(nnz, dtype=torch.int64, device=device)
    rows = []
    for extent in reversed(tuple(sparse_shape)):
        rows.append(flat % extent)
        flat = flat // extent
    return torch.stack(rows[::-1])


def _build_inputs_fn(plan, dtype, device):
    """Build the sparse input of one plan on ``device``."""
    row = plan.builder_args
    if row[0] == "coo":
        _, sparse_shape, dense_shape, nnz = row
        values = _stored_values((nnz,) + tuple(dense_shape), dtype, device)
        inp = torch.sparse_coo_tensor(
            _coo_indices(sparse_shape, nnz, device),
            values,
            tuple(sparse_shape) + tuple(dense_shape),
            device=device,
            is_coalesced=True,
        )
        return inp, {}

    layout, size, nnz, blocks, nbatch = row
    batch, groups, _ = _compressed_plan(layout, size, nbatch, blocks)
    values = _stored_values(_stored_shape(row), dtype, device)
    counts = torch.full((groups,), nnz // groups, dtype=_INDEX_DTYPE, device=device)
    counts[: nnz % groups] += 1
    compressed = torch.cat(
        [
            torch.zeros(1, dtype=_INDEX_DTYPE, device=device),
            counts.cumsum(0, dtype=_INDEX_DTYPE),
        ]
    )
    plain = torch.arange(nnz, device=device, dtype=_INDEX_DTYPE) - (
        torch.repeat_interleave(compressed[:-1], counts)
    )
    if batch:
        compressed = compressed.expand(*batch, -1).contiguous()
        plain = plain.expand(*batch, -1).contiguous()
    inp = _LAYOUTS[layout](compressed, plain, values, size=tuple(size), device=device)
    return inp, {}


class ValuesBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        rows = []
        for shape in list(self.shapes) + _BENCH_CASES:
            if shape and isinstance(shape[0], str):
                row = tuple(
                    tuple(item) if isinstance(item, list) else item for item in shape
                )
            else:
                # Preserve the declared logical shape with a deterministic 1/8 density.
                size = tuple(shape)
                capacity = math.prod(size)
                row = ("coo", size, (), max(1, capacity // 8) if capacity else 0)
            if row not in rows:
                rows.append(row)
        self.shapes = rows


@pytest.mark.values
def test_values():
    bench = ValuesBenchmark(
        op_name="values",
        torch_op=torch.ops.aten.values,
        gems_op=getattr(flag_gems, "values", None),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
