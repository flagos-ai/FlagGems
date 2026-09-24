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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

# aten::sparse_bsc_tensor.ccol_row_value_size(Tensor ccol_indices,
#     Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# constructs a sparse BSC tensor. Construction work scales with the stored nnz
# (copying ccol/row/values and building the sparse structure), not with the
# logical matrix size, so the benchmark pairs large logical matrices with a
# bounded nnz. Each triple is a (logical matrix shape, block size, nnz) case;
# every matrix dimension is an exact multiple of its block dimension.
_BSC_SHAPES = [
    ((1024, 1024), (2, 2), 65536),
    ((1024, 1024), (2, 2), 262144),
    ((2048, 2048), (4, 4), 131072),
    ((4096, 4096), (8, 8), 65536),
    ((8192, 8192), (16, 16), 65536),
    ((1024, 2048), (4, 4), 131072),
]


def _make_bsc_inputs(shape, block, nnz, dtype, device):
    M, N = shape
    Br, Bc = block
    assert M % Br == N % Bc == 0
    n_row_blocks = M // Br
    n_col_blocks = N // Bc
    # Spread the nnz entries uniformly across the column blocks so the
    # compressed column pointers are dense enough to be representative.
    assert 0 <= nnz <= n_row_blocks * n_col_blocks
    q, r = divmod(nnz, n_col_blocks)
    counts = torch.full((n_col_blocks,), q, dtype=torch.long, device=device)
    if r:
        counts[:r] += 1
    ccol = torch.zeros(n_col_blocks + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, 0, out=ccol[1:])
    row = torch.arange(nnz, device=device) - torch.repeat_interleave(ccol[:-1], counts)
    values = torch.randn((nnz, Br, Bc), dtype=dtype, device=device)
    return ccol, row, values


def _case_fn(shape, dtype):
    del dtype
    matrix_shape, block, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"matrix": matrix_shape, "block": block, "nnz": nnz},
        params={"nnz": nnz},
        builder_args=(matrix_shape, block, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    matrix_shape, block, nnz = plan.builder_args
    ccol, row, values = _make_bsc_inputs(matrix_shape, block, nnz, dtype, device)
    # The trailing dict is unpacked into kwargs by the benchmark runner, so
    # torch_op and gems_op both receive (ccol, row, values, size=...,
    # dtype=..., layout=..., device=...) with identical call semantics. dtype is
    # passed explicitly: without it the sparse tensor defaults to Float
    # regardless of the values dtype.
    return (
        ccol,
        row,
        values,
        {
            "size": list(matrix_shape),
            "dtype": dtype,
            "layout": torch.sparse_bsc,
            "device": device,
        },
    )


class SparseBscTensorBenchmark(OperatorBenchmark):
    # sparse_bsc_tensor is a sparse construction op; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated
    # (matrix shape, block, nnz) triples instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BSC_SHAPES)


@pytest.mark.sparse_bsc_tensor
def test_sparse_bsc_tensor():
    bench = SparseBscTensorBenchmark(
        op_name="sparse_bsc_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size,
        gems_op=getattr(flag_gems, "sparse_bsc_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
