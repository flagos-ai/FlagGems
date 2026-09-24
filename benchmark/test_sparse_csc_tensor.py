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

# aten::sparse_csc_tensor.ccol_row_value_size(Tensor ccol_indices,
#     Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# constructs a sparse CSC tensor. Construction work scales with the stored nnz
# (copying ccol/row/values and building the sparse structure), not with the
# logical matrix size, so the benchmark pairs large logical matrices with a
# bounded nnz. Each case is a (logical matrix shape, nnz) pair; the nnz
# entries are spread evenly across the columns.
#
# The shape-inferred sibling (ccol_row_value, 3 positional args) shares the
# same public callable and the same copy-bound cost, so benchmarking the
# explicit-size overload covers the operator.
_CSC_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 262144),
    ((2048, 2048), 131072),
    ((4096, 4096), 65536),
    ((8192, 8192), 65536),
    ((1024, 2048), 131072),
    ((2048, 1024), 131072),
]


def _make_csc_inputs(shape, nnz, dtype, device):
    M, N = shape
    # Spread the nnz entries uniformly across the columns so the compressed
    # column pointers are dense enough to be representative.
    assert 0 <= nnz <= M * N
    q, r = divmod(nnz, N)
    counts = torch.full((N,), q, dtype=torch.long, device=device)
    if r:
        counts[:r] += 1
    ccol = torch.zeros(N + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, 0, out=ccol[1:])
    row = torch.arange(nnz, device=device) - torch.repeat_interleave(ccol[:-1], counts)
    values = torch.randn((nnz,), dtype=dtype, device=device)
    return ccol, row, values


def _case_fn(shape, dtype):
    del dtype
    matrix_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"matrix": matrix_shape, "nnz": nnz},
        params={"nnz": nnz},
        builder_args=(matrix_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    matrix_shape, nnz = plan.builder_args
    ccol, row, values = _make_csc_inputs(matrix_shape, nnz, dtype, device)
    # dtype and device are passed explicitly: without them the aten op forces
    # float32 storage and (on CUDA) fails to infer the target device from the
    # input tensors. The trailing dict is unpacked into kwargs by the
    # benchmark runner, so torch_op and gems_op both receive
    # (ccol, row, values, size=..., dtype=..., layout=..., device=...).
    return (
        ccol,
        row,
        values,
        {
            "size": list(matrix_shape),
            "dtype": dtype,
            "layout": torch.sparse_csc,
            "device": device,
        },
    )


class SparseCscTensorBenchmark(OperatorBenchmark):
    # sparse_csc_tensor is a sparse construction op; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated (matrix shape,
    # nnz) pairs instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CSC_SHAPES)


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor():
    bench = SparseCscTensorBenchmark(
        op_name="sparse_csc_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_csc_tensor,
        # None here lets Benchmark._candidate_call fall back to a KernelGen-injected
        # override first; the benchmark harness resolves the candidate through
        # the same process-local registry as the correctness tests.
        gems_op=getattr(flag_gems, "sparse_csc_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
