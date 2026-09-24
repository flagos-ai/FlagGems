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
from . import base, consts, utils

# aten::dim(Tensor self) -> int reports the number of dimensions of a tensor:
# ``len(self.size())`` for strided tensors and the full logical rank for sparse
# COO/CSR layouts. It is a pure metadata query (the measured work is dispatch
# and layout introspection, never data movement), but the candidate must accept
# every layout the operator dispatches to, so every case below is materialized
# as a dense, sparse COO or sparse CSR tensor.
_DIM_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 128, 128, 16),
]

# Number of stored entries for every sparse case. dim is O(1), so nnz only
# affects input allocation, not the measured call.
_DIM_NNZ = 4096


def _make_coo_input(shape, sparse_dim, dtype, device, nnz=_DIM_NNZ, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = utils.generate_tensor_input((nnz,) + tuple(dense_shape), dtype, device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _make_csr_input(shape, dtype, device, nnz=_DIM_NNZ):
    # 2-D (rows, cols) or batched 3-D (batch, rows, cols); every batch stores
    # the same nnz entries (shared crow/col pattern).
    if len(shape) == 2:
        rows, cols = shape
    else:
        _, rows, cols = shape
    assert 0 <= nnz <= rows * cols
    counts = torch.full((rows,), nnz // rows, dtype=torch.long)
    counts[: nnz % rows] += 1
    crow_indices = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    col_indices = torch.arange(nnz) - torch.repeat_interleave(crow_indices[:-1], counts)
    if len(shape) == 3:
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = utils.generate_tensor_input((shape[0], nnz), dtype, device)
    else:
        values = utils.generate_tensor_input((nnz,), dtype, device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=device
    )


def _case_fn(shape, dtype):
    del dtype
    # Dense (strided) layout: dim == len(shape).
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"layout": "dense"},
        builder_args=(shape, "dense", None),
    )
    # Sparse COO layout: all-sparse for 2-D, hybrid sparse+dense for higher
    # ranks. sparse_dim stays in [1, ndim] so shapes merged in by other bench
    # levels remain valid.
    sparse_dim = len(shape) if len(shape) <= 2 else len(shape) - 1
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"layout": "coo", "sparse_dim": sparse_dim, "nnz": _DIM_NNZ},
        builder_args=(shape, "coo", sparse_dim),
    )
    # Sparse CSR layout where a 2-D / batched 3-D compressed layout exists.
    if len(shape) in (2, 3):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"layout": "csr", "nnz": _DIM_NNZ},
            builder_args=(shape, "csr", None),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, layout, sparse_dim = plan.builder_args
    if layout == "dense":
        inp = utils.generate_tensor_input(shape, dtype, device)
    elif layout == "coo":
        inp = _make_coo_input(shape, sparse_dim, dtype, device)
    else:
        inp = _make_csr_input(shape, dtype, device)
    return inp, {}


class DimBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark whose inputs are dense, sparse COO and sparse
    CSR tensors."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DIM_SHAPES)


@pytest.mark.dim
def test_dim():
    bench = DimBenchmark(
        op_name="dim",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.dim,
        # flag_gems has no public ``dim`` direct callable yet; the candidate is
        # supplied by the process-local override keyed on the public operator
        # name "dim" (resolved via Benchmark._candidate_call inside the
        # benchmark runner).
        gems_op=getattr(flag_gems, "dim", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
