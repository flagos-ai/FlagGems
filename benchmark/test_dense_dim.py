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

# aten::dense_dim(Tensor self) -> int reports the number of dense dimensions of
# a tensor: ``self.dim()`` for strided tensors, the trailing dense dims for
# sparse COO tensors, and 0 for sparse CSR tensors. It is a pure metadata query
# (the measured work is dispatch and layout introspection, never data
# movement), but the candidate must accept every layout the operator dispatches
# to, so the benchmark covers all three input classes below.
#
# Case descriptors:
#   ("dense", shape)
#   ("coo", sparse_shape, dense_shape, nnz)
#   ("csr", shape, rows, cols, nnz)
_BENCH_CASES = [
    ("dense", ()),
    ("dense", (1024, 1024)),
    ("dense", (4096, 4096)),
    ("dense", (64, 512, 512)),
    ("dense", (16, 128, 64, 60)),
    ("coo", (1024, 1024), (), 65536),
    ("coo", (1024, 1024), (32,), 262144),
    ("coo", (256, 256, 256), (16,), 1048576),
    ("coo", (4096, 4096), (8, 8), 131072),
    ("csr", (1024, 1024), 1024, 1024, 4096),
    ("csr", (64, 512, 512), 512, 512, 8192),
]


def _case_fn(case, dtype):
    del dtype
    kind = case[0]
    if kind == "dense":
        shape = case[1]
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"layout": "dense"},
            builder_args=case,
        )
    elif kind == "coo":
        _, sparse_shape, dense_shape, nnz = case
        yield base.BenchmarkCasePlan(
            shape={"input": sparse_shape + dense_shape},
            params={"layout": "coo", "nnz": nnz},
            builder_args=case,
        )
    else:  # "csr"
        _, shape, _, _, nnz = case
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"layout": "csr", "nnz": nnz},
            builder_args=case,
        )


def _build_inputs_fn(plan, dtype, device):
    case = plan.builder_args
    kind = case[0]
    if kind == "dense":
        inp = torch.randn(case[1], dtype=dtype, device=device)
        return inp, {}
    if kind == "coo":
        _, sparse_shape, dense_shape, nnz = case
        indices = torch.stack(
            [
                torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
                for dim in sparse_shape
            ]
        )
        values = torch.randn((nnz,) + tuple(dense_shape), dtype=dtype, device=device)
        inp = torch.sparse_coo_tensor(
            indices, values, sparse_shape + dense_shape, device=device
        )
        return inp, {}
    # "csr"
    _, shape, rows, cols, nnz = case
    assert 0 <= nnz <= rows * cols
    counts = torch.full((rows,), nnz // rows, dtype=torch.long)
    counts[: nnz % rows] += 1
    crow_indices = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    col_indices = torch.arange(nnz) - torch.repeat_interleave(crow_indices[:-1], counts)
    values = torch.randn(nnz, dtype=dtype, device=device)
    if len(shape) == 3:
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = values.expand(shape[0], -1).contiguous()
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=device
    )
    return inp, {}


class DenseDimBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark whose inputs are dense, sparse COO and sparse
    CSR tensors."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.dense_dim
def test_dense_dim():
    bench = DenseDimBenchmark(
        op_name="dense_dim",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.dense_dim,
        gems_op=getattr(flag_gems, "dense_dim", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
