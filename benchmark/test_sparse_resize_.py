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

# aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim)
# -> Tensor(a!) resizes a sparse COO tensor in place to ``size`` with
# ``sparse_dim`` sparse and ``dense_dim`` dense dimensions. The cost scales
# with nnz and the sparse/dense storage touched, so each case below is a
# (logical_src_shape, sparse_dim, nnz, dst_size, dst_sparse_dim, dst_dense_dim)
# tuple. Only non-shrinking resizes on non-empty tensors and free reshapes of
# the empty tensor are benchmarked (the reference rejects the other directions
# for non-empty tensors). There are no meaningful sparse shapes in
# core_shapes.yaml for this op, so dedicated sparse cases are used.
_SPARSE_RESIZE_CASES = [
    ((1024, 1024), 2, 65536, [1024, 2048], 2, 0),
    ((1024, 1024), 2, 1048576, [2048, 1024], 2, 0),
    ((4096, 4096), 2, 1048576, [4096, 8192], 2, 0),
    ((64, 512, 512), 3, 524288, [64, 512, 1024], 3, 0),
    ((16, 1024, 1024), 2, 8192, [32, 1024, 1024], 2, 1),
    ((1024, 1024), 2, 0, [2048, 2048], 2, 0),
]


def _make_sparse_input(shape, sparse_dim, nnz, dtype, device):
    # Sparse COO input built directly on the benchmark device. Indices are
    # drawn with replacement (a valid, possibly uncoalesced sparse structure);
    # the resize cost depends only on nnz and the sparse/dense storage.
    dense_shape = tuple(shape[sparse_dim:])
    values_shape = (nnz,) + dense_shape
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in shape[:sparse_dim]
        ]
    )
    values = torch.randn(values_shape, device=device).to(dtype)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _case_fn(shape, dtype):
    del dtype
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = shape
    yield base.BenchmarkCasePlan(
        shape={"input": src_shape},
        params={
            "size": size,
            "sparse_dim": new_sparse_dim,
            "dense_dim": new_dense_dim,
        },
        builder_args=(
            src_shape,
            sparse_dim,
            nnz,
            size,
            new_sparse_dim,
            new_dense_dim,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = plan.builder_args
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype, device)
    return inp, {
        "size": size,
        "sparse_dim": new_sparse_dim,
        "dense_dim": new_dense_dim,
    }


class SparseResizeBenchmark(OperatorBenchmark):
    # sparse_resize_ reshapes a sparse COO tensor; the dense default shapes in
    # core_shapes.yaml do not apply, so benchmark dedicated
    # (logical_src_shape, sparse_dim, nnz, dst_size, dst_sparse_dim,
    # dst_dense_dim) tuples instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SPARSE_RESIZE_CASES)


@pytest.mark.sparse_resize_
def test_sparse_resize_():
    bench = SparseResizeBenchmark(
        op_name="sparse_resize_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_resize_,
        gems_op=getattr(flag_gems, "sparse_resize_", None),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
