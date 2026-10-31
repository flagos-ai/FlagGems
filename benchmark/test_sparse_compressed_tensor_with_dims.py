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

from . import base

# The constructor allocates shell buffers only; the benchmark measures the
# allocation geometry against the native op.
WITH_DIMS_NNZ = [1, 64, 4096]


class SparseCompressedTensorWithDimsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = WITH_DIMS_NNZ

    def get_input_iter(self, cur_dtype):
        for nnz in self.shapes:
            yield nnz, 0, [4, 4], [], torch.int64


@pytest.mark.sparse_compressed_tensor_with_dims
def test_sparse_compressed_tensor_with_dims():
    """Benchmark against the native with-dims constructor."""

    def native_with_dims(nnz, dense_dim, size, blocksize, index_dtype):
        return torch.ops.aten._sparse_compressed_tensor_with_dims(
            nnz,
            dense_dim,
            size,
            blocksize,
            index_dtype,
            dtype=torch.float32,
            layout=torch.sparse_csr,
            device="cuda",
        )

    def gems_with_dims(nnz, dense_dim, size, blocksize, index_dtype):
        return flag_gems.sparse_compressed_tensor_with_dims(
            nnz,
            dense_dim,
            size,
            blocksize,
            index_dtype,
            dtype=torch.float32,
            layout=torch.sparse_csr,
            device="cuda",
        )

    bench = SparseCompressedTensorWithDimsBenchmark(
        op_name="sparse_compressed_tensor_with_dims",
        torch_op=native_with_dims,
        gems_op=gems_with_dims,
        dtypes=[torch.float32],
    )
    bench.run()
