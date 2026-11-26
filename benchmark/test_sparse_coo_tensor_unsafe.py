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

# The unsafe constructor is a host-metadata assembly; the benchmark measures
# the assembly path against the native op.
COO_UNSAFE_NNZ = [1, 64, 4096]


class SparseCooTensorUnsafeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = COO_UNSAFE_NNZ

    def get_input_iter(self, cur_dtype):
        for nnz in self.shapes:
            indices = torch.stack(
                [
                    torch.randint(0, 4, (nnz,), device=self.device),
                    torch.randint(0, 4, (nnz,), device=self.device),
                ]
            ).to(torch.int64)
            values = torch.randn((nnz,), dtype=cur_dtype, device=self.device)
            yield indices, values, (4, 4)


@pytest.mark.sparse_coo_tensor_unsafe
def test_sparse_coo_tensor_unsafe():
    """Benchmark against the native unsafe COO constructor."""

    def native_coo(indices, values, size):
        return torch.ops.aten._sparse_coo_tensor_unsafe(
            indices, values, size, device=values.device
        )

    bench = SparseCooTensorUnsafeBenchmark(
        op_name="sparse_coo_tensor_unsafe",
        torch_op=native_coo,
        gems_op=flag_gems.sparse_coo_tensor_unsafe,
        dtypes=[torch.float32],
    )
    bench.run()
