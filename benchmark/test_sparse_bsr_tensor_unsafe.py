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
# the assembly path (no dtype conversion requested) against the native op.
BSR_UNSAFE_NNZ = [1, 64, 4096]


class SparseBsrTensorUnsafeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = BSR_UNSAFE_NNZ

    def get_input_iter(self, cur_dtype):
        for nnz in self.shapes:
            crow = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=self.device)
            col = torch.randint(0, 2, (nnz,), device=self.device).to(torch.int64)
            values = torch.randn((nnz, 2, 2), dtype=cur_dtype, device=self.device)
            yield crow, col, values, (4, 4, 2)


@pytest.mark.sparse_bsr_tensor_unsafe
def test_sparse_bsr_tensor_unsafe():
    """Benchmark against the native unsafe BSR constructor.

    The native unsafe constructor requires an explicit device kwarg (it fails
    its internal values/instance device check when omitted), so the baseline is
    wrapped to pass one.
    """

    def native_bsr(crow, col, values, size):
        return torch.ops.aten._sparse_bsr_tensor_unsafe(
            crow, col, values, size, device=values.device
        )

    bench = SparseBsrTensorUnsafeBenchmark(
        op_name="sparse_bsr_tensor_unsafe",
        torch_op=native_bsr,
        gems_op=flag_gems.sparse_bsr_tensor_unsafe,
        dtypes=[torch.float32],
    )
    bench.run()
