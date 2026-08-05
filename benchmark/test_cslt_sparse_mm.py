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

"""
Benchmark for _cslt_sparse_mm operator.

NOTE: This benchmarks the native PyTorch implementation only, as
_cslt_sparse_mm is NOT implemented in FlagGems (vendor-specific operator).
"""

import pytest
import torch

from . import base

# cuSPARSELt sparse MM shapes
CSLT_SPARSE_MM_SHAPES = [
    (64, 128, 64),
    (128, 256, 128),
    (256, 512, 256),
    (512, 1024, 512),
]


class CsltSparseMMBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CSLT_SPARSE_MM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            M, K, N = shape
            # Create sparse matrix with 50% sparsity
            A = torch.randn(M, K, dtype=cur_dtype, device=self.device)
            mask = torch.rand_like(A) > 0.5
            A_sparse = A * mask
            compressed_A = torch._cslt_compress(A_sparse)
            B = torch.randn(K, N, dtype=cur_dtype, device=self.device)
            yield compressed_A, B


@pytest.mark.cslt_sparse_mm
def test_cslt_sparse_mm_perf():
    """
    Benchmark native PyTorch _cslt_sparse_mm implementation.

    Note: FlagGems does not provide its own implementation of this operator.
    """
    bench = CsltSparseMMBenchmark(
        op_name="cslt_sparse_mm",
        torch_op=torch._cslt_sparse_mm,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
