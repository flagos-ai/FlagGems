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
Benchmark for the _cslt_sparse_mm operator.

Compares the native cuSPARSELt implementation (``torch._cslt_sparse_mm``)
against the FlagGems Triton implementation, which decodes the 2:4 compressed
blob and runs a dense matmul. The gems path is exercised automatically by the
benchmark harness via ``flag_gems.use_gems()``.
"""

import pytest
import torch

from flag_gems.ops._cslt_sparse_mm import _cslt_sparse_mm_enabled

from . import base

# cuSPARSELt sparse MM shapes (M, K, N)
CSLT_SPARSE_MM_SHAPES = [
    (64, 128, 64),
    (128, 256, 128),
    (256, 512, 256),
    (512, 1024, 512),
]


def _make_2to4(M, K, dtype, device):
    """Build an M x K matrix with an exact 2:4 sparsity pattern."""
    a = torch.randn(M, K, dtype=dtype, device=device).view(M, K // 4, 4)
    idx = a.abs().argsort(dim=-1)
    mask = torch.zeros_like(a)
    mask.scatter_(-1, idx[..., 2:], 1.0)
    return (a * mask).view(M, K).contiguous()


class CsltSparseMMBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CSLT_SPARSE_MM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            M, K, N = shape
            A_sparse = _make_2to4(M, K, cur_dtype, self.device)
            compressed_A = torch._cslt_compress(A_sparse)
            B = torch.randn(K, N, dtype=cur_dtype, device=self.device)
            yield compressed_A, B


@pytest.mark.skipif(
    not _cslt_sparse_mm_enabled(),
    reason=(
        "the Triton _cslt_sparse_mm decoder models the Hopper cuSPARSELt "
        "metadata layout; not registered on this architecture"
    ),
)
@pytest.mark.cslt_sparse_mm
def test_cslt_sparse_mm_perf():
    """Benchmark native cuSPARSELt vs the FlagGems Triton _cslt_sparse_mm."""
    bench = CsltSparseMMBenchmark(
        op_name="cslt_sparse_mm",
        torch_op=torch._cslt_sparse_mm,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
