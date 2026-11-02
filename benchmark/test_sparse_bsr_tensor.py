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

from . import base, consts

# Source shapes chosen to cover small, mid, and large block grids; each is a
# 2-D block-compressed sparse row (BSR) matrix with 16x16 blocks.
SPARSE_BSR_TENSOR_SHAPES = [
    (256, 256),
    (512, 128),
    (1024, 1024),
]


def _make_bsr(shape, dtype, device, density=0.3, blocksize=(16, 16)):
    """Build a block-compressed sparse row tensor on the target device."""
    x = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < density
    return (x * mask).to_sparse_bsr(blocksize)


class SparseBsrTensorBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_BSR_TENSOR_SHAPES
        self.shape_desc = "M, N (BSR, blocksize 16x16)"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            sp = _make_bsr(shape, cur_dtype, self.device)
            yield sp.crow_indices(), sp.col_indices(), sp.values(), shape


@pytest.mark.sparse_bsr_tensor
def test_sparse_bsr_tensor():
    bench = SparseBsrTensorBenchmark(
        op_name="sparse_bsr_tensor",
        torch_op=torch.sparse_bsr_tensor,
        gems_op=flag_gems.sparse_bsr_tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
