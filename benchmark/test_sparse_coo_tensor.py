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

# Source shapes chosen to cover a 1-D case, a mid-rank case, and a higher-rank
# case; each benchmark call passes an explicit size, which is the structural
# fast path of the constructor.
SPARSE_COO_TENSOR_SHAPES = [
    (256,),
    (20, 320, 15),
    (16, 128, 64, 60),
]


def _make_indices_values(shape, dtype, device, density=0.3):
    """Build a (indices, values) COO component pair on the target device."""
    x = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < density
    sp = (x * mask).to_sparse().coalesce()
    return sp.indices(), sp.values()


class SparseCooTensorBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_COO_TENSOR_SHAPES
        self.shape_desc = "M [, N [, ...]] (COO)"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            indices, values = _make_indices_values(shape, cur_dtype, self.device)
            yield indices, values, shape


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor():
    bench = SparseCooTensorBenchmark(
        op_name="sparse_coo_tensor",
        torch_op=torch.sparse_coo_tensor,
        gems_op=flag_gems.sparse_coo_tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
