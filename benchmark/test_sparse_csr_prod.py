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

# (M, N) compressed grids; density is picked so the column reduction exercises
# both dense and sparse column populations.
SPARSE_CSR_PROD_SHAPES = [
    (1024, 1024),
    (4096, 512),
    (512, 4096),
]


def _make_csr(shape, dtype, device, density=0.3):
    """Build a compressed sparse row tensor on the target device."""
    x = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < density
    return (x * mask).to_sparse_csr()


class SparseCsrProdBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_CSR_PROD_SHAPES
        self.shape_desc = "M, N (CSR)"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            sp = _make_csr(shape, cur_dtype, self.device)
            yield (sp, [0], True, {"dtype": cur_dtype})


@pytest.mark.sparse_csr_prod
def test_sparse_csr_prod():
    bench = SparseCsrProdBenchmark(
        op_name="sparse_csr_prod",
        torch_op=torch.ops.aten._sparse_csr_prod,
        gems_op=flag_gems.sparse_csr_prod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
