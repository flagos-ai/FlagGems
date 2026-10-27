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

# Shapes covering 2D and 3D tensors for renorm benchmarking
RENORM_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (16, 32, 64),
    (8, 16, 128),
]


class RenormBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RENORM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            p = 2.0
            dim = 1 if len(shape) > 1 else 0
            maxnorm = 1.0
            yield x, p, dim, maxnorm


@pytest.mark.renorm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_renorm():
    bench = RenormBenchmark(
        op_name="renorm",
        torch_op=torch.renorm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
