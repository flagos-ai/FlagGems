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

from . import base, consts

# cov takes a 2D (variables x observations) matrix and produces an N x N
# covariance matrix. Shapes range from small to large in both dimensions so the
# mean reduction and the centered Gram GEMM are both exercised.
COV_SHAPES = [
    (32, 256),
    (64, 1024),
    (128, 1024),
    (256, 1024),
    (256, 4096),
    (512, 2048),
    (512, 4096),
]


class CovBenchmark(base.Benchmark):
    """Benchmark for aten::cov (covariance matrix)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = COV_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            yield inp,


@pytest.mark.cov
def test_cov():
    bench = CovBenchmark(
        op_name="cov",
        torch_op=torch.cov,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
