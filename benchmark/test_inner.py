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

# torch.inner contracts the last dimension of both operands. The cases cover the
# three performance regimes: the 1-D dot-product reduction (memory bound), the
# 2-D matrix product (tensor-core bound), and the batched/outer-product forms.
INNER_SHAPES = [
    ((4096,), (4096,)),
    ((16384,), (16384,)),
    ((512, 1024), (1024, 1024)),
    ((1024, 4096), (2048, 4096)),
    ((2048, 4096), (2048, 4096)),
    ((256, 128), (128,)),
]


class InnerBenchmark(base.Benchmark):
    """Benchmark for aten::inner (generalized inner product)."""

    DEFAULT_SHAPE_DESC = "input shape, other shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = INNER_SHAPES

    def get_input_iter(self, dtype):
        for input_shape, other_shape in self.shapes:
            inp = torch.randn(input_shape, dtype=dtype, device=self.device)
            other = torch.randn(other_shape, dtype=dtype, device=self.device)
            yield inp, other


@pytest.mark.inner
def test_inner():
    bench = InnerBenchmark(
        op_name="inner",
        torch_op=torch.inner,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
