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

# Shapes cover small to large NCHW combinations typical for unpooling benchmarks
MAX_UNPOOL2D_SHAPES = [
    (1, 1, 8, 8),
    (1, 1, 16, 16),
    (2, 3, 16, 16),
    (4, 8, 32, 32),
    (1, 16, 32, 32),
]


class MaxUnpool2dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MAX_UNPOOL2D_SHAPES

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n, c, h, w = shape
            # Create input tensor
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Apply max_pool2d to get pooled output and indices
            pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
            pooled, indices = pool(x.contiguous())
            output_size = [h, w]
            yield pooled, indices.to(torch.int64), output_size

    def get_tflops(self, op, *args, **kwargs):
        pooled, indices, output_size = args
        return pooled.numel()


@pytest.mark.max_unpool2d
def test_max_unpool2d():
    def torch_max_unpool2d(pooled, indices, output_size):
        return torch.ops.aten.max_unpool2d(pooled, indices, output_size)

    bench = MaxUnpool2dBenchmark(
        op_name="max_unpool2d",
        torch_op=torch_max_unpool2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
