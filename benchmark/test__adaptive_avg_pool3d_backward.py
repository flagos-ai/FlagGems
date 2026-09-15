# Copyright 2026, The FlagOS Contributors.
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
from _pytest.mark.structures import Mark, MarkDecorator

from . import base, consts

# ``_adaptive_avg_pool3d_backward`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register it directly on the MarkGenerator so
# ``@pytest.mark._adaptive_avg_pool3d_backward`` and ``-m
# _adaptive_avg_pool3d_backward`` both work.
setattr(
    pytest.mark,
    "_adaptive_avg_pool3d_backward",
    MarkDecorator(
        Mark("_adaptive_avg_pool3d_backward", (), {}, _ispytest=True), _ispytest=True
    ),
)

# Shapes for _adaptive_avg_pool3d_backward benchmark
ADAPTIVE_AVG_POOL3D_BACKWARD_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 32, 32, 32),
    (4, 8, 64, 64, 64),
]


class AdaptiveAvgPool3DBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ADAPTIVE_AVG_POOL3D_BACKWARD_SHAPES
        self.output_sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]

    def get_input_iter(self, cur_dtype):
        for shape, output_size in zip(self.shapes, self.output_sizes):
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Compute forward to get output shape
            out = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
            grad = torch.ones_like(out)
            yield grad, x


@pytest.mark._adaptive_avg_pool3d_backward
def test_adaptive_avg_pool3d_backward():
    bench = AdaptiveAvgPool3DBackwardBenchmark(
        op_name="_adaptive_avg_pool3d_backward",
        torch_op=torch.ops.aten._adaptive_avg_pool3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
