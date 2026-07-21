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

from . import base
from .consts import FLOAT_DTYPES

# 3D volumes of varying sizes for benchmarking reflection padding backward
REFLECTION_PAD3D_BACKWARD_SHAPES = [
    (1, 1, 4, 4, 4),
    (2, 3, 8, 8, 8),
    (1, 1, 16, 16, 16),
    (2, 4, 8, 16, 32),
]

# Padding values must be strictly less than corresponding dimension size
REFLECTION_PAD3D_PADDINGS = [
    (1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2),
    (1, 2, 1, 2, 1, 2),
]


class ReflectionPad3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Generate all combinations of shapes and paddings
        self.shapes = [
            (shape, padding)
            for shape in REFLECTION_PAD3D_BACKWARD_SHAPES
            for padding in REFLECTION_PAD3D_PADDINGS
        ]

    def get_input_iter(self, cur_dtype):
        for shape, padding in self.shapes:
            N, C, D, H, W = shape
            pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding
            D_out = D + pad_d0 + pad_d1
            H_out = H + pad_h0 + pad_h1
            W_out = W + pad_w0 + pad_w1

            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            grad_output = torch.ones(
                (N, C, D_out, H_out, W_out), dtype=cur_dtype, device=self.device
            )
            yield grad_output, x, padding


@pytest.mark.reflection_pad3d_backward
def test_reflection_pad3d_backward():
    bench = ReflectionPad3dBackwardBenchmark(
        op_name="reflection_pad3d_backward",
        torch_op=torch.ops.aten.reflection_pad3d_backward,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
