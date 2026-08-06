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


def reflection_pad2d_backward_input_fn(shape, dtype, device):
    """Generate input for reflection_pad2d_backward benchmark."""
    inp = torch.randn(shape, dtype=dtype, device=device)
    # Use moderate padding: left, right, top, bottom
    padding = [2, 2, 2, 2]
    padded = torch.nn.functional.pad(inp, padding, mode="reflect")
    grad_output = torch.randn_like(padded)
    yield grad_output, inp, padding


class ReflectionPad2dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative 4D shapes from ResNet-like workloads for pad2d backward
        self.shapes = [
            (4, 3, 224, 224),
            (16, 64, 56, 56),
            (32, 128, 28, 28),
            (64, 256, 14, 14),
            (128, 512, 7, 7),
        ]

    def set_more_shapes(self):
        # Additional shapes covering 3D-like and non-ResNet sizes
        return [
            (1, 3, 512, 512),
            (8, 64, 128, 128),
            (4, 128, 64, 64),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from reflection_pad2d_backward_input_fn(shape, cur_dtype, self.device)


@pytest.mark.reflection_pad2d_backward
def test_reflection_pad2d_backward():
    bench = ReflectionPad2dBackwardBenchmark(
        op_name="reflection_pad2d_backward",
        torch_op=torch.ops.aten.reflection_pad2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
