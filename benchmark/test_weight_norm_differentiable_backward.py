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


class WeightNormDifferentiableBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((4, 8), 0),
            ((64, 128), 1),
            ((256, 256), 0),
            ((512, 1024), 1),
            ((2048, 1024), 0),
        ]

    def get_input_iter(self, dtype):
        for shape, dim in self.shapes:
            broadcast_shape = [1] * len(shape)
            broadcast_shape[dim] = shape[dim]
            grad_w = torch.randn(shape, dtype=dtype, device=self.device)
            saved_v = torch.randn(shape, dtype=dtype, device=self.device)
            saved_g = torch.randn(broadcast_shape, dtype=dtype, device=self.device)
            _, saved_norms = torch._weight_norm_interface(saved_v, saved_g, dim)
            yield grad_w, saved_v, saved_g, saved_norms, dim


@pytest.mark.weight_norm_differentiable_backward
def test_weight_norm_differentiable_backward_benchmark():
    bench = WeightNormDifferentiableBackwardBenchmark(
        op_name="weight_norm_differentiable_backward",
        torch_op=torch.ops.aten._weight_norm_differentiable_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._weight_norm_differentiable_backward)
    bench.run()
