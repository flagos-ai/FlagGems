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

from . import base

# cudnn_affine_grid_generator_backward takes (grad, N, C, H, W) and reduces over
# the (H, W) plane, so the benchmark uses (N, C, H, W) tuples as shapes.
AFFINE_GRID_BACKWARD_SHAPES = [
    (1, 3, 8, 8),
    (2, 3, 32, 32),
    (4, 3, 64, 64),
    (8, 3, 128, 128),
    (2, 3, 256, 256),
]


class CudnnAffineGridBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = AFFINE_GRID_BACKWARD_SHAPES
        self.shape_desc = "N, C, H, W"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for N, C, H, W in self.shapes:
            grad = torch.randn((N, H, W, 2), dtype=cur_dtype, device=self.device)
            yield grad, N, C, H, W

    def get_gbps(self, args, latency=None):
        grad = args[0]
        # read the whole grad tensor, write the [N, 2, 3] grad_theta
        total_bytes = (
            grad.numel() * grad.element_size() + args[1] * 6 * grad.element_size()
        )
        return total_bytes / (latency * 1e3) / 1e3


class CudnnAffineGridBackwardOutBenchmark(CudnnAffineGridBackwardBenchmark):
    def get_input_iter(self, cur_dtype):
        for N, C, H, W in self.shapes:
            grad = torch.randn((N, H, W, 2), dtype=cur_dtype, device=self.device)
            out = torch.empty((N, 2, 3), dtype=cur_dtype, device=self.device)
            yield grad, N, C, H, W, {"out": out}


@pytest.mark.cudnn_affine_grid_generator_backward
def test_cudnn_affine_grid_generator_backward():
    bench = CudnnAffineGridBackwardBenchmark(
        op_name="cudnn_affine_grid_generator_backward",
        torch_op=torch.ops.aten.cudnn_affine_grid_generator_backward,
        # cudnn rejects bfloat16 input for this operator
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.cudnn_affine_grid_generator_backward_out
def test_cudnn_affine_grid_generator_backward_out():
    bench = CudnnAffineGridBackwardOutBenchmark(
        op_name="cudnn_affine_grid_generator_backward_out",
        torch_op=torch.ops.aten.cudnn_affine_grid_generator_backward.out,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
