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

# cudnn_affine_grid_generator takes (theta, N, C, H, W) and returns a
# [N, H, W, 2] grid, so the generic 1D/3D shape defaults do not apply.
# Shapes are given as (N, H, W, C).
CUDNN_AFFINE_GRID_SHAPES = [
    (1, 16, 16, 3),
    (2, 32, 32, 3),
    (4, 64, 64, 3),
    (8, 128, 128, 3),
    (16, 256, 256, 3),
]


class CudnnAffineGridBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CUDNN_AFFINE_GRID_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for N, H, W, C in self.shapes:
            theta = torch.randn((N, 2, 3), dtype=cur_dtype, device=self.device)
            yield theta, N, C, H, W

    def get_gbps(self, args, latency):
        theta, N, C, H, W = args
        io_amount = theta.numel() * theta.element_size() + N * H * W * 2 * (
            theta.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.cudnn_affine_grid_generator
def test_cudnn_affine_grid_generator():
    bench = CudnnAffineGridBenchmark(
        op_name="cudnn_affine_grid_generator",
        torch_op=torch.cudnn_affine_grid_generator,
        # bfloat16 is rejected by cuDNN (CUDNN_STATUS_BAD_PARAM); float16,
        # float32 and float64 are all accepted.
        dtypes=[torch.float16, torch.float32, torch.float64],
    )
    bench.run()


class CudnnAffineGridOutBenchmark(CudnnAffineGridBenchmark):
    def get_input_iter(self, cur_dtype):
        for N, H, W, C in self.shapes:
            theta = torch.randn((N, 2, 3), dtype=cur_dtype, device=self.device)
            out = torch.empty((N, H, W, 2), dtype=cur_dtype, device=self.device)
            yield theta, N, C, H, W, {"out": out}


@pytest.mark.cudnn_affine_grid_generator_out
def test_cudnn_affine_grid_generator_out():
    bench = CudnnAffineGridOutBenchmark(
        op_name="cudnn_affine_grid_generator_out",
        torch_op=torch.ops.aten.cudnn_affine_grid_generator.out,
        dtypes=[torch.float16, torch.float32, torch.float64],
    )
    bench.run()
