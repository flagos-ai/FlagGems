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

import flag_gems
from flag_gems.utils import shape_utils

from . import base

# (N, C, H_in, W_in, H_out, W_out) tuples for the 2D cuDNN grid sampler.
GRID_SAMPLER_SHAPES = [
    (1, 3, 32, 32, 32, 32),
    (2, 32, 64, 64, 64, 64),
    (4, 16, 128, 128, 64, 64),
    (2, 64, 64, 64, 128, 128),
    (8, 32, 256, 256, 64, 64),
]

# cudnn does not support bfloat16 for this operator (CUDNN_STATUS_BAD_PARAM);
# float16 and float32 are supported.
GRID_SAMPLER_DTYPES = [torch.float16, torch.float32]


class GridSamplerBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GRID_SAMPLER_SHAPES
        self.shape_desc = "N, C, H_in, W_in, H_out, W_out"

    def set_more_shapes(self):
        return []

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency=None):
        inp = args[0]
        grid = args[1]
        # read input + grid, write an output of the sampled (H_out, W_out) plane
        io_amount = (
            shape_utils.size_in_bytes(inp)
            + shape_utils.size_in_bytes(grid)
            + shape_utils.size_in_bytes(inp)
            * grid.shape[1]
            * grid.shape[2]
            // (inp.shape[2] * inp.shape[3])
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, cur_dtype):
        for shape in GRID_SAMPLER_SHAPES:
            N, C, H_in, W_in, H_out, W_out = shape
            inp = torch.randn(N, C, H_in, W_in, dtype=cur_dtype, device=self.device)
            grid = (
                torch.rand(N, H_out, W_out, 2, dtype=cur_dtype, device=self.device) * 2
                - 1
            )
            yield inp, grid


class GridSamplerOutBenchmark(GridSamplerBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in GRID_SAMPLER_SHAPES:
            N, C, H_in, W_in, H_out, W_out = shape
            inp = torch.randn(N, C, H_in, W_in, dtype=cur_dtype, device=self.device)
            grid = (
                torch.rand(N, H_out, W_out, 2, dtype=cur_dtype, device=self.device) * 2
                - 1
            )
            out = torch.empty(N, C, H_out, W_out, dtype=cur_dtype, device=self.device)
            yield inp, grid, {"out": out}


@pytest.mark.cudnn_grid_sampler
def test_cudnn_grid_sampler():
    bench = GridSamplerBenchmark(
        op_name="cudnn_grid_sampler",
        torch_op=torch.ops.aten.cudnn_grid_sampler,
        # cudnn rejects bfloat16 input for this operator
        dtypes=GRID_SAMPLER_DTYPES,
        gems_op=flag_gems.cudnn_grid_sampler,
    )
    bench.run()


@pytest.mark.cudnn_grid_sampler_out
def test_cudnn_grid_sampler_out():
    bench = GridSamplerOutBenchmark(
        op_name="cudnn_grid_sampler_out",
        torch_op=torch.ops.aten.cudnn_grid_sampler.out,
        dtypes=GRID_SAMPLER_DTYPES,
        gems_op=flag_gems.cudnn_grid_sampler_out,
    )
    bench.run()
