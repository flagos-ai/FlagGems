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
"""Performance benchmark for ``cudnn_grid_sampler_backward``.

``cudnn_grid_sampler_backward`` consumes three tensors (``input``, ``grid`` and
``grad_output``), so it uses a dedicated benchmark class with 4D image shapes
instead of the pointwise templates.
"""

import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base

# cuDNN does not accept bfloat16 for this operator, so the reference can only be
# timed for float16/float32/float64 (a subset of utils.FLOAT_DTYPES).
GRID_SAMPLER_BACKWARD_DTYPES = [torch.float16, torch.float32, torch.float64]

# (N, C, H_in, W_in); the grid keeps the spatial size of the input.
SHAPES = [
    (1, 3, 32, 32),
    (2, 8, 64, 64),
    (2, 32, 64, 64),
    (4, 16, 128, 128),
    (8, 32, 56, 56),
]


class GridSamplerBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Shapes are grid-sample specific; do not read the shared shape file.
        self.shapes = SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            N, C, H_in, W_in = shape
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            grid = torch.empty(
                (N, H_in, W_in, 2), dtype=dtype, device=self.device
            ).uniform_(-1.0, 1.0)
            grad_output = torch.randn(
                (N, C, H_in, W_in), dtype=dtype, device=self.device
            )
            yield inp, grid, grad_output

    def get_gbps(self, args, latency=None):
        inp, grid, grad_output = args
        grad_input = torch.empty_like(inp)
        grad_grid = torch.empty_like(grid)
        io_amount = (
            shape_utils.size_in_bytes(inp)
            + shape_utils.size_in_bytes(grid)
            + shape_utils.size_in_bytes(grad_output)
            + shape_utils.size_in_bytes(grad_input)
            + shape_utils.size_in_bytes(grad_grid)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_metrics(self):
        return ["gbps"]


@pytest.mark.cudnn_grid_sampler_backward
def test_cudnn_grid_sampler_backward():
    bench = GridSamplerBackwardBenchmark(
        op_name="cudnn_grid_sampler_backward",
        torch_op=torch.ops.aten.cudnn_grid_sampler_backward,
        dtypes=GRID_SAMPLER_BACKWARD_DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_grid_sampler_backward)
    bench.run()


@pytest.mark.cudnn_grid_sampler_backward_out
def test_cudnn_grid_sampler_backward_out():
    outs = {}

    def _bind_outs(torch_op):
        def op(input, grid, grad_output):
            # Reuse the pre-allocated output buffers for each shape so the
            # timing only covers the operator itself.
            key = (tuple(input.shape), tuple(grid.shape), input.dtype)
            if key not in outs:
                out = (torch.empty_like(input), torch.empty_like(grid))
                outs[key] = out
            else:
                out = outs[key]
            out0, out1 = out
            return torch_op(input, grid, grad_output, out0=out0, out1=out1)

        return op

    bench = GridSamplerBackwardBenchmark(
        op_name="cudnn_grid_sampler_backward_out",
        torch_op=_bind_outs(torch.ops.aten.cudnn_grid_sampler_backward.out),
        dtypes=GRID_SAMPLER_BACKWARD_DTYPES,
    )
    bench.set_gems(_bind_outs(flag_gems.cudnn_grid_sampler_backward_out))
    bench.run()
