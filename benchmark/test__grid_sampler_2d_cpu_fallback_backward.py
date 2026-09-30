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

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::_grid_sampler_2d_cpu_fallback_backward(grad_output, input, grid,
# interpolation_mode, padding_mode, align_corners) -> (grad_input, grad_grid) is
# a host-only ATen composite: it has no accelerator kernel (accelerator operands
# segfault inside the CPU kernel), so the builders create CPU float32 tensors -
# the operator's real contract. float32 is the only accepted dtype. A case is
# (N, C, H, W, out_H, out_W): input (N, C, H, W), grid (N, out_H, out_W, 2) and
# grad_output (N, C, out_H, out_W).
_SHAPES = [
    (1, 1, 4, 4, 2, 2),
    (2, 3, 8, 6, 5, 7),
    (8, 3, 32, 32, 16, 16),
    (16, 3, 64, 64, 32, 32),
    (4, 16, 32, 24, 17, 13),
    (16, 128, 64, 60, 4, 4),
    (2, 64, 48, 40, 33, 25),
]

# Keep all native interpolation, padding and align_corners branches.
_BENCH_MODES = [(i, p, a) for i in (0, 1, 2) for p in (0, 1, 2) for a in (False, True)]


def _case_fn(shape, dtype):
    del dtype
    for mode in _BENCH_MODES:
        yield base.BenchmarkCasePlan(
            shape={"input": shape[:4], "grid": (shape[0], shape[4], shape[5], 2)},
            params={
                "interpolation_mode": mode[0],
                "padding_mode": mode[1],
                "align_corners": mode[2],
            },
            builder_args=(shape, mode),
        )


def _build_inputs_fn(plan, dtype, device):
    # The operator fixes its own dtype and device; both are host-side.
    del dtype, device
    n, c, h, w, out_h, out_w = plan.builder_args[0]
    interpolation_mode, padding_mode, align_corners = plan.builder_args[1]
    grad_output = torch.randn(n, c, out_h, out_w, dtype=torch.float32)
    inp = torch.randn(n, c, h, w, dtype=torch.float32)
    # Coordinates span [-1.2, 1.2] so the padding branches run alongside the
    # in-bounds interpolation path.
    grid = torch.rand(n, out_h, out_w, 2, dtype=torch.float32) * 2.4 - 1.2
    return (grad_output, inp, grid, interpolation_mode, padding_mode, align_corners)


class GridSampler2dCpuFallbackBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        requested = []
        for shape in self.shapes:
            shape = tuple(shape)
            if len(shape) == 4:
                # A shared NCHW image keeps its geometry and spatial output size.
                requested.append((*shape, shape[-2], shape[-1]))
            elif len(shape) == 6:
                requested.append(shape)
        self.shapes = list(dict.fromkeys(requested + _SHAPES))


@pytest.mark.grid_sampler_2d_cpu_fallback_backward
def test__grid_sampler_2d_cpu_fallback_backward():
    bench = GridSampler2dCpuFallbackBackwardBenchmark(
        op_name="_grid_sampler_2d_cpu_fallback_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._grid_sampler_2d_cpu_fallback_backward.default,
        gems_op=getattr(flag_gems, "_grid_sampler_2d_cpu_fallback_backward", None),
        dtypes=[torch.float32],
    )
    bench.run()
