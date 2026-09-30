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

# aten::_grid_sampler_2d_cpu_fallback is CPU + float32 only: its body hardcodes
# `using scalar_t = float` and reads host memory (an accelerator input segfaults
# the CUDA backend instead of raising), and every other dtype raises
# "RuntimeError: expected scalar type Float but found <T>". The operands are
# therefore allocated on the CPU, which keeps the native timing baseline (also
# CPU) a meaningful comparison.
_CPU = torch.device("cpu")

# (N, C, H, W) images; the grid is (N, OH, OW, 2). A six-element row is
# (N, C, H, W, OH, OW) and decouples the grid resolution from the image
# resolution; every other row samples the image at its own resolution.
BENCH_SHAPES = [
    (1, 3, 256, 256),
    (2, 3, 256, 256, 128, 128),
    (8, 16, 128, 128),
    (16, 128, 64, 60),
    (32, 64, 32, 32),
]


def _coord_grid(n, out_h, out_w, span=1.5):
    """(N, OH, OW, 2) coordinates covering in-bounds, boundary and out-of-bounds
    samples."""
    xs = torch.linspace(-span, span, out_w, dtype=torch.float32, device=_CPU)
    ys = torch.linspace(-span, span, out_h, dtype=torch.float32, device=_CPU)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    shift = torch.arange(n, dtype=torch.float32, device=_CPU) - (n - 1) / 2
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0) + (0.05 * shift).view(n, 1, 1, 1)
    return grid.contiguous()


def _case_fn(shape, dtype):
    del dtype
    n, c, h, w = shape[:4]
    out_h, out_w = (shape[4], shape[5]) if len(shape) > 4 else (h, w)
    for interpolation in (0, 1, 2):
        for padding in (0, 1, 2):
            for align in (False, True):
                yield base.BenchmarkCasePlan(
                    shape={"input": (n, c, h, w), "grid": (n, out_h, out_w, 2)},
                    params={
                        "interpolation_mode": interpolation,
                        "padding_mode": padding,
                        "align_corners": align,
                    },
                    builder_args=(n, c, h, w, out_h, out_w),
                )


def _build_inputs_fn(plan, dtype, device):
    # `device` is the harness device; this operator's native contract is the CPU.
    del device
    n, c, h, w, out_h, out_w = plan.builder_args
    inp = torch.randn((n, c, h, w), dtype=dtype, device=_CPU)
    grid = _coord_grid(n, out_h, out_w)
    return (
        inp,
        grid,
        plan.params["interpolation_mode"],
        plan.params["padding_mode"],
        plan.params["align_corners"],
    )


class GridSampler2dCpuFallbackBenchmark(base.GenericBenchmark):
    """Rank-4 CPU workloads for aten::_grid_sampler_2d_cpu_fallback."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # The shared grid is 1-D/2-D/3-D/5-D, which the native
        # check_grid_sampler_2d rejects before any index math (the operator is
        # rank-4 only), so those rows are dropped. Rows a shape file provides
        # with an accepted rank are kept and the operator's own native-valid
        # geometry is appended.
        self.shapes = list(
            dict.fromkeys(
                [tuple(s) for s in self.shapes if len(s) in (4, 6)] + BENCH_SHAPES
            )
        )


@pytest.mark.grid_sampler_2d_cpu_fallback
def test__grid_sampler_2d_cpu_fallback():
    bench = GridSampler2dCpuFallbackBenchmark(
        op_name="_grid_sampler_2d_cpu_fallback",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._grid_sampler_2d_cpu_fallback,
        gems_op=getattr(flag_gems, "_grid_sampler_2d_cpu_fallback", None),
        dtypes=[torch.float32],
    )
    bench.run()
