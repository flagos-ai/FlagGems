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


"""Performance benchmark for aten::cudnn_grid_sampler.

The operator samples a rank-4 input at rank-4 grid coordinates, so each case is
an (input_shape, grid_shape) pair with ``grid.shape[-1] == 2`` and equal batch
sizes.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_HAS_FP64 = bool(getattr(flag_gems.runtime.device, "support_fp64", False))
_DTYPES = [torch.float32, torch.float16] + ([torch.float64] if _HAS_FP64 else [])

# (input_shape, grid_shape) pairs; grid last dim must stay 2.
_BENCH_CASES = [
    ((16, 128, 64, 60), (16, 8, 8, 2)),
    ((1, 1, 1024, 1024), (1, 32, 32, 2)),
    ((8, 16, 128, 128), (8, 32, 32, 2)),
    ((4, 3, 256, 256), (4, 64, 64, 2)),
    ((2, 3, 512, 512), (2, 16, 16, 2)),
    ((2, 6, 128, 128), (2, 16, 16, 2)),
]


def _case_fn(case, dtype):
    # set_shapes feeds one (input_shape, grid_shape) descriptor per case.
    del dtype
    input_shape, grid_shape = case
    yield base.BenchmarkCasePlan(
        shape={"input": list(input_shape), "grid": list(grid_shape)},
        params={"samples": list(grid_shape[:3])},
        builder_args=(input_shape, grid_shape),
    )


def _build_inputs_fn(plan, dtype, device):
    input_shape, grid_shape = plan.builder_args
    inp = torch.randn(input_shape, dtype=dtype, device=device)
    # Measured native behaviour: coordinates outside [-1, 1] fall back to the
    # zero padding, so a [-1, 1] grid keeps the timed samples on real pixels.
    grid = torch.rand(grid_shape, dtype=dtype, device=device) * 2 - 1
    return inp, grid


class CudnnGridSamplerBenchmark(OperatorBenchmark):
    """Two-phase benchmark over rank-4 (input, grid) pairs of growing size."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)

    def set_more_shapes(self):
        # The level-2 extras are plain 1-D/2-D/3-D shapes and cannot express the
        # (input, grid) pair this operator needs, so listing and execution use
        # exactly _BENCH_CASES at every bench level.
        return []


@pytest.mark.cudnn_grid_sampler
def test_cudnn_grid_sampler():
    bench = CudnnGridSamplerBenchmark(
        op_name="cudnn_grid_sampler",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cudnn_grid_sampler,
        gems_op=getattr(flag_gems, "cudnn_grid_sampler", None),
        dtypes=_DTYPES,
    )
    bench.run()
