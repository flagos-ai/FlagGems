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
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(1, 3, 8, 8, 8), (2, 8, 16, 16, 16), (4, 16, 32, 32, 32), (8, 32, 64, 64, 64)]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    grid = (dims[0], dims[2] // 2, dims[3] // 2, dims[4] // 2, 3)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "grid": list(grid)},
        params={"interpolation_mode": 0, "padding_mode": 0, "align_corners": False},
        builder_args=(dims, grid),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, grid_shape = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    grid = torch.randn(grid_shape, dtype=dtype, device=device) * 1.5
    return inp, grid, 0, 0, False


class ScopedBenchmark(OperatorBenchmark):
    def build_inputs(self, case):
        if flag_gems.vendor_name == "kunlunxin" and case.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            # Keep the original catalog/IDs, skipping only unsupported raw
            # reference dtypes. No CPU/F32 timing substitutes for their latency.
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("Kunlunxin native grid_sampler_3d supports float32 inputs only")
        return super().build_inputs(case)

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)


@pytest.mark.grid_sampler_3d
def test_grid_sampler_3d():
    ScopedBenchmark(
        op_name="grid_sampler_3d",
        torch_op=torch.ops.aten.grid_sampler_3d.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
