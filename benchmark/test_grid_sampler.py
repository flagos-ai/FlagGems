# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# (input shape, grid shape, interpolation_mode, padding_mode, align_corners).
# The shared shape files carry no grid_sampler entry and the generic default
# shapes have the wrong rank, so each row pins the operand pair and the three mode
# arguments; an explicit --shape_file still overrides them. Every row is a valid
# native call, and the 5-D rows keep modes 0/1 because 3-D sampling has no
# bicubic path.
GRID_SAMPLER_CASES = [
    ((2, 3, 64, 64), (2, 32, 32, 2), 0, 0, False),
    ((4, 8, 128, 128), (4, 64, 64, 2), 0, 0, False),
    ((8, 16, 256, 256), (8, 128, 128, 2), 1, 1, True),
    ((16, 32, 128, 128), (16, 64, 64, 2), 2, 2, False),
    ((2, 4, 32, 32, 32), (2, 16, 16, 16, 3), 0, 0, False),
    ((4, 8, 64, 64, 64), (4, 32, 32, 32, 3), 1, 2, True),
]

GRID_SAMPLER_COMPREHENSIVE_CASES = [
    ((4, 16, 512, 512), (4, 128, 128, 2), 0, 0, False),
    ((8, 8, 1024, 1024), (8, 256, 256, 2), 1, 1, True),
    ((1, 32, 64, 64, 64), (1, 32, 32, 32, 3), 0, 0, False),
    ((8, 16, 32, 32, 32), (8, 16, 16, 16, 3), 1, 2, True),
]


def _case_fn(shape, dtype):
    del dtype
    inp_shape, grid_shape, interpolation_mode, padding_mode, align_corners = shape
    yield base.BenchmarkCasePlan(
        shape={"input": list(inp_shape), "grid": list(grid_shape)},
        params={
            "interpolation_mode": interpolation_mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
        },
        builder_args=(inp_shape, grid_shape),
    )


def _build_inputs_fn(plan, dtype, device):
    inp_shape, grid_shape = plan.builder_args
    inp = utils.generate_tensor_input(inp_shape, dtype, device)
    # Sample coordinates are normalized to [-1, 1] by definition rather than
    # drawn from the image generator; the range is slightly wider so the border,
    # reflection and zeros padding paths are actually taken.
    grid = (torch.rand(grid_shape, dtype=torch.float32, device=device) * 2.5 - 1.25).to(
        dtype
    )
    return inp, grid, dict(plan.params)


class GridSamplerBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        rows = []
        for shape in self.shapes:
            if shape and isinstance(shape[0], (tuple, list)):
                row = tuple(shape)
            elif len(shape) in (4, 5):
                image = tuple(shape)
                grid = (image[0],) + image[2:] + (len(image) - 2,)
                row = (image, grid, 0, 0, False)
            else:
                # Native grid_sampler only accepts NCHW and NCDHW operands.
                continue
            if row not in rows:
                rows.append(row)
        for row in GRID_SAMPLER_CASES:
            if row not in rows:
                rows.append(row)
        self.shapes = rows

    def set_more_shapes(self):
        return list(super().set_more_shapes()) + GRID_SAMPLER_COMPREHENSIVE_CASES


@pytest.mark.grid_sampler
def test_grid_sampler():
    bench = GridSamplerBenchmark(
        op_name="grid_sampler",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.grid_sampler,
        gems_op=getattr(flag_gems, "grid_sampler", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
