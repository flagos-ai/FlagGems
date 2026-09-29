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

# The CUDA dtype table rejects bfloat16, so the measurable set drops it from the
# framework float list; float64 is included when the backend provides it.
FP64_SUPPORTED = bool(getattr(flag_gems.runtime.device, "support_fp64", False))
BENCH_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16] + (
    [torch.float64] if FP64_SUPPORTED else []
)

# (N, C, H, W) grid sizes; C is ignored metadata, so N/H/W carry the work.
DEFAULT_SHAPES = [
    (64, 3, 512, 512),
    (256, 3, 512, 512),
    (64, 3, 1024, 1024),
]


def _checked_size(shape):
    """Return a usable (N, C, H, W) descriptor or report an invalid row."""
    values = tuple(shape)
    if len(values) != 4 or not all(isinstance(value, int) for value in values):
        raise ValueError(
            f"cudnn_affine_grid_generator expects (N, C, H, W) integer sizes, "
            f"got {shape!r}"
        )
    n, _c, h, w = values
    if n < 1 or h < 1 or w < 1:
        raise ValueError(
            f"cudnn_affine_grid_generator requires positive N, H and W, got {shape!r}"
        )
    return values


def _case_fn(shape, dtype):
    # One plan per (N, C, H, W) row; theta is always (N, 2, 3).
    del dtype
    n, c, h, w = shape
    yield base.BenchmarkCasePlan(
        shape={"size": [n, c, h, w], "theta": [n, 2, 3]},
        params={},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    n, c, h, w = plan.builder_args[0]
    # theta is built explicitly: the shared generator has no float64 branch.
    theta = torch.randn(n, 2, 3, dtype=torch.float32, device=device).to(dtype)
    return theta, n, c, h, w


class CudnnAffineGridGeneratorBenchmark(OperatorBenchmark):
    """Two-phase benchmark over (N, C, H, W) sizes and the CUDA dtype set."""

    DEFAULT_SHAPE_DESC = "N, C, H, W"

    def set_shapes(self, shape_file_path=None):
        # OperatorBenchmark default-shape contract: a shape-file entry for this
        # operator or class wins, otherwise DEFAULT_SHAPES apply. Requested rows
        # are used as given; an unusable descriptor is an error, not a swap.
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_SHAPES)
        self.shapes = [_checked_size(shape) for shape in self.shapes]


@pytest.mark.cudnn_affine_grid_generator
def test_cudnn_affine_grid_generator():
    bench = CudnnAffineGridGeneratorBenchmark(
        op_name="cudnn_affine_grid_generator",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cudnn_affine_grid_generator,
        gems_op=getattr(flag_gems, "cudnn_affine_grid_generator", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
