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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark
from .test_fused_rms_norm import _RmsBackwardBaseline, _torch_fused_rms_norm_backward

SHAPES = [(4, 64), (8, 256), (16, 1024)]


class OperatorCaseBenchmark(_RmsBackwardBaseline, OperatorBenchmark):
    baseline_benchmark = "benchmark/test_fused_rms_norm_backward.py"

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "operator-specific shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(shape[-1], dtype=dtype, device=device)
    rstd = torch.rsqrt(inp.to(torch.float32).pow(2).mean(dim=-1) + 1e-5)
    return grad, inp, [shape[-1]], rstd, weight, [True, True]


@pytest.mark.fused_rms_norm_backward
@pytest.mark.underscore_fused_rms_norm_backward
def test_fused_rms_norm_backward():
    OperatorCaseBenchmark(
        op_name="_fused_rms_norm_backward",
        torch_op=_torch_fused_rms_norm_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
