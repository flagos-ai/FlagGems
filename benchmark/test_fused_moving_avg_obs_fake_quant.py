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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(1 << 27,), (1 << 28,), (1 << 29,), (1024, 1024, 1024), (1 << 30,)]


class FusedMovingAvgObsFakeQuantBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "per-tensor QAT activation shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "qparams": [1]},
        params={"ch_axis": -1, "per_row_fake_quant": False},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    return (
        inp,
        torch.tensor([1], dtype=torch.int64, device=device),
        torch.tensor([1], dtype=torch.int64, device=device),
        torch.tensor([-0.5], dtype=torch.float32, device=device),
        torch.tensor([0.5], dtype=torch.float32, device=device),
        torch.ones(1, dtype=torch.float32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        0.01,
        0,
        255,
        -1,
        False,
        False,
    )


@pytest.mark.fused_moving_avg_obs_fake_quant
def test_fused_moving_avg_obs_fake_quant():
    FusedMovingAvgObsFakeQuantBenchmark(
        op_name="fused_moving_avg_obs_fake_quant",
        torch_op=torch.fused_moving_avg_obs_fake_quant,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
        fresh_inputs=True,
    ).run()
