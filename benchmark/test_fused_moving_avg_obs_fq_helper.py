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

SHAPES = [(1 << 26,), (1 << 28,), (2048, 16384), (4096, 16384)]


class FusedMovingAvgObsFqBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "QAT tensor shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    per_channel = len(dims) > 1
    channels = dims[0] if per_channel else 1
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "qparams": [channels]},
        params={"per_channel": per_channel, "ch_axis": 0},
        builder_args=(dims, per_channel, channels),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    shape, per_channel, channels = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    observer_on = torch.tensor(1, dtype=torch.long, device=device)
    fake_quant_on = torch.tensor(1, dtype=torch.long, device=device)
    running_min = torch.full((channels,), -0.5, dtype=torch.float32, device=device)
    running_max = torch.full((channels,), 0.5, dtype=torch.float32, device=device)
    scale = torch.ones((channels,), dtype=torch.float32, device=device)
    zero_point = torch.zeros((channels,), dtype=torch.int32, device=device)
    return (
        inp,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        0.01,
        0,
        255,
        0,
        per_channel,
        False,
    )


@pytest.mark.fused_moving_avg_obs_fq_helper
@pytest.mark.underscore_fused_moving_avg_obs_fq_helper
def test_fused_moving_avg_obs_fq_helper():
    FusedMovingAvgObsFqBenchmark(
        op_name="_fused_moving_avg_obs_fq_helper",
        torch_op=torch.ops.aten._fused_moving_avg_obs_fq_helper,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
        fresh_inputs=True,
    ).run()
