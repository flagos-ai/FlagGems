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


class FusedMovingAvgObsFqBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Representative QAT tensor sizes: large per-tensor activations and
        # per-channel conv/linear weight tensors. Small tensors are dominated by
        # fixed kernel-launch overhead and are not a realistic QAT workload.
        return [
            (1 << 26,),  # 64M per-tensor activation
            (1 << 28,),  # 256M per-tensor activation
            (2048, 16384),  # per-channel weights
            (4096, 16384),  # per-channel weights
        ]


def fused_moving_avg_obs_fq_helper_input_fn(shape, dtype, device):
    # aten._fused_moving_avg_obs_fq_helper requires a float32 `self`; the qparam
    # channel count is 1 for per-tensor and shape[0] for per-channel.
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    per_channel = inp.ndim > 1
    n = shape[0] if per_channel else 1

    observer_on = torch.tensor(1, dtype=torch.long, device=device)
    fake_quant_on = torch.tensor(1, dtype=torch.long, device=device)
    running_min = torch.full((n,), -0.5, dtype=torch.float32, device=device)
    running_max = torch.full((n,), 0.5, dtype=torch.float32, device=device)
    scale = torch.ones((n,), dtype=torch.float32, device=device)
    zero_point = torch.zeros((n,), dtype=torch.int32, device=device)

    yield (
        inp,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        0.01,  # averaging_const
        0,  # quant_min
        255,  # quant_max
        0,  # ch_axis
        per_channel,  # per_row_fake_quant
        False,  # symmetric_quant
    )


def torch_fused_moving_avg_obs_fq_helper(
    inp,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant,
    symmetric_quant,
):
    return torch.ops.aten._fused_moving_avg_obs_fq_helper(
        inp,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        averaging_const,
        quant_min,
        quant_max,
        ch_axis,
        per_row_fake_quant,
        symmetric_quant,
    )


@pytest.mark.fused_moving_avg_obs_fq_helper
def test_fused_moving_avg_obs_fq_helper():
    bench = FusedMovingAvgObsFqBenchmark(
        input_fn=fused_moving_avg_obs_fq_helper_input_fn,
        op_name="fused_moving_avg_obs_fq_helper",
        torch_op=torch_fused_moving_avg_obs_fq_helper,
        dtypes=[torch.float32],
    )
    bench.run()
