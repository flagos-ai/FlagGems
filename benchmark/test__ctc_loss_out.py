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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

CTC_DTYPES = [torch.float32, torch.float64]


class CtcLossOutBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def set_shapes(self, shape_file_path=None):
        # This recipe needs T/N/C/S, not a backend's generic matrix shapes.
        super().set_shapes(shape_file_path, default_shapes=self.DEFAULT_SHAPES)

    def set_more_shapes(self):
        return []


def _case_fn(shape, dtype):
    del dtype
    t_steps, batch, classes, max_target = shape
    yield base.BenchmarkCasePlan(
        shape={
            "log_probs": [t_steps, batch, classes],
            "targets": [batch, max_target],
            "out0": [batch],
            "out1": [batch, t_steps, 2 * max_target + 1],
        },
        params={"blank": 0, "zero_infinity": False},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    t_steps, batch, classes, max_target = shape
    torch.manual_seed(0)
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double values")
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin native CTC does not preserve float64 outputs")
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    targets = torch.randint(1, classes, (batch, max_target), dtype=torch.long, device=device)
    input_lengths = [t_steps] * batch
    target_lengths = [max_target] * batch
    out0 = torch.empty(batch, dtype=dtype, device=device)
    out1 = torch.empty(batch, t_steps, 2 * max_target + 1, dtype=dtype, device=device)
    return (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        {"blank": 0, "zero_infinity": False, "out0": out0, "out1": out1},
    )


@pytest.mark.underscore_ctc_loss_out
def test_perf__ctc_loss_out():
    CtcLossOutBenchmark(
        op_name="_ctc_loss_out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._ctc_loss.out,
        dtypes=CTC_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
