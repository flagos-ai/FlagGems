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
from .generated_operator_utils import OperatorBenchmark

# CTC loss is implemented for the single and double precision types.
CTC_DTYPES = [torch.float32, torch.float64]


def ctc_loss_internal_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"log_probs": [shape[0], shape[1], shape[2]]},
        builder_args=(shape,),
    )


def ctc_loss_internal_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    t_steps, batch, classes, max_target = shape
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin native CTC does not preserve float64 outputs")

    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    targets = torch.randint(
        1, classes, (batch, max_target), dtype=torch.long, device=device
    )
    input_lengths = [t_steps] * batch
    target_lengths = [max_target] * batch

    return log_probs, targets, input_lengths, target_lengths, {"blank": 0}


def _ctc_loss_build_inputs_fn(plan, dtype, device):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double values")
    return ctc_loss_internal_build_inputs_fn(plan, dtype, device)


class CtcLossInternalBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=self.DEFAULT_SHAPES)

    def set_more_shapes(self):
        return []


@pytest.mark.underscore_ctc_loss
def test_perf__ctc_loss():
    bench = CtcLossInternalBenchmark(
        op_name="_ctc_loss",
        case_fn=ctc_loss_internal_case_fn,
        build_inputs_fn=_ctc_loss_build_inputs_fn,
        torch_op=torch.ops.aten._ctc_loss,
        dtypes=CTC_DTYPES,
    )
    bench.run()


def ctc_loss_internal_out_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"log_probs": [shape[0], shape[1], shape[2]]},
        builder_args=(shape,),
    )


def ctc_loss_internal_out_build_inputs_fn(plan, dtype, device):
    """Same inputs as the base variant, plus the two preallocated outputs."""
    (shape,) = plan.builder_args
    t_steps, batch, classes, max_target = shape

    *args, kwargs = ctc_loss_internal_build_inputs_fn(plan, dtype, device)
    # _ctc_loss returns (neg_log_likelihood, log_alpha), so out0 is per
    # batch entry and out1 spans the (T, 2S + 1) alpha lattice.
    out0 = torch.empty(batch, dtype=dtype, device=device)
    out1 = torch.empty(batch, t_steps, 2 * max_target + 1, dtype=dtype, device=device)
    return (*args, {**kwargs, "out0": out0, "out1": out1})


@pytest.mark.ctc_loss_out
def test_perf__ctc_loss_out():
    bench = CtcLossInternalBenchmark(
        op_name="_ctc_loss_out",
        case_fn=ctc_loss_internal_out_case_fn,
        build_inputs_fn=ctc_loss_internal_out_build_inputs_fn,
        torch_op=torch.ops.aten._ctc_loss.out,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems.ops._ctc_loss_out)
    bench.run()
