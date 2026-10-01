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

"""Benchmark for ``aten::_use_cudnn_ctc_loss``.

The op is a boolean capability predicate: it inspects rank, dtype, device and
the two length lists, then returns a Python bool without allocating anything, so
what is measured is dispatch + predicate cost for the accepted float32 / rank-3
log_probs with CPU int32 targets, in both native call forms (Python ``int[]``
lengths and tensor lengths). A rejected dtype does not open a second code path,
so only float32 log_probs is timed.

``torch_op`` is the ATen reference (the perf comparison baseline) and ``gems_op``
is the FlagGems candidate resolved through ``--override``; both receive the same
positional operands and the same ``blank`` keyword. Cases come from ``case_fn``
only, so ``--list-cases`` and ``--case-id`` replay allocate no tensors and call no
operator, and the plans used for listing are the same ones used for execution.
"""

import pytest
import torch

import flag_gems

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

# Rank-3 (T, N, C) log_probs: every input_length equals T and every target length
# stays below 256, which keeps the predicate on its accepted path.
CTC_SHAPES = [
    (128, 8, 16),
    (256, 16, 32),
    (512, 32, 64),
    (1024, 64, 128),
    (2048, 8, 64),
]

_TARGET_LENGTH = 8
_LENGTHS_FORMS = ("list", "tensor")


def _case_fn(shape, dtype):
    del dtype
    log_probs_shape = tuple(shape)
    batch = log_probs_shape[1] if len(log_probs_shape) == 3 else 1
    time_extent = log_probs_shape[0] if log_probs_shape else 0
    input_lengths = [time_extent] * batch
    target_lengths = [_TARGET_LENGTH] * batch
    for form in _LENGTHS_FORMS:
        yield base.BenchmarkCasePlan(
            shape={
                "log_probs": list(log_probs_shape),
                "targets": [batch * _TARGET_LENGTH],
            },
            params={"blank": 0, "lengths_form": form},
            builder_args=(log_probs_shape, input_lengths, target_lengths, form),
        )


def _build_inputs_fn(plan, dtype, device):
    log_probs_shape, input_lengths, target_lengths, form = plan.builder_args
    log_probs = utils.generate_tensor_input(log_probs_shape, dtype, device)
    # targets must be rank-1 int32 and, for the int[] form, on CPU: the native
    # predicate tests targets.device().type() == at::kCPU.
    targets = torch.zeros(
        len(target_lengths) * _TARGET_LENGTH, dtype=torch.int32, device="cpu"
    )
    if form == "tensor":
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device=device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32, device=device)
    # unpack_to_args_kwargs keeps tensors/lists positional and merges a trailing
    # dict into the call kwargs, so the operands are returned flat -- a nested
    # args tuple would be passed as a single operand.
    return (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        {"blank": plan.params["blank"]},
    )


class UseCudnnCtcLossBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Other ranks are valid predicate workloads returning False. Retain
        # their requested shapes alongside the rank-3 cuDNN eligibility cases.
        self.shapes = list(
            dict.fromkeys([tuple(shape) for shape in self.shapes] + CTC_SHAPES)
        )


@pytest.mark.use_cudnn_ctc_loss
def test__use_cudnn_ctc_loss():
    bench = UseCudnnCtcLossBenchmark(
        op_name="_use_cudnn_ctc_loss",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._use_cudnn_ctc_loss,
        gems_op=getattr(flag_gems, "_use_cudnn_ctc_loss", None),
        dtypes=[torch.float32],
    )
    bench.run()
