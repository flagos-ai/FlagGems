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
import torch.nn.functional as F

import flag_gems

from . import base, consts

CTC_DTYPES = [torch.float32, torch.float16]


def _ctc_loss_reference(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    original_dtype = log_probs.dtype
    work_log_probs = log_probs
    if work_log_probs.dtype in (torch.float16, torch.bfloat16):
        work_log_probs = work_log_probs.float()
    out = F.ctc_loss(
        work_log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    return out.to(original_dtype) if out.dtype != original_dtype else out


def _make_targets(batch, max_target, classes, device, target_layout):
    target_lengths = torch.empty(batch, device=device, dtype=torch.long)
    padded = torch.zeros(batch, max_target, device=device, dtype=torch.long)
    pieces = []
    for row in range(batch):
        length = max(1, max_target - (row % 5))
        target_lengths[row] = length
        values = (torch.arange(length, device=device, dtype=torch.long) + row) % (
            classes - 1
        )
        values = values + 1
        padded[row, :length] = values
        pieces.append(values)

    if target_layout == "padded":
        targets = padded
    else:
        targets = torch.cat(pieces)
    return targets, target_lengths


def ctc_loss_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"log_probs": [shape[0], shape[1], shape[2]]},
        params={"layout": "padded", "reduction": "mean"},
        builder_args=(shape, "padded", "mean"),
    )
    yield base.BenchmarkCasePlan(
        shape={"log_probs": [shape[0], shape[1], shape[2]]},
        params={"layout": "concatenated", "reduction": "mean"},
        builder_args=(shape, "concatenated", "mean"),
    )
    if base.Config.bench_level.value == consts.BenchLevel.COMPREHENSIVE.value:
        yield base.BenchmarkCasePlan(
            shape={"log_probs": [shape[0], shape[1], shape[2]]},
            params={"layout": "concatenated", "reduction": "sum"},
            builder_args=(shape, "concatenated", "sum"),
        )


def ctc_loss_build_inputs_fn(plan, dtype, device):
    shape, target_layout, reduction = plan.builder_args
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    input_lengths = torch.full((batch,), t_steps, dtype=torch.long, device=device)

    targets, target_lengths = _make_targets(
        batch, max_target, classes, device, target_layout
    )
    return (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        {"blank": 0, "reduction": reduction, "zero_infinity": False},
    )


class CtcLossBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def set_more_shapes(self):
        return []


@pytest.mark.ctc_loss
def test_ctc_loss():
    bench = CtcLossBenchmark(
        op_name="ctc_loss",
        case_fn=ctc_loss_case_fn,
        build_inputs_fn=ctc_loss_build_inputs_fn,
        torch_op=_ctc_loss_reference,
        dtypes=CTC_DTYPES,
    )
    bench.run()


@pytest.mark.ctc_loss_backward
def test_ctc_loss_backward():
    bench = CtcLossBenchmark(
        op_name="ctc_loss",
        case_fn=ctc_loss_case_fn,
        build_inputs_fn=ctc_loss_build_inputs_fn,
        torch_op=_ctc_loss_reference,
        dtypes=CTC_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.ctc_loss)
    bench.run()
