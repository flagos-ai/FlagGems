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

# CTC loss backward is implemented for the single and double precision types.
BACKWARD_DTYPES = [torch.float32, torch.float64]


def ctc_loss_backward_input_fn(shape, dtype, device):
    """Yield backward inputs, running the forward pass once to obtain them.

    _ctc_loss_backward consumes neg_log_likelihood and log_alpha from the
    forward pass, so the forward is run here in setup rather than inside the
    timed region.
    """
    t_steps, batch, classes, max_target = shape

    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    targets = torch.randint(
        1, classes, (batch, max_target), dtype=torch.long, device=device
    )
    input_lengths = [t_steps] * batch
    target_lengths = [max_target] * batch

    neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
        log_probs, targets, input_lengths, target_lengths
    )
    grad_output = torch.randn_like(neg_log_likelihood)

    yield (
        grad_output,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        {"blank": 0},
    )


class CtcLossBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def set_more_shapes(self):
        return []


@pytest.mark.ctc_loss_backward
def test_perf__ctc_loss_backward():
    bench = CtcLossBackwardBenchmark(
        op_name="_ctc_loss_backward",
        input_fn=ctc_loss_backward_input_fn,
        torch_op=torch.ops.aten._ctc_loss_backward,
        dtypes=BACKWARD_DTYPES,
    )
    bench.set_gems(flag_gems.ops._ctc_loss_backward)
    bench.run()


def ctc_loss_backward_out_input_fn(shape, dtype, device):
    """Same inputs as the base variant, plus a preallocated ``out`` tensor."""
    for item in ctc_loss_backward_input_fn(shape, dtype, device):
        *args, kwargs = item
        out = torch.empty_like(args[1])
        yield (*args, {**kwargs, "out": out})


@pytest.mark.ctc_loss_backward_out
def test_perf__ctc_loss_backward_out():
    bench = CtcLossBackwardBenchmark(
        op_name="_ctc_loss_backward_out",
        input_fn=ctc_loss_backward_out_input_fn,
        torch_op=torch.ops.aten._ctc_loss_backward.out,
        dtypes=BACKWARD_DTYPES,
    )
    bench.set_gems(flag_gems.ops._ctc_loss_backward_out)
    bench.run()
