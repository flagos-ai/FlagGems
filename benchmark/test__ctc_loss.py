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

# aten::_ctc_loss only has a float32 CUDA kernel, so the torch baseline cannot
# run in half precision.
CTC_DTYPES = [torch.float32]


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

    targets = padded if target_layout == "padded" else torch.cat(pieces)
    return targets, target_lengths


def _base_inputs(shape, dtype, device, target_layout):
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    input_lengths = torch.full((batch,), t_steps, dtype=torch.long, device=device)
    targets, target_lengths = _make_targets(
        batch, max_target, classes, device, target_layout
    )
    return log_probs, targets, input_lengths, target_lengths


def _ctc_loss_input_fn(shape, dtype, device):
    # 1D concatenated targets hit the default overload.
    log_probs, targets, input_lengths, target_lengths = _base_inputs(
        shape, dtype, device, "concatenated"
    )
    yield log_probs, targets, input_lengths, target_lengths, {
        "blank": 0,
        "zero_infinity": False,
    }

    if base.Config.bench_level.value == consts.BenchLevel.COMPREHENSIVE.value:
        yield log_probs, targets, input_lengths, target_lengths, {
            "blank": 0,
            "zero_infinity": True,
        }


def _ctc_loss_tensor_input_fn(shape, dtype, device):
    # 2D padded targets hit the Tensor overload.
    log_probs, targets, input_lengths, target_lengths = _base_inputs(
        shape, dtype, device, "padded"
    )
    yield log_probs, targets, input_lengths, target_lengths, {
        "blank": 0,
        "zero_infinity": False,
    }


def _ctc_loss_out_input_fn(shape, dtype, device):
    # aten::_ctc_loss.out takes int[] lengths, so pass plain Python lists and
    # let the wrapper allocate through the caller-provided out buffers.
    log_probs, targets, input_lengths, target_lengths = _base_inputs(
        shape, dtype, device, "concatenated"
    )
    out0 = torch.empty(0, dtype=log_probs.dtype, device=device)
    out1 = torch.empty(0, dtype=log_probs.dtype, device=device)
    yield log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), {
        "blank": 0,
        "zero_infinity": False,
        "out0": out0,
        "out1": out1,
    }


class CtcLossInternalBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def set_shapes(self, shape_file_path=None):
        # core_shapes.yaml has no entry for the `_ctc_loss` op names, and the
        # generic fallback yields 1D/3D shapes that cannot describe (T, N, C, S).
        self.shapes = [tuple(shape) for shape in self.DEFAULT_SHAPES]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return []


@pytest.mark.underscore_ctc_loss
def test_perf__ctc_loss():
    bench = CtcLossInternalBenchmark(
        op_name="_ctc_loss",
        input_fn=_ctc_loss_input_fn,
        torch_op=torch.ops.aten._ctc_loss,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems._ctc_loss)
    bench.run()


@pytest.mark.underscore_ctc_loss
def test_perf__ctc_loss_tensor():
    bench = CtcLossInternalBenchmark(
        op_name="_ctc_loss.Tensor",
        input_fn=_ctc_loss_tensor_input_fn,
        torch_op=torch.ops.aten._ctc_loss.Tensor,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems._ctc_loss)
    bench.run()


@pytest.mark.underscore_ctc_loss
def test_perf__ctc_loss_out():
    bench = CtcLossInternalBenchmark(
        op_name="_ctc_loss.out",
        input_fn=_ctc_loss_out_input_fn,
        torch_op=torch.ops.aten._ctc_loss.out,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems._ctc_loss_out)
    bench.run()
