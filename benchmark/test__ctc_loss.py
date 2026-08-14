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


def _ctc_loss_input_fn(shape, dtype, device):
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    input_lengths = torch.full((batch,), t_steps, dtype=torch.long, device=device)

    # 1D concatenated targets hit the default overload.
    targets, target_lengths = _make_targets(
        batch, max_target, classes, device, "concatenated"
    )
    yield log_probs, targets, input_lengths, target_lengths, {
        "blank": 0,
        "zero_infinity": False,
    }

    if base.Config.bench_level.value == consts.BenchLevel.COMPREHENSIVE.value:
        # 2D padded targets hit the Tensor overload.
        targets, target_lengths = _make_targets(
            batch, max_target, classes, device, "padded"
        )
        yield log_probs, targets, input_lengths, target_lengths, {
            "blank": 0,
            "zero_infinity": False,
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
        # core_shapes.yaml has no `_ctc_loss` entry, so the base implementation
        # falls back along the MRO and picks up generic 1-D/2-D shapes. CTC needs
        # (T, N, C, S) 4-tuples, so pin them instead of consulting the yaml.
        self.shapes = [tuple(shape) for shape in self.DEFAULT_SHAPES]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return []


@pytest.mark.ctc_loss_internal
def test_perf__ctc_loss():
    bench = CtcLossInternalBenchmark(
        op_name="_ctc_loss",
        input_fn=_ctc_loss_input_fn,
        torch_op=torch.ops.aten._ctc_loss,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems._ctc_loss)
    bench.run()
