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
from .conftest import Config

CORE_SHAPES = [
    (1, 32, False),
    (8, 64, True),
    (32, 128, False),
    (64, 256, True),
    (128, 512, False),
    (256, 1024, True),
]
COMPREHENSIVE_SHAPES = [
    (512, 1024, False),
    (512, 2048, True),
    (1024, 2048, True),
]
DTYPES = list(consts.FLOAT_DTYPES)
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)


def _thnn_fused_lstm_cell_backward_no_grad(*args):
    with torch.no_grad():
        return torch.ops.aten._thnn_fused_lstm_cell_backward(*args)


class FusedLSTMCellBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_input_iter(self, cur_dtype):
        for batch_size, hidden_size, has_bias in self.shapes:
            input_gates = torch.randn(
                batch_size,
                4 * hidden_size,
                dtype=cur_dtype,
                device=self.device,
            )
            hidden_gates = torch.randn_like(input_gates)
            cx = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            bias = (
                torch.randn(4 * hidden_size, dtype=cur_dtype, device=self.device)
                if has_bias
                else None
            )
            with torch.no_grad():
                hy, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
                    input_gates, hidden_gates, cx, bias, bias
                )
            yield torch.randn_like(hy), torch.randn_like(
                cy
            ), cx, cy, workspace, has_bias


@pytest.mark.thnn_fused_lstm_cell_backward
def test_thnn_fused_lstm_cell_backward():
    bench = FusedLSTMCellBackwardBenchmark(
        op_name="thnn_fused_lstm_cell_backward",
        torch_op=_thnn_fused_lstm_cell_backward_no_grad,
        dtypes=DTYPES,
    )
    bench.run()
