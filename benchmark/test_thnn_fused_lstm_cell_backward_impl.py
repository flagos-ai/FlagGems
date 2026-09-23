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

# Hardcoded shapes: LSTM cell backward requires (batch, hidden) shapes;
# CI --level core overrides GenericBenchmark shapes,
# so we use a custom Benchmark subclass with set_shapes override.
LSTM_SHAPES = [
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),
    (1, 64),
    (2, 4),
    (2, 8),
    (2, 16),
    (2, 32),
    (2, 64),
    (4, 4),
    (4, 8),
    (4, 16),
    (4, 32),
    (4, 64),
    (8, 4),
    (8, 8),
    (8, 16),
    (8, 32),
    (8, 64),
    (16, 4),
    (16, 8),
    (16, 16),
    (16, 32),
    (16, 64),
]


class LSTMCellBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LSTM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch_size, hidden_size = shape
            input_gates = torch.randn(
                batch_size, 4 * hidden_size, dtype=cur_dtype, device=self.device
            )
            hidden_gates = torch.randn(
                batch_size, 4 * hidden_size, dtype=cur_dtype, device=self.device
            )
            cx = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            input_bias = torch.zeros(
                4 * hidden_size, dtype=cur_dtype, device=self.device
            )
            hidden_bias = torch.randn(
                4 * hidden_size, dtype=cur_dtype, device=self.device
            )

            # Forward pass to get workspace
            hx, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
                input_gates, hidden_gates, cx, input_bias, hidden_bias
            )

            # Create gradient tensors
            grad_hy = torch.randn_like(hx)
            grad_cy = torch.randn_like(cy)

            yield grad_hy, grad_cy, cx, cy, workspace, True


@pytest.mark.thnn_fused_lstm_cell_backward_impl
def test_thnn_fused_lstm_cell_backward_impl():
    bench = LSTMCellBackwardBenchmark(
        op_name="thnn_fused_lstm_cell_backward_impl",
        torch_op=torch.ops.aten._thnn_fused_lstm_cell_backward_impl,
        # Half/BFloat16 not supported by the underlying aten operator
        dtypes=[torch.float32],
    )
    bench.run()
