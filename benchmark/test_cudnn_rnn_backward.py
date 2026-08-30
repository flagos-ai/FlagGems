# Copyright 2026, The FlagOS Contributors.
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

# Hardcoded shapes: cuDNN RNN backward requires
# (seq_len, batch, input_size, hidden_size) shapes; CI --level core overrides
# GenericBenchmark shapes, so we use a custom Benchmark subclass with a
# set_shapes override.
CUDNN_RNN_SHAPES = [
    (3, 2, 4, 5),
    (5, 4, 8, 8),
    (4, 3, 16, 16),
    (6, 8, 32, 32),
    (8, 16, 64, 64),
    (10, 32, 128, 128),
]


class CudnnRnnBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CUDNN_RNN_SHAPES

    def get_input_iter(self, cur_dtype):
        gates = 4  # LSTM
        for shape in self.shapes:
            seq_len, batch, input_size, hidden_size = shape
            input = torch.randn(
                seq_len, batch, input_size, dtype=cur_dtype, device=self.device
            )
            hx = torch.randn(1, batch, hidden_size, dtype=cur_dtype, device=self.device)
            cx = torch.randn(1, batch, hidden_size, dtype=cur_dtype, device=self.device)
            w_ih = torch.randn(
                gates * hidden_size, input_size, dtype=cur_dtype, device=self.device
            )
            w_hh = torch.randn(
                gates * hidden_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            b_ih = torch.randn(gates * hidden_size, dtype=cur_dtype, device=self.device)
            b_hh = torch.randn(gates * hidden_size, dtype=cur_dtype, device=self.device)
            weight = [w_ih, w_hh, b_ih, b_hh]

            # Forward pass to obtain the cuDNN reserve / weight_buf.
            output, hy, cy, reserve, weight_buf = torch.ops.aten._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,
                hidden_size,
                0,
                1,
                False,
                0.0,
                True,
                False,
                [],
                None,
            )
            grad_output = torch.randn_like(output)
            grad_hy = torch.randn_like(hy)
            grad_cy = torch.randn_like(cy)

            yield (
                input,
                weight,
                4,
                weight_buf,
                hx,
                cx,
                output,
                grad_output,
                grad_hy,
                grad_cy,
                2,
                hidden_size,
                0,
                1,
                False,
                0.0,
                True,
                False,
                [],
                None,
                reserve,
                [True, True, True, True],
            )


@pytest.mark.cudnn_rnn_backward
def test_cudnn_rnn_backward():
    bench = CudnnRnnBackwardBenchmark(
        op_name="cudnn_rnn_backward",
        torch_op=torch.ops.aten._cudnn_rnn_backward,
        # cuDNN RNN forward is CUDA-only; use float32 for stable timing.
        dtypes=[torch.float32],
    )
    bench.run()
