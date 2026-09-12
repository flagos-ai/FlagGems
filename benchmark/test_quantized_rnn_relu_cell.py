# Copyright 2026 FlagOS Contributors.
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

# (batch, input_size, hidden_size) shapes representative of a single
# quantized RNN (ReLU) cell step.
QUANTIZED_RNN_RELU_CELL_SHAPES = [
    (8, 64, 64),
    (16, 128, 128),
    (32, 256, 256),
    (64, 512, 512),
]


class QuantizedRnnReluCellBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QUANTIZED_RNN_RELU_CELL_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch, input_size, hidden_size = shape
            input = torch.randn(batch, input_size, dtype=cur_dtype, device=self.device)
            hx = torch.randn(batch, hidden_size, dtype=cur_dtype, device=self.device)
            # Quantize fp32 weights to int8 (symmetric, zero_point = 0) as
            # ``fbgemm_linear_quantize_weight`` would.
            w_ih_fp = torch.randn(hidden_size, input_size, dtype=torch.float32)
            w_hh_fp = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
            scale_ih = w_ih_fp.abs().max().item() / 127.0
            scale_hh = w_hh_fp.abs().max().item() / 127.0
            w_ih_int8 = torch.round(w_ih_fp / scale_ih).clamp(-127, 127).to(torch.int8)
            w_hh_int8 = torch.round(w_hh_fp / scale_hh).clamp(-127, 127).to(torch.int8)
            b_ih = torch.randn(hidden_size, dtype=torch.float32, device=self.device)
            b_hh = torch.randn(hidden_size, dtype=torch.float32, device=self.device)
            # The Triton kernel reads the int8 weight directly, so the FBGEMM
            # ``packed_*`` / ``col_offsets_*`` buffers are passed as stand-ins
            # (they do not affect the numeric result of the GPU path).
            packed_ih = w_ih_int8.clone().to(self.device)
            packed_hh = w_hh_int8.clone().to(self.device)
            col_offsets_ih = torch.zeros(
                hidden_size, dtype=torch.int32, device=self.device
            )
            col_offsets_hh = torch.zeros(
                hidden_size, dtype=torch.int32, device=self.device
            )
            yield (
                input,
                hx,
                w_ih_int8.to(self.device),
                w_hh_int8.to(self.device),
                b_ih,
                b_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                scale_ih,
                scale_hh,
                0,  # zero_point_ih
                0,  # zero_point_hh
            )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell():
    # PyTorch has no working CUDA reference for the FBGEMM-packed quantized RNN
    # cell, so both the baseline and the FlagGems path measure the same Triton
    # implementation (the speedup is therefore expected to be ~1.0).
    bench = QuantizedRnnReluCellBenchmark(
        op_name="quantized_rnn_relu_cell",
        torch_op=flag_gems.ops.quantized_rnn_relu_cell,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
