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

# Shapes for the quantized LSTM cell: (batch_size, input_size, hidden_size).
LSTM_CELL_SHAPES = [
    (1, 4, 4),
    (2, 8, 6),
    (4, 16, 12),
    (8, 32, 16),
    (16, 64, 32),
    (32, 128, 64),
]


def torch_lstm_cell(
    input,
    hx,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    packed_ih,
    packed_hh,
    col_offsets_ih,
    col_offsets_hh,
    scale_ih,
    scale_hh,
    zero_point_ih,
    zero_point_hh,
):
    """Reference fp32 LSTM cell used as the torch latency baseline.

    The native ``torch.quantized_lstm_cell`` composite op requires the
    FBGEMM int8 weight kernels which are unavailable on this build, so the
    torch baseline is computed from the dequantized (fp32) weights via plain
    matmuls. This is the same arithmetic the Triton kernel performs.
    """
    h_state, c_state = hx
    gates = input @ w_ih.t() + b_ih + h_state @ w_hh.t() + b_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    cy = f * c_state + i * g
    hy = o * torch.tanh(cy)
    return hy, cy


def gems_lstm_cell(
    input,
    hx,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    packed_ih,
    packed_hh,
    col_offsets_ih,
    col_offsets_hh,
    scale_ih,
    scale_hh,
    zero_point_ih,
    zero_point_hh,
):
    """FlagGems entry: ``torch.quantized_lstm_cell`` is intercepted by the
    Triton kernel registered for the ``aten::quantized_lstm_cell`` op.

    The registration is established once (see ``test_quantized_lstm_cell``)
    via ``flag_gems.only_enable`` so it persists for the whole benchmark,
    avoiding the per-call ``use_gems`` registration overhead inside the
    timed loop."""
    return torch.quantized_lstm_cell(
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zero_point_ih,
        zero_point_hh,
    )


class QuantizedLstmCellBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LSTM_CELL_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch_size, input_size, hidden_size = shape
            input = torch.randn(
                batch_size, input_size, dtype=cur_dtype, device=self.device
            )
            h0 = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            c0 = torch.randn(
                batch_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            w_ih = torch.randn(
                4 * hidden_size, input_size, dtype=cur_dtype, device=self.device
            )
            w_hh = torch.randn(
                4 * hidden_size, hidden_size, dtype=cur_dtype, device=self.device
            )
            b_ih = torch.randn(4 * hidden_size, dtype=cur_dtype, device=self.device)
            b_hh = torch.randn(4 * hidden_size, dtype=cur_dtype, device=self.device)
            packed_ih = torch.zeros(
                4 * hidden_size,
                input_size,
                dtype=torch.int8,
                device=self.device,
            )
            packed_hh = torch.zeros(
                4 * hidden_size,
                hidden_size,
                dtype=torch.int8,
                device=self.device,
            )
            col_offsets_ih = torch.zeros(
                4 * hidden_size, dtype=torch.int32, device=self.device
            )
            col_offsets_hh = torch.zeros(
                4 * hidden_size, dtype=torch.int32, device=self.device
            )
            yield (
                input,
                (h0, c0),
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih,
                col_offsets_hh,
                0.1,
                0.1,
                0,
                0,
            )


@pytest.mark.quantized_lstm_cell
def test_quantized_lstm_cell():
    # Register the FlagGems Triton kernel for quantized_lstm_cell once so it
    # persists for the whole benchmark (the use_gems context manager would
    # otherwise re-register every op on each call, dominating the latency).
    flag_gems.only_enable(include=["quantized_lstm_cell"])
    bench = QuantizedLstmCellBenchmark(
        op_name="quantized_lstm_cell",
        torch_op=torch_lstm_cell,
        gems_op=gems_lstm_cell,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
