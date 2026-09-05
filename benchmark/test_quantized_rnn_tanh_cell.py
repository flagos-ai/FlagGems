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


def torch_quantized_rnn_tanh_cell_ref(
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
    """Pure-PyTorch float reference for the quantized tanh RNN cell.

    The ATen ``torch.quantized_rnn_tanh_cell`` baseline is CPU/FBGEMM-only and
    segfaults on CUDA, so this float reference (which consumes the dequantized
    weights directly) serves as the GPU-runnable baseline for benchmarking.
    """
    return torch.tanh(input @ w_ih.t() + b_ih + hx @ w_hh.t() + b_hh)


def quantized_rnn_tanh_cell_input_fn(shape, dtype, device):
    batch, input_size, hidden_size = shape
    input = torch.randn(batch, input_size, dtype=dtype, device=device)
    hx = torch.randn(batch, hidden_size, dtype=dtype, device=device)
    w_ih = torch.randn(hidden_size, input_size, dtype=dtype, device=device)
    w_hh = torch.randn(hidden_size, hidden_size, dtype=dtype, device=device)
    b_ih = torch.randn(hidden_size, dtype=dtype, device=device)
    b_hh = torch.randn(hidden_size, dtype=dtype, device=device)
    # Packing artifacts are unused by the GPU kernels; provide zero placeholders
    # of the expected dtypes to match the ATen schema.
    packed_ih = torch.zeros(input_size, dtype=torch.int8, device=device)
    packed_hh = torch.zeros(hidden_size, dtype=torch.int8, device=device)
    col_offsets_ih = torch.zeros(hidden_size, dtype=torch.int32, device=device)
    col_offsets_hh = torch.zeros(hidden_size, dtype=torch.int32, device=device)
    yield input, hx, w_ih, w_hh, b_ih, b_hh, {
        "packed_ih": packed_ih,
        "packed_hh": packed_hh,
        "col_offsets_ih": col_offsets_ih,
        "col_offsets_hh": col_offsets_hh,
        "scale_ih": 1.0,
        "scale_hh": 1.0,
        "zero_point_ih": 0,
        "zero_point_hh": 0,
    }


class QuantizedRnnTanhCellBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Override the yaml-driven shape loading: this op uses (batch,
        # input_size, hidden_size) 3-tuples that the generic core_shapes.yaml
        # does not describe.
        self.shapes = [
            (2, 8, 8),
            (4, 16, 32),
            (8, 64, 128),
            (16, 128, 256),
            (32, 256, 512),
        ]


@pytest.mark.quantized_rnn_tanh_cell
def test_quantized_rnn_tanh_cell():
    bench = QuantizedRnnTanhCellBenchmark(
        op_name="quantized_rnn_tanh_cell",
        torch_op=torch_quantized_rnn_tanh_cell_ref,
        input_fn=quantized_rnn_tanh_cell_input_fn,
        gems_op=flag_gems.quantized_rnn_tanh_cell,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
