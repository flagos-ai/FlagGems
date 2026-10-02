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

# (batch, input_size, hidden_size) tuples representative of RNN cell usage.
QUANTIZED_RNN_SHAPES = [
    (2, 8, 8),
    (4, 16, 32),
    (8, 64, 128),
    (16, 128, 256),
    (32, 256, 512),
]


def torch_quantized_rnn_tanh_cell_ref(
    input,
    hx,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    **kwargs,
):
    """GPU-runnable float reference for the quantized tanh RNN cell.

    ``aten::quantized_rnn_tanh_cell`` dispatches to CPU-only FBGEMM, so it
    cannot serve as the baseline for a GPU benchmark. The dominant cost of the
    quantized path is the int8 GEMM over the hidden/input reduction, which the
    dequantized-weight float GEMM mirrors shape-for-shape.
    """
    return torch.tanh(input @ w_ih.t() + b_ih + hx @ w_hh.t() + b_hh)


def quantized_rnn_tanh_cell_input_fn(shape, dtype, device):
    """Build one benchmark case with real int8 weights and quantization metadata.

    The weights are int8-quantized per tensor exactly like
    ``aten::make_quantized_cell_params`` does (``fbgemm_linear_quantize_weight``
    + ``fbgemm_pack_quantized_matrix`` + ``CalcColOffsetsTranspose``), and the
    scale / zero-point / column offsets handed to both paths are the real
    quantization parameters of those int8 weights, not placeholders.
    """
    batch, input_size, hidden_size = shape
    gen = torch.Generator().manual_seed(hash((shape, str(dtype))) % (2**31))
    input = torch.randn(batch, input_size, dtype=torch.float32, generator=gen).to(
        device=device
    )
    hx = torch.randn(batch, hidden_size, dtype=torch.float32, generator=gen).to(
        device=device
    )
    w_ih = torch.randn(hidden_size, input_size, dtype=torch.float32, generator=gen).to(
        device=device
    ) / (input_size**0.5)
    w_hh = torch.randn(hidden_size, hidden_size, dtype=torch.float32, generator=gen).to(
        device=device
    ) / (hidden_size**0.5)
    b_ih = torch.randn(hidden_size, dtype=torch.float32, generator=gen).to(
        device=device
    )
    b_hh = torch.randn(hidden_size, dtype=torch.float32, generator=gen).to(
        device=device
    )

    # Real quantization metadata for the int8 weights (host side, matching the
    # CPU/FBGEMM packing API), including non-zero zero-points.
    zero_point_ih, zero_point_hh = 5, -7
    scale_ih = w_ih.detach().cpu().abs().max().item() / 127.0
    scale_hh = w_hh.detach().cpu().abs().max().item() / 127.0
    qw_ih = torch.quantize_per_tensor(
        w_ih.detach().cpu(),
        scale=scale_ih,
        zero_point=zero_point_ih,
        dtype=torch.qint8,
    )
    qw_hh = torch.quantize_per_tensor(
        w_hh.detach().cpu(),
        scale=scale_hh,
        zero_point=zero_point_hh,
        dtype=torch.qint8,
    )
    packed_ih = torch.ops.aten.fbgemm_pack_quantized_matrix(qw_ih)
    packed_hh = torch.ops.aten.fbgemm_pack_quantized_matrix(qw_hh)
    w_int_ih = qw_ih.int_repr().to(torch.int32)
    w_int_hh = qw_hh.int_repr().to(torch.int32)
    col_offsets_ih = (w_int_ih.sum(dim=1) - zero_point_ih * input_size).to(torch.int32)
    col_offsets_hh = (w_int_hh.sum(dim=1) - zero_point_hh * hidden_size).to(torch.int32)

    yield (
        input,
        hx,
        qw_ih.dequantize().to(device),
        qw_hh.dequantize().to(device),
        b_ih,
        b_hh,
        {
            "packed_ih": packed_ih,
            "packed_hh": packed_hh,
            "col_offsets_ih": col_offsets_ih,
            "col_offsets_hh": col_offsets_hh,
            "scale_ih": scale_ih,
            "scale_hh": scale_hh,
            "zero_point_ih": zero_point_ih,
            "zero_point_hh": zero_point_hh,
        },
    )


class QuantizedRnnTanhCellBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Override the yaml-driven shape loading: this op uses (batch,
        # input_size, hidden_size) 3-tuples that the generic core_shapes.yaml
        # does not describe.
        self.shapes = QUANTIZED_RNN_SHAPES


@pytest.mark.quantized_rnn_tanh_cell
def test_quantized_rnn_tanh_cell():
    bench = QuantizedRnnTanhCellBenchmark(
        op_name="quantized_rnn_tanh_cell",
        torch_op=torch_quantized_rnn_tanh_cell_ref,
        input_fn=quantized_rnn_tanh_cell_input_fn,
        gems_op=flag_gems.quantized_rnn_tanh_cell,
        dtypes=[torch.float32],
    )
    bench.run()
