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

from . import base

# (batch, input_size, hidden_size) shapes representative of a single
# quantized RNN (ReLU) cell step.
QUANTIZED_RNN_RELU_CELL_SHAPES = [
    (8, 64, 64),
    (16, 128, 128),
    (32, 256, 256),
    (64, 512, 512),
]

# ``aten::quantized_rnn_relu_cell`` is float32-only (fp16/bf16 activations
# raise "expected scalar type Float but found Half/BFloat16"), so the FlagGems
# kernel matches ATen and only the fp32 case is benchmarked.
QUANTIZED_RNN_RELU_CELL_DTYPES = [torch.float32]


def _torch_quantized_rnn_relu_cell_gpu(
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
    """GPU-runnable reference for ``aten::quantized_rnn_relu_cell``.

    Reproduces the FBGEMM legacy path ATen runs on CPU: per-tensor dynamic
    quint8 activation quantization followed by the zero-point-corrected
    integer GEMM (the packed buffers are x86 layout artifacts and are not
    needed here).  Runs entirely on the accelerator, so it is a meaningful
    latency baseline for the fused Triton kernel.
    """

    def _gates(a, w_q, col_offsets, w_scale, w_zp, bias):
        xq = torch.quantize_per_tensor_dynamic(a, torch.quint8, reduce_range=False)
        aq = xq.int_repr().to(torch.int32).to(torch.float32)
        azp = xq.q_zero_point()
        # col_offsets == rowsum(w_q) - K * w_zp, so this equals
        # (aq - azp) @ (w_q - w_zp)^T in exact integer arithmetic.
        acc = (
            aq @ w_q.to(torch.float32).t()
            - azp * col_offsets.to(torch.float32)[None, :]
            - w_zp * aq.sum(dim=1, keepdim=True)
        )
        return (xq.q_scale() * w_scale) * acc + bias

    igates = _gates(input, w_ih, col_offsets_ih, scale_ih, zero_point_ih, b_ih)
    hgates = _gates(hx, w_hh, col_offsets_hh, scale_hh, zero_point_hh, b_hh)
    return torch.relu(igates + hgates)


class QuantizedRnnReluCellBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QUANTIZED_RNN_RELU_CELL_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch, input_size, hidden_size = shape
            input = torch.randn(
                batch, input_size, dtype=torch.float32, device=self.device
            )
            hx = torch.randn(
                batch, hidden_size, dtype=torch.float32, device=self.device
            )
            # Quantize fp32 weights with ``fbgemm_linear_quantize_weight`` (the
            # quantization ATen itself uses) and pack with the real FBGEMM API.
            w_ih_int8, col_offsets_ih, scale_ih, zp_ih = (
                torch.fbgemm_linear_quantize_weight(
                    torch.randn(hidden_size, input_size, dtype=torch.float32)
                )
            )
            w_hh_int8, col_offsets_hh, scale_hh, zp_hh = (
                torch.fbgemm_linear_quantize_weight(
                    torch.randn(hidden_size, hidden_size, dtype=torch.float32)
                )
            )
            b_ih = torch.randn(hidden_size, dtype=torch.float32, device=self.device)
            b_hh = torch.randn(hidden_size, dtype=torch.float32, device=self.device)
            packed_ih = torch.fbgemm_pack_quantized_matrix(w_ih_int8).to(self.device)
            packed_hh = torch.fbgemm_pack_quantized_matrix(w_hh_int8).to(self.device)
            yield (
                input,
                hx,
                w_ih_int8.to(self.device),
                w_hh_int8.to(self.device),
                b_ih,
                b_hh,
                packed_ih,
                packed_hh,
                col_offsets_ih.to(self.device),
                col_offsets_hh.to(self.device),
                float(scale_ih),
                float(scale_hh),
                int(zp_ih),
                int(zp_hh),
            )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell():
    # aten::quantized_rnn_relu_cell only dispatches to CPU (FBGEMM), so the
    # baseline is a GPU-runnable reimplementation of its numerics: dynamic
    # quint8 activation quantization + zero-point-corrected integer GEMM.
    bench = QuantizedRnnReluCellBenchmark(
        op_name="quantized_rnn_relu_cell",
        torch_op=_torch_quantized_rnn_relu_cell_gpu,
        dtypes=QUANTIZED_RNN_RELU_CELL_DTYPES,
        gems_op=flag_gems.quantized_rnn_relu_cell,
    )
    bench.run()
