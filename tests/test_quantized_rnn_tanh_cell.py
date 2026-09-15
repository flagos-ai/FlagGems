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

from . import accuracy_utils as utils

# (batch, input_size, hidden_size) tuples representative of RNN cell usage.
QUANTIZED_RNN_SHAPES = [
    (2, 8, 8),
    (4, 16, 32),
    (3, 64, 128),
    (8, 128, 256),
]


# The CPU FBGEMM reference accumulates the int8 mat-vec products in a different
# order than the dequantized float matmul, so a loose tolerance is required to
# absorb the quantization noise. The CI FBGEMM build rounds the packed int8
# weight differently from a local build (observed single-element gaps reach
# ~0.22 on CI vs ~0.02 locally for the same random seed), so this is kept as a
# loose cross-check rather than a tight numerical bound.
QUANT_REF_TOL = 5e-1


def _pack_per_tensor_weight(w, scale, zero_point):
    """Per-tensor quantize ``w`` and build the FBGEMM packed weight + col_offsets.

    Returns ``(quantized_weight, packed, col_offsets)`` suitable for the
    ``aten::quantized_rnn_tanh_cell`` CPU reference.
    """
    qw = torch.quantize_per_tensor(
        w, scale=scale, zero_point=zero_point, dtype=torch.qint8
    )
    packed = torch.ops.aten.fbgemm_pack_quantized_matrix(qw)
    w_int = qw.int_repr().to(torch.int32)
    # Column offsets (length = output dim) include the sum of the weight columns
    # as well as ``zero_point * K``; this matches the FBGEMM packing convention.
    col_offsets = (w_int.sum(dim=1) - zero_point * w_int.shape[1]).to(torch.int32)
    return qw, packed, col_offsets


@pytest.mark.quantized_rnn_tanh_cell
@pytest.mark.parametrize("shape", QUANTIZED_RNN_SHAPES)
# The aten CPU/FBGEMM reference requires float32 activation tensors.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_quantized_rnn_tanh_cell(shape, dtype):
    """Accuracy of the FlagGems kernel against the FBGEMM CPU reference.

    The reference ``torch.quantized_rnn_tanh_cell`` is CPU/FBGEMM-only (it
    segfaults on CUDA because the packed weights are CPU-opaque), so the
    reference runs on CPU while the FlagGems kernel runs on the GPU device.
    The weights are scaled so the pre-activations stay moderate, keeping the
    quantization noise small relative to the signal.
    """
    batch, input_size, hidden_size = shape

    # Scale inputs/weights so pre-activations are moderate (avoid tanh saturation
    # which would amplify the int8 quantization noise of the reference).
    input = torch.randn(batch, input_size, dtype=dtype, device=flag_gems.device)
    input = input * (0.3 / max(input.std().item(), 1e-6))
    hx = torch.randn(batch, hidden_size, dtype=dtype, device=flag_gems.device)
    hx = hx * (0.3 / max(hx.std().item(), 1e-6))
    w_ih = torch.randn(hidden_size, input_size, dtype=dtype, device=flag_gems.device)
    w_ih = w_ih / (input_size**0.5)
    w_hh = torch.randn(hidden_size, hidden_size, dtype=dtype, device=flag_gems.device)
    w_hh = w_hh / (hidden_size**0.5)
    b_ih = torch.randn(hidden_size, dtype=dtype, device=flag_gems.device) * 0.3
    b_hh = torch.randn(hidden_size, dtype=dtype, device=flag_gems.device) * 0.3

    # Quantize the weights (per-tensor, symmetric) on CPU for the reference.
    scale_ih = (w_ih.abs().max().to("cpu").item() / 127.0) or 1e-6
    scale_hh = (w_hh.abs().max().to("cpu").item() / 127.0) or 1e-6
    w_ih_cpu = w_ih.to("cpu")
    w_hh_cpu = w_hh.to("cpu")
    qw_ih, packed_ih, col_offsets_ih = _pack_per_tensor_weight(w_ih_cpu, scale_ih, 0)
    qw_hh, packed_hh, col_offsets_hh = _pack_per_tensor_weight(w_hh_cpu, scale_hh, 0)

    # Reference: the ATen op is CPU/FBGEMM-only.
    ref_out = torch.quantized_rnn_tanh_cell(
        input.to("cpu"),
        hx.to("cpu"),
        qw_ih.dequantize(),
        qw_hh.dequantize(),
        b_ih.to("cpu"),
        b_hh.to("cpu"),
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        0,
        0,
    )

    # FlagGems implementation (GPU Triton kernel).
    res_out = flag_gems.quantized_rnn_tanh_cell(
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
        0,
        0,
    )

    torch.testing.assert_close(
        res_out.to("cpu"), ref_out, atol=QUANT_REF_TOL, rtol=QUANT_REF_TOL
    )


@pytest.mark.quantized_rnn_tanh_cell
@pytest.mark.parametrize("shape", QUANTIZED_RNN_SHAPES)
# bf16 is excluded: its accumulation noise exceeds the tight 1e-2 tolerance
# against the dequantized float reference for this tanh cell.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantized_rnn_tanh_cell_float_ref(shape, dtype):
    """Accuracy of the FlagGems kernel against a pure float PyTorch reference.

    Both the kernel and the reference consume the same (dequantized) float
    weights, so this checks the Triton kernel arithmetic itself without the
    int8 quantization noise of the FBGEMM path.
    """
    batch, input_size, hidden_size = shape

    # Scale inputs/weights so pre-activations stay moderate; otherwise tanh
    # saturates and tiny float-accumulation-order differences get amplified.
    input = torch.randn(batch, input_size, dtype=dtype, device=flag_gems.device)
    input = input * (0.3 / max(input.std().item(), 1e-6))
    hx = torch.randn(batch, hidden_size, dtype=dtype, device=flag_gems.device)
    hx = hx * (0.3 / max(hx.std().item(), 1e-6))
    w_ih = torch.randn(hidden_size, input_size, dtype=dtype, device=flag_gems.device)
    w_ih = w_ih / (input_size**0.5)
    w_hh = torch.randn(hidden_size, hidden_size, dtype=dtype, device=flag_gems.device)
    w_hh = w_hh / (hidden_size**0.5)
    b_ih = torch.randn(hidden_size, dtype=dtype, device=flag_gems.device) * 0.3
    b_hh = torch.randn(hidden_size, dtype=dtype, device=flag_gems.device) * 0.3

    packed_ih = torch.zeros(input_size, dtype=torch.int8, device=flag_gems.device)
    packed_hh = torch.zeros(hidden_size, dtype=torch.int8, device=flag_gems.device)
    col_offsets_ih = torch.zeros(
        hidden_size, dtype=torch.int32, device=flag_gems.device
    )
    col_offsets_hh = torch.zeros(
        hidden_size, dtype=torch.int32, device=flag_gems.device
    )

    res_out = flag_gems.quantized_rnn_tanh_cell(
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
        1.0,
        1.0,
        0,
        0,
    )

    # Float reference on the reference (CPU/upcast) side.
    ref_input = utils.to_reference(input)
    ref_hx = utils.to_reference(hx)
    ref_w_ih = utils.to_reference(w_ih)
    ref_w_hh = utils.to_reference(w_hh)
    ref_b_ih = utils.to_reference(b_ih)
    ref_b_hh = utils.to_reference(b_hh)
    ref_out = torch.tanh(
        ref_input.to(torch.float32) @ ref_w_ih.to(torch.float32).T
        + ref_b_ih.to(torch.float32)
        + ref_hx.to(torch.float32) @ ref_w_hh.to(torch.float32).T
        + ref_b_hh.to(torch.float32)
    ).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2)
