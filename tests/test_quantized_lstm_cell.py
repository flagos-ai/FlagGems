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

import contextlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for the quantized LSTM cell: (batch_size, input_size, hidden_size).
# Covers the 1D (no-batch) path, small/typical cells and larger cells.
LSTM_CELL_SHAPES = [
    (1, 4, 4),
    (2, 8, 6),
    (4, 16, 12),
    (8, 32, 16),
    (16, 64, 32),
]


@contextlib.contextmanager
def _disable_tf32():
    """Disable TF32 matmul on CUDA so the reference matches the strict-fp32
    Triton kernel (which accumulates in true fp32)."""
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul


def _make_inputs(shape, dtype, device):
    batch_size, input_size, hidden_size = shape
    input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    h0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    c0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    w_ih = torch.randn(4 * hidden_size, input_size, dtype=dtype, device=device)
    w_hh = torch.randn(4 * hidden_size, hidden_size, dtype=dtype, device=device)
    b_ih = torch.randn(4 * hidden_size, dtype=dtype, device=device)
    b_hh = torch.randn(4 * hidden_size, dtype=dtype, device=device)
    # The packed/quantization descriptors are accepted for interface
    # compatibility; on the fp32 Triton path they do not affect the result,
    # so placeholders of the expected dtypes are sufficient.
    packed_ih = torch.zeros(
        4 * hidden_size, input_size, dtype=torch.int8, device=device
    )
    packed_hh = torch.zeros(
        4 * hidden_size, hidden_size, dtype=torch.int8, device=device
    )
    col_offsets_ih = torch.zeros(4 * hidden_size, dtype=torch.int32, device=device)
    col_offsets_hh = torch.zeros(4 * hidden_size, dtype=torch.int32, device=device)
    return (
        input,
        h0,
        c0,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        (
            col_offsets_ih,
            col_offsets_hh,
        ),
    )


def _reference(input, h0, c0, w_ih, w_hh, b_ih, b_hh):
    """Standard fp32 LSTM cell math, computed on the reference device."""
    ref_input = utils.to_reference(input)
    ref_h0 = utils.to_reference(h0)
    ref_c0 = utils.to_reference(c0)
    ref_w_ih = utils.to_reference(w_ih)
    ref_w_hh = utils.to_reference(w_hh)
    ref_b_ih = utils.to_reference(b_ih)
    ref_b_hh = utils.to_reference(b_hh)

    ref_input = ref_input.to(torch.float32)
    ref_h0 = ref_h0.to(torch.float32)
    ref_c0 = ref_c0.to(torch.float32)
    ref_w_ih = ref_w_ih.to(torch.float32)
    ref_w_hh = ref_w_hh.to(torch.float32)
    ref_b_ih = ref_b_ih.to(torch.float32) if ref_b_ih is not None else None
    ref_b_hh = ref_b_hh.to(torch.float32) if ref_b_hh is not None else None

    # Match the strict-fp32 accumulation of the Triton kernel.
    with _disable_tf32():
        gates = ref_input @ ref_w_ih.t()
        if ref_b_ih is not None:
            gates = gates + ref_b_ih
        gates = gates + ref_h0 @ ref_w_hh.t()
        if ref_b_hh is not None:
            gates = gates + ref_b_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    ref_cy = f * ref_c0 + i * g
    ref_hy = o * torch.tanh(ref_cy)
    return ref_hy, ref_cy


def _call_gems(
    input, h0, c0, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets
):
    col_offsets_ih, col_offsets_hh = col_offsets
    res_hy, res_cy = flag_gems.quantized_lstm_cell(
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
    return res_hy, res_cy


def _atol(dtype):
    if dtype == torch.bfloat16:
        return 1e-2
    if dtype == torch.float16:
        return 1.5e-3
    return 1e-4


@pytest.mark.quantized_lstm_cell
@pytest.mark.parametrize("shape", LSTM_CELL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_lstm_cell(shape, dtype):
    """Test quantized_lstm_cell accuracy against the standard LSTM cell math."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    (
        input,
        h0,
        c0,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets,
    ) = _make_inputs(shape, dtype, flag_gems.device)

    ref_hy, ref_cy = _reference(input, h0, c0, w_ih, w_hh, b_ih, b_hh)
    res_hy, res_cy = _call_gems(
        input, h0, c0, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets
    )

    atol = _atol(dtype)
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)


@pytest.mark.quantized_lstm_cell
@pytest.mark.parametrize("shape", LSTM_CELL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_lstm_cell_zero_bias(shape, dtype):
    """Test quantized_lstm_cell with zero biases (equivalent to no bias)."""
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    batch_size, input_size, hidden_size = shape
    device = flag_gems.device
    input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    h0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    c0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    w_ih = torch.randn(4 * hidden_size, input_size, dtype=dtype, device=device)
    w_hh = torch.randn(4 * hidden_size, hidden_size, dtype=dtype, device=device)
    # The op schema takes bias tensors (not Optional), so the "no-bias" case
    # is represented by zero-filled biases.
    b_ih = torch.zeros(4 * hidden_size, dtype=dtype, device=device)
    b_hh = torch.zeros(4 * hidden_size, dtype=dtype, device=device)
    packed_ih = torch.zeros(
        4 * hidden_size, input_size, dtype=torch.int8, device=device
    )
    packed_hh = torch.zeros(
        4 * hidden_size, hidden_size, dtype=torch.int8, device=device
    )
    col_offsets_ih = torch.zeros(4 * hidden_size, dtype=torch.int32, device=device)
    col_offsets_hh = torch.zeros(4 * hidden_size, dtype=torch.int32, device=device)

    ref_hy, ref_cy = _reference(input, h0, c0, w_ih, w_hh, b_ih, b_hh)
    res_hy, res_cy = _call_gems(
        input,
        h0,
        c0,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        (col_offsets_ih, col_offsets_hh),
    )

    atol = _atol(dtype)
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)


@pytest.mark.quantized_lstm_cell
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_lstm_cell_single_batch(dtype):
    """Test the single-sample batch=1 path."""
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    batch_size = 1
    input_size = 16
    hidden_size = 20
    device = flag_gems.device
    input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    h0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    c0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    w_ih = torch.randn(4 * hidden_size, input_size, dtype=dtype, device=device)
    w_hh = torch.randn(4 * hidden_size, hidden_size, dtype=dtype, device=device)
    b_ih = torch.randn(4 * hidden_size, dtype=dtype, device=device)
    b_hh = torch.randn(4 * hidden_size, dtype=dtype, device=device)
    packed_ih = torch.zeros(
        4 * hidden_size, input_size, dtype=torch.int8, device=device
    )
    packed_hh = torch.zeros(
        4 * hidden_size, hidden_size, dtype=torch.int8, device=device
    )
    col_offsets_ih = torch.zeros(4 * hidden_size, dtype=torch.int32, device=device)
    col_offsets_hh = torch.zeros(4 * hidden_size, dtype=torch.int32, device=device)

    ref_hy, ref_cy = _reference(input, h0, c0, w_ih, w_hh, b_ih, b_hh)
    res_hy, res_cy = _call_gems(
        input,
        h0,
        c0,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        (col_offsets_ih, col_offsets_hh),
    )

    atol = _atol(dtype)
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)
