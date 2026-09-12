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
from . import conftest as cfg

# Larger reduction dimensions to exercise the K-tiled accumulation loops.
RNN_CELL_LARGE_SHAPES = (
    [(2, 256, 64)] if cfg.QUICK_MODE else [(1, 256, 128), (4, 512, 256), (8, 1024, 128)]
)


# Shapes used for the quantized RNN cell: (batch, input_size, hidden_size).
# The cell computes a single time step, so only 2D activations are involved.
RNN_CELL_SHAPES = (
    [(2, 8, 4)]
    if cfg.QUICK_MODE
    else [(2, 8, 4), (4, 16, 13), (8, 32, 20), (16, 64, 48), (32, 128, 64)]
)


def _make_inputs(shape, dtype, device, zero_point=0):
    batch, input_size, hidden_size = shape
    input = torch.randn(batch, input_size, dtype=dtype, device=device)
    hx = torch.randn(batch, hidden_size, dtype=dtype, device=device)
    w_ih_fp = torch.randn(hidden_size, input_size, dtype=torch.float32)
    w_hh_fp = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
    if zero_point == 0:
        w_ih_int8, scale_ih, zp_ih = _quantize_weight(w_ih_fp)
        w_hh_int8, scale_hh, zp_hh = _quantize_weight(w_hh_fp)
    else:
        w_ih_int8, scale_ih, zp_ih = _quantize_weight_zp(w_ih_fp, zero_point)
        w_hh_int8, scale_hh, zp_hh = _quantize_weight_zp(w_hh_fp, zero_point)
    b_ih = torch.randn(hidden_size, dtype=torch.float32, device=device)
    b_hh = torch.randn(hidden_size, dtype=torch.float32, device=device)
    # FBGEMM pack/col-offset buffers: real values are not needed by the
    # FlagGems kernel (it reads the int8 weight directly), so we pass the
    # int8 weight as a stand-in for `packed_*` and zeros for `col_offsets_*`.
    packed_ih = w_ih_int8.clone().to(device)
    packed_hh = w_hh_int8.clone().to(device)
    col_offsets_ih = torch.zeros(hidden_size, dtype=torch.int32, device=device)
    col_offsets_hh = torch.zeros(hidden_size, dtype=torch.int32, device=device)
    return (
        input,
        hx,
        w_ih_int8.to(device),
        w_hh_int8.to(device),
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )


# Per-dtype tolerances. The kernel accumulates in fp32, so float32 is tight;
# float16/bfloat16 carry the dtype's natural rounding when storing back.
_ATOL = {torch.float32: 1e-4, torch.float16: 1e-3, torch.bfloat16: 2e-2}

pytestmark = pytest.mark.quantized_rnn_relu_cell


def _reference_quantized_rnn_relu_cell(
    input, hx, w_ih_int8, w_hh_int8, b_ih, b_hh, scale_ih, scale_hh, zp_ih, zp_hh
):
    """fp64 reference for ``quantized_rnn_relu_cell``.

    The FBGEMM reference kernel is unavailable on this build, but the numeric
    result is the dequantized-weight matmul followed by ReLU::

        igates = scale_ih * (input @ w_ih_int8.T - zp_ih * input.rowsum) + b_ih
        hgates = scale_hh * (hx    @ w_hh_int8.T - zp_hh * hx.rowsum)    + b_hh
        hy     = relu(igates + hgates)

    The packed / col_offsets buffers are FBGEMM layout artifacts and do not
    affect the numeric output, so they are not needed here.
    """
    ref_dtype = (
        torch.float64 if flag_gems.runtime.device.support_fp64 else torch.float32
    )
    # Route the reference inputs through to_reference so the golden reference
    # is computed on the reference device/precision (a no-op here while the
    # CUDA path runs, since these tests skip on TO_CPU).
    inp = utils.to_reference(input).to(ref_dtype)
    h = utils.to_reference(hx).to(ref_dtype)
    w_ih = utils.to_reference(w_ih_int8).to(ref_dtype)
    w_hh = utils.to_reference(w_hh_int8).to(ref_dtype)
    ig = scale_ih * (inp @ w_ih.t() - zp_ih * inp.sum(dim=1, keepdim=True)) + (
        utils.to_reference(b_ih).to(ref_dtype)
    )
    hg = scale_hh * (h @ w_hh.t() - zp_hh * h.sum(dim=1, keepdim=True)) + (
        utils.to_reference(b_hh).to(ref_dtype)
    )
    return torch.relu(ig + hg)


def _quantize_weight_zp(w_fp, zero_point):
    """Per-tensor quantization with a caller-chosen integer zero point."""
    max_val = w_fp.abs().max().item()
    # Keep the quantized range within int8 bounds given the offset.
    lo = -127 - zero_point
    hi = 127 - zero_point
    bound = max(abs(lo), abs(hi))
    scale = (max_val / bound) if max_val > 0 else 1.0
    w_int8 = torch.round(w_fp / scale + zero_point).clamp(-127, 127).to(torch.int8)
    return w_int8, scale, zero_point


def _quantize_weight(w_fp, dtype=torch.qint8):
    """Per-tensor symmetric-ish quantization mimicking FBGEMM
    ``fbgemm_linear_quantize_weight``: returns ``(w_int8, scale, zero_point)``.

    We use a zero point of 0 (symmetric) by default; tests can pass an explicit
    non-zero zero point to exercise the dequantization correction term.
    """
    max_val = w_fp.abs().max().item()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    zero_point = 0
    w_int8 = torch.round(w_fp / scale + zero_point).clamp(-127, 127).to(torch.int8)
    return w_int8, scale, zero_point


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("shape", RNN_CELL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_rnn_relu_cell(shape, dtype):
    """Accuracy of quantized_rnn_relu_cell vs the fp64 dequantized-matmul reference."""
    torch.backends.cuda.matmul.allow_tf32 = False
    args = _make_inputs(shape, dtype, flag_gems.device, zero_point=0)
    (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    ) = args

    ref = _reference_quantized_rnn_relu_cell(
        input, hx, w_ih, w_hh, b_ih, b_hh, scale_ih, scale_hh, zp_ih, zp_hh
    ).to(dtype)

    res = flag_gems.quantized_rnn_relu_cell(
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )

    utils.gems_assert_close(res, ref, dtype, atol=_ATOL[dtype])


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("shape", RNN_CELL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_rnn_relu_cell_nonzero_zero_point(shape, dtype):
    """Exercise the dequantization zero-point correction term (zp != 0)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    args = _make_inputs(shape, dtype, flag_gems.device, zero_point=7)
    (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    ) = args

    ref = _reference_quantized_rnn_relu_cell(
        input, hx, w_ih, w_hh, b_ih, b_hh, scale_ih, scale_hh, zp_ih, zp_hh
    ).to(dtype)

    res = flag_gems.quantized_rnn_relu_cell(
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )

    utils.gems_assert_close(res, ref, dtype, atol=_ATOL[dtype])


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("shape", RNN_CELL_LARGE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_rnn_relu_cell_large(shape, dtype):
    """Large reduction dimensions exercise the K-tiled accumulation loops."""
    torch.backends.cuda.matmul.allow_tf32 = False
    args = _make_inputs(shape, dtype, flag_gems.device, zero_point=0)
    (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    ) = args

    ref = _reference_quantized_rnn_relu_cell(
        input, hx, w_ih, w_hh, b_ih, b_hh, scale_ih, scale_hh, zp_ih, zp_hh
    ).to(dtype)

    res = flag_gems.quantized_rnn_relu_cell(
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )

    # Allow slightly looser tolerance for large reductions.
    atol = _ATOL[dtype] * 2
    utils.gems_assert_close(res, ref, dtype, atol=atol)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_rnn_relu_cell_all_negative(dtype):
    """When pre-activation is entirely negative, ReLU output must be all zeros."""
    torch.backends.cuda.matmul.allow_tf32 = False
    batch, input_size, hidden_size = 4, 16, 8
    device = flag_gems.device
    input = torch.randn(batch, input_size, dtype=dtype, device=device) * 0.01
    hx = torch.randn(batch, hidden_size, dtype=dtype, device=device) * 0.01
    w_ih_fp = torch.randn(hidden_size, input_size, dtype=torch.float32) * 0.01
    w_hh_fp = torch.randn(hidden_size, hidden_size, dtype=torch.float32) * 0.01
    w_ih_int8, scale_ih, zp_ih = _quantize_weight(w_ih_fp)
    w_hh_int8, scale_hh, zp_hh = _quantize_weight(w_hh_fp)
    # Large negative biases guarantee a negative pre-activation everywhere.
    b_ih = torch.full((hidden_size,), -100.0, dtype=torch.float32, device=device)
    b_hh = torch.full((hidden_size,), -100.0, dtype=torch.float32, device=device)
    packed_ih = w_ih_int8.clone().to(device)
    packed_hh = w_hh_int8.clone().to(device)
    col_ih = torch.zeros(hidden_size, dtype=torch.int32, device=device)
    col_hh = torch.zeros(hidden_size, dtype=torch.int32, device=device)
    w_ih = w_ih_int8.to(device)
    w_hh = w_hh_int8.to(device)

    res = flag_gems.quantized_rnn_relu_cell(
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )
    assert res.shape == (batch, hidden_size)
    assert res.dtype == dtype
    utils.gems_assert_equal(res, torch.zeros_like(res))


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_batch_one():
    """batch == 1 is a degenerate but valid case (single sequence element)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float32
    args = _make_inputs((1, 8, 5), dtype, flag_gems.device, zero_point=0)
    (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    ) = args

    ref = _reference_quantized_rnn_relu_cell(
        input, hx, w_ih, w_hh, b_ih, b_hh, scale_ih, scale_hh, zp_ih, zp_hh
    ).to(dtype)

    res = flag_gems.quantized_rnn_relu_cell(
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )
    utils.gems_assert_close(res, ref, dtype, atol=_ATOL[dtype])
