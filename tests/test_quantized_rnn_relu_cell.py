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

import zlib

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


def _quantize_weight_zp(w_fp, zero_point):
    """Per-tensor quantization with a caller-chosen integer zero point.

    Produces (int8 weight, scale, zero_point) plus the FBGEMM column offsets
    ``rowsum(w_int8) - K * zero_point`` so the produced buffers are consistent
    with what ``aten::quantized_rnn_relu_cell`` receives.
    """
    max_val = w_fp.abs().max().item()
    # Keep the quantized range within int8 bounds given the offset.
    lo = -127 - zero_point
    hi = 127 - zero_point
    bound = max(abs(lo), abs(hi))
    scale = (max_val / bound) if max_val > 0 else 1.0
    w_int8 = torch.round(w_fp / scale + zero_point).clamp(-127, 127).to(torch.int8)
    col_offsets = w_int8.sum(dim=1, dtype=torch.int32) - zero_point * w_int8.shape[1]
    return w_int8, scale, zero_point, col_offsets


def _pack_weight(w_int8):
    """FBGEMM-packed representation of an int8 weight (real packed buffer)."""
    return torch.fbgemm_pack_quantized_matrix(w_int8.contiguous())


def _quantize_weight(w_fp):
    """``torch.fbgemm_linear_quantize_weight`` with real FBGEMM col offsets.

    Returns ``(w_int8, col_offsets, scale, zero_point)`` exactly as ATen's
    legacy quantized-linear path receives them.
    """
    w_int8, col_offsets, scale, zero_point = torch.fbgemm_linear_quantize_weight(w_fp)
    return w_int8, col_offsets, scale, zero_point


def _make_inputs(shape, device, zero_point=None):
    """Build quantized-RNN-cell inputs through the real FBGEMM quantize/pack APIs.

    ``zero_point=None`` uses ``torch.fbgemm_linear_quantize_weight`` (the exact
    quantization ATen uses); an integer zero point uses a manual asymmetric
    quantization to exercise non-zero weight zero points.  All tensors are
    returned on CPU; callers move the CUDA arguments to device.
    """
    batch, input_size, hidden_size = shape
    # Draw from a local generator seeded off the case, not the global RNG: the
    # global stream depends on how many tests ran (and in what order) before
    # this one, so a failure on CI could not be reproduced locally even with
    # torch.manual_seed() in the test body.
    gen = torch.Generator().manual_seed(
        abs(zlib.crc32(repr((shape, str(zero_point))).encode())) % (2**31)
    )
    input = torch.randn(batch, input_size, dtype=torch.float32, generator=gen)
    hx = torch.randn(batch, hidden_size, dtype=torch.float32, generator=gen)
    w_ih_fp = (
        torch.randn(hidden_size, input_size, dtype=torch.float32, generator=gen) * 0.5
    )
    w_hh_fp = (
        torch.randn(hidden_size, hidden_size, dtype=torch.float32, generator=gen) * 0.5
    )
    b_ih = torch.randn(hidden_size, dtype=torch.float32, generator=gen) * 0.2
    b_hh = torch.randn(hidden_size, dtype=torch.float32, generator=gen) * 0.2
    if zero_point is None:
        w_ih_int8, col_ih, scale_ih, zp_ih = _quantize_weight(w_ih_fp)
        w_hh_int8, col_hh, scale_hh, zp_hh = _quantize_weight(w_hh_fp)
    else:
        w_ih_int8, scale_ih, zp_ih, col_ih = _quantize_weight_zp(w_ih_fp, zero_point)
        w_hh_int8, scale_hh, zp_hh, col_hh = _quantize_weight_zp(w_hh_fp, zero_point)
    packed_ih = _pack_weight(w_ih_int8)
    packed_hh = _pack_weight(w_hh_int8)
    return (
        input,
        hx,
        w_ih_int8,
        w_hh_int8,
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


def _to_device(args, device):
    """Move the tensor arguments of ``_make_inputs`` to ``device``."""
    idx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}  # input, hx, w_ih, w_hh, b_ih, b_hh,
    out = list(args)  # packed, col offsets
    for i in idx:
        out[i] = out[i].to(device)
    return tuple(out)


# ``aten::quantized_rnn_relu_cell`` is float32-only (fp16/bf16 activations
# raise "expected scalar type Float but found Half/BFloat16"), so the FlagGems
# kernel matches ATen by accepting float32 only.
RNN_CELL_DTYPES = [torch.float32]
# The kernel reproduces ATen's integer path, so a local wheel matches
# bit-exactly; across 200 random inputs the worst local gap is 0.0233, under
# one quantization bin (scale ~0.025). The tolerance stays tight on purpose:
# CI has reported gaps up to 1.851 -- ~66 bins -- which is a real arithmetic
# divergence, not cross-build rounding, and must not be hidden behind a loose
# budget.
_ATOL = {torch.float32: 1e-4}

pytestmark = pytest.mark.quantized_rnn_relu_cell


def _run_cell(args, device):
    """Run the FlagGems kernel on device-side copies of ``args``."""
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
    ) = _to_device(args, device)
    return flag_gems.quantized_rnn_relu_cell(
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


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("shape", RNN_CELL_SHAPES)
@pytest.mark.parametrize("dtype", RNN_CELL_DTYPES)
def test_quantized_rnn_relu_cell_aten_parity(shape, dtype):
    """Parity of the Triton kernel against the native CPU ATen operator.

    The reference is ``torch.quantized_rnn_relu_cell`` fed with real FBGEMM
    packed weights / column offsets from ``torch.fbgemm_linear_quantize_weight``,
    so the comparison exercises the full native quantization/correction path
    (dynamic per-tensor quint8 activation quantization + integer GEMM
    correction), not a mirrored kernel formula.

    Seeded: without a fixed seed the inputs differ on every run, so a CI
    failure cannot be reproduced locally (CI reported a 1.851 gap -- ~66
    quantization bins, a real divergence -- that no local run reproduced).
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    args = _make_inputs(shape, flag_gems.device, zero_point=None)
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

    ref = torch.quantized_rnn_relu_cell(
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
        float(scale_ih),
        float(scale_hh),
        int(zp_ih),
        int(zp_hh),
    )

    res = _run_cell(args, flag_gems.device).cpu()

    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    utils.gems_assert_close(res, ref.to(dtype), dtype, atol=_ATOL[dtype])


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("shape", RNN_CELL_SHAPES)
@pytest.mark.parametrize("dtype", RNN_CELL_DTYPES)
def test_quantized_rnn_relu_cell_nonzero_zero_point_aten_parity(shape, dtype):
    """ATen parity with non-zero weight zero points (asymmetric quantization)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    args = _make_inputs(shape, flag_gems.device, zero_point=7)
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

    ref = torch.quantized_rnn_relu_cell(
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
        float(scale_ih),
        float(scale_hh),
        int(zp_ih),
        int(zp_hh),
    )

    res = _run_cell(args, flag_gems.device).cpu()

    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    utils.gems_assert_close(res, ref.to(dtype), dtype, atol=_ATOL[dtype])


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("shape", RNN_CELL_LARGE_SHAPES)
@pytest.mark.parametrize("dtype", RNN_CELL_DTYPES)
def test_quantized_rnn_relu_cell_large_aten_parity(shape, dtype):
    """Large reduction dimensions exercise the K-tiled integer GEMM loops."""
    torch.backends.cuda.matmul.allow_tf32 = False
    args = _make_inputs(shape, flag_gems.device, zero_point=None)
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

    ref = torch.quantized_rnn_relu_cell(
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
        float(scale_ih),
        float(scale_hh),
        int(zp_ih),
        int(zp_hh),
    )

    res = _run_cell(args, flag_gems.device).cpu()

    # Allow slightly looser tolerance for large reductions.
    atol = _ATOL[dtype] * 2  # keep the same relative headroom
    utils.gems_assert_close(res, ref.to(dtype), dtype, atol=atol)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda a: (a[0], torch.randn(a[0].shape[0] + 1, a[0].shape[1]), *a[2:]),
            id="hx-batch-mismatch",
        ),
        pytest.param(
            lambda a: (
                a[0],
                a[1],
                torch.zeros(a[2].shape[0], a[0].shape[1] + 1, dtype=torch.int8),
                *a[3:],
            ),
            id="w_ih-wrong-K",
        ),
        pytest.param(
            lambda a: (a[0], a[1], a[2], a[3].to(torch.float32), *a[4:]),
            id="w_hh-fp32-dtype",
        ),
        pytest.param(
            lambda a: (a[0], a[1], a[2], a[3], torch.randn(a[4].shape[0] + 1), *a[5:]),
            id="b_ih-wrong-len",
        ),
        pytest.param(
            lambda a: (torch.randn(a[0].shape[1]), *a[1:]),
            id="input-1d",
        ),
        pytest.param(
            lambda a: (a[0].half(), *a[1:]),
            id="input-fp16",
        ),
    ],
)
def test_quantized_rnn_relu_cell_validation(mutate):
    """Bad shapes/dtypes must be rejected before the kernel launches."""
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        pytest.skip("Triton kernel is CUDA-only")
    args = _make_inputs((4, 16, 8), flag_gems.device, zero_point=None)
    bad = mutate(args)
    with pytest.raises(RuntimeError):
        flag_gems.quantized_rnn_relu_cell(*bad)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("dtype", RNN_CELL_DTYPES)
def test_quantized_rnn_relu_cell_all_negative(dtype):
    """When pre-activation is entirely negative, ReLU output must be all zeros."""
    torch.backends.cuda.matmul.allow_tf32 = False
    batch, input_size, hidden_size = 4, 16, 8
    device = flag_gems.device
    input = torch.randn(batch, input_size, dtype=torch.float32, device=device) * 0.01
    hx = torch.randn(batch, hidden_size, dtype=torch.float32, device=device) * 0.01
    w_ih_fp = torch.randn(hidden_size, input_size, dtype=torch.float32) * 0.01
    w_hh_fp = torch.randn(hidden_size, hidden_size, dtype=torch.float32) * 0.01
    w_ih_int8, scale_ih, zp_ih, col_ih_cpu = _quantize_weight_zp(w_ih_fp, 0)
    w_hh_int8, scale_hh, zp_hh, col_hh_cpu = _quantize_weight_zp(w_hh_fp, 0)
    # Large negative biases guarantee a negative pre-activation everywhere.
    b_ih = torch.full((hidden_size,), -100.0, dtype=torch.float32, device=device)
    b_hh = torch.full((hidden_size,), -100.0, dtype=torch.float32, device=device)
    packed_ih = _pack_weight(w_ih_int8).to(device)
    packed_hh = _pack_weight(w_hh_int8).to(device)
    col_ih = col_ih_cpu.to(device)
    col_hh = col_hh_cpu.to(device)
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
    args = _make_inputs((1, 8, 5), flag_gems.device, zero_point=None)
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

    ref = torch.quantized_rnn_relu_cell(
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
        float(scale_ih),
        float(scale_hh),
        int(zp_ih),
        int(zp_hh),
    )

    res = _run_cell(args, flag_gems.device).cpu()
    utils.gems_assert_close(res, ref.to(dtype), dtype, atol=_ATOL[dtype])
