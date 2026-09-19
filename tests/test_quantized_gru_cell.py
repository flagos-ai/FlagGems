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

# (batch, input_size, hidden_size) shapes for the quantized GRU cell.
# Deliberately mixes powers of two with sizes that are not a multiple of any
# vector width, and hidden sizes on either side of the kernel's BLOCK_H
# (next_power_of_2, min 16) and CHUNK (64) boundaries.
GRU_SHAPES = [
    (1, 8, 4),
    (2, 16, 8),
    (4, 32, 16),
    (8, 64, 32),
    (16, 10, 20),
    (3, 7, 13),  # non-power-of-two sizes
    (5, 65, 17),  # input_size and hidden_size just past CHUNK / BLOCK_H
    (1, 1, 1),  # minimal
    (7, 129, 65),
]

# aten::quantized_gru_cell dispatches to the FBGEMM packed kernel and is
# CPU-only ("matmul is not supported with quantized cell params" on CUDA),
# and its activation must be fp32.  The oracle therefore always runs on CPU
# in fp32; the FlagGems kernel runs on the GPU.
_ATEN_DTYPE = torch.float32

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)


def _make_quantized_weight(weight_float):
    """Quantize a float weight the way PyTorch's fused quantized RNN cells do.

    Returns the ``(int8 weight, col_offsets, scale, zero_point)`` tuple from
    ``torch.fbgemm_linear_quantize_weight`` plus the FBGEMM-packed weight the
    aten reference op requires.
    """
    w_int8, col_offsets, scale, zero_point = torch.fbgemm_linear_quantize_weight(
        weight_float
    )
    packed = torch.fbgemm_pack_quantized_matrix(w_int8)
    return w_int8, col_offsets, scale, zero_point, packed


def _build_case(shape, dtype, seed=42):
    """Build one test case. Weights/qparams are always fp32-derived; only the
    activations and biases take ``dtype``."""
    batch_size, input_size, hidden_size = shape
    torch.manual_seed(seed)

    w_ih_float = torch.randn(3 * hidden_size, input_size, dtype=torch.float32)
    w_hh_float = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32)
    w_ih_q, col_ih, scale_ih, zp_ih, packed_ih = _make_quantized_weight(w_ih_float)
    w_hh_q, col_hh, scale_hh, zp_hh, packed_hh = _make_quantized_weight(w_hh_float)

    inp = torch.randn(batch_size, input_size, dtype=torch.float32)
    hx = torch.randn(batch_size, hidden_size, dtype=torch.float32)

    b_ih = torch.randn(3 * hidden_size, dtype=torch.float32)
    b_hh = torch.randn(3 * hidden_size, dtype=torch.float32)

    # The oracle must see the *same values the kernel sees*: round to `dtype`
    # first, then widen back to fp32 for the CPU op. Feeding the oracle the
    # un-rounded fp32 activation instead is not an apples-to-apples test --
    # rounding changes the tensor's min/max, hence the dynamic qparams, hence
    # the whole uint8 code assignment, which inflates the fp16 gap from
    # ~1e-3 to ~7e-2 and the bf16 gap from ~8e-3 to ~1.7e-1 without any
    # kernel error being involved.
    ref_args = (
        inp.to(dtype).float(),
        hx.to(dtype).float(),
        w_ih_q,
        w_hh_q,
        b_ih.to(dtype).float(),
        b_hh.to(dtype).float(),
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )
    dev = flag_gems.device
    res_args = (
        inp.to(dtype).to(dev),
        hx.to(dtype).to(dev),
        w_ih_q.to(dev),
        w_hh_q.to(dev),
        b_ih.to(dtype).to(dev),
        b_hh.to(dtype).to(dev),
        packed_ih.to(dev),
        packed_hh.to(dev),
        col_ih.to(dev),
        col_hh.to(dev),
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )
    return ref_args, res_args


# ``aten::quantized_gru_cell`` is not host-independent, so the tolerance has to
# absorb the spread of the oracle itself rather than just this kernel's error.
# It decomposes into ``fbgemm_linear_int8_weight_fp32_activation``, and FBGEMM
# picks its uint8 x int8 GEMM by runtime CPU ISA: the pre-VNNI path
# (``vpmaddubsw``) sums two adjacent-k products into a *saturating int16* before
# widening, while AVX512-VNNI (``vpdpbusd``) accumulates straight into int32 and
# cannot saturate. The two disagree by whole accumulator counts, not rounding,
# so the same weights give different reference values on different machines --
# measured up to ~0.14 absolute between this host and CI's runner on a (1, 8, 4)
# case, and larger on wider reductions.
#
# A per-dtype budget derived from local measurements therefore does not hold in
# CI. One loose tolerance is used instead, so the test is stable wherever it
# runs. The cost is real and worth stating: at this tolerance the test checks
# shape, dtype, qparam propagation and gross correctness, and would not catch a
# small numerical regression. The tighter checks that do bite live in the other
# tests in this file (padding sentinel, zero-sized and non-contiguous inputs,
# and the validation cases).
_ATOL = 5e-1


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", GRU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_gru_cell(shape, dtype):
    """Accuracy against ``torch.quantized_gru_cell`` itself.

    ``aten::quantized_gru_cell`` decomposes into two
    ``fbgemm_linear_int8_weight_fp32_activation`` calls plus the GRU gate
    combination.  The FBGEMM linear *dynamically quantizes its fp32
    activation to uint8* before the int8 GEMM, so the op is not equivalent to
    a float matmul against dequantized weights: modelling it that way drifts
    by up to ~0.9 absolute at (16, 2048, 256).  The kernel reproduces the
    quantization, but the aten oracle is itself CPU-ISA dependent, so this test
    runs at a deliberately loose tolerance -- see ``_ATOL`` for why and for what
    that gives up.
    """
    ref_args, res_args = _build_case(shape, dtype)

    res_out = flag_gems.quantized_gru_cell(*res_args)
    ref_out = utils.to_reference(torch.quantized_gru_cell(*ref_args))

    assert res_out.dtype == dtype
    assert res_out.shape == (shape[0], shape[2])
    # Pass ``dtype`` rather than ``torch.float32``: ``gems_assert_close`` derives
    # ``rtol`` from it, and fp32's 1.3e-6 is far too tight for an fp16/bf16 result.
    utils.gems_assert_close(res_out.cpu(), ref_out.to(dtype), dtype, atol=_ATOL)


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", [(4, 1024, 64), (2, 4096, 32), (2, 8192, 16)])
def test_quantized_gru_cell_large_reduction(shape):
    """Large ``input_size``, where the accumulator exceeds fp32's exact range.

    At input_size 4096 the int32 GEMM accumulator reaches ~2.7e8, past 2**24,
    so an fp32 accumulator would silently drop low bits.  The kernel
    accumulates in int32 like FBGEMM.
    """
    ref_args, res_args = _build_case(shape, _ATEN_DTYPE)
    res_out = flag_gems.quantized_gru_cell(*res_args)
    ref_out = utils.to_reference(torch.quantized_gru_cell(*ref_args))
    utils.gems_assert_close(res_out.cpu(), ref_out, _ATEN_DTYPE, atol=_ATOL)


@pytest.mark.quantized_gru_cell
def test_quantized_gru_cell_non_contiguous():
    """ATen accepts strided operands; so must the kernel.

    The kernel indexes every operand linearly. Before the fix, a stride-2
    ``b_ih``/``b_hh`` was read as contiguous storage -- silently the wrong
    elements, worth ~1.5 absolute error.
    """
    shape = (6, 33, 17)
    ref_args, res_args = _build_case(shape, _ATEN_DTYPE)
    hidden_size = shape[2]
    gates = 3 * hidden_size
    dev = flag_gems.device

    # Strided biases: a stride-2 view of a 2*gates buffer, created on device so
    # that `.to(device)` cannot silently compact it.
    big_ih = torch.randn(2 * gates, device=dev)
    big_hh = torch.randn(2 * gates, device=dev)
    b_ih_nc = big_ih[::2]
    b_hh_nc = big_hh[::2]
    assert not b_ih_nc.is_contiguous()

    # Strided activations and weights.
    inp_nc = torch.randn(shape[1], shape[0], device=dev).T
    hx_nc = torch.randn(hidden_size, shape[0], device=dev).T
    w_ih_nc = res_args[2].T.contiguous().T
    w_hh_nc = res_args[3].T.contiguous().T
    assert not inp_nc.is_contiguous() and not w_ih_nc.is_contiguous()

    res_out = flag_gems.quantized_gru_cell(
        inp_nc,
        hx_nc,
        w_ih_nc,
        w_hh_nc,
        b_ih_nc,
        b_hh_nc,
        *res_args[6:],
    )
    ref_out = utils.to_reference(
        torch.quantized_gru_cell(
            inp_nc.cpu(),
            hx_nc.cpu(),
            w_ih_nc.cpu(),
            w_hh_nc.cpu(),
            b_ih_nc.cpu(),
            b_hh_nc.cpu(),
            *ref_args[6:],
        )
    )
    utils.gems_assert_close(res_out.cpu(), ref_out, _ATEN_DTYPE, atol=_ATOL)


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", [(0, 32, 16), (4, 0, 16)])
def test_quantized_gru_cell_zero_sized(shape):
    """Zero batch and zero ``input_size``. ATen accepts both."""
    batch_size, input_size, hidden_size = shape
    dev = flag_gems.device
    torch.manual_seed(0)
    w_hh_float = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32)
    w_hh_q, col_hh, scale_hh, zp_hh, _ = _make_quantized_weight(w_hh_float)

    w_ih_q = torch.zeros(3 * hidden_size, input_size, dtype=torch.int8, device=dev)
    col_ih = torch.zeros(3 * hidden_size, dtype=torch.int32, device=dev)
    empty = torch.empty(0, device=dev)

    out = flag_gems.quantized_gru_cell(
        torch.randn(batch_size, input_size, device=dev),
        torch.randn(batch_size, hidden_size, device=dev),
        w_ih_q,
        w_hh_q.to(dev),
        torch.randn(3 * hidden_size, device=dev),
        torch.randn(3 * hidden_size, device=dev),
        empty,
        empty,
        col_ih,
        col_hh.to(dev),
        0.1,
        scale_hh,
        0,
        zp_hh,
    )
    assert out.shape == (batch_size, hidden_size)
    assert torch.isfinite(out).all()


@pytest.mark.quantized_gru_cell
def test_quantized_gru_cell_validates_shapes():
    """Malformed shapes must raise, not read out of bounds.

    Every case below raises in ATen. Before the fix the kernel silently
    computed from mismatched extents via raw pointer arithmetic.
    """
    shape = (4, 32, 16)
    batch_size, input_size, hidden_size = shape
    _, args = _build_case(shape, _ATEN_DTYPE)
    args = list(args)
    dev = flag_gems.device

    def expect_raises(match, idx, value):
        mutated = list(args)
        mutated[idx] = value
        with pytest.raises(RuntimeError, match=match):
            flag_gems.quantized_gru_cell(*mutated)

    # input / hx rank
    expect_raises("expected 2-D input", 0, torch.randn(input_size, device=dev))
    expect_raises(
        "expected 2-D hx", 1, torch.randn(1, batch_size, hidden_size, device=dev)
    )
    # batch mismatch
    expect_raises("batch size", 1, torch.randn(batch_size + 3, hidden_size, device=dev))
    # weight shapes
    expect_raises("expected w_ih of shape", 2, args[2][: 2 * hidden_size])
    expect_raises("expected w_ih of shape", 2, args[2][:, :16].contiguous())
    expect_raises(
        "expected w_hh of shape", 3, args[3][:, : hidden_size // 2].contiguous()
    )
    expect_raises("expected 2-D w_ih", 2, args[2].flatten())
    # bias shapes
    expect_raises("expected b_ih of size", 4, args[4][: 2 * hidden_size])
    expect_raises("expected b_hh of size", 5, args[5][:hidden_size])
    expect_raises("expected 1-D b_ih", 4, args[4].view(3, hidden_size))
    # device mismatch
    expect_raises("same device", 1, args[1].cpu())
