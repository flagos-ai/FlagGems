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

from . import accuracy_utils as utils

# ``quantize_per_tensor`` only accepts float32 tensors on the quantized CUDA
# backend (Half/BFloat16/Double raise "Quantize only works on Float Tensor"),
# so there is no ``FLOAT_DTYPES`` parametrization here. The output is a
# quantized tensor whose ``int_repr`` matches the reference exactly (round to
# nearest, ties to even), hence ``gems_assert_equal`` on the int representation.
#
# The computation is fp32 end to end -- matching ATen, which narrows the double
# scale to float and multiplies by the fp32 reciprocal -- so the result does not
# depend on whether the device supports fp64.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]
QUANT_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [(), (1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
)
SCALES = [0.1, 0.01, 1.0]
# ``zero_point`` must lie within the representable integer range of *every*
# tested quantized dtype. quint8 covers [0, 255], qint8 covers [-128, 127] and
# qint32 covers the full int32 range, so the common range is [0, 127]. PyTorch
# validates this bound on ``quantize_per_tensor`` and rejects out-of-range values.
ZERO_POINTS = [0, 10, 64]


def _make_input(shape, device="cuda"):
    # Spread values across a wide range so that clamping to the integer range
    # is exercised alongside ordinary in-range values.
    return torch.randn(shape, dtype=torch.float32, device=device) * 3.0


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("zero_point", ZERO_POINTS)
def test_quantize_per_tensor(shape, in_dtype, scale, zero_point):
    res_inp = _make_input(shape)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor(ref_inp, scale, zero_point, in_dtype)
    res_out = flag_gems.quantize_per_tensor(res_inp, scale, zero_point, in_dtype)

    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    assert res_out.dtype == in_dtype
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()


@pytest.mark.quantize_per_tensor_out
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("zero_point", ZERO_POINTS)
def test_quantize_per_tensor_out(shape, in_dtype, scale, zero_point):
    res_inp = _make_input(shape)
    ref_inp = utils.to_reference(res_inp)

    # Pre-allocate a quantized `out` buffer with *different* scale/zero_point so
    # that we verify the kernel writes the passed parameters back onto it.
    res_out = torch.quantize_per_tensor(res_inp, 0.5, 100, in_dtype)
    ref_out = torch.quantize_per_tensor(ref_inp, 0.5, 100, in_dtype)

    ref_r = torch.ops.aten.quantize_per_tensor.out(
        ref_inp, scale, zero_point, in_dtype, out=ref_out
    )
    res_r = flag_gems.quantize_per_tensor_out(
        res_inp, scale, zero_point, in_dtype, out=res_out
    )

    assert res_r is res_out
    utils.gems_assert_equal(res_r.int_repr(), ref_r.int_repr())
    assert res_r.q_scale() == ref_r.q_scale()
    assert res_r.q_zero_point() == ref_r.q_zero_point()


@pytest.mark.quantize_per_tensor
@pytest.mark.skipif(
    utils.TO_CPU,
    reason="half-way quotients are backend-specific: CPU aten rounds the fp32 "
    "product, CUDA aten rounds the fp64 quotient, and the two disagree by 1. "
    "This kernel targets CUDA, so the comparison is only meaningful there.",
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantize_per_tensor_half_way_values(in_dtype):
    """Values whose quotient lands exactly on ``k + 0.5``.

    Half-way quotients are the only input class that pins down the precision of
    the division and the position of the ``zero_point`` add; uniformly random
    inputs almost never produce one. A non-zero ``zero_point`` is essential here,
    since adding it before rounding rather than after only changes the result at
    a tie.

    Skipped under ``--ref=cpu``: at exactly these inputs the two aten backends
    genuinely disagree (for x=0.85, scale=0.1 CPU gives 8 and CUDA gives 9), so no
    single kernel can be bit-exact against both. Every other test in this file
    passes under either reference, since random inputs essentially never tie.
    """
    scale = 0.14897697696685788
    ks = torch.arange(-400, 400, dtype=torch.float64) + 0.5
    res_inp = (ks * scale).to(torch.float32).cuda()

    ref_out = torch.quantize_per_tensor(res_inp, scale, 5, in_dtype)
    res_out = flag_gems.quantize_per_tensor(res_inp, scale, 5, in_dtype)
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("layout", ["transpose", "slice", "narrow"])
def test_quantize_per_tensor_non_contiguous(in_dtype, layout):
    base = _make_input((64, 64))
    view = {
        "transpose": lambda t: t.t(),
        "slice": lambda t: t[:, ::2],
        "narrow": lambda t: t[8:24, 4:20],
    }[layout](base)
    assert not view.is_contiguous()
    ref_view = utils.to_reference(view)

    ref_out = torch.quantize_per_tensor(ref_view, 0.05, 7, in_dtype)
    res_out = flag_gems.quantize_per_tensor(view, 0.05, 7, in_dtype)
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    assert tuple(res_out.shape) == tuple(view.shape)


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("bad_dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_quantize_per_tensor_rejects_non_float32(bad_dtype):
    # ATen raises "Quantize only works on Float Tensor" rather than upcasting.
    inp = torch.randn(32, dtype=bad_dtype, device="cuda")
    with pytest.raises(RuntimeError, match="torch.float32"):
        flag_gems.quantize_per_tensor(inp, 0.1, 0, torch.quint8)


@pytest.mark.quantize_per_tensor
def test_quantize_per_tensor_empty():
    inp = torch.empty(0, dtype=torch.float32, device="cuda")
    ref_out = torch.quantize_per_tensor(utils.to_reference(inp), 0.1, 0, torch.quint8)
    res_out = flag_gems.quantize_per_tensor(inp, 0.1, 0, torch.quint8)
    assert res_out.numel() == 0
    assert res_out.dtype == ref_out.dtype
    assert res_out.q_scale() == ref_out.q_scale()


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_dtype_mismatch():
    inp = _make_input((8, 8))
    # Requesting quint8 while handing in a qint8 buffer: ATen validates the out
    # dtype against the *requested* quantized dtype and rejects the mismatch.
    out = torch._empty_affine_quantized(
        (8, 8), scale=0.1, zero_point=0, dtype=torch.qint8, device="cuda"
    )
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems.quantize_per_tensor_out(inp, 0.1, 0, torch.quint8, out=out)


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_shape_mismatch():
    inp = _make_input((8, 8))
    out = torch._empty_affine_quantized(
        (2, 2), scale=0.1, zero_point=0, dtype=torch.quint8, device="cuda"
    )
    # A shape mismatch is a resize in ATen, but ``aten::resize_`` has no
    # QuantizedCUDA kernel, so the native op fails here too.
    with pytest.raises(RuntimeError):
        flag_gems.quantize_per_tensor_out(inp, 0.1, 0, torch.quint8, out=out)
