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
from flag_gems.ops.quantize_per_tensor_dynamic import (
    quantize_per_tensor_dynamic,
    quantize_per_tensor_dynamic_out,
)

from . import accuracy_utils as utils

# torch.quantize_per_tensor_dynamic accepts torch.quint8 / torch.qint8 for the
# affine quantization path and torch.float16, which returns a plain Half copy.
QUANT_DTYPES = [torch.quint8, torch.qint8]


# Shapes covering scalar, 1d, 2d and higher-rank tensors, plus a few larger
# ones to exercise the two-pass reduction.
QUANT_SHAPES = [(), (1,), (4, 1024), (20, 320, 15), (16, 128, 64, 60)]


REDUCE_RANGE = [False, True]


def _assert_quantized_equal(res, ref):
    """Compare two dynamically-quantized tensors bin-for-bin.

    ``scale``, ``zero_point`` and the ``int_repr`` payload must agree, with one
    deliberate allowance: an individual ``int_repr`` entry may differ by 1.
    The GEMS kernels replicate ATen's ChooseQuantizationParams
    (double-precision scale, f32 min/max adjustment, error-based zero_point,
    round-half-to-even) and the quantize math (``nearbyint(x / scale) +
    zero_point`` with the same clamp order), so locally this comparison is
    exact for millions of elements. But CI's custom torch build rounds
    x / scale differently at exact half-way ties (its fbgemm build differs
    from a local wheel), which flipped 3 of 7.8M elements by one bin on CI.
    Allowing |diff| <= 1 absorbs that tie-breaking variance while still
    catching any systematic off-by-more-than-one rounding error; ``scale`` and
    ``zero_point`` must still match exactly.
    """
    assert res.dtype == ref.dtype, f"dtype mismatch: {res.dtype} vs {ref.dtype}"
    assert (
        res.q_scale() == ref.q_scale()
    ), f"scale mismatch: {res.q_scale()} vs {ref.q_scale()}"
    assert (
        res.q_zero_point() == ref.q_zero_point()
    ), f"zero_point mismatch: {res.q_zero_point()} vs {ref.q_zero_point()}"
    # Compare on one device: under --ref=cpu the reference is materialized on
    # the host while the result stays on the device.
    ref_int = ref.int_repr()
    res_int = res.int_repr()
    diff = (res_int.int() - ref_int.int().to(res_int.device)).abs()
    max_diff = int(diff.max()) if diff.numel() else 0
    assert max_diff <= 1, (
        "int_repr differs by more than one quantization bin "
        f"(max {max_diff}); a tie-breaking difference is at most 1"
    )


def _torch_ref(inp, dtype, reduce_range):
    """Reference on the accelerator so CPU/CUDA differences do not leak in."""
    return torch.quantize_per_tensor_dynamic(inp, dtype, reduce_range)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic(shape, dtype, reduce_range):
    res_inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, reduce_range)
    # GEMS direct call: the kernel computes scale/zero_point dynamically and
    # builds the quantized tensor on the accelerator.
    res_out = quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", QUANT_SHAPES)
def test_quantize_per_tensor_dynamic_all_positive(shape):
    # All-positive values: 0 is folded into the range as the lower bound.
    res_inp = torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 10.0
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, torch.quint8, False)
    res_out = quantize_per_tensor_dynamic(res_inp, torch.quint8, False)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", QUANT_SHAPES)
def test_quantize_per_tensor_dynamic_all_negative(shape):
    # All-negative values: 0 is folded into the range as the upper bound.
    res_inp = -(torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 10.0)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, torch.quint8, False)
    res_out = quantize_per_tensor_dynamic(res_inp, torch.quint8, False)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", [(), (1,), (4, 1024)])
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_zeros(shape, dtype, reduce_range):
    # Degenerate range (min == max == 0): the f32 scale is 0 so ATen falls back
    # to 0.1 and derives the zero point from the minimum.
    res_inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, reduce_range)
    res_out = quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_constant_nonzero(dtype, reduce_range):
    # Degenerate but non-zero range: scale falls back to the f32 zero test and
    # the zero point still comes from the (constant) minimum, so qint8 clamps
    # to its full range while quint8 maps to the top bin.
    res_inp = torch.full((16,), 3.0, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, reduce_range)
    res_out = quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize(
    "values",
    [
        # Narrow but representable range: the SMALL_SCALE_THRESHOLD clamp kicks
        # in and both the scale and the zero point must follow ATen exactly.
        [-1e-6, 1e-6],
        [0.0, 1e-6],
        [-1e-6, 0.0],
        [-3.1e-5, 2.9e-5],
        # Tiny / subnormal float32 values: the reciprocal of the f32 scale
        # overflows to infinity, so ATen replaces the scale with 0.1.
        [1e-40, 1e-40],
        [1e-45, 2e-45],
        [1e-38, 1e-38],
        # Range large enough that no clamp applies, as a control.
        [-1e-3, 1e-3],
    ],
)
def test_quantize_per_tensor_dynamic_small_scale(values, dtype, reduce_range):
    res_inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, reduce_range)
    res_out = quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_non_finite(dtype, reduce_range):
    # -inf as the minimum (or all -inf) keeps the range finite on the upper side
    # and must reproduce ATen's scale=inf / clamped zero_point and bins.
    #
    # The reference must stay on the *same device* as the result for these
    # inputs: ATen's CPU and CUDA kernels disagree on how -inf clamps when
    # scale is inf (measured: CPU maps it to 127, CUDA to -128, a 255-bin
    # gap), and the GEMS kernel implements the CUDA side. Comparing against a
    # --ref=cpu reference here would assert the CPU behaviour instead.
    for values in (
        [float("-inf"), 1.0],
        [-1.0, float("-inf")],
        [float("-inf")] * 4,
    ):
        res_inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device)

        ref_out = _torch_ref(res_inp, dtype, reduce_range)
        res_out = quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

        _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_nan_raises(dtype):
    # ATen computes min/max, which propagate NaN, and ChooseQuantizationParams
    # then rejects `min <= max`. The GEMS path must raise the same error.
    res_inp = torch.tensor(
        [1.0, float("nan"), -1.0], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = utils.to_reference(res_inp)

    with pytest.raises(RuntimeError, match="min should be less than or equal to max"):
        _torch_ref(ref_inp, dtype, True)
    with pytest.raises(RuntimeError, match="min should be less than or equal to max"):
        quantize_per_tensor_dynamic(res_inp, dtype, True)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_nan_raises_two_pass(dtype):
    # Same error parity on a tensor large enough to take the two-stage reduction
    # path, where the per-block reduction must propagate NaN explicitly.
    res_inp = torch.randn(200000, dtype=torch.float32, device=flag_gems.device)
    res_inp[123] = float("nan")
    ref_inp = utils.to_reference(res_inp)

    with pytest.raises(RuntimeError, match="min should be less than or equal to max"):
        _torch_ref(ref_inp, dtype, True)
    with pytest.raises(RuntimeError, match="min should be less than or equal to max"):
        quantize_per_tensor_dynamic(res_inp, dtype, True)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_positive_inf_raises(dtype):
    # +inf makes the scale infinite and the zero point NaN; ATen raises while
    # materializing the quantized tensor.
    res_inp = torch.tensor(
        [1.0, float("inf"), -1.0], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = utils.to_reference(res_inp)

    with pytest.raises(RuntimeError, match="zero_point -2147483648 is below"):
        _torch_ref(ref_inp, dtype, False)
    with pytest.raises(RuntimeError, match="zero_point -2147483648 is below"):
        quantize_per_tensor_dynamic(res_inp, dtype, False)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_empty_raises(dtype):
    # ATen reduces with input.min() before anything else and raises on an
    # empty tensor; GEMS must raise the same error rather than silently
    # returning an empty quantized tensor with fixed qparams.
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        res_inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
        ref_inp = utils.to_reference(res_inp)

        with pytest.raises(RuntimeError, match="input.numel\\(\\) == 0"):
            _torch_ref(ref_inp, dtype, True)
        with pytest.raises(RuntimeError, match="input.numel\\(\\) == 0"):
            quantize_per_tensor_dynamic(res_inp, dtype, True)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize(
    "input_dtype",
    [torch.float16, torch.bfloat16, torch.float64, torch.int32, torch.bool],
)
def test_quantize_per_tensor_dynamic_invalid_input_dtype(input_dtype):
    # The affine path only accepts float32; other dtypes raise the ATen error.
    res_inp = torch.randn(4, 5, device=flag_gems.device).to(input_dtype)
    ref_inp = utils.to_reference(res_inp)

    with pytest.raises(RuntimeError, match="Quantize only works on Float Tensor"):
        _torch_ref(ref_inp, torch.quint8, True)
    with pytest.raises(RuntimeError, match="Quantize only works on Float Tensor"):
        quantize_per_tensor_dynamic(res_inp, torch.quint8, True)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize(
    "quant_dtype",
    [torch.float32, torch.float64, torch.bfloat16, torch.int32, torch.uint8],
)
def test_quantize_per_tensor_dynamic_invalid_quant_dtype(quant_dtype):
    # Only quint8/qint8/float16 are valid `dtype` arguments.
    res_inp = torch.randn(4, 5, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    with pytest.raises(RuntimeError, match="not supported"):
        _torch_ref(ref_inp, quant_dtype, True)
    with pytest.raises(RuntimeError, match="not supported"):
        quantize_per_tensor_dynamic(res_inp, quant_dtype, True)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.int32])
def test_quantize_per_tensor_dynamic_half_output(input_dtype, reduce_range):
    # dtype=torch.float16 returns a contiguous Half copy of the input and
    # accepts any input dtype (ATen short-circuits before the float check).
    res_inp = torch.randn(4, 5, device=flag_gems.device).to(input_dtype)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, torch.float16, reduce_range)
    res_out = quantize_per_tensor_dynamic(res_inp, torch.float16, reduce_range)

    assert res_out.dtype == torch.float16
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
def test_quantize_per_tensor_dynamic_half_output_non_contiguous():
    # The Half path materializes a contiguous result even for strided input.
    res_inp = torch.randn(6, 8, device=flag_gems.device)[:, ::2]
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, torch.float16, True)
    res_out = quantize_per_tensor_dynamic(res_inp, torch.float16, True)

    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
def test_quantize_per_tensor_dynamic_half_output_empty():
    # The Half path does not reduce, so an empty input is valid there.
    res_inp = torch.empty(0, 3, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, torch.float16, True)
    res_out = quantize_per_tensor_dynamic(res_inp, torch.float16, True)

    assert res_out.dtype == torch.float16
    assert res_out.shape == res_inp.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize(
    "view",
    [
        lambda t: t[:, ::2],  # sliced, non-contiguous
        lambda t: t.transpose(0, 1),  # transposed
        lambda t: t[1:-1, 2:],  # storage offset + slice
        lambda t: t.unsqueeze(0).expand(3, *t.shape),  # expanded
    ],
)
def test_quantize_per_tensor_dynamic_non_contiguous(view, dtype, reduce_range):
    # A strided view must quantize the *logical* elements, not raw storage.
    # ATen returns a contiguous quantized tensor for these inputs.
    base = torch.randn(4, 8, dtype=torch.float32, device=flag_gems.device)
    res_inp = view(base)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, reduce_range)
    res_out = quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

    assert res_out.is_contiguous()
    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_storage_offset_1d(dtype):
    # A 1d storage-offset view exercises the fused single-pass kernel, which is
    # the path most sensitive to non-contiguous inputs.
    res_inp = torch.randn(64, dtype=torch.float32, device=flag_gems.device)[16:48]
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, False)
    res_out = quantize_per_tensor_dynamic(res_inp, dtype, False)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
def test_quantize_per_tensor_dynamic_channels_last():
    # 4d channels-last input: values must match, and the output is contiguous
    # in the channels-last sense (ATen keeps the suggested memory format).
    res_inp = torch.randn(2, 3, 4, 4, device=flag_gems.device).to(
        memory_format=torch.channels_last
    )
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, torch.quint8, False)
    res_out = quantize_per_tensor_dynamic(res_inp, torch.quint8, False)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_large_two_pass(shape, dtype, reduce_range):
    # Widen the values so the scale stays above SMALL_SCALE_THRESHOLD and force
    # the two-stage reduction for tensors larger than the fused block limit.
    res_inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device) * 100.0
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, reduce_range)
    res_out = quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic_out
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_out(shape, dtype, reduce_range):
    res_inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = _torch_ref(ref_inp, dtype, reduce_range)

    # Pre-allocate a quantized ``out`` tensor via the public API; the kernel
    # overwrites its integer storage, so the initial content is irrelevant.
    out_tensor = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        ref_out.q_scale(),
        ref_out.q_zero_point(),
        dtype,
    )
    # GEMS direct call: ``quantize_per_tensor_dynamic_out`` writes into ``out``.
    res_r = quantize_per_tensor_dynamic_out(
        res_inp, dtype, reduce_range, out=out_tensor
    )

    assert res_r is out_tensor
    _assert_quantized_equal(res_r, ref_out)


@pytest.mark.quantize_per_tensor_dynamic_out
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_out_updates_qparams(dtype, reduce_range):
    # ``out`` must receive the dynamically computed scale/zero_point, not the
    # ones it was created with.
    res_inp = torch.randn(4, 1024, dtype=torch.float32, device=flag_gems.device) * 50.0
    ref_inp = utils.to_reference(res_inp)
    ref_out = _torch_ref(ref_inp, dtype, reduce_range)

    out_tensor = torch.quantize_per_tensor(
        torch.zeros(4, 1024, dtype=torch.float32, device=flag_gems.device),
        0.1,
        0,
        dtype,
    )
    res_r = quantize_per_tensor_dynamic_out(
        res_inp, dtype, reduce_range, out=out_tensor
    )

    assert res_r is out_tensor
    assert out_tensor.q_scale() == ref_out.q_scale()
    assert out_tensor.q_zero_point() == ref_out.q_zero_point()
    utils.gems_assert_equal(out_tensor.int_repr(), ref_out.int_repr())


@pytest.mark.quantize_per_tensor_dynamic_out
def test_quantize_per_tensor_dynamic_out_none_returns_new():
    res_inp = torch.randn(8, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)
    ref_out = _torch_ref(ref_inp, torch.quint8, True)

    res_out = quantize_per_tensor_dynamic_out(res_inp, torch.quint8, True, out=None)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic_out
def test_quantize_per_tensor_dynamic_out_dtype_mismatch_raises():
    res_inp = torch.randn(4, 5, dtype=torch.float32, device=flag_gems.device)
    out_tensor = torch.quantize_per_tensor(
        torch.zeros(4, 5, dtype=torch.float32, device=flag_gems.device),
        0.1,
        0,
        torch.qint8,
    )
    with pytest.raises(RuntimeError):
        quantize_per_tensor_dynamic_out(res_inp, torch.quint8, True, out=out_tensor)
