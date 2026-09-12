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

# quantize_per_channel expects a Float (float32) input -- float16/bfloat16 are
# rejected by PyTorch, so we parametrize over the *quantized* output dtypes
# (quint8 / qint8 / qint32) rather than utils.FLOAT_DTYPES and keep the input
# dtype fixed at float32, plus over the channel `axis`.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Shapes spanning 2D / 3D / 4D / 5D with a couple of large reduction-like dims.
QUANT_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [
        (8, 16),
        (2, 3, 4),
        (64, 128),
        (1024, 1024),
        (16, 128, 64),
        (16, 7, 57, 32),
    ]
)


def _make_inputs(shape, axis, device=flag_gems.device):
    """Build float32 input plus matching per-channel scales/zero_points."""
    inp = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    n_channels = shape[axis]
    scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    # Keep zero_points within the valid range of every quantized dtype so the
    # same parametrization covers quint8 / qint8 / qint32.
    zero_points = torch.randint(0, 50, (n_channels,), device=device, dtype=torch.int32)
    return inp, scales, zero_points


def _assert_quantized_equal(res, ref):
    """Compare two per-channel quantized tensors.

    The scale/zero_point/axis are exact parameters, so they must match
    exactly. The integer representation is derived via ``nearbyint(x /
    scale)``; CPU and GPU compute the fp64 division identically, but a few
    values on a rounding boundary (``.5``) can round to the adjacent bin
    depending on the device's last-bit rounding. That is a benign 1-ULP
    boundary effect, so ``int_repr`` is compared with ``atol=1``.
    """
    assert res.dtype == ref.dtype, f"dtype mismatch: {res.dtype} vs {ref.dtype}"
    assert res.q_per_channel_axis() == ref.q_per_channel_axis(), "axis mismatch"
    utils.gems_assert_equal(res.q_per_channel_scales(), ref.q_per_channel_scales())
    utils.gems_assert_equal(
        res.q_per_channel_zero_points(), ref.q_per_channel_zero_points()
    )
    # int_repr is uint8/int8/int32; compare in float32 with atol=1 to absorb
    # the rounding-boundary 1-ULP diff between CPU and GPU references.
    utils.gems_assert_close(
        res.int_repr().to(torch.float32),
        ref.int_repr().to(torch.float32),
        torch.float32,
        atol=1,
        equal_nan=True,
    )


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel(shape, axis, dtype):
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    inp, scales, zero_points = _make_inputs(shape, axis)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.quantize_per_channel(
        ref_inp, ref_scales, ref_zero_points, axis, dtype
    )
    # GEMS direct call: the kernel computes the per-channel quantization on the
    # accelerator with fp64 division to match PyTorch's accuracy.
    res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_last_axis(shape, dtype):
    """Exercise the last (innermost) channel axis specifically."""
    axis = len(shape) - 1
    inp, scales, zero_points = _make_inputs(shape, axis)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.quantize_per_channel(
        ref_inp, ref_scales, ref_zero_points, axis, dtype
    )
    res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_extremes(dtype):
    """Inputs that exercise clamping: very large, very small, zeros, and inf."""
    # 4 channels x 32 spatial elements per channel
    shape = (4, 32)
    axis = 1
    n_channels = shape[axis]
    scales = torch.full((n_channels,), 0.1, device=flag_gems.device)
    zero_points = torch.zeros(n_channels, device=flag_gems.device, dtype=torch.int32)

    base = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    cases = {
        "zeros": torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        "large": torch.full(shape, 1e4, dtype=torch.float32, device=flag_gems.device),
        "neg_large": torch.full(
            shape, -1e4, dtype=torch.float32, device=flag_gems.device
        ),
        "with_inf": torch.where(base > 0, base, torch.full_like(base, float("inf"))),
    }

    for name, inp in cases.items():
        ref_inp = utils.to_reference(inp)
        ref_scales = utils.to_reference(scales)
        ref_zero_points = utils.to_reference(zero_points)
        ref_out = torch.quantize_per_channel(
            ref_inp, ref_scales, ref_zero_points, axis, dtype
        )
        res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)
        _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_round_trip(shape, axis, dtype):
    """dequantize(quantize(x)) should match torch for the round-trip values."""
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    inp, scales, zero_points = _make_inputs(shape, axis)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    res_q = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)
    ref_q = torch.quantize_per_channel(
        ref_inp, ref_scales, ref_zero_points, axis, dtype
    )

    # `dequantize()` outside the GEMS call so it goes through torch's native
    # per-channel dequantize (the GEMS dequantize op only supports per-tensor).
    res_out = res_q.dequantize()
    ref_out = ref_q.dequantize()

    utils.gems_assert_close(res_out, ref_out, dtype=torch.float32)


@pytest.mark.quantize_per_channel_out
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_out(shape, axis, dtype):
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    inp, scales, zero_points = _make_inputs(shape, axis)

    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.quantize_per_channel(
        ref_inp, ref_scales, ref_zero_points, axis, dtype
    )

    # Pre-allocate a per-channel quantized ``out`` tensor via the public API
    # with matching shape/scales/zero_points/axis; the FlagGems out kernel
    # overwrites its integer storage.
    out_tensor = torch.quantize_per_channel(
        torch.zeros(shape, dtype=torch.float32, device=inp.device),
        scales.double().to(inp.device),
        zero_points.long().to(inp.device),
        axis,
        dtype,
    )
    res_r = flag_gems.quantize_per_channel_out(
        inp, scales, zero_points, axis, dtype, out=out_tensor
    )

    assert res_r is out_tensor
    _assert_quantized_equal(res_r, ref_out)
