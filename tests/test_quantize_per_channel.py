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

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# quantize_per_channel expects a Float (float32) input -- float16/bfloat16 are
# rejected by PyTorch, so we parametrize over the *quantized* output dtypes
# (quint8 / qint8 / qint32) rather than utils.FLOAT_DTYPES and keep the input
# dtype fixed at float32, plus over the channel `axis`.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# The float-zero-point scheme keeps its zero-points in float32. Native PyTorch
# only accepts a float zero-point *inside the integer range of the target
# dtype*, so quint8 requires [0, 255) and qint8 [-128, 127).
FLOAT_QPARAM_DTYPES = [torch.quint8, torch.qint8]

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


def _pow2_scales(n_channels, device=flag_gems.device):
    """Per-channel scales that are exact powers of two.

    ``quantize_per_channel`` derives its result from a division by the scale.
    A power-of-two scale is exact in both the fp32 reciprocal the device kernel
    uses and the fp64 division the CPU kernel uses, so every implementation
    rounds the identical real value. That makes ``int_repr()`` comparison able
    to be exact instead of tolerant, for the arbitrary-scale cases as well as
    for the rounding-tie cases below.
    """
    exponents = torch.randint(-3, 4, (n_channels,), device=device)
    return torch.pow(2.0, exponents.to(torch.float32))


def _make_inputs(shape, axis, device=flag_gems.device, pow2_scales=False):
    """Build float32 input plus matching per-channel scales/zero_points."""
    inp = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    n_channels = shape[axis]
    if pow2_scales:
        scales = _pow2_scales(n_channels, device)
    else:
        scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    # Keep zero_points within the valid range of every quantized dtype so the
    # same parametrization covers quint8 / qint8 / qint32.
    zero_points = torch.randint(0, 50, (n_channels,), device=device, dtype=torch.int32)
    return inp, scales, zero_points


def _make_float_zero_points(n_channels, dtype, device=flag_gems.device):
    """Float zero-points inside ``dtype``'s integer range.

    ``per_channel_affine_float_qparams`` still requires the zero-point to lie
    within the quantized dtype's range (ATen checks it in the kernel), so
    quint8 needs a non-negative value and qint8 a signed one.
    """
    if dtype == torch.quint8:
        return torch.rand(n_channels, device=device)
    return (torch.rand(n_channels, device=device) - 0.5) * 200.0


def _assert_quantized_identical(res, ref):
    """Compare two per-channel quantized tensors exactly.

    The scale/zero_point/axis are exact parameters and ``int_repr`` is the
    stored integer payload, so both are compared bit-for-bit:

    * ``float32`` rounding of qint32 payloads is lossy -- 16777217.0f is
      16777216.0f -- and ``atol=1`` would also hide a genuine one-bin
      quantization difference at large magnitudes. Comparing the integer
      tensors directly avoids both problems.
    * ``q_per_channel_zero_points`` is compared including its dtype, which is
      how ``per_channel_affine`` (int64) and
      ``per_channel_affine_float_qparams`` (float32) are told apart.
    """
    assert res.dtype == ref.dtype, f"dtype mismatch: {res.dtype} vs {ref.dtype}"
    assert (
        res.qscheme() == ref.qscheme()
    ), f"qscheme mismatch: {res.qscheme()} vs {ref.qscheme()}"
    assert res.q_per_channel_axis() == ref.q_per_channel_axis(), "axis mismatch"
    utils.gems_assert_equal(res.q_per_channel_scales(), ref.q_per_channel_scales())
    res_zp = res.q_per_channel_zero_points()
    ref_zp = ref.q_per_channel_zero_points()
    assert (
        res_zp.dtype == ref_zp.dtype
    ), f"zero_point dtype mismatch: {res_zp.dtype} vs {ref_zp.dtype}"
    utils.gems_assert_equal(res_zp, ref_zp)
    # int_repr is uint8/int8/int32; compare the integers themselves.
    utils.gems_assert_equal(res.int_repr(), ref.int_repr())


def _reference(inp, scales, zero_points, axis, dtype):
    """Reference quantized tensor, on the device selected by ``--ref``."""
    return torch.quantize_per_channel(
        utils.to_reference(inp),
        utils.to_reference(scales),
        utils.to_reference(zero_points),
        axis,
        dtype,
    )


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel(shape, axis, dtype):
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    # Power-of-two scales keep the comparison exact even under --ref=cpu.
    inp, scales, zero_points = _make_inputs(shape, axis, pow2_scales=True)

    ref_out = _reference(inp, scales, zero_points, axis, dtype)
    # GEMS direct call: the kernel computes the per-channel quantization on the
    # accelerator with fp64 division to match PyTorch's accuracy.
    res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)

    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_last_axis(shape, dtype):
    """Exercise the last (innermost) channel axis specifically."""
    axis = len(shape) - 1
    inp, scales, zero_points = _make_inputs(shape, axis, pow2_scales=True)

    ref_out = _reference(inp, scales, zero_points, axis, dtype)
    res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)

    _assert_quantized_identical(res_out, ref_out)


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
        ref_out = _reference(inp, scales, zero_points, axis, dtype)
        res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)
        _assert_quantized_identical(res_out, ref_out)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_tie_cases(dtype):
    """Explicit +/- half-way values must round half-to-even, like PyTorch.

    With a unit scale the input *is* the pre-rounding value, so each listed
    input lands exactly on a ``.5`` boundary and the expected integer is the
    even neighbour (``nearbyint`` semantics): 0.5 -> 0, 1.5 -> 2, 2.5 -> 2,
    -0.5 -> 0, -1.5 -> -2, -2.5 -> -2.
    """
    values = [-7.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 6.5, 7.5]
    inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device).reshape(
        1, len(values)
    )
    n_channels = len(values)
    scales = torch.ones(n_channels, device=flag_gems.device)
    zero_points = torch.zeros(n_channels, device=flag_gems.device, dtype=torch.int32)

    ref_out = _reference(inp, scales, zero_points, 1, dtype)
    res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, 1, dtype)

    _assert_quantized_identical(res_out, ref_out)

    # The expected half-to-even payload, independent of the reference.
    expected = [int(torch.round(torch.tensor(v))) for v in values]
    observed = res_out.int_repr().flatten().tolist()
    for value, want, got in zip(values, expected, observed):
        if dtype == torch.quint8 and want < 0:
            # Clamped into quint8's [0, 255] range.
            want = 0
        assert got == want, f"{value} rounded to {got}, expected {want} (half-to-even)"


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_round_trip(shape, axis, dtype):
    """dequantize(quantize(x)) should match torch for the round-trip values."""
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    inp, scales, zero_points = _make_inputs(shape, axis)

    res_q = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)
    ref_q = _reference(inp, scales, zero_points, axis, dtype)

    # `dequantize()` outside the GEMS call so it goes through torch's native
    # per-channel dequantize (the GEMS dequantize op only supports per-tensor).
    res_out = res_q.dequantize()
    ref_out = ref_q.dequantize()

    utils.gems_assert_close(res_out, ref_out, dtype=torch.float32)


# ---------------------------------------------------------------------------
# per_channel_affine_float_qparams
# ---------------------------------------------------------------------------


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_QPARAM_DTYPES)
def test_quantize_per_channel_float_zero_points(shape, axis, dtype):
    """A floating-point zero_points must select the float-qparams scheme.

    That scheme stores both scales and zero-points as float32, reports
    ``torch.per_channel_affine_float_qparams``, and rounds
    ``x * (1/scale) + zero_point`` before clamping -- a different formula from
    the integer scheme, which rounds ``x / scale`` and then adds the
    zero-point.
    """
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    inp, scales, _ = _make_inputs(shape, axis, pow2_scales=True)
    zero_points = _make_float_zero_points(shape[axis], dtype)

    ref_out = _reference(inp, scales, zero_points, axis, dtype)
    res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, axis, dtype)

    assert (
        res_out.qscheme() == torch.per_channel_affine_float_qparams
    ), f"expected float qparams, got {res_out.qscheme()}"
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("dtype", FLOAT_QPARAM_DTYPES)
def test_quantize_per_channel_float_zero_points_tie_cases(dtype):
    """Ties in the float-qparams formula round half-to-even.

    The native kernel adds the zero-point *before* rounding, so a zero-point
    that itself sits on a ``.5`` boundary -- and an ``x / scale`` landing on one
    -- must both fall to the even neighbour. Unit scale keeps ``x / scale``
    exact, so the pre-rounding value is ``x + zero_point``.
    """
    values = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device).reshape(
        1, len(values)
    )
    n_channels = len(values)
    scales = torch.ones(n_channels, device=flag_gems.device)
    zero_points = torch.full((n_channels,), 0.5, device=flag_gems.device)

    ref_out = _reference(inp, scales, zero_points, 1, dtype)
    res_out = flag_gems.quantize_per_channel(inp, scales, zero_points, 1, dtype)

    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("dtype", FLOAT_QPARAM_DTYPES)
def test_quantize_per_channel_float_zero_points_qscheme_differs(dtype):
    """The same scales with integer vs float zero-points give different results.

    Guards against the float path being silently collapsed into the integer
    one (the two schemes round different expressions).
    """
    shape = (4, 64)
    axis = 1
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device) * 10.0
    scales = torch.full((shape[axis],), 0.1, device=flag_gems.device)

    float_zp = torch.full((shape[axis],), 0.9, device=flag_gems.device)
    int_zp = torch.full((shape[axis],), 1, device=flag_gems.device, dtype=torch.int32)

    float_out = flag_gems.quantize_per_channel(inp, scales, float_zp, axis, dtype)
    int_out = flag_gems.quantize_per_channel(inp, scales, int_zp, axis, dtype)

    assert float_out.qscheme() == torch.per_channel_affine_float_qparams
    assert int_out.qscheme() == torch.per_channel_affine
    assert not torch.equal(float_out.int_repr(), int_out.int_repr())


# ---------------------------------------------------------------------------
# out= overload
# ---------------------------------------------------------------------------


def _empty_per_channel_quantized(shape, scales, zero_points, axis, dtype):
    """Allocate an uninitialised per-channel quantized tensor."""
    return torch.ops.aten._empty_per_channel_affine_quantized(
        list(shape),
        scales=scales.clone(),
        zero_points=zero_points.clone(),
        axis=axis,
        dtype=dtype,
        device=scales.device,
    )


@pytest.mark.quantize_per_channel_out
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_out(shape, axis, dtype):
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    inp, scales, zero_points = _make_inputs(shape, axis, pow2_scales=True)

    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)
    ref_out = torch.quantize_per_channel(
        ref_inp, ref_scales, ref_zero_points, axis, dtype
    )

    # Freshly allocated ``out`` with deliberately wrong qparams: the native out=
    # overwrites the destination's integer storage *and* adopts the new
    # scale/zero_point/axis, so the stale values must not survive.
    stale_scales = scales.double() * 7.0
    stale_zps = torch.full((shape[axis],), 3, device=inp.device, dtype=torch.int64)
    out_tensor = _empty_per_channel_quantized(
        shape, stale_scales, stale_zps, axis, dtype
    )
    res_r = flag_gems.quantize_per_channel_out(
        inp, scales, zero_points, axis, dtype, out=out_tensor
    )

    assert res_r is out_tensor, "out= must return the same object"
    _assert_quantized_identical(res_r, ref_out)


@pytest.mark.quantize_per_channel_out
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_channel_out_matches_aten_out(dtype):
    """The out= overload must agree with ``torch.ops.aten.quantize_per_channel.out``.

    Runs both overloads on identically-initialised out tensors and compares the
    payload, the returned object identity, and the adopted qparams.
    """
    shape = (8, 32)
    axis = 0
    inp, scales, zero_points = _make_inputs(shape, axis, pow2_scales=True)

    gems_out = _empty_per_channel_quantized(
        shape, scales.double(), zero_points.long(), axis, dtype
    )
    aten_out = _empty_per_channel_quantized(
        shape, scales.double(), zero_points.long(), axis, dtype
    )

    res_r = flag_gems.quantize_per_channel_out(
        inp, scales, zero_points, axis, dtype, out=gems_out
    )
    ref_r = torch.ops.aten.quantize_per_channel.out(
        inp, scales, zero_points, axis, dtype, out=aten_out
    )

    assert res_r is gems_out
    assert ref_r is aten_out
    assert torch.equal(res_r.int_repr(), ref_r.int_repr())
    assert torch.equal(res_r.q_per_channel_scales(), ref_r.q_per_channel_scales())
    assert torch.equal(
        res_r.q_per_channel_zero_points(), ref_r.q_per_channel_zero_points()
    )
    assert res_r.q_per_channel_axis() == ref_r.q_per_channel_axis()
    assert res_r.qscheme() == ref_r.qscheme()


@pytest.mark.quantize_per_channel_out
def test_quantize_per_channel_out_rejects_wrong_dtype():
    """An ``out`` of the wrong quantized dtype fails like native ``out=``."""
    shape = (4, 8)
    axis = 0
    inp, scales, zero_points = _make_inputs(shape, axis)
    out_tensor = _empty_per_channel_quantized(
        shape, scales.double(), zero_points.long(), axis, torch.qint8
    )

    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        flag_gems.quantize_per_channel_out(
            inp, scales, zero_points, axis, torch.quint8, out=out_tensor
        )


@pytest.mark.quantize_per_channel_out
def test_quantize_per_channel_out_noncontiguous():
    """A strided ``out`` view is written through its own layout."""
    shape = (4, 12)
    axis = 0
    inp, scales, zero_points = _make_inputs(shape, axis, pow2_scales=True)

    big = _empty_per_channel_quantized(
        (4, 24), scales.double(), zero_points.long(), axis, torch.quint8
    )
    view = big[:, ::2]
    assert not view.is_contiguous()

    ref_out = torch.quantize_per_channel(
        utils.to_reference(inp),
        utils.to_reference(scales),
        utils.to_reference(zero_points),
        axis,
        torch.quint8,
    )
    res_r = flag_gems.quantize_per_channel_out(
        inp, scales, zero_points, axis, torch.quint8, out=view
    )
    assert res_r is view
    utils.gems_assert_equal(res_r.int_repr(), ref_out.int_repr())


# ---------------------------------------------------------------------------
# input-contract validation
# ---------------------------------------------------------------------------


@pytest.mark.quantize_per_channel
def test_quantize_per_channel_rejects_non_float_input():
    """Native per-channel quantization only accepts float32."""
    inp = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    scales = torch.rand(4, dtype=torch.float64, device=flag_gems.device) + 0.1
    zero_points = torch.zeros(4, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="expects a Float Tensor, got Half"):
        flag_gems.quantize_per_channel(inp, scales, zero_points, 0, torch.quint8)


@pytest.mark.quantize_per_channel
def test_quantize_per_channel_rejects_float_zero_point_non_float_input():
    """The float-qparams quantizer phrases its dtype check differently."""
    inp = torch.randn(4, 8, dtype=torch.float16, device=flag_gems.device)
    scales = torch.rand(4, dtype=torch.float64, device=flag_gems.device) + 0.1
    zero_points = torch.rand(4, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="Quantize only works on Float Tensor"):
        flag_gems.quantize_per_channel(inp, scales, zero_points, 0, torch.quint8)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize("axis", [-1, 2, 5])
def test_quantize_per_channel_rejects_bad_axis(axis):
    """Out-of-range (including negative) axes are rejected, as ATen does."""
    inp = torch.randn(4, 8, device=flag_gems.device)
    scales = torch.rand(4, dtype=torch.float64, device=flag_gems.device) + 0.1
    zero_points = torch.zeros(4, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="Channel axis out of range"):
        flag_gems.quantize_per_channel(inp, scales, zero_points, axis, torch.quint8)


@pytest.mark.quantize_per_channel
def test_quantize_per_channel_rejects_multi_dim_qparams():
    """scales/zero_points must be 1-D vectors."""
    inp = torch.randn(4, 8, device=flag_gems.device)
    scales_2d = torch.rand(1, 4, dtype=torch.float64, device=flag_gems.device) + 0.1
    zero_points = torch.zeros(4, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="scale tensor must have dimension 1"):
        flag_gems.quantize_per_channel(inp, scales_2d, zero_points, 0, torch.quint8)

    scales = torch.rand(4, dtype=torch.float64, device=flag_gems.device) + 0.1
    zero_points_2d = torch.zeros(1, 4, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="zero_points tensor must have dimension 1"):
        flag_gems.quantize_per_channel(inp, scales, zero_points_2d, 0, torch.quint8)


@pytest.mark.quantize_per_channel
def test_quantize_per_channel_rejects_short_qparams():
    """Short qparam vectors are rejected before the kernel reads past the end.

    The kernel indexes ``scales``/``zero_points`` by channel, so a vector
    shorter than ``input.size(axis)`` used to load out of bounds.
    """
    inp = torch.randn(4, 8, device=flag_gems.device)
    scales = torch.rand(4, dtype=torch.float64, device=flag_gems.device) + 0.1
    zero_points = torch.zeros(4, dtype=torch.int64, device=flag_gems.device)

    # A zero_points vector of a *different* length is caught by the
    # parameter-dimension check first, which is the order ATen applies them in
    # (``checkPerChannelParamDims`` runs before per-channel sizing).
    with pytest.raises(
        RuntimeError, match="number of elements in scales and zero_points must match"
    ):
        flag_gems.quantize_per_channel(
            inp,
            scales,
            torch.zeros(2, dtype=torch.int64, device=flag_gems.device),
            0,
            torch.quint8,
        )

    # Equal-length vectors that are both too short trip the per-channel length
    # check, and are the case that used to read past the end of the qparams.
    with pytest.raises(RuntimeError, match="length of scales must equal to channel"):
        flag_gems.quantize_per_channel(
            inp, scales[:3], zero_points[:3], 0, torch.quint8
        )


@pytest.mark.quantize_per_channel
def test_quantize_per_channel_rejects_integer_scales():
    """Scales must be floating point."""
    inp = torch.randn(4, 8, device=flag_gems.device)
    scales = torch.full((4,), 1, device=flag_gems.device, dtype=torch.int64)
    zero_points = torch.zeros(4, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="scale tensor must be floating point"):
        flag_gems.quantize_per_channel(inp, scales, zero_points, 0, torch.quint8)


@pytest.mark.quantize_per_channel
def test_quantize_per_channel_rejects_unsupported_dtype():
    """Only quint8/qint8/qint32 have a per-channel implementation."""
    inp = torch.randn(4, 8, device=flag_gems.device)
    scales = torch.rand(4, dtype=torch.float64, device=flag_gems.device) + 0.1
    zero_points = torch.zeros(4, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(NotImplementedError, match="not implemented for"):
        flag_gems.quantize_per_channel(inp, scales, zero_points, 0, torch.quint4x2)


_DEVICE_TRAP_CASE = textwrap.dedent("""
    import torch
    import flag_gems

    input = torch.randn(4, 8, device=flag_gems.device)
    scales = torch.full((4,), 0.1, dtype=torch.float64, device=flag_gems.device)
    zero_points = torch.full((4,), {zp}, device=flag_gems.device, dtype={dtype})
    flag_gems.quantize_per_channel(input, scales, zero_points, 0, torch.{qtype})
    torch.cuda.synchronize()
    print("no error")
    """)


@pytest.mark.quantize_per_channel
@pytest.mark.parametrize(
    "qtype,zp,dtype",
    [
        ("quint8", "300", "torch.int64"),
        ("qint8", "-129", "torch.int64"),
        ("quint8", "-0.5", "torch.float32"),
        ("qint8", "300.0", "torch.float32"),
    ],
)
def test_quantize_per_channel_rejects_out_of_range_zero_point(qtype, zp, dtype):
    """Zero-points outside the target dtype's range fail on device.

    The check is a device-side trap because deciding it on the host would
    require a synchronising read of the qparam tensor. A trap faults the CUDA
    context, so the call runs in a subprocess and only its failure is asserted.
    """
    code = _DEVICE_TRAP_CASE.format(zp=zp, dtype=dtype, qtype=qtype)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert (
        proc.returncode != 0
    ), f"out-of-range zero_point was accepted: {proc.stdout!r}"
    assert "no error" not in proc.stdout


@pytest.mark.quantize_per_channel
def test_quantize_per_channel_accepts_valid_zero_points_in_subprocess():
    """The in-range zero-points used elsewhere do not trip the device trap."""
    code = textwrap.dedent("""
        import torch
        import flag_gems

        input = torch.randn(4, 8, device=flag_gems.device)
        scales = torch.full((4,), 0.1, dtype=torch.float64, device=flag_gems.device)
        for zp, dtype, qtype in [
            (0, torch.int64, torch.quint8),
            (255, torch.int64, torch.quint8),
            (-128, torch.int64, torch.qint8),
            (127, torch.int64, torch.qint8),
            (0.0, torch.float32, torch.quint8),
            (254.5, torch.float32, torch.quint8),
            (-127.5, torch.float32, torch.qint8),
        ]:
            zero_points = torch.full((4,), zp, device=flag_gems.device, dtype=dtype)
            flag_gems.quantize_per_channel(input, scales, zero_points, 0, qtype)
        torch.cuda.synchronize()
        print("no error")
        """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "no error" in proc.stdout
