# Copyright 2026, The FlagOS Contributors.
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
"""Accuracy tests for ``aten::_empty_per_channel_affine_quantized``.

This operator is a factory: it allocates a fresh, *uninitialized* buffer and
attaches a per-channel affine quantizer.  The values in the buffer are
unspecified by contract (the ATen reference returns whatever the allocator
hands back), so nothing here compares ``int_repr()`` between the two
implementations.  What is checked instead is the metadata that the op actually
promises:

* dtype, shape, device and layout of the result;
* ``qscheme()`` (``per_channel_affine`` for integral zero points,
  ``per_channel_affine_float_qparams`` for floating point ones);
* ``q_per_channel_axis()``, which is stored verbatim (negative axes included);
* the scales and zero points read back through the public
  ``torch.ops.aten.q_per_channel_scales`` / ``q_per_channel_zero_points`` ops.

Error behaviour is delegated to the native factory, so the rejection cases
(no quantized dtype, malformed scales, string ``memory_format``) are also
pinned against the reference.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

pytestmark = [pytest.mark.empty_per_channel_affine_quantized]

_QUANTIZED_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Shapes cover the ordinary 2-D/4-D cases plus a zero-extent output.
_SHAPES = [(2, 4), (3, 8), (2, 3, 4, 5), (0, 4)]


def _ref(size, **kwargs):
    return torch.ops.aten._empty_per_channel_affine_quantized.default(size, **kwargs)


def _gems(size, **kwargs):
    return flag_gems._empty_per_channel_affine_quantized(size, **kwargs)


def _scales(n, device, dtype=torch.float64):
    return (torch.arange(1, n + 1, dtype=dtype, device=device) * 0.25).contiguous()


def _zero_points(n, device, dtype=torch.int64):
    return torch.zeros(n, dtype=dtype, device=device)


def _assert_matches_reference(res, ref, *, size, dtype, axis, scales, zero_points):
    """Check the metadata contract shared by the result and the reference."""
    assert tuple(res.shape) == tuple(size)
    assert res.dtype == dtype
    assert res.device == ref.device
    assert res.layout == ref.layout
    assert res.is_quantized

    assert res.qscheme() == ref.qscheme()
    assert res.q_per_channel_axis() == ref.q_per_channel_axis() == axis

    # Read the qparams back through the aten accessors the op is expected to
    # feed, not only through the Tensor methods.
    res_scales = torch.ops.aten.q_per_channel_scales(res)
    res_zps = torch.ops.aten.q_per_channel_zero_points(res)
    ref_scales = torch.ops.aten.q_per_channel_scales(ref)
    ref_zps = torch.ops.aten.q_per_channel_zero_points(ref)

    utils.gems_assert_equal(res_scales.to(torch.float64), ref_scales.to(torch.float64))
    utils.gems_assert_equal(res_zps.to(torch.float64), ref_zps.to(torch.float64))
    utils.gems_assert_equal(res_scales.to(torch.float64), scales.to(torch.float64))
    utils.gems_assert_equal(res_zps.to(torch.float64), zero_points.to(torch.float64))

    # The int representation exists and has the requested shape; its contents
    # are uninitialized and intentionally not compared.
    int_repr = res.int_repr()
    assert tuple(int_repr.shape) == tuple(size)
    assert int_repr.device == res.device


@pytest.mark.empty_per_channel_affine_quantized
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _QUANTIZED_DTYPES)
def test_empty_per_channel_affine_quantized(shape, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    axis = 1
    scales = _scales(shape[axis], flag_gems.device)
    zero_points = _zero_points(shape[axis], flag_gems.device)

    ref_out = _ref(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=dtype,
        device=flag_gems.device,
    )
    res_out = _gems(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=dtype,
        device=flag_gems.device,
    )

    _assert_matches_reference(
        res_out,
        ref_out,
        size=shape,
        dtype=dtype,
        axis=axis,
        scales=scales,
        zero_points=zero_points,
    )
    assert res_out.qscheme() == torch.per_channel_affine


@pytest.mark.empty_per_channel_affine_quantized
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_empty_per_channel_affine_quantized_axis(axis):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape = (4, 6)
    channel = shape[axis]
    scales = _scales(channel, flag_gems.device)
    zero_points = _zero_points(channel, flag_gems.device)

    res_out = _gems(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=torch.quint8,
        device=flag_gems.device,
    )

    assert res_out.q_per_channel_axis() == axis
    assert res_out.qscheme() == torch.per_channel_affine
    utils.gems_assert_equal(
        torch.ops.aten.q_per_channel_scales(res_out).to(torch.float64),
        scales.to(torch.float64),
    )
    utils.gems_assert_equal(
        torch.ops.aten.q_per_channel_zero_points(res_out).to(torch.float64),
        zero_points.to(torch.float64),
    )


@pytest.mark.empty_per_channel_affine_quantized
def test_empty_per_channel_affine_quantized_float_zero_points():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape = (2, 4)
    scales = _scales(shape[1], flag_gems.device)
    zero_points = torch.full(
        (shape[1],), 0.5, dtype=torch.float32, device=flag_gems.device
    )

    ref_out = _ref(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=1,
        dtype=torch.qint8,
        device=flag_gems.device,
    )
    res_out = _gems(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=1,
        dtype=torch.qint8,
        device=flag_gems.device,
    )

    assert res_out.qscheme() == torch.per_channel_affine_float_qparams
    assert res_out.qscheme() == ref_out.qscheme()
    utils.gems_assert_equal(
        torch.ops.aten.q_per_channel_zero_points(res_out).to(torch.float32),
        zero_points,
    )


@pytest.mark.empty_per_channel_affine_quantized
def test_empty_per_channel_affine_quantized_device_default_cpu():
    """``device=None`` inherits the native factory's CPU default, not scales'."""
    shape = (2, 4)
    scales = torch.ones(shape[1], dtype=torch.float64)
    zero_points = torch.zeros(shape[1], dtype=torch.int64)

    res_out = _gems(
        list(shape), scales=scales, zero_points=zero_points, axis=1, dtype=torch.quint8
    )

    assert res_out.device.type == "cpu"
    assert res_out.qscheme() == torch.per_channel_affine
    assert tuple(res_out.shape) == shape


@pytest.mark.empty_per_channel_affine_quantized
def test_empty_per_channel_affine_quantized_memory_format():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape = (2, 3, 4, 5)
    scales = _scales(shape[1], flag_gems.device)
    zero_points = _zero_points(shape[1], flag_gems.device)

    ref_out = _ref(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=1,
        dtype=torch.quint8,
        device=flag_gems.device,
        memory_format=torch.channels_last,
    )
    res_out = _gems(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=1,
        dtype=torch.quint8,
        device=flag_gems.device,
        memory_format=torch.channels_last,
    )

    assert res_out.stride() == ref_out.stride()
    assert res_out.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.empty_per_channel_affine_quantized
def test_empty_per_channel_affine_quantized_requires_quantized_dtype():
    shape = (2, 4)
    scales = torch.ones(shape[1], dtype=torch.float64)
    zero_points = torch.zeros(shape[1], dtype=torch.int64)

    # The exact message depends on which dispatch key ATen resolves (the CPU
    # stub says "Creation of quantized tensor requires quantized dtype ...",
    # the real factory says "Must provide data type ..."), so only the
    # rejection itself is pinned.
    with pytest.raises(RuntimeError):
        _ref(list(shape), scales=scales, zero_points=zero_points, axis=1)
    with pytest.raises(RuntimeError):
        _gems(list(shape), scales=scales, zero_points=zero_points, axis=1)


@pytest.mark.empty_per_channel_affine_quantized
def test_empty_per_channel_affine_quantized_rejects_bad_qparams():
    shape = (2, 4)
    zero_points = torch.zeros(shape[1], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="scales and zero_points must match"):
        _gems(
            list(shape),
            scales=torch.ones(shape[1] + 1, dtype=torch.float64),
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
        )
    with pytest.raises(RuntimeError, match="dimension 1"):
        _gems(
            list(shape),
            scales=torch.ones(shape[1], 1, dtype=torch.float64),
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
        )
    with pytest.raises(RuntimeError, match="must be floating point"):
        _gems(
            list(shape),
            scales=torch.ones(shape[1], dtype=torch.int64),
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
        )


@pytest.mark.empty_per_channel_affine_quantized
def test_empty_per_channel_affine_quantized_rejects_string_memory_format():
    shape = (2, 4)
    scales = torch.ones(shape[1], dtype=torch.float64)
    zero_points = torch.zeros(shape[1], dtype=torch.int64)

    # ``memory_format`` must be a ``torch.memory_format``; the ATen packet
    # rejects a string during argument casting, and FlagGems reports the same
    # rejection through redispatch.
    with pytest.raises(RuntimeError):
        _ref(
            list(shape),
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
            memory_format="contiguous_format",
        )
    with pytest.raises(RuntimeError):
        _gems(
            list(shape),
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
            memory_format="contiguous_format",
        )
