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

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
for _name in (
    "_empty_per_channel_affine_quantized",
    "_empty_per_channel_affine_quantized_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# Allocate uninitialized quantized storage; compare layout and per-channel qparams.
QINT_DTYPES = [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]

# CUDA out metadata is unreliable for sub-byte dtypes; default still covers them.
QINT_DTYPES_OUT = [torch.qint8, torch.quint8, torch.qint32]

# (shape, axis), including empty and negative-axis cases.
SHAPE_AXIS = tu.selected_cases(
    [
        ((4,), 0),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), -1),
        ((2, 3, 4), 1),
        ((2, 3, 4), -1),
        ((2, 3, 4, 5), 1),
        ((2, 3, 4, 5), -1),
        ((2, 3, 4, 5, 6), 2),
        ((2, 3, 4, 5, 6), -2),
        ((0, 2, 3), 1),
        ((2, 0), 1),
        ((256,), 0),
        ((1024, 1024), 0),
        ((1024, 1024), 1),
        ((20, 320, 15), 1),
        ((16, 128, 64, 60), 2),
        ((16, 7, 57, 32, 29), 3),
    ],
    quick=[((2, 19, 7), 1)],
)

SCALES_DTYPES = [torch.float32, torch.float64]

ZERO_POINT_DTYPES = [torch.int32, torch.int64]


def _make_metadata(shape, axis, scale_dtype, zero_point_dtype):
    # Use zero points accepted by the CUDA per-channel quantizer.
    num_channels = shape[axis]
    scales = torch.arange(
        1, num_channels + 1, dtype=scale_dtype, device=flag_gems.device
    )
    zero_points = torch.randint(
        0, 8, (num_channels,), dtype=zero_point_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _make_ranged_metadata(shape, axis, value_range):
    num_channels = shape[axis]
    scales = tu.make_input(torch.float64, (num_channels,), value_range).to(
        flag_gems.device
    )
    zero_points = tu.make_input(torch.int64, (num_channels,), value_range).to(
        flag_gems.device
    )
    return scales, zero_points


def _assert_per_channel_metadata(res_out, ref_out):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.stride() == ref_out.stride()
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.qscheme() == ref_out.qscheme()
    assert res_out.q_per_channel_axis() == ref_out.q_per_channel_axis()
    tu.assert_result_equal(
        res_out.q_per_channel_scales(), ref_out.q_per_channel_scales()
    )
    tu.assert_result_equal(
        res_out.q_per_channel_zero_points(), ref_out.q_per_channel_zero_points()
    )


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("quantized_dtype", QINT_DTYPES)
@pytest.mark.parametrize("scale_dtype", SCALES_DTYPES)
@pytest.mark.parametrize("zero_point_dtype", ZERO_POINT_DTYPES)
def test__empty_per_channel_affine_quantized(
    shape, axis, quantized_dtype, scale_dtype, zero_point_dtype
):
    scales, zero_points = _make_metadata(shape, axis, scale_dtype, zero_point_dtype)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=ref_scales,
        zero_points=ref_zero_points,
        axis=axis,
        dtype=quantized_dtype,
        device=ref_device,
    )

    res_out = flag_gems._empty_per_channel_affine_quantized(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=quantized_dtype,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("quantized_dtype", QINT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__empty_per_channel_affine_quantized_metadata_value_ranges(
    shape, axis, quantized_dtype, value_range
):
    scales, zero_points = _make_ranged_metadata(shape, axis, value_range)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=ref_scales,
        zero_points=ref_zero_points,
        axis=axis,
        dtype=quantized_dtype,
        device=ref_device,
    )

    res_out = flag_gems._empty_per_channel_affine_quantized(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=quantized_dtype,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._empty_per_channel_affine_quantized_out
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("quantized_dtype", QINT_DTYPES_OUT)
@pytest.mark.parametrize("scale_dtype", SCALES_DTYPES)
@pytest.mark.parametrize("zero_point_dtype", ZERO_POINT_DTYPES)
def test__empty_per_channel_affine_quantized_out(
    shape, axis, quantized_dtype, scale_dtype, zero_point_dtype
):
    scales, zero_points = _make_metadata(shape, axis, scale_dtype, zero_point_dtype)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    # The .out variant writes the quantizer metadata into the provided out
    # buffer and returns that same tensor (alias semantics). The buffer is
    # created with deliberately *different* metadata so the overwrite is
    # observable. Note: the .out variant does not change the out buffer dtype,
    # so the buffer is created with the tested quantized dtype.
    ref_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=tu.to_reference(
            torch.tensor([9.0], dtype=torch.float64, device=flag_gems.device),
        ),
        zero_points=tu.to_reference(
            torch.tensor([9], dtype=torch.int64, device=flag_gems.device),
        ),
        axis=0,
        dtype=quantized_dtype,
        device=ref_device,
    )
    torch.ops.aten._empty_per_channel_affine_quantized.out(
        shape,
        scales=ref_scales,
        zero_points=ref_zero_points,
        axis=axis,
        out=ref_out_buf,
    )

    act_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.tensor([9.0], dtype=torch.float64, device=flag_gems.device),
        zero_points=torch.tensor([9], dtype=torch.int64, device=flag_gems.device),
        axis=0,
        dtype=quantized_dtype,
        device=flag_gems.device,
    )
    res_ret = flag_gems._empty_per_channel_affine_quantized(
        shape, scales=scales, zero_points=zero_points, axis=axis, out=act_out_buf
    )
    assert res_ret is act_out_buf

    _assert_per_channel_metadata(act_out_buf, ref_out_buf)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_cases([(3, 4)]))
@pytest.mark.parametrize("scenario", tu.selected_cases(["nan", "inf", "mixed"]))
def test__empty_per_channel_affine_quantized_nan_inf_scales(shape, scenario):
    scales = tu.make_special_input(torch.float64, scenario)[: shape[0]]
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=ref_scales,
        zero_points=ref_zero_points,
        axis=0,
        dtype=torch.quint8,
        device=ref_device,
    )

    res_out = flag_gems._empty_per_channel_affine_quantized(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=0,
        dtype=torch.quint8,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._empty_per_channel_affine_quantized
def test__empty_per_channel_affine_quantized_fp64_scales_preserved():
    # These scale values cannot round-trip through float32.
    shape = (3, 4)
    scales = torch.tensor(
        [0.1 + 1e-17, 0.2, 1 / 3], dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.tensor([1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=ref_scales,
        zero_points=ref_zero_points,
        axis=0,
        dtype=torch.quint8,
        device=ref_device,
    )

    res_out = flag_gems._empty_per_channel_affine_quantized(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=0,
        dtype=torch.quint8,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("invalid_dtype", [torch.float32, torch.float64, torch.int32])
def test__empty_per_channel_affine_quantized_negative_invalid_dtype(invalid_dtype):
    shape = (2, 3)
    scales = torch.tensor([1.0, 2.0], dtype=torch.float64, device=flag_gems.device)
    zero_points = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._empty_per_channel_affine_quantized(
            shape,
            scales=tu.to_reference(scales),
            zero_points=tu.to_reference(zero_points),
            axis=1,
            dtype=invalid_dtype,
            device=ref_device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._empty_per_channel_affine_quantized(
            shape,
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=invalid_dtype,
            device=flag_gems.device,
        )


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("scale_dtype", [torch.int32, torch.int64, torch.uint8])
def test__empty_per_channel_affine_quantized_negative_non_float_scales(scale_dtype):
    shape = (2, 3)
    scales = torch.tensor([1, 2], dtype=scale_dtype, device=flag_gems.device)
    zero_points = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._empty_per_channel_affine_quantized(
            shape,
            scales=tu.to_reference(scales),
            zero_points=tu.to_reference(zero_points),
            axis=1,
            dtype=torch.quint8,
            device=ref_device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._empty_per_channel_affine_quantized(
            shape,
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
            device=flag_gems.device,
        )


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("scale_len,zero_point_len", [(1, 2), (3, 1), (0, 2), (2, 0)])
def test__empty_per_channel_affine_quantized_negative_metadata_length_mismatch(
    scale_len, zero_point_len
):
    # Metadata lengths must match each other; the factory does not check size[axis].
    shape = (2, 3)
    scales = torch.arange(
        1, scale_len + 1, dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.zeros(
        zero_point_len, dtype=torch.int64, device=flag_gems.device
    )
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._empty_per_channel_affine_quantized(
            shape,
            scales=tu.to_reference(scales),
            zero_points=tu.to_reference(zero_points),
            axis=1,
            dtype=torch.quint8,
            device=ref_device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._empty_per_channel_affine_quantized(
            shape,
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
            device=flag_gems.device,
        )
