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

from . import conftest as cfg
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
for _name in (
    "_make_per_channel_quantized_tensor",
    "_make_per_channel_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# Copy integer storage and attach per-channel scale/zero-point metadata.
# Integer zero points use affine qparams; floating zero points use float_qparams.
_STORAGE_DTYPES = [torch.uint8, torch.int8, torch.int32]

_QUANT_DTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}

_WRONG_QUANT_DTYPE = {
    torch.uint8: torch.qint8,
    torch.int8: torch.quint8,
    torch.int32: torch.quint8,
}

_SCALE_DTYPES = [torch.float32, torch.float64]

_REJECTED_INPUT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int16,
    torch.int64,
    torch.bool,
]

_NON_FINITE = [float("nan"), float("inf"), float("-inf")]

_AXIS_BY_RANK = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3}

_SHAPE_AXIS = [(shape, _AXIS_BY_RANK[len(shape)]) for shape in tu.selected_shapes()]

if tu.QUICK_MODE:
    _SHAPE_AXIS += [((2, 3, 4), 1), ((4, 5), 0)]

# Exercise positive and negative axis encodings.
_AXIS_SHAPES = tu.selected_cases(
    [
        ((7,), 0),
        ((7,), -1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), -1),
        ((2, 3, 4), 0),
        ((2, 3, 4), 1),
        ((2, 3, 4), 2),
        ((2, 3, 4), -2),
        ((2, 3, 4), -3),
        ((7, 13, 29), 2),
        ((7, 13, 29), -1),
        ((2, 3, 4, 5), 2),
        ((2, 3, 4, 5), -1),
        ((2, 3, 4, 5, 6), 3),
        ((2, 3, 4, 5, 6), -1),
    ],
    quick=[((2, 3, 4), 1), ((2, 3, 4), -2), ((2, 3, 4), 0), ((7,), -1)],
)


def _ref_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _num_channels(shape, axis):
    # A scalar storage tensor carries one channel.
    return 1 if len(shape) == 0 else shape[axis]


def _zero_point_bounds(dtype):
    if dtype == torch.uint8:
        return 0, 256
    if dtype == torch.int8:
        return -128, 128
    info = torch.iinfo(torch.int32)
    return info.min, info.max


def _make_metadata(shape, axis, storage_dtype, scale_dtype):
    # Use positive scales and zero points inside the storage dtype range.
    num_channels = _num_channels(shape, axis)
    scales = torch.rand(num_channels, dtype=scale_dtype, device=flag_gems.device) + 0.1
    low, high = _zero_point_bounds(storage_dtype)
    zero_points = torch.randint(
        low, high, (num_channels,), dtype=storage_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _make_float_metadata(shape, axis, zero_point_dtype):
    # Floating zero points select per_channel_affine_float_qparams.
    num_channels = _num_channels(shape, axis)
    scales = (
        torch.rand(num_channels, dtype=torch.float32, device=flag_gems.device) + 0.1
    )
    zero_points = torch.rand(
        num_channels, dtype=zero_point_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _make_out_buffer(shape, axis, storage_dtype, device, reference=False):
    # Prefill different qparams so out must overwrite existing metadata.
    num_channels = _num_channels(shape, axis)
    scales = torch.full((num_channels,), 9.0, dtype=torch.float64, device=device)
    zero_points = torch.full((num_channels,), 9, dtype=torch.int64, device=device)
    if reference:
        scales = tu.to_reference(scales)
        zero_points = tu.to_reference(zero_points)
    return torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=_QUANT_DTYPE[storage_dtype],
        device=device,
    )


def _assert_per_channel_metadata(res_out, ref_out):
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
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
    tu.assert_result_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor_value_ranges(
    shape, axis, value_range, storage_dtype
):
    num_channels = _num_channels(shape, axis)
    inp = tu.make_input(storage_dtype, shape, value_range)
    scales = tu.make_input(torch.float64, (num_channels,), value_range)
    zero_points = tu.make_input(storage_dtype, (num_channels,), value_range)
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor(shape, axis, storage_dtype, scale_dtype):
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, scale_dtype)
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize("shape,axis", _AXIS_SHAPES)
def test__make_per_channel_quantized_tensor_axis(
    shape, axis, storage_dtype, scale_dtype
):
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, scale_dtype)
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_boundary_values(storage_dtype):
    info = torch.iinfo(storage_dtype)
    values = [info.min, info.max, 0, 1]
    if storage_dtype != torch.uint8:
        values.append(-1)
    num_channels = len(values)
    inp = torch.tensor(
        values * 4, dtype=storage_dtype, device=flag_gems.device
    ).reshape(4, num_channels)
    axis = 1
    scales = torch.linspace(
        0.5, 1.5, num_channels, dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.tensor(values, dtype=torch.int64, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test__make_per_channel_quantized_tensor_non_contiguous(storage_dtype, scale_dtype):
    base = tu.make_input(storage_dtype, (4, 3, 8), ["0", "max"])
    inp = base.transpose(0, 1)
    assert not inp.is_contiguous()  # shape (3, 4, 8)
    axis = 1
    scales, zero_points = _make_metadata(inp.shape, axis, storage_dtype, scale_dtype)
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("zero_point_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor_float_zero_points(
    shape, axis, storage_dtype, zero_point_dtype
):
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_float_metadata(shape, axis, zero_point_dtype)
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("bad", _NON_FINITE)
def test__make_per_channel_quantized_tensor_non_finite_scales(storage_dtype, bad):
    shape, axis = (2, 3, 4), 1
    num_channels = _num_channels(shape, axis)
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales = torch.full(
        (num_channels,), bad, dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.zeros(
        num_channels, dtype=storage_dtype, device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("bad", _NON_FINITE)
def test__make_per_channel_quantized_tensor_non_finite_float_zero_points(
    storage_dtype, bad
):
    shape, axis = (2, 3, 4), 1
    num_channels = _num_channels(shape, axis)
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales = torch.full(
        (num_channels,), 0.5, dtype=torch.float32, device=flag_gems.device
    )
    zero_points = torch.full(
        (num_channels,), bad, dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor_out(shape, axis, storage_dtype):
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = tu.to_reference(inp)
    ref_scales = tu.to_reference(scales)
    ref_zero_points = tu.to_reference(zero_points)

    ref_out_buf = _make_out_buffer(
        shape, axis, storage_dtype, _ref_device(), reference=True
    )
    torch.ops.aten._make_per_channel_quantized_tensor.out(
        ref_inp, ref_scales, ref_zero_points, axis, out=ref_out_buf
    )

    act_out_buf = _make_out_buffer(shape, axis, storage_dtype, flag_gems.device)
    res_ret = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis, out=act_out_buf
    )
    assert res_ret is act_out_buf

    _assert_per_channel_metadata(act_out_buf, ref_out_buf)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(scales, ref_scales)
    tu.assert_result_equal(zero_points, ref_zero_points)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("invalid_dtype", _REJECTED_INPUT_DTYPES)
def test__make_per_channel_quantized_tensor_rejects_non_storage_dtype(invalid_dtype):
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=invalid_dtype, device=flag_gems.device)
    scales = torch.full((3,), 0.5, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.zeros(3, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            tu.to_reference(inp),
            tu.to_reference(scales),
            tu.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_channel_quantized_tensor(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("scale_dtype", [torch.int32, torch.int64])
def test__make_per_channel_quantized_tensor_rejects_non_float_scales(scale_dtype):
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.tensor([1, 2, 3], dtype=scale_dtype, device=flag_gems.device)
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            tu.to_reference(inp),
            tu.to_reference(scales),
            tu.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_channel_quantized_tensor(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("bad_metadata", ["scale", "zero_point"])
def test__make_per_channel_quantized_tensor_rejects_non_1d_metadata(bad_metadata):
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.rand(3, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.zeros(3, dtype=torch.int64, device=flag_gems.device)
    if bad_metadata == "scale":
        scales = torch.rand(2, 3, dtype=torch.float32, device=flag_gems.device)
    else:
        zero_points = torch.zeros(2, 3, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            tu.to_reference(inp),
            tu.to_reference(scales),
            tu.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_channel_quantized_tensor(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("scale_len,zero_point_len", [(2, 3), (3, 2), (0, 3), (3, 0)])
def test__make_per_channel_quantized_tensor_rejects_metadata_length_mismatch(
    scale_len, zero_point_len
):
    # Metadata lengths must match each other; the factory does not check size[axis].
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.rand(scale_len, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.zeros(
        zero_point_len, dtype=torch.int64, device=flag_gems.device
    )

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            tu.to_reference(inp),
            tu.to_reference(scales),
            tu.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_channel_quantized_tensor(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_out_rejects_non_quantized_buffer(
    storage_dtype,
):
    shape, axis = (2, 3), 1
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = tu.to_reference(inp)

    ref_buf = torch.empty(shape, dtype=torch.float32, device=_ref_device())
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor.out(
            ref_inp,
            tu.to_reference(scales),
            tu.to_reference(zero_points),
            axis,
            out=ref_buf,
        )

    act_buf = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_channel_quantized_tensor(
            inp, scales, zero_points, axis, out=act_buf
        )


@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_out_rejects_wrong_quantized_dtype(
    storage_dtype,
):
    shape, axis = (2, 3), 1
    num_channels = _num_channels(shape, axis)
    inp = tu.make_input(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = tu.to_reference(inp)

    ref_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=tu.to_reference(
            torch.full(
                (num_channels,), 9.0, dtype=torch.float64, device=flag_gems.device
            ),
        ),
        zero_points=tu.to_reference(
            torch.full((num_channels,), 9, dtype=torch.int64, device=flag_gems.device),
        ),
        axis=axis,
        dtype=_WRONG_QUANT_DTYPE[storage_dtype],
        device=_ref_device(),
    )
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor.out(
            ref_inp,
            tu.to_reference(scales),
            tu.to_reference(zero_points),
            axis,
            out=ref_buf,
        )

    act_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.full(
            (num_channels,), 9.0, dtype=torch.float64, device=flag_gems.device
        ),
        zero_points=torch.full(
            (num_channels,), 9, dtype=torch.int64, device=flag_gems.device
        ),
        axis=axis,
        dtype=_WRONG_QUANT_DTYPE[storage_dtype],
        device=flag_gems.device,
    )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_channel_quantized_tensor(
            inp, scales, zero_points, axis, out=act_buf
        )
