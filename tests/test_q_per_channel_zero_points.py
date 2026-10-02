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
from . import test_utils as tu

# aten::q_per_channel_zero_points returns the zero_points tensor a per-channel
# quantizer already holds: per_channel_affine stores int64 metadata and
# per_channel_affine_float_qparams stores float32 metadata. There is no dense
# result of its own, so the shared value-range grid drives the stored metadata
# and the operator-specific coverage is the metadata dtype matrix, both qschemes,
# the scalar/empty-channel operands, identity/alias semantics and the .out
# buffer contract.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Sub-byte element dtypes share the same metadata handling; kept apart so they
# add coverage without multiplying the value-range grid.
SUB_BYTE_QUANT_DTYPES = [torch.quint4x2, torch.quint2x4]

# The two result families of the operator, i.e. the qscheme axis of the grid.
METADATA_FAMILY = [
    pytest.param(torch.int64, torch.int64, id="affine"),
    pytest.param(torch.float32, torch.float32, id="float_qparams"),
]

# (metadata dtype, result dtype). Integer parameters are held as int64 and
# floating parameters (the float-qparams scheme) as float32.
METADATA_DTYPES = [
    (torch.int8, torch.int64),
    (torch.uint8, torch.int64),
    (torch.int32, torch.int64),
    (torch.int64, torch.int64),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32),
    (torch.float64, torch.float32),
]

FLOAT_METADATA_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

# (shape, axis) rows for the value-range grid, covering the shared shape family.
# Rank 0 is a legal per-channel operand (one channel on axis 0); higher ranks use
# an interior, a trailing and a negative axis.
SHAPE_AXIS = tu.selected_cases(
    [
        ((), 0),
        ((1,), 0),
        ((256,), 0),
        ((1024, 1024), 1),
        ((20, 320, 15), 2),
        ((16, 128, 64, 60), -1),
        ((16, 7, 57, 32, 29), 3),
    ],
    quick=[((), 0), ((2, 19, 7), 1)],
)

PARAMETER_SHAPE_AXIS = tu.selected_cases(
    [((256,), 0), ((20, 320, 15), 2), ((16, 128, 64, 60), 1)],
    quick=[((2, 19, 7), 1)],
)

SUB_BYTE_AXIS = tu.selected_cases(
    [((1024, 1024), 0), ((20, 320, 15), 2)], quick=[((2, 19, 7), 1)]
)

# Both orientations are small boundary cases and stay in quick.
EMPTY_CHANNEL_AXIS = tu.selected_cases(
    [((0, 3), 0), ((3, 0), 1)], quick=[((0, 3), 0), ((3, 0), 1)]
)

OUT_SHAPE_AXIS = tu.selected_cases(
    [
        ((0, 3), 0),
        ((3, 0), 1),
        ((1,), 0),
        ((256,), 0),
        ((1024, 1024), 0),
        ((20, 320, 15), 2),
        ((16, 128, 64, 60), -1),
        ((16, 7, 57, 32, 29), 0),
    ],
    quick=[((1,), 0), ((0, 3), 0), ((3, 0), 1), ((2, 19, 7), 1)],
)

OUT_FLOAT_SHAPE_AXIS = tu.selected_cases(
    [((256,), 0), ((20, 320, 15), 2), ((3, 4), 0)], quick=[((2, 19, 7), 1)]
)

SCALAR_METADATA = [
    pytest.param(torch.int64, torch.int64, id="affine"),
    pytest.param(torch.float32, torch.float32, id="float_qparams"),
]

# Positive special-value scenarios are default-only.
SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(FLOAT_METADATA_DTYPES), quick=[]
)


def _reference_device():
    # The configured reference may run on CPU, so reference-side quantizers and
    # out buffers are built there.
    return "cpu" if utils.TO_CPU else flag_gems.device


def _num_channels(shape, axis):
    return shape[axis] if shape else 1


def _quantized(shape, axis, quantized_dtype, zero_points, device=None):
    # aten::_empty_per_channel_affine_quantized stores the supplied parameters
    # verbatim, so it can hold int64 extremes and float parameters that
    # torch.quantize_per_channel would range-check away.
    device = flag_gems.device if device is None else device
    scales = torch.ones(_num_channels(shape, axis), dtype=torch.float64, device=device)
    return torch.ops.aten._empty_per_channel_affine_quantized(
        tuple(shape),
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=quantized_dtype,
        device=device,
    )


def _quantized_pair(shape, axis, quantized_dtype, zero_points):
    """Candidate quantizer plus an independent reference-side one."""
    ref_zero_points = tu.to_reference(zero_points)
    ref_quantized = _quantized(
        shape, axis, quantized_dtype, ref_zero_points, device=_reference_device()
    )
    return _quantized(shape, axis, quantized_dtype, zero_points), ref_quantized


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("quantized_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("metadata_dtype,out_dtype", METADATA_FAMILY)
def test_q_per_channel_zero_points_value_range(
    shape, axis, quantized_dtype, metadata_dtype, out_dtype, value_range
):
    zero_points = tu.make_input(
        metadata_dtype, (_num_channels(shape, axis),), value_range
    )
    quantized, ref_quantized = _quantized_pair(
        shape, axis, quantized_dtype, zero_points
    )

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)

    assert res_out.dtype == out_dtype
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("shape,axis", PARAMETER_SHAPE_AXIS)
@pytest.mark.parametrize("metadata_dtype,out_dtype", METADATA_DTYPES)
def test_q_per_channel_zero_points_metadata_dtype(
    shape, axis, metadata_dtype, out_dtype
):
    zero_points = tu.make_input(
        metadata_dtype, (_num_channels(shape, axis),), ["-1", "1"]
    )
    quantized, ref_quantized = _quantized_pair(shape, axis, torch.quint8, zero_points)

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)

    assert res_out.dtype == out_dtype
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("shape,axis", SUB_BYTE_AXIS)
@pytest.mark.parametrize("quantized_dtype", SUB_BYTE_QUANT_DTYPES)
def test_q_per_channel_zero_points_sub_byte_dtype(shape, axis, quantized_dtype):
    zero_points = tu.make_input(torch.int64, (_num_channels(shape, axis),), ["-1", "1"])
    quantized, ref_quantized = _quantized_pair(
        shape, axis, quantized_dtype, zero_points
    )

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("metadata_dtype,scenario", SPECIAL_CASES)
def test_q_per_channel_zero_points_float_qparams_special_values(
    metadata_dtype, scenario
):
    # make_special_input yields five values, so the (5, 4) operand carries five
    # channels on axis 0.
    zero_points = tu.make_special_input(metadata_dtype, scenario)
    quantized, ref_quantized = _quantized_pair((5, 4), 0, torch.quint8, zero_points)

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("shape,axis", EMPTY_CHANNEL_AXIS)
def test_q_per_channel_zero_points_empty_channel(shape, axis):
    # A zero-length channel axis still carries an (empty) parameter tensor.
    zero_points = torch.empty(
        _num_channels(shape, axis), dtype=torch.int64, device=flag_gems.device
    )
    quantized, ref_quantized = _quantized_pair(shape, axis, torch.quint8, zero_points)

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)

    assert res_out.shape == ref_out.shape == (0,)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("metadata_dtype,out_dtype", SCALAR_METADATA)
def test_q_per_channel_zero_points_scalar_operand(metadata_dtype, out_dtype):
    # A rank-0 per-channel quantizer holds one channel on axis 0, so the getter
    # reports a length-1 tensor rather than a scalar.
    value = 0.25 if metadata_dtype.is_floating_point else 7
    zero_points = torch.tensor([value], dtype=metadata_dtype, device=flag_gems.device)
    quantized, ref_quantized = _quantized_pair((), 0, torch.quint8, zero_points)

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)

    assert res_out.shape == (1,)
    assert res_out.dtype == out_dtype
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("metadata_dtype,out_dtype", SCALAR_METADATA)
def test_q_per_channel_zero_points_out_scalar_operand(metadata_dtype, out_dtype):
    value = 0.25 if metadata_dtype.is_floating_point else 7
    zero_points = torch.tensor([value], dtype=metadata_dtype, device=flag_gems.device)
    quantized, ref_quantized = _quantized_pair((), 0, torch.quint8, zero_points)

    ref_buf = torch.full((1,), 0, dtype=out_dtype, device=_reference_device())
    torch.ops.aten.q_per_channel_zero_points.out(ref_quantized, out=ref_buf)
    buf = torch.full((1,), 0, dtype=out_dtype, device=flag_gems.device)
    res_ret = flag_gems.q_per_channel_zero_points(quantized, out=buf)

    assert res_ret is buf
    tu.assert_result_equal(buf, ref_buf)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("shape,axis", OUT_SHAPE_AXIS)
@pytest.mark.parametrize("quantized_dtype", QUANT_DTYPES)
def test_q_per_channel_zero_points_out(shape, axis, quantized_dtype):
    channels = _num_channels(shape, axis)
    zero_points = tu.make_input(torch.int64, (channels,), ["-1", "1"])
    quantized, ref_quantized = _quantized_pair(
        shape, axis, quantized_dtype, zero_points
    )

    ref_buf = torch.full((channels,), -1, dtype=torch.int64, device=_reference_device())
    torch.ops.aten.q_per_channel_zero_points.out(ref_quantized, out=ref_buf)
    buf = torch.full((channels,), -1, dtype=torch.int64, device=flag_gems.device)
    res_ret = flag_gems.q_per_channel_zero_points(quantized, out=buf)

    assert res_ret is buf
    tu.assert_result_equal(buf, ref_buf)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("shape,axis", OUT_FLOAT_SHAPE_AXIS)
def test_q_per_channel_zero_points_out_float_qparams(shape, axis):
    channels = _num_channels(shape, axis)
    zero_points = torch.full(
        (channels,), 0.5, dtype=torch.float32, device=flag_gems.device
    )
    quantized, ref_quantized = _quantized_pair(shape, axis, torch.quint8, zero_points)

    ref_buf = torch.full(
        (channels,), -2.0, dtype=torch.float32, device=_reference_device()
    )
    torch.ops.aten.q_per_channel_zero_points.out(ref_quantized, out=ref_buf)
    buf = torch.full((channels,), -2.0, dtype=torch.float32, device=flag_gems.device)
    res_ret = flag_gems.q_per_channel_zero_points(quantized, out=buf)

    assert res_ret is buf
    tu.assert_result_equal(buf, ref_buf)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("buffer_len", [0, 1, 5])
def test_q_per_channel_zero_points_out_resizes_buffer(buffer_len):
    # Native resizes a mismatched out buffer in place and returns that same
    # object; 3 channels with 0/1/5 elements covers both resize directions.
    channels = 3
    zero_points = torch.tensor([0, 1, 7], dtype=torch.int64, device=flag_gems.device)
    quantized, ref_quantized = _quantized_pair((3, 4), 0, torch.quint8, zero_points)

    ref_buf = torch.full(
        (buffer_len,), -1, dtype=torch.int64, device=_reference_device()
    )
    torch.ops.aten.q_per_channel_zero_points.out(ref_quantized, out=ref_buf)
    buf = torch.full((buffer_len,), -1, dtype=torch.int64, device=flag_gems.device)
    res_ret = flag_gems.q_per_channel_zero_points(quantized, out=buf)

    assert res_ret is buf
    assert buf.shape == ref_buf.shape == (channels,)
    tu.assert_result_equal(buf, ref_buf)


@pytest.mark.q_per_channel_zero_points
def test_q_per_channel_zero_points_out_strided_buffer():
    channels = 256
    zero_points = tu.make_input(torch.int64, (channels,), ["-1", "1"])
    quantized, ref_quantized = _quantized_pair(
        (channels,), 0, torch.quint8, zero_points
    )

    # The whole backing storage is compared, so the untouched sentinel holes
    # prove that only the view elements were written.
    ref_base = torch.full(
        (2 * channels,), -1, dtype=torch.int64, device=_reference_device()
    )
    torch.ops.aten.q_per_channel_zero_points.out(ref_quantized, out=ref_base[::2])
    buf_base = torch.full(
        (2 * channels,), -1, dtype=torch.int64, device=flag_gems.device
    )
    buf = buf_base[::2]
    res_ret = flag_gems.q_per_channel_zero_points(quantized, out=buf)

    assert res_ret is buf
    assert buf.stride() == (2,)
    tu.assert_result_equal(buf_base, ref_base)


@pytest.mark.q_per_channel_zero_points
def test_q_per_channel_zero_points_returns_stored_parameter_object():
    zero_points = torch.tensor(
        [3, 1, 4, 1, 5], dtype=torch.int64, device=flag_gems.device
    )
    quantized, ref_quantized = _quantized_pair((5, 4), 0, torch.quint8, zero_points)

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)

    # Identity, not merely equal pointers: the result is the object handed to the
    # quantizer, and every call returns that same object.
    assert res_out is zero_points
    assert flag_gems.q_per_channel_zero_points(quantized) is res_out
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.q_per_channel_zero_points
def test_q_per_channel_zero_points_result_aliases_metadata_storage():
    # torch.quantize_per_channel initializes the quantized storage, so the
    # snapshot below reads defined values; the raw factory leaves it undefined.
    data = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    scales = torch.tensor(
        [0.5, 0.25, 0.125], dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.tensor([1, 2, 3], dtype=torch.int64, device=flag_gems.device)

    quantized = torch.quantize_per_channel(data, scales, zero_points, 0, torch.quint8)
    ref_quantized = torch.quantize_per_channel(
        tu.to_reference(data),
        tu.to_reference(scales),
        tu.to_reference(zero_points),
        0,
        torch.quint8,
    )
    data_before = quantized.int_repr().clone()

    ref_out = torch.ops.aten.q_per_channel_zero_points(ref_quantized)
    res_out = flag_gems.q_per_channel_zero_points(quantized)
    tu.assert_result_equal(res_out, ref_out)

    res_out.fill_(9)

    # The write reaches the quantizer's own parameter (a returned copy would not
    # survive the second read) and leaves the data storage untouched.
    assert flag_gems.q_per_channel_zero_points(quantized).tolist() == [9, 9, 9]
    tu.assert_result_equal(quantized.int_repr(), data_before)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.float32, torch.float64])
def test_q_per_channel_zero_points_negative_out_dtype_affine(out_dtype):
    # The integer scheme stores long parameters and accepts only a long buffer.
    zero_points = torch.tensor([1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    quantized = _quantized((3, 4), 0, torch.quint8, zero_points)
    buf = torch.zeros(3, dtype=out_dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.q_per_channel_zero_points(quantized, out=buf)


@pytest.mark.q_per_channel_zero_points
@pytest.mark.parametrize("out_dtype", [torch.int64, torch.float64])
def test_q_per_channel_zero_points_negative_out_dtype_float_qparams(out_dtype):
    zero_points = torch.tensor(
        [0.25, 0.5, 0.75], dtype=torch.float32, device=flag_gems.device
    )
    quantized = _quantized((3, 4), 0, torch.quint8, zero_points)
    buf = torch.zeros(3, dtype=out_dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.q_per_channel_zero_points(quantized, out=buf)


@pytest.mark.q_per_channel_zero_points
def test_q_per_channel_zero_points_negative_per_tensor_qscheme():
    quantized = torch.quantize_per_tensor(
        tu.make_input(torch.float32, (4,), ["-1", "1"]), 0.1, 3, torch.quint8
    )

    with pytest.raises(RuntimeError):
        flag_gems.q_per_channel_zero_points(quantized)


@pytest.mark.q_per_channel_zero_points
def test_q_per_channel_zero_points_negative_non_quantized_input():
    # Native raises NotImplementedError for a dense tensor.
    dense = tu.make_input(torch.float32, (4,), ["-1", "1"])

    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems.q_per_channel_zero_points(dense)
