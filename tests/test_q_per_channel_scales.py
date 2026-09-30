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

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# q_per_channel_scales returns the scale tensor a per-channel quantizer already holds
# (float64 for per_channel_affine, float32 for per_channel_affine_float_qparams); the
# out overload copies it into a caller buffer. Coverage is the per-channel shape/axis
# grid with its rank-0, zero-channel and negative-axis boundaries, the quantized
# storage dtype, both qparam families, both call forms and the invalid inputs.
#
# A quantized operand cannot be built by tu.make_input and cannot be cloned for the
# float_qparams scheme, so each case builds two independent operands from the same
# parameters. Value ranges and nan/inf are applied to the stored parameter tensors,
# the only floating values this accessor returns.

STORAGE_BYTES = {
    torch.qint8: torch.int8,
    torch.quint8: torch.uint8,
    torch.qint32: torch.int32,
}

# per_channel_affine keeps float64 scales with integer zero points (int32 for qint32
# storage); per_channel_affine_float_qparams keeps float32 for both and exists for
# 8-bit storage only.
QPARAM_FAMILIES = [
    ("affine", torch.float64, torch.int64, [torch.qint8, torch.quint8, torch.qint32]),
    ("float_qparams", torch.float32, torch.float32, [torch.qint8, torch.quint8]),
]

RESULT_DTYPE = {"affine": torch.float64, "float_qparams": torch.float32}


def _channels(shape, axis):
    # A rank-0 quantized tensor reports a single scale on axis 0.
    return 1 if len(shape) == 0 else shape[axis]


def _axes(shape):
    if len(shape) == 0:
        return (0,)
    # First, middle and last dimension, so the per-channel axis is not always the last.
    return tuple(sorted({0, len(shape) // 2, len(shape) - 1}))


# Rank-0, single-element, empty-tensor and negative-axis boundaries. (4, 0, 3) axis 1 is
# the zero-channel case (empty tensor and metadata); (0, 5) axis -1 is its
# converse (empty tensor, five channels of metadata).
_BOUNDARY_ROWS = [
    ((), 0),
    ((1,), 0),
    ((0,), 0),
    ((4, 0, 3), 1),
    ((0, 5), -1),
    ((7, 5), -1),
    ((3, 4, 5), -3),
]


def _unique_rows(rows):
    seen = set()
    unique = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            unique.append(row)
    return unique


SHAPE_AXIS_CASES = _unique_rows(
    [(shape, axis) for shape in tu.selected_shapes() for axis in _axes(shape)]
    + _BOUNDARY_ROWS
)

CASE_ROWS = [
    (family, scales_dtype, zero_points_dtype, storage_dtype, shape, axis)
    for family, scales_dtype, zero_points_dtype, storages in QPARAM_FAMILIES
    for storage_dtype in storages
    for shape, axis in SHAPE_AXIS_CASES
]

# The out= contract depends on the qparam family rather than on the storage dtype, so it
# runs once per family. 'guard' uses a strided view inside a sentinel-filled backing
# tensor; 'resize' passes a buffer of the wrong length, which native resizes in place.
_OUT_PLAIN_ROWS = [
    (family, scales_dtype, zero_points_dtype, shape, axis, "plain")
    for family, scales_dtype, zero_points_dtype, _ in QPARAM_FAMILIES
    for shape, axis in SHAPE_AXIS_CASES
]

_OUT_EDGE_SHAPES = {
    "guard": [((7, 5), -1), ((), 0), ((1024, 1024), 0)],
    "resize": [((7, 5), -1), ((4, 0, 3), 1), ((1024, 1024), 0)],
}


def _out_edge_rows(shapes_by_variant):
    return [
        (family, scales_dtype, zero_points_dtype, shape, axis, variant)
        for variant, shapes in shapes_by_variant.items()
        for family, scales_dtype, zero_points_dtype, _ in QPARAM_FAMILIES
        for shape, axis in shapes
    ]


OUT_CASES = _OUT_PLAIN_ROWS + tu.selected_cases(
    _out_edge_rows(_OUT_EDGE_SHAPES),
    quick=_out_edge_rows(
        {
            variant: [row for row in rows if row[0] != (1024, 1024)]
            for variant, rows in _OUT_EDGE_SHAPES.items()
        }
    ),
)

ALIAS_CASES = [
    (family, scales_dtype, zero_points_dtype, storage_dtype)
    for family, scales_dtype, zero_points_dtype, storages in QPARAM_FAMILIES
    for storage_dtype in storages
]

# qint32 storage is affine-only and pairs with int32 zero points.
INT32_ZERO_POINT_CASES = [
    ((256,), 0),
    ((7, 5), -1),
    ((1024, 1024), 1),
    ((1024, 1024), -1),
    ((20, 320, 15), -1),
    ((20, 320, 15), 1),
    ((16, 7, 57, 32, 29), 2),
]

SCALE_RANGE_CASES = [
    (value_range, family, scales_dtype, zero_points_dtype, storage_dtype)
    for value_range in tu.selected_ranges()
    for family, scales_dtype, zero_points_dtype, storages in QPARAM_FAMILIES
    for storage_dtype in storages
]

SPECIAL_SCALE_CASES = tu.selected_cases(
    [
        (scenario, family, scales_dtype, zero_points_dtype, storage_dtype)
        for scenario in ("nan", "inf", "mixed")
        for family, scales_dtype, zero_points_dtype, storages in QPARAM_FAMILIES
        for storage_dtype in storages
    ],
    quick=[],
)

NEGATIVE_CASES = [
    "non_tensor_input",
    "plain_float_tensor",
    "not_per_channel",
    "out_dtype_mismatch",
    "out_not_tensor",
]


def _storage_values(shape, storage_dtype):
    # Real initialized integer storage; the integer payload never reaches the result,
    # so it only has to be initialized and non-trivial.
    numel = math.prod(shape)
    return (
        (torch.arange(numel, dtype=torch.int32, device=flag_gems.device) % 100)
        .to(STORAGE_BYTES[storage_dtype])
        .reshape(shape)
    )


def _make_operand(
    shape, axis, storage_dtype, scales_dtype, zero_points_dtype, scale_values=None
):
    channels = _channels(shape, axis)
    if scale_values is None:
        scales = (
            torch.arange(1, channels + 1, dtype=scales_dtype, device=flag_gems.device)
            * 0.5
        )
    else:
        scales = torch.as_tensor(
            scale_values, dtype=scales_dtype, device=flag_gems.device
        ).reshape(-1)
    zero_points = (
        torch.arange(channels, dtype=torch.int32, device=flag_gems.device) % 7
    ).to(zero_points_dtype)
    return torch.ops.aten._make_per_channel_quantized_tensor(
        _storage_values(shape, storage_dtype), scales, zero_points, axis
    )


def _quantized_pair(
    shape, axis, storage_dtype, scales_dtype, zero_points_dtype, scale_values=None
):
    inp = _make_operand(
        shape, axis, storage_dtype, scales_dtype, zero_points_dtype, scale_values
    )
    ref_inp = _make_operand(
        shape, axis, storage_dtype, scales_dtype, zero_points_dtype, scale_values
    )
    return inp, ref_inp


@pytest.mark.q_per_channel_scales
@pytest.mark.parametrize(
    "family,scales_dtype,zero_points_dtype,storage_dtype,shape,axis", CASE_ROWS
)
def test_q_per_channel_scales(
    family, scales_dtype, zero_points_dtype, storage_dtype, shape, axis
):
    inp, ref_inp = _quantized_pair(
        shape, axis, storage_dtype, scales_dtype, zero_points_dtype
    )
    # The native result aliases the quantizer's parameter tensor, so it is snapshotted
    # before the candidate runs; ref_inp is an independent operand.
    ref_snapshot = torch.ops.aten.q_per_channel_scales(ref_inp).detach().clone()

    res_out = flag_gems.q_per_channel_scales(inp)

    # One scale per channel, in the dtype the stored qparams carry, on the input device.
    assert res_out.dtype == RESULT_DTYPE[family]
    assert res_out.device == inp.device
    assert res_out.shape == (_channels(shape, axis),)
    tu.assert_result_equal(res_out, ref_snapshot)


@pytest.mark.q_per_channel_scales
@pytest.mark.filterwarnings(
    "ignore:An output with one or more elements was resized:UserWarning",
)
@pytest.mark.parametrize(
    "family,scales_dtype,zero_points_dtype,shape,axis,variant", OUT_CASES
)
def test_q_per_channel_scales_out(
    family, scales_dtype, zero_points_dtype, shape, axis, variant
):
    inp, ref_inp = _quantized_pair(
        shape, axis, torch.qint8, scales_dtype, zero_points_dtype
    )
    channels = _channels(shape, axis)
    out_dtype = RESULT_DTYPE[family]
    guard = ref_guard = None

    if variant == "guard":
        # Non-contiguous view inside a sentinel-filled backing tensor; the whole backing
        # tensor is compared, so a write outside the view becomes visible.
        guard = torch.full(
            (2 * channels + 3,), -1234.5, dtype=out_dtype, device=inp.device
        )
        ref_guard = torch.full(
            (2 * channels + 3,), -1234.5, dtype=out_dtype, device=ref_inp.device
        )
        out_buf = guard[1 : 1 + 2 * channels : 2]
        ref_buf = ref_guard[1 : 1 + 2 * channels : 2]
    else:
        size = channels + 2 if variant == "resize" else channels
        out_buf = torch.empty(size, dtype=out_dtype, device=inp.device)
        ref_buf = torch.empty(size, dtype=out_dtype, device=ref_inp.device)

    ref_out = torch.ops.aten.q_per_channel_scales.out(ref_inp, out=ref_buf)
    res_out = flag_gems.q_per_channel_scales(inp, out=out_buf)

    # The caller's object is filled and returned, resizing a mismatched buffer.
    assert res_out is out_buf
    assert res_out.dtype == out_dtype
    assert res_out.shape == (channels,)
    tu.assert_result_equal(res_out, ref_out)

    if variant == "guard":
        # Same storage at the same offset with the same stride, and nothing written
        # outside the view.
        assert out_buf.data_ptr() == guard.data_ptr() + guard.element_size()
        assert out_buf.stride() == (2,)
        tu.assert_result_equal(guard, ref_guard)


@pytest.mark.q_per_channel_scales
@pytest.mark.parametrize(
    "family,scales_dtype,zero_points_dtype,storage_dtype", ALIAS_CASES
)
def test_q_per_channel_scales_returns_stored_parameter(
    family, scales_dtype, zero_points_dtype, storage_dtype
):
    inp, ref_inp = _quantized_pair(
        (16, 128, 64, 60), 1, storage_dtype, scales_dtype, zero_points_dtype
    )
    stored = inp.q_per_channel_scales()
    ref_snapshot = torch.ops.aten.q_per_channel_scales(ref_inp).detach().clone()

    res_out = flag_gems.q_per_channel_scales(inp)

    # The default entry point hands back the quantizer's own parameter tensor instead of
    # recomputing or copying it, and the stored metadata survives the call untouched.
    assert res_out is stored
    assert res_out is not inp
    assert res_out.data_ptr() == stored.data_ptr()
    assert res_out.dtype == RESULT_DTYPE[family]
    assert res_out.shape == (128,)
    assert res_out.requires_grad is False
    assert inp.q_per_channel_scales() is stored
    assert inp.q_per_channel_axis() == 1
    tu.assert_result_equal(res_out, ref_snapshot)


@pytest.mark.q_per_channel_scales
@pytest.mark.parametrize("shape,axis", INT32_ZERO_POINT_CASES)
def test_q_per_channel_scales_int32_zero_points(shape, axis):
    inp, ref_inp = _quantized_pair(
        shape, axis, torch.qint32, torch.float64, torch.int32
    )

    ref_snapshot = torch.ops.aten.q_per_channel_scales(ref_inp).detach().clone()
    res_out = flag_gems.q_per_channel_scales(inp)

    assert res_out.dtype == torch.float64
    assert res_out.shape == (_channels(shape, axis),)
    tu.assert_result_equal(res_out, ref_snapshot)


@pytest.mark.q_per_channel_scales
@pytest.mark.parametrize(
    "value_range,family,scales_dtype,zero_points_dtype,storage_dtype", SCALE_RANGE_CASES
)
def test_q_per_channel_scales_scale_values(
    value_range, family, scales_dtype, zero_points_dtype, storage_dtype
):
    # The spec's five value ranges applied to the scale parameter, the only place an
    # arithmetic value can live for this metadata accessor.
    scale_values = tu.make_input(scales_dtype, (1024,), value_range).tolist()
    inp, ref_inp = _quantized_pair(
        (1024, 1024),
        -1,
        storage_dtype,
        scales_dtype,
        zero_points_dtype,
        scale_values=scale_values,
    )

    ref_snapshot = torch.ops.aten.q_per_channel_scales(ref_inp).detach().clone()
    res_out = flag_gems.q_per_channel_scales(inp)

    assert res_out.dtype == RESULT_DTYPE[family]
    assert res_out.shape == (1024,)
    tu.assert_result_equal(res_out, ref_snapshot)


@pytest.mark.q_per_channel_scales
@pytest.mark.parametrize(
    "scenario,family,scales_dtype,zero_points_dtype,storage_dtype", SPECIAL_SCALE_CASES
)
def test_q_per_channel_scales_special_scale_values(
    scenario, family, scales_dtype, zero_points_dtype, storage_dtype
):
    # nan / inf / -0.0 can only enter through the stored parameter tensors; the list is
    # rebuilt per operand so the two operands stay independent.
    scale_values = tu.make_special_input(torch.float32, scenario).tolist()
    shape = (4, len(scale_values), 6)
    inp, ref_inp = _quantized_pair(
        shape,
        1,
        storage_dtype,
        scales_dtype,
        zero_points_dtype,
        scale_values=scale_values,
    )

    ref_snapshot = torch.ops.aten.q_per_channel_scales(ref_inp).detach().clone()
    res_out = flag_gems.q_per_channel_scales(inp)

    assert res_out.dtype == RESULT_DTYPE[family]
    assert res_out.shape == (len(scale_values),)
    tu.assert_result_equal(res_out, ref_snapshot)


def _invalid_args(case):
    # Only the operands the selected case needs are built, so nothing but the candidate
    # call inside pytest.raises can raise.
    if case == "non_tensor_input":
        return ([1.0, 2.0, 3.0], {})
    if case == "plain_float_tensor":
        return (torch.zeros((4, 4), dtype=torch.float32, device=flag_gems.device), {})
    if case == "not_per_channel":
        return (
            torch.quantize_per_tensor(
                torch.zeros((4, 4), device=flag_gems.device), 0.25, 0, torch.quint8
            ),
            {},
        )
    inp = _make_operand((4, 32, 8), 1, torch.qint8, torch.float64, torch.int64)
    if case == "out_dtype_mismatch":
        # Affine metadata is float64, so a float32 buffer must be rejected.
        return (
            inp,
            {"out": torch.empty(32, dtype=torch.float32, device=flag_gems.device)},
        )
    return (inp, {"out": [0.0] * 32})


@pytest.mark.q_per_channel_scales
@pytest.mark.parametrize("case", NEGATIVE_CASES)
def test_q_per_channel_scales_invalid_inputs(case):
    inp, kwargs = _invalid_args(case)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.q_per_channel_scales(inp, **kwargs)
