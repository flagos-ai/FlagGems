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

import flag_gems

from . import test_utils as tu

# aten::q_per_channel_axis(Tensor self) -> int reports the channel axis stored by
# a per-channel quantizer. Only saved metadata is read, so the result is a Python
# int compared directly with the native one: this operator has no tensor value, no
# broadcast form, no gradient and no element-wise rounding.
#
# Quantized operands are materialized, never computed: torch.quantize_per_channel
# provides the value-range grid over a real element pattern, and
# torch.ops.aten._empty_per_channel_affine_quantized provides the rows that must
# store a raw negative axis, the 0-dim shape, an empty dim or a sub-byte dtype.
# Both are input builders for this accessor.

QUANTIZER_DTYPES = [torch.qint8, torch.quint8, torch.qint32]

# (shape, axis) rows for the value-range grid. The axes are non-negative because
# torch.quantize_per_channel validates the channel axis against the raw rank
# ("Channel axis out of range in per channel affine quantization. Got: -1
# Expected: [0, 2)") and a 0-dim operand has no channel axis at all; both forms
# are covered by FACTORY_ROWS instead.
GRID_GEOMETRY = tu.selected_cases(
    [
        ((1,), 0),
        ((256,), 0),
        ((1024, 1024), 1),
        ((20, 320, 15), 1),
        ((2, 19, 7), 1),
        ((16, 128, 64, 60), 2),
        ((16, 7, 57, 32, 29), 0),
    ],
    quick=[((2, 19, 7), 1)],
)

# (shape, axis, dtype, zero_point_kind) rows built by
# torch.ops.aten._empty_per_channel_affine_quantized, which stores the axis
# verbatim (probed for -1/-2 at rank 2 and -3 at rank 3) and accepts the sub-byte
# types plus the 0-dim and empty-dim forms the public quantizer cannot express.
# Integer zero points select torch.per_channel_affine, float32 zero points select
# torch.per_channel_affine_float_qparams. These rows are small semantic cases and
# are kept in both modes.
FACTORY_ROWS = [
    ((), 0, torch.qint8, "int"),
    ((2, 3), 0, torch.quint8, "int"),
    ((2, 3), 1, torch.qint32, "int"),
    ((4, 6), -1, torch.qint8, "int"),
    ((4, 6), -2, torch.qint8, "int"),
    ((4, 6), -1, torch.quint4x2, "int"),
    ((4, 6), -1, torch.quint2x4, "int"),
    ((2, 3, 4), -1, torch.qint8, "int"),
    ((2, 3, 4), -3, torch.qint8, "int"),
    ((8, 2, 3), 2, torch.qint8, "int"),
    ((0, 2, 3), 1, torch.qint8, "int"),
    ((2, 0), 1, torch.qint8, "int"),
    ((2, 0), -1, torch.qint8, "int"),
    ((), 0, torch.qint8, "float"),
    ((4, 6), 1, torch.qint8, "float"),
    ((4, 6), -1, torch.qint8, "float"),
    ((4, 6), -1, torch.quint8, "float"),
    ((4, 6), -1, torch.quint4x2, "float"),
    ((4, 6), -1, torch.quint2x4, "float"),
    ((2, 3, 4), -2, torch.qint32, "float"),
]

# (layout, expected axis, expected shape). Every row is a view of one (4, 6)
# axis-1 affine tensor, so the accessor must report the view's own saved axis. The
# row_index row stores axis 0 against the base tensor's axis 1, which is what makes
# the distinction observable. Kept in both modes.
VIEW_ROWS = [
    ("row_step", 1, (2, 6)),
    ("col_step", 1, (4, 3)),
    ("row_offset", 1, (3, 6)),
    ("row_index", 0, (6,)),
    ("detached", 1, (4, 6)),
    ("reshaped", 1, (2, 12)),
]

# The accessor returns a Python int, so it has no output tensor and no output
# device. These rows check that a device-resident operand, a host-resident operand
# (reachable in this checkout through .to("cpu")) and a clone owning fresh storage
# all report the stored axis.
STATE_ROWS = ["device_resident", "cpu_resident", "clone"]

# Positive special-value cases are default-only.
SPECIAL_ROWS = tu.selected_cases(["nan", "inf", "mixed"], quick=[])

# Invalid-input rows, kept in both modes.
INVALID_ROWS = [
    "per_tensor_device",
    "per_tensor_cpu",
    "plain_float_tensor",
    "non_tensor_int",
]


def _quantize_per_channel(source, axis, dtype):
    """Per-channel quantized operand built from a source element pattern."""
    channels = source.shape[axis]
    scales = torch.ones(channels, dtype=torch.float64, device=source.device)
    zero_points = torch.zeros(channels, dtype=torch.int64, device=source.device)
    return torch.quantize_per_channel(source, scales, zero_points, axis, dtype)


def _empty_per_channel(shape, axis, dtype, zero_point_kind):
    """Per-channel quantized operand that stores the requested axis verbatim."""
    # A 0-dim operand carries a single channel, the contract the factory accepts.
    channels = 1 if len(shape) == 0 else shape[axis]
    float_qparams = zero_point_kind == "float"
    scales = torch.ones(
        channels,
        dtype=torch.float32 if float_qparams else torch.float64,
        device=flag_gems.device,
    )
    zero_points = torch.zeros(
        channels,
        dtype=torch.float32 if float_qparams else torch.int64,
        device=flag_gems.device,
    )
    return torch.ops.aten._empty_per_channel_affine_quantized(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=dtype,
        device=flag_gems.device,
    )


def _axis1_affine_tensor(dtype=torch.qint8):
    """(4, 6) per-channel affine tensor with the saved axis 1."""
    source = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    return _quantize_per_channel(source, 1, dtype)


def _apply_layout(tensor, layout):
    if layout == "row_step":
        return tensor[::2]
    if layout == "col_step":
        return tensor[:, ::2]
    if layout == "row_offset":
        return tensor[1:]
    if layout == "row_index":
        return tensor[0]
    if layout == "detached":
        return tensor.detach()
    if layout == "reshaped":
        return tensor.reshape(2, 12)
    raise ValueError("unsupported layout " + repr(layout))


@pytest.mark.q_per_channel_axis
@pytest.mark.parametrize("shape,axis", GRID_GEOMETRY)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", QUANTIZER_DTYPES)
def test_q_per_channel_axis_value_ranges(shape, axis, value_range, dtype):
    source = tu.make_input(torch.float32, shape, value_range)
    ref_source = tu.to_reference(source)
    inp = _quantize_per_channel(source, axis, dtype)
    ref_inp = _quantize_per_channel(ref_source, axis, dtype)

    ref_out = torch.ops.aten.q_per_channel_axis(ref_inp)
    res_out = flag_gems.q_per_channel_axis(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    assert res_out == axis


@pytest.mark.q_per_channel_axis
@pytest.mark.parametrize("shape,axis,dtype,zero_point_kind", FACTORY_ROWS)
def test_q_per_channel_axis_factory_metadata(shape, axis, dtype, zero_point_kind):
    inp = _empty_per_channel(shape, axis, dtype, zero_point_kind)
    ref_inp = _empty_per_channel(shape, axis, dtype, zero_point_kind)

    ref_out = torch.ops.aten.q_per_channel_axis(ref_inp)
    res_out = flag_gems.q_per_channel_axis(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    assert res_out == axis


@pytest.mark.q_per_channel_axis
@pytest.mark.parametrize("layout,axis,shape", VIEW_ROWS)
def test_q_per_channel_axis_view_layout(layout, axis, shape):
    base = _axis1_affine_tensor()
    ref_base = tu.to_reference(base)
    inp = _apply_layout(base, layout)
    ref_inp = _apply_layout(ref_base, layout)
    base_axis = torch.ops.aten.q_per_channel_axis(ref_base)

    ref_out = torch.ops.aten.q_per_channel_axis(ref_inp)
    res_out = flag_gems.q_per_channel_axis(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    # Each row aliases the base storage, so only the view's own saved axis can
    # explain the reported value (row_index differs from the base axis).
    assert res_out == axis
    # Reading the view must not disturb the base tensor's saved axis.
    assert flag_gems.q_per_channel_axis(base) == base_axis
    tu.assert_result_equal(base.int_repr(), ref_base.int_repr())


@pytest.mark.q_per_channel_axis
@pytest.mark.parametrize("state", STATE_ROWS)
def test_q_per_channel_axis_operand_state(state):
    inp = _axis1_affine_tensor()
    ref_inp = _axis1_affine_tensor()
    if state == "cpu_resident":
        # A host-resident quantized operand must be accepted exactly like a
        # device-resident one: only the stored metadata is read.
        inp = inp.to("cpu")
        ref_inp = ref_inp.to("cpu")
    elif state == "clone":
        ref_inp = tu.to_reference(inp)
        inp = inp.clone()

    ref_out = torch.ops.aten.q_per_channel_axis(ref_inp)
    res_out = flag_gems.q_per_channel_axis(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    assert res_out == 1


@pytest.mark.q_per_channel_axis
@pytest.mark.parametrize("scenario", SPECIAL_ROWS)
def test_q_per_channel_axis_special_source(scenario):
    source = tu.make_special_input(torch.float32, scenario)
    ref_source = tu.to_reference(source)
    inp = _quantize_per_channel(source, 0, torch.qint8)
    ref_inp = _quantize_per_channel(ref_source, 0, torch.qint8)

    ref_out = torch.ops.aten.q_per_channel_axis(ref_inp)
    res_out = flag_gems.q_per_channel_axis(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    assert res_out == 0


@pytest.mark.q_per_channel_axis
@pytest.mark.parametrize("kind", INVALID_ROWS)
def test_q_per_channel_axis_rejects_invalid_input(kind):
    if kind == "non_tensor_int":
        with pytest.raises((TypeError, RuntimeError)):
            flag_gems.q_per_channel_axis(3)
        return

    if kind == "plain_float_tensor":
        # Only a per-channel quantized operand is valid; the native dispatch has
        # no CUDA/CPU kernel for a plain float tensor.
        inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
        with pytest.raises((NotImplementedError, RuntimeError)):
            flag_gems.q_per_channel_axis(inp)
        return

    device = "cpu" if kind == "per_tensor_cpu" else flag_gems.device
    inp = torch.quantize_per_tensor(
        torch.zeros(4, dtype=torch.float32, device=device), 0.1, 0, torch.qint8
    )
    with pytest.raises(RuntimeError):
        flag_gems.q_per_channel_axis(inp)
