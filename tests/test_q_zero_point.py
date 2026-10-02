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

from . import test_utils as tu

# aten::q_zero_point(Tensor self) -> int reads the zero point already stored in a
# per-tensor affine quantizer; it does not quantize anything. Per-tensor is the
# only supported scheme, the result is a Python int, and the quantizer scale is
# the sole carrier of a non-finite value, so the grids below span the storage
# dtype, the stored zero point, the input layout and the scale.
_STORAGE_TO_QUANT = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}
_STORAGE_DTYPES = list(_STORAGE_TO_QUANT)
_SCALE = 0.5


def _quantized(storage_dtype, shape, value_range, zero_point, scale=_SCALE):
    """Per-tensor affine quantized tensor on the active device."""
    storage = tu.make_input(storage_dtype, shape, value_range)
    return torch.ops.aten._make_per_tensor_quantized_tensor(storage, scale, zero_point)


def _view(inp, layout):
    """Return the quantized view named by ``layout``."""
    if layout == "slice":  # stride-2 slice, non-contiguous
        return inp[:, ::2]
    if layout == "transpose":  # non-contiguous transposed view
        return inp.t()
    if layout == "narrow":  # view with a non-zero storage offset
        return inp.narrow(1, 2, 3)
    if layout == "reshape":
        return inp.reshape(-1)
    raise ValueError(f"unknown layout: {layout}")


# tu.selected_shapes() plus (0,): an empty quantized tensor is a valid input.
@pytest.mark.q_zero_point
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes() + [(0,)])
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_q_zero_point(storage_dtype, shape, value_range):
    inp = _quantized(storage_dtype, shape, value_range, 7)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.q_zero_point(ref_inp)
    res_out = flag_gems.q_zero_point(inp)

    assert type(res_out) is int
    assert res_out == ref_out


# The stored zero point is this operator's value dimension: zero, both signs, the
# dtype boundaries and out-of-range values all round-trip verbatim. The rows are
# a single scalar tensor each, so they stay active in the default and --quick
# suites alike.
_ZERO_POINT_CASES = [
    (torch.uint8, 0),
    (torch.uint8, 1),
    (torch.uint8, 128),
    (torch.uint8, 255),
    (torch.uint8, 256),
    (torch.uint8, -1),
    (torch.uint8, 300),
    (torch.int8, 0),
    (torch.int8, 1),
    (torch.int8, -1),
    (torch.int8, 127),
    (torch.int8, -128),
    (torch.int8, -300),
    (torch.int32, 0),
    (torch.int32, 1),
    (torch.int32, -1),
    (torch.int32, 2**31 - 1),
    (torch.int32, -(2**31)),
]

# The scale is the only field that can hold a non-finite value and it must not
# affect the returned int. The finite boundaries (negative, zero) are ordinary
# parameter values and run in both modes; NaN/Inf are positive special-value
# scenarios, so they are default-only. The scale is a single scalar, so nan-only
# and inf-only are the representable scenarios.
_FINITE_SCALE_CASES = [
    (storage_dtype, scale) for storage_dtype in _STORAGE_DTYPES for scale in (-0.5, 0.0)
]
_NON_FINITE_SCALE_CASES = tu.selected_cases(
    [
        (storage_dtype, scale)
        for storage_dtype in _STORAGE_DTYPES
        for scale in (float("nan"), float("inf"), float("-inf"))
    ],
    quick=[],
)

_LAYOUTS = ["slice", "transpose", "narrow", "reshape"]
# (4, 6) inputs: the layout rows are small, so they keep their coverage of all
# three quantized dtypes and all four view forms in --quick as well.
_LAYOUT_CASES = [
    (storage_dtype, layout) for storage_dtype in _STORAGE_DTYPES for layout in _LAYOUTS
]

# Non-quantized dtypes have no aten::q_zero_point kernel: the native operator
# raises NotImplementedError for all of them. These rows stay active in both
# suites.
_NON_QUANTIZED_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


@pytest.mark.q_zero_point
@pytest.mark.parametrize("storage_dtype,zero_point", _ZERO_POINT_CASES)
def test_q_zero_point_stored_value(storage_dtype, zero_point):
    inp = _quantized(storage_dtype, (256,), ["-1", "1"], zero_point)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.q_zero_point(ref_inp)
    res_out = flag_gems.q_zero_point(inp)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.q_zero_point
@pytest.mark.parametrize("storage_dtype,scale", _FINITE_SCALE_CASES)
def test_q_zero_point_finite_scale(storage_dtype, scale):
    inp = _quantized(storage_dtype, (16, 4), ["-1", "1"], 7, scale=scale)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.q_zero_point(ref_inp)
    res_out = flag_gems.q_zero_point(inp)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.q_zero_point
@pytest.mark.parametrize("storage_dtype,scale", _NON_FINITE_SCALE_CASES)
def test_q_zero_point_non_finite_scale(storage_dtype, scale):
    inp = _quantized(storage_dtype, (16, 4), ["-1", "1"], 7, scale=scale)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.q_zero_point(ref_inp)
    res_out = flag_gems.q_zero_point(inp)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.q_zero_point
@pytest.mark.parametrize("storage_dtype,layout", _LAYOUT_CASES)
def test_q_zero_point_quantized_view(storage_dtype, layout):
    inp = _quantized(storage_dtype, (4, 6), ["-1", "1"], 11)
    ref_inp = tu.to_reference(inp)
    view = _view(inp, layout)
    ref_view = _view(ref_inp, layout)
    snapshot = view.detach().clone()

    ref_out = torch.ops.aten.q_zero_point(ref_view)
    res_out = flag_gems.q_zero_point(view)

    assert type(res_out) is int
    assert res_out == ref_out
    # Reading the quantizer must not rewrite the view's storage.
    tu.assert_result_equal(view, snapshot)


@pytest.mark.q_zero_point
@pytest.mark.parametrize("dtype", _NON_QUANTIZED_DTYPES)
def test_q_zero_point_rejects_non_quantized(dtype):
    inp = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.q_zero_point(inp)


@pytest.mark.q_zero_point
@pytest.mark.parametrize("axis", [0, 1])
def test_q_zero_point_rejects_per_channel(axis):
    # A valid per-channel quantizer (one scale/zero point per channel along
    # ``axis``) is still rejected: the operator accepts per-tensor affine only.
    shape = (2, 3)
    channels = shape[axis]
    storage = torch.zeros(shape, dtype=torch.int8, device=flag_gems.device)
    scales = torch.full((channels,), 0.5, dtype=torch.float64, device=flag_gems.device)
    zero_points = torch.zeros(channels, dtype=torch.int64, device=flag_gems.device)
    inp = torch.ops.aten._make_per_channel_quantized_tensor(
        storage, scales, zero_points, axis
    )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.q_zero_point(inp)


@pytest.mark.q_zero_point
@pytest.mark.parametrize("value", [3, 3.14, [1, 2]])
def test_q_zero_point_rejects_non_tensor(value):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.q_zero_point(value)
