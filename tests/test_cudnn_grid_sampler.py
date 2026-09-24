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


"""Correctness tests for aten::cudnn_grid_sampler.

The native operator takes rank-4 operands with ``grid.size(-1) == 2`` and
``grid.size(0) == input.size(0)``, so the spec's shape levels are expressed as
(input_shape, grid_shape) pairs and values come from the shared value-range
framework.
"""

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Static capability flags: read once, no tensor work at import or collection.
_HAS_FP64 = bool(getattr(flag_gems.runtime.device, "support_fp64", False))
_SUPPORTS = {
    "bf16": bool(getattr(flag_gems.runtime.device, "support_bf16", False)),
    "fp8": bool(getattr(flag_gems.runtime.device, "support_fp8", False)),
    "int64": bool(getattr(flag_gems.runtime.device, "support_int64", False)),
}
_VENDOR = getattr(flag_gems, "vendor_name", "")

# The native kernel accepts these floating dtypes only.
_FLOAT_DTYPES = [torch.float32, torch.float16] + ([torch.float64] if _HAS_FP64 else [])

# Every other operand dtype is rejected by the kernel. A dtype is listed only
# when this device can even create it, and the rejection expectation itself is
# scoped to the backend it was measured on.
_UNSUPPORTED_DTYPES = []
if _VENDOR == "nvidia":
    if _SUPPORTS["bf16"]:
        _UNSUPPORTED_DTYPES.append(torch.bfloat16)
    _UNSUPPORTED_DTYPES.append(torch.int8)
    _UNSUPPORTED_DTYPES.append(torch.uint8)
    if _SUPPORTS["fp8"]:
        _UNSUPPORTED_DTYPES.extend((torch.float8_e4m3fn, torch.float8_e5m2))
    _UNSUPPORTED_DTYPES.append(torch.int32)
    if _SUPPORTS["int64"]:
        _UNSUPPORTED_DTYPES.append(torch.int64)
    _UNSUPPORTED_DTYPES.append(torch.bool)

# Input and grid must share one dtype; float32 is always available to pair with.
_MISMATCH_DTYPES = (
    [torch.float16] + ([torch.float64] if _HAS_FP64 else [])
    if _VENDOR == "nvidia"
    else []
)

_INVALID_INPUT_ERRORS = (RuntimeError, TypeError)

# Sampling coordinates stay inside [-1, 1] so every case samples real pixels.
_GRID_RANGE = ["-1", "1"]

# (input_shape, grid_shape) pairs: rank-4 operands, grid last dim 2, equal batch.
_SHAPE_CASES = [
    ((16, 128, 64, 60), (16, 8, 8, 2)),  # spec 4-D level, batch 16
    ((1, 1, 1024, 1024), (1, 32, 32, 2)),  # 1024x1024 spatial boundary
    ((2, 3, 5, 7), (2, 4, 4, 2)),  # tiny spatial extent
    ((1, 1, 1, 1), (1, 1, 1, 2)),  # single pixel
    ((4, 8, 33, 17), (4, 3, 5, 2)),  # odd spatial sizes
    ((3, 2, 64, 64), (3, 64, 64, 2)),  # one sample per pixel
    ((1, 16, 8, 8), (1, 2, 2, 2)),  # many channels, few pixels
    ((2, 6, 128, 128), (2, 16, 16, 2)),  # 128x128 input sampled down to 16x16
]
_QUICK_SHAPE_CASES = [((2, 3, 19, 7), (2, 8, 7, 2))]

_SHAPE_PARAMS = tu.selected_cases(_SHAPE_CASES, quick=_QUICK_SHAPE_CASES)
# The remaining dimensions are default-only sweeps; quick keeps the required
# negatives and one representative positive workload.
_OUT_SHAPE_CASES = tu.selected_cases(_SHAPE_CASES[:4], quick=[])
_BACKWARD_CASES = tu.selected_cases(
    [((2, 3, 16, 16), (2, 4, 4, 2)), ((1, 2, 8, 8), (1, 3, 5, 2))], quick=[]
)
_LAYOUT_CASES = tu.selected_cases(
    ["input_strided", "grid_strided", "both_strided", "storage_offset"], quick=[]
)
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])
_BOUNDARY_DTYPES = tu.selected_cases(_FLOAT_DTYPES, quick=[])
_SPECIAL_COORD_CASES = tu.selected_cases(
    [
        (dtype, scenario)
        for dtype in _FLOAT_DTYPES
        for scenario in ("nan", "inf", "mixed")
    ],
    quick=[],
)

# Coordinates on the border and just outside it are native-valid; whether a
# sample keeps any edge weight there is decided by the native kernel, so every
# row is compared against it and no zero fill is assumed for any coordinate.
_BOUNDARY_COORDS = (
    (-1.0, -1.0),
    (1.0, 1.0),
    (0.999, 0.999),
    (-0.999, -0.999),
    (1.001, 1.001),
    (-1.001, -1.001),
    (1.001, -1.001),
    (1.5, 1.5),
    (-1.5, -1.5),
    (3.0, -3.0),
)

# Deterministic sample points spread over the sampled plane.
_SPECIAL_COORDS = (
    (-1.0, -1.0),
    (-0.5, 0.25),
    (0.0, 0.0),
    (0.25, -0.5),
    (0.5, 0.5),
    (0.75, -0.25),
    (1.0, 1.0),
    (0.125, 0.875),
)

# Non-finite sampling coordinates are native-valid: measured on this backend the
# affected samples come back as NaN instead of raising, so they are compared
# against the native result and NaN matching is left to the shared helper.
_COORD_PAYLOADS = {
    "nan": ((float("nan"), 0.0), (0.0, float("nan")), (-0.5, 0.5)),
    "inf": ((float("inf"), 0.0), (0.0, float("-inf")), (0.5, 0.5)),
    "mixed": ((float("nan"), float("inf")), (float("-inf"), 0.0), (0.25, -0.25)),
}

_INVALID_SHAPE_CASES = [
    ((1, 3, 8, 8), (1, 4, 4, 3)),  # grid last dim must be 2
    ((1, 3, 8, 8), (1, 4, 4)),  # grid must be rank 4
    ((3, 8, 8), (1, 4, 4, 2)),  # input must be rank 4
    ((1, 3, 8, 8), (2, 4, 4, 2)),  # batch sizes must match
    ((), ()),  # 0-dim operands
]


def _out_shape(input_shape, grid_shape):
    """Output shape (N, C, grid_h, grid_w) from operand metadata."""
    return (grid_shape[0], input_shape[1], grid_shape[1], grid_shape[2])


def _coordinate_grid(dtype, coords):
    """Rank-4 batch-1 grid of (x, y) pairs, built directly in ``dtype``."""
    grid = torch.tensor(coords, dtype=dtype, device=flag_gems.device)
    return grid.reshape(1, 1, len(coords), 2)


def _special_input(dtype, payload):
    """(1, 1, 8, 8) input tiling ``payload`` over every pixel.

    Tiling means each sample point in ``_SPECIAL_COORDS`` lands on a payload
    element, so the comparison exercises NaN/Inf propagation instead of reading
    the finite interior between a few scattered pixels.
    """
    shape = (1, 1, 8, 8)
    index = torch.arange(math.prod(shape), device=payload.device) % payload.numel()
    return payload[index].reshape(shape).to(dtype)


def _layout_inputs(layout, dtype):
    """Operand pair with non-contiguous strides or a non-zero storage offset."""
    inp = tu.make_input(dtype, (2, 4, 16, 16), _GRID_RANGE)
    grid = tu.make_input(dtype, (2, 4, 8, 2), _GRID_RANGE)
    # A wider contiguous grid whose last dim is strided down to 2.
    wide_grid = tu.make_input(dtype, (2, 4, 8, 4), _GRID_RANGE)[:, :, :, ::2]
    if layout == "input_strided":
        return inp[:, :, :, ::2], grid
    if layout == "grid_strided":
        return inp, wide_grid
    if layout == "both_strided":
        return inp[:, :, ::2, :], wide_grid
    if layout == "storage_offset":
        padded = torch.zeros(2, 4, 16, 18, dtype=dtype, device=flag_gems.device)
        padded[:, :, :, 1:17] = inp
        return padded[:, :, :, 1:17], grid
    raise AssertionError(f"unknown layout {layout}")


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("input_shape, grid_shape", _SHAPE_PARAMS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_cudnn_grid_sampler(input_shape, grid_shape, value_range, dtype):
    inp = tu.make_input(dtype, input_shape, value_range)
    grid = tu.make_input(dtype, grid_shape, _GRID_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("input_shape, grid_shape", _OUT_SHAPE_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_cudnn_grid_sampler_out(input_shape, grid_shape, dtype):
    inp = tu.make_input(dtype, input_shape, _GRID_RANGE)
    grid = tu.make_input(dtype, grid_shape, _GRID_RANGE)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)
    out_shape = _out_shape(input_shape, grid_shape)

    ref_buf = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_ret = torch.ops.aten.cudnn_grid_sampler.out(ref_inp, ref_grid, out=ref_buf)

    res_buf = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    res_ret = flag_gems.cudnn_grid_sampler(inp, grid, out=res_buf)

    # The out overload must return the caller's buffer itself: an aliased view
    # that merely shares its storage is not the buffer object.
    assert res_ret is res_buf
    tu.assert_result_close(res_ret, ref_ret)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("layout", _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_cudnn_grid_sampler_layout(layout, dtype):
    inp, grid = _layout_inputs(layout, dtype)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype, scenario", _SPECIAL_CASES)
def test_cudnn_grid_sampler_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    inp = _special_input(dtype, payload)
    grid = _coordinate_grid(dtype, _SPECIAL_COORDS)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype, scenario", _SPECIAL_COORD_CASES)
def test_cudnn_grid_sampler_special_coordinates(dtype, scenario):
    inp = tu.make_input(dtype, (1, 4, 16, 16), _GRID_RANGE)
    grid = _coordinate_grid(dtype, _COORD_PAYLOADS[scenario])
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
def test_cudnn_grid_sampler_border_coordinates(dtype):
    inp = tu.make_input(dtype, (1, 3, 8, 8), _GRID_RANGE)
    grid = _coordinate_grid(dtype, _BOUNDARY_COORDS)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("input_shape, grid_shape", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_cudnn_grid_sampler_backward(input_shape, grid_shape, dtype):
    inp = tu.make_input(dtype, input_shape, _GRID_RANGE).requires_grad_(True)
    grid = tu.make_input(dtype, grid_shape, _GRID_RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    ref_grid = tu.to_reference(grid)

    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)
    tu.assert_result_close(res_out, ref_out)

    upstream = tu.make_input(dtype, tuple(ref_out.shape), _GRID_RANGE)
    ref_upstream = tu.to_reference(upstream)
    ref_grads = torch.autograd.grad(ref_out, (ref_inp, ref_grid), ref_upstream)
    res_grads = torch.autograd.grad(res_out, (inp, grid), upstream)

    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_cudnn_grid_sampler_unsupported_dtype(dtype):
    inp = torch.zeros((1, 1, 8, 8), dtype=dtype, device=flag_gems.device)
    grid = torch.zeros((1, 1, 1, 2), dtype=dtype, device=flag_gems.device)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.cudnn_grid_sampler(inp, grid)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("grid_dtype", _MISMATCH_DTYPES)
def test_cudnn_grid_sampler_dtype_mismatch(grid_dtype):
    inp = torch.zeros((1, 1, 8, 8), dtype=torch.float32, device=flag_gems.device)
    grid = torch.zeros((1, 1, 1, 2), dtype=grid_dtype, device=flag_gems.device)
    with pytest.raises(_INVALID_INPUT_ERRORS):
        flag_gems.cudnn_grid_sampler(inp, grid)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("input_shape, grid_shape", _INVALID_SHAPE_CASES)
def test_cudnn_grid_sampler_invalid_shapes(input_shape, grid_shape):
    inp = torch.zeros(input_shape, dtype=torch.float32, device=flag_gems.device)
    grid = torch.zeros(grid_shape, dtype=torch.float32, device=flag_gems.device)
    # The 0-dim form fails inside ATen with IndexError rather than RuntimeError.
    with pytest.raises((*_INVALID_INPUT_ERRORS, IndexError)):
        flag_gems.cudnn_grid_sampler(inp, grid)
