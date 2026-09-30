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

# The native FBGEMM kernel is CPU-only and float32-only: the activation must be
# float32 with dim() >= 2 (input.size(-1) is the reduction dim), the weight is
# int8 (N, K), col_offsets int32, the bias float32 of length N, weight_scale a
# Python number and weight_zero_point an integral Python number. Every operand
# is therefore built on CPU.
DTYPES = [torch.float32]

# Rank >= 2 only: 0-dim/1-dim activations are rejected by the native kernel.
_GRID_SHAPES = [
    (2, 19, 7),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
SHAPES = tu.selected_cases(_GRID_SHAPES, quick=[(2, 19, 7)])

# Small geometry shared by the layout, parameter and special-value workloads.
_SMALL_SHAPE = (4, 32)
_MAX_OUTPUT_WIDTH = 8

# Native-valid activation layouts. These are cheap single-operand layout
# fixtures (not multi-operand broadcasts), so every mode runs all of them;
# the last two rows are the singleton and the native empty activation.
_LAYOUT_ROWS = [
    ("plain", (4, 32)),
    ("transposed", (4, 32)),
    ("inner_stride2", (4, 32)),
    ("inner_offset_stride2", (4, 32)),
    ("inner_broadcast", (4, 32)),
    ("plain", (1, 32)),
    ("plain", (0, 32)),
]
LAYOUT_CASES = tu.selected_cases(
    [
        (layout, shape, value_range)
        for layout, shape in _LAYOUT_ROWS
        for value_range in tu.REQUIRED_RANGES
    ],
    quick=[(layout, shape, ["-1", "1"]) for layout, shape in _LAYOUT_ROWS],
)

_BIAS_LAYOUT_ROWS = [
    "plain",
    "inner_stride2",
    "inner_offset_stride2",
    "inner_broadcast",
]
BIAS_LAYOUT_CASES = tu.selected_cases(
    [
        (layout, value_range)
        for layout in _BIAS_LAYOUT_ROWS
        for value_range in tu.REQUIRED_RANGES
    ],
    quick=[(layout, ["-1", "1"]) for layout in _BIAS_LAYOUT_ROWS],
)

# weight_scale is a Python number: positive / negative / zero, magnitude
# boundaries and the nan/inf float boundaries.
_SCALE_VALUES = [1.0, -0.5, 0.0, 1e-4, 1e4, float("nan"), float("inf"), float("-inf")]
# weight_zero_point is an integral Python number; the kernel accepts any int.
_ZERO_POINT_VALUES = [0, 1, -1, 127, -128, 255, 3, -3]
_PARAM_SHAPES = [(1024, 1024), (20, 320, 15), (16, 128, 64, 60)]

SCALE_CASES = tu.selected_cases(
    [(_SMALL_SHAPE, scale) for scale in _SCALE_VALUES]
    + [(shape, scale) for shape in _PARAM_SHAPES for scale in (1.0, 0.0, float("inf"))],
    quick=[(_SMALL_SHAPE, scale) for scale in _SCALE_VALUES if math.isfinite(scale)],
)
ZERO_POINT_CASES = tu.selected_cases(
    [(_SMALL_SHAPE, zero_point) for zero_point in _ZERO_POINT_VALUES]
    + [(shape, zero_point) for shape in _PARAM_SHAPES for zero_point in (0, 127, -128)],
    quick=[(_SMALL_SHAPE, zero_point) for zero_point in _ZERO_POINT_VALUES],
)

_SPECIAL_SHAPES = [(4, 8), (3, 5, 6)]
SPECIAL_CASES = tu.selected_cases(
    [
        (shape, dtype, scenario)
        for shape in _SPECIAL_SHAPES
        for dtype, scenario in tu.special_value_cases(DTYPES)
    ],
    quick=[],
)
_BIAS_SPECIAL_SCENARIOS = ["nan", "inf", "mixed"]
BIAS_SPECIAL_CASES = tu.selected_cases(
    [
        (shape, scenario)
        for shape in _SPECIAL_SHAPES
        for scenario in _BIAS_SPECIAL_SCENARIOS
    ],
    quick=[],
)

_INVALID_INPUT_DTYPES = [
    dtype for dtype in tu.REQUIRED_DTYPES if dtype != torch.float32
] + [torch.float64, torch.bool]

_INVALID_SCALAR_ROWS = [
    ("tensor_scale", 0),  # materialized inside the test
    (1.0, 0.5),  # weight_zero_point must be integral
]


def _activation(layout, shape, dtype, value_range):
    """Activation in one of the native-valid layouts, built on CPU."""
    if layout == "plain":
        return tu.make_input(dtype, shape, value_range).cpu()
    rows, inner = math.prod(shape[:-1]), shape[-1]
    if layout == "transposed":
        # (inner, rows).t(): non-contiguous view, still dim() == 2.
        return tu.make_input(dtype, (inner, rows), value_range).cpu().t()
    if layout == "inner_stride2":
        return tu.make_input(dtype, (rows, 2 * inner), value_range).cpu()[:, ::2]
    if layout == "inner_offset_stride2":
        return tu.make_input(dtype, (rows, 2 * inner + 1), value_range).cpu()[:, 1::2]
    if layout == "inner_broadcast":
        return tu.make_input(dtype, (1, inner), value_range).cpu().expand(rows, inner)
    raise AssertionError(layout)


def _bias(layout, size, dtype, value_range):
    """Bias operand of length ``size`` in the requested native-valid layout."""
    if layout == "plain":
        return tu.make_input(dtype, (size,), value_range).cpu()
    if layout == "inner_stride2":
        return tu.make_input(dtype, (2 * size,), value_range).cpu()[::2]
    if layout == "inner_offset_stride2":
        return tu.make_input(dtype, (2 * size,), value_range).cpu()[1::2]
    if layout == "inner_broadcast":
        return tu.make_input(dtype, (1,), value_range).cpu().expand(size)
    raise AssertionError(layout)


def _native_operands(inner_dim):
    """One native operand set: int8 weight, its packed handle, int32 column
    offsets and a float32 bias of the matching output width."""
    n = min(_MAX_OUTPUT_WIDTH, inner_dim)
    weight = torch.randint(-128, 128, (n, inner_dim), dtype=torch.int8)
    packed = torch.ops.aten.fbgemm_pack_quantized_matrix(weight)
    col_offsets = torch.sum(weight, dim=1, dtype=torch.int32)
    bias = torch.randn(n, dtype=torch.float32)
    return weight, packed, col_offsets, bias


def _reference_operands(weight, col_offsets, bias):
    """Value-identical independent operands for the oracle call.

    The opaque packed handle stays alive and is passed separately to each call;
    dense operands have independent storage.
    """
    return (
        tu.to_reference(weight),
        tu.to_reference(col_offsets),
        tu.to_reference(bias),
    )


def _special_activation(shape, dtype, scenario):
    """Full-size activation holding the shared special-value payload."""
    payload = tu.make_special_input(dtype, scenario).cpu()
    inp = torch.zeros(shape, dtype=dtype)
    inp.reshape(-1)[: payload.numel()] = payload
    return inp


def _special_bias(size, dtype, scenario):
    """Bias whose first entries are the shared special-value payload."""
    payload = tu.make_special_input(dtype, scenario).cpu()
    bias = torch.zeros(size, dtype=dtype)
    bias[: payload.numel()] = payload
    return bias


def _invalid_bias(kind, size):
    """Bias violating one native bias constraint."""
    bias = torch.randn(size, dtype=torch.float32)
    if kind == "dtype":
        return bias.double()
    if kind == "rank":
        return bias.unsqueeze(1)
    return bias[:-1].clone()


def _snapshot(*tensors):
    return [tensor.clone() for tensor in tensors]


def _assert_unmodified(operands, snapshots):
    for operand, before in zip(operands, snapshots):
        tu.assert_result_equal(operand, before)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_linear_int8_weight_fp32_activation(shape, dtype, value_range):
    inp = _activation("plain", shape, dtype, value_range)
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, bias = _native_operands(shape[-1])
    ref_weight, ref_col_offsets, ref_bias = _reference_operands(
        weight, col_offsets, bias
    )
    snapshots = _snapshot(inp, weight, packed, col_offsets, bias)

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, 1.0, 0, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, 1.0, 0, bias
    )

    _assert_unmodified((inp, weight, packed, col_offsets, bias), snapshots)
    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("layout,shape,value_range", LAYOUT_CASES)
def test_fbgemm_linear_int8_weight_fp32_activation_layout(layout, shape, value_range):
    inp = _activation(layout, shape, torch.float32, value_range)
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, bias = _native_operands(shape[-1])
    ref_weight, ref_col_offsets, ref_bias = _reference_operands(
        weight, col_offsets, bias
    )
    snapshots = _snapshot(inp, weight, packed, col_offsets, bias)

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, 1.0, 0, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, 1.0, 0, bias
    )

    _assert_unmodified((inp, weight, packed, col_offsets, bias), snapshots)
    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("bias_layout,value_range", BIAS_LAYOUT_CASES)
def test_fbgemm_linear_int8_weight_fp32_activation_bias_layout(
    bias_layout, value_range
):
    shape = _SMALL_SHAPE
    inp = _activation("plain", shape, torch.float32, value_range)
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, _ = _native_operands(shape[-1])
    ref_weight, ref_col_offsets = tu.to_reference(weight), tu.to_reference(col_offsets)
    bias = _bias(bias_layout, weight.shape[0], torch.float32, value_range)
    ref_bias = tu.to_reference(bias)
    snapshots = _snapshot(inp, weight, packed, col_offsets, bias)

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, 1.0, 0, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, 1.0, 0, bias
    )

    _assert_unmodified((inp, weight, packed, col_offsets, bias), snapshots)
    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("shape,weight_scale", SCALE_CASES)
def test_fbgemm_linear_int8_weight_fp32_activation_weight_scale(shape, weight_scale):
    inp = _activation("plain", shape, torch.float32, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, bias = _native_operands(shape[-1])
    ref_weight, ref_col_offsets, ref_bias = _reference_operands(
        weight, col_offsets, bias
    )

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, weight_scale, 0, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, weight_scale, 0, bias
    )

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("shape,weight_zero_point", ZERO_POINT_CASES)
def test_fbgemm_linear_int8_weight_fp32_activation_weight_zero_point(
    shape, weight_zero_point
):
    inp = _activation("plain", shape, torch.float32, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, bias = _native_operands(shape[-1])
    ref_weight, ref_col_offsets, ref_bias = _reference_operands(
        weight, col_offsets, bias
    )

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, 1.0, weight_zero_point, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, 1.0, weight_zero_point, bias
    )

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("shape,dtype,scenario", SPECIAL_CASES)
def test_fbgemm_linear_int8_weight_fp32_activation_special_values(
    shape, dtype, scenario
):
    inp = _special_activation(shape, dtype, scenario)
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, bias = _native_operands(shape[-1])
    ref_weight, ref_col_offsets, ref_bias = _reference_operands(
        weight, col_offsets, bias
    )

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, 1.0, 0, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, 1.0, 0, bias
    )

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("shape,scenario", BIAS_SPECIAL_CASES)
def test_fbgemm_linear_int8_weight_fp32_activation_bias_special_values(shape, scenario):
    inp = _activation("plain", shape, torch.float32, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, _ = _native_operands(shape[-1])
    ref_weight, ref_col_offsets = tu.to_reference(weight), tu.to_reference(col_offsets)
    bias = _special_bias(weight.shape[0], torch.float32, scenario)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, 1.0, 0, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, 1.0, 0, bias
    )

    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("shape", tu.selected_cases([(4, 32), (3, 5, 6)], quick=[]))
def test_fbgemm_linear_int8_weight_fp32_activation_requires_grad(shape):
    inp = _activation("plain", shape, torch.float32, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    weight, packed, col_offsets, bias = _native_operands(shape[-1])
    ref_weight, ref_col_offsets = tu.to_reference(weight), tu.to_reference(col_offsets)
    ref_bias = tu.to_reference(bias).requires_grad_(True)
    bias = bias.requires_grad_(True)

    ref_out = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        ref_inp, ref_weight, packed, ref_col_offsets, 1.0, 0, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        inp, weight, packed, col_offsets, 1.0, 0, bias
    )

    # The native kernel defines no autograd formula: the output stays detached
    # even with requires_grad inputs, so there is no backward to cover.
    assert res_out.requires_grad == ref_out.requires_grad
    assert (res_out.grad_fn is None) == (ref_out.grad_fn is None)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("dtype", _INVALID_INPUT_DTYPES)
def test_fbgemm_linear_int8_weight_fp32_activation_invalid_input_dtype(dtype):
    inp = torch.zeros(4, 6, dtype=dtype)
    weight, packed, col_offsets, bias = _native_operands(6)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, weight, packed, col_offsets, 1.0, 0, bias
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("shape", [(), (8,)])
def test_fbgemm_linear_int8_weight_fp32_activation_invalid_input_rank(shape):
    inp = torch.zeros(shape, dtype=torch.float32)
    weight, packed, col_offsets, bias = _native_operands(6)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, weight, packed, col_offsets, 1.0, 0, bias
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("weight_shape", [(), (6,), (4, 6, 1)])
def test_fbgemm_linear_int8_weight_fp32_activation_invalid_weight_rank(weight_shape):
    inp = _activation("plain", (4, 6), torch.float32, ["-1", "1"])
    _, packed, col_offsets, bias = _native_operands(6)
    invalid_weight = torch.zeros(weight_shape, dtype=torch.int8)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, invalid_weight, packed, col_offsets, 1.0, 0, bias
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_invalid_col_offsets_dtype():
    inp = _activation("plain", (4, 6), torch.float32, ["-1", "1"])
    weight, packed, col_offsets, bias = _native_operands(6)
    invalid_col_offsets = col_offsets.long()

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, weight, packed, invalid_col_offsets, 1.0, 0, bias
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("bias_kind", ["dtype", "rank", "length"])
def test_fbgemm_linear_int8_weight_fp32_activation_invalid_bias(bias_kind):
    inp = _activation("plain", (4, 6), torch.float32, ["-1", "1"])
    weight, packed, col_offsets, bias = _native_operands(6)
    invalid_bias = _invalid_bias(bias_kind, bias.shape[0])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, weight, packed, col_offsets, 1.0, 0, invalid_bias
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_mismatched_inner_dim():
    inp = _activation("plain", (4, 6), torch.float32, ["-1", "1"])
    weight, packed, col_offsets, bias = _native_operands(5)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, weight, packed, col_offsets, 1.0, 0, bias
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_invalid_packed_dtype():
    inp = _activation("plain", (4, 6), torch.float32, ["-1", "1"])
    weight, _, col_offsets, bias = _native_operands(6)
    invalid_packed = torch.zeros(4, 6, dtype=torch.float32)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, weight, invalid_packed, col_offsets, 1.0, 0, bias
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("weight_scale,weight_zero_point", _INVALID_SCALAR_ROWS)
def test_fbgemm_linear_int8_weight_fp32_activation_invalid_scalars(
    weight_scale, weight_zero_point
):
    if weight_scale == "tensor_scale":
        weight_scale = torch.tensor(1.0)
    inp = _activation("plain", (4, 6), torch.float32, ["-1", "1"])
    weight, packed, col_offsets, bias = _native_operands(6)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
        )
