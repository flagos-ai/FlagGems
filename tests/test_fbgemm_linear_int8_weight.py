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

"""Correctness tests for fbgemm_linear_int8_weight (host FBGEMM int8 linear).

Measured native contract of
``aten::fbgemm_linear_int8_weight(input, weight, packed, col_offsets,
weight_scale, weight_zero_point, bias)`` (CPU kernel):
- CPU-only; activation and bias must be float32. Weight supplies shape metadata,
  while numerical weights come from the packed handle.
- rank(input) >= 2, weight (N, K) with K == input.size(-1), bias length N.
- weight_scale / weight_zero_point must be Python numbers (integral zero point).
- ``packed`` must be the genuine buffer produced by
  ``fbgemm_pack_quantized_matrix``; copies of that handle (clone, set_,
  storage clone) are rejected with "Expected temporary cpp type wrapper", so
  the same handle is shared by the oracle and the candidate.
- col_offsets is accepted but not read.
- No autograd: the result never carries a grad_fn, even with requires_grad
  operands.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

DTYPE = torch.float32

_WORKLOAD_SHAPE = (8, 16)
_WORKLOAD_N = 4

# (input shape, output width N); K = shape[-1].
# The spec's rank-0/rank-1 shapes cannot appear here: the native op rejects
# rank < 2 ("Expected input.dim() >= 2"), so only its rank >= 2 shapes apply.
# (0, 16) is the native-valid empty workload (K must stay positive).
_GRID_ROWS = [
    ((1024, 1024), 128),
    ((1024, 1024), 1),
    ((256, 256), 256),
    ((64, 33), 7),
    ((20, 320, 15), 64),
    ((20, 320, 15), 1),
    ((20, 320, 15), 33),
    ((16, 128, 64, 60), 16),
    ((16, 128, 64, 60), 1),
    ((16, 7, 57, 32, 29), 8),
    ((16, 7, 57, 32, 29), 3),
    ((2, 19, 7), 5),
    ((5, 1), 1),
    ((1, 5), 4),
    ((3, 7, 9, 11), 6),
    ((2, 3, 4, 5, 6), 2),
    ((128, 3), 9),
    ((8, 8), 8),
    ((1, 1), 1),
    ((1, 16), 4),
    ((0, 16), 4),
]
_QUICK_GRID_ROWS = [((2, 19, 7), 5), ((1, 1), 1), ((1, 16), 4), ((0, 16), 4)]
GRID_ROWS = tu.selected_cases(_GRID_ROWS, quick=_QUICK_GRID_ROWS)

# Strided operands: a transposed activation, an activation view with a storage
# offset (offset 8), and a stride-0 expanded activation. All three measured to
# return the same values as their contiguous copies. Every row also uses a
# non-contiguous bias column view (stride 2, storage_offset 1).
_LAYOUT_ROWS = ["transpose", "offset", "expand"]
LAYOUT_ROWS = tu.selected_cases(_LAYOUT_ROWS, quick=_LAYOUT_ROWS)

# Scalar parameter domains. Native accepts nan/inf scales (the whole result
# becomes nan/inf), which stays in the default grid only.
_WEIGHT_SCALE_ROWS = [1.0, -1.0, 0.0, 1e-3, 1e38, float("inf"), float("nan")]
_QUICK_WEIGHT_SCALE_ROWS = [1.0, -1.0, 0.0, 1e-3, 1e38]
WEIGHT_SCALE_CASES = tu.selected_cases(
    _WEIGHT_SCALE_ROWS, quick=_QUICK_WEIGHT_SCALE_ROWS
)

# Negative / zero / int8 boundary zero points; the native kernel does not clamp
# them, it only shifts the dequantized weights.
WEIGHT_ZERO_POINT_CASES = [0, 1, -1, 127, -128]

# Positive special values are default-only and cover every operand that can
# carry them.
_SPECIAL_OPERANDS = ("input", "weight", "bias")
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases([DTYPE]), quick=[])

# Default-only: differentiating the result is a backward-dimension check.
_AUTOGRAD_CASES = tu.selected_cases(["no_graph"], quick=[])

# "expected scalar type Float but found X" for each of these.
_REJECTED_DTYPES = [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.complex64,
    torch.complex128,
]

_REJECTED_INPUT_RANKS = [(), (16,)]
_REJECTED_WEIGHT_RANKS = [(16,), (1, _WORKLOAD_N, 16)]
_REJECTED_BIAS_LENGTHS = [3, 5]
_REJECTED_SCALAR_FORMS = [
    "tensor_weight_scale",
    "tensor_weight_zero_point",
    "fractional_zero_point",
]


def _pack_weights(weight):
    """Native quantization + packing; the op consumes both produced buffers."""
    (
        qweight,
        col_offsets,
        weight_scale,
        weight_zero_point,
    ) = torch.ops.aten.fbgemm_linear_quantize_weight(weight)
    packed = torch.ops.aten.fbgemm_pack_quantized_matrix(qweight)
    return packed, col_offsets, float(weight_scale), int(weight_zero_point)


def _make_linear_operands(shape, n, value_range):
    """CPU operands for one workload (the native op has no device kernel)."""
    inp = tu.make_input(DTYPE, shape, value_range).cpu()
    weight = tu.make_input(DTYPE, (n, shape[-1]), value_range).cpu()
    bias = tu.make_input(DTYPE, (n,), value_range).cpu()
    return inp, weight, bias


def _strided_operands(layout, value_range):
    inp = tu.make_input(DTYPE, (16, 8), value_range).cpu().t()
    if layout == "offset":
        inp = tu.make_input(DTYPE, (8, 32), value_range).cpu()[:, 8:24]
    elif layout == "expand":
        inp = tu.make_input(DTYPE, (1, 16), value_range).cpu().expand(8, 16)
    weight = tu.make_input(DTYPE, (_WORKLOAD_N, 16), value_range).cpu()
    bias = tu.make_input(DTYPE, (_WORKLOAD_N, 2), value_range).cpu()[:, 1]
    return inp, weight, bias


def _snapshot(*tensors):
    return tuple(tensor.clone() for tensor in tensors)


def _assert_read_only(operands, snapshot):
    """The op documents no mutation: dense operands and buffers must hold."""
    for operand, before in zip(operands, snapshot):
        tu.assert_result_equal(operand, before)


def _reference(inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias):
    """Native oracle on independent dense operands.

    ``packed`` is passed through unchanged: the native op validates that opaque
    FBGEMM handle, and any copy of it (clone / set_ / storage clone) fails with
    "Expected temporary cpp type wrapper".
    """
    return torch.ops.aten.fbgemm_linear_int8_weight(
        tu.to_reference(inp),
        tu.to_reference(weight),
        packed,
        col_offsets.clone(),
        weight_scale,
        weight_zero_point,
        tu.to_reference(bias),
    )


def _rejected_scalars(case):
    """One invalid (weight_scale, weight_zero_point) pair per native check."""
    weight_scale, weight_zero_point = 1.0, 0
    if case == "tensor_weight_scale":
        weight_scale = torch.tensor(1.0)
    elif case == "tensor_weight_zero_point":
        weight_zero_point = torch.tensor(0)
    else:
        weight_zero_point = 0.5
    return weight_scale, weight_zero_point


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("shape,n", GRID_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_fbgemm_linear_int8_weight(shape, n, value_range):
    inp, weight, bias = _make_linear_operands(shape, n, value_range)
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)
    before = _snapshot(inp, weight, packed, col_offsets, bias)

    ref = _reference(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    res = flag_gems.fbgemm_linear_int8_weight(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    tu.assert_result_close(res, ref)
    _assert_read_only((inp, weight, packed, col_offsets, bias), before)


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("layout", LAYOUT_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_fbgemm_linear_int8_weight_strided_operands(layout, value_range):
    inp, weight, bias = _strided_operands(layout, value_range)
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)
    before = _snapshot(inp, weight, packed, col_offsets, bias)

    ref = _reference(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    res = flag_gems.fbgemm_linear_int8_weight(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    tu.assert_result_close(res, ref)
    _assert_read_only((inp, weight, packed, col_offsets, bias), before)


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("weight_scale", WEIGHT_SCALE_CASES)
def test_fbgemm_linear_int8_weight_weight_scale(weight_scale):
    inp, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, _, weight_zero_point = _pack_weights(weight)

    ref = _reference(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    res = flag_gems.fbgemm_linear_int8_weight(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    tu.assert_result_close(res, ref)


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("weight_zero_point", WEIGHT_ZERO_POINT_CASES)
def test_fbgemm_linear_int8_weight_weight_zero_point(weight_zero_point):
    inp, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, weight_scale, _ = _pack_weights(weight)
    before = _snapshot(inp, weight, bias)

    ref = _reference(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    res = flag_gems.fbgemm_linear_int8_weight(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    tu.assert_result_close(res, ref)
    _assert_read_only((inp, weight, bias), before)


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("operand", _SPECIAL_OPERANDS)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_fbgemm_linear_int8_weight_special_values(dtype, scenario, operand):
    inp, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    payload = tu.make_special_input(dtype, scenario).cpu()
    target = {"input": inp, "weight": weight, "bias": bias}[operand]
    flat = target.flatten()
    count = min(flat.numel(), payload.numel())
    flat[:count] = payload[:count]
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)

    ref = _reference(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    res = flag_gems.fbgemm_linear_int8_weight(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    tu.assert_result_close(res, ref)


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("case", _AUTOGRAD_CASES)
def test_fbgemm_linear_int8_weight_autograd_metadata(case):
    del case
    inp, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    inp.requires_grad_(True)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(
        weight.detach()
    )

    ref = _reference(
        inp.detach(),
        weight.detach(),
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
        bias.detach(),
    )
    res = flag_gems.fbgemm_linear_int8_weight(
        inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    tu.assert_result_close(res, ref)
    # Measured native behavior: requires_grad operands still produce a
    # graph-free result, so differentiating it must fail.
    assert res.requires_grad is False
    assert res.grad_fn is None
    with pytest.raises(RuntimeError):
        torch.autograd.grad(res, inp, grad_outputs=torch.ones_like(res))


# Negative cases: the invalid argument is always built before the raises block
# so a construction failure cannot be mistaken for the op's rejection.
@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test_fbgemm_linear_int8_weight_rejects_dtype(dtype):
    _, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)
    # CPU-only op: the invalid operand sits next to the CPU operands it must match.
    bad_inp = torch.ones(_WORKLOAD_SHAPE, dtype=dtype)
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_int8_weight(
            bad_inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
        )


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("shape", _REJECTED_INPUT_RANKS)
def test_fbgemm_linear_int8_weight_rejects_input_rank(shape):
    _, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)
    bad_inp = torch.ones(shape, dtype=DTYPE)
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_int8_weight(
            bad_inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
        )


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight_rejects_k_mismatch():
    _, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)
    bad_inp = torch.ones(8, 15, dtype=DTYPE)
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_int8_weight(
            bad_inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
        )


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("weight_shape", _REJECTED_WEIGHT_RANKS)
def test_fbgemm_linear_int8_weight_rejects_weight_rank(weight_shape):
    inp, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)
    bad_weight = torch.ones(weight_shape, dtype=DTYPE)
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_int8_weight(
            inp, bad_weight, packed, col_offsets, weight_scale, weight_zero_point, bias
        )


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("bias_length", _REJECTED_BIAS_LENGTHS)
def test_fbgemm_linear_int8_weight_rejects_bias_length(bias_length):
    inp, weight, _ = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, weight_scale, weight_zero_point = _pack_weights(weight)
    bad_bias = torch.ones(bias_length, dtype=DTYPE)
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_int8_weight(
            inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bad_bias
        )


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("case", _REJECTED_SCALAR_FORMS)
def test_fbgemm_linear_int8_weight_rejects_scalar_form(case):
    inp, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, _, _ = _pack_weights(weight)
    weight_scale, weight_zero_point = _rejected_scalars(case)
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_int8_weight(
            inp, weight, packed, col_offsets, weight_scale, weight_zero_point, bias
        )


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.complex64,
        torch.complex128,
    ],
)
def test_fbgemm_linear_int8_weight_weight_metadata_dtype(dtype):
    inp, weight, bias = _make_linear_operands(_WORKLOAD_SHAPE, _WORKLOAD_N, ["-1", "1"])
    packed, col_offsets, scale, zero_point = _pack_weights(weight)
    weight = weight.to(dtype)
    ref_out = torch.ops.aten.fbgemm_linear_int8_weight(
        tu.to_reference(inp),
        tu.to_reference(weight),
        packed,
        tu.to_reference(col_offsets),
        scale,
        zero_point,
        tu.to_reference(bias),
    )
    res_out = flag_gems.fbgemm_linear_int8_weight(
        inp, weight, packed, col_offsets, scale, zero_point, bias
    )

    tu.assert_result_close(res_out, ref_out)
