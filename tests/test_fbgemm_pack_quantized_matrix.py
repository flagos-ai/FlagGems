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

"""Correctness tests for ``aten::fbgemm_pack_quantized_matrix``.

CPU-only FBGEMM int8 host packer: rank >= 2, int8, returning a fixed-size opaque
``uint8`` descriptor that holds library heap state. Two independently allocated
handles are therefore never compared byte-wise; the packed payload is validated
through the genuine native consumer chain, each side packing its own weight.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Argument error paths: ``None`` is read by the native parser as a missing
# dimension (IndexError), everything else fails the schema check.
_REJECTED = (RuntimeError, TypeError, IndexError)

# int8 is the only dtype the native CPU kernel accepts.
_REJECT_DTYPES = [
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.bool,
]

# The packer reads dimensions 0 and 1, so rank < 2 is native-invalid and
# the shared rank-0/1 spec shapes are negative rows only.
_RANK_ROWS = [(), (1,), (256,)]
_NON_TENSORS = [3, 1.5, "int8", [1, 2, 3], (2, 2), None]
_BAD_KN = [(1.5, 2), (2, 1.5), ("1", 1), (2, None)]
_SPECIAL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _cpu_input(dtype, shape, value_range):
    """Shared value-range input, placed on CPU for this CPU-only operator."""
    return tu.make_input(dtype, shape, value_range).cpu()


def _layout_view(src, layout):
    """Value-preserving view of ``src`` with the requested memory layout.

    Layouts are only requested for rank-2 rows, so the packed shape stays the
    case shape while stride and storage offset differ.
    """
    if layout == "contiguous":
        return src
    rows, cols = src.shape
    if layout == "transposed":
        buffer = torch.empty(cols, rows, dtype=src.dtype)
        buffer.copy_(src.t())
        return buffer.t()
    if layout == "stepped":
        buffer = torch.empty(rows, 2 * cols + 1, dtype=src.dtype)
        view = buffer[:, ::2].narrow(1, 0, cols)
        view.copy_(src)
        return view
    if layout == "offset":
        buffer = torch.empty(rows + 3, cols + 5, dtype=src.dtype)
        view = buffer.narrow(0, 2, rows).narrow(1, 3, cols)
        view.copy_(src)
        return view
    raise AssertionError(f"unknown layout {layout!r}")


def _pack_operands(shape, layout, value_range):
    """Independent candidate / reference operands with equal values and layout."""
    inp = _cpu_input(torch.int8, shape, value_range)
    return _layout_view(inp, layout), _layout_view(inp.clone(), layout)


def _assert_blob_contract(res_out, ref_out, inp):
    """Descriptor properties a candidate must reproduce.

    The bytes are heap state and are not compared across independent
    allocations; the packed content is checked by the native consumer rows.
    """
    assert res_out.dtype == ref_out.dtype == torch.uint8
    assert res_out.dim() == ref_out.dim() == 1
    assert res_out.shape == ref_out.shape
    assert res_out.numel() > 0
    assert res_out.stride() == ref_out.stride()
    assert res_out.device == inp.device


def _quantized_weight(rows, cols, value_range):
    """Native FBGEMM quantizer: int8 weight plus its consumer scalars."""
    weight = _cpu_input(torch.float32, (rows, cols), value_range)
    (
        quantized,
        col_offsets,
        scale,
        zero_point,
    ) = torch.ops.aten.fbgemm_linear_quantize_weight(weight)
    return quantized, col_offsets, scale, zero_point


def _linear_output(quantized, packed, col_offsets, scale, zero_point, activation, bias):
    return torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        activation, quantized, packed, col_offsets, scale, zero_point, bias
    )


def _linear_operands(shape, layout, batch, value_range):
    """Quantized weight views plus the consumer's activation/bias scaffolding."""
    rows, cols = shape
    quantized, col_offsets, scale, zero_point = _quantized_weight(
        rows, cols, value_range
    )
    activation = _cpu_input(torch.float32, (batch, cols), ["-1", "1"])
    bias = _cpu_input(torch.float32, (rows,), ["-1", "1"])
    return (
        _layout_view(quantized, layout),
        _layout_view(quantized.clone(), layout),
        col_offsets,
        scale,
        zero_point,
        activation,
        bias,
    )


# Native-valid rank >= 2 workloads: the shared spec shapes plus every extra
# shape and layout carried over from the previous revision.
_PACK_ROWS = [
    ((1024, 1024), "contiguous"),
    ((20, 320, 15), "contiguous"),
    ((16, 128, 64, 60), "contiguous"),
    ((16, 7, 57, 32, 29), "contiguous"),
    ((2, 19, 7), "contiguous"),
    ((1, 1), "contiguous"),
    ((1, 1, 1, 1), "contiguous"),
    ((2, 3, 5), "contiguous"),
    ((4, 4), "contiguous"),
    ((8, 16), "contiguous"),
    ((33, 17), "contiguous"),
    ((256, 256), "contiguous"),
    ((0, 4), "contiguous"),
    ((4, 0), "contiguous"),
    ((4, 16), "transposed"),
    ((8, 16), "stepped"),
    ((33, 17), "offset"),
]

_LARGE_PACK_SHAPES = {
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
}

# Quick only trims the large extents: every layout, empty and singleton row stays.
PACK_ROWS = tu.selected_cases(
    _PACK_ROWS,
    quick=[row for row in _PACK_ROWS if row[0] not in _LARGE_PACK_SHAPES],
)

_KN_SHAPES = [(4, 4), (8, 16), (33, 17)]

# Boundaries accepted by the native ``.KN`` overload; the descriptor carries the
# input's own dims, so these are exercised as arguments, not asserted back.
_KN_VALUES = [(0, 0), (1, 1), (-1, -1), (256, 256), (2, 19)]

# (weight shape, layout, consumer batch): the activation's last dim must equal
# the weight's column count and the bias length its row count.
_CONSUMER_ROWS = [
    ((8, 16), "contiguous", 4),
    ((4, 8), "contiguous", 16),
    ((16, 32), "contiguous", 8),
    ((4, 4), "contiguous", 5),
    ((8, 16), "transposed", 5),
    ((8, 16), "stepped", 5),
    ((8, 16), "offset", 5),
    ((33, 17), "contiguous", 5),
    ((64, 128), "contiguous", 5),
]

CONSUMER_ROWS = tu.selected_cases(
    _CONSUMER_ROWS,
    quick=[
        row for row in _CONSUMER_ROWS if row[0] in ((8, 16), (4, 8), (16, 32), (4, 4))
    ],
)

# float payloads are the only way to present NaN/Inf and int8 cannot encode
# them, so this dimension is the native rejection of every float dtype.
SPECIAL_ROWS = tu.special_value_cases(_SPECIAL_DTYPES)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,layout", PACK_ROWS)
def test_fbgemm_pack_quantized_matrix(shape, layout, value_range):
    inp, ref_inp = _pack_operands(shape, layout, value_range)

    ref_out = torch.ops.aten.fbgemm_pack_quantized_matrix(ref_inp)
    res_out = flag_gems.fbgemm_pack_quantized_matrix(inp)

    _assert_blob_contract(res_out, ref_out, inp)
    tu.assert_result_equal(inp, ref_inp)
    # The native packer reads N=size(0), K=size(1), including for rank>2.
    # Observe that packed prefix through its genuine consumer. Empty handles
    # have no payload and are covered by the descriptor checks above.
    rows, cols = shape[:2]
    if rows and cols:
        weight = ref_inp.contiguous().reshape(-1)[: rows * cols].reshape(rows, cols)
        offsets = weight.sum(dim=1, dtype=torch.int32)
        activation = torch.randn((2, cols), dtype=torch.float32)
        bias = torch.zeros(rows, dtype=torch.float32)
        ref_values = _linear_output(weight, ref_out, offsets, 1.0, 0, activation, bias)
        res_values = _linear_output(weight, res_out, offsets, 1.0, 0, activation, bias)
        tu.assert_result_close(res_values, ref_values)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("kn", _KN_VALUES)
@pytest.mark.parametrize("shape", _KN_SHAPES)
def test_fbgemm_pack_quantized_matrix_kn(shape, kn, value_range):
    inp, ref_inp = _pack_operands(shape, "contiguous", value_range)

    ref_out = torch.ops.aten.fbgemm_pack_quantized_matrix(ref_inp, kn[0], kn[1])
    res_out = flag_gems.fbgemm_pack_quantized_matrix(inp, kn[0], kn[1])

    _assert_blob_contract(res_out, ref_out, inp)
    tu.assert_result_equal(inp, ref_inp)
    # The native packer reads N=size(0), K=size(1), including for rank>2.
    # Observe that packed prefix through its genuine consumer. Empty handles
    # have no payload and are covered by the descriptor checks above.
    rows, cols = shape[:2]
    if rows and cols:
        weight = ref_inp.contiguous().reshape(-1)[: rows * cols].reshape(rows, cols)
        offsets = weight.sum(dim=1, dtype=torch.int32)
        activation = torch.randn((2, cols), dtype=torch.float32)
        bias = torch.zeros(rows, dtype=torch.float32)
        ref_values = _linear_output(weight, ref_out, offsets, 1.0, 0, activation, bias)
        res_values = _linear_output(weight, res_out, offsets, 1.0, 0, activation, bias)
        tu.assert_result_close(res_values, ref_values)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,layout,batch", CONSUMER_ROWS)
def test_fbgemm_pack_quantized_matrix_feeds_native_linear(
    shape, layout, batch, value_range
):
    # The packed payload is heap-internal, so it is observed through the genuine
    # native FBGEMM consumer: both sides pack their own quantized weight and are
    # consumed with identical quantization metadata.
    res_in, ref_in, col_offsets, scale, zero_point, activation, bias = _linear_operands(
        shape, layout, batch, value_range
    )

    ref_out = _linear_output(
        ref_in,
        torch.ops.aten.fbgemm_pack_quantized_matrix(ref_in),
        col_offsets,
        scale,
        zero_point,
        activation,
        bias,
    )
    res_out = _linear_output(
        res_in,
        flag_gems.fbgemm_pack_quantized_matrix(res_in),
        col_offsets,
        scale,
        zero_point,
        activation,
        bias,
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("kn", _KN_VALUES)
@pytest.mark.parametrize("shape,layout,batch", CONSUMER_ROWS)
def test_fbgemm_pack_quantized_matrix_kn_feeds_native_linear(
    shape, layout, batch, kn, value_range
):
    # Same end-to-end payload check for the ``.KN`` call form.
    res_in, ref_in, col_offsets, scale, zero_point, activation, bias = _linear_operands(
        shape, layout, batch, value_range
    )

    ref_out = _linear_output(
        ref_in,
        torch.ops.aten.fbgemm_pack_quantized_matrix(ref_in, kn[0], kn[1]),
        col_offsets,
        scale,
        zero_point,
        activation,
        bias,
    )
    res_out = _linear_output(
        res_in,
        flag_gems.fbgemm_pack_quantized_matrix(res_in, kn[0], kn[1]),
        col_offsets,
        scale,
        zero_point,
        activation,
        bias,
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("dtype", _REJECT_DTYPES)
def test_fbgemm_pack_quantized_matrix_rejects_dtype(dtype):
    inp = _cpu_input(dtype, (4, 4), ["-1", "1"])
    with pytest.raises(_REJECTED):
        flag_gems.fbgemm_pack_quantized_matrix(inp)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("shape", _RANK_ROWS)
def test_fbgemm_pack_quantized_matrix_rejects_low_rank(shape):
    inp = _cpu_input(torch.int8, shape, ["-1", "1"])
    with pytest.raises(_REJECTED):
        flag_gems.fbgemm_pack_quantized_matrix(inp)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("value", _NON_TENSORS)
def test_fbgemm_pack_quantized_matrix_rejects_non_tensor(value):
    with pytest.raises(_REJECTED):
        flag_gems.fbgemm_pack_quantized_matrix(value)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("kn", _BAD_KN)
def test_fbgemm_pack_quantized_matrix_rejects_bad_kn(kn):
    inp = _cpu_input(torch.int8, (4, 4), ["-1", "1"])
    with pytest.raises(_REJECTED):
        flag_gems.fbgemm_pack_quantized_matrix(inp, kn[0], kn[1])


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("dtype,scenario", SPECIAL_ROWS)
def test_fbgemm_pack_quantized_matrix_rejects_special_values(dtype, scenario):
    # Rank is valid here, so the only reason to reject is the float payload.
    inp = tu.make_special_input(dtype, scenario).reshape(1, -1).cpu()
    with pytest.raises(_REJECTED):
        flag_gems.fbgemm_pack_quantized_matrix(inp)
