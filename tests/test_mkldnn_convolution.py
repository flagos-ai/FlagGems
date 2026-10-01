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

# aten::mkldnn_convolution is CPU-only (oneDNN), so every operand stays a CPU
# tensor: the reference and the injected candidate receive the exact same CPU
# arguments. Native dtype probes on this host: float32, float16, bfloat16, int8
# and uint8 run; each dtype in _UNSUPPORTED_DTYPES raises
# "itensor_view_from_dense expects float/bfloat16/half/int8 tensor input".
_SUPPORTED_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
]
_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_UNSUPPORTED_DTYPES = [
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
]
_UNIT_RANGE = ["-1", "1"]


def _base_input(dtype, shape, value_range):
    """CPU operand of the requested extents and value range.

    tu.make_input builds on flag_gems.device; this operator is CPU-only, so the
    generated values are moved to CPU and both call paths use them unchanged.
    """
    return tu.make_input(dtype, shape, value_range).to("cpu")


def _bias(dtype, channels, value_range):
    # oneDNN rejects an int8/uint8 bias descriptor; quantized inputs take a
    # float32 bias.
    bias_dtype = dtype if dtype.is_floating_point else torch.float32
    return _base_input(bias_dtype, (channels,), value_range)


def _operand(dtype, shape, value_range, layout):
    """Operand in the requested memory layout.

    oneDNN reorders non-contiguous, shifted-storage and channels-last CPU
    operands internally, so these produce real layouts instead of a copy.
    """
    if layout == "channels-last":
        return _base_input(dtype, shape, value_range).contiguous(
            memory_format=torch.channels_last
        )
    if layout == "input-slice":
        base = (shape[0], shape[1], shape[2] + 2, shape[3] + 2)
        return _base_input(dtype, base, value_range)[:, :, : shape[2], : shape[3]]
    if layout == "input-offset":
        base = (shape[0], shape[1], shape[2] + 2, shape[3] + 2)
        return _base_input(dtype, base, value_range)[
            :, :, 1 : shape[2] + 1, 1 : shape[3] + 1
        ]
    if layout == "input-transposed":
        base = (shape[0], shape[1], shape[3], shape[2])
        return _base_input(dtype, base, value_range).transpose(2, 3)
    if layout == "weight-strided":
        return _base_input(dtype, (shape[0] * 2,) + tuple(shape[1:]), value_range)[::2]
    if layout == "weight-offset":
        return _base_input(dtype, (shape[0] + 1,) + tuple(shape[1:]), value_range)[1:]
    return _base_input(dtype, shape, value_range)


def _out_shape(input_shape, weight_shape, padding, stride, dilation):
    return (input_shape[0], weight_shape[0]) + tuple(
        (
            input_shape[i + 2]
            + 2 * padding[i]
            - dilation[i] * (weight_shape[i + 2] - 1)
            - 1
        )
        // stride[i]
        + 1
        for i in range(len(input_shape) - 2)
    )


def _conv_args(
    input_shape,
    weight_shape,
    padding,
    stride,
    dilation,
    groups,
    dtype=torch.float32,
    bias_shape=None,
    bias_dtype=None,
    use_bias=True,
):
    """Exact native positional arguments for one convolution call."""
    if bias_shape is None:
        bias_shape = (weight_shape[0],)
    if bias_dtype is None:
        bias_dtype = dtype if dtype.is_floating_point else torch.float32
    inp = _base_input(dtype, input_shape, _UNIT_RANGE)
    weight = _base_input(dtype, weight_shape, _UNIT_RANGE)
    bias = _base_input(bias_dtype, bias_shape, _UNIT_RANGE) if use_bias else None
    return (inp, weight, bias, list(padding), list(stride), list(dilation), groups)


def _unsupported_operand(shape, dtype):
    # Any real operand of that dtype reaches the same native dtype check; the
    # shared generator does not build bool/complex/fp8 tensors.
    return torch.randn(shape, device="cpu").to(dtype)


# tu.selected_shapes() holds rank 0-2 and rank 6 shapes, which the native kernel
# rejects, so these rows carry the same spec extents at the ranks it accepts:
# 3 (NCL), 4 (NCHW) and 5 (NCDHW), plus cheap non-contiguous / shifted-storage
# operands. The last field is the memory layout of both operands.
_SINGLETON_ROW = (
    (1, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0),
    (1, 1),
    (1, 1),
    1,
    "contiguous",
)
_QUICK_RANK3_ROW = ((2, 19, 7), (3, 19, 3), (1,), (1,), (1,), 1, "contiguous")
_CHANNELS_LAST_ROW = (
    (2, 3, 8, 8),
    (4, 3, 3, 3),
    (1, 1),
    (1, 1),
    (1, 1),
    1,
    "channels-last",
)

_SHAPE_ROWS_ALL = [
    ((1, 1, 1024, 1024), (1, 1, 3, 3), (0, 0), (1, 1), (1, 1), 1, "contiguous"),
    ((20, 1, 320, 15), (2, 1, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((20, 320, 15), (4, 320, 3), (1,), (1,), (1,), 1, "contiguous"),
    (
        (16, 7, 57, 32, 29),
        (4, 7, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
        "contiguous",
    ),
    _SINGLETON_ROW,
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((1, 1, 1024, 1024), (1, 1, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((20, 320, 15), (16, 320, 3), (1,), (1,), (1,), 1, "contiguous"),
    ((16, 128, 64, 60), (32, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    ((7, 57, 32, 29), (16, 57, 3, 3), (1, 1), (1, 1), (1, 1), 1, "contiguous"),
    _QUICK_RANK3_ROW,
    (
        (4, 7, 15, 29, 32),
        (8, 7, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
        "contiguous",
    ),
    _CHANNELS_LAST_ROW,
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "input-slice"),
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "input-offset"),
    ((2, 3, 10, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "input-transposed"),
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "weight-strided"),
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, "weight-offset"),
]
_SHAPE_ROWS = tu.selected_cases(
    _SHAPE_ROWS_ALL,
    quick=[_SINGLETON_ROW, _QUICK_RANK3_ROW, _CHANNELS_LAST_ROW] + _SHAPE_ROWS_ALL[-5:],
)
_SHAPE_IDS = ["-".join(map(str, row[0])) + ":" + row[6] for row in _SHAPE_ROWS]


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("row", _SHAPE_ROWS, ids=_SHAPE_IDS)
def test_mkldnn_convolution(row, value_range, dtype):
    input_shape, weight_shape, padding, stride, dilation, groups, layout = row
    inp = _operand(dtype, input_shape, value_range, layout)
    weight = _operand(dtype, weight_shape, value_range, layout)
    bias = _bias(dtype, weight_shape[0], value_range)
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.mkldnn_convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        list(padding),
        list(stride),
        list(dilation),
        groups,
    )
    res_out = flag_gems.mkldnn_convolution(
        inp, weight, bias, list(padding), list(stride), list(dilation), groups
    )

    # oneDNN keeps the operand layout: a channels-last input yields a
    # channels-last-strided output, contiguous operands yield a contiguous
    # output. Check the layout before comparing values, since a dense value
    # comparison alone would hide a layout mismatch.
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(weight, ref_weight)


# Parameter rows: small, legal oneDNN calls on the (20, 320, 15) spec shape used
# as a rank-3 (N, C, L) input, covering padding/stride/dilation/kernel/groups and
# the bias-absent form. All of them are cheap, so quick keeps every branch.
_PARAM_ROWS_ALL = [
    ("asymmetric-2d", (1, 4, 128, 60), (4, 4, 3, 3), (1, 2), (2, 3), (1, 1), 1, True),
    ("depthwise-2d", (1, 4, 128, 60), (4, 1, 3, 3), (1, 1), (1, 1), (1, 1), 4, True),
    ("base-params", (20, 320, 15), (16, 320, 3), (1,), (1,), (1,), 1, True),
    ("padding-zero", (20, 320, 15), (16, 320, 3), (0,), (1,), (1,), 1, True),
    ("padding-two", (20, 320, 15), (16, 320, 3), (2,), (1,), (1,), 1, True),
    ("stride-two", (20, 320, 15), (16, 320, 3), (1,), (2,), (1,), 1, True),
    (
        "padding-zero-stride-three",
        (20, 320, 15),
        (16, 320, 3),
        (0,),
        (3,),
        (1,),
        1,
        True,
    ),
    ("dilation-two", (20, 320, 15), (16, 320, 3), (1,), (1,), (2,), 1, True),
    ("kernel-one", (20, 320, 15), (16, 320, 1), (0,), (1,), (1,), 1, True),
    ("groups-two", (20, 320, 15), (16, 160, 3), (1,), (1,), (1,), 2, True),
    ("groups-eight", (20, 320, 15), (16, 40, 3), (1,), (1,), (1,), 8, True),
    ("bias-none", (20, 320, 15), (16, 320, 3), (1,), (1,), (1,), 1, False),
]
_PARAM_ROWS = tu.selected_cases(_PARAM_ROWS_ALL, quick=_PARAM_ROWS_ALL)


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("row", _PARAM_ROWS, ids=[row[0] for row in _PARAM_ROWS])
def test_mkldnn_convolution_with_params(row, dtype):
    _, input_shape, weight_shape, padding, stride, dilation, groups, use_bias = row
    inp = _base_input(dtype, input_shape, _UNIT_RANGE)
    weight = _base_input(dtype, weight_shape, _UNIT_RANGE)
    bias = _bias(dtype, weight_shape[0], _UNIT_RANGE) if use_bias else None
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = None if bias is None else tu.to_reference(bias)

    ref_out = torch.ops.aten.mkldnn_convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        list(padding),
        list(stride),
        list(dilation),
        groups,
    )
    res_out = flag_gems.mkldnn_convolution(
        inp, weight, bias, list(padding), list(stride), list(dilation), groups
    )

    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_close(res_out, ref_out)


# The out overload is really callable natively for these CPU forms: it returns
# the passed buffer itself and resizes an empty or undersized buffer.
_OUT_ROWS_ALL = ["exact-buffer", "empty-buffer", "undersized-buffer"]
_OUT_ROWS = tu.selected_cases(_OUT_ROWS_ALL, quick=_OUT_ROWS_ALL)


def _out_buffer(out_kind, dtype, expected_shape):
    if out_kind == "exact-buffer":
        return torch.empty(expected_shape, dtype=dtype, device="cpu")
    if out_kind == "empty-buffer":
        return torch.empty(0, dtype=dtype, device="cpu")
    return torch.empty(
        (expected_shape[0], expected_shape[1], 1, 1), dtype=dtype, device="cpu"
    )


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("out_kind", _OUT_ROWS)
def test_mkldnn_convolution_out(out_kind, dtype):
    input_shape, weight_shape, padding = (2, 3, 8, 8), (4, 3, 3, 3), (1, 1)
    stride, dilation = (1, 1), (1, 1)
    expected = _out_shape(input_shape, weight_shape, padding, stride, dilation)
    inp = _base_input(dtype, input_shape, _UNIT_RANGE)
    weight = _base_input(dtype, weight_shape, _UNIT_RANGE)
    bias = _bias(dtype, weight_shape[0], _UNIT_RANGE)

    ref_buf = _out_buffer(out_kind, dtype, expected)
    ref_out = torch.ops.aten.mkldnn_convolution.out(
        tu.to_reference(inp),
        tu.to_reference(weight),
        tu.to_reference(bias),
        list(padding),
        list(stride),
        list(dilation),
        1,
        out=ref_buf,
    )
    out_buf = _out_buffer(out_kind, dtype, expected)
    res_out = flag_gems.mkldnn_convolution(
        inp,
        weight,
        bias,
        list(padding),
        list(stride),
        list(dilation),
        1,
        out=out_buf,
    )

    # Return identity means the very same object, not just equal contents.
    assert res_out is out_buf
    assert res_out.shape == expected
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_close(res_out, ref_out)


_BACKWARD_ROWS_ALL = [
    ((2, 3, 8, 8), (4, 3, 3, 3), (2, 2), (1, 1), (2, 2), 1),
    ((2, 4, 8, 8), (4, 2, 3, 3), (1, 1), (1, 1), (1, 1), 2),
    ((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((2, 3, 19), (4, 3, 3), (1,), (1,), (1,), 1),
    ((2, 3, 8, 6, 4), (4, 3, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1),
]
# Backward is default-only: no backward case, not even a smoke one, is a quick
# workload. int8/uint8 inputs are excluded because integer tensors cannot carry
# gradients at all (probed: requires_grad_ on int8 raises).
_BACKWARD_ROWS = tu.selected_cases(_BACKWARD_ROWS_ALL, quick=[])


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
@pytest.mark.parametrize(
    "row", _BACKWARD_ROWS, ids=[f"rank{len(row[0])}-conv" for row in _BACKWARD_ROWS]
)
def test_mkldnn_convolution_backward(row, dtype):
    input_shape, weight_shape, padding, stride, dilation, groups = row
    inp = _base_input(dtype, input_shape, _UNIT_RANGE).requires_grad_(True)
    weight = _base_input(dtype, weight_shape, _UNIT_RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    ref_weight = tu.to_reference(weight.detach()).requires_grad_(True)
    bias = _bias(dtype, weight_shape[0], _UNIT_RANGE).requires_grad_(True)
    ref_bias = tu.to_reference(bias)
    args = (list(padding), list(stride), list(dilation), groups)

    ref_out = torch.ops.aten.mkldnn_convolution(ref_inp, ref_weight, ref_bias, *args)
    res_out = flag_gems.mkldnn_convolution(inp, weight, bias, *args)

    # The forward values are compared first: gradients are only meaningful when
    # the values they differentiate already agree.
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_close(res_out, ref_out)

    upstream = tu.make_input(dtype, tuple(ref_out.shape), _UNIT_RANGE).to("cpu")
    res_grad_inp, res_grad_weight, res_grad_bias = torch.autograd.grad(
        res_out, (inp, weight, bias), grad_outputs=upstream
    )
    ref_grad_inp, ref_grad_weight, ref_grad_bias = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), grad_outputs=upstream
    )
    tu.assert_result_close(res_grad_inp, ref_grad_inp)
    tu.assert_result_close(res_grad_weight, ref_grad_weight)
    tu.assert_result_close(res_grad_bias, ref_grad_bias)


# Positive special values are default-only. A 1x1 kernel with unit weight and
# zero bias replicates each input element into the two output channels, so the
# nan/inf payloads reach the output unchanged.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_mkldnn_convolution_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario).to("cpu")
    inp = payload.reshape(1, 1, 5, 1).expand(1, 1, 5, 4).contiguous()
    weight = torch.ones((2, 1, 1, 1), dtype=dtype, device="cpu")
    bias = torch.zeros((2,), dtype=dtype, device="cpu")
    ref_out = torch.ops.aten.mkldnn_convolution(
        tu.to_reference(inp),
        tu.to_reference(weight),
        tu.to_reference(bias),
        [0, 0],
        [1, 1],
        [1, 1],
        1,
    )
    res_out = flag_gems.mkldnn_convolution(inp, weight, bias, [0, 0], [1, 1], [1, 1], 1)
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_close(res_out, ref_out)


# Every row overrides one field of an otherwise valid call. All of them are kept
# in quick as well as in default mode.
#
# A 1-element padding array is deliberately not used as a negative row: the
# convolution argument handling broadcasts a single padding value across all
# spatial dims, so [1] is a legal rank-4 call that both the native reference and
# the candidate accept and execute.
_NEGATIVE_ROWS_ALL = [
    ("rank2-input", {"input_shape": (2, 3)}),
    ("rank1-weight", {"weight_shape": (4,)}),
    ("rank6-input", {"input_shape": (1, 1, 2, 3, 8, 8)}),
    ("input-channel-mismatch", {"weight_shape": (4, 5, 3, 3)}),
    ("bias-size-mismatch", {"bias_shape": (3,)}),
    ("groups-mismatch", {"groups": 2}),
    ("channels-not-divisible-by-groups", {"groups": 2}),
    ("groups-zero", {"groups": 0}),
    ("groups-negative", {"groups": -1}),
    ("zero-stride", {"stride": (0, 1)}),
    ("zero-dilation", {"dilation": (0, 1)}),
    ("negative-padding", {"padding": (-1, 1)}),
    ("kernel-larger-than-input", {"weight_shape": (4, 3, 11, 11)}),
    ("int8-bias", {"dtype": torch.int8, "bias_dtype": torch.int8}),
    ("uint8-bias", {"dtype": torch.uint8, "bias_dtype": torch.uint8}),
    ("dilation-kernel-out", {"dilation": (5, 5)}),
]
_NEGATIVE_ROWS = tu.selected_cases(_NEGATIVE_ROWS_ALL, quick=_NEGATIVE_ROWS_ALL)
_NEGATIVE_DEFAULTS = {
    "input_shape": (2, 3, 8, 8),
    "weight_shape": (4, 3, 3, 3),
    "padding": (1, 1),
    "stride": (1, 1),
    "dilation": (1, 1),
    "groups": 1,
    "dtype": torch.float32,
    "bias_shape": None,
    "bias_dtype": None,
}


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize("row", _NEGATIVE_ROWS, ids=[row[0] for row in _NEGATIVE_ROWS])
def test_mkldnn_convolution_rejects_invalid_args(row):
    params = dict(_NEGATIVE_DEFAULTS)
    params.update(row[1])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_convolution(*_conv_args(**params))


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize(
    "dtype", _UNSUPPORTED_DTYPES, ids=[str(d) for d in _UNSUPPORTED_DTYPES]
)
def test_mkldnn_convolution_rejects_unsupported_dtype(dtype):
    inp = _unsupported_operand((2, 3, 8, 8), dtype)
    weight = _unsupported_operand((4, 3, 3, 3), dtype)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_convolution(inp, weight, None, [1, 1], [1, 1], [1, 1], 1)


# Native out requires the buffer dtype to equal the input dtype.
_OUT_DTYPE_ROWS = [
    (torch.float32, torch.float64),
    (torch.float16, torch.float32),
    (torch.int8, torch.uint8),
    (torch.int8, torch.float32),
]


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    _OUT_DTYPE_ROWS,
    ids=[f"{i}-to-{o}" for i, o in _OUT_DTYPE_ROWS],
)
def test_mkldnn_convolution_rejects_out_dtype_mismatch(in_dtype, out_dtype):
    inp = _base_input(in_dtype, (2, 3, 8, 8), _UNIT_RANGE)
    weight = _base_input(in_dtype, (4, 3, 3, 3), _UNIT_RANGE)
    out_buf = torch.empty((2, 4, 8, 8), dtype=out_dtype, device="cpu")
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_convolution(
            inp,
            weight,
            _bias(in_dtype, 4, _UNIT_RANGE),
            [1, 1],
            [1, 1],
            [1, 1],
            1,
            out=out_buf,
        )


@pytest.mark.mkldnn_convolution
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("packed_weight", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
def test_mkldnn_convolution_mkldnn_layout(dtype, packed_weight, use_bias):
    dense = _base_input(dtype, (2, 3, 8, 8), _UNIT_RANGE)
    weight = _base_input(dtype, (4, 3, 3, 3), _UNIT_RANGE)
    ref_weight = tu.to_reference(weight)
    if packed_weight:
        weight, ref_weight = weight.to_mkldnn(), ref_weight.to_mkldnn()
    bias = _bias(dtype, 4, _UNIT_RANGE) if use_bias else None
    ref_bias = tu.to_reference(bias) if use_bias else None
    inp, ref_inp = dense.to_mkldnn(), tu.to_reference(dense).to_mkldnn()

    ref_out = torch.ops.aten.mkldnn_convolution(
        ref_inp, ref_weight, ref_bias, [1, 1], [1, 1], [1, 1], 1
    )
    res_out = flag_gems.mkldnn_convolution(inp, weight, bias, [1, 1], [1, 1], [1, 1], 1)

    assert res_out.layout == torch._mkldnn
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())
    tu.assert_result_equal(inp.to_dense(), ref_inp.to_dense())
    tu.assert_result_equal(weight.to_dense(), ref_weight.to_dense())
