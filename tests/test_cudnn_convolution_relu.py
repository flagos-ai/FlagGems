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

# aten::cudnn_convolution_relu(self, weight, bias, stride, padding, dilation,
# groups) is the fused N-D convolution + ReLU; stride / padding / dilation carry
# one entry per spatial dimension. Every parameter is required, so there is no
# optional argument to omit.
#   * the accepted ranks are 4 (2-D convolution) and 5 (3-D convolution), so the
#     spec's 0-3 dim shapes collapse onto those two ranks.
#   * convolution semantics tie the weight to the input only through the rank and
#     the channel relation: the weight has shape
#     (out_channels, in_channels // groups, ...), so the input and kernel spatial
#     extents need not match and no broadcasting is involved there. The only
#     broadcasting the operator performs is on bias, covered as the three
#     native-valid forms (C,), (C, 1, ..) and (1, C, 1, ..) in _PARAM_CASES.
#   * the schema has no scalar operand, so there is no tensor-vs-scalar workload.
#   * no autograd formula is registered for the operator ('derivative for
#     aten::cudnn_convolution_relu is not implemented'), so backward is exempt
#     rather than skipped.
_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)

# Probe-measured capability. The dtype and rank rejections encoded below are a
# cuDNN implementation limit on the vendor whose probes established them, so the
# negative lists are scoped by the measured vendor identity rather than by a
# device string, and each dtype is listed only when the backend can build
# tensors of it (shared constructibility flags). Another vendor routes the
# operator to a different native implementation with different limits.
_IS_MEASURED_VENDOR = flag_gems.runtime.device.vendor_name == "nvidia"

_UNSUPPORTED_DTYPES = [torch.bool, torch.int8, torch.uint8, torch.int32]
if utils.fp64_is_supported:
    _UNSUPPORTED_DTYPES.append(torch.float64)
if utils.int64_is_supported:
    _UNSUPPORTED_DTYPES.append(torch.int64)
if utils.fp8_is_supported:
    _UNSUPPORTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if not _IS_MEASURED_VENDOR:
    _UNSUPPORTED_DTYPES = []

# (input_shape, out_channels, kernel, groups). The rank-4 rows keep the spec's
# size progression up to the prescribed (16, 128, 64, 60) plus 1x1 / 5x5
# kernels, stride > 1, groups 1 and 2 and the depthwise form; the rank-5 rows
# include the prescribed (16, 7, 57, 32, 29) and a depthwise variant.
_CONV_CASES = [
    ((1, 1, 4, 4), 1, 1, 1),
    ((1, 2, 8, 8), 4, 3, 1),
    ((2, 3, 9, 9), 4, 3, 1),
    ((2, 3, 10, 10), 4, 5, 1),
    ((2, 4, 32, 32), 8, 5, 2),
    ((4, 8, 16, 16), 8, 3, 1),
    ((4, 8, 16, 16), 8, 3, 2),
    ((2, 4, 256, 256), 4, 3, 1),
    ((1, 3, 1024, 1024), 4, 3, 1),
    ((16, 128, 64, 60), 8, 3, 1),
    ((1, 2, 5, 5, 5), 2, 3, 1),
    ((2, 3, 6, 6, 6), 4, 3, 1),
    ((4, 4, 8, 8, 8), 6, 3, 2),
    ((2, 6, 12, 12, 12), 6, 5, 3),
    ((1, 3, 7, 7, 7), 3, 1, 3),
    ((16, 7, 57, 32, 29), 8, 3, 1),
]

# The spec's quick shape (2, 19, 7) gains the leading channel axis that rank 4
# requires; kernel size and groups take valid representative values while
# stride / padding / dilation keep their identity values.
_CONV_CASES_QUICK = [((2, 3, 19, 7), 4, 3, 1)]

_CONV_SELECTED = tu.selected_cases(_CONV_CASES, quick=_CONV_CASES_QUICK)

# (input_shape, out_channels, kernel, stride, padding, dilation, groups, bias).
# The first eleven rows sweep the parameters at (4, 8, 32, 32): stride 1 / 2 / 3,
# padding 0 / 1 / 2, dilation 1 / 2 / 3, the combined stride 2 + padding 1,
# groups 1 / 2 / 4 and the optional bias=None, each value exercised with the
# others at their identity setting. The (4, 8, 16, 16) rows add asymmetric
# stride / padding / dilation combinations (including an asymmetric stride with
# groups 4), the three native-valid bias forms and a second bias=None; the last
# two rows are 3-D.
_PARAM_CASES = tu.selected_cases(
    [
        ((4, 8, 32, 32), 8, 3, [1, 1], [0, 0], [1, 1], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [2, 2], [0, 0], [1, 1], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [3, 3], [0, 0], [1, 1], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [1, 1], [1, 1], [1, 1], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [1, 1], [2, 2], [1, 1], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [1, 1], [0, 0], [2, 2], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [1, 1], [0, 0], [3, 3], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [2, 2], [1, 1], [1, 1], 1, "vector"),
        ((4, 8, 32, 32), 8, 3, [1, 1], [0, 0], [1, 1], 1, None),
        ((4, 8, 32, 32), 8, 3, [1, 1], [0, 0], [1, 1], 2, "vector"),
        ((4, 8, 32, 32), 8, 3, [1, 1], [0, 0], [1, 1], 4, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [0, 0], [1, 1], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 2], [0, 1], [1, 1], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [2, 1], [1, 0], [1, 1], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [2, 1], [1, 1], [1, 1], 4, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 2], [1, 2], [1, 2], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [2, 2], [1, 1], [1, 1], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [2, 2], [1, 1], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [1, 1], [2, 2], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [1, 1], [3, 3], 1, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [1, 1], [1, 1], 2, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [1, 1], [1, 1], 4, "vector"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [0, 0], [1, 1], 1, "channel"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [0, 0], [1, 1], 1, "batch"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [0, 0], [1, 1], 4, "channel"),
        ((4, 8, 16, 16), 8, 3, [1, 1], [0, 0], [1, 1], 1, None),
        ((2, 6, 12, 12, 12), 6, 3, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1, "vector"),
        ((2, 4, 9, 9, 9), 4, 3, [2, 1, 1], [1, 0, 1], [1, 2, 1], 2, "vector"),
    ],
    quick=[],
)

_OUT_CASES = tu.selected_cases(
    [((2, 3, 8, 8), 4, 3, 1), ((2, 3, 6, 6, 6), 4, 3, 1)], quick=[]
)

# Each row changes the memory layout of one operand of the same convolution
# (in_channels = 4, out_channels = 4, 3x3 kernel) while keeping every operand
# shape consistent with it - a storage-offset or strided form is built from
# wider storage and viewed to the correct length instead of being shortened.
_LAYOUT_CASES = tu.selected_cases(
    [
        "input_transposed",
        "input_channels_last",
        "input_storage_offset",
        "weight_sliced",
        "weight_storage_offset",
        "bias_storage_offset",
        "bias_strided",
    ],
    quick=[],
)

# One special-value payload is installed in one operand at a time.
_SPECIAL_TARGETS = ("input", "weight", "bias")
_SPECIAL_VALUES = tu.special_value_cases(_DTYPES)
_SPECIAL_CASES = tu.selected_cases(
    [
        (target, dtype, scenario)
        for target in _SPECIAL_TARGETS
        for dtype, scenario in _SPECIAL_VALUES
    ],
    quick=[],
)
_SPECIAL_SHAPE = (1, 5, 6, 6)
_SPECIAL_CHANNELS = 5

# A rank-3 operand asks for a 1-D convolution, which the measured vendor's
# convolution descriptor rejects; that limit is vendor-scoped, so the list is
# empty on any other backend.
_BAD_RANK_CASES = (
    [((2, 3, 10), (4, 3, 3), [1], [0], [1], 1)] if _IS_MEASURED_VENDOR else []
)

# Operand-shape mismatches the descriptor rejects: a 1-D weight against a 2-D
# input and an input-channel / weight-channel disagreement. A wrong-length bias
# is deliberately absent: probes show that after any valid call in the same
# process cuDNN reuses a cached execution plan and then accepts it silently, so
# it is order-dependent instead of a stable invalid input.
_MISMATCHED_CASES = [
    ((2, 3, 8, 8), (4, 3, 3), [1, 1], [0, 0], [1, 1], 1),
    ((2, 3, 8, 8), (4, 5, 3, 3), [1, 1], [0, 0], [1, 1], 1),
]

# Parameter values cuDNN rejects through the convolution descriptor or the
# output-shape computation. A stride of 0 aborts the process inside cuDNN and a
# groups value that does not divide the channels is accepted natively, so
# neither is a negative case here. The argument lists are data, so the only
# statement inside pytest.raises is the candidate call itself.
_INVALID_PARAM_CASES = [
    ((2, 6, 16, 16), (4, 6, 3, 3), [-1, 1], [1, 1], [1, 1], 1),
    ((2, 6, 16, 16), (4, 6, 3, 3), [1, 1], [1, 1], [0, 1], 1),
    ((2, 6, 16, 16), (4, 6, 3, 3), [1, 1], [-1, 1], [1, 1], 1),
]


def _make_bias(dtype, out_channels, spatial, bias_form, value_range):
    if bias_form is None:
        return None
    if bias_form == "vector":
        shape = (out_channels,)
    elif bias_form == "channel":
        shape = (out_channels,) + (1,) * spatial
    else:
        shape = (1, out_channels) + (1,) * spatial
    return tu.make_input(dtype, shape, value_range)


def _make_operands(
    dtype,
    shape,
    out_channels,
    kernel,
    n_groups,
    value_range,
    *,
    stride=None,
    padding=None,
    dilation=None,
    bias_form="vector",
):
    # (input, weight, bias, stride, padding, dilation, groups) on flag_gems.device.
    # The value range applies to every operand, bias included.
    spatial = len(shape) - 2
    inp = tu.make_input(dtype, shape, value_range)
    weight = tu.make_input(
        dtype,
        (out_channels, shape[1] // n_groups) + (kernel,) * spatial,
        value_range,
    )
    bias = _make_bias(dtype, out_channels, spatial, bias_form, value_range)
    return (
        inp,
        weight,
        bias,
        [1] * spatial if stride is None else stride,
        [0] * spatial if padding is None else padding,
        [1] * spatial if dilation is None else dilation,
        n_groups,
    )


def _conv_output_shape(shape, out_channels, kernel, stride, padding, dilation):
    # Allocation metadata only; the reference value still comes from the native op.
    spatial = [
        (size + 2 * pad - dil * (kernel - 1) - 1) // st + 1
        for size, st, pad, dil in zip(shape[2:], stride, padding, dilation)
    ]
    return (shape[0], out_channels, *spatial)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("case", _CONV_SELECTED)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cudnn_convolution_relu_value_ranges(case, value_range, dtype):
    shape, out_channels, kernel, n_groups = case
    inp, weight, bias, stride, padding, dilation, groups = _make_operands(
        dtype, shape, out_channels, kernel, n_groups, value_range
    )
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.cudnn_convolution_relu(
        ref_inp, ref_weight, ref_bias, stride, padding, dilation, groups
    )
    res_out = flag_gems.cudnn_convolution_relu(
        inp, weight, bias, stride, padding, dilation, groups
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("case", _PARAM_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cudnn_convolution_relu_params(case, dtype):
    shape, out_channels, kernel, stride, padding, dilation, n_groups, bias_form = case
    inp, weight, bias, stride, padding, dilation, groups = _make_operands(
        dtype,
        shape,
        out_channels,
        kernel,
        n_groups,
        ["-1", "1"],
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias_form=bias_form,
    )
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.cudnn_convolution_relu(
        ref_inp, ref_weight, ref_bias, stride, padding, dilation, groups
    )
    res_out = flag_gems.cudnn_convolution_relu(
        inp, weight, bias, stride, padding, dilation, groups
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("case", _OUT_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cudnn_convolution_relu_out(case, dtype):
    shape, out_channels, kernel, n_groups = case
    inp, weight, bias, stride, padding, dilation, groups = _make_operands(
        dtype, shape, out_channels, kernel, n_groups, ["-1", "1"]
    )
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)

    out_shape = _conv_output_shape(
        shape, out_channels, kernel, stride, padding, dilation
    )
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_ret = torch.ops.aten.cudnn_convolution_relu.out(
        ref_inp, ref_weight, ref_bias, stride, padding, dilation, groups, out=ref_out
    )

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    res_ret = flag_gems.cudnn_convolution_relu(
        inp, weight, bias, stride, padding, dilation, groups, out=out
    )

    assert res_ret is out
    tu.assert_result_close(res_ret, ref_ret)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("layout", _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cudnn_convolution_relu_operand_layouts(layout, dtype):
    inp = tu.make_input(dtype, (2, 4, 10, 10), ["-1", "1"])
    weight = tu.make_input(dtype, (4, 4, 3, 3), ["-1", "1"])
    bias = tu.make_input(dtype, (4,), ["-1", "1"])
    if layout == "input_transposed":
        inp = inp.transpose(2, 3)
    elif layout == "input_channels_last":
        inp = inp.to(memory_format=torch.channels_last)
    elif layout == "input_storage_offset":
        inp = tu.make_input(dtype, (3, 4, 10, 10), ["-1", "1"])[1:]
    elif layout == "weight_sliced":
        weight = tu.make_input(dtype, (4, 4, 4, 4), ["-1", "1"])[:, :, :3, :3]
    elif layout == "weight_storage_offset":
        weight = tu.make_input(dtype, (5, 4, 3, 3), ["-1", "1"])[1:]
    elif layout == "bias_storage_offset":
        bias = tu.make_input(dtype, (5,), ["-1", "1"])[1:]
    else:
        # Wider storage viewed down to the correct bias length, so the row tests
        # a non-unit stride instead of a shortened bias.
        bias = tu.make_input(dtype, (8,), ["-1", "1"])[::2]

    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)
    stride, padding, dilation, groups = [1, 1], [0, 0], [1, 1], 1

    ref_out = torch.ops.aten.cudnn_convolution_relu(
        ref_inp, ref_weight, ref_bias, stride, padding, dilation, groups
    )
    res_out = flag_gems.cudnn_convolution_relu(
        inp, weight, bias, stride, padding, dilation, groups
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("target,dtype,scenario", _SPECIAL_CASES)
def test_cudnn_convolution_relu_special_values(target, dtype, scenario):
    # The payload is expanded so that it stays channel-constant: with
    # groups == channels the depthwise layout maps payload entry c to output
    # channel c, so every payload value - NaN, the Inf entries and the finite
    # ones - stays observable instead of any of them hiding another. The other
    # operands are finite positive constants that cannot cancel the payload.
    payload = tu.make_special_input(dtype, scenario)
    inp = torch.ones(_SPECIAL_SHAPE, dtype=dtype, device=flag_gems.device)
    weight = torch.full(
        (_SPECIAL_CHANNELS, 1, 3, 3), 0.25, dtype=dtype, device=flag_gems.device
    )
    bias = torch.zeros(_SPECIAL_CHANNELS, dtype=dtype, device=flag_gems.device)
    if target == "input":
        inp = payload.view(1, _SPECIAL_CHANNELS, 1, 1).expand(_SPECIAL_SHAPE)
        inp = inp.contiguous()
    elif target == "weight":
        weight = payload.view(_SPECIAL_CHANNELS, 1, 1, 1).expand(
            _SPECIAL_CHANNELS, 1, 3, 3
        )
        weight = weight.contiguous()
    else:
        bias = payload

    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)
    stride, padding, dilation, groups = [1, 1], [1, 1], [1, 1], _SPECIAL_CHANNELS

    ref_out = torch.ops.aten.cudnn_convolution_relu(
        ref_inp, ref_weight, ref_bias, stride, padding, dilation, groups
    )
    res_out = flag_gems.cudnn_convolution_relu(
        inp, weight, bias, stride, padding, dilation, groups
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_cudnn_convolution_relu_rejects_unsupported_dtype(dtype):
    inp = tu.make_input(dtype, (2, 3, 8, 8), ["-1", "1"])
    weight = tu.make_input(dtype, (4, 3, 3, 3), ["-1", "1"])
    bias = tu.make_input(dtype, (4,), ["-1", "1"])
    stride, padding, dilation, groups = [1, 1], [0, 0], [1, 1], 1

    with pytest.raises(RuntimeError):
        flag_gems.cudnn_convolution_relu(
            inp, weight, bias, stride, padding, dilation, groups
        )


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize(
    "shape,weight_shape,stride,padding,dilation,groups", _BAD_RANK_CASES
)
def test_cudnn_convolution_relu_rejects_bad_input_rank(
    shape, weight_shape, stride, padding, dilation, groups
):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    weight = tu.make_input(torch.float32, weight_shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.cudnn_convolution_relu(
            inp, weight, None, stride, padding, dilation, groups
        )


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize(
    "shape,weight_shape,stride,padding,dilation,groups", _MISMATCHED_CASES
)
def test_cudnn_convolution_relu_rejects_mismatched_operands(
    shape, weight_shape, stride, padding, dilation, groups
):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    weight = tu.make_input(torch.float32, weight_shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.cudnn_convolution_relu(
            inp, weight, None, stride, padding, dilation, groups
        )


@pytest.mark.cudnn_convolution_relu
@pytest.mark.parametrize(
    "shape,weight_shape,stride,padding,dilation,groups", _INVALID_PARAM_CASES
)
def test_cudnn_convolution_relu_rejects_invalid_params(
    shape, weight_shape, stride, padding, dilation, groups
):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    weight = tu.make_input(torch.float32, weight_shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.cudnn_convolution_relu(
            inp, weight, None, stride, padding, dilation, groups
        )
