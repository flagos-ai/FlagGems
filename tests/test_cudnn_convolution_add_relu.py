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

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z,
#     Scalar? alpha, Tensor? bias, SymInt[] stride, SymInt[] padding,
#     SymInt[] dilation, int groups) -> Tensor
#
# ``alpha`` and ``bias`` are Optional without a schema default, so the absent
# value is passed positionally as None.
#
# Constraints that shape this file, all measured on the nvidia descriptor path:
#   * The accepted ranks are 4-D NCHW and 5-D NCDHW. 2-D/3-D ``self``, a 3-D
#     ``weight``, a rank mismatch between ``self`` and ``weight``, and a
#     stride/padding/dilation list whose length disagrees with the input rank
#     all raise RuntimeError with CUDNN_STATUS_BAD_PARAM from descriptor
#     creation, so the spec shape grid keeps its 4-D and 5-D entries and is
#     extended with further geometries of both ranks.
#   * ``bias`` accepts None, (C,), (C,1,1) and (1,C,1,1). Other bias shapes and
#     residual shapes outside the output shape fail with 'GET was unable to find
#     an engine to execute this computation' in one call order and succeed in
#     another for the same shapes, which is cudnn engine selection rather than a
#     stable argument contract, so no case asserts those forms.
#   * ``.out`` resizes a caller buffer whose shape differs from the result, so
#     that is covered as a supported boundary instead of as an invalid argument.
#     The buffer's dtype must match the computation dtype, which is the property
#     the out negative asserts (with float32 operands).
#   * There is no backward case: autograd raises RuntimeError('derivative for
#     aten::cudnn_convolution_add_relu is not implemented').

# The float types the native kernel executes; bf16 follows the capability flag.
SUPPORTED_DTYPES = [torch.float32, torch.float16]
if flag_gems.runtime.device.support_bf16:
    SUPPORTED_DTYPES.append(torch.bfloat16)

# The measured rank and dtype rejections come from the convolution descriptor
# and its dtype map, so they are asserted only for the vendor that produced
# them. The schema-level negatives further down (tensor ``alpha``, ``out``
# dtype) are independent of that and stay unconditional.
_IS_DESCRIPTOR_VENDOR = flag_gems.runtime.device.vendor_name == "nvidia"

_NEGATIVE_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.bool]
if flag_gems.runtime.device.support_int64:
    _NEGATIVE_DTYPES.append(torch.int64)
if flag_gems.runtime.device.support_fp8:
    _NEGATIVE_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if utils.fp64_is_supported:
    _NEGATIVE_DTYPES.append(torch.float64)
NEGATIVE_DTYPES = _NEGATIVE_DTYPES if _IS_DESCRIPTOR_VENDOR else []

RANK_NEGATIVE_CASES = (
    [
        "self_2d",
        "self_3d",
        "weight_3d",
        "weight_rank_mismatch",
        "stride_length_mismatch",
    ]
    if _IS_DESCRIPTOR_VENDOR
    else []
)

# The float32 operands below always run, so each row only needs a buffer dtype
# that this backend can construct and that differs from the computation dtype.
OUT_DTYPE_NEGATIVE_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    OUT_DTYPE_NEGATIVE_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    OUT_DTYPE_NEGATIVE_DTYPES.append(torch.float64)

# One convolution geometry per row:
# (input/self, weight, stride, padding, dilation, groups).
QUICK_CONV_CASES = [
    ((1, 3, 19, 7), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1),
]

CONV_CASES = [
    ((16, 128, 64, 60), (32, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1),  # spec 4-D
    ((16, 128, 64, 60), (16, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((1, 1, 4, 4), (1, 1, 1, 1), (1, 1), (0, 0), (1, 1), 1),
    ((0, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1),  # empty batch
    ((4, 3, 19, 7), (8, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((2, 4, 13, 11), (6, 4, 3, 3), (2, 2), (1, 1), (1, 1), 1),  # stride 2
    ((2, 4, 8, 8), (4, 4, 3, 3), (1, 1), (0, 0), (1, 1), 1),
    ((2, 4, 16, 16), (4, 4, 3, 3), (1, 1), (2, 2), (2, 2), 1),  # dilation 2
    ((2, 4, 10, 8), (4, 2, 3, 2), (2, 1), (1, 2), (1, 2), 2),  # asymmetric kernel
    ((2, 6, 12, 12), (6, 3, 3, 3), (1, 1), (1, 1), (2, 2), 2),
    ((3, 8, 20, 17), (8, 2, 5, 5), (2, 2), (2, 2), (1, 1), 4),  # 5x5, groups 4
    ((2, 3, 7, 5), (5, 3, 1, 3), (1, 1), (0, 0), (1, 1), 1),  # asymmetric kernel
    ((1, 8, 16, 16), (8, 1, 3, 3), (1, 1), (1, 1), (1, 1), 8),  # depthwise
    ((2, 16, 64, 60), (16, 16, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((1, 256, 16, 16), (64, 256, 1, 1), (1, 1), (0, 0), (1, 1), 1),  # 1x1, 256 ch
    ((2, 4, 32, 24), (8, 4, 3, 5), (2, 3), (1, 0), (2, 1), 1),  # asymmetric params
    ((1, 2, 1024, 64), (4, 2, 3, 3), (1, 1), (1, 1), (1, 1), 1),  # wide spatial
    ((2, 3, 8, 9, 7), (4, 3, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1),  # NCDHW
    ((2, 4, 6, 7, 5), (4, 2, 3, 2, 2), (2, 1, 1), (1, 0, 1), (1, 2, 1), 2),  # 5-D, g2
]
CONV_CASE_ROWS = tu.selected_cases(CONV_CASES, quick=QUICK_CONV_CASES)

# Parameter sweeps use the two output depths of the spec 4-D shape.
PARAM_CASES = [
    ((16, 128, 64, 60), (32, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((16, 128, 64, 60), (16, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1),
]
PARAM_CASE_ROWS = tu.selected_cases(PARAM_CASES, quick=[])

# ``.out`` shares the single public entry point and has its own call form.
OUT_CASES = [
    ((2, 3, 19, 7), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((2, 4, 8, 8), (4, 4, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((2, 4, 16, 16), (4, 4, 3, 3), (1, 1), (2, 2), (2, 2), 1),
    ((2, 4, 10, 8), (4, 2, 3, 2), (2, 1), (1, 2), (1, 2), 2),
]
OUT_CASE_ROWS = tu.selected_cases(OUT_CASES, quick=[])

# A caller buffer whose shape is not the result shape is resized rather than
# rejected, so the three prelude shapes below are supported inputs.
OUT_RESIZE_CASE = ((2, 4, 8, 8), (4, 4, 3, 3), (1, 1), (1, 1), (1, 1), 1)
OUT_RESIZE_ROWS = tu.selected_cases(
    [
        (prelude, dtype)
        for prelude in [(2, 4, 9, 9), (1,), (0,)]
        for dtype in SUPPORTED_DTYPES
    ],
    quick=[],
)

# ``alpha`` scales ``z``; None is the unspecified Optional value, and the rest
# cover zero, positive, negative, an int and the float nan/inf boundaries.
ALPHA_VALUES = [None, 0.0, 1.0, 0.5, -1.0, 2, float("inf"), float("-inf"), float("nan")]
ALPHA_CASE_ROWS = tu.selected_cases(ALPHA_VALUES, quick=[])

# Bias forms: unspecified, the 1-dim (C,) broadcast, the (C,1,1) broadcast and
# the (1,C,1,1) broadcast.
BIAS_FORMS = ["none", "channel", "channel_11", "broadcast_1C11"]
BIAS_CASE_ROWS = tu.selected_cases(BIAS_FORMS, quick=[])

# Positive nan/inf coverage per operand position. The geometry is depthwise with
# a 1x1 kernel, so no reduction mixes the payload with itself, and the operands
# that do not carry a payload are finite and positive; each payload value then
# keeps its own classification in the result.
SPECIAL_CASE = ((2, 8, 8, 8), (8, 1, 1, 1), (1, 1), (0, 0), (1, 1), 8)
SPECIAL_PLACEMENTS = ["input", "weight", "residual", "bias", "alpha_zero_residual"]
SPECIAL_CASE_ROWS = tu.selected_cases(
    [
        (placement, dtype, scenario)
        for placement in SPECIAL_PLACEMENTS
        for dtype, scenario in tu.special_value_cases(SUPPORTED_DTYPES)
    ],
    quick=[],
)

# Layout states: an alternative memory format, non-contiguous operands and a
# nonzero storage offset on each operand. ``bias`` element order changes the
# result, so the strided bias keeps the values of a (C,) bias in the same order.
LAYOUT_CASE = ((2, 4, 8, 8), (4, 4, 3, 3), (1, 1), (1, 1), (1, 1), 1)
LAYOUT_VARIANTS = [
    "input_channels_last",
    "input_noncontiguous",
    "input_storage_offset",
    "weight_noncontiguous",
    "weight_storage_offset",
    "residual_noncontiguous",
    "residual_storage_offset",
    "bias_storage_offset",
    "bias_noncontiguous",
]
LAYOUT_CASE_ROWS = tu.selected_cases(LAYOUT_VARIANTS, quick=[])


def _output_shape(x_shape, w_shape, stride, padding, dilation):
    return (x_shape[0], w_shape[0]) + tuple(
        (x_shape[i + 2] + 2 * padding[i] - dilation[i] * (w_shape[i + 2] - 1) - 1)
        // stride[i]
        + 1
        for i in range(len(stride))
    )


def _positive_tensor(dtype, shape):
    # A strictly positive finite constant in the tested dtype: multiplying the
    # special payload by it cannot turn an Inf into a NaN or zero it out, so each
    # payload value keeps its own classification in the result.
    return torch.full(shape, 0.25, dtype=dtype, device=flag_gems.device)


def _special_tensor(dtype, scenario, shape):
    # Tile the shared special-value payload over ``shape`` so every value in it
    # appears in the operand.
    payload = tu.make_special_input(dtype, scenario)
    size = math.prod(shape)
    return payload.repeat(-(-size // payload.numel()))[:size].reshape(shape)


def _relayout(tensor, offset=0):
    # Same values inside a wider buffer: non-contiguous strides and/or a nonzero
    # storage offset.
    pad = max(offset, 1)
    buffer = torch.zeros(
        tensor.shape[:-1] + (tensor.shape[-1] + pad,),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    buffer[..., offset : offset + tensor.shape[-1]] = tensor
    return buffer[..., offset : offset + tensor.shape[-1]]


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("conv_case", CONV_CASE_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cudnn_convolution_add_relu(conv_case, value_range, dtype):
    x_shape, w_shape, stride, padding, dilation, groups = conv_case
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)

    inp = tu.make_input(dtype, x_shape, value_range)
    weight = tu.make_input(dtype, w_shape, value_range)
    z = tu.make_input(dtype, out_shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_z = tu.to_reference(z)

    ref_out = torch.ops.aten.cudnn_convolution_add_relu(
        ref_inp,
        ref_weight,
        ref_z,
        None,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )
    res_out = flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        None,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("conv_case", OUT_CASE_ROWS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cudnn_convolution_add_relu_out(conv_case, value_range, dtype):
    x_shape, w_shape, stride, padding, dilation, groups = conv_case
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)

    inp = tu.make_input(dtype, x_shape, value_range)
    weight = tu.make_input(dtype, w_shape, value_range)
    z = tu.make_input(dtype, out_shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_z = tu.to_reference(z)

    # The out overload is callable on this backend and returns the caller
    # buffer, so both paths call it directly.
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.cudnn_convolution_add_relu.out(
        ref_inp,
        ref_weight,
        ref_z,
        None,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
        out=ref_out,
    )
    res_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_ret = flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        None,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
        out=res_out,
    )

    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("prelude,dtype", OUT_RESIZE_ROWS)
def test_cudnn_convolution_add_relu_out_resize(prelude, dtype):
    x_shape, w_shape, stride, padding, dilation, groups = OUT_RESIZE_CASE
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)

    inp = tu.make_input(dtype, x_shape, ["-1", "1"])
    weight = tu.make_input(dtype, w_shape, ["-1", "1"])
    z = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_z = tu.to_reference(z)

    ref_out = torch.empty(prelude, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.cudnn_convolution_add_relu.out(
        ref_inp,
        ref_weight,
        ref_z,
        None,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
        out=ref_out,
    )
    res_out = torch.empty(prelude, dtype=dtype, device=flag_gems.device)
    res_ret = flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        None,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
        out=res_out,
    )

    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("param_case", PARAM_CASE_ROWS)
@pytest.mark.parametrize("alpha", ALPHA_CASE_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cudnn_convolution_add_relu_alpha(param_case, alpha, dtype):
    x_shape, w_shape, stride, padding, dilation, groups = param_case
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)

    inp = tu.make_input(dtype, x_shape, ["-1", "1"])
    weight = tu.make_input(dtype, w_shape, ["-1", "1"])
    z = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_z = tu.to_reference(z)

    ref_out = torch.ops.aten.cudnn_convolution_add_relu(
        ref_inp,
        ref_weight,
        ref_z,
        alpha,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )
    res_out = flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        alpha,
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("param_case", PARAM_CASE_ROWS)
@pytest.mark.parametrize("bias_form", BIAS_CASE_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cudnn_convolution_add_relu_bias(param_case, bias_form, dtype):
    x_shape, w_shape, stride, padding, dilation, groups = param_case
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)
    channels = out_shape[1]

    inp = tu.make_input(dtype, x_shape, ["-1", "1"])
    weight = tu.make_input(dtype, w_shape, ["-1", "1"])
    z = tu.make_input(dtype, out_shape, ["-1", "1"])

    # Each advertised form is native-valid against this output.
    if bias_form == "none":
        bias = None
    elif bias_form == "channel":
        bias = tu.make_input(dtype, (channels,), ["-1", "1"])
    elif bias_form == "channel_11":
        bias = tu.make_input(dtype, (channels, 1, 1), ["-1", "1"])
    else:
        bias = tu.make_input(dtype, (1, channels, 1, 1), ["-1", "1"])

    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_z = tu.to_reference(z)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.cudnn_convolution_add_relu(
        ref_inp,
        ref_weight,
        ref_z,
        None,
        ref_bias,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )
    res_out = flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        None,
        bias,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("placement,dtype,scenario", SPECIAL_CASE_ROWS)
def test_cudnn_convolution_add_relu_special_values(placement, dtype, scenario):
    x_shape, w_shape, stride, padding, dilation, groups = SPECIAL_CASE
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)

    inp = _positive_tensor(dtype, x_shape)
    weight = _positive_tensor(dtype, w_shape)
    z = _positive_tensor(dtype, out_shape)
    bias = None
    alpha = None

    if placement == "input":
        inp = _special_tensor(dtype, scenario, x_shape)
    elif placement == "weight":
        weight = _special_tensor(dtype, scenario, w_shape)
    elif placement == "residual":
        z = _special_tensor(dtype, scenario, out_shape)
    elif placement == "alpha_zero_residual":
        # ``alpha`` still multiplies ``z`` when it is zero, so the payload
        # reaches the result through 0 * payload.
        z = _special_tensor(dtype, scenario, out_shape)
        alpha = 0.0
    else:
        bias = _special_tensor(dtype, scenario, (out_shape[1], 1, 1))

    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_z = tu.to_reference(z)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.cudnn_convolution_add_relu(
        ref_inp,
        ref_weight,
        ref_z,
        alpha,
        ref_bias,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )
    res_out = flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        alpha,
        bias,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("variant", LAYOUT_CASE_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cudnn_convolution_add_relu_layout(variant, dtype):
    x_shape, w_shape, stride, padding, dilation, groups = LAYOUT_CASE
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)
    channels = out_shape[1]

    inp = tu.make_input(dtype, x_shape, ["-1", "1"])
    weight = tu.make_input(dtype, w_shape, ["-1", "1"])
    z = tu.make_input(dtype, out_shape, ["-1", "1"])
    bias = None

    if variant == "input_channels_last":
        inp = inp.to(memory_format=torch.channels_last)
    elif variant == "input_noncontiguous":
        inp = _relayout(inp)
    elif variant == "input_storage_offset":
        inp = _relayout(inp, offset=1)
    elif variant == "weight_noncontiguous":
        weight = _relayout(weight)
    elif variant == "weight_storage_offset":
        weight = _relayout(weight, offset=1)
    elif variant == "residual_noncontiguous":
        z = _relayout(z)
    elif variant == "residual_storage_offset":
        z = _relayout(z, offset=1)
    elif variant == "bias_noncontiguous":
        bias = tu.make_input(dtype, (channels, 2), ["-1", "1"])[:, 0]
    else:
        buffer = tu.make_input(dtype, (channels + 2,), ["-1", "1"])
        bias = buffer[1 : channels + 1]

    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_z = tu.to_reference(z)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.cudnn_convolution_add_relu(
        ref_inp,
        ref_weight,
        ref_z,
        None,
        ref_bias,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )
    res_out = flag_gems.cudnn_convolution_add_relu(
        inp,
        weight,
        z,
        None,
        bias,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("dtype", NEGATIVE_DTYPES)
def test_cudnn_convolution_add_relu_negative_dtype(dtype):
    inp = tu.make_input(dtype, (2, 4, 8, 8), ["0", "1"])
    weight = tu.make_input(dtype, (4, 4, 3, 3), ["0", "1"])
    z = tu.make_input(dtype, (2, 4, 8, 8), ["0", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_convolution_add_relu(
            inp, weight, z, None, None, [1, 1], [1, 1], [1, 1], 1
        )


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("rank_case", RANK_NEGATIVE_CASES)
def test_cudnn_convolution_add_relu_negative_rank(rank_case):
    inp = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    weight = tu.make_input(torch.float32, (4, 4, 3, 3), ["-1", "1"])
    z = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    stride, padding, dilation = [1, 1], [1, 1], [1, 1]

    if rank_case == "self_2d":
        inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
    elif rank_case == "self_3d":
        inp = tu.make_input(torch.float32, (2, 4, 8), ["-1", "1"])
    elif rank_case == "weight_3d":
        weight = tu.make_input(torch.float32, (4, 4, 3), ["-1", "1"])
    elif rank_case == "weight_rank_mismatch":
        # A 5-D input with a 4-D weight: the two descriptors disagree on rank.
        inp = tu.make_input(torch.float32, (2, 4, 8, 8, 8), ["-1", "1"])
        z = tu.make_input(torch.float32, (2, 4, 8, 8, 8), ["-1", "1"])
        stride, padding, dilation = [1, 1, 1], [1, 1, 1], [1, 1, 1]
    else:
        # Three-element parameter lists do not describe a 4-D convolution.
        stride, padding, dilation = [1, 1, 1], [1, 1, 1], [1, 1, 1]

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_convolution_add_relu(
            inp, weight, z, None, None, stride, padding, dilation, 1
        )


@pytest.mark.cudnn_convolution_add_relu
def test_cudnn_convolution_add_relu_negative_channel_mismatch():
    # weight.size(1) must equal self.size(1): 8 in-channels against 4 input
    # channels is invalid, and the residual already matches the true output.
    inp = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    weight = tu.make_input(torch.float32, (4, 8, 3, 3), ["-1", "1"])
    z = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_convolution_add_relu(
            inp, weight, z, None, None, [1, 1], [1, 1], [1, 1], 1
        )


@pytest.mark.cudnn_convolution_add_relu
def test_cudnn_convolution_add_relu_negative_groups():
    # groups must satisfy in_channels == groups * weight.size(1). With 6 input
    # channels and a weight holding 2 in-channels per group, groups=4 would
    # require 8 input channels, so these arguments are inconsistent.
    inp = tu.make_input(torch.float32, (2, 6, 8, 8), ["-1", "1"])
    weight = tu.make_input(torch.float32, (4, 2, 3, 3), ["-1", "1"])
    z = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_convolution_add_relu(
            inp, weight, z, None, None, [1, 1], [1, 1], [1, 1], 4
        )


@pytest.mark.cudnn_convolution_add_relu
def test_cudnn_convolution_add_relu_negative_padding():
    # Negative padding is invalid. The residual matches the (2,4,4,4) output
    # such padding would imply, so this stays invalid whether the padding is
    # rejected outright or ignored (the residual cannot combine with the
    # (2,4,6,6) output an ignored padding produces).
    inp = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    weight = tu.make_input(torch.float32, (4, 4, 3, 3), ["-1", "1"])
    z = tu.make_input(torch.float32, (2, 4, 4, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_convolution_add_relu(
            inp, weight, z, None, None, [1, 1], [-1, -1], [1, 1], 1
        )


@pytest.mark.cudnn_convolution_add_relu
def test_cudnn_convolution_add_relu_negative_tensor_alpha():
    # ``alpha`` is a Scalar?: a 0-dim tensor does not match the schema.
    inp = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    weight = tu.make_input(torch.float32, (4, 4, 3, 3), ["-1", "1"])
    z = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    alpha = torch.tensor(1.0, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_convolution_add_relu(
            inp, weight, z, alpha, None, [1, 1], [1, 1], [1, 1], 1
        )


@pytest.mark.cudnn_convolution_add_relu
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_NEGATIVE_DTYPES)
def test_cudnn_convolution_add_relu_negative_out_dtype(out_dtype):
    # The operands are float32, so the buffer must be float32 too: a buffer of
    # any other float dtype is rejected. Its shape may differ, because the out
    # overload resizes it.
    inp = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    weight = tu.make_input(torch.float32, (4, 4, 3, 3), ["-1", "1"])
    z = tu.make_input(torch.float32, (2, 4, 8, 8), ["-1", "1"])
    out = torch.empty((2, 4, 8, 8), dtype=out_dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_convolution_add_relu(
            inp, weight, z, None, None, [1, 1], [1, 1], [1, 1], 1, out=out
        )
