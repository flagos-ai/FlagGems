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

import logging
import math

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.utils import libentry

from .conv2d import conv2d_output_size

logger = logging.getLogger(__name__)

_BLOCK_NI_DO_HO_WO = 32
_BLOCK_CI = 32
_BLOCK_CO = 32

# Zero bias for the bias-less case, kept per (size, device): the kernel only reads
# it, and re-allocating it per call showed up in the launch cost.
_ZERO_BIAS = {}


def _zero_bias(out_c, device):
    key = (out_c, str(device))
    t = _ZERO_BIAS.get(key)
    if t is None:
        t = torch.zeros(out_c, device=device, dtype=torch.float)
        if len(_ZERO_BIAS) < 64:
            _ZERO_BIAS[key] = t
    return t


def conv3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 3D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.

    Returns:
        Output size of 3D convolution.
    """
    return conv2d_output_size(in_size, kernel_size, stride, padding, dilation)


# @triton.autotune(
#     configs=runtime.get_tuned_config("conv3d_forward"),
#     key=[
#         "in_n",
#         "weight_c",
#         "input_depth",
#         "input_height",
#         "input_width",
#         "out_c",
#         "out_depth",
#         "out_height",
#         "out_width",
#         "weight_depth",
#         "weight_height",
#         "weight_width",
#         "stride_depth",
#         "stride_height",
#         "stride_width",
#         "padding_depth",
#         "padding_height",
#         "padding_width",
#         "groups",
#     ],
# )
# Scalar arguments are kept out of the specialization list so that the launch
# keeps a stable argument list for every geometry.
_NO_SPECIALIZE_ARGS = [
    "in_n",
    "input_depth",
    "input_height",
    "input_width",
    "out_c",
    "out_depth",
    "out_height",
    "out_width",
    "input_n_stride",
    "input_c_stride",
    "input_depth_stride",
    "input_height_stride",
    "input_width_stride",
    "weight_n_stride",
    "weight_c_stride",
    "weight_depth_stride",
    "weight_height_stride",
    "weight_width_stride",
    "output_n_stride",
    "output_c_stride",
    "output_depth_stride",
    "output_height_stride",
    "output_width_stride",
]


@libentry()
@triton.jit(do_not_specialize=_NO_SPECIALIZE_ARGS)
def conv3d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_depth,
    input_height,
    input_width,
    out_c,
    out_depth,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_depth: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_DO_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_do_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_depth out_height out_weight value in kernel
    ni_do_ho_wo_offset = pid_ni_do_ho_wo * BLOCK_NI_DO_HO_WO + tl.arange(
        0, BLOCK_NI_DO_HO_WO
    )
    ni_do_ho_offset = ni_do_ho_wo_offset // out_width
    ni_do_offset = ni_do_ho_offset // out_height
    in_n_point_value = ni_do_offset // out_depth
    output_depth_point_value = ni_do_offset % out_depth
    output_height_point_value = ni_do_ho_offset % out_height
    output_width_point_value = ni_do_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_DO_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for dhwc in range(weight_depth * weight_height * weight_width * BLOCK_CI_COUNT):
        c = (dhwc % BLOCK_CI_COUNT) * BLOCK_CI
        dhw = dhwc // BLOCK_CI_COUNT
        dh = dhw // weight_width
        d = dh // weight_height
        h = dh % weight_height
        w = dhw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_depth_offset = (
            d * dilation_depth - padding_depth + stride_depth * output_depth_point_value
        )
        input_height_offset = (
            h * dilation_height
            - padding_height
            + stride_height * output_height_point_value
        )
        input_width_offset = (
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_depth_stride * input_depth_offset)[:, None]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_depth_stride * d)
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_depth_offset)[:, None]
            & (input_depth_offset < input_depth)[:, None]
            & (0 <= input_height_offset)[:, None]
            & (input_height_offset < input_height)[:, None]
            & (0 <= input_width_offset)[:, None]
            & (input_width_offset < input_width)[:, None]
        )
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0).to(
            tl.float32
        )
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0).to(
            tl.float32
        )

        accum += tl.sum(input_block[:, :, None] * weight_block[None, :, :], axis=1)
    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask=mask_bias, other=0.0).to(tl.float32)
    accum += bias
    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_depth_stride * output_depth_point_value)[:, None]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_depth_point_value < out_depth)[:, None]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


# One lane per output element, accumulating the taps in float32. Used when the
# padding is at least one effective kernel extent, so some output element's
# whole receptive field lies in the padding and the result is just the bias.
@libentry()
@triton.jit(do_not_specialize=_NO_SPECIALIZE_ARGS)
def conv3d_padded_tile_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_depth,
    input_height,
    input_width,
    out_c,
    out_depth,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_depth: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_DO_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK_NI_DO_HO_WO + tl.arange(0, BLOCK_NI_DO_HO_WO)
    total = in_n * out_c * out_depth * out_height * out_width
    ow = lane % out_width
    q = lane // out_width
    oh = q % out_height
    q = q // out_height
    od = q % out_depth
    q = q // out_depth
    oc = q % out_c
    ni = q // out_c
    group = oc // (out_c // groups)

    live = lane < total
    acc = tl.zeros((BLOCK_NI_DO_HO_WO,), tl.float32)
    for r in range(weight_depth):
        idd = od * stride_depth - padding_depth + r * dilation_depth
        for s in range(weight_height):
            ih = oh * stride_height - padding_height + s * dilation_height
            for t in range(weight_width):
                iw = ow * stride_width - padding_width + t * dilation_width
                valid = (
                    live
                    & (idd >= 0)
                    & (idd < input_depth)
                    & (ih >= 0)
                    & (ih < input_height)
                    & (iw >= 0)
                    & (iw < input_width)
                )
                idd_s = tl.where(valid, idd, 0)
                ih_s = tl.where(valid, ih, 0)
                iw_s = tl.where(valid, iw, 0)
                for ci in range(weight_c):
                    xv = tl.load(
                        input_pointer
                        + ni * input_n_stride
                        + (group * weight_c + ci) * input_c_stride
                        + idd_s * input_depth_stride
                        + ih_s * input_height_stride
                        + iw_s * input_width_stride,
                        mask=live,
                        other=0.0,
                    )
                    xv = tl.where(valid, xv, 0.0).to(tl.float32)
                    wv = tl.load(
                        weight_pointer
                        + oc * weight_n_stride
                        + ci * weight_c_stride
                        + r * weight_depth_stride
                        + s * weight_height_stride
                        + t * weight_width_stride,
                        mask=live,
                        other=0.0,
                    ).to(tl.float32)
                    acc += xv * wv
    acc += tl.load(bias_pointer + oc, mask=live, other=0.0).to(tl.float32)
    tl.store(
        output_pointer
        + ni * output_n_stride
        + oc * output_c_stride
        + od * output_depth_stride
        + oh * output_height_stride
        + ow * output_width_stride,
        acc,
        mask=live,
    )


# class Conv3d(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
#         pass


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_KUNLUNXIN CONV3D")

    assert weight.ndim == 5, "Weights must be 5D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), "Bias must be 1D, received shape {bias.shape}"

    assert (
        input.shape[1] == groups * weight.shape[1]
    ), "Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
    assert (
        bias is None or weight.shape[0] == bias.shape[0]
    ), "Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

    if isinstance(stride, (list, tuple)):
        stride_depth, stride_height, stride_width = stride
    else:
        stride_depth = stride_height = stride_width = stride

    if isinstance(dilation, (list, tuple)):
        dilation_depth, dilation_height, dilation_width = dilation
    else:
        dilation_depth = dilation_height = dilation_width = dilation

    if isinstance(padding, str):
        if padding == "same":
            assert (
                stride_depth == 1 and stride_height == 1 and stride_width == 1
            ), "Doesn't support any stride values other than 1 in padding = 'same' mode"
            padding_depth = math.ceil(dilation_depth * (weight.shape[-3] - 1) / 2)
            padding_height = math.ceil(dilation_height * (weight.shape[-2] - 1) / 2)
            padding_width = math.ceil(dilation_width * (weight.shape[-1] - 1) / 2)
        elif padding == "valid":
            padding_depth = padding_height = padding_width = 0
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_depth, padding_height, padding_width = padding
    else:
        padding_depth = padding_height = padding_width = padding

    in_n, _, input_depth, input_height, input_width = input.shape
    out_c, weight_c, weight_depth, weight_height, weight_width = weight.shape
    out_depth = conv3d_output_size(
        input_depth, weight_depth, stride_depth, padding_depth, dilation_depth
    )

    out_height = conv3d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv3d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    output_dtype = input.dtype

    # The kernel up-casts both operands to float32 before multiplying and keeps a
    # float32 accumulator, so half precision inputs do not need a host-side
    # promotion: a promoted round trip costs more than the convolution itself at
    # these sizes.
    compute_dtype = output_dtype

    output = torch.empty(
        (in_n, out_c, out_depth, out_height, out_width),
        device=input.device,
        dtype=compute_dtype,
    )

    # BLOCK_NI_HO_WO along the in_n, out_height, and out_width dimensions,
    # BLOCK_CO along the out_c,
    # one group per cat. The block sizes are fixed, so the grid is a plain tuple.
    grid = (
        triton.cdiv(in_n * out_depth * out_height * out_width, _BLOCK_NI_DO_HO_WO),
        triton.cdiv(out_c // groups, _BLOCK_CO),
        groups,
    )

    if bias is None:
        bias_pointer = _zero_bias(out_c, input.device)
    else:
        bias_pointer = bias.to(torch.float)

    # When the padding is at least one effective kernel extent, some output
    # element's whole receptive field lies outside the input and its result is
    # just the bias, so compute those geometries with the tile kernel instead.
    kernel = conv3d_forward_kernel
    if (
        padding_depth >= dilation_depth * (weight_depth - 1) + 1
        or padding_height >= dilation_height * (weight_height - 1) + 1
        or padding_width >= dilation_width * (weight_width - 1) + 1
    ):
        kernel = conv3d_padded_tile_kernel
        grid = (
            triton.cdiv(
                in_n * out_c * out_depth * out_height * out_width, _BLOCK_NI_DO_HO_WO
            ),
        )

    kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        input_depth,
        input_height,
        input_width,
        out_c,
        out_depth,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_depth,
        weight_height,
        weight_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        groups=groups,
        BLOCK_NI_DO_HO_WO=_BLOCK_NI_DO_HO_WO,
        BLOCK_CI=_BLOCK_CI,
        BLOCK_CO=_BLOCK_CO,
    )

    if padding == "same":
        output = output[
            ...,
            (out_depth - input_depth) :,
            (out_height - input_height) :,
            (out_width - input_width) :,
        ]

    return output
