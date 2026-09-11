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

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv2d import conv2d, conv2d_output_size
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def conv_depthwise2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_n_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
    channel_multiplier: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_oc = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    n_oh_offsets = m_offsets // out_width
    n_offsets = n_oh_offsets // out_height
    oh_offsets = n_oh_offsets % out_height
    ow_offsets = m_offsets % out_width
    input_c_offsets = oc_offsets // channel_multiplier

    accum = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    for kh in range(weight_height):
        ih_offsets = oh_offsets * stride_height + kh * dilation_height - padding_height
        valid_h = (0 <= ih_offsets) & (ih_offsets < input_height)
        for kw in range(weight_width):
            iw_offsets = ow_offsets * stride_width + kw * dilation_width - padding_width
            valid_w = (0 <= iw_offsets) & (iw_offsets < input_width)
            input_ptrs = (
                input_pointer
                + (n_offsets[:, None] * input_n_stride)
                + (input_c_offsets[None, :] * input_c_stride)
                + (ih_offsets[:, None] * input_height_stride)
                + (iw_offsets[:, None] * input_width_stride)
            )
            weight_ptrs = (
                weight_pointer
                + oc_offsets[None, :] * weight_n_stride
                + kh * weight_height_stride
                + kw * weight_width_stride
            )
            mask = (
                (m_offsets < in_n * out_height * out_width)[:, None]
                & (oc_offsets < out_c)[None, :]
                & valid_h[:, None]
                & valid_w[:, None]
            )
            input_block = tl.load(input_ptrs, mask=mask, other=0.0)
            weight_block = tl.load(
                weight_ptrs, mask=(oc_offsets < out_c)[None, :], other=0.0
            )
            accum += input_block * weight_block

    if HAS_BIAS:
        bias = tl.load(bias_pointer + oc_offsets, mask=oc_offsets < out_c, other=0.0)
        accum += bias[None, :]

    output_ptrs = (
        output_pointer
        + (n_offsets[:, None] * output_n_stride)
        + (oc_offsets[None, :] * output_c_stride)
        + (oh_offsets[:, None] * output_height_stride)
        + (ow_offsets[:, None] * output_width_stride)
    )
    output_mask = (m_offsets < in_n * out_height * out_width)[:, None] & (
        oc_offsets < out_c
    )[None, :]
    tl.store(output_ptrs, accum, mask=output_mask)


def _conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation):
    logger.debug("GEMS ILUVATAR DEPTHWISE")
    assert (
        input.ndim == 4
    ), "Invalid input tensor must be 4D, recevied shape {input.shape}"
    assert (
        weight.shape[0] % input.shape[1] == 0
    ), "Output channels must be multiple of input, recevied output {weught.shape[0], input {input.shape[0]}}"
    assert (
        weight.shape[1] == 1
    ), "input channels of per goups must be 1, recevied {weight.shape[1]}"
    groups = input.shape[1]
    if weight.shape[2] * weight.shape[3] <= 4:
        return conv2d(input, weight, bias, stride, padding, dilation, groups)

    if isinstance(stride, (list, tuple)):
        stride_height, stride_width = stride
    else:
        stride_height = stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_height, padding_width = padding
    else:
        padding_height = padding_width = padding

    if isinstance(dilation, (list, tuple)):
        dilation_height, dilation_width = dilation
    else:
        dilation_height = dilation_width = dilation

    in_n, in_c, input_height, input_width = input.shape
    out_c, _, weight_height, weight_width = weight.shape
    channel_multiplier = out_c // in_c
    out_height = conv2d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv2d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )
    output = torch.empty(
        (in_n, out_c, out_height, out_width), device=input.device, dtype=input.dtype
    )

    grid = lambda META: (
        triton.cdiv(in_n * out_height * out_width, META["BLOCK_M"]),
        triton.cdiv(out_c, META["BLOCK_OC"]),
    )
    block_oc = 1
    bias_pointer = (
        bias
        if bias is not None
        else torch.empty(0, device=input.device, dtype=input.dtype)
    )

    conv_depthwise2d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        *input.stride(),
        weight.stride(0),
        weight.stride(2),
        weight.stride(3),
        *output.stride(),
        channel_multiplier,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        HAS_BIAS=(bias is not None),
        BLOCK_M=256,
        BLOCK_OC=block_oc,
        num_warps=4,
    )
    return output
