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

# GCU400 专属 conv_depthwise2d：不再委托通用 conv2d。
#   * 公版路径（ops/conv2d.py 的 im2col + tl.dot）在 GCU 上 fp32 全量错值
#     （PAD_LANES=12 lane workaround + tune 配置缺陷，见 conv2d__opt.py 头注）；
#   * conv2d__opt 路径虽正确，但 weight_c=1 < 16 时走 host 侧 F.pad 通道预填充
#     （aten 计算外流 + 多次 kernel launch + 16x 输入物化）；
#   * 2D tile (M, C) FMA 外积形态触发 gcuir PassManager legalization 崩溃
#     （P02 家族，同 cudnn_convolution_relu_direct2d 实例签名）。
# 此处采用已验证可编的 1D-tile 行程式（镜像 generic _direct_conv3d_kernel：
# 每 program 单输出通道 + 标量 weight tap），host 仅分配输出后一次 launch，
# 累加全程 fp32。

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def conv_depthwise2d_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    num_sp_tiles,
    in_n,
    input_height,
    input_width,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    multiplier: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_channel = pid // num_sp_tiles
    pid_sp = pid % num_sp_tiles

    sp_offset = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ow = sp_offset % out_width
    tmp = sp_offset // out_width
    oh = tmp % out_height
    n = tmp // out_height
    sp_ok = n < in_n

    iw0 = ow * stride_width - padding_width

    # Depthwise: output channel pid_channel reads input channel
    # pid_channel // multiplier.
    in_base = (
        input_pointer
        + n * input_n_stride
        + (pid_channel // multiplier) * input_c_stride
        + iw0 * input_width_stride
    )
    w_base = weight_pointer + pid_channel * weight_n_stride

    accum = tl.zeros((BLOCK_SP,), dtype=tl.float32)
    for kh in tl.static_range(weight_height):
        ih = oh * stride_height + kh * dilation_height - padding_height
        row_ok = sp_ok & (ih >= 0) & (ih < input_height)
        p = in_base + ih * input_height_stride
        for kw in tl.static_range(weight_width):
            iw = iw0 + kw * dilation_width
            in_mask = row_ok & (iw >= 0) & (iw < input_width)
            x = tl.load(p, mask=in_mask, other=0.0).to(tl.float32)
            w = tl.load(
                w_base + kh * weight_height_stride + kw * weight_width_stride
            ).to(tl.float32)
            accum += x * w
            p += dilation_width * input_width_stride

    if HAS_BIAS:
        accum += tl.load(bias_pointer + pid_channel).to(tl.float32)

    tl.store(
        output_pointer
        + pid_channel * output_c_stride
        + n * output_n_stride
        + oh * output_height_stride
        + ow * output_width_stride,
        accum.to(output_pointer.dtype.element_ty),
        mask=sp_ok,
    )


def _conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation):
    logger.debug("GEMS DEPTHWISE")

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
    multiplier = out_c // in_c

    out_height = (
        input_height + 2 * padding_height - dilation_height * (weight_height - 1) - 1
    ) // stride_height + 1
    out_width = (
        input_width + 2 * padding_width - dilation_width * (weight_width - 1) - 1
    ) // stride_width + 1

    output = torch.empty(
        (in_n, out_c, out_height, out_width),
        device=input.device,
        dtype=input.dtype,
    )
    total_m = in_n * out_height * out_width

    # Flat 1D grid keeps every axis off the tight grid.y/z (<=255) limits; grow
    # BLOCK_SP when the channel-spatial product would exceed the grid.x cap.
    BLOCK_SP = 256
    while triton.cdiv(total_m, BLOCK_SP) * out_c > 65535 and BLOCK_SP < 4096:
        BLOCK_SP *= 2
    num_sp_tiles = triton.cdiv(total_m, BLOCK_SP)
    grid = (num_sp_tiles * out_c,)
    conv_depthwise2d_kernel[grid](
        input,
        weight,
        bias if bias is not None else weight,
        output,
        num_sp_tiles,
        in_n,
        input_height,
        input_width,
        out_height,
        out_width,
        *input.stride(),
        weight.stride(0),
        weight.stride(2),
        weight.stride(3),
        *output.stride(),
        stride_height=stride_height,
        stride_width=stride_width,
        padding_height=padding_height,
        padding_width=padding_width,
        dilation_height=dilation_height,
        dilation_width=dilation_width,
        weight_height=weight_height,
        weight_width=weight_width,
        multiplier=multiplier,
        HAS_BIAS=bias is not None,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
    )
    return output
