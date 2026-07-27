import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.ops.conv_transpose2d import (
    _pair,
    _validate_conv_transpose2d_args,
)

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _conv_transpose2d_output_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    output_padding_height: tl.constexpr,
    output_padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    channel_blocks: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    pid_width = tl.program_id(0)
    row_id = tl.program_id(1)
    output_channel = tl.program_id(2)

    offsets_width = pid_width
    safe_width = pid_width
    batch = row_id // output_height
    output_row = row_id % output_height
    group = output_channel // output_channels_per_group
    output_channel_in_group = output_channel % output_channels_per_group

    accumulator = tl.zeros((), dtype=tl.float32)
    for kh in tl.static_range(0, kernel_height):
        input_row_numerator = output_row + padding_height - kh * dilation_height
        row_valid = (input_row_numerator >= 0) & (
            input_row_numerator % stride_height == 0
        )
        input_row = input_row_numerator // stride_height
        row_in_bounds = (input_row >= 0) & (input_row < input_height)
        safe_input_row = tl.where(row_in_bounds, input_row, input_row * 0)
        for kw in tl.static_range(0, kernel_width):
            input_col_numerator = (
                safe_width + padding_width - kw * dilation_width
            )
            col_valid = (input_col_numerator >= 0) & (
                input_col_numerator % stride_width == 0
            )
            input_col = input_col_numerator // stride_width
            col_in_bounds = (input_col >= 0) & (input_col < input_width)
            safe_input_col = tl.where(col_in_bounds, input_col, input_col * 0)
            valid = row_valid & col_valid & row_in_bounds & col_in_bounds

            for ci_base in tl.static_range(0, channel_blocks):
                input_channel_in_group = (
                    ci_base * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
                )
                input_channel = group * input_channels_per_group + input_channel_in_group
                input_offsets = (
                    (batch * input_channels + input_channel)
                    * input_height
                    + safe_input_row
                ) * input_width + safe_input_col
                weight_offsets = (
                    (input_channel * output_channels_per_group + output_channel_in_group)
                    * kernel_height
                    * kernel_width
                    + kh * kernel_width
                    + kw
                )
                channel_mask = input_channel_in_group < input_channels_per_group
                input_values = tl.load(
                    input_ptr + input_offsets,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                input_values = tl.where(valid, input_values, 0.0)
                weight_values = tl.load(
                    weight_ptr + weight_offsets,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(input_values * weight_values, axis=0)

    if bias_ptr is not None:
        accumulator += tl.load(bias_ptr + output_channel).to(tl.float32)
    tl.store(
        output_ptr
        + (batch * output_channels + output_channel) * output_height * output_width
        + output_row * output_width
        + offsets_width,
        accumulator,
    )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE2D")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if (
        input.dtype not in supported_dtypes
        or weight.dtype not in supported_dtypes
        or (bias is not None and bias.dtype not in supported_dtypes)
    ):
        raise NotImplementedError(
            "conv_transpose2d does not support the requested dtype"
        )
    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)
    stride_h, stride_w = _pair(stride)
    padding_h, padding_w = _pair(padding)
    output_padding_h, output_padding_w = _pair(output_padding)
    dilation_h, dilation_w = _pair(dilation)
    _validate_conv_transpose2d_args(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    )
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    batch_size, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, kernel_height, kernel_width = weight.shape
    output_channels = output_channels_per_group * groups
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (kernel_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (kernel_width - 1)
        + output_padding_w
        + 1
    )
    output = torch.empty(
        (batch_size, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    if batch_size != 0:
        grid = (
            output_width,
            batch_size * output_height,
            output_channels,
        )
        with torch_device_fn.device(input.device):
            _conv_transpose2d_output_kernel[grid](
                input,
                weight,
                bias,
                output,
                batch_size,
                input_channels,
                input_height,
                input_width,
                output_channels,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                input_channels // groups,
                output_channels_per_group,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                output_padding_h,
                output_padding_w,
                dilation_h,
                dilation_w,
                groups,
                channel_blocks=triton.cdiv(input_channels // groups, 32),
                BLOCK_WIDTH=8,
                BLOCK_CHANNELS=32,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    if input_was_unbatched:
        return output.squeeze(0)
    return output
