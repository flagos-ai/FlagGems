# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _parse_2tuple(value, name):
    if isinstance(value, int):
        return value, value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} must be an int or a tuple/list of two ints")


@libentry()
@triton.jit
def _im2col_kernel(
    input_ptr,
    output_ptr,
    total,
    C,
    H,
    W,
    KH,
    KW,
    DH,
    DW,
    PH,
    PW,
    SH,
    SW,
    OUT_W,
    ROWS,
    L,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0).to(tl.int32) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total

    output_pos = offsets % L
    row_batch = offsets // L
    row = row_batch % ROWS
    batch = row_batch // ROWS

    kernel_area = KH * KW
    channel = row // kernel_area
    kernel_pos = row % kernel_area
    kernel_h = kernel_pos // KW
    kernel_w = kernel_pos % KW
    output_h = output_pos // OUT_W
    output_w = output_pos % OUT_W
    input_h = output_h * SH - PH + kernel_h * DH
    input_w = output_w * SW - PW + kernel_w * DW
    in_bounds = (input_h >= 0) & (input_h < H) & (input_w >= 0) & (input_w < W)

    input_offsets = ((batch * C + channel) * H + input_h) * W + input_w
    values = tl.load(input_ptr + input_offsets, mask=valid & in_bounds, other=0.0)
    values = tl.where(in_bounds, values, 0.0)
    tl.store(output_ptr + offsets, values, mask=valid)


def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    logger.debug("GEMS_KUNLUNXIN IM2COL")
    was_unbatched = input.ndim == 3
    x = input.unsqueeze(0) if was_unbatched else input
    if x.ndim != 4:
        raise ValueError("im2col expects input of shape (N, C, H, W) or (C, H, W)")

    kernel_h, kernel_w = _parse_2tuple(kernel_size, "kernel_size")
    dilation_h, dilation_w = _parse_2tuple(dilation, "dilation")
    padding_h, padding_w = _parse_2tuple(padding, "padding")
    stride_h, stride_w = _parse_2tuple(stride, "stride")

    batch, channels, height, width = x.shape
    output_h = (
        height
        + 2 * padding_h
        - (dilation_h * (kernel_h - 1) + 1)
    ) // stride_h + 1
    output_w = (
        width
        + 2 * padding_w
        - (dilation_w * (kernel_w - 1) + 1)
    ) // stride_w + 1
    rows = channels * kernel_h * kernel_w
    locations = output_h * output_w
    output = torch.empty(
        (batch, rows, locations), dtype=input.dtype, device=input.device
    )
    total = output.numel()
    if total:
        x = x.contiguous()
        block = 256
        with torch_device_fn.device(input.device):
            _im2col_kernel[(triton.cdiv(total, block),)](
                x,
                output,
                total,
                channels,
                height,
                width,
                kernel_h,
                kernel_w,
                dilation_h,
                dilation_w,
                padding_h,
                padding_w,
                stride_h,
                stride_w,
                output_w,
                rows,
                locations,
                BLOCK=block,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    return output.squeeze(0) if was_unbatched else output
