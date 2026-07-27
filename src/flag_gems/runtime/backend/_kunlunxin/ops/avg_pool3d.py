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


def _triple(value, name):
    if isinstance(value, int):
        return value, value, value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, int) for item in value)
    ):
        return int(value[0]), int(value[1]), int(value[2])
    raise ValueError(f"{name} must be an int or a tuple/list of three ints")


def _output_size(input_size, kernel, stride, padding, ceil_mode):
    numerator = input_size + 2 * padding - kernel
    output = (numerator + stride - 1) // stride + 1 if ceil_mode else numerator // stride + 1
    if ceil_mode and (output - 1) * stride >= input_size + padding:
        output -= 1
    return output


@libentry()
@triton.jit
def _avg_pool3d_kernel(
    input_ptr,
    output_ptr,
    total,
    C,
    D,
    H,
    W,
    OUT_D,
    OUT_H,
    OUT_W,
    SD,
    SH,
    SW,
    PD,
    PH,
    PW,
    DIVISOR,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    HAS_DIVISOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0).to(tl.int32) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total
    output_w = offsets % OUT_W
    remaining = offsets // OUT_W
    output_h = remaining % OUT_H
    remaining = remaining // OUT_H
    output_d = remaining % OUT_D
    channel_batch = remaining // OUT_D

    start_d = output_d * SD - PD
    start_h = output_h * SH - PH
    start_w = output_w * SW - PW
    padded_end_d = tl.minimum(start_d + KD, D + PD)
    padded_end_h = tl.minimum(start_h + KH, H + PH)
    padded_end_w = tl.minimum(start_w + KW, W + PW)
    pool_size = (
        (padded_end_d - start_d)
        * (padded_end_h - start_h)
        * (padded_end_w - start_w)
    )

    accumulator = tl.zeros((BLOCK,), tl.float32)
    valid_count = tl.zeros((BLOCK,), tl.int32)
    for kd in tl.static_range(0, KD):
        input_d = start_d + kd
        for kh in tl.static_range(0, KH):
            input_h = start_h + kh
            for kw in tl.static_range(0, KW):
                input_w = start_w + kw
                in_bounds = (
                    (input_d >= 0)
                    & (input_d < D)
                    & (input_h >= 0)
                    & (input_h < H)
                    & (input_w >= 0)
                    & (input_w < W)
                )
                input_offset = (
                    ((channel_batch * D + input_d) * H + input_h) * W + input_w
                )
                value = tl.load(
                    input_ptr + input_offset,
                    mask=valid & in_bounds,
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.where(in_bounds, value, 0.0)
                valid_count += tl.where(in_bounds, 1, 0)

    if HAS_DIVISOR:
        divisor = DIVISOR
    elif COUNT_INCLUDE_PAD:
        divisor = pool_size
    else:
        divisor = valid_count
    tl.store(output_ptr + offsets, accumulator / divisor, mask=valid)


def avg_pool3d(
    input,
    kernel_size,
    stride=(),
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    logger.debug("GEMS_KUNLUNXIN AVG_POOL3D")
    was_unbatched = input.ndim == 4
    x = input.unsqueeze(0) if was_unbatched else input
    if x.ndim != 5:
        raise ValueError("avg_pool3d expects a 4D or 5D input tensor")

    kernel_d, kernel_h, kernel_w = _triple(kernel_size, "kernel_size")
    if stride == () or stride == []:
        stride_d, stride_h, stride_w = kernel_d, kernel_h, kernel_w
    else:
        stride_d, stride_h, stride_w = _triple(stride, "stride")
    padding_d, padding_h, padding_w = _triple(padding, "padding")

    batch, channels, depth, height, width = x.shape
    output_d = _output_size(depth, kernel_d, stride_d, padding_d, ceil_mode)
    output_h = _output_size(height, kernel_h, stride_h, padding_h, ceil_mode)
    output_w = _output_size(width, kernel_w, stride_w, padding_w, ceil_mode)
    output = torch.empty(
        (batch, channels, output_d, output_h, output_w),
        dtype=input.dtype,
        device=input.device,
    )
    total = output.numel()
    if total:
        x = x.contiguous()
        block = 128
        divisor = 0 if divisor_override is None else divisor_override
        with torch_device_fn.device(input.device):
            _avg_pool3d_kernel[(triton.cdiv(total, block),)](
                x,
                output,
                total,
                channels,
                depth,
                height,
                width,
                output_d,
                output_h,
                output_w,
                stride_d,
                stride_h,
                stride_w,
                padding_d,
                padding_h,
                padding_w,
                divisor,
                KD=kernel_d,
                KH=kernel_h,
                KW=kernel_w,
                COUNT_INCLUDE_PAD=count_include_pad,
                HAS_DIVISOR=divisor_override is not None,
                BLOCK=block,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    return output.squeeze(0) if was_unbatched else output
