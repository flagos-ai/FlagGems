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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext


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
def _avg_pool3d_backward_gather_kernel(
    grad_output_ptr,
    grad_input_ptr,
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
    input_w = offsets % W
    remaining = offsets // W
    input_h = remaining % H
    remaining = remaining // H
    input_d = remaining % D
    channel_batch = remaining // D

    accumulator = tl.zeros((BLOCK,), tl.float32)
    for kd in tl.static_range(0, KD):
        output_d = (input_d + PD - kd) // SD
        d_matches = (output_d * SD + kd == input_d + PD) & (output_d >= 0) & (output_d < OUT_D)
        start_d = output_d * SD - PD
        end_d = tl.minimum(start_d + KD, D + PD)
        for kh in tl.static_range(0, KH):
            output_h = (input_h + PH - kh) // SH
            h_matches = (output_h * SH + kh == input_h + PH) & (output_h >= 0) & (output_h < OUT_H)
            start_h = output_h * SH - PH
            end_h = tl.minimum(start_h + KH, H + PH)
            for kw in tl.static_range(0, KW):
                output_w = (input_w + PW - kw) // SW
                w_matches = (output_w * SW + kw == input_w + PW) & (output_w >= 0) & (output_w < OUT_W)
                start_w = output_w * SW - PW
                end_w = tl.minimum(start_w + KW, W + PW)
                output_offset = (
                    ((channel_batch * OUT_D + output_d) * OUT_H + output_h) * OUT_W
                    + output_w
                )
                matches = valid & d_matches & h_matches & w_matches
                value = tl.load(grad_output_ptr + output_offset, mask=matches, other=0.0).to(tl.float32)
                pool_d = tl.zeros((BLOCK,), tl.int32)
                valid_d = tl.zeros((BLOCK,), tl.int32)
                for pool_kd in tl.static_range(0, KD):
                    position_d = start_d + pool_kd
                    pool_d += tl.where(position_d < D + PD, 1, 0)
                    valid_d += tl.where((position_d >= 0) & (position_d < D), 1, 0)
                pool_h = tl.zeros((BLOCK,), tl.int32)
                valid_h = tl.zeros((BLOCK,), tl.int32)
                for pool_kh in tl.static_range(0, KH):
                    position_h = start_h + pool_kh
                    pool_h += tl.where(position_h < H + PH, 1, 0)
                    valid_h += tl.where((position_h >= 0) & (position_h < H), 1, 0)
                pool_w = tl.zeros((BLOCK,), tl.int32)
                valid_w = tl.zeros((BLOCK,), tl.int32)
                for pool_kw in tl.static_range(0, KW):
                    position_w = start_w + pool_kw
                    pool_w += tl.where(position_w < W + PW, 1, 0)
                    valid_w += tl.where((position_w >= 0) & (position_w < W), 1, 0)
                pool_size = pool_d * pool_h * pool_w
                valid_size = valid_d * valid_h * valid_w
                if HAS_DIVISOR:
                    divisor = DIVISOR
                elif COUNT_INCLUDE_PAD:
                    divisor = pool_size
                else:
                    divisor = valid_size
                accumulator += tl.fdiv(value, divisor.to(tl.float32))

    tl.store(grad_input_ptr + offsets, accumulator, mask=valid)


def avg_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    was_unbatched = input.ndim == 4
    x = input.unsqueeze(0) if was_unbatched else input
    if x.ndim != 5:
        raise ValueError("avg_pool3d_backward expects a 4D or 5D input tensor")

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
    expected_shape = (batch, channels, output_d, output_h, output_w)
    grad_output = grad_output.unsqueeze(0) if was_unbatched else grad_output
    if tuple(grad_output.shape) != expected_shape:
        raise ValueError(f"grad_output has shape {tuple(grad_output.shape)}, expected {expected_shape}")

    grad_input = torch.empty_like(x)
    total = grad_input.numel()
    if total:
        block = 128
        divisor = 0 if divisor_override is None else divisor_override
        with torch_device_fn.device(input.device):
            _avg_pool3d_backward_gather_kernel[(triton.cdiv(total, block),)](
                grad_output.contiguous(),
                grad_input,
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
            )
    return grad_input.squeeze(0) if was_unbatched else grad_input
