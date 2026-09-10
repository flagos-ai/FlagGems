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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _pool_output_size(input_size, kernel_size, stride, padding, ceil_mode):
    numerator = input_size + 2 * padding - kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= input_size + padding:
            output_size -= 1
        return output_size
    return numerator // stride + 1


def _triple(value, name, default=None):
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        value = default
    if isinstance(value, int):
        return value, value, value
    if len(value) == 1:
        return value[0], value[0], value[0]
    if len(value) == 3:
        return value[0], value[1], value[2]
    raise ValueError(f"{name} must be a single int or a tuple of three ints")


def _parse_pool3d_params(kernel_size, stride, padding):
    kernel_d, kernel_h, kernel_w = _triple(kernel_size, "kernel_size")
    stride_d, stride_h, stride_w = _triple(stride, "stride", kernel_size)
    padding_d, padding_h, padding_w = _triple(padding, "padding")

    if min(kernel_d, kernel_h, kernel_w) <= 0:
        raise ValueError("kernel_size must be greater than zero")
    if min(stride_d, stride_h, stride_w) <= 0:
        raise ValueError("stride must be greater than zero")
    if min(padding_d, padding_h, padding_w) < 0:
        raise ValueError("padding must be non-negative")
    if (
        padding_d > kernel_d // 2
        or padding_h > kernel_h // 2
        or padding_w > kernel_w // 2
    ):
        raise ValueError("pad should be smaller than or equal to half of kernel size")

    return (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    )


@libentry()
@triton.jit
def _avg_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    output_numel,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    input_stride_n,
    input_stride_c,
    input_stride_d,
    input_stride_h,
    input_stride_w,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_D: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offset < output_numel

    out_w_index = output_offset % out_w
    output_offset_dh = output_offset // out_w
    out_h_index = output_offset_dh % out_h
    output_offset_dc = output_offset_dh // out_h
    out_d_index = output_offset_dc % out_d
    output_offset_cn = output_offset_dc // out_d
    channel_index = output_offset_cn % in_c
    batch_index = output_offset_cn // in_c

    input_d_start = out_d_index * STRIDE_D - PADDING_D
    input_h_start = out_h_index * STRIDE_H - PADDING_H
    input_w_start = out_w_index * STRIDE_W - PADDING_W

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    valid_count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for kernel_d_index in tl.static_range(0, KERNEL_D):
        input_d_index = input_d_start + kernel_d_index
        input_d_valid = (input_d_index >= 0) & (input_d_index < in_d)
        for kernel_h_index in tl.static_range(0, KERNEL_H):
            input_h_index = input_h_start + kernel_h_index
            input_h_valid = (input_h_index >= 0) & (input_h_index < in_h)
            for kernel_w_index in tl.static_range(0, KERNEL_W):
                input_w_index = input_w_start + kernel_w_index
                input_w_valid = (input_w_index >= 0) & (input_w_index < in_w)
                input_mask = output_mask & input_d_valid & input_h_valid & input_w_valid

                safe_d_index = tl.where(input_mask, input_d_index, 0)
                safe_h_index = tl.where(input_mask, input_h_index, 0)
                safe_w_index = tl.where(input_mask, input_w_index, 0)
                input_offset = (
                    batch_index * input_stride_n
                    + channel_index * input_stride_c
                    + safe_d_index * input_stride_d
                    + safe_h_index * input_stride_h
                    + safe_w_index * input_stride_w
                )
                value = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
                accumulator += tl.where(input_mask, value, 0.0)
                valid_count += input_mask.to(tl.int32)

    if DIVISOR_OVERRIDE != 0:
        divisor = tl.full((BLOCK_SIZE,), DIVISOR_OVERRIDE, tl.float32)
    elif COUNT_INCLUDE_PAD:
        padded_d_count = tl.minimum(
            input_d_start + KERNEL_D, in_d + PADDING_D
        ) - tl.maximum(input_d_start, -PADDING_D)
        padded_h_count = tl.minimum(
            input_h_start + KERNEL_H, in_h + PADDING_H
        ) - tl.maximum(input_h_start, -PADDING_H)
        padded_w_count = tl.minimum(
            input_w_start + KERNEL_W, in_w + PADDING_W
        ) - tl.maximum(input_w_start, -PADDING_W)
        padded_d_count = tl.maximum(padded_d_count, 0)
        padded_h_count = tl.maximum(padded_h_count, 0)
        padded_w_count = tl.maximum(padded_w_count, 0)
        divisor = (padded_d_count * padded_h_count * padded_w_count).to(tl.float32)
    else:
        divisor = valid_count.to(tl.float32)

    result = tl.where(divisor != 0, accumulator / divisor, 0.0)
    tl.store(output_ptr + output_offset, result, mask=output_mask)


@libentry()
@triton.jit
def _avg_pool3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    input_numel,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    grad_output_stride_n,
    grad_output_stride_c,
    grad_output_stride_d,
    grad_output_stride_h,
    grad_output_stride_w,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_D: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offset < input_numel

    input_w_index = input_offset % in_w
    input_offset_dh = input_offset // in_w
    input_h_index = input_offset_dh % in_h
    input_offset_dc = input_offset_dh // in_h
    input_d_index = input_offset_dc % in_d
    input_offset_cn = input_offset_dc // in_d
    channel_index = input_offset_cn % in_c
    batch_index = input_offset_cn // in_c

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kernel_d_index in tl.static_range(0, KERNEL_D):
        out_d_numerator = input_d_index + PADDING_D - kernel_d_index
        out_d_valid = (out_d_numerator >= 0) & (out_d_numerator % STRIDE_D == 0)
        out_d_index = out_d_numerator // STRIDE_D
        out_d_valid &= (out_d_index >= 0) & (out_d_index < out_d)

        for kernel_h_index in tl.static_range(0, KERNEL_H):
            out_h_numerator = input_h_index + PADDING_H - kernel_h_index
            out_h_valid = (out_h_numerator >= 0) & (out_h_numerator % STRIDE_H == 0)
            out_h_index = out_h_numerator // STRIDE_H
            out_h_valid &= (out_h_index >= 0) & (out_h_index < out_h)

            for kernel_w_index in tl.static_range(0, KERNEL_W):
                out_w_numerator = input_w_index + PADDING_W - kernel_w_index
                out_w_valid = (out_w_numerator >= 0) & (out_w_numerator % STRIDE_W == 0)
                out_w_index = out_w_numerator // STRIDE_W
                out_w_valid &= (out_w_index >= 0) & (out_w_index < out_w)

                output_mask = input_mask & out_d_valid & out_h_valid & out_w_valid
                safe_out_d = tl.where(output_mask, out_d_index, 0)
                safe_out_h = tl.where(output_mask, out_h_index, 0)
                safe_out_w = tl.where(output_mask, out_w_index, 0)

                if DIVISOR_OVERRIDE != 0:
                    divisor = tl.full((BLOCK_SIZE,), DIVISOR_OVERRIDE, tl.float32)
                else:
                    input_d_start = safe_out_d * STRIDE_D - PADDING_D
                    input_h_start = safe_out_h * STRIDE_H - PADDING_H
                    input_w_start = safe_out_w * STRIDE_W - PADDING_W
                    if COUNT_INCLUDE_PAD:
                        count_d = tl.minimum(
                            input_d_start + KERNEL_D, in_d + PADDING_D
                        ) - tl.maximum(input_d_start, -PADDING_D)
                        count_h = tl.minimum(
                            input_h_start + KERNEL_H, in_h + PADDING_H
                        ) - tl.maximum(input_h_start, -PADDING_H)
                        count_w = tl.minimum(
                            input_w_start + KERNEL_W, in_w + PADDING_W
                        ) - tl.maximum(input_w_start, -PADDING_W)
                    else:
                        count_d = tl.minimum(
                            input_d_start + KERNEL_D, in_d
                        ) - tl.maximum(input_d_start, 0)
                        count_h = tl.minimum(
                            input_h_start + KERNEL_H, in_h
                        ) - tl.maximum(input_h_start, 0)
                        count_w = tl.minimum(
                            input_w_start + KERNEL_W, in_w
                        ) - tl.maximum(input_w_start, 0)
                    count_d = tl.maximum(count_d, 0)
                    count_h = tl.maximum(count_h, 0)
                    count_w = tl.maximum(count_w, 0)
                    divisor = (count_d * count_h * count_w).to(tl.float32)
                    divisor = tl.where(divisor == 0, 1.0, divisor)

                grad_output_offset = (
                    batch_index * grad_output_stride_n
                    + channel_index * grad_output_stride_c
                    + safe_out_d * grad_output_stride_d
                    + safe_out_h * grad_output_stride_h
                    + safe_out_w * grad_output_stride_w
                )
                grad_output = tl.load(
                    grad_output_ptr + grad_output_offset,
                    mask=output_mask,
                    other=0.0,
                )
                accumulator += tl.where(output_mask, grad_output / divisor, 0.0)

    tl.store(grad_input_ptr + input_offset, accumulator, mask=input_mask)


def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    logger.debug("GEMS KUNLUNXIN AVG_POOL3D")
    if input.ndim not in (4, 5):
        raise RuntimeError("non-empty 4D or 5D tensor expected for input")
    if divisor_override == 0:
        raise RuntimeError("divisor must be not zero")

    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    ) = _parse_pool3d_params(kernel_size, stride, padding)

    unbatched = input.ndim == 4
    input_5d = input.unsqueeze(0) if unbatched else input
    in_n, in_c, in_d, in_h, in_w = input_5d.shape
    out_d = _pool_output_size(in_d, kernel_d, stride_d, padding_d, ceil_mode)
    out_h = _pool_output_size(in_h, kernel_h, stride_h, padding_h, ceil_mode)
    out_w = _pool_output_size(in_w, kernel_w, stride_w, padding_w, ceil_mode)
    output_5d = torch.empty(
        (in_n, in_c, out_d, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )
    if output_5d.numel() != 0:
        block_size = 256
        _avg_pool3d_forward_kernel[(triton.cdiv(output_5d.numel(), block_size),)](
            input_5d,
            output_5d,
            output_5d.numel(),
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            input_5d.stride(0),
            input_5d.stride(1),
            input_5d.stride(2),
            input_5d.stride(3),
            input_5d.stride(4),
            KERNEL_D=kernel_d,
            KERNEL_H=kernel_h,
            KERNEL_W=kernel_w,
            STRIDE_D=stride_d,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PADDING_D=padding_d,
            PADDING_H=padding_h,
            PADDING_W=padding_w,
            COUNT_INCLUDE_PAD=count_include_pad,
            DIVISOR_OVERRIDE=divisor_override or 0,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return output_5d.squeeze(0) if unbatched else output_5d


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
    logger.debug("GEMS KUNLUNXIN AVG_POOL3D_BACKWARD")
    if input.ndim not in (4, 5):
        raise RuntimeError("non-empty 4D or 5D tensor expected for input")
    if divisor_override == 0:
        raise RuntimeError("divisor must be not zero")

    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    ) = _parse_pool3d_params(kernel_size, stride, padding)

    unbatched = input.ndim == 4
    input_5d = input.unsqueeze(0) if unbatched else input
    grad_output_5d = grad_output.unsqueeze(0) if unbatched else grad_output
    in_n, in_c, in_d, in_h, in_w = input_5d.shape
    _, _, out_d, out_h, out_w = grad_output_5d.shape
    grad_input_5d = torch.empty(input_5d.shape, device=input.device, dtype=input.dtype)

    if grad_input_5d.numel() != 0:
        block_size = 256
        _avg_pool3d_backward_kernel[(triton.cdiv(grad_input_5d.numel(), block_size),)](
            grad_output_5d,
            grad_input_5d,
            grad_input_5d.numel(),
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            grad_output_5d.stride(0),
            grad_output_5d.stride(1),
            grad_output_5d.stride(2),
            grad_output_5d.stride(3),
            grad_output_5d.stride(4),
            KERNEL_D=kernel_d,
            KERNEL_H=kernel_h,
            KERNEL_W=kernel_w,
            STRIDE_D=stride_d,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PADDING_D=padding_d,
            PADDING_H=padding_h,
            PADDING_W=padding_w,
            COUNT_INCLUDE_PAD=count_include_pad,
            DIVISOR_OVERRIDE=divisor_override or 0,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return grad_input_5d.squeeze(0) if unbatched else grad_input_5d
