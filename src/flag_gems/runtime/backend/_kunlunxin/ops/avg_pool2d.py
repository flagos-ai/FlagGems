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


def pool2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.jit
def avg_pool2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    numel,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = offsets < numel

    w_in = offsets % in_w
    remaining = offsets // in_w
    h_in = remaining % in_h
    remaining = remaining // in_h
    c_idx = remaining % in_c
    n_idx = remaining // in_c

    grad_output_base_ptr = (
        grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    )
    grad_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_out_num = h_in + padding_h - kh * dilation_h
            w_out_num = w_in + padding_w - kw * dilation_w
            h_valid_map = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
            w_valid_map = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)

            h_out = h_out_num // stride_h
            w_out = w_out_num // stride_w
            out_mask = (
                input_mask
                & h_valid_map
                & w_valid_map
                & (h_out < out_h)
                & (w_out < out_w)
            )

            h_start = h_out * stride_h - padding_h
            w_start = w_out * stride_w - padding_w
            if COUNT_INCLUDE_PAD:
                h_lower, h_upper = -padding_h, in_h + padding_h
                w_lower, w_upper = -padding_w, in_w + padding_w
            else:
                h_lower, h_upper = 0, in_h
                w_lower, w_upper = 0, in_w

            h_first = (h_lower - h_start + dilation_h - 1) // dilation_h
            h_last = (h_upper - h_start + dilation_h - 1) // dilation_h
            w_first = (w_lower - w_start + dilation_w - 1) // dilation_w
            w_last = (w_upper - w_start + dilation_w - 1) // dilation_w
            h_first = tl.maximum(h_first, 0)
            h_last = tl.minimum(h_last, kernel_h)
            w_first = tl.maximum(w_first, 0)
            w_last = tl.minimum(w_last, kernel_w)
            h_count = tl.maximum(h_last - h_first, 0)
            w_count = tl.maximum(w_last - w_first, 0)
            default_divisor = (h_count * w_count).to(tl.float32)
            divisor = tl.where(
                divisor_override != 0,
                divisor_override + default_divisor * 0,
                default_divisor,
            )
            divisor = tl.where(divisor == 0, 1.0, divisor)

            safe_h_out = tl.where(out_mask, h_out, 0)
            safe_w_out = tl.where(out_mask, w_out, 0)
            grad_out_ptr = (
                grad_output_base_ptr
                + safe_h_out * out_stride_h
                + safe_w_out * out_stride_w
            )
            grad_out = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
            grad_acc += tl.where(out_mask, grad_out / divisor, 0.0)

    grad_input_ptrs = (
        grad_input_ptr
        + n_idx * in_stride_n
        + c_idx * in_stride_c
        + h_in * in_stride_h
        + w_in * in_stride_w
    )
    tl.store(
        grad_input_ptrs,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=input_mask,
    )


@libentry()
@triton.jit
def avg_pool2d_backward_plane_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_c = tl.program_id(1)
    input_mask = offsets < in_h * in_w
    h_in = offsets // in_w
    w_in = offsets % in_w

    n_idx = n_c // in_c
    c_idx = n_c % in_c
    grad_output_base_ptr = (
        grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    )
    grad_input_base_ptr = (
        grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    )
    grad_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_out_num = h_in + padding_h - kh * dilation_h
            w_out_num = w_in + padding_w - kw * dilation_w
            h_valid_map = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
            w_valid_map = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)
            h_out = h_out_num // stride_h
            w_out = w_out_num // stride_w
            out_mask = (
                input_mask
                & h_valid_map
                & w_valid_map
                & (h_out < out_h)
                & (w_out < out_w)
            )

            h_start = h_out * stride_h - padding_h
            w_start = w_out * stride_w - padding_w
            if COUNT_INCLUDE_PAD:
                h_lower, h_upper = -padding_h, in_h + padding_h
                w_lower, w_upper = -padding_w, in_w + padding_w
            else:
                h_lower, h_upper = 0, in_h
                w_lower, w_upper = 0, in_w

            h_first = (h_lower - h_start + dilation_h - 1) // dilation_h
            h_last = (h_upper - h_start + dilation_h - 1) // dilation_h
            w_first = (w_lower - w_start + dilation_w - 1) // dilation_w
            w_last = (w_upper - w_start + dilation_w - 1) // dilation_w
            h_first = tl.maximum(h_first, 0)
            h_last = tl.minimum(h_last, kernel_h)
            w_first = tl.maximum(w_first, 0)
            w_last = tl.minimum(w_last, kernel_w)
            default_divisor = ((h_last - h_first) * (w_last - w_first)).to(
                tl.float32
            )
            divisor = tl.where(
                divisor_override != 0,
                divisor_override + default_divisor * 0,
                default_divisor,
            )
            divisor = tl.where(divisor == 0, 1.0, divisor)

            safe_h_out = tl.where(out_mask, h_out, 0)
            safe_w_out = tl.where(out_mask, w_out, 0)
            grad_out_ptr = (
                grad_output_base_ptr
                + safe_h_out * out_stride_h
                + safe_w_out * out_stride_w
            )
            grad_out = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
            grad_acc += tl.where(out_mask, grad_out / divisor, 0.0)

    grad_input_ptrs = grad_input_base_ptr + h_in * in_stride_h + w_in * in_stride_w
    tl.store(
        grad_input_ptrs,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=input_mask,
    )


@libentry()
@triton.jit
def avg_pool2d_forward_flat_kernel(
    input_ptr,
    output_ptr,
    numel,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < numel

    w_out = offsets % out_w
    remaining = offsets // out_w
    h_out = remaining % out_h
    remaining = remaining // out_h
    c_idx = remaining % in_c
    n_idx = remaining // in_c

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    output_ptrs = (
        output_ptr
        + n_idx * out_stride_n
        + c_idx * out_stride_c
        + h_out * out_stride_h
        + w_out * out_stride_w
    )
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out * stride_h - padding_h + kh * dilation_h
            w_in = w_out * stride_w - padding_w + kw * dilation_w
            in_mask = (
                output_mask
                & (h_in >= 0)
                & (h_in < in_h)
                & (w_in >= 0)
                & (w_in < in_w)
            )
            padded_mask = (
                output_mask
                & (h_in >= -padding_h)
                & (h_in < in_h + padding_h)
                & (w_in >= -padding_w)
                & (w_in < in_w + padding_w)
            )
            safe_h_in = tl.where(in_mask, h_in, 0)
            safe_w_in = tl.where(in_mask, w_in, 0)
            input_ptrs = input_base_ptr + safe_h_in * in_stride_h + safe_w_in * in_stride_w
            value = tl.load(input_ptrs, mask=in_mask, other=0.0)
            sum_acc += tl.where(in_mask, value, 0.0)
            count_acc += tl.where(
                COUNT_INCLUDE_PAD, padded_mask, in_mask
            ).to(tl.int32)

    divisor = count_acc.to(tl.float32)
    divisor = tl.where(
        divisor_override != 0,
        divisor_override + divisor * 0,
        divisor,
    )
    divisor = tl.where(divisor == 0, 1.0, divisor)
    result = sum_acc / divisor
    tl.store(
        output_ptrs,
        result.to(output_ptr.type.element_ty),
        mask=output_mask,
    )


def _parse_pool_params(kernel_size, stride, padding):
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if stride is None or (isinstance(stride, (list, tuple)) and not stride):
        stride_h, stride_w = kernel_h, kernel_w
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride must be greater than zero")

    if padding_h < 0 or padding_w < 0:
        raise ValueError("padding must be non-negative")

    if padding_h > kernel_h // 2 or padding_w > kernel_w // 2:
        raise ValueError("pad should be smaller than or equal to half of kernel size")

    return kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w


@libentry()
@triton.jit
def avg_pool2d_forward_plane_kernel(
    input_ptr,
    output_ptr,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_c = tl.program_id(1)
    h_out = offsets // out_w
    w_out = offsets % out_w
    output_mask = offsets < out_h * out_w

    h_in_base = h_out * stride_h - padding_h
    w_in_base = w_out * stride_w - padding_w
    input_base = n_c * in_h * in_w
    output_base = n_c * out_h * out_w

    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for kh in tl.static_range(kernel_h):
        h_in = h_in_base + kh
        h_valid = (h_in >= 0) & (h_in < in_h)
        for kw in tl.static_range(kernel_w):
            w_in = w_in_base + kw
            w_valid = (w_in >= 0) & (w_in < in_w)
            input_mask = output_mask & h_valid & w_valid
            safe_h_in = tl.where(input_mask, h_in, 0)
            safe_w_in = tl.where(input_mask, w_in, 0)
            value = tl.load(
                input_ptr + input_base + safe_h_in * in_w + safe_w_in,
                mask=input_mask,
                other=0.0,
            )
            sum_acc += tl.where(input_mask, value, 0.0)
            count_acc += tl.where(
                COUNT_INCLUDE_PAD,
                output_mask,
                input_mask,
            ).to(tl.int32)

    divisor = count_acc.to(tl.float32)
    divisor = tl.where(
        divisor_override != 0,
        divisor_override + divisor * 0,
        divisor,
    )
    divisor = tl.where(divisor == 0, 1.0, divisor)
    tl.store(
        output_ptr + output_base + offsets,
        (sum_acc / divisor).to(output_ptr.type.element_ty),
        mask=output_mask,
    )


def avg_pool2d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    logger.debug("GEMS_KUNLUNXIN AVG_POOL2D")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    input = input.contiguous()

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )
    dilation_h, dilation_w = 1, 1

    in_n, in_c, in_h, in_w = input.shape

    out_h = pool2d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = pool2d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )

    if output.numel() == 0:
        return output

    grid = lambda meta: (
        triton.cdiv(out_h * out_w, meta["BLOCK_SIZE"]),
        in_n * in_c,
    )

    avg_pool2d_forward_plane_kernel[grid](
        input,
        output,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        COUNT_INCLUDE_PAD=count_include_pad,
        divisor_override=divisor_override if divisor_override is not None else 0.0,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    return output


def avg_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    logger.debug("GEMS_KUNLUNXIN AVG_POOL2D_BACKWARD")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    grad_output = grad_output.contiguous()

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )
    dilation_h, dilation_w = 1, 1

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_output.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        triton.cdiv(in_h * in_w, meta["BLOCK_SIZE"]),
        in_n * in_c,
    )

    avg_pool2d_backward_plane_kernel[grid](
        grad_output,
        grad_input,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        COUNT_INCLUDE_PAD=count_include_pad,
        divisor_override=divisor_override if divisor_override is not None else 0.0,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    return grad_input.to(grad_output.dtype)
