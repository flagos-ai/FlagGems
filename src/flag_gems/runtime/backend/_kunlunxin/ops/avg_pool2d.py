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
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["out_h", "out_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def avg_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # AvgPool specific parameters
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=0.0
            )

            sum_acc += tl.where(in_mask, current_val, 0.0)
            count_acc += in_mask.to(tl.int32)

    count_divisor = count_acc.to(tl.float32)

    if COUNT_INCLUDE_PAD:
        default_divisor = tl.where(
            count_divisor >= 0, float(kernel_h * kernel_w), count_divisor
        )
    else:
        default_divisor = count_divisor

    divisor = tl.where(
        divisor_override != 0, divisor_override + default_divisor * 0, default_divisor
    )

    output_vals = tl.where(divisor != 0, sum_acc / divisor, 0.0)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(
        output_block_ptr, output_vals.to(output_ptr.type.element_ty), mask=out_mask
    )


@libentry()
@triton.jit
def pool2d_input_grad_kernel(
    grad_output_ptr,
    grad_input_ptr,
    input_numel,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    grad_output_stride_n,
    grad_output_stride_c,
    grad_output_stride_h,
    grad_output_stride_w,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offset < input_numel

    input_w_index = input_offset % in_w
    input_offset_hc = input_offset // in_w
    input_h_index = input_offset_hc % in_h
    input_offset_nc = input_offset_hc // in_h
    input_c_index = input_offset_nc % in_c
    input_n_index = input_offset_nc // in_c

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
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

            output_mask = input_mask & out_h_valid & out_w_valid
            safe_out_h = tl.where(output_mask, out_h_index, 0)
            safe_out_w = tl.where(output_mask, out_w_index, 0)

            if DIVISOR_OVERRIDE != 0:
                divisor = tl.full((BLOCK_SIZE,), DIVISOR_OVERRIDE, tl.float32)
            else:
                input_h_start = safe_out_h * STRIDE_H - PADDING_H
                input_w_start = safe_out_w * STRIDE_W - PADDING_W
                if COUNT_INCLUDE_PAD:
                    count_h = tl.minimum(
                        input_h_start + KERNEL_H, in_h + PADDING_H
                    ) - tl.maximum(input_h_start, -PADDING_H)
                    count_w = tl.minimum(
                        input_w_start + KERNEL_W, in_w + PADDING_W
                    ) - tl.maximum(input_w_start, -PADDING_W)
                else:
                    count_h = tl.minimum(input_h_start + KERNEL_H, in_h) - tl.maximum(
                        input_h_start, 0
                    )
                    count_w = tl.minimum(input_w_start + KERNEL_W, in_w) - tl.maximum(
                        input_w_start, 0
                    )
                count_h = tl.maximum(count_h, 0)
                count_w = tl.maximum(count_w, 0)
                divisor = (count_h * count_w).to(tl.float32)

            grad_output_offset = (
                input_n_index * grad_output_stride_n
                + input_c_index * grad_output_stride_c
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
        in_n * in_c,
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    avg_pool2d_forward_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
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
        dilation_h,
        dilation_w,
        COUNT_INCLUDE_PAD=count_include_pad,
        divisor_override=divisor_override if divisor_override is not None else 0.0,
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

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )

    if input.ndim not in (3, 4):
        raise RuntimeError("avg_pool2d_backward expects a 3D or 4D input")

    unbatched = input.ndim == 3
    input_4d = input.unsqueeze(0) if unbatched else input
    grad_output_4d = grad_output.unsqueeze(0) if unbatched else grad_output

    _, in_c, in_h, in_w = input_4d.shape
    out_h, out_w = grad_output_4d.shape[2], grad_output_4d.shape[3]
    input_numel = input_4d.numel()
    grad_input_4d = torch.empty(
        input_4d.shape, device=input.device, dtype=torch.float32
    )

    if input_numel != 0:
        block_size = 2048
        grid = (triton.cdiv(input_numel, block_size),)
        pool2d_input_grad_kernel[grid](
            grad_output_4d,
            grad_input_4d,
            input_numel,
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            grad_output_4d.stride(0),
            grad_output_4d.stride(1),
            grad_output_4d.stride(2),
            grad_output_4d.stride(3),
            KERNEL_H=kernel_h,
            KERNEL_W=kernel_w,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PADDING_H=padding_h,
            PADDING_W=padding_w,
            COUNT_INCLUDE_PAD=count_include_pad,
            DIVISOR_OVERRIDE=divisor_override or 0,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )

    grad_input_4d = grad_input_4d.to(grad_output.dtype)
    return grad_input_4d.squeeze(0) if unbatched else grad_input_4d
