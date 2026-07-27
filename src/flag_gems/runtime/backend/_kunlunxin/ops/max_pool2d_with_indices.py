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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool2d_output_size(
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
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.jit
def max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
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
    # Meta-parameters for tiling
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

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            h_safe = tl.where(in_mask, h_in, 0)
            w_safe = tl.where(in_mask, w_in, 0)
            input_offset = h_safe * in_stride_h + w_safe * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=min_val
            )
            current_val = tl.where(in_mask, current_val, min_val)
            current_idx = h_safe * in_w + w_safe

            is_new_max = current_val > max_val_acc
            max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
            max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.jit
def max_pool2d_forward_flat_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    total,
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
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total
    out_hw = out_h * out_w
    nc_idx = offsets // out_hw
    rem = offsets % out_hw
    oh = rem // out_w
    ow = rem % out_w
    nc_safe = tl.where(output_mask, nc_idx, 0)

    max_val = tl.full((BLOCK,), float("-inf"), tl.float32)
    max_idx = tl.full((BLOCK,), -1, tl.int64)
    for kh in tl.static_range(kernel_h):
        for kw in tl.static_range(kernel_w):
            ih = oh * stride_h - padding_h + kh * dilation_h
            iw = ow * stride_w - padding_w + kw * dilation_w
            valid = output_mask & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)
            input_offset = nc_safe * (in_h * in_w) + ih_safe * in_w + iw_safe
            value = tl.load(input_ptr + input_offset, mask=valid, other=float("-inf"))
            value = tl.where(valid, value.to(tl.float32), float("-inf"))
            is_new_max = valid & (value > max_val)
            max_val = tl.where(is_new_max, value, max_val)
            max_idx = tl.where(is_new_max, ih_safe * in_w + iw_safe, max_idx)

    tl.store(output_ptr + offsets, max_val, mask=output_mask)
    tl.store(indices_ptr + offsets, max_idx, mask=output_mask)


@libentry()
@triton.jit
def max_pool2d_backward_flat_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    total,
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
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    input_mask = offsets < total
    in_hw = in_h * in_w
    nc_idx = offsets // in_hw
    rem = offsets % in_hw
    ih = rem // in_w
    iw = rem % in_w
    nc_safe = tl.where(input_mask, nc_idx, 0)
    input_flat_idx = ih * in_w + iw

    grad_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in tl.static_range(kernel_h):
        for kw in tl.static_range(kernel_w):
            h_num = ih + padding_h - kh * dilation_h
            w_num = iw + padding_w - kw * dilation_w
            h_nonnegative = h_num >= 0
            w_nonnegative = w_num >= 0
            h_num_safe = tl.where(h_nonnegative, h_num, 0)
            w_num_safe = tl.where(w_nonnegative, w_num, 0)
            oh = h_num_safe // stride_h
            ow = w_num_safe // stride_w
            h_rem = h_num_safe - oh * stride_h
            w_rem = w_num_safe - ow * stride_w
            valid = (
                input_mask
                & h_nonnegative
                & w_nonnegative
                & (h_rem == 0)
                & (w_rem == 0)
                & (oh < out_h)
                & (ow < out_w)
            )
            oh_safe = tl.where(valid, oh, 0)
            ow_safe = tl.where(valid, ow, 0)
            out_offset = nc_safe * (out_h * out_w) + oh_safe * out_w + ow_safe
            index_value = tl.load(indices_ptr + out_offset, mask=valid, other=-1)
            index_value = tl.where(valid, index_value, -1)
            match = valid & (index_value == input_flat_idx)
            grad_value = tl.load(grad_output_ptr + out_offset, mask=match, other=0.0)
            grad_acc += tl.where(match, grad_value.to(tl.float32), 0.0)

    tl.store(grad_input_ptr + offsets, grad_acc, mask=input_mask)


@libentry()
@triton.jit
def max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Strides for grad_output/indices
    out_stride_nc,
    out_stride_h,
    out_stride_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Tiling parameters
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    current_input_flat_idx = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    grad_acc = tl.zeros((BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_stride_nc

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            numerator_h = h_in_offsets[:, None] + padding_h - kh * dilation_h
            numerator_w = w_in_offsets[None, :] + padding_w - kw * dilation_w

            valid_map_mask = (numerator_h % stride_h == 0) & (
                numerator_w % stride_w == 0
            )
            h_out = numerator_h // stride_h
            w_out = numerator_w // stride_w
            out_bounds_mask = (
                (h_out >= 0) & (h_out < out_h) & (w_out >= 0) & (w_out < out_w)
            )
            load_mask = valid_map_mask & out_bounds_mask

            safe_h_out = tl.where(load_mask, h_out, 0)
            safe_w_out = tl.where(load_mask, w_out, 0)
            out_offsets = safe_h_out * out_stride_h + safe_w_out

            indices_block = tl.load(
                indices_base_ptr + out_offsets, mask=load_mask, other=-1
            )
            match_mask = indices_block == current_input_flat_idx

            grad_block = tl.load(
                grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
            )
            grad_acc += grad_block

    grad_input_base_ptr = grad_input_ptr + nc_idx * in_h * in_w
    grad_input_offsets = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    store_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def _parse_pool_params(kernel_size, stride, padding, dilation):
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_h, stride_w = _parse_param(stride, "stride", default=(kernel_h, kernel_w))
    padding_h, padding_w = _parse_param(padding, "padding", default=(0, 0))
    dilation_h, dilation_w = _parse_param(dilation, "dilation", default=(1, 1))

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got stride=({stride_h}, {stride_w})"
        )
    if padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({padding_h}, {padding_w})"
        )
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dilation_h}, {dilation_w})"
        )

    return (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )


def max_pool2d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL2D_WITH_INDICES")
    input = input.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h = max_pool2d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool2d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int32
    )

    if output.numel() == 0:
        return output, indices

    total = output.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)

    with torch_device_fn.device(input.device):
        max_pool2d_forward_flat_kernel[grid](
            input,
            output,
            indices,
            total,
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
            block,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return output, indices


def max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL2D_BACKWARD")
    original_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32).contiguous()
    indices = indices.to(torch.int32).contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(original_dtype)

    total = grad_input.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)

    with torch_device_fn.device(grad_input.device):
        max_pool2d_backward_flat_kernel[grid](
            grad_output,
            indices,
            grad_input,
            total,
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
            block,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return grad_input.to(original_dtype)
