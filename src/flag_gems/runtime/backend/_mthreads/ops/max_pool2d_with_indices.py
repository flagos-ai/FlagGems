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
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1
    return output_size


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 2, "BLOCK_HW": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 2, "BLOCK_HW": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 2, "BLOCK_HW": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_stages=2, num_warps=8),
    ],
    key=[
        "total_channels",
        "out_hw",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    ],
)
@triton.jit
def max_pool2d_channel_packed_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    total_channels,
    in_h,
    in_w,
    out_w,
    out_hw,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    c_mask = c_offsets < total_channels
    hw_mask = hw_offsets < out_hw

    h_out = hw_offsets // out_w
    w_out = hw_offsets % out_w

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_C, BLOCK_HW), min_val, dtype=tl.float32)
    best_k = tl.full((BLOCK_C, BLOCK_HW), -1, dtype=tl.int32)

    in_hw_size = in_h * in_w
    c_safe = tl.minimum(c_offsets, total_channels - 1)
    c_base_offset = c_safe[:, None] * in_hw_size

    h_base = h_out * stride_h - padding_h
    w_base = w_out * stride_w - padding_w

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_base + kh * dilation_h
            w_in = w_base + kw * dilation_w

            h_in_safe = tl.maximum(0, tl.minimum(h_in, in_h - 1))
            w_in_safe = tl.maximum(0, tl.minimum(w_in, in_w - 1))
            spatial_offset_safe = h_in_safe * in_w + w_in_safe

            input_offset = c_base_offset + spatial_offset_safe[None, :]

            raw_val_f32 = tl.load(input_ptr + input_offset).to(tl.float32)

            valid_spatial = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

            is_new_max = (raw_val_f32 > max_val_acc) & valid_spatial[None, :]
            max_val_acc = tl.where(is_new_max, raw_val_f32, max_val_acc)
            best_k = tl.where(is_new_max, kh * kernel_w + kw, best_k)

    kh_best = best_k // kernel_w
    kw_best = best_k % kernel_w
    best_h_in = h_base[None, :] + kh_best * dilation_h
    best_w_in = w_base[None, :] + kw_best * dilation_w
    max_idx_acc = tl.where(best_k >= 0, (best_h_in * in_w + best_w_in).to(tl.int32), -1)

    out_base_offset = c_offsets[:, None] * out_hw + hw_offsets[None, :]
    store_mask = c_mask[:, None] & hw_mask[None, :]
    tl.store(
        output_ptr + out_base_offset,
        max_val_acc.to(dtype),
        mask=store_mask,
    )
    tl.store(indices_ptr + out_base_offset, max_idx_acc.to(tl.int64), mask=store_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=2, num_warps=8),
    ],
    key=[
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    ],
)
@triton.jit
def max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
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
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
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
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=tl.float32)
    best_k = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    h_base = h_out_offsets[:, None] * stride_h - padding_h
    w_base = w_out_offsets[None, :] * stride_w - padding_w

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_base + kh * dilation_h
            w_in = w_base + kw * dilation_w

            h_in_safe = tl.maximum(0, tl.minimum(h_in, in_h - 1))
            w_in_safe = tl.maximum(0, tl.minimum(w_in, in_w - 1))
            safe_offset = h_in_safe * in_stride_h + w_in_safe * in_stride_w

            raw_val_f32 = tl.load(input_base_ptr + safe_offset).to(tl.float32)

            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

            is_new_max = (raw_val_f32 > max_val_acc) & in_mask
            max_val_acc = tl.where(is_new_max, raw_val_f32, max_val_acc)
            best_k = tl.where(is_new_max, kh * kernel_w + kw, best_k)

    kh_best = best_k // kernel_w
    kw_best = best_k % kernel_w
    best_h_in = h_base + kh_best * dilation_h
    best_w_in = w_base + kw_best * dilation_w
    max_idx_acc = tl.where(best_k >= 0, (best_h_in * in_w + best_w_in).to(tl.int32), -1)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    output_block_ptr = (
        out_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc.to(dtype), mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc.to(tl.int64), mask=out_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 16}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 8, "BLOCK_IN_W": 32}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 32}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 64}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 64, "BLOCK_IN_W": 16}, num_warps=8),
    ],
    key=[
        "in_h",
        "in_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    ],
)
@triton.jit
def max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_h,
    in_w,
    out_h,
    out_w,
    out_stride_nc,
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
    logger.debug("GEMS MAX_POOL2D_WITH_INDICES FORWARD")
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
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    assert (
        in_h * in_w < 2147483647
    ), "Feature map elements exceed int32 representable range."

    out_hw = out_h * out_w
    total_channels = in_n * in_c

    use_packed = (
        out_hw <= 2048
        and total_channels >= 512
        and total_channels * in_h * in_w < 2147483647
    )

    if use_packed:
        grid = lambda meta: (
            triton.cdiv(total_channels, meta["BLOCK_C"]),
            triton.cdiv(out_hw, meta["BLOCK_HW"]),
        )
        max_pool2d_channel_packed_kernel[grid](
            input,
            output,
            indices,
            total_channels,
            in_h,
            in_w,
            out_w,
            out_hw,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        )
    else:
        grid = lambda meta: (
            in_n * in_c,
            triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
        )
        max_pool2d_forward_kernel[grid](
            input,
            output,
            indices,
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
    logger.debug("GEMS MAX_POOL2D BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

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
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_h, meta["BLOCK_IN_H"]) * triton.cdiv(in_w, meta["BLOCK_IN_W"]),
    )

    out_stride_nc = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

    max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_h,
        in_w,
        out_h,
        out_w,
        out_stride_nc,
        out_stride_h,
        out_stride_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )

    return grad_input.to(grad_output.dtype)


def max_pool2d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
    indices: torch.Tensor,
):
    logger.debug("GEMS MAX_POOL2D_WITH_INDICES_BACKWARD")
    return max_pool2d_backward(
        grad_output, self, indices, kernel_size, stride, padding, dilation, ceil_mode
    )
