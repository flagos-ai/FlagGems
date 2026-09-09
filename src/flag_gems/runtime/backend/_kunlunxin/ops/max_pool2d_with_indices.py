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
    # Input/Output shapes
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
    BLOCK_W: tl.constexpr,
):
    # Keep the height and N*C coordinates scalar.  The previous 2-D tile used
    # div/mod to decode a flattened H*W grid and exposed a bad lane mapping in
    # the XPU lowering.  A 1-D W tile has the same work with a much simpler
    # address calculation.
    h_out = tl.program_id(0)
    w_block = tl.program_id(1)
    pid_nc = tl.program_id(2)
    w_out_offsets = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_out_offsets < out_w

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_W,), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_W,), -1, dtype=tl.int32)

    input_base_ptr = input_ptr + pid_nc * in_h * in_w

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            input_offset = h_in * in_w + w_in
            safe_input_offset = tl.where(in_mask, input_offset, 0)
            loaded_val = tl.load(
                input_base_ptr + safe_input_offset, mask=w_mask, other=0
            )
            current_val = tl.where(in_mask, loaded_val, min_val)
            current_idx = h_in * in_w + w_in

            is_new_max = current_val > max_val_acc
            max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
            max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    output_ptrs = out_base_ptr + h_out * out_w + w_out_offsets
    indices_ptrs = indices_base_ptr + h_out * out_w + w_out_offsets
    tl.store(output_ptrs, max_val_acc, mask=w_mask)
    tl.store(indices_ptrs, max_idx_acc, mask=w_mask)


@libentry()
@triton.jit
def max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
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
    # Tiling parameters
    BLOCK_IN_W: tl.constexpr,
):
    # One input row and a W tile per program.  This mirrors the forward grid
    # and avoids the 2-D accumulator/lane mapping that corrupted indices on XPU.
    h_in = tl.program_id(0)
    w_block = tl.program_id(1)
    nc_idx = tl.program_id(2)
    w_in_offsets = w_block * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)
    store_mask = w_in_offsets < in_w

    current_input_flat_idx = h_in * in_w + w_in_offsets
    grad_acc = tl.zeros((BLOCK_IN_W,), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_h * out_w
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_h * out_w

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            numerator_h = h_in + padding_h - kh * dilation_h
            numerator_w = w_in_offsets + padding_w - kw * dilation_w

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
            out_offsets = safe_h_out * out_w + safe_w_out

            safe_out_offsets = tl.where(load_mask, out_offsets, 0)
            indices_block = tl.load(
                indices_base_ptr + safe_out_offsets, mask=store_mask, other=0
            )
            match_mask = load_mask & (indices_block == current_input_flat_idx)

            # Load only from a valid output coordinate, then select by index in
            # registers.  The XPU lowering mishandles a data-dependent mask on
            # this load and can return every candidate gradient for every lane.
            grad_block = tl.load(
                grad_output_base_ptr + safe_out_offsets, mask=store_mask, other=0.0
            )
            grad_acc += tl.where(match_mask, grad_block, 0.0)

    grad_input_base_ptr = grad_input_ptr + nc_idx * in_h * in_w
    grad_input_offsets = h_in * in_w + w_in_offsets
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

    # A program handles one output row and one W tile.  Keep the tile 1-D so the
    # compiler does not have to decode a flattened H*W program id.
    block_w = min(triton.next_power_of_2(out_w), 64)

    grid = (
        out_h,
        triton.cdiv(out_w, block_w),
        in_n * in_c,
    )

    with torch_device_fn.device(input.device):
        max_pool2d_forward_kernel[grid](
            input,
            output,
            indices,
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
            block_w,
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
    input = input.contiguous()
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

    # Mirror the forward grid: one input row and one W tile per program.
    block_in_w = min(triton.next_power_of_2(in_w), 32)

    grid = (
        in_h,
        triton.cdiv(in_w, block_in_w),
        in_n * in_c,
    )

    with torch_device_fn.device(grad_input.device):
        max_pool2d_backward_kernel[grid](
            grad_output,
            indices,
            grad_input,
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
            block_in_w,
        )

    return grad_input.to(original_dtype)


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
    """Vendor implementation for the aten backward schema."""
    return max_pool2d_backward(
        grad_output,
        self,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
