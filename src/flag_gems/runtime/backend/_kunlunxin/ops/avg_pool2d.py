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
    # Number of (n, c) channels
    nc_total,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)

    # Batch multiple (n, c) channels into one program with a serial 2D-tile
    # loop (same recipe as max_pool2d_with_indices): with small outputs (late
    # ResNet stages) the per-(n,c) grid made the kernel launch-bound (up to
    # 65536 tiny programs). Tail channels are clamped to the last valid
    # channel, so their stores are idempotent (same values, same addresses).
    nc_start = pid_nc * BLOCK_NC
    for i in tl.range(0, BLOCK_NC):
        nc = nc_start + i
        nc_safe = tl.minimum(nc, nc_total - 1)
        n_idx = nc_safe // in_c
        c_idx = nc_safe % in_c

        sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

        input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                h_in = (
                    h_out_offsets[:, None] * stride_h
                    - padding_h
                    + kh * dilation_h
                )
                w_in = (
                    w_out_offsets[None, :] * stride_w
                    - padding_w
                    + kw * dilation_w
                )
                in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

                # On XPU, masked loads are unreliable: `other` is not honored and even
                # the valid lanes of a partially-masked load can return corrupted data
                # (same hazard as max_pool2d_with_indices). Out-of-bounds input
                # coordinates are clamped to (0, 0) so the load address is always in
                # bounds, and padding lanes are restored to 0 in registers.
                # Affine clamp (min/max): keeps the address expression
                # piecewise affine so the XPU offset analysis can prove
                # unit/constant strides (tl.where on the address forces
                # per-lane discrete gather). Padding lanes are still restored
                # to 0 in registers below.
                h_in_safe = tl.minimum(tl.maximum(h_in, 0), in_h - 1)
                w_in_safe = tl.minimum(tl.maximum(w_in, 0), in_w - 1)
                input_offset = h_in_safe * in_stride_h + w_in_safe * in_stride_w
                current_val = tl.load(input_base_ptr + input_offset)

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
            divisor_override != 0,
            divisor_override + default_divisor * 0,
            default_divisor,
        )

        output_vals = tl.where(divisor != 0, sum_acc / divisor, 0.0)

        out_base_ptr = output_ptr + nc_safe * out_h * out_w
        output_block_ptr = (
            out_base_ptr
            + h_out_offsets[:, None] * out_w
            + w_out_offsets[None, :]
        )

        tl.store(
            output_block_ptr,
            output_vals.to(output_ptr.type.element_ty),
            mask=out_mask,
        )


@libentry()
@triton.jit
def avg_pool2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
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
    # AvgPool specific parameters
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    # Number of (n, c) channels
    nc_total,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_W)

    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    in_write_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)

    # Batch multiple (n, c) channels into one program with a serial 2D-tile
    # loop (same recipe as max_pool2d_with_indices): with small inputs (late
    # ResNet stages) the per-(n,c) grid made the kernel launch-bound (up to
    # 65536 tiny programs). Tail channels are clamped to the last valid
    # channel, so their stores are idempotent (same values, same addresses).
    nc_start = pid_nc * BLOCK_NC
    for i in tl.range(0, BLOCK_NC):
        nc = nc_start + i
        nc_safe = tl.minimum(nc, nc_total - 1)
        n_idx = nc_safe // in_c
        c_idx = nc_safe % in_c

        grad_input_block_ptr = (
            grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
        )
        grad_output_base_ptr = (
            grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
        )

        grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

        for kh_loop in tl.static_range(0, kernel_h):
            for kw_loop in tl.static_range(0, kernel_w):
                h_out_num = h_in_offsets[:, None] + padding_h - kh_loop * dilation_h
                w_out_num = w_in_offsets[None, :] + padding_w - kw_loop * dilation_w

                h_valid_map = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
                w_valid_map = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)

                h_out = h_out_num // stride_h
                w_out = w_out_num // stride_w

                h_out_mask = h_valid_map & (h_out < out_h)
                w_out_mask = w_valid_map & (w_out < out_w)
                out_mask = h_out_mask & w_out_mask

                # Compute count for this output position (for count_include_pad=False)
                if COUNT_INCLUDE_PAD:
                    # With count_include_pad=True the valid-lane count is always
                    # >= 0, so the original code path degenerated to the constant
                    # window size: the previous O(kernel_h*kernel_w)^2 count loop
                    # was pure compute waste and is skipped entirely here.
                    default_divisor = float(kernel_h * kernel_w)
                else:
                    h_start = h_out * stride_h - padding_h
                    w_start = w_out * stride_w - padding_w
                    if dilation_h == 1 and dilation_w == 1:
                        # The valid set of a window is a cartesian product of the
                        # valid rows and columns, so the count is separable and
                        # can be computed in O(1) per axis instead of a K*K loop.
                        cnt_h = (
                            tl.minimum(tl.maximum(h_start + kernel_h, 0), in_h)
                            - tl.maximum(h_start, 0)
                        )
                        cnt_h = tl.maximum(tl.minimum(cnt_h, kernel_h), 0)
                        cnt_w = (
                            tl.minimum(tl.maximum(w_start + kernel_w, 0), in_w)
                            - tl.maximum(w_start, 0)
                        )
                        cnt_w = tl.maximum(tl.minimum(cnt_w, kernel_w), 0)
                        default_divisor = (cnt_h * cnt_w).to(tl.float32)
                    else:
                        count_h = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
                        for kh_count in tl.static_range(0, kernel_h):
                            h_in_for_count = h_start + kh_count * dilation_h
                            h_valid_count = (h_in_for_count >= 0) & (
                                h_in_for_count < in_h
                            )
                            count_h += h_valid_count.to(tl.int32)
                        count_w = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
                        for kw_count in tl.static_range(0, kernel_w):
                            w_in_for_count = w_start + kw_count * dilation_w
                            w_valid_count = (w_in_for_count >= 0) & (
                                w_in_for_count < in_w
                            )
                            count_w += w_valid_count.to(tl.int32)
                        default_divisor = (count_h * count_w).to(tl.float32)

                divisor = tl.where(
                    divisor_override != 0,
                    divisor_override + default_divisor * 0,
                    default_divisor,
                )
                divisor = tl.where(divisor == 0, 1.0, divisor)

                # On XPU, masked loads are unreliable: `other` is not honored and
                # even the valid lanes of a partially-masked load can return
                # corrupted data (same hazard as max_pool2d_with_indices).
                # Out-of-range output coordinates are clamped to (0, 0) so the
                # load address is always in bounds, and invalid lanes are
                # restored to 0 in registers.
                # Affine clamp (min/max), same rationale as the forward
                # kernel: address stays in bounds and piecewise affine.
                h_out_safe = tl.minimum(tl.maximum(h_out, 0), out_h - 1)
                w_out_safe = tl.minimum(tl.maximum(w_out, 0), out_w - 1)
                grad_out_ptr = (
                    grad_output_base_ptr
                    + h_out_safe * out_stride_h
                    + w_out_safe * out_stride_w
                )
                grad_out_val = tl.load(grad_out_ptr)
                grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)

        grad_input_store_ptr = (
            grad_input_block_ptr
            + h_in_offsets[:, None] * in_stride_h
            + w_in_offsets[None, :] * in_stride_w
        )
        tl.store(
            grad_input_store_ptr,
            grad_acc.to(grad_input_ptr.type.element_ty),
            mask=in_write_mask,
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

    # Adaptive tiling + channel batching (same recipe as
    # max_pool2d_with_indices): size the (BLOCK_H, BLOCK_W) tile to the actual
    # output (next pow2 capped) so tiny outputs (4x4/7x7) don't waste a 64x64
    # tile per program, then batch channels serially per program so each
    # program keeps around 2048 lanes (uni_sram safe budget).
    nc_total = in_n * in_c
    block_h = min(triton.next_power_of_2(out_h), 64)
    block_w = min(triton.next_power_of_2(out_w), 64)
    block_nc = max(1, min(nc_total, 2048 // (block_h * block_w)))

    grid = (
        triton.cdiv(nc_total, block_nc),
        triton.cdiv(out_h, block_h) * triton.cdiv(out_w, block_w),
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
        nc_total=nc_total,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
        BLOCK_NC=block_nc,
        num_warps=8,
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

    # Allocate grad_input directly in the output dtype: the kernel writes every
    # element of grad_input (grid fully tiles the input), so neither the fp32
    # intermediate nor the final whole-tensor `.to()` cast pass is needed.
    grad_input = torch.empty_like(input, dtype=grad_output.dtype)

    if grad_output.numel() == 0:
        return grad_input

    # Adaptive tiling + channel batching (same recipe as
    # max_pool2d_with_indices): the old per-(n,c) grid with a fixed 64x16 tile
    # made small inputs (7x7/14x14/28x28) launch-bound with up to 65536 tiny
    # programs. Size the (BLOCK_H, BLOCK_W) tile to the actual input (next pow2
    # capped), then batch channels serially per program so each program keeps
    # around 2048 lanes (uni_sram safe budget, >2048 triggers OOM).
    nc_total = in_n * in_c
    block_h = min(triton.next_power_of_2(in_h), 64)
    block_w = min(triton.next_power_of_2(in_w), 32)
    block_nc = max(1, min(nc_total, 2048 // (block_h * block_w)))

    grid = (
        triton.cdiv(nc_total, block_nc),
        triton.cdiv(in_h, block_h) * triton.cdiv(in_w, block_w),
    )

    avg_pool2d_backward_kernel[grid](
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
        nc_total=nc_total,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
        BLOCK_NC=block_nc,
        num_warps=8,
    )

    return grad_input
