# Optimized avg_pool2d_backward for mthreads

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.avg_pool2d import (
    _parse_pool_params,
    avg_pool2d,
    avg_pool2d_forward_kernel,
    pool2d_output_size,
)
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@libentry()
@triton.autotune(
    configs=[
        # Small block configs for small inputs
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 4}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=4, num_warps=4),
        # Medium block configs
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        # Large block configs
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["in_h", "in_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def avg_pool2d_backward_kernel_optimized(
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
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_W)

    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    grad_input_block_ptr = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    grad_output_base_ptr = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Precompute fixed divisor for COUNT_INCLUDE_PAD=True case
    fixed_divisor = tl.full((BLOCK_H, BLOCK_W), kernel_h * kernel_w, dtype=tl.float32)

    for kh_loop in range(kernel_h):
        for kw_loop in range(kernel_w):
            h_out_num = h_in_offsets[:, None] + padding_h - kh_loop * dilation_h
            w_out_num = w_in_offsets[None, :] + padding_w - kw_loop * dilation_w

            h_valid_map = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
            w_valid_map = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)

            h_out = h_out_num // stride_h
            w_out = w_out_num // stride_w

            h_out_mask = h_valid_map & (h_out < out_h)
            w_out_mask = w_valid_map & (w_out < out_w)
            out_mask = h_out_mask & w_out_mask

            if divisor_override != 0:
                divisor = tl.full(
                    (BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32
                )
            elif COUNT_INCLUDE_PAD:
                divisor = fixed_divisor
            else:
                # Optimized divisor calculation for count_include_pad=False
                # Calculate the valid range for each output position
                h_start = h_out * stride_h - padding_h
                w_start = w_out * stride_w - padding_w

                # Calculate valid h range
                h_begin = tl.maximum(0, -h_start // dilation_h + ((-h_start % dilation_h) != 0).to(tl.int32))
                h_end_raw = (in_h - h_start + dilation_h - 1) // dilation_h
                h_end = tl.minimum(kernel_h, h_end_raw)
                h_count = tl.maximum(0, h_end - h_begin)

                # Calculate valid w range
                w_begin = tl.maximum(0, -w_start // dilation_w + ((-w_start % dilation_w) != 0).to(tl.int32))
                w_end_raw = (in_w - w_start + dilation_w - 1) // dilation_w
                w_end = tl.minimum(kernel_w, w_end_raw)
                w_count = tl.maximum(0, w_end - w_begin)

                divisor = (h_count * w_count).to(tl.float32)

            divisor = tl.where(divisor == 0, 1.0, divisor)

            grad_out_ptr = (
                grad_output_base_ptr + h_out * out_stride_h + w_out * out_stride_w
            )
            grad_out_val = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
            grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)

    grad_input_store_ptr = (
        grad_input_block_ptr
        + h_in_offsets[:, None] * in_stride_h
        + w_in_offsets[None, :] * in_stride_w
    )
    in_write_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=in_write_mask,
    )


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
    logger.debug("GEMS_MTHREADS AVG_POOL2D BACKWARD")

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
        in_n * in_c,
        triton.cdiv(in_h, meta["BLOCK_H"]) * triton.cdiv(in_w, meta["BLOCK_W"]),
    )

    avg_pool2d_backward_kernel_optimized[grid](
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
    )

    return grad_input.to(grad_output.dtype)
