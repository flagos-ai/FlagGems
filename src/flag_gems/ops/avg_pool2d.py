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
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
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

    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=0.0
            )

            sum_acc += tl.where(in_mask, current_val, 0.0)
            count_acc += in_mask.to(tl.int32)

    if divisor_override != 0:
        divisor = tl.full((BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32)
    elif COUNT_INCLUDE_PAD:
        divisor = tl.full((BLOCK_H, BLOCK_W), kernel_h * kernel_w, dtype=tl.float32)
    else:
        divisor = count_acc.to(tl.float32)

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
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["in_h", "in_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
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
                divisor = tl.full(
                    (BLOCK_H, BLOCK_W), kernel_h * kernel_w, dtype=tl.float32
                )
            else:
                h_start = h_out * stride_h - padding_h
                w_start = w_out * stride_w - padding_w
                count = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
                for kh_count in range(0, kernel_h):
                    for kw_count in range(0, kernel_w):
                        h_in_for_count = h_start + kh_count * dilation_h
                        w_in_for_count = w_start + kw_count * dilation_w
                        is_valid = (
                            (h_in_for_count >= 0)
                            & (h_in_for_count < in_h)
                            & (w_in_for_count >= 0)
                            & (w_in_for_count < in_w)
                        )
                        count += is_valid.to(tl.int32)
                divisor = count.to(tl.float32)

            divisor = tl.where(divisor == 0, 1.0, divisor)

            grad_out_ptr = (
                grad_output_base_ptr + h_out * out_stride_h + w_out * out_stride_w
            )
            grad_out_val = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
            grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)
            # grad_to_add = grad_out_val.to(tl.float32) / divisor.to(tl.float32)
            # grad_acc += tl.where(out_mask, grad_to_add, 0.0)

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
    logger.debug("GEMS AVG_POOL2D FORWARD")

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
    logger.debug("GEMS AVG_POOL2D BACKWARD")

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
    )

    return grad_input.to(grad_output.dtype)
