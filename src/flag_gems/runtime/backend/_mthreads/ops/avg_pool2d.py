import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


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
        triton.Config({"BLOCK_SIZE": 256}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def avg_pool2d_forward_kernel_1d(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_n,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    n_elements,
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose linear index to (n, c, h_out, w_out)
    out_hw = out_h * out_w
    out_chw = in_c * out_hw

    n_idx = offsets // out_chw
    remaining = offsets % out_chw
    c_idx = remaining // out_hw
    remaining2 = remaining % out_hw
    h_out_idx = remaining2 // out_w
    w_out_idx = remaining2 % out_w

    # Compute sum and count for each output position
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_idx * stride_h - padding_h + kh * dilation_h
            w_in = w_out_idx * stride_w - padding_w + kw * dilation_w
            in_mask = mask & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

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

    tl.store(
        output_ptr + offsets, output_vals.to(output_ptr.type.element_ty), mask=mask
    )


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def avg_pool2d_backward_kernel_1d(
    grad_output_ptr,
    grad_input_ptr,
    # Input/Output shapes
    in_n,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    n_elements,
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose linear index to (n, c, h_in, w_in)
    in_hw = in_h * in_w
    in_chw = in_c * in_hw

    n_idx = offsets // in_chw
    remaining = offsets % in_chw
    c_idx = remaining // in_hw
    remaining2 = remaining % in_hw
    h_in_idx = remaining2 // in_w
    w_in_idx = remaining2 % in_w

    grad_input_ptr_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    grad_output_ptr_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

    grad_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kh_loop in tl.static_range(0, kernel_h):
        for kw_loop in tl.static_range(0, kernel_w):
            h_out_num = h_in_idx + padding_h - kh_loop * dilation_h
            w_out_num = w_in_idx + padding_w - kw_loop * dilation_w

            h_valid_map = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
            w_valid_map = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)

            h_out = h_out_num // stride_h
            w_out = w_out_num // stride_w

            out_mask = (
                mask & h_valid_map & w_valid_map & (h_out < out_h) & (w_out < out_w)
            )

            # Compute count for this output position (for count_include_pad=False)
            h_start = h_out * stride_h - padding_h
            w_start = w_out * stride_w - padding_w
            count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
            for kh_count in tl.static_range(0, kernel_h):
                for kw_count in tl.static_range(0, kernel_w):
                    h_in_for_count = h_start + kh_count * dilation_h
                    w_in_for_count = w_start + kw_count * dilation_w
                    is_valid = (
                        (h_in_for_count >= 0)
                        & (h_in_for_count < in_h)
                        & (w_in_for_count >= 0)
                        & (w_in_for_count < in_w)
                    )
                    count += is_valid.to(tl.int32)

            count_divisor = count.to(tl.float32)

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
            divisor = tl.where(divisor == 0, 1.0, divisor)

            grad_out_ptr = (
                grad_output_ptr_base + h_out * out_stride_h + w_out * out_stride_w
            )
            grad_out_val = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
            grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)

    grad_input_store_ptr = (
        grad_input_ptr_base + h_in_idx * in_stride_h + w_in_idx * in_stride_w
    )
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=mask,
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
    logger.debug("GEMS_MTHREADS AVG_POOL2D FORWARD")

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

    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    avg_pool2d_forward_kernel_1d[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        n_elements,
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

    n_elements = grad_input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    avg_pool2d_backward_kernel_1d[grid](
        grad_output,
        grad_input,
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        n_elements,
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
