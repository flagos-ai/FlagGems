import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    """Compute the output size for one spatial dimension of a 3D pooling operation."""
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
    key=["out_d", "out_h", "out_w", "kernel_d", "kernel_h", "kernel_w"],
)
@triton.jit
def avg_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # AvgPool specific parameters
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (N*C, out_d * cdiv(out_h, BLOCK_H) * cdiv(out_w, BLOCK_W))
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    # Decompose pid_dhw into d_idx, h_block_idx, w_block_idx
    d_idx = pid_dhw // num_hw_blocks
    hw_remainder = pid_dhw % num_hw_blocks
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kd in range(0, kernel_d):
        d_in = d_idx * stride_d - padding_d + kd * dilation_d
        d_valid = (d_in >= 0) & (d_in < in_d)
        for kh in range(0, kernel_h):
            for kw in range(0, kernel_w):
                h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
                hw_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
                in_mask = hw_mask & d_valid

                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=0.0
                )

                sum_acc += tl.where(in_mask, current_val, 0.0)
                count_acc += in_mask.to(tl.int32)

    if divisor_override != 0:
        divisor = tl.full((BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32)
    elif COUNT_INCLUDE_PAD:
        # Count positions within padded boundary (correct for ceil_mode edges)
        d_start_fwd = d_idx * stride_d - padding_d
        d_padded_count = tl.minimum(d_start_fwd + kernel_d, in_d + padding_d) - (
            tl.maximum(d_start_fwd, -padding_d)
        )
        d_padded_count = tl.maximum(d_padded_count, 0)

        h_start_fwd = h_out_offsets[:, None] * stride_h - padding_h
        h_padded_count = tl.minimum(h_start_fwd + kernel_h, in_h + padding_h) - (
            tl.maximum(h_start_fwd, -padding_h)
        )
        h_padded_count = tl.maximum(h_padded_count, 0)

        w_start_fwd = w_out_offsets[None, :] * stride_w - padding_w
        w_padded_count = tl.minimum(w_start_fwd + kernel_w, in_w + padding_w) - (
            tl.maximum(w_start_fwd, -padding_w)
        )
        w_padded_count = tl.maximum(w_padded_count, 0)

        divisor = (d_padded_count * h_padded_count * w_padded_count).to(tl.float32)
    else:
        divisor = count_acc.to(tl.float32)

    output_vals = tl.where(divisor != 0, sum_acc / divisor, 0.0)

    out_base_ptr = output_ptr + pid_nc * out_d * out_h * out_w + d_idx * out_h * out_w
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
    key=["in_h", "in_w", "kernel_d", "kernel_h", "kernel_w"],
)
@triton.jit
def avg_pool3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Strides for grad_input
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Strides for grad_output
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    # AvgPool specific parameters
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Input-centric backward: iterate over input positions, gather from output.
    # Uses tl.store (not atomic_add), safe with autotune.
    # Grid: (N*C, in_d * cdiv(in_h, BLOCK_H) * cdiv(in_w, BLOCK_W))
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    num_h_blocks = tl.cdiv(in_h, BLOCK_H)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_in_idx = pid_dhw // num_hw_blocks
    hw_remainder = pid_dhw % num_hw_blocks
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    grad_input_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    grad_output_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for kd in range(kernel_d):
        d_out_num = d_in_idx + padding_d - kd
        d_out_valid = (d_out_num >= 0) & ((d_out_num % stride_d) == 0)
        d_out = d_out_num // stride_d
        d_out_valid = d_out_valid & (d_out >= 0) & (d_out < out_d)

        for kh in range(kernel_h):
            for kw in range(kernel_w):
                h_out_num = h_in_offsets[:, None] + padding_h - kh
                w_out_num = w_in_offsets[None, :] + padding_w - kw

                h_valid = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
                w_valid = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)

                h_out = h_out_num // stride_h
                w_out = w_out_num // stride_w

                out_mask = (
                    d_out_valid & h_valid & w_valid & (h_out < out_h) & (w_out < out_w)
                )

                if divisor_override != 0:
                    divisor = tl.full(
                        (BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32
                    )
                elif COUNT_INCLUDE_PAD:
                    # Count positions within padded boundary (ceil_mode)
                    d_start_bwd = d_out * stride_d - padding_d
                    d_pc = tl.minimum(
                        d_start_bwd + kernel_d, in_d + padding_d
                    ) - tl.maximum(d_start_bwd, -padding_d)
                    d_pc = tl.maximum(d_pc, 0)

                    h_start_bwd = h_out * stride_h - padding_h
                    h_pc = tl.minimum(
                        h_start_bwd + kernel_h, in_h + padding_h
                    ) - tl.maximum(h_start_bwd, -padding_h)
                    h_pc = tl.maximum(h_pc, 0)

                    w_start_bwd = w_out * stride_w - padding_w
                    w_pc = tl.minimum(
                        w_start_bwd + kernel_w, in_w + padding_w
                    ) - tl.maximum(w_start_bwd, -padding_w)
                    w_pc = tl.maximum(w_pc, 0)

                    divisor = (d_pc * h_pc * w_pc).to(tl.float32)
                else:
                    d_start = d_out * stride_d - padding_d
                    d_count = tl.minimum(d_start + kernel_d, in_d) - tl.maximum(
                        d_start, 0
                    )
                    d_count = tl.maximum(d_count, 0)

                    h_start = h_out * stride_h - padding_h
                    h_count = tl.minimum(h_start + kernel_h, in_h) - tl.maximum(
                        h_start, 0
                    )
                    h_count = tl.maximum(h_count, 0)

                    w_start = w_out * stride_w - padding_w
                    w_count = tl.minimum(w_start + kernel_w, in_w) - tl.maximum(
                        w_start, 0
                    )
                    w_count = tl.maximum(w_count, 0)

                    divisor = (d_count * h_count * w_count).to(tl.float32)

                divisor = tl.where(divisor == 0, 1.0, divisor)

                grad_out_ptr = (
                    grad_output_base
                    + d_out * out_stride_d
                    + h_out * out_stride_h
                    + w_out * out_stride_w
                )
                grad_out_val = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
                grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)

    grad_input_store_ptr = (
        grad_input_base
        + d_in_idx * in_stride_d
        + h_in_offsets[:, None] * in_stride_h
        + w_in_offsets[None, :] * in_stride_w
    )
    in_write_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=in_write_mask,
    )


def _parse_pool3d_params(kernel_size, stride, padding):
    """Parse and validate 3D pooling parameters."""
    if isinstance(kernel_size, int):
        kernel_d = kernel_h = kernel_w = kernel_size
    else:
        kernel_d, kernel_h, kernel_w = kernel_size

    if stride is None or (isinstance(stride, (list, tuple)) and not stride):
        stride_d, stride_h, stride_w = kernel_d, kernel_h, kernel_w
    elif isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(padding, int):
        padding_d = padding_h = padding_w = padding
    else:
        padding_d, padding_h, padding_w = padding

    if stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride must be greater than zero")

    if padding_d < 0 or padding_h < 0 or padding_w < 0:
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


def avg_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    """Compute 3D average pooling over an input signal composed of several input
    planes.

    Args:
        input: 5D tensor of shape (N, C, D, H, W).
        kernel_size: Size of the pooling window. Can be int or (kD, kH, kW).
        stride: Stride of the pooling window. Default: kernel_size.
        padding: Implicit zero padding on both sides. Default: 0.
        ceil_mode: Use ceil instead of floor to compute output shape. Default: False.
        count_include_pad: Include zero-padding in the averaging calculation.
            Default: True.
        divisor_override: If specified, use this as the divisor instead of the
            pool size. Default: None.

    Returns:
        5D tensor of shape (N, C, D_out, H_out, W_out).
    """
    logger.debug("GEMS AVG_POOL3D FORWARD")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    input = input.contiguous()

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
    dilation_d, dilation_h, dilation_w = 1, 1, 1

    in_n, in_c, in_d, in_h, in_w = input.shape

    out_d = pool3d_output_size(
        in_d, kernel_d, stride_d, padding_d, dilation_d, ceil_mode
    )
    out_h = pool3d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = pool3d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )

    if output.numel() == 0:
        return output

    grid = lambda meta: (
        in_n * in_c,
        out_d
        * triton.cdiv(out_h, meta["BLOCK_H"])
        * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    avg_pool3d_forward_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        COUNT_INCLUDE_PAD=count_include_pad,
        divisor_override=divisor_override if divisor_override is not None else 0.0,
    )

    return output


def avg_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    """Compute the gradient of avg_pool3d.

    Args:
        grad_output: Gradient of the output tensor.
        input: Original input tensor (used for shape information).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        ceil_mode: Whether ceil was used for output shape.
        count_include_pad: Whether padding was included in averaging.
        divisor_override: Custom divisor override.

    Returns:
        Gradient with respect to the input tensor.
    """
    logger.debug("GEMS AVG_POOL3D BACKWARD")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    grad_output = grad_output.contiguous()

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

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    grad_input = torch.empty_like(input)

    if grad_output.numel() == 0:
        return grad_input.zero_()

    # Input-centric grid: iterate over input positions
    grid = lambda meta: (
        in_n * in_c,
        in_d * triton.cdiv(in_h, meta["BLOCK_H"]) * triton.cdiv(in_w, meta["BLOCK_W"]),
    )

    avg_pool3d_backward_kernel[grid](
        grad_output,
        grad_input,
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_input.stride(4),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_output.stride(4),
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        COUNT_INCLUDE_PAD=count_include_pad,
        divisor_override=divisor_override if divisor_override is not None else 0.0,
    )

    return grad_input
