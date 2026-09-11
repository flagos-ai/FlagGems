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

logger = logging.getLogger(__name__)


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
    # Tiling meta-parameter
    BLOCK_SIZE: tl.constexpr,
):
    # Flattened input-centric backward: each program handles a contiguous chunk
    # of the input's D*H*W volume for a single (n, c) plane, so every lane maps
    # to a real input element instead of a masked-out corner of an oversized
    # 2D block. This keeps utilization high for shapes with small spatial dims
    # but large N*C. Uses tl.store (not atomic_add), safe with autotune.
    # Grid: (N*C, cdiv(in_d * in_h * in_w, BLOCK_SIZE))
    pid_nc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    in_dhw = in_d * in_h * in_w
    offs = pid_blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    elem_mask = offs < in_dhw

    # Decompose the flat input index into (d_in, h_in, w_in).
    w_in = offs % in_w
    dh_tmp = offs // in_w
    h_in = dh_tmp % in_h
    d_in = dh_tmp // in_h

    grad_input_base = (
        grad_input_ptr
        + n_idx.to(tl.int64) * in_stride_n
        + c_idx.to(tl.int64) * in_stride_c
    )
    grad_output_base = (
        grad_output_ptr
        + n_idx.to(tl.int64) * out_stride_n
        + c_idx.to(tl.int64) * out_stride_c
    )

    grad_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kd in range(kernel_d):
        d_out_num = d_in + padding_d - kd
        d_out_valid = (d_out_num >= 0) & ((d_out_num % stride_d) == 0)
        d_out = d_out_num // stride_d
        d_out_valid = d_out_valid & (d_out >= 0) & (d_out < out_d)

        for kh in range(kernel_h):
            h_out_num = h_in + padding_h - kh
            h_valid = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
            h_out = h_out_num // stride_h
            h_valid = h_valid & (h_out < out_h)

            for kw in range(kernel_w):
                w_out_num = w_in + padding_w - kw
                w_valid = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)
                w_out = w_out_num // stride_w
                w_valid = w_valid & (w_out < out_w)

                out_mask = elem_mask & d_out_valid & h_valid & w_valid

                if divisor_override != 0:
                    divisor = tl.full((BLOCK_SIZE,), divisor_override, dtype=tl.float32)
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
        grad_input_base + d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
    )
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=elem_mask,
    )


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
    """Compute the gradient of avg_pool3d (hygon specialization).

    Uses a flattened input-centric kernel so each lane maps to a real input
    element, avoiding the wasted-lane blow-up of the generic 2D-tiled kernel
    on shapes with small spatial dims and large N*C.

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
    logger.debug("GEMS_HYGON AVG_POOL3D BACKWARD")

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

    # Flattened input-centric grid: one axis over N*C, one over the input volume.
    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_d * in_h * in_w, meta["BLOCK_SIZE"]),
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
        BLOCK_SIZE=256,
    )

    return grad_input
