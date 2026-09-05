# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MTHREADS avg_pool2d implementations.

The generic FlagGems backward kernel maps one program to an input tile and
enumerates every KxK tap.  That is particularly expensive on MTT because the
receiver test contains vector integer modulo/division.  The kernels here use
the inverse pooling relation instead: an input element directly computes the
small output interval that can produce it.
"""

import logging

# Triton is optional in some backend environments; keep this import block
# stable across isort's installed-module classification.
# isort: off
import triton
import triton.language as tl
import torch
from flag_gems.ops.avg_pool2d import _parse_pool_params
from flag_gems.ops.avg_pool2d import avg_pool2d_backward as _generic_avg_pool2d_backward
from flag_gems.utils import libentry

# isort: on

logger = logging.getLogger(
    f"flag_gems.runtime.backend._mthreads.ops.{__name__.split('.')[-1]}"
)


@triton.jit
def _k3_divisor(
    oh,
    ow,
    in_h,
    in_w,
    CEIL_TAIL_H: tl.constexpr,
    CEIL_TAIL_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    divisor = tl.full((BLOCK_H, BLOCK_W), 9.0, tl.float32)
    if CEIL_TAIL_H:
        hs = oh * 2 - 1
        vh = tl.maximum(tl.minimum(hs + 3, in_h + 1) - tl.maximum(hs, 0), 0)
        divisor = divisor * tl.where(hs + 3 > in_h + 1, vh, 3).to(tl.float32) / 3.0
    if CEIL_TAIL_W:
        ws = ow * 2 - 1
        vw = tl.maximum(tl.minimum(ws + 3, in_w + 1) - tl.maximum(ws, 0), 0)
        divisor = divisor * tl.where(ws + 3 > in_w + 1, vw, 3).to(tl.float32) / 3.0
    return divisor


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=[
        "in_h",
        "in_w",
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def _avg_pool2d_backward_direct_range_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    CEIL_TAIL_H: tl.constexpr,
    CEIL_TAIL_W: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    h_block = pid_hw // num_w_blocks
    w_block = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    h2 = h[:, None]
    w2 = w[None, :]
    in_mask = (h2 < in_h) & (w2 < in_w)

    # oh_min = ceil((h + p - (k - 1)) / s), oh_max=floor((h+p)/s).
    # The numerator of oh_min can be negative, so use a sign-safe ceil-div.
    h_min_num = h2 + padding_h - (kernel_h - 1)
    w_min_num = w2 + padding_w - (kernel_w - 1)
    h_min = tl.where(
        h_min_num >= 0,
        (h_min_num + stride_h - 1) // stride_h,
        -((-h_min_num) // stride_h),
    )
    w_min = tl.where(
        w_min_num >= 0,
        (w_min_num + stride_w - 1) // stride_w,
        -((-w_min_num) // stride_w),
    )
    h_max = (h2 + padding_h) // stride_h
    w_max = (w2 + padding_w) // stride_w

    # At most ceil(K/S) producers exist in each dimension.  All loops are
    # compile-time unrolled and masked by the exact direct range.
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    go_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    for dh in tl.static_range(0, (kernel_h + stride_h - 1) // stride_h):
        oh = h_min + dh
        oh_mask = (oh >= h_min) & (oh <= h_max) & (oh >= 0) & (oh < out_h)
        if DIVISOR_OVERRIDE != 0:
            divisor_h = tl.full((BLOCK_H, BLOCK_W), float(DIVISOR_OVERRIDE), tl.float32)
        elif COUNT_INCLUDE_PAD:
            if CEIL_TAIL_H:
                h_start = oh * stride_h - padding_h
                valid_h = tl.minimum(h_start + kernel_h, in_h + padding_h) - tl.maximum(
                    h_start, 0
                )
                valid_h = tl.maximum(valid_h, 0)
                # PyTorch counts explicit padding, but not the shortened tail
                # introduced solely by ceil_mode.
                divisor_h = tl.where(
                    h_start + kernel_h > in_h + padding_h, valid_h, kernel_h
                ).to(tl.float32)
            else:
                divisor_h = tl.full((BLOCK_H, BLOCK_W), float(kernel_h), tl.float32)
        else:
            h_start = oh * stride_h - padding_h
            valid_h = tl.minimum(h_start + kernel_h, in_h) - tl.maximum(h_start, 0)
            valid_h = tl.maximum(valid_h, 0)
            divisor_h = valid_h.to(tl.float32)

        for dw in tl.static_range(0, (kernel_w + stride_w - 1) // stride_w):
            ow = w_min + dw
            ow_mask = (ow >= w_min) & (ow <= w_max) & (ow >= 0) & (ow < out_w)
            producer_mask = in_mask & oh_mask & ow_mask
            if DIVISOR_OVERRIDE != 0:
                divisor = divisor_h
            elif COUNT_INCLUDE_PAD:
                if CEIL_TAIL_W:
                    w_start = ow * stride_w - padding_w
                    valid_w = tl.minimum(
                        w_start + kernel_w, in_w + padding_w
                    ) - tl.maximum(w_start, 0)
                    valid_w = tl.maximum(valid_w, 0)
                    divisor = divisor_h * tl.where(
                        w_start + kernel_w > in_w + padding_w, valid_w, kernel_w
                    ).to(tl.float32)
                else:
                    divisor = divisor_h * kernel_w
            else:
                w_start = ow * stride_w - padding_w
                valid_w = tl.minimum(w_start + kernel_w, in_w) - tl.maximum(w_start, 0)
                valid_w = tl.maximum(valid_w, 0)
                divisor = divisor_h * valid_w.to(tl.float32)
            divisor = tl.where(divisor == 0, 1.0, divisor)
            go_ptr = go_base + oh * out_stride_h + ow * out_stride_w
            go = tl.load(go_ptr, mask=producer_mask, other=0.0).to(tl.float32)
            grad_acc += tl.where(producer_mask, go / divisor, 0.0)

    gi_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    gi_ptr = gi_base + h2 * in_stride_h + w2 * in_stride_w
    tl.store(gi_ptr, grad_acc, mask=in_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
)
@triton.jit
def _avg_pool2d_backward_k3s2p1_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    CEIL_TAIL_H: tl.constexpr,
    CEIL_TAIL_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    h_block = pid_hw // num_w_blocks
    w_block = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    h2 = h[:, None]
    w2 = w[None, :]
    in_mask = (h2 < in_h) & (w2 < in_w)

    # For K=3,S=2,P=1, the inverse range is exactly [h//2,(h+1)//2]
    # (and analogously for W).  This removes both modulo and general divide.
    oh0 = h2 >> 1
    oh1 = (h2 + 1) >> 1
    ow0 = w2 >> 1
    ow1 = (w2 + 1) >> 1
    go_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    m00 = in_mask & (oh0 < out_h) & (ow0 < out_w)
    m01 = in_mask & (oh0 < out_h) & (ow1 < out_w) & (ow1 != ow0)
    m10 = in_mask & (oh1 < out_h) & (ow0 < out_w) & (oh1 != oh0)
    m11 = in_mask & (oh1 < out_h) & (ow1 < out_w) & (oh1 != oh0) & (ow1 != ow0)

    grad_acc += tl.load(
        go_base + oh0 * out_stride_h + ow0 * out_stride_w, mask=m00, other=0.0
    ).to(tl.float32) / _k3_divisor(
        oh0, ow0, in_h, in_w, CEIL_TAIL_H, CEIL_TAIL_W, BLOCK_H, BLOCK_W
    )
    grad_acc += tl.load(
        go_base + oh0 * out_stride_h + ow1 * out_stride_w, mask=m01, other=0.0
    ).to(tl.float32) / _k3_divisor(
        oh0, ow1, in_h, in_w, CEIL_TAIL_H, CEIL_TAIL_W, BLOCK_H, BLOCK_W
    )
    grad_acc += tl.load(
        go_base + oh1 * out_stride_h + ow0 * out_stride_w, mask=m10, other=0.0
    ).to(tl.float32) / _k3_divisor(
        oh1, ow0, in_h, in_w, CEIL_TAIL_H, CEIL_TAIL_W, BLOCK_H, BLOCK_W
    )
    grad_acc += tl.load(
        go_base + oh1 * out_stride_h + ow1 * out_stride_w, mask=m11, other=0.0
    ).to(tl.float32) / _k3_divisor(
        oh1, ow1, in_h, in_w, CEIL_TAIL_H, CEIL_TAIL_W, BLOCK_H, BLOCK_W
    )

    gi_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    tl.store(gi_base + h2 * in_stride_h + w2 * in_stride_w, grad_acc, mask=in_mask)


def _launch_backward(
    kernel,
    grad_output,
    grad_input,
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
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    grid = lambda meta: (
        grad_input.shape[0] * grad_input.shape[1],
        triton.cdiv(in_h, meta["BLOCK_H"]) * triton.cdiv(in_w, meta["BLOCK_W"]),
    )
    args = [
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
    ]
    if kernel is _avg_pool2d_backward_direct_range_kernel:
        args += [kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w]
        ceil_tail_h = bool(ceil_mode and ((in_h + 2 * padding_h - kernel_h) % stride_h))
        ceil_tail_w = bool(ceil_mode and ((in_w + 2 * padding_w - kernel_w) % stride_w))
        kernel[grid](
            *args,
            COUNT_INCLUDE_PAD=count_include_pad,
            CEIL_TAIL_H=ceil_tail_h,
            CEIL_TAIL_W=ceil_tail_w,
            DIVISOR_OVERRIDE=divisor_override or 0,
        )
    else:
        ceil_tail_h = bool(ceil_mode and ((in_h + 2 * padding_h - kernel_h) % stride_h))
        ceil_tail_w = bool(ceil_mode and ((in_w + 2 * padding_w - kernel_w) % stride_w))
        kernel[grid](*args, CEIL_TAIL_H=ceil_tail_h, CEIL_TAIL_W=ceil_tail_w)


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
    """MTHREADS dispatch with a semantics-preserving generic fallback."""
    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    # Keep the existing implementation for layouts/dtypes outside the tuned
    # FP32 contiguous contract.  In particular this preserves arbitrary
    # strides and all PyTorch corner-case behavior.
    if input.dtype != torch.float32 or grad_output.dtype != torch.float32:
        return _generic_avg_pool2d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )
    # With no spatial decimation the inverse range has exactly K^2 producers,
    # so the extra range/divisor arithmetic costs more than the old kernel.
    # A runtime divisor override is likewise already a cheap constant path;
    # retain the established implementation for these non-winning cases.
    if (stride_h == 1 and stride_w == 1) or divisor_override is not None:
        return _generic_avg_pool2d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    if not input.is_contiguous() or not grad_output.is_contiguous():
        return _generic_avg_pool2d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    grad_output = grad_output.contiguous()
    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]
    # The kernel writes every input element, including receivers with no
    # producer, so no zero-initialization launch is needed.
    grad_input = torch.empty_like(input, dtype=torch.float32)
    if grad_output.numel() == 0:
        return grad_input.to(grad_output.dtype)

    if (
        kernel_h == 3
        and kernel_w == 3
        and stride_h == 2
        and stride_w == 2
        and padding_h == 1
        and padding_w == 1
        and count_include_pad
        and divisor_override is None
    ):
        _launch_backward(
            _avg_pool2d_backward_k3s2p1_kernel,
            grad_output,
            grad_input,
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
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    else:
        _launch_backward(
            _avg_pool2d_backward_direct_range_kernel,
            grad_output,
            grad_input,
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
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    return grad_input.to(grad_output.dtype)


__all__ = ["avg_pool2d_backward"]
