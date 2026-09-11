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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _triple(value, name):
    if isinstance(value, int):
        return value, value, value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, int) for item in value)
    ):
        return int(value[0]), int(value[1]), int(value[2])
    raise ValueError(f"{name} must be an int or a tuple/list of three ints")


def _output_size(input_size, kernel, stride, padding, ceil_mode):
    numerator = input_size + 2 * padding - kernel
    output = (
        (numerator + stride - 1) // stride + 1 if ceil_mode else numerator // stride + 1
    )
    if ceil_mode and (output - 1) * stride >= input_size + padding:
        output -= 1
    return output


@libentry()
@triton.jit
def _avg_pool3d_backward_gather_kernel(
    grad_output_ptr,
    grad_input_ptr,
    out_last,  # out_total - 1, used to clamp the tap address in range
    D,
    H,
    W,
    OUT_D,
    OUT_H,
    OUT_W,
    SD,
    SH,
    SW,
    PD,
    PH,
    PW,
    INV_DIVISOR,  # 1 / divisor_override (fp32), only used when HAS_DIVISOR
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    TD: tl.constexpr,  # cdiv(KD, SD): number of output planes covering one input
    TH: tl.constexpr,
    TW: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    HAS_DIVISOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One flat tile of *input* positions per program.  grad_input is over
    # allocated to a whole multiple of BLOCK by the caller, so this kernel never
    # needs a store mask (masked stores are not honoured on this backend and a
    # vector store always touches a full 64-element granule).
    offsets = ext.program_id(0).to(tl.int32) * BLOCK + tl.arange(0, BLOCK)
    input_w = offsets % W
    remaining = offsets // W
    input_h = remaining % H
    remaining = remaining // H
    input_d = remaining % D
    channel_batch = remaining // D

    # For a fixed input position only ceil(K / S) output planes per dimension can
    # cover it: output = (input + P) // S - j, j in [0, ceil(K / S)).  Enumerating
    # those instead of all K kernel taps removes the (S ** 3 - 1) / S ** 3 fraction
    # of taps that can never match.
    base_od = (input_d + PD) // SD
    base_oh = (input_h + PH) // SH
    base_ow = (input_w + PW) // SW

    accumulator = tl.zeros((BLOCK,), tl.float32)
    for jd in tl.static_range(0, TD):
        output_d = base_od - jd
        tap_d = input_d + PD - output_d * SD
        start_d = output_d * SD - PD
        d_matches = (output_d >= 0) & (output_d < OUT_D) & (tap_d < KD)
        if COUNT_INCLUDE_PAD:
            count_d = tl.minimum(start_d + KD, D + PD) - start_d
        else:
            count_d = tl.minimum(start_d + KD, D) - tl.maximum(start_d, 0)
        count_d = tl.maximum(count_d, 1)
        for jh in tl.static_range(0, TH):
            output_h = base_oh - jh
            tap_h = input_h + PH - output_h * SH
            start_h = output_h * SH - PH
            h_matches = (output_h >= 0) & (output_h < OUT_H) & (tap_h < KH)
            if COUNT_INCLUDE_PAD:
                count_h = tl.minimum(start_h + KH, H + PH) - start_h
            else:
                count_h = tl.minimum(start_h + KH, H) - tl.maximum(start_h, 0)
            count_h = tl.maximum(count_h, 1)
            for jw in tl.static_range(0, TW):
                output_w = base_ow - jw
                tap_w = input_w + PW - output_w * SW
                start_w = output_w * SW - PW
                w_matches = (output_w >= 0) & (output_w < OUT_W) & (tap_w < KW)
                if COUNT_INCLUDE_PAD:
                    count_w = tl.minimum(start_w + KW, W + PW) - start_w
                else:
                    count_w = tl.minimum(start_w + KW, W) - tl.maximum(start_w, 0)
                count_w = tl.maximum(count_w, 1)

                matches = d_matches & h_matches & w_matches
                output_offset = (
                    (channel_batch * OUT_D + output_d) * OUT_H + output_h
                ) * OUT_W + output_w
                # Clamped, *unmasked* tap: a masked tl.load is not honoured here
                # (`other=` silently leaks into lanes whose mask is false), so the
                # address is forced in range and the lane is gated afterwards.
                output_offset = tl.minimum(tl.maximum(output_offset, 0), out_last)
                value = tl.load(grad_output_ptr + output_offset).to(tl.float32)
                if HAS_DIVISOR:
                    contribution = value * INV_DIVISOR
                else:
                    divisor = (count_d * count_h * count_w).to(tl.float32)
                    contribution = value / divisor
                accumulator += tl.where(matches, contribution, 0.0)

    tl.store(grad_input_ptr + offsets, accumulator.to(grad_input_ptr.type.element_ty))


def _pick_block(total):
    block = 1024
    while block > 64 and block > total:
        block //= 2
    return block


def avg_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    logger.debug("GEMS_KUNLUNXIN AVG_POOL3D_BACKWARD")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    was_unbatched = input.ndim == 4
    x = input.unsqueeze(0) if was_unbatched else input
    if x.ndim != 5:
        raise ValueError("avg_pool3d_backward expects a 4D or 5D input tensor")

    kernel_d, kernel_h, kernel_w = _triple(kernel_size, "kernel_size")
    if stride == () or stride == []:
        stride_d, stride_h, stride_w = kernel_d, kernel_h, kernel_w
    else:
        stride_d, stride_h, stride_w = _triple(stride, "stride")
    padding_d, padding_h, padding_w = _triple(padding, "padding")

    batch, channels, depth, height, width = x.shape
    output_d = _output_size(depth, kernel_d, stride_d, padding_d, ceil_mode)
    output_h = _output_size(height, kernel_h, stride_h, padding_h, ceil_mode)
    output_w = _output_size(width, kernel_w, stride_w, padding_w, ceil_mode)
    expected_shape = (batch, channels, output_d, output_h, output_w)
    grad_output = grad_output.unsqueeze(0) if was_unbatched else grad_output
    if tuple(grad_output.shape) != expected_shape:
        raise ValueError(
            f"grad_output has shape {tuple(grad_output.shape)}, expected {expected_shape}"
        )

    total = batch * channels * depth * height * width
    if total == 0:
        grad_input = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        return grad_input.squeeze(0) if was_unbatched else grad_input

    block = _pick_block(total)
    n_tiles = triton.cdiv(total, block)
    # Over-allocate so the kernel can store without any mask; only the leading
    # `total` elements are handed back.
    padded = torch.empty(n_tiles * block, dtype=x.dtype, device=x.device)
    grad_output = grad_output.contiguous()
    out_total = grad_output.numel()
    inv_divisor = 1.0 if divisor_override is None else 1.0 / float(divisor_override)
    with torch_device_fn.device(x.device):
        _avg_pool3d_backward_gather_kernel[(n_tiles,)](
            grad_output,
            padded,
            out_total - 1,
            depth,
            height,
            width,
            output_d,
            output_h,
            output_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            inv_divisor,
            KD=kernel_d,
            KH=kernel_h,
            KW=kernel_w,
            TD=triton.cdiv(kernel_d, stride_d),
            TH=triton.cdiv(kernel_h, stride_h),
            TW=triton.cdiv(kernel_w, stride_w),
            COUNT_INCLUDE_PAD=count_include_pad,
            HAS_DIVISOR=divisor_override is not None,
            BLOCK=block,
        )
    grad_input = padded[:total].view(x.shape)
    return grad_input.squeeze(0) if was_unbatched else grad_input
