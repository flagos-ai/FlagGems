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
import math
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _bilinear_backward_isinf(value):
    if value.dtype == tl.float64:
        bits = value.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        return bits == 0x7FF0000000000000
    else:
        bits = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        return bits == 0x7F800000


@triton.jit
def _bilinear_backward_weight(
    output_index,
    input_index,
    INPUT_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    ACC: tl.constexpr,
):
    scale = tl.full((), SCALE, ACC)
    if ALIGN_CORNERS:
        real = output_index.to(ACC) * scale
    else:
        real = tl.maximum((output_index.to(ACC) + 0.5) * scale - 0.5, 0.0)
    lower = tl.minimum(real, INPUT_SIZE - 1).to(input_index.dtype)
    lower = tl.minimum(lower, INPUT_SIZE - 1)
    upper = tl.minimum(lower + 1, INPUT_SIZE - 1)
    fraction = tl.minimum(real - lower.to(ACC), 1.0)
    weight = tl.where(input_index == lower, 1.0 - fraction, 0.0)
    weight += tl.where(input_index == upper, fraction, 0.0)
    if ACC == tl.float64:
        bits = fraction.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
        zero_fraction = bits == 0
        one_fraction = bits == 0x3FF0000000000000
    else:
        zero_fraction = fraction == 0.0
        one_fraction = fraction == 1.0
    zero_tap = ((input_index == lower) & one_fraction) | (
        (input_index == upper) & zero_fraction
    )
    return weight, (input_index == lower) | (input_index == upper), zero_tap


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_kernel(
    GradOutput,
    GradInput,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    IS3: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FP64: tl.constexpr,
    INDEX64: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    COPY: tl.constexpr,
    ZERO: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.cdiv(N * C * IH * IW, PROGRAMS * BLOCK)):
        block_id = tl.program_id(0) + tile * PROGRAMS
        if INDEX64:
            offsets = block_id.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        else:
            offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N * C * IH * IW
        if CHANNELS_LAST:
            c = offsets % C
            x = offsets // C % IW
            y = offsets // (C * IW) % IH
            n = offsets // (C * IW * IH)
        else:
            x = offsets % IW
            y = offsets // IW % IH
            c = offsets // (IW * IH) % C
            n = offsets // (IW * IH * C)
        destination = GradInput + n * IS0 + c * IS1 + y * IS2 + x * IS3
        source = GradOutput + n * GS0 + c * GS1
        if ZERO:
            tl.store(destination, 0.0, valid)
        elif COPY:
            value = tl.load(source + y * GS2 + x * GS3, valid, other=0)
            tl.store(destination, value, valid)
        else:
            if FP64:
                acc_dtype: tl.constexpr = tl.float64
            else:
                acc_dtype: tl.constexpr = tl.float32
            shift: tl.constexpr = 0.0 if ALIGN_CORNERS else 0.5
            if SH == 0.0 or IH == 1:
                start_y = tl.full((BLOCK,), 0, offsets.dtype)
            else:
                inv_h = tl.full((), 1.0 / SH, acc_dtype)
                start_y = tl.maximum(
                    tl.floor((y.to(acc_dtype) - 1.0 + shift) * inv_h - shift).to(
                        offsets.dtype
                    ),
                    0,
                )
            if SW == 0.0 or IW == 1:
                start_x = tl.full((BLOCK,), 0, offsets.dtype)
            else:
                inv_w = tl.full((), 1.0 / SW, acc_dtype)
                start_x = tl.maximum(
                    tl.floor((x.to(acc_dtype) - 1.0 + shift) * inv_w - shift).to(
                        offsets.dtype
                    ),
                    0,
                )
            result = tl.full((BLOCK,), 0.0, acc_dtype)
            for dy in range(KH):
                oy = start_y + dy
                wy, match_y, zero_y = _bilinear_backward_weight(
                    oy, y, IH, SH, ALIGN_CORNERS, acc_dtype
                )
                for dx in range(KW):
                    ox = start_x + dx
                    wx, match_x, zero_x = _bilinear_backward_weight(
                        ox, x, IW, SW, ALIGN_CORNERS, acc_dtype
                    )
                    active = valid & (oy < OH) & (ox < OW) & match_y & match_x
                    value = tl.load(source + oy * GS2 + ox * GS3, active, other=0).to(
                        acc_dtype
                    )
                    contribution = (wy * wx) * value
                    contribution = tl.where(
                        (zero_y | zero_x) & _bilinear_backward_isinf(value),
                        float("nan"),
                        contribution,
                    )
                    result += tl.where(active, contribution, 0.0)
            tl.store(destination, result, valid)


def upsample_bilinear2d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
    *,
    grad_input=None,
):
    logging.getLogger("flag_gems.ops.upsample_bilinear2d_backward").debug(
        "GEMS UPSAMPLE_BILINEAR2D_BACKWARD"
    )
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_BILINEAR2D_BACKWARD")
    if len(output_size) != 2 or len(input_size) != 4:
        raise RuntimeError(
            "output_size must have 2 elements and input_size must have 4 elements"
        )
    n, c, ih, iw = input_size
    oh, ow = output_size
    if min(ih, iw, oh, ow) <= 0 or n < 0 or c < 0:
        raise RuntimeError("input and output spatial sizes must be greater than 0")
    if grad_output.ndim != 4:
        raise RuntimeError("Expected grad_output to be a tensor of dimension 4")
    if tuple(grad_output.shape) != (n, c, oh, ow):
        raise RuntimeError("Expected grad_output to have the same shape as output")
    if any(scale is not None and not scale > 0 for scale in (scales_h, scales_w)):
        raise RuntimeError("scales_h and scales_w must be greater than 0")
    if grad_output.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(
            "upsample_bilinear2d_backward requires a floating point dtype"
        )

    if grad_input is None:
        if grad_output.is_contiguous(memory_format=torch.channels_last):
            grad_input = torch.empty_strided(
                input_size,
                (c * ih * iw, 1, c * iw, c),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
        else:
            grad_input = torch.empty(
                input_size, dtype=grad_output.dtype, device=grad_output.device
            )
    else:
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                "Expected grad_input to have the same dtype as grad_output"
            )
        if grad_input.device != grad_output.device:
            raise RuntimeError("Expected all tensors to be on the same device")
        if tuple(grad_input.shape) != tuple(input_size):
            if grad_input.numel() != 0:
                warnings.warn(
                    "An output with one or more elements was resized because its shape did not match input_size",
                    UserWarning,
                    stacklevel=2,
                )
            grad_input.resize_(input_size)
    if n == 0 or c == 0:
        return grad_input

    copy = ih == oh and iw == ow
    if align_corners:
        sh = (ih - 1) / (oh - 1) if oh > 1 else 0.0
        sw = (iw - 1) / (ow - 1) if ow > 1 else 0.0
    else:
        sh = 1.0 / scales_h if scales_h is not None else ih / oh
        sw = 1.0 / scales_w if scales_w is not None else iw / ow
    if not align_corners:
        if (oh - 0.5) * sh <= 0.25:
            sh = 0.0
        if (ow - 0.5) * sw <= 0.25:
            sw = 0.0
    kh = oh if sh == 0 or ih == 1 else min(oh, math.ceil(2 / sh) + 2)
    kw = ow if sw == 0 or iw == 1 else min(ow, math.ceil(2 / sw) + 2)
    shift = 0.0 if align_corners else 0.5
    if sh > 0 and ih > 1:
        tail_h = oh - max(0, math.floor((ih - 2 + shift) / sh - shift))
        kh = min(oh, max(kh, tail_h))
    if sw > 0 and iw > 1:
        tail_w = ow - max(0, math.floor((iw - 2 + shift) / sw - shift))
        kw = min(ow, max(kw, tail_w))
    block = 128
    gs = grad_output.stride()
    strides = grad_input.stride()
    largest_offset = max(
        (n - 1) * gs[0] + (c - 1) * gs[1] + (oh - 1) * gs[2] + (ow - 1) * gs[3],
        (n - 1) * strides[0]
        + (c - 1) * strides[1]
        + (ih - 1) * strides[2]
        + (iw - 1) * strides[3],
    )
    index64 = (
        max(n * c * ih * iw, n * c * oh * ow, largest_offset) + block >= 2**31
        or max(ih, iw, oh, ow) >= 2**24
    )
    with torch_device_fn.device(grad_output.device):
        programs = triton.cdiv(n * c * ih * iw, block)
        _upsample_bilinear2d_backward_kernel[(programs,)](
            grad_output,
            grad_input,
            n,
            c,
            ih,
            iw,
            oh,
            ow,
            *gs,
            *strides,
            sh,
            sw,
            align_corners,
            kh,
            kw,
            grad_output.dtype == torch.float64,
            index64,
            strides[1] == 1,
            copy,
            False,
            programs,
            block,
            enable_fp_fusion=False,
        )
    return grad_input


def upsample_bilinear2d_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    return upsample_bilinear2d_backward(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
        grad_input=grad_input,
    )


def _install_into_flag_gems_ops():
    try:
        import flag_gems.ops as _fg_ops
    except Exception:  # pragma: no cover - defensive; import ordering
        return
    _fg_ops.upsample_bilinear2d_backward = upsample_bilinear2d_backward
    _fg_ops.upsample_bilinear2d_backward_grad_input = (
        upsample_bilinear2d_backward_grad_input
    )


_install_into_flag_gems_ops()
