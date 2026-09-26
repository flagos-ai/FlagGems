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
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _cubic_coefficients(t):
    a = -0.75
    p = t + 1.0
    w0 = ((a * p - 5.0 * a) * p + 8.0 * a) * p - 4.0 * a
    w1 = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    p = 1.0 - t
    w2 = ((a + 2.0) * p - (a + 3.0)) * p * p + 1.0
    p = 2.0 - t
    w3 = ((a * p - 5.0 * a) * p + 8.0 * a) * p - 4.0 * a
    return w0, w1, w2, w3


@triton.jit
def _cubic_axis_weight(
    index,
    output_index,
    INPUT: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
):
    scale = tl.full((), SCALE, tl.float32)
    output_f = output_index.to(tl.float32)
    if ALIGN:
        source = output_f * scale
    else:
        source = (output_f + 0.5) * scale - 0.5
    base = source.to(index.dtype)
    base -= (base.to(tl.float32) > source).to(index.dtype)
    fraction = source - base.to(tl.float32)
    w0, w1, w2, w3 = _cubic_coefficients(fraction)
    m0 = tl.minimum(tl.maximum(base - 1, 0), INPUT - 1) == index
    m1 = tl.minimum(tl.maximum(base, 0), INPUT - 1) == index
    m2 = tl.minimum(tl.maximum(base + 1, 0), INPUT - 1) == index
    m3 = tl.minimum(tl.maximum(base + 2, 0), INPUT - 1) == index
    weight = tl.where(m0, w0, 0.0)
    weight += tl.where(m1, w1, 0.0)
    weight += tl.where(m2, w2, 0.0)
    weight += tl.where(m3, w3, 0.0)
    touched = m0 | m1 | m2 | m3
    positive = (m0 & (w0 > 0)) | (m1 & (w1 > 0)) | (m2 & (w2 > 0)) | (m3 & (w3 > 0))
    negative = (m0 & (w0 < 0)) | (m1 & (w1 < 0)) | (m2 & (w2 < 0)) | (m3 & (w3 < 0))
    zero = (m0 & (w0 == 0)) | (m1 & (w1 == 0)) | (m2 & (w2 == 0)) | (m3 & (w3 == 0))
    nonfinite_nan = zero | (positive & negative)
    return weight, touched, nonfinite_nan


@triton.jit
def _cubic_contributors(
    index,
    INPUT: tl.constexpr,
    OUTPUT: tl.constexpr,
    SCALE: tl.constexpr,
    INVERSE: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if SCALE < (INPUT + 4) / 1073741824.0 or INPUT == 1:
        start = tl.full(index.shape, 0, index.dtype)
        end = tl.full(index.shape, OUTPUT, index.dtype)
    else:
        inverse = tl.full((), INVERSE, tl.float32)
        shift: tl.constexpr = 0.0 if ALIGN else 0.5
        center = (index.to(tl.float32) + shift) * inverse - shift
        start = (center - 2.0 * inverse).to(index.dtype) - 1
        end = (center + 2.0 * inverse).to(index.dtype) + 2
        start = tl.minimum(tl.maximum(start, 0), OUTPUT)
        end = tl.minimum(tl.maximum(end, 0), OUTPUT)
        start = tl.where(index == 0, 0, start)
        end = tl.where(index == INPUT - 1, OUTPUT, end)
    return start, end


@triton.jit
def _bicubic_backward_scalar(
    source,
    destination,
    total,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    IH_INV: tl.constexpr,
    IW_INV: tl.constexpr,
    ALIGN: tl.constexpr,
    COPY: tl.constexpr,
):
    start = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)
    for pixel in range(start, total, step):
        x = pixel % IW
        y = pixel // IW % IH
        c = pixel // (IW * IH) % C
        n = pixel // (IW * IH * C)
        destination_offset = n * D0 + c * D1 + y * D2 + x * D3
        if COPY:
            value = tl.load(source + n * S0 + c * S1 + y * S2 + x * S3)
            tl.store(destination + destination_offset, value)
        else:
            ys, ye = _cubic_contributors(y, IH, OH, SH, IH_INV, ALIGN)
            xs, xe = _cubic_contributors(x, IW, OW, SW, IW_INV, ALIGN)
            acc = tl.full((), 0.0, tl.float32)
            for oy in range(ys, ye):
                wy, ty, ny = _cubic_axis_weight(y, oy, IH, SH, ALIGN)
                for ox in range(xs, xe):
                    wx, tx, nx = _cubic_axis_weight(x, ox, IW, SW, ALIGN)
                    value = tl.load(source + n * S0 + c * S1 + oy * S2 + ox * S3)
                    value = tl.where(tx & ty, value.to(tl.float32), 0.0)
                    contribution = (value * wx) * wy
                    contribution = tl.where(
                        (nx | ny) & (tl.abs(value) == float("inf")),
                        float("nan"),
                        contribution,
                    )
                    acc += contribution
            tl.store(
                destination + destination_offset,
                acc.to(destination.dtype.element_ty),
            )


def _axis_inverse(size, scale):
    if scale < (size + 4) / 1073741824.0 or size == 1:
        return 0.0
    return 1.0 / scale


def _upsample_bicubic2d_backward_impl(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h,
    scales_w,
    grad_input,
):
    if len(input_size) != 4 or len(output_size) != 2:
        raise RuntimeError("Expected input_size with 4 and output_size with 2 elements")
    n, c, ih, iw = (int(size) for size in input_size)
    oh, ow = (int(size) for size in output_size)
    if min(ih, iw, oh, ow) <= 0 or n < 0 or c < 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if grad_output.ndim != 4 or tuple(grad_output.shape) != (n, c, oh, ow):
        raise RuntimeError("Expected grad_output to have the same 4D shape as output")
    if grad_output.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(
            "upsample_bicubic2d_backward requires a floating point dtype"
        )
    if any(scale is not None and not scale > 0.0 for scale in (scales_h, scales_w)):
        raise RuntimeError("scales_h and scales_w must be positive")
    shape = (n, c, ih, iw)
    if grad_input is None:
        grad_input = torch.empty(
            shape, dtype=grad_output.dtype, device=grad_output.device
        )
    else:
        if grad_input.device != grad_output.device:
            raise RuntimeError("Expected grad_input and grad_output on the same device")
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                "Expected grad_input and grad_output to have the same dtype"
            )
        if tuple(grad_input.shape) != shape:
            if grad_input.numel():
                warnings.warn(
                    "An output with one or more elements was resized because its shape "
                    "did not match the required output shape.",
                    UserWarning,
                    stacklevel=3,
                )
            grad_input.resize_(shape)
    if grad_input.numel() == 0:
        return grad_input

    if align_corners:
        sh = (ih - 1) / (oh - 1) if oh > 1 else 0.0
        sw = (iw - 1) / (ow - 1) if ow > 1 else 0.0
    else:
        sh = 1.0 / scales_h if scales_h is not None else ih / oh
        sw = 1.0 / scales_w if scales_w is not None else iw / ow
    copy = ih == oh and iw == ow
    total = n * c * ih * iw
    with torch_device_fn.device(grad_output.device):
        _bicubic_backward_scalar[(min(total, 65535),)](
            grad_output,
            grad_input,
            total,
            c,
            ih,
            iw,
            oh,
            ow,
            *grad_output.stride(),
            *grad_input.stride(),
            sh,
            sw,
            _axis_inverse(ih, sh),
            _axis_inverse(iw, sw),
            align_corners,
            copy,
            enable_fp_fusion=False,
        )
    return grad_input


def upsample_bicubic2d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_BICUBIC2D_BACKWARD")
    return _upsample_bicubic2d_backward_impl(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
        None,
    )


def upsample_bicubic2d_backward_grad_input(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_BICUBIC2D_BACKWARD.GRAD_INPUT")
    return _upsample_bicubic2d_backward_impl(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
        grad_input,
    )


def _install_into_flag_gems_ops():
    try:
        import flag_gems.ops as _fg_ops
    except Exception:  # pragma: no cover - defensive; import ordering
        return
    _fg_ops.upsample_bicubic2d_backward = upsample_bicubic2d_backward
    _fg_ops.upsample_bicubic2d_backward_grad_input = (
        upsample_bicubic2d_backward_grad_input
    )


_install_into_flag_gems_ops()
