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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_ASCEND_CANN_VERSION = None
if device.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _ASCEND_CANN_VERSION = get_cann_version()


@triton.jit
def _cubic_coefficients(t):
    # Keys' cubic convolution, matching ATen's coefficient evaluation order.
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
    FP64: tl.constexpr,
):
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    scale = tl.full((), SCALE, dtype)
    output_f = output_index.to(dtype)
    if ALIGN:
        source = output_f * scale
    else:
        source = (output_f + 0.5) * scale - 0.5
    base = source.to(index.dtype)
    base -= (base.to(dtype) > source).to(index.dtype)
    fraction = source - base.to(dtype)
    w0, w1, w2, w3 = _cubic_coefficients(fraction)
    # The coordinate is deliberately NOT clamped: only the four addresses are.
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
    # Repeated boundary taps can turn infinity into NaN before their weights
    # are combined. Keep that IEEE behavior without atomics or a native call.
    nonfinite_nan = zero | (positive & negative)
    return weight, touched, nonfinite_nan


@triton.jit
def _cubic_contributors(
    index,
    INPUT: tl.constexpr,
    OUTPUT: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
    FP64: tl.constexpr,
):
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    # Inverting a tiny positive scale can overflow even when all forward
    # coordinates are finite. A conservative full candidate range avoids that
    # inverse and its float-to-index conversion; geometric tap masks still
    # select the actual contributors, including zero-weight nonfinite taps.
    if SCALE < (INPUT + 4) / 1073741824.0 or INPUT == 1:
        start = tl.full(index.shape, 0, index.dtype)
        end = tl.full(index.shape, OUTPUT, index.dtype)
    else:
        inverse = tl.full((), 1.0 / SCALE, dtype)
        shift: tl.constexpr = 0.0 if ALIGN else 0.5
        center = (index.to(dtype) + shift) * inverse - shift
        # An extra output element on each side protects inverse-rounding boundaries.
        start = (center - 2.0 * inverse).to(index.dtype) - 1
        end = (center + 2.0 * inverse).to(index.dtype) + 2
        start = tl.minimum(tl.maximum(start, 0), OUTPUT)
        end = tl.minimum(tl.maximum(end, 0), OUTPUT)
        start = tl.where(index == 0, 0, start)
        end = tl.where(index == INPUT - 1, OUTPUT, end)
    return start, end


@libentry()
@triton.jit
def _upsample_bicubic2d_backward_axis(
    source,
    destination,
    numel,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    INPUT: tl.constexpr,
    OUTPUT: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
    X_AXIS: tl.constexpr,
    FP64: tl.constexpr,
    COPY: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    itype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    offset = tl.program_id(0).to(itype) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < numel
    x = offset % W
    y = offset // W % H
    c = offset // (W * H) % C
    n = offset // (W * H * C)
    destination_offset = n * D0 + c * D1 + y * D2 + x * D3
    if COPY:
        value = tl.load(
            source + n * S0 + c * S1 + y * S2 + x * S3, mask=valid, other=0.0
        )
        tl.store(destination + destination_offset, value, mask=valid)
    else:
        dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
        index = x if X_AXIS else y
        start, end = _cubic_contributors(index, INPUT, OUTPUT, SCALE, ALIGN, FP64)
        count = tl.max(tl.where(valid, end - start, 0), 0)
        acc = tl.full((BLOCK,), 0.0, dtype)
        if X_AXIS:
            source_base = n * S0 + c * S1 + y * S2
            stride: tl.constexpr = S3
        else:
            source_base = n * S0 + c * S1 + x * S3
            stride: tl.constexpr = S2
        for contributor in range(count):
            output_index = start + contributor
            weight, touched, nonfinite_nan = _cubic_axis_weight(
                index, output_index, INPUT, SCALE, ALIGN, FP64
            )
            value = tl.load(
                source + source_base + output_index * stride,
                mask=valid & (output_index < end) & touched,
                other=0.0,
            ).to(dtype)
            contribution = value * weight
            contribution = tl.where(
                nonfinite_nan & (tl.abs(value) == float("inf")),
                float("nan"),
                contribution,
            )
            acc += contribution
        tl.store(destination + destination_offset, acc, mask=valid)


@libentry()
@triton.jit
def _upsample_bicubic2d_backward_gather(
    source,
    destination,
    numel,
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
    ALIGN: tl.constexpr,
    FP64: tl.constexpr,
    INDEX64: tl.constexpr,
    RH: tl.constexpr,
    RW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    itype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    offset = tl.program_id(0).to(itype) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < numel
    x = offset % IW
    y = offset // IW % IH
    c = offset // (IW * IH) % C
    n = offset // (IW * IH * C)
    ys, ye = _cubic_contributors(y, IH, OH, SH, ALIGN, FP64)
    xs, xe = _cubic_contributors(x, IW, OW, SW, ALIGN, FP64)
    r = tl.arange(0, RH * RW)
    oy = ys[:, None] + (r // RW)[None, :]
    ox = xs[:, None] + (r % RW)[None, :]
    wy, ty, ny = _cubic_axis_weight(y[:, None], oy, IH, SH, ALIGN, FP64)
    wx, tx, nx = _cubic_axis_weight(x[:, None], ox, IW, SW, ALIGN, FP64)
    source_offset = n[:, None] * S0 + c[:, None] * S1 + oy * S2 + ox * S3
    mask = valid[:, None] & (oy < ye[:, None]) & (ox < xe[:, None])
    mask = mask & tx & ty
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    value = tl.load(source + source_offset, mask, other=0.0).to(dtype)
    contribution = (value * wx) * wy
    contribution = tl.where(
        (nx | ny) & (tl.abs(value) == float("inf")), float("nan"), contribution
    )
    result = tl.sum(contribution, 1)
    tl.store(destination + n * D0 + c * D1 + y * D2 + x * D3, result, valid)


@libentry()
@triton.jit
def _upsample_bicubic2d_backward_wide_scalar(
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
    ALIGN: tl.constexpr,
    FP64: tl.constexpr,
    COPY: tl.constexpr,
    BF16_RNE: tl.constexpr = False,
):
    # Vector offsets spanning 2**31 elements failed on tested CANN 8.5/9.0
    # and Hygon runtimes. The cause is not yet established; use scalar addresses.
    # Bound the launch and keep the complete loop/index/address chain in int64.
    for pixel in range(
        tl.program_id(0).to(tl.int64),
        total.to(tl.int64),
        tl.num_programs(0).to(tl.int64),
    ):
        x = pixel % IW
        y = pixel // IW % IH
        c = pixel // (IW * IH) % C
        n = pixel // (IW * IH * C)
        destination_offset = n * D0 + c * D1 + y * D2 + x * D3
        if COPY:
            value = tl.load(source + n * S0 + c * S1 + y * S2 + x * S3)
            tl.store(destination + destination_offset, value)
        else:
            ys, ye = _cubic_contributors(y, IH, OH, SH, ALIGN, FP64)
            xs, xe = _cubic_contributors(x, IW, OW, SW, ALIGN, FP64)
            dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
            acc = tl.full((), 0.0, dtype)
            for oy in range(ys, ye):
                wy, ty, ny = _cubic_axis_weight(y, oy, IH, SH, ALIGN, FP64)
                for ox in range(xs, xe):
                    wx, tx, nx = _cubic_axis_weight(x, ox, IW, SW, ALIGN, FP64)
                    value = tl.load(
                        source + n * S0 + c * S1 + oy * S2 + ox * S3,
                        mask=tx & ty,
                        other=0.0,
                    ).to(dtype)
                    contribution = (value * wx) * wy
                    contribution = tl.where(
                        (nx | ny) & (tl.abs(value) == float("inf")),
                        float("nan"),
                        contribution,
                    )
                    acc += contribution
            if BF16_RNE:
                # CANN 8.5 scalar FP32-to-BF16 casts round halfway values away
                # from zero even with explicit rtne. Construct the RNE bits.
                bits = acc.to(tl.int32, bitcast=True)
                rounded = bits + 0x7FFF + ((bits >> 16) & 1)
                output = tl.where(
                    (bits & 0x7FFFFFFF) > 0x7F800000,
                    (bits >> 16) | 0x40,
                    rounded >> 16,
                ).to(tl.int16)
                tl.store(
                    destination + destination_offset,
                    output.to(tl.bfloat16, bitcast=True),
                )
            else:
                tl.store(destination + destination_offset, acc)


@libentry()
@triton.jit
def _bicubic_unit_input(
    source,
    destination,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    itype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    plane = tl.program_id(0).to(itype)
    n = plane // C
    c = plane % C
    r = tl.arange(0, BLOCK).to(itype)
    value = tl.load(
        source + n * S0 + c * S1 + r // OW * S2 + r % OW * S3, r < OH * OW, other=0.0
    ).to(tl.float32)
    result = tl.sum(value, 0)
    infinite = tl.sum((tl.abs(value) == float("inf")).to(tl.int32), 0) > 0
    result = tl.where(infinite, float("nan"), result)
    tl.store(destination + n * D0 + c * D1, result)


# CANN 8.5 needs the earlier masked-load axis layout: its row-gather
# lowering raised a measured UB vector alignment exception for short rows.
@libentry()
@triton.jit
def _upsample_bicubic2d_backward_axis_vector(
    source,
    destination,
    numel,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    INPUT: tl.constexpr,
    OUTPUT: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
    X_AXIS: tl.constexpr,
    FP64: tl.constexpr,
    INDEX64: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    itype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    for tile in range(
        tl.program_id(0), (numel + BLOCK - 1) // BLOCK, tl.num_programs(0)
    ):
        offset = tile.to(itype) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < numel
        x = offset % W
        y = offset // W % H
        c = offset // (W * H) % C
        n = offset // (W * H * C)
        index = x if X_AXIS else y
        start, end = _cubic_contributors(index, INPUT, OUTPUT, SCALE, ALIGN, FP64)
        output_index = start[None, :] + tl.arange(0, R)[:, None]
        weight, touched, nonfinite_nan = _cubic_axis_weight(
            index[None, :], output_index, INPUT, SCALE, ALIGN, FP64
        )
        if X_AXIS:
            source_base = n * S0 + c * S1 + y * S2
            stride: tl.constexpr = S3
        else:
            source_base = n * S0 + c * S1 + x * S3
            stride: tl.constexpr = S2
        dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
        value = tl.load(
            source + source_base[None, :] + output_index * stride,
            valid[None, :] & (output_index < end[None, :]) & touched,
            other=0.0,
        ).to(dtype)
        contribution = value * weight
        contribution = tl.where(
            nonfinite_nan & (tl.abs(value) == float("inf")),
            float("nan"),
            contribution,
        )
        result = tl.sum(contribution, 0)
        tl.store(destination + n * D0 + c * D1 + y * D2 + x * D3, result, valid)


@libentry()
@triton.jit
def _bicubic_copy_bounded(
    source,
    destination,
    numel,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    itype: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    if INDEX64:
        # Widen before ceil division: numel can still arrive as an int32 scalar.
        total = numel.to(tl.int64)
    else:
        total = numel
    for tile in range(
        tl.program_id(0), (total + BLOCK - 1) // BLOCK, tl.num_programs(0)
    ):
        i = tile.to(itype) * BLOCK + tl.arange(0, BLOCK)
        if CONTIGUOUS:
            source_offset = i
            destination_offset = i
        else:
            x = i % W
            y = i // W % H
            c = i // (W * H) % C
            n = i // (W * H * C)
            source_offset = n * S0 + c * S1 + y * S2 + x * S3
            destination_offset = n * D0 + c * D1 + y * D2 + x * D3
        value = tl.load(source + source_offset, i < total, other=0.0)
        tl.store(destination + destination_offset, value, i < total)


@libentry()
@triton.jit
def _upsample_bicubic2d_backward_row_x(
    source,
    destination,
    NC: tl.constexpr,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    IW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    SW: tl.constexpr,
    ALIGN: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
):
    local = tl.arange(0, L)
    r = tl.arange(0, R)[:, None]
    for row in range(tl.program_id(0), NC * OH, tl.num_programs(0)):
        oy = row % OH
        c = row // OH % C
        n = row // (OH * C)
        data = tl.load(
            source + n * S0 + c * S1 + oy * S2 + local * S3, local < OW, other=0
        ).to(tl.float32)
        for tile in range((IW + B - 1) // B):
            x = tile * B + tl.arange(0, B)
            start, end = _cubic_contributors(x, IW, OW, SW, ALIGN, False)
            ox = start[None, :] + r
            wx, touched, nan = _cubic_axis_weight(x[None, :], ox, IW, SW, ALIGN, False)
            valid = (x[None, :] < IW) & (ox < end[None, :]) & touched
            localidx = tl.where(valid, ox, 0).reshape((R * B,)).to(tl.int32)
            value = tl.gather(data, localidx, 0).reshape((R, B))
            value = tl.where(valid, value, 0.0)
            term = value * wx
            term = tl.where(nan & (tl.abs(value) == float("inf")), float("nan"), term)
            result = tl.sum(tl.where(valid, term, 0.0), 0)
            tl.store(destination + row * IW + x, result, x < IW)


@libentry()
@triton.jit
def _upsample_bicubic2d_backward_row_y(
    source,
    destination,
    NC: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    SH: tl.constexpr,
    ALIGN: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.arange(0, R)
    for row in range(tl.program_id(0), NC * IH, tl.num_programs(0)):
        y = row % IH
        c = row // IH % C
        n = row // (IH * C)
        start, end = _cubic_contributors(y, IH, OH, SH, ALIGN, False)
        oy = start + r
        wy, touched, nan = _cubic_axis_weight(y, oy, IH, SH, ALIGN, False)
        for tile in range((IW + B - 1) // B):
            x = tile * B + tl.arange(0, B)
            valid = (x[None, :] < IW) & (oy[:, None] < end) & touched[:, None]
            value = tl.load(
                source + ((n * C + c) * OH + oy[:, None]) * IW + x[None, :],
                valid,
                other=0,
            ).to(tl.float32)
            term = value * wy[:, None]
            term = tl.where(
                nan[:, None] & (tl.abs(value) == float("inf")), float("nan"), term
            )
            result = tl.sum(tl.where(valid, term, 0.0), 0)
            tl.store(destination + n * D0 + c * D1 + y * D2 + x * D3, result, x < IW)


def _max_contributors(input_size, output_size, scale, align_corners):
    if scale < (input_size + 4) / 1073741824.0 or input_size == 1:
        return output_size
    inverse = 1.0 / scale
    shift = 0.0 if align_corners else 0.5
    first_center = shift * inverse - shift
    last_center = (input_size - 1 + shift) * inverse - shift
    first_end = min(max(int(first_center + 2 * inverse) + 2, 0), output_size)
    last_start = min(max(int(last_center - 2 * inverse) - 1, 0), output_size)
    return min(
        output_size,
        max(math.ceil(4 * inverse) + 4, first_end, output_size - last_start),
    )


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
    if grad_output.dtype == torch.float64 and device.vendor_name == "iluvatar":
        raise RuntimeError(
            "upsample_bicubic2d_backward requires FP64 arithmetic unsupported on Iluvatar"
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
    fp64 = grad_output.dtype == torch.float64
    copy = ih == oh and iw == ow
    block = 1024 if copy and device.vendor_name == "mthreads" else 128
    # Address bounds use elements, as do Triton pointer offsets. Keep the full-width
    # path for large strided views and large coordinate/index ranges.
    address_bound = max(
        sum(
            (size - 1) * stride
            for size, stride in zip(grad_output.shape, grad_output.stride())
        ),
        sum(
            (size - 1) * stride
            for size, stride in zip(grad_input.shape, grad_input.stride())
        ),
    )
    index_bound = max(
        grad_output.numel(),
        grad_input.numel(),
        n * c * oh * iw,
        address_bound,
        (oh + 4096) * sh,
        (ow + 4096) * sw,
    )
    index64 = index_bound >= 2**31 - 4096 or (fp64 and device.vendor_name == "mthreads")
    # Ascend's row gathers return incorrect results for the full contributor range
    # used by extremely small effective scales. Preserve the negative half-pixel
    # coordinates and weights through the ordered scalar implementation.
    wide_scalar = (
        device.vendor_name in ("hygon", "ascend") and address_bound >= 2**31
    ) or (
        device.vendor_name == "ascend"
        and not fp64
        and not copy
        and not align_corners
        and (sh < (ih + 4) / 1073741824.0 or sw < (iw + 4) / 1073741824.0)
    )
    with torch_device_fn.device(grad_output.device):
        if (
            device.vendor_name == "mthreads"
            and not fp64
            and not copy
            and ih == 1
            and iw == 1
            and oh * ow <= 4096
        ):
            # All taps address the single input pixel and their weights sum to one.
            # An infinite gradient still produces NaN through zero or mixed-sign taps.
            _bicubic_unit_input[(n * c,)](
                grad_output,
                grad_input,
                c,
                oh,
                ow,
                *grad_output.stride(),
                *grad_input.stride()[:2],
                index64,
                triton.next_power_of_2(oh * ow),
                enable_fp_fusion=False,
            )
        elif wide_scalar:
            _upsample_bicubic2d_backward_wide_scalar[(min(grad_input.numel(), 4096),)](
                grad_output,
                grad_input,
                grad_input.numel(),
                c,
                ih,
                iw,
                oh,
                ow,
                *grad_output.stride(),
                *grad_input.stride(),
                sh,
                sw,
                align_corners,
                fp64,
                copy,
                BF16_RNE=(
                    device.vendor_name == "ascend"
                    and _ASCEND_CANN_VERSION == "8.5.0"
                    and grad_input.dtype == torch.bfloat16
                    and not copy
                ),
                enable_fp_fusion=False,
            )
        elif copy and device.vendor_name == "ascend":
            cores = triton.runtime.driver.active.utils.get_device_properties(
                grad_output.device.index
            )["num_vectorcore"]
            contiguous = grad_output.is_contiguous() and grad_input.is_contiguous()
            copy_block = 8192 if contiguous else 256
            # Include the ceil-division addition, even when all tensor addresses
            # and the common coordinate bound still fit in signed int32.
            copy_index64 = index64 or grad_input.numel() >= 2**31 - copy_block
            _bicubic_copy_bounded[
                (min(cores, triton.cdiv(grad_input.numel(), copy_block)),)
            ](
                grad_output,
                grad_input,
                grad_input.numel(),
                c,
                ih,
                iw,
                *grad_output.stride(),
                *grad_input.stride(),
                contiguous,
                copy_index64,
                copy_block,
                enable_fp_fusion=False,
            )
        elif copy:
            _upsample_bicubic2d_backward_axis[
                (triton.cdiv(grad_input.numel(), block),)
            ](
                grad_output,
                grad_input,
                grad_input.numel(),
                c,
                ih,
                iw,
                iw,
                ow,
                *grad_output.stride(),
                *grad_input.stride(),
                sw,
                align_corners,
                True,
                fp64,
                True,
                index64,
                block,
                enable_fp_fusion=False,
            )
        elif (
            device.vendor_name == "ascend"
            and _ASCEND_CANN_VERSION == "8.5.0"
            and not fp64
            and not index64
        ):
            intermediate = torch.empty(
                (n, c, oh, iw), device=grad_output.device, dtype=torch.float32
            )
            cores = triton.runtime.driver.active.utils.get_device_properties(
                grad_output.device.index
            )["num_vectorcore"]
            rh = triton.next_power_of_2(_max_contributors(ih, oh, sh, align_corners))
            rw = triton.next_power_of_2(_max_contributors(iw, ow, sw, align_corners))
            bx = min(256, max(1, 2048 // rw))
            by = min(256, max(1, 2048 // rh))
            _upsample_bicubic2d_backward_axis_vector[
                (min(cores, triton.cdiv(intermediate.numel(), bx)),)
            ](
                grad_output,
                intermediate,
                intermediate.numel(),
                c,
                oh,
                iw,
                iw,
                ow,
                *grad_output.stride(),
                *intermediate.stride(),
                sw,
                align_corners,
                True,
                fp64,
                index64,
                rw,
                bx,
                enable_fp_fusion=False,
            )
            _upsample_bicubic2d_backward_axis_vector[
                (min(cores, triton.cdiv(grad_input.numel(), by)),)
            ](
                intermediate,
                grad_input,
                grad_input.numel(),
                c,
                ih,
                iw,
                ih,
                oh,
                *intermediate.stride(),
                *grad_input.stride(),
                sh,
                align_corners,
                False,
                fp64,
                index64,
                rh,
                by,
                enable_fp_fusion=False,
            )
        elif device.vendor_name == "ascend" and not fp64:
            rh = triton.next_power_of_2(_max_contributors(ih, oh, sh, align_corners))
            rw = triton.next_power_of_2(_max_contributors(iw, ow, sw, align_corners))
            if (
                index64
                or ow > 4096
                or max(rh, rw) > 1024
                or max(rh, rw) * max(iw, 8) > 4096
            ):
                # Keep scalar addressing when a complete row patch cannot fit
                # in the bounded local buffers used by the vector path.
                _upsample_bicubic2d_backward_wide_scalar[
                    (min(grad_input.numel(), 4096),)
                ](
                    grad_output,
                    grad_input,
                    grad_input.numel(),
                    c,
                    ih,
                    iw,
                    oh,
                    ow,
                    *grad_output.stride(),
                    *grad_input.stride(),
                    sh,
                    sw,
                    align_corners,
                    fp64,
                    False,
                    BF16_RNE=(
                        device.vendor_name == "ascend"
                        and _ASCEND_CANN_VERSION == "8.5.0"
                        and grad_input.dtype == torch.bfloat16
                        and not copy
                    ),
                    enable_fp_fusion=False,
                )
            else:
                intermediate = torch.empty(
                    (n, c, oh, iw), device=grad_output.device, dtype=torch.float32
                )
                cores = triton.runtime.driver.active.utils.get_device_properties(
                    grad_output.device.index
                )["num_vectorcore"]
                # Short non-scalar vector rows need 32-byte FP32 alignment.
                # B=4 triggered a measured UB vector alignment fault on CANN 9.0.
                bx = max(8, min(triton.next_power_of_2(iw), max(1, 1024 // rw)))
                by = max(8, min(triton.next_power_of_2(iw), max(1, 4096 // rh)))
                _upsample_bicubic2d_backward_row_x[(min(cores, n * c * oh),)](
                    grad_output,
                    intermediate,
                    n * c,
                    c,
                    oh,
                    ow,
                    iw,
                    *grad_output.stride(),
                    sw,
                    align_corners,
                    rw,
                    bx,
                    triton.next_power_of_2(ow),
                    enable_fp_fusion=False,
                )
                # A strided FP32 row store is much slower on CANN 9.0. Compute
                # contiguous rows, then copy through the caller's real strides.
                strided_f32 = (
                    _ASCEND_CANN_VERSION == "9.0.0"
                    and grad_input.dtype == torch.float32
                    and not grad_input.is_contiguous()
                )
                row_destination = (
                    torch.empty(shape, dtype=torch.float32, device=grad_output.device)
                    if strided_f32
                    else grad_input
                )
                _upsample_bicubic2d_backward_row_y[(min(cores, n * c * ih),)](
                    intermediate,
                    row_destination,
                    n * c,
                    c,
                    ih,
                    iw,
                    oh,
                    *row_destination.stride(),
                    sh,
                    align_corners,
                    rh,
                    by,
                    enable_fp_fusion=False,
                )
                if strided_f32:
                    _bicubic_copy_bounded[
                        (min(cores, triton.cdiv(grad_input.numel(), 256)),)
                    ](
                        row_destination,
                        grad_input,
                        grad_input.numel(),
                        c,
                        ih,
                        iw,
                        *row_destination.stride(),
                        *grad_input.stride(),
                        False,
                        index64,
                        256,
                        enable_fp_fusion=False,
                    )
        elif (
            not fp64
            and grad_input.numel() <= 16384
            and _max_contributors(ih, oh, sh, align_corners)
            * _max_contributors(iw, ow, sw, align_corners)
            <= 4096
        ):
            rh = triton.next_power_of_2(_max_contributors(ih, oh, sh, align_corners))
            rw = triton.next_power_of_2(_max_contributors(iw, ow, sw, align_corners))
            gather_block = min(128, max(1, 2048 // (rh * rw)))
            if device.vendor_name == "mthreads":
                # The MUSA 32-bit gather tile (32, 64) miscompiles. Smaller
                # tiles also avoid the measured register-pressure slowdown.
                gather_block = 4 if rh * rw <= 64 else 1
            _upsample_bicubic2d_backward_gather[
                (triton.cdiv(grad_input.numel(), gather_block),)
            ](
                grad_output,
                grad_input,
                grad_input.numel(),
                c,
                ih,
                iw,
                oh,
                ow,
                *grad_output.stride(),
                *grad_input.stride(),
                sh,
                sw,
                align_corners,
                fp64,
                index64,
                rh,
                rw,
                gather_block,
                enable_fp_fusion=False,
            )
        else:
            intermediate = torch.empty(
                (n, c, oh, iw),
                device=grad_output.device,
                dtype=torch.float64 if fp64 else torch.float32,
            )
            _upsample_bicubic2d_backward_axis[
                (triton.cdiv(intermediate.numel(), block),)
            ](
                grad_output,
                intermediate,
                intermediate.numel(),
                c,
                oh,
                iw,
                iw,
                ow,
                *grad_output.stride(),
                *intermediate.stride(),
                sw,
                align_corners,
                True,
                fp64,
                False,
                index64,
                block,
                enable_fp_fusion=False,
            )
            _upsample_bicubic2d_backward_axis[
                (triton.cdiv(grad_input.numel(), block),)
            ](
                intermediate,
                grad_input,
                grad_input.numel(),
                c,
                ih,
                iw,
                ih,
                oh,
                *intermediate.stride(),
                *grad_input.stride(),
                sh,
                align_corners,
                False,
                fp64,
                False,
                index64,
                block,
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
    logger.debug("GEMS UPSAMPLE_BICUBIC2D_BACKWARD")
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
    logger.debug("GEMS UPSAMPLE_BICUBIC2D_BACKWARD.GRAD_INPUT")
    return _upsample_bicubic2d_backward_impl(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
        grad_input,
    )
