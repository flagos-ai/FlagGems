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

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_ILUVATAR = tl.constexpr(runtime_device.vendor_name == "iluvatar")

_CANN_VERSION = None
_CANN_SIZE_RATIO = False
if runtime_device.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _CANN_VERSION = get_cann_version()
    _CANN_SIZE_RATIO = _CANN_VERSION in ("8.5.0", "9.0.0")


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
    if _ILUVATAR:
        # Preserve the coordinate arithmetic validated on CoreX 4.4.
        position = input_index.to(ACC)
        last = input_index == INPUT_SIZE - 1
        right = real >= position
        beyond = real >= position + 1.0
        weight = tl.where(right, 1.0 - (real - position), real - (position - 1.0))
        weight = tl.where(last & right, 1.0, weight)
        match = (real >= position - 1.0) & (last | ~beyond)
        zero_tap = last & ((real == position) | beyond)
        return weight, match, zero_tap
    else:
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
            # Each lane owns one input pixel, so no atomics or scratch buffer are needed.
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
                    # Merging clamped taps must preserve an original 0 * Inf update.
                    contribution = tl.where(
                        (zero_y | zero_x) & _bilinear_backward_isinf(value),
                        float("nan"),
                        contribution,
                    )
                    result += tl.where(active, contribution, 0.0)
            tl.store(destination, result, valid)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_ascend_copy(
    Source,
    Destination,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Matching dense layouts can copy their physical storage in linear tiles.
    for start in range(tl.program_id(0) * BLOCK, TOTAL, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        values = tl.load(Source + offsets, offsets < TOTAL, other=0)
        tl.store(Destination + offsets, values, offsets < TOTAL)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_scalar_kernel(
    GradOutput,
    GradInput,
    TOTAL: tl.constexpr,
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
    ALIGN: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    ACC: tl.constexpr,
    COPY: tl.constexpr,
):
    # Scalar 64-bit address arithmetic avoids DTK's wide vector-store truncation.
    for offset in range(tl.program_id(0).to(tl.int64), TOTAL, tl.num_programs(0)):
        x = offset % IW
        y = offset // IW % IH
        c = offset // (IH * IW) % C
        n = offset // (IH * IW * C)
        source = GradOutput + n * GS0 + c * GS1
        if COPY:
            result = tl.load(source + y * GS2 + x * GS3)
        else:
            shift: tl.constexpr = 0.0 if ALIGN else 0.5
            if SH == 0 or IH == 1:
                start_y = tl.full((), 0, tl.int64)
            else:
                start_y = tl.maximum(
                    tl.floor(
                        (y.to(ACC) - 1.0 + shift) * tl.full((), 1.0 / SH, ACC) - shift
                    ),
                    0.0,
                ).to(tl.int64)
            if SW == 0 or IW == 1:
                start_x = tl.full((), 0, tl.int64)
            else:
                start_x = tl.maximum(
                    tl.floor(
                        (x.to(ACC) - 1.0 + shift) * tl.full((), 1.0 / SW, ACC) - shift
                    ),
                    0.0,
                ).to(tl.int64)
            result = tl.full((), 0.0, ACC)
            for dy in range(KH):
                oy = start_y + dy
                wy, match_y, zero_y = _bilinear_backward_weight(
                    oy, y, IH, SH, ALIGN, ACC
                )
                for dx in range(KW):
                    ox = start_x + dx
                    wx, match_x, zero_x = _bilinear_backward_weight(
                        ox, x, IW, SW, ALIGN, ACC
                    )
                    active = (oy < OH) & (ox < OW) & match_y & match_x
                    value = tl.load(source + oy * GS2 + ox * GS3, active, other=0.0).to(
                        ACC
                    )
                    contribution = tl.where(
                        (zero_y | zero_x) & _bilinear_backward_isinf(value),
                        float("nan"),
                        (wy * wx) * value,
                    )
                    result += tl.where(active, contribution, 0.0)
        tl.store(GradInput + n * IS0 + c * IS1 + y * IS2 + x * IS3, result)


def _bilinear_backward_fill_or_copy(source, destination, zero):
    n, c, h, w = destination.shape
    is_ascend = runtime_device.vendor_name == "ascend"
    block = 1024 if is_ascend else 256
    programs = triton.cdiv(destination.numel(), block)
    if is_ascend:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            destination.device.index
        )
        programs = min(programs, properties.get("num_vectorcore", 40))
    largest_offset = max(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        for t in (source, destination)
    )
    _upsample_bilinear2d_backward_kernel[(programs,)](
        source,
        destination,
        n,
        c,
        h,
        w,
        h,
        w,
        *source.stride(),
        *destination.stride(),
        1.0,
        1.0,
        False,
        1,
        1,
        False,
        max(source.numel(), destination.numel(), largest_offset) + block >= 2**31,
        destination.stride(1) == 1,
        not zero,
        zero,
        programs,
        block,
        enable_fp_fusion=False,
    )


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_exact2x_kernel(
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
    DOWN: tl.constexpr,
    ACC: tl.constexpr,
    INDEX64: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if INDEX64:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    source = GradOutput + n * GS0 + c * GS1
    if DOWN:
        value = tl.load(source + (y // 2) * GS2 + (x // 2) * GS3, valid, other=0)
        result = value.to(ACC) * 0.25
    else:
        result = tl.full((BLOCK,), 0.0, ACC)
        # Four quarter-pixel taps contribute in the interior. The extra -2
        # candidate preserves the clamped first output's zero-weight update.
        for dy in tl.static_range(-2, 3):
            oy = 2 * y + dy
            if dy == -2:
                wy = tl.full((BLOCK,), 0.0, ACC)
                match_y = y == 1
            elif dy == 0:
                wy = tl.where(y == 0, 1.0, 0.75).to(ACC)
                match_y = oy < OH
            elif dy == 1:
                wy = tl.where(y == IH - 1, 1.0, 0.75).to(ACC)
                match_y = oy < OH
            else:
                wy = tl.full((BLOCK,), 0.25, ACC)
                match_y = (oy >= 0) & (oy < OH)
            for dx in tl.static_range(-2, 3):
                ox = 2 * x + dx
                if dx == -2:
                    wx = tl.full((BLOCK,), 0.0, ACC)
                    match_x = x == 1
                elif dx == 0:
                    wx = tl.where(x == 0, 1.0, 0.75).to(ACC)
                    match_x = ox < OW
                elif dx == 1:
                    wx = tl.where(x == IW - 1, 1.0, 0.75).to(ACC)
                    match_x = ox < OW
                else:
                    wx = tl.full((BLOCK,), 0.25, ACC)
                    match_x = (ox >= 0) & (ox < OW)
                active = valid & match_y & match_x
                value = tl.load(source + oy * GS2 + ox * GS3, active, other=0).to(ACC)
                contribution = (wy * wx) * value
                if (IH == 1 and dy == 0) or (IW == 1 and dx == 0):
                    contribution = tl.where(
                        _bilinear_backward_isinf(value), float("nan"), contribution
                    )
                result += tl.where(active, contribution, 0.0)
    tl.store(GradInput + n * IS0 + c * IS1 + y * IS2 + x * IS3, result, valid)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_ascend_rows(
    GradOutput,
    GradInput,
    PLANES: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    PATCH: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(IH, ROWS)
    tasks: tl.constexpr = PLANES * tiles
    count = tl.cdiv(tasks, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    row = tl.floor(tl.div_rn(index.to(tl.float32), IW)).to(tl.int32)
    x = index - row * IW
    patch_index = tl.arange(0, PATCH)
    shift: tl.constexpr = 0.0 if ALIGN_CORNERS else 0.5
    start_x = tl.maximum(
        tl.floor((x.to(tl.float32) - 1.0 + shift) * (1.0 / SW) - shift), 0.0
    ).to(tl.int32)
    for task in range(first, tl.minimum(first + count, tasks)):
        plane = task // tiles
        first_row = (task - plane * tiles) * ROWS
        y = first_row + row
        valid = index < tl.minimum(ROWS, IH - first_row) * IW
        start_y = tl.maximum(
            tl.floor((y.to(tl.float32) - 1.0 + shift) * (1.0 / SH) - shift), 0.0
        ).to(tl.int32)
        patch_row = tl.maximum(
            tl.floor((first_row.to(tl.float32) - 1.0 + shift) * (1.0 / SH) - shift),
            0.0,
        ).to(tl.int32)
        # A contiguous global read feeds all tap gathers from local memory.
        values = tl.load(
            GradOutput + plane * OH * OW + patch_row * OW + patch_index,
            patch_index < (OH - patch_row) * OW,
            other=0,
        ).to(tl.float32)
        result = tl.full((BLOCK,), 0.0, tl.float32)
        for dy in range(KH):
            oy = start_y + dy
            wy, match_y, zero_y = _bilinear_backward_weight(
                oy, y, IH, SH, ALIGN_CORNERS, tl.float32
            )
            for dx in range(KW):
                ox = start_x + dx
                wx, match_x, zero_x = _bilinear_backward_weight(
                    ox, x, IW, SW, ALIGN_CORNERS, tl.float32
                )
                local = (oy - patch_row) * OW + ox
                value = tl.gather(
                    values, tl.minimum(tl.maximum(local, 0), PATCH - 1), 0
                )
                active = valid & (oy < OH) & (ox < OW) & match_y & match_x
                contribution = tl.where(
                    (zero_y | zero_x) & _bilinear_backward_isinf(value),
                    float("nan"),
                    (wy * wx) * value,
                )
                result += tl.where(active, contribution, 0.0)
        tl.store(GradInput + plane * IH * IW + first_row * IW + index, result, valid)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_ascend_axis(
    Source,
    Destination,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    SOURCE_H: tl.constexpr,
    SOURCE_W: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS3: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    HORIZONTAL: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    K: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    PATCH: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(H, ROWS)
    planes: tl.constexpr = N if CHANNELS_LAST else N * C
    width: tl.constexpr = W * C if CHANNELS_LAST else W
    count = tl.cdiv(planes * tiles, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    row = tl.floor(tl.div_rn(index.to(tl.float32), width)).to(tl.int32)
    column = index - row * width
    if CHANNELS_LAST:
        x = tl.floor(tl.div_rn(column.to(tl.float32), C)).to(tl.int32)
        channel = column - x * C
    else:
        x = column
        channel = tl.full((BLOCK,), 0, tl.int32)
    patch_index = tl.arange(0, PATCH)
    shift: tl.constexpr = 0.0 if ALIGN_CORNERS else 0.5
    for task in range(first, tl.minimum(first + count, planes * tiles)):
        plane = task // tiles
        first_row = (task - plane * tiles) * ROWS
        y = first_row + row
        valid = index < tl.minimum(ROWS, H - first_row) * width
        if CHANNELS_LAST:
            source = Source + plane * SS0
            destination = Destination + plane * DS0
        else:
            batch = plane // C
            ch = plane - batch * C
            source = Source + batch * SS0 + ch * SS1
            destination = Destination + batch * DS0 + ch * DS1
        if HORIZONTAL:
            coordinate = x
            input_size: tl.constexpr = W
            output_size: tl.constexpr = SOURCE_W
            patch_row = first_row
        else:
            coordinate = y
            input_size: tl.constexpr = H
            output_size: tl.constexpr = SOURCE_H
            patch_row = tl.maximum(
                tl.floor((first_row.to(tl.float32) - 1.0 + shift) / SCALE - shift), 0.0
            ).to(tl.int32)
        start = tl.maximum(
            tl.floor((coordinate.to(tl.float32) - 1.0 + shift) / SCALE - shift), 0.0
        ).to(tl.int32)
        remaining = (SOURCE_H - patch_row - 1) * SS2 + (SOURCE_W - 1) * SS3 + 1
        if CHANNELS_LAST:
            remaining += C - 1
        values = tl.load(
            source + patch_row * SS2 + patch_index, patch_index < remaining, other=0
        ).to(tl.float32)
        result = tl.full((BLOCK,), 0.0, tl.float32)
        for tap in range(K):
            output = start + tap
            weight, match, zero_tap = _bilinear_backward_weight(
                output, coordinate, input_size, SCALE, ALIGN_CORNERS, tl.float32
            )
            if HORIZONTAL:
                local = row * SS2 + output * SS3 + channel * SS1
            else:
                local = (output - patch_row) * SS2 + x * SS3 + channel * SS1
            value = tl.gather(values, tl.minimum(tl.maximum(local, 0), PATCH - 1), 0)
            active = valid & match & (output < output_size)
            contribution = tl.where(
                zero_tap & _bilinear_backward_isinf(value), float("nan"), weight * value
            )
            result += tl.where(active, contribution, 0.0)
        if CHANNELS_LAST:
            address = first_row * DS2 + index
        else:
            address = first_row * DS2 + index * DS3
        tl.store(destination + address, result, valid)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_unrolled_kernel(
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
    INDEX64: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC: tl.constexpr,
):
    if INDEX64:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    shift: tl.constexpr = 0.0 if ALIGN_CORNERS else 0.5
    if SH == 0.0 or IH == 1:
        start_y = tl.full((BLOCK,), 0, offsets.dtype)
    else:
        start_y = tl.maximum(
            tl.floor((y.to(ACC) - 1.0 + shift) * tl.full((), 1.0 / SH, ACC) - shift).to(
                offsets.dtype
            ),
            0,
        )
    if SW == 0.0 or IW == 1:
        start_x = tl.full((BLOCK,), 0, offsets.dtype)
    else:
        start_x = tl.maximum(
            tl.floor((x.to(ACC) - 1.0 + shift) * tl.full((), 1.0 / SW, ACC) - shift).to(
                offsets.dtype
            ),
            0,
        )
    source = GradOutput + n * GS0 + c * GS1
    result = tl.full((BLOCK,), 0.0, ACC)
    # Bounded unrolling lets common width/height coordinate expressions be reused.
    for dy in tl.static_range(KH):
        oy = start_y + dy
        wy, match_y, zero_y = _bilinear_backward_weight(
            oy, y, IH, SH, ALIGN_CORNERS, ACC
        )
        for dx in tl.static_range(KW):
            ox = start_x + dx
            wx, match_x, zero_x = _bilinear_backward_weight(
                ox, x, IW, SW, ALIGN_CORNERS, ACC
            )
            active = valid & (oy < OH) & (ox < OW) & match_y & match_x
            value = tl.load(source + oy * GS2 + ox * GS3, active, other=0.0).to(ACC)
            contribution = tl.where(
                (zero_y | zero_x) & _bilinear_backward_isinf(value),
                float("nan"),
                (wy * wx) * value,
            )
            result += tl.where(active, contribution, 0.0)
    tl.store(GradInput + n * IS0 + c * IS1 + y * IS2 + x * IS3, result, valid)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_disjoint_kernel(
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
    FP64: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if INDEX64:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N * C * OH * OW
    ox = offsets % OW
    oy = offsets // OW % OH
    c = offsets // (OW * OH) % C
    n = offsets // (OW * OH * C)
    if FP64:
        acc_dtype: tl.constexpr = tl.float64
    else:
        acc_dtype: tl.constexpr = tl.float32
    scale_h = tl.full((), SH, acc_dtype)
    scale_w = tl.full((), SW, acc_dtype)
    if ALIGN_CORNERS:
        ry = oy.to(acc_dtype) * scale_h
        rx = ox.to(acc_dtype) * scale_w
    else:
        ry = tl.maximum((oy.to(acc_dtype) + 0.5) * scale_h - 0.5, 0.0)
        rx = tl.maximum((ox.to(acc_dtype) + 0.5) * scale_w - 0.5, 0.0)
    y0 = tl.minimum(ry.to(offsets.dtype), IH - 1)
    x0 = tl.minimum(rx.to(offsets.dtype), IW - 1)
    y1 = tl.minimum(y0 + 1, IH - 1)
    x1 = tl.minimum(x0 + 1, IW - 1)
    hy1 = ry - y0.to(acc_dtype)
    wx1 = rx - x0.to(acc_dtype)
    hy0 = tl.where(y1 == y0, 1.0, 1.0 - hy1)
    wx0 = tl.where(x1 == x0, 1.0, 1.0 - wx1)
    value = tl.load(
        GradOutput + n * GS0 + c * GS1 + oy * GS2 + ox * GS3, valid, other=0
    ).to(acc_dtype)
    zero_duplicate = ((y0 == y1) & ((hy1 == 0.0) | (hy1 == 1.0))) | (
        (x0 == x1) & ((wx1 == 0.0) | (wx1 == 1.0))
    )
    value = tl.where(
        zero_duplicate & _bilinear_backward_isinf(value), float("nan"), value
    )
    destination = GradInput + n * IS0 + c * IS1
    # Successive output pixels are at least two input pixels apart on both axes.
    tl.store(destination + y0 * IS2 + x0 * IS3, (hy0 * wx0) * value, valid)
    tl.store(destination + y0 * IS2 + x1 * IS3, (hy0 * wx1) * value, valid & (x1 != x0))
    tl.store(destination + y1 * IS2 + x0 * IS3, (hy1 * wx0) * value, valid & (y1 != y0))
    tl.store(
        destination + y1 * IS2 + x1 * IS3,
        (hy1 * wx1) * value,
        valid & (y1 != y0) & (x1 != x0),
    )


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_table_kernel(
    Indices,
    Weights,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    HEIGHT_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < HEIGHT_BLOCKS:
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        _bilinear_backward_table_axis(
            Indices, Weights, offset, IH, OH, SH, KH, ALIGN_CORNERS
        )
    else:
        offset = (pid - HEIGHT_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
        _bilinear_backward_table_axis(
            Indices + IH * KH,
            Weights + IH * KH,
            offset,
            IW,
            OW,
            SW,
            KW,
            ALIGN_CORNERS,
        )


@triton.jit
def _bilinear_backward_table_axis(
    Indices,
    Weights,
    offset,
    SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    K: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
):
    index = offset // K
    delta = offset % K
    shift: tl.constexpr = 0.0 if ALIGN_CORNERS else 0.5
    if SCALE == 0.0 or SIZE == 1:
        start = tl.full(offset.shape, 0, tl.int64)
    else:
        inverse = tl.full((), 1.0 / SCALE, tl.float64)
        start = tl.maximum(
            tl.floor((index.to(tl.float64) - 1.0 + shift) * inverse - shift).to(
                tl.int64
            ),
            0,
        )
    output = start + delta
    weight, match, zero_tap = _bilinear_backward_weight(
        output, index, SIZE, SCALE, ALIGN_CORNERS, tl.float64
    )
    active = (output < OUTPUT_SIZE) & match
    # Encode the zero-tap flag alongside the source index for the separable path.
    encoded = output + tl.where(zero_tap, OUTPUT_SIZE, 0)
    tl.store(Indices + offset, tl.where(active, encoded, -1), index < SIZE)
    tl.store(Weights + offset, weight, index < SIZE)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_axis_kernel(
    Source,
    Destination,
    Indices,
    Weights,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS3: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    HORIZONTAL: tl.constexpr,
    TABLE_OFFSET: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N * C * H * W
    x = offsets % W
    y = offsets // W % H
    c = offsets // (W * H) % C
    n = offsets // (W * H * C)
    if HORIZONTAL:
        index = x
        source = Source + n * SS0 + c * SS1 + y * SS2
        step: tl.constexpr = SS3
    else:
        index = y
        source = Source + n * SS0 + c * SS1 + x * SS3
        step: tl.constexpr = SS2
    result = tl.full((BLOCK,), 0.0, tl.float64)
    for delta in range(K):
        table = TABLE_OFFSET + index * K + delta
        output = tl.load(Indices + table, valid, other=-1)
        weight = tl.load(Weights + table, valid, other=0.0)
        active = valid & (output >= 0)
        zero_tap = output >= OUTPUT_SIZE
        output = tl.where(zero_tap, output - OUTPUT_SIZE, output)
        value = tl.load(source + output * step, active, other=0)
        contribution = tl.where(
            zero_tap & _bilinear_backward_isinf(value),
            float("nan"),
            weight * value,
        )
        result += tl.where(active, contribution, 0.0)
    tl.store(Destination + n * DS0 + c * DS1 + y * DS2 + x * DS3, result, valid)


@libentry()
@triton.jit
def _upsample_bilinear2d_backward_axis_fp32(
    Source,
    Destination,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    INPUT: tl.constexpr,
    OUTPUT: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS3: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    HORIZONTAL: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N * C * H * W
    x = offset % W
    y = offset // W % H
    c = offset // (W * H) % C
    n = offset // (W * H * C)
    if HORIZONTAL:
        index = x
        source = Source + n * SS0 + c * SS1 + y * SS2
        step: tl.constexpr = SS3
    else:
        index = y
        source = Source + n * SS0 + c * SS1 + x * SS3
        step: tl.constexpr = SS2
    shift: tl.constexpr = 0.0 if ALIGN else 0.5
    start = tl.maximum(
        tl.floor((index.to(tl.float32) - 1.0 + shift) * (1.0 / SCALE) - shift), 0.0
    ).to(tl.int32)
    result = tl.full((BLOCK,), 0.0, tl.float32)
    for delta in tl.static_range(K):
        output = start + delta
        weight, matches, zero_tap = _bilinear_backward_weight(
            output, index, INPUT, SCALE, ALIGN, tl.float32
        )
        active = valid & (output < OUTPUT) & matches
        value = tl.load(source + output * step, active, other=0.0).to(tl.float32)
        contribution = tl.where(
            zero_tap & _bilinear_backward_isinf(value), float("nan"), weight * value
        )
        result += tl.where(active, contribution, 0.0)
    tl.store(Destination + n * DS0 + c * DS1 + y * DS2 + x * DS3, result, valid)


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
    logger.debug("GEMS UPSAMPLE_BILINEAR2D_BACKWARD")
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
    # Explicit scales are required to be positive; omitted scales are inferred.
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

    if grad_output.dtype == torch.float64 and runtime_device.vendor_name == "iluvatar":
        raise RuntimeError(
            "Iluvatar bilinear2d backward does not support FP64 arithmetic"
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
    # Verified CANN 8.5.0/9.0.0 nonidentity backward uses size ratios. Identity keeps
    # the explicit copy contract, including when scales were supplied.
    if _CANN_SIZE_RATIO and not copy:
        scales_h = scales_w = None
    if align_corners:
        sh = (ih - 1) / (oh - 1) if oh > 1 else 0.0
        sw = (iw - 1) / (ow - 1) if ow > 1 else 0.0
    else:
        sh = 1.0 / scales_h if scales_h is not None else ih / oh
        sw = 1.0 / scales_w if scales_w is not None else iw / ow
    if not align_corners:
        # Far inside the left clamp, every source coordinate is exactly zero.
        # The margin avoids an FP32 rounding boundary and unbounded inverse windows.
        if (oh - 0.5) * sh <= 0.25:
            sh = 0.0
        if (ow - 0.5) * sw <= 0.25:
            sw = 0.0
    exact_down = not align_corners and sh == sw == 2.0 and ih == 2 * oh and iw == 2 * ow
    exact_up = not align_corners and sh == sw == 0.5 and oh == 2 * ih and ow == 2 * iw
    exact_down = exact_down and runtime_device.vendor_name != "iluvatar"
    exact_down = exact_down and max(ih, iw, oh, ow) <= 2**20
    exact_up = exact_up and runtime_device.vendor_name != "iluvatar"
    exact_up = exact_up and max(ih, iw, oh, ow) <= 2**20
    kh = oh if sh == 0 or ih == 1 else min(oh, math.ceil(2 / sh) + 2)
    kw = ow if sw == 0 or iw == 1 else min(ow, math.ceil(2 / sw) + 2)
    # Inconsistent explicit scales can map the output tail beyond the input.
    # The last input pixel then owns every remaining clamped contribution.
    shift = 0.0 if align_corners else 0.5
    if sh > 0 and ih > 1:
        tail_h = oh - max(0, math.floor((ih - 2 + shift) / sh - shift))
        kh = min(oh, max(kh, tail_h))
    if sw > 0 and iw > 1:
        tail_w = ow - max(0, math.floor((iw - 2 + shift) / sw - shift))
        kw = min(ow, max(kw, tail_w))
    is_ascend = runtime_device.vendor_name == "ascend"
    block = 1024 if is_ascend else 128
    last_h = (oh - 1) * sh if align_corners else (oh - 0.5) * sh - 0.5
    last_w = (ow - 1) * sw if align_corners else (ow - 0.5) * sw - 0.5
    disjoint = (oh == 1 or sh >= 2.0) and (ow == 1 or sw >= 2.0)
    disjoint = disjoint and last_h < ih and last_w < iw and not copy
    # Bound rounding error so adjacent output pixels cannot share a destination.
    coordinate_epsilon = 2.0**-23
    disjoint = disjoint and max(ih, iw, oh, ow) <= 2**20
    disjoint = disjoint and all(
        size == 1
        or scale == 2.0
        or scale - 2.0 >= 8 * coordinate_epsilon * max(1.0, last)
        for size, scale, last in ((oh, sh, last_h), (ow, sw, last_w))
    )
    # Ascend cannot legalize the disjoint kernel's multiple masked stores.
    disjoint = disjoint and not is_ascend
    disjoint = disjoint and n * c * ih * iw >= 65536 and oh * ow > 1
    with torch_device_fn.device(grad_output.device):
        if (
            _CANN_VERSION == "9.0.0"
            and copy
            and n * c * ih * iw < 2**30
            and grad_output.stride() == grad_input.stride()
            and (
                grad_output.is_contiguous()
                or grad_output.is_contiguous(memory_format=torch.channels_last)
            )
        ):
            properties = triton.runtime.driver.active.utils.get_device_properties(
                grad_output.device.index
            )
            programs = min(
                triton.cdiv(n * c * ih * iw, 4096),
                properties.get("num_vectorcore", 40),
            )
            _upsample_bilinear2d_backward_ascend_copy[(programs,)](
                grad_output, grad_input, n * c * ih * iw, 4096
            )
            return grad_input
        if (runtime_device.vendor_name == "hygon" or _CANN_VERSION == "8.5.0") and max(
            sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
            for tensor in (grad_output, grad_input)
        ) >= 2**31:
            programs = min(n * c * ih * iw, 40 if is_ascend else 4096)
            _upsample_bilinear2d_backward_scalar_kernel[(programs,)](
                grad_output,
                grad_input,
                n * c * ih * iw,
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
                kh,
                kw,
                tl.float64 if grad_output.dtype == torch.float64 else tl.float32,
                copy,
                enable_fp_fusion=False,
            )
            return grad_input
        if (
            grad_output.dtype == torch.float64
            and n * c * ih * iw >= 65536
            and oh > 1
            and ow > 1
            and not copy
            and not disjoint
            and not exact_up
        ):
            # Compute FP64 coordinates once per spatial axis, shared by N and C.
            count = ih * kh + iw * kw
            indices = torch.empty(count, device=grad_output.device, dtype=torch.int64)
            weights = torch.empty(count, device=grad_output.device, dtype=torch.float64)
            height_blocks = triton.cdiv(ih * kh, block)
            _upsample_bilinear2d_backward_table_kernel[
                (height_blocks + triton.cdiv(iw * kw, block),)
            ](
                indices,
                weights,
                ih,
                iw,
                oh,
                ow,
                sh,
                sw,
                kh,
                kw,
                align_corners,
                height_blocks,
                block,
                enable_fp_fusion=False,
            )
            # Separability reduces the FP64 work from KH * KW to KH + KW.
            temporary = torch.empty(
                (n, c, oh, iw), device=grad_output.device, dtype=torch.float64
            )
            _upsample_bilinear2d_backward_axis_kernel[
                (triton.cdiv(n * c * oh * iw, block),)
            ](
                grad_output,
                temporary,
                indices,
                weights,
                n,
                c,
                oh,
                iw,
                ow,
                *grad_output.stride(),
                *temporary.stride(),
                True,
                ih * kh,
                kw,
                block,
                enable_fp_fusion=False,
            )
            _upsample_bilinear2d_backward_axis_kernel[
                (triton.cdiv(n * c * ih * iw, block),)
            ](
                temporary,
                grad_input,
                indices,
                weights,
                n,
                c,
                ih,
                iw,
                oh,
                *temporary.stride(),
                *grad_input.stride(),
                False,
                0,
                kh,
                block,
                enable_fp_fusion=False,
            )
            return grad_input
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
        # Bound the Ascend axis temporary and every core's final tile as well.
        index64 = index64 or (
            is_ascend
            and max(n * c * ih * iw, n * c * oh * ow, n * c * oh * iw, largest_offset)
            >= 2**30
        )
        if (
            runtime_device.vendor_name == "mthreads"
            and grad_output.dtype != torch.float64
            and not copy
            and not exact_down
            and not exact_up
            and not disjoint
            and not index64
            and sh > 0
            and sw > 0
            and ih > 1
            and iw > 1
            and n * c * ih * iw >= 4096
            and 9 <= kh * kw
            and max(kh, kw) <= 16
        ):
            temporary = torch.empty(
                (n, c, oh, iw), dtype=torch.float32, device=grad_output.device
            )
            _upsample_bilinear2d_backward_axis_fp32[
                (triton.cdiv(n * c * oh * iw, 128),)
            ](
                grad_output,
                temporary,
                n,
                c,
                oh,
                iw,
                iw,
                ow,
                *gs,
                *temporary.stride(),
                True,
                sw,
                align_corners,
                kw,
                128,
                enable_fp_fusion=False,
            )
            _upsample_bilinear2d_backward_axis_fp32[
                (triton.cdiv(n * c * ih * iw, 128),)
            ](
                temporary,
                grad_input,
                n,
                c,
                ih,
                iw,
                ih,
                oh,
                *temporary.stride(),
                *strides,
                False,
                sh,
                align_corners,
                kh,
                128,
                enable_fp_fusion=False,
            )
            return grad_input
        if exact_down or (exact_up and not is_ascend):
            _upsample_bilinear2d_backward_exact2x_kernel[
                (triton.cdiv(n * c * ih * iw, block),)
            ](
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
                exact_down,
                tl.float64 if grad_output.dtype == torch.float64 else tl.float32,
                index64,
                strides[1] == 1,
                block,
                enable_fp_fusion=False,
            )
            return grad_input
        if (
            runtime_device.vendor_name == "nvidia"
            and grad_output.dtype == torch.float64
            and not copy
            and not disjoint
            and 9 <= kh * kw <= 64
        ):
            # Bound the unrolled instruction footprint and retain scalar accumulators.
            _upsample_bilinear2d_backward_unrolled_kernel[
                (triton.cdiv(n * c * ih * iw, 128),)
            ](
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
                index64,
                strides[1] == 1,
                128,
                tl.float64 if grad_output.dtype == torch.float64 else tl.float32,
                enable_fp_fusion=False,
            )
            return grad_input
        if (
            _CANN_VERSION == "9.0.0"
            and not copy
            and not index64
            and sh > 0
            and sw > 0
            and ih > 1
            and iw > 1
            and n * c * ih * iw >= 4096
            and 9 <= kh * kw
            and max(kh, kw) <= 16
        ):
            channels_last = grad_output.is_contiguous(
                memory_format=torch.channels_last
            ) and grad_input.is_contiguous(memory_format=torch.channels_last)
            regular = (
                gs[2] == ow * gs[3]
                and strides[2] == iw * strides[3]
                and gs[3] in (1, 2)
                and strides[3] in (1, 2)
            )
            if channels_last or regular:
                if channels_last:
                    temporary = torch.empty_strided(
                        (n, c, oh, iw),
                        (c * oh * iw, 1, c * iw, c),
                        dtype=torch.float32,
                        device=grad_output.device,
                    )
                else:
                    temporary = torch.empty(
                        (n, c, oh, iw), dtype=torch.float32, device=grad_output.device
                    )
                properties = triton.runtime.driver.active.utils.get_device_properties(
                    grad_output.device.index
                )
                axis_output = grad_input
                if not channels_last and not grad_input.is_contiguous():
                    axis_output = torch.empty(
                        input_size, dtype=grad_input.dtype, device=grad_input.device
                    )
                axes = [
                    (grad_output, temporary, oh, iw, oh, ow, True, sw, kw),
                    (temporary, axis_output, ih, iw, oh, iw, False, sh, kh),
                ]
                configurations = []
                for (
                    source,
                    destination,
                    h,
                    w,
                    source_h,
                    source_w,
                    horizontal,
                    scale,
                    taps,
                ) in axes:
                    effective_width = w * c if channels_last else w
                    rows = min(16, h)
                    while (
                        rows > 1
                        and triton.next_power_of_2(rows * effective_width) > 1024
                    ):
                        rows //= 2
                    while rows > 1:
                        patch_rows = (
                            rows
                            if horizontal
                            else min(source_h, math.ceil((rows - 1) / scale) + taps + 2)
                        )
                        if (
                            triton.next_power_of_2(patch_rows * source.stride(2))
                            <= 8192
                        ):
                            break
                        rows //= 2
                    patch_rows = (
                        rows
                        if horizontal
                        else min(source_h, math.ceil((rows - 1) / scale) + taps + 2)
                    )
                    patch = triton.next_power_of_2(patch_rows * source.stride(2))
                    configurations.append((rows, patch))
                if all(patch <= 8192 for _, patch in configurations):
                    for (
                        source,
                        destination,
                        h,
                        w,
                        source_h,
                        source_w,
                        horizontal,
                        scale,
                        taps,
                    ), (rows, patch) in zip(axes, configurations):
                        programs = min(
                            (n if channels_last else n * c) * triton.cdiv(h, rows),
                            properties.get("num_vectorcore", 40),
                        )
                        _upsample_bilinear2d_backward_ascend_axis[(programs,)](
                            source,
                            destination,
                            n,
                            c,
                            h,
                            w,
                            source_h,
                            source_w,
                            *source.stride(),
                            *destination.stride(),
                            horizontal,
                            scale,
                            align_corners,
                            taps,
                            channels_last,
                            rows,
                            triton.next_power_of_2(
                                rows * (w * c if channels_last else w)
                            ),
                            patch,
                            enable_fp_fusion=False,
                        )
                    if axis_output is not grad_input:
                        _bilinear_backward_fill_or_copy(axis_output, grad_input, False)
                    return grad_input
        if (
            _CANN_VERSION == "9.0.0"
            and not copy
            and not index64
            and sh > 0
            and sw > 0
            and ih > 1
            and iw > 1
            and 9 <= kh * kw <= 64
        ):
            rows = min(8, ih)
            while rows > 1 and (
                triton.next_power_of_2(rows * iw) > 1024
                or triton.next_power_of_2(
                    min(oh, math.ceil((rows - 1) / sh) + kh + 2) * ow
                )
                > 16384
            ):
                rows //= 2
            patch = triton.next_power_of_2(
                min(oh, math.ceil((rows - 1) / sh) + kh + 2) * ow
            )
            if patch <= 16384 and triton.next_power_of_2(rows * iw) <= 1024:
                source = grad_output
                destination = grad_input
                if not source.is_contiguous():
                    source = torch.empty(
                        grad_output.shape,
                        dtype=grad_output.dtype,
                        device=grad_output.device,
                    )
                    _bilinear_backward_fill_or_copy(grad_output, source, False)
                if not destination.is_contiguous():
                    destination = torch.empty(
                        input_size, dtype=grad_input.dtype, device=grad_input.device
                    )
                properties = triton.runtime.driver.active.utils.get_device_properties(
                    grad_output.device.index
                )
                programs = min(
                    n * c * triton.cdiv(ih, rows), properties.get("num_vectorcore", 40)
                )
                _upsample_bilinear2d_backward_ascend_rows[(programs,)](
                    source,
                    destination,
                    n * c,
                    ih,
                    iw,
                    oh,
                    ow,
                    sh,
                    sw,
                    align_corners,
                    kh,
                    kw,
                    rows,
                    triton.next_power_of_2(rows * iw),
                    patch,
                    enable_fp_fusion=False,
                )
                if destination is not grad_input:
                    _bilinear_backward_fill_or_copy(destination, grad_input, False)
                return grad_input
        programs = triton.cdiv(n * c * ih * iw, block)
        if is_ascend:
            # Bound the grid by the vector cores and process remaining tiles locally.
            properties = triton.runtime.driver.active.utils.get_device_properties(
                grad_output.device.index
            )
            programs = min(programs, properties.get("num_vectorcore", 40))
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
            disjoint,
            programs,
            block,
            enable_fp_fusion=False,
        )
        if disjoint:
            _upsample_bilinear2d_backward_disjoint_kernel[
                (triton.cdiv(n * c * oh * ow, block),)
            ](
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
                grad_output.dtype == torch.float64,
                index64,
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
