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
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name


@triton.jit
def _lanczos3(x):
    abs_x = tl.abs(x)
    pix = x * 3.141592653589793
    pix_over_three = pix / 3.0
    sinc_x = tl.where(abs_x == 0.0, 1.0, tl.sin(pix) / pix)
    sinc_x_over_three = tl.where(
        abs_x == 0.0, 1.0, tl.sin(pix_over_three) / pix_over_three
    )
    return tl.where(abs_x < 3.0, sinc_x * sinc_x_over_three, 0.0)


@triton.jit
def _lanczos_weights_kernel(
    weights,
    index_mins,
    index_sizes,
    input_size,
    output_size,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    center = SCALE * (offsets.to(compute_dtype) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0)
    index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_size)
    index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)

    for tap in range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        weight = tl.where(tap_mask, weight, 0.0)
        tl.store(weights + offsets * MAX_TAPS + tap, weight, mask=mask)
        total_weight += weight

    total_weight = tl.where(total_weight != 0.0, total_weight, 1.0)
    for tap in range(MAX_TAPS):
        weight = tl.load(weights + offsets * MAX_TAPS + tap, mask=mask)
        tl.store(
            weights + offsets * MAX_TAPS + tap,
            weight / total_weight,
            mask=mask,
        )
    tl.store(index_mins + offsets, index_min, mask=mask)
    tl.store(index_sizes + offsets, index_size, mask=mask)


@triton.jit
def _lanczos_horizontal_kernel(
    input,
    output,
    weights,
    index_mins,
    index_sizes,
    numel,
    input_w,
    output_w,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    output_x = offsets % output_w
    row = offsets // output_w
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    if PRECOMPUTED:
        index_min = tl.load(index_mins + output_x, mask=mask)
        index_size = tl.load(index_sizes + output_x, mask=mask)
    else:
        center = SCALE * (output_x.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0)
        index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_w)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for tap in range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + output_x * MAX_TAPS + tap,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        sample = tl.load(
            input + row * input_w + index_min + tap,
            mask=tap_mask,
            other=0.0,
        ).to(compute_dtype)
        weight = tl.where(tap_mask, weight, 0.0)
        value += sample * weight
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lanczos_vertical_kernel(
    input,
    output,
    weights,
    index_mins,
    index_sizes,
    numel,
    input_h,
    output_h,
    output_w,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    output_x = offsets % output_w
    output_y = (offsets // output_w) % output_h
    nc = offsets // (output_h * output_w)
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    if PRECOMPUTED:
        index_min = tl.load(index_mins + output_y, mask=mask)
        index_size = tl.load(index_sizes + output_y, mask=mask)
    else:
        center = SCALE * (output_y.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0)
        index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_h)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for tap in range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + output_y * MAX_TAPS + tap,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        input_offset = (nc * input_h + index_min + tap) * output_w + output_x
        sample = tl.load(input + input_offset, mask=tap_mask, other=0.0).to(
            compute_dtype
        )
        weight = tl.where(tap_mask, weight, 0.0)
        value += sample * weight
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(output + offsets, value, mask=mask)


def _reciprocal_scale(
    input_size: int,
    output_size: int,
    align_corners: bool,
    scale: Optional[float],
) -> float:
    if align_corners:
        return (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    if scale is not None and scale > 0.0:
        return 1.0 / scale
    return input_size / output_size


def _support_and_taps(scale: float) -> Tuple[float, int]:
    support = 3.0 * scale if scale >= 1.0 else 3.0
    return support, math.ceil(support) * 2 + 1


def _validate(input: torch.Tensor, output_size: Sequence[int]) -> Tuple[int, int]:
    if input.device.type != device:
        raise RuntimeError(f"Expected input on {device}, but got {input.device.type}")
    if input.ndim != 4:
        raise RuntimeError(
            f"It is expected input_size equals to 4, but got size {input.ndim}"
        )
    if len(output_size) != 2:
        raise RuntimeError(
            f"It is expected output_size equals to 2, but got size {len(output_size)}"
        )
    if not input.is_floating_point() and input.dtype != torch.uint8:
        raise RuntimeError(
            f'"compute_index_ranges_weights" not implemented for {input.dtype}'
        )

    output_h, output_w = int(output_size[0]), int(output_size[1])
    if min(input.shape[-2], input.shape[-1], output_h, output_w) <= 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if input.shape[1] == 0:
        raise RuntimeError("Non-empty 4D data tensor expected")
    return output_h, output_w


def _make_lanczos_weights(
    input_size: int,
    output_size: int,
    scale: float,
    support: float,
    taps: int,
    device: torch.device,
    is_fp64: bool,
):
    weights = torch.empty(
        (output_size, taps),
        device=device,
        dtype=torch.float64 if is_fp64 else torch.float32,
    )
    index_mins = torch.empty(output_size, device=device, dtype=torch.int64)
    index_sizes = torch.empty(output_size, device=device, dtype=torch.int64)
    block_size = 128
    _lanczos_weights_kernel[(triton.cdiv(output_size, block_size),)](
        weights,
        index_mins,
        index_sizes,
        input_size,
        output_size,
        SCALE=scale,
        SUPPORT=support,
        INVSCALE=1.0 / scale if scale >= 1.0 else 1.0,
        MAX_TAPS=taps,
        IS_FP64=is_fp64,
        BLOCK_SIZE=block_size,
    )
    return weights, index_mins, index_sizes


def _upsample_lanczos2d_aa_contiguous(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scales_h: Optional[float],
    scales_w: Optional[float],
) -> torch.Tensor:
    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    if input_h == output_h and input_w == output_w:
        return input.clone()

    scale_w = _reciprocal_scale(input_w, output_w, align_corners, scales_w)
    scale_h = _reciprocal_scale(input_h, output_h, align_corners, scales_h)
    support_w, taps_w = _support_and_taps(scale_w)
    support_h, taps_h = _support_and_taps(scale_h)
    is_uint8 = input.dtype == torch.uint8
    is_fp64 = input.dtype == torch.float64
    precompute_weights = n * c >= 128
    block_size = 256

    with torch_device_fn.device(input.device):
        if input_w != output_w:
            if precompute_weights:
                weights_w, index_mins_w, index_sizes_w = _make_lanczos_weights(
                    input_w,
                    output_w,
                    scale_w,
                    support_w,
                    taps_w,
                    input.device,
                    is_fp64,
                )
            else:
                weights_w = index_mins_w = index_sizes_w = input
            horizontal = torch.empty(
                (n, c, input_h, output_w), device=input.device, dtype=input.dtype
            )
            horizontal_numel = horizontal.numel()
            _lanczos_horizontal_kernel[(triton.cdiv(horizontal_numel, block_size),)](
                input,
                horizontal,
                weights_w,
                index_mins_w,
                index_sizes_w,
                horizontal_numel,
                input_w,
                output_w,
                SCALE=scale_w,
                SUPPORT=support_w,
                INVSCALE=1.0 / scale_w if scale_w >= 1.0 else 1.0,
                MAX_TAPS=taps_w,
                IS_UINT8=is_uint8,
                IS_FP64=is_fp64,
                PRECOMPUTED=precompute_weights,
                BLOCK_SIZE=block_size,
            )
        else:
            horizontal = input

        if input_h != output_h:
            if precompute_weights:
                weights_h, index_mins_h, index_sizes_h = _make_lanczos_weights(
                    input_h,
                    output_h,
                    scale_h,
                    support_h,
                    taps_h,
                    input.device,
                    is_fp64,
                )
            else:
                weights_h = index_mins_h = index_sizes_h = input
            output = torch.empty(
                (n, c, output_h, output_w), device=input.device, dtype=input.dtype
            )
            output_numel = output.numel()
            _lanczos_vertical_kernel[(triton.cdiv(output_numel, block_size),)](
                horizontal,
                output,
                weights_h,
                index_mins_h,
                index_sizes_h,
                output_numel,
                input_h,
                output_h,
                output_w,
                SCALE=scale_h,
                SUPPORT=support_h,
                INVSCALE=1.0 / scale_h if scale_h >= 1.0 else 1.0,
                MAX_TAPS=taps_h,
                IS_UINT8=is_uint8,
                IS_FP64=is_fp64,
                PRECOMPUTED=precompute_weights,
                BLOCK_SIZE=block_size,
            )
        else:
            output = horizontal
    return output


def _upsample_lanczos2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE LANCZOS2D AA")
    output_size = _validate(input, output_size)
    channels_last = input.is_contiguous(memory_format=torch.channels_last)
    if input.shape[0] == 0:
        output = torch.empty(
            (*input.shape[:2], *output_size), device=input.device, dtype=input.dtype
        )
    else:
        output = _upsample_lanczos2d_aa_contiguous(
            input.contiguous(),
            output_size,
            align_corners,
            scales_h,
            scales_w,
        )
    if channels_last:
        output = output.contiguous(memory_format=torch.channels_last)
    return output


def _upsample_lanczos2d_aa_out(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor on {input.device}, but got {out.device}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype}"
        )
    result = _upsample_lanczos2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def _upsample_lanczos2d_aa_vec(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    if (output_size is None) == (scale_factors is None):
        raise RuntimeError("Must specify exactly one of output_size and scale_factors")
    if output_size is not None:
        return _upsample_lanczos2d_aa(input, tuple(output_size), align_corners)
    if len(scale_factors) != 2:
        raise RuntimeError("scale_factors must have two elements")
    output_size = (
        int(input.shape[-2] * scale_factors[0]),
        int(input.shape[-1] * scale_factors[1]),
    )
    return _upsample_lanczos2d_aa(
        input,
        output_size,
        align_corners,
        float(scale_factors[0]),
        float(scale_factors[1]),
    )
