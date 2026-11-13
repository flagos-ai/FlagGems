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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _round_half_to_even(x):
    floor_x = tl.floor(x)
    fraction = x - floor_x
    floor_is_even = (floor_x % 2.0) == 0.0
    return tl.where(
        fraction > 0.5,
        floor_x + 1.0,
        tl.where((fraction < 0.5) | floor_is_even, floor_x, floor_x + 1.0),
    )


@triton.jit
def fake_quantize_per_channel_affine_cachemask_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    cachemask_ptr,
    n_elements,
    n_channels,
    channel_stride,
    quant_min,
    quant_max,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=valid, other=0.0)
    channel_idx = (offsets // channel_stride) % n_channels
    scale = tl.load(scale_ptr + channel_idx, mask=valid, other=1.0)
    zero_point = tl.load(zero_point_ptr + channel_idx, mask=valid, other=0.0)

    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    zero_point_fp32 = zero_point.to(tl.float32)
    quantized = _round_half_to_even(x_fp32 / scale_fp32) + zero_point_fp32
    cachemask = (quantized >= quant_min) & (quantized <= quant_max)
    quantized = tl.minimum(tl.maximum(quantized, quant_min), quant_max)
    output = (quantized - zero_point_fp32) * scale_fp32

    tl.store(output_ptr + offsets, output, mask=valid)
    tl.store(cachemask_ptr + offsets, cachemask, mask=valid)


def _fake_quantize_per_channel_affine_cachemask_impl(
    input,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    output=None,
    cachemask=None,
):
    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    if output is None:
        output = torch.empty_like(input)
    if cachemask is None:
        cachemask = torch.empty_like(input, dtype=torch.bool)

    n_elements = input.numel()
    if n_elements == 0:
        return output, cachemask

    n_channels = input.shape[axis]
    channel_stride = 1
    for size in input.shape[axis + 1 :]:
        channel_stride *= size

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        fake_quantize_per_channel_affine_cachemask_kernel[grid](
            input,
            scale,
            zero_point,
            output,
            cachemask,
            n_elements,
            n_channels,
            channel_stride,
            quant_min,
            quant_max,
            BLOCK_SIZE=1024,
        )
    return output, cachemask


def fake_quantize_per_channel_affine_cachemask(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input, scale, zero_point, axis, quant_min, quant_max
    )


def fake_quantize_per_channel_affine_cachemask_out(
    input, scale, zero_point, axis, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_OUT")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input,
        scale,
        zero_point,
        axis,
        quant_min,
        quant_max,
        output=out0,
        cachemask=out1,
    )
