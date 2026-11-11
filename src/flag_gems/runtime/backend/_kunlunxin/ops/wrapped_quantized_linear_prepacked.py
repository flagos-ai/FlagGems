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

from flag_gems.ops._wrapped_linear_prepack import unpack_linear_weight
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

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


@libentry()
@triton.jit
def _wrapped_quantized_linear_prepacked_empty_k_kernel(
    output,
    output_scale,
    output_zero_point,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = -tl.load(output_zero_point).to(tl.float32)
    value *= tl.load(output_scale).to(tl.float32)
    tl.store(output + offsets, value, mask=offsets < numel)


@libentry()
@triton.jit
def _quantize_input_kernel(
    input,
    input_scale,
    input_zero_point,
    quantized,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    values = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(input_scale).to(tl.float32)
    zero_point = tl.load(input_zero_point).to(tl.int32)
    quantized_values = _round_half_to_even(values / scale) + zero_point
    quantized_values = tl.minimum(tl.maximum(quantized_values, 0.0), 255.0)
    quantized_values = quantized_values.to(tl.int32) - 128
    tl.store(quantized + offsets, quantized_values.to(tl.int8), mask=mask)


@libentry()
@triton.jit
def _wrapped_quantized_linear_prepacked_kernel(
    input,
    input_scale,
    input_zero_point,
    weight,
    weight_metadata,
    bias,
    output_scale,
    output_zero_point,
    output,
    M,
    N,
    K,
    stride_im,
    BLOCK: tl.constexpr,
):
    """One lane per dense output element, folding the K reduce in registers.

    The per-tile int8 ``tl.dot`` form of this kernel does not lower on this
    backend, so the products are accumulated with plain integer multiplies;
    the two zero-point correction sums are folded in the same sweep.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < M * N
    rows = offs // N
    cols = offs % N

    input_scale_value = tl.load(input_scale).to(tl.float32)
    input_zero_point_value = tl.load(input_zero_point).to(tl.int32)
    weight_scale_value = tl.load(weight_metadata).to(tl.float32)
    weight_zero_point_value = tl.load(weight_metadata + 1).to(tl.int32)

    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    input_sum = tl.zeros((BLOCK,), dtype=tl.int32)
    weight_sum = tl.zeros((BLOCK,), dtype=tl.int32)

    for k in range(0, K):
        quantized_input = tl.load(input + rows * K + k, mask=valid, other=0).to(
            tl.int32
        )
        quantized_weight = tl.load(weight + cols * K + k, mask=valid, other=0).to(
            tl.int32
        )
        acc += quantized_input * quantized_weight
        input_sum += quantized_input
        weight_sum += quantized_weight

    shifted_input_zero_point = input_zero_point_value - 128
    acc = acc - weight_zero_point_value * input_sum
    acc = acc - shifted_input_zero_point * weight_sum
    acc = acc + K * shifted_input_zero_point * weight_zero_point_value

    bias_values = tl.load(bias + cols, mask=valid, other=0.0)
    real_output = acc.to(tl.float32) * input_scale_value * weight_scale_value
    real_output = real_output + bias_values

    output_scale_value = tl.load(output_scale).to(tl.float32)
    output_zero_point_value = tl.load(output_zero_point).to(tl.float32)
    quantized_output = (
        _round_half_to_even(real_output / output_scale_value) + output_zero_point_value
    )
    quantized_output = tl.minimum(tl.maximum(quantized_output, 0.0), 255.0)
    dequantized_output = (
        quantized_output - output_zero_point_value
    ) * output_scale_value

    tl.store(output + offs, dequantized_output, mask=valid)


def _wrapped_quantized_linear_prepacked(
    input,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    logger.debug("GEMS_KUNLUNXIN _WRAPPED_QUANTIZED_LINEAR_PREPACKED")
    if input.dtype != torch.float32:
        raise RuntimeError(f"Quantize only works on Float Tensor, got {input.dtype}")
    if input.ndim < 2:
        raise RuntimeError(
            "The dimension of input tensor should be larger than or equal to 2"
        )
    if out_channel < 0:
        raise RuntimeError("out_channel must be non-negative")
    for name, parameter in (
        ("input_scale", input_scale),
        ("input_zero_point", input_zero_point),
        ("output_scale", output_scale),
        ("output_zero_point", output_zero_point),
    ):
        if parameter.numel() != 1:
            raise RuntimeError(f"{name} must contain one element")
        if parameter.device != input.device:
            raise RuntimeError(f"{name} must be on the input device")
    if packed_weight.device != input.device:
        raise RuntimeError("packed_weight must be on the input device")

    K = input.shape[-1]
    quantized_weight, weight_metadata, bias = unpack_linear_weight(
        packed_weight, out_channel, K
    )

    output_shape = (*input.shape[:-1], out_channel)
    output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    M = 1
    for dimension in input.shape[:-1]:
        M *= dimension
    if M == 0 or out_channel == 0:
        return output
    if K == 0:
        block_size = 256
        grid = (triton.cdiv(output.numel(), block_size),)
        with torch_device_fn.device(input.device):
            _wrapped_quantized_linear_prepacked_empty_k_kernel[grid](
                output,
                output_scale,
                output_zero_point,
                output.numel(),
                BLOCK_SIZE=block_size,
            )
        return output

    input_2d = input.reshape(M, K).contiguous()
    # Quantize once up front: the per-k loop then only multiplies int8 values,
    # which keeps its memory-op count inside the per-kernel pipeline budget.
    quantized_input = torch.empty((M, K), dtype=torch.int8, device=input.device)
    quantize_block = 1024
    with torch_device_fn.device(input.device):
        _quantize_input_kernel[(triton.cdiv(M * K, quantize_block),)](
            input_2d,
            input_scale,
            input_zero_point,
            quantized_input,
            M * K,
            BLOCK_SIZE=quantize_block,
        )
    block_size = 64
    grid = (triton.cdiv(M * out_channel, block_size),)
    with torch_device_fn.device(input.device):
        _wrapped_quantized_linear_prepacked_kernel[grid](
            quantized_input,
            input_scale,
            input_zero_point,
            quantized_weight,
            weight_metadata,
            bias,
            output_scale,
            output_zero_point,
            output,
            M,
            out_channel,
            K,
            input_2d.stride(0),
            BLOCK=block_size,
            num_stages=1,
        )
    return output
