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

from flag_gems.ops._wrapped_linear_prepack import (
    _round_half_to_even,
    unpack_linear_weight,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


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
    stride_ik,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    input_scale_value = tl.load(input_scale).to(tl.float32)
    input_zero_point_value = tl.load(input_zero_point).to(tl.int32)
    weight_scale_value = tl.load(weight_metadata).to(tl.float32)
    weight_zero_point_value = tl.load(weight_metadata + 1).to(tl.int32)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    input_sum = tl.zeros((BLOCK_M,), dtype=tl.int32)
    weight_sum = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        current_k = k_block * BLOCK_K + offsets_k
        input_ptrs = (
            input + offsets_m[:, None] * stride_im + current_k[None, :] * stride_ik
        )
        input_mask = (offsets_m[:, None] < M) & (current_k[None, :] < K)
        input_values = tl.load(input_ptrs, mask=input_mask, other=0.0).to(tl.float32)
        quantized_input = (
            _round_half_to_even(input_values / input_scale_value)
            + input_zero_point_value
        )
        quantized_input = tl.minimum(tl.maximum(quantized_input, 0.0), 255.0).to(
            tl.uint8
        )
        shifted_input = (quantized_input.to(tl.int32) - 128).to(tl.int8)
        shifted_input = tl.where(input_mask, shifted_input, 0)

        weight_ptrs = weight + offsets_n[None, :] * K + current_k[:, None]
        weight_mask = (offsets_n[None, :] < N) & (current_k[:, None] < K)
        quantized_weight = tl.load(weight_ptrs, mask=weight_mask, other=0)

        accumulator += tl.dot(shifted_input, quantized_weight)
        input_sum += tl.sum(shifted_input.to(tl.int32), axis=1)
        weight_sum += tl.sum(quantized_weight.to(tl.int32), axis=0)

    shifted_input_zero_point = input_zero_point_value - 128
    accumulator -= weight_zero_point_value * input_sum[:, None]
    accumulator -= shifted_input_zero_point * weight_sum[None, :]
    accumulator += K * shifted_input_zero_point * weight_zero_point_value

    bias_values = tl.load(bias + offsets_n, mask=offsets_n < N, other=0.0)
    real_output = (
        accumulator.to(tl.float32) * input_scale_value * weight_scale_value
        + bias_values[None, :]
    )
    output_scale_value = tl.load(output_scale).to(tl.float32)
    output_zero_point_value = tl.load(output_zero_point).to(tl.float32)
    quantized_output = (
        _round_half_to_even(real_output / output_scale_value) + output_zero_point_value
    )
    quantized_output = tl.minimum(tl.maximum(quantized_output, 0.0), 255.0)
    dequantized_output = (
        quantized_output - output_zero_point_value
    ) * output_scale_value

    output_ptrs = output + offsets_m[:, None] * N + offsets_n[None, :]
    output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(output_ptrs, dequantized_output, mask=output_mask)


def _wrapped_quantized_linear_prepacked(
    input,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    logger.debug("GEMS _WRAPPED_QUANTIZED_LINEAR_PREPACKED")
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

    input_2d = input.reshape(M, K)
    block_m = 32
    block_n = 64 if M >= 256 and out_channel >= 1024 else 32
    block_k = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(out_channel, block_n))
    with torch_device_fn.device(input.device):
        _wrapped_quantized_linear_prepacked_kernel[grid](
            input_2d,
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
            input_2d.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
    return output
