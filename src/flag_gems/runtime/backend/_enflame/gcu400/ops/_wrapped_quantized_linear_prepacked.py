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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Persistent-grid sizing for the input quantization pass (one program per SIP).
_QUANTIZE_NUM_SIPS = 24
_QUANTIZE_BLOCK = 8192


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
def _wrapped_quantized_linear_prepacked_quantize_kernel(
    input,
    quantized_input,
    numel,
    input_scale,
    input_zero_point,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize the whole input once, storing (q - 128) as int8.

    Doing this in a separate streaming pass removes the per-``pid_n``
    re-quantization that dominated the fused kernel for large output channels.
    """
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    scale = tl.load(input_scale).to(tl.float32)
    zero_point = tl.load(input_zero_point).to(tl.int32)
    num_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_programs):
        index = block_id * BLOCK_SIZE + offsets
        mask = index < numel
        values = tl.load(input + index, mask=mask, other=0.0).to(tl.float32)
        quantized = _round_half_to_even(values / scale) + zero_point
        quantized = tl.minimum(tl.maximum(quantized, 0.0), 255.0)
        tl.store(
            quantized_input + index,
            (quantized.to(tl.int32) - 128).to(tl.int8),
            mask=mask,
        )


@libentry()
@triton.jit
def _wrapped_quantized_linear_prepacked_kernel(
    quantized_input,
    weight,
    weight_metadata,
    bias,
    output_scale,
    output_zero_point,
    output,
    M,
    N,
    K,
    input_scale,
    input_zero_point,
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
        k_mask = current_k < K
        input_mask = (offsets_m[:, None] < M) & k_mask[None, :]
        shifted_input = tl.load(
            quantized_input + offsets_m[:, None] * K + current_k[None, :],
            mask=input_mask,
            other=0,
        )

        weight_ptrs = weight + offsets_n[None, :] * K + current_k[:, None]
        weight_mask = (offsets_n[None, :] < N) & k_mask[:, None]
        quantized_weight = tl.load(weight_ptrs, mask=weight_mask, other=0)
        quantized_weight = tl.where(weight_mask, quantized_weight, 0)

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


def _gemm_tile(out_elements):
    """Large output tiles win once the output is big enough to fill the SIPs."""
    if out_elements >= 256 * 1024:
        return 256, 256, 64, 16
    return 128, 64, 64, 8


def _wrapped_quantized_linear_prepacked(
    input,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    logger.debug("GEMS_ENFLAME _WRAPPED_QUANTIZED_LINEAR_PREPACKED")
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
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()
    quantized_input = torch.empty((M, K), dtype=torch.int8, device=input.device)
    block_m, block_n, block_k, num_warps = _gemm_tile(M * out_channel)
    grid = (triton.cdiv(M, block_m), triton.cdiv(out_channel, block_n))
    with torch_device_fn.device(input.device):
        _wrapped_quantized_linear_prepacked_quantize_kernel[
            (min(triton.cdiv(M * K, _QUANTIZE_BLOCK), _QUANTIZE_NUM_SIPS),)
        ](
            input_2d,
            quantized_input,
            M * K,
            input_scale,
            input_zero_point,
            BLOCK_SIZE=_QUANTIZE_BLOCK,
            num_warps=4,
        )
        _wrapped_quantized_linear_prepacked_kernel[grid](
            quantized_input,
            quantized_weight,
            weight_metadata,
            bias,
            output_scale,
            output_zero_point,
            output,
            M,
            out_channel,
            K,
            input_scale,
            input_zero_point,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
    return output
