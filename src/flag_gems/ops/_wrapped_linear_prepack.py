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
import triton.language.extra.libdevice as libdevice

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_HEADER_BYTES = 16
_PACK_MAGIC = 260821.0
_PACK_VERSION = 1.0


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
def _wrapped_linear_prepack_kernel(
    weight,
    weight_scale,
    weight_zero_point,
    bias,
    metadata,
    packed_bias,
    packed_weight,
    N,
    K,
    stride_wn,
    stride_wk,
    stride_bias,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    weight_mask = offsets < N * K
    rows = offsets // K
    cols = offsets % K

    scale = tl.load(weight_scale).to(tl.float32)
    inverse_scale = libdevice.rcp_rn(scale)
    zero_point = tl.load(weight_zero_point).to(tl.float32)
    values = tl.load(
        weight + rows * stride_wn + cols * stride_wk,
        mask=weight_mask,
        other=0.0,
    ).to(tl.float32)
    # Match ATen quantize_per_tensor, which multiplies by the reciprocal
    # scale. Division can round differently at FP32 half-way boundaries.
    quantized = _round_half_to_even(values * inverse_scale) + zero_point
    quantized = tl.minimum(tl.maximum(quantized, -128.0), 127.0)
    tl.store(packed_weight + offsets, quantized.to(tl.int8), mask=weight_mask)

    bias_mask = offsets < N
    bias_values = tl.load(bias + offsets * stride_bias, mask=bias_mask, other=0.0).to(
        tl.float32
    )
    tl.store(packed_bias + offsets, bias_values, mask=bias_mask)

    metadata_mask = offsets < 4
    metadata_values = tl.where(
        offsets == 0,
        scale,
        tl.where(
            offsets == 1,
            zero_point,
            tl.where(offsets == 2, PACK_MAGIC, PACK_VERSION),
        ),
    )
    tl.store(metadata + offsets, metadata_values, mask=metadata_mask)


@libentry()
@triton.jit
def _wrapped_linear_prepack_empty_kernel(
    weight_scale,
    weight_zero_point,
    bias,
    metadata,
    packed_bias,
    N,
    stride_bias,
    PACK_MAGIC: tl.constexpr,
    PACK_VERSION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bias_mask = offsets < N
    bias_values = tl.load(bias + offsets * stride_bias, mask=bias_mask, other=0.0).to(
        tl.float32
    )
    tl.store(packed_bias + offsets, bias_values, mask=bias_mask)

    scale = tl.load(weight_scale).to(tl.float32)
    zero_point = tl.load(weight_zero_point).to(tl.float32)
    metadata_values = tl.where(
        offsets == 0,
        scale,
        tl.where(
            offsets == 1,
            zero_point,
            tl.where(offsets == 2, PACK_MAGIC, PACK_VERSION),
        ),
    )
    tl.store(metadata + offsets, metadata_values, mask=offsets < 4)


def unpack_linear_weight(packed_weight, out_channel, in_channel=None):
    """Return views of the private GPU packing format used by the paired op."""
    if packed_weight.dtype != torch.uint8 or packed_weight.ndim != 1:
        raise RuntimeError("packed_weight must be a one-dimensional uint8 tensor")
    if out_channel < 0:
        raise RuntimeError("out_channel must be non-negative")
    payload_bytes = packed_weight.numel() - _HEADER_BYTES - 4 * out_channel
    if payload_bytes < 0:
        raise RuntimeError("packed_weight has an invalid FlagGems linear layout")
    if out_channel == 0:
        if payload_bytes != 0:
            raise RuntimeError("packed_weight has an invalid FlagGems linear layout")
        K = 0 if in_channel is None else in_channel
    else:
        if payload_bytes % out_channel != 0:
            raise RuntimeError("packed_weight has an invalid FlagGems linear layout")
        K = payload_bytes // out_channel
        if in_channel is not None and K != in_channel:
            raise RuntimeError("input and packed weight dimensions do not match")
    metadata = packed_weight[:_HEADER_BYTES].view(torch.float32)
    bias_start = _HEADER_BYTES
    bias_end = bias_start + 4 * out_channel
    bias = packed_weight[bias_start:bias_end].view(torch.float32)
    weight = packed_weight[bias_end:].view(torch.int8).view(out_channel, K)
    return weight, metadata, bias


def _wrapped_linear_prepack(weight, weight_scale, weight_zero_point, bias):
    logger.debug("GEMS _WRAPPED_LINEAR_PREPACK")
    if weight.dtype != torch.float32:
        raise RuntimeError(f"Quantize only works on Float Tensor, got {weight.dtype}")
    if weight.ndim != 2:
        raise RuntimeError("fbgemm weight packing only packs matrices not vectors.")
    N, K = weight.shape
    if weight_scale.numel() != 1 or weight_zero_point.numel() != 1:
        raise RuntimeError("weight scale and zero point must contain one element")
    if bias.dtype != torch.float32 or bias.ndim != 1 or bias.numel() != N:
        raise RuntimeError("bias must be a float32 vector with out_channel elements")
    if not (
        weight.device == weight_scale.device == weight_zero_point.device == bias.device
    ):
        raise RuntimeError("all prepack inputs must be on the same device")

    packed_numel = _HEADER_BYTES + 4 * N + N * K
    packed = torch.empty(packed_numel, dtype=torch.uint8, device=weight.device)
    metadata = packed[:_HEADER_BYTES].view(torch.float32)
    bias_end = _HEADER_BYTES + 4 * N
    packed_bias = packed[_HEADER_BYTES:bias_end].view(torch.float32)
    quantized_weight = packed[bias_end:].view(torch.int8)

    block_size = 1024
    with torch_device_fn.device(weight.device):
        if N == 0 or K == 0:
            grid = (triton.cdiv(max(N, 4), block_size),)
            _wrapped_linear_prepack_empty_kernel[grid](
                weight_scale,
                weight_zero_point,
                bias,
                metadata,
                packed_bias,
                N,
                bias.stride(0),
                PACK_MAGIC=_PACK_MAGIC,
                PACK_VERSION=_PACK_VERSION,
                BLOCK_SIZE=block_size,
            )
        else:
            grid = (triton.cdiv(max(N * K, N), block_size),)
            _wrapped_linear_prepack_kernel[grid](
                weight,
                weight_scale,
                weight_zero_point,
                bias,
                metadata,
                packed_bias,
                quantized_weight,
                N,
                K,
                weight.stride(0),
                weight.stride(1),
                bias.stride(0),
                PACK_MAGIC=_PACK_MAGIC,
                PACK_VERSION=_PACK_VERSION,
                BLOCK_SIZE=block_size,
            )
    return packed
