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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

_ALPHA = 1.7580993408473766


@libentry()
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def feature_alpha_dropout_mask_kernel(
    mask_ptr,
    n_features,
    p,
    philox_seed,
    philox_offset,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_features

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 += offsets.to(tl.uint32)
    zero = c0 * 0
    random, _, _, _ = tl.philox(philox_seed, c0, c1, zero, zero)
    keep = uint_to_uniform_float(random) > p
    tl.store(mask_ptr + offsets, keep, mask=active)


@libentry()
@triton.jit(do_not_specialize=["scale", "shift", "dropped_value"])
def feature_alpha_dropout_apply_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    n_channels,
    spatial_size,
    scale,
    shift,
    dropped_value,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    channel_batch = offsets // spatial_size
    feature_offsets = (channel_batch // n_channels) * n_channels + (
        channel_batch % n_channels
    )

    value = tl.load(input_ptr + offsets, mask=active)
    keep = tl.load(mask_ptr + feature_offsets, mask=active)
    kept_value = value * scale + shift
    output = tl.where(keep, kept_value, dropped_value)
    tl.store(output_ptr + offsets, output, mask=active)


def feature_alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS FEATURE_ALPHA_DROPOUT")

    if not (0.0 <= p <= 1.0):
        raise RuntimeError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    if p == 0.0 or not train or input.numel() == 0:
        return input
    if p == 1.0:
        return input * 0
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    if not input.dtype.is_floating_point:
        raise RuntimeError("feature_alpha_dropout only supports floating-point inputs")

    original_input = input
    input = input.contiguous()
    output = torch.empty_like(input)
    n_elements = input.numel()
    n_batch = input.shape[0]
    n_channels = input.shape[1]
    spatial_size = n_elements // (n_batch * n_channels)
    n_features = n_batch * n_channels

    scale = 1.0 / math.sqrt((_ALPHA * _ALPHA * p + 1.0) * (1.0 - p))
    shift = _ALPHA * scale * p
    dropped_value = _ALPHA * scale * (p - 1.0)

    feature_mask = torch.empty(n_features, dtype=torch.bool, device=input.device)
    mask_block_size = 256
    apply_block_size = 1024
    mask_grid = (triton.cdiv(n_features, mask_block_size),)
    apply_grid = (triton.cdiv(n_elements, apply_block_size),)

    with torch_device_fn.device(input.device):
        philox_seed, philox_offset = philox_backend_seed_offset(n_features)
        feature_alpha_dropout_mask_kernel[mask_grid](
            feature_mask,
            n_features,
            p,
            philox_seed,
            philox_offset,
            BLOCK_SIZE=mask_block_size,
        )
        feature_alpha_dropout_apply_kernel[apply_grid](
            input,
            feature_mask,
            output,
            n_elements,
            n_channels,
            spatial_size,
            scale,
            shift,
            dropped_value,
            BLOCK_SIZE=apply_block_size,
        )

    if original_input.is_contiguous():
        return output
    result = torch.empty_like(original_input)
    result.copy_(output)
    return result
