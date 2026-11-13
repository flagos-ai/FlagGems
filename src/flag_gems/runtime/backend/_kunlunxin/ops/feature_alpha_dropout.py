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

# Kunlunxin(XPU) override for feature_alpha_dropout.
#
# The generic implementation (src/flag_gems/ops/feature_alpha_dropout.py) is
# catastrophically slow on XPU for the large shapes: the apply kernel computes
# the feature index per element via `offsets // spatial_size` (soft integer
# division, ~30 instructions/element) and then redundantly recomputes it via a
# div/mod pair `(channel_batch // n_channels) * n_channels + (channel_batch %
# n_channels)` which is mathematically the identity.  On the (32,64,128,128)
# case this measures ~93 ms vs torch 0.16 ms (0.002x).
#
# Fix: restructure the apply kernel as a 2-D grid (feature, spatial-block) so
# the feature index is `program_id(0)` and the flat input offset is
# `feature_id * spatial_size + spatial_offset` -- zero division, zero modulo.
# The mask is loaded as a per-feature scalar.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

_ALPHA = 1.7580993408473766


@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def _fad_mask_kernel(
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


@triton.jit(do_not_specialize=["spatial_size", "scale", "shift", "dropped_value"])
def _fad_apply_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    spatial_size,
    scale,
    shift,
    dropped_value,
    BLOCK_SPATIAL: tl.constexpr,
):
    feature_id = tl.program_id(0)
    pid = tl.program_id(1)
    spatial_offsets = pid * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    active = spatial_offsets < spatial_size

    keep = tl.load(mask_ptr + feature_id)
    offsets = feature_id * spatial_size + spatial_offsets

    value = tl.load(input_ptr + offsets, mask=active)
    kept_value = value * scale + shift
    output = tl.where(keep, kept_value, dropped_value)
    tl.store(output_ptr + offsets, output, mask=active)


def feature_alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_ALPHA_DROPOUT")

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
    # One program per feature (or per spatial chunk when spatial_size is
    # large); a full-feature block avoids the per-element integer
    # division and keeps the mask load a per-feature scalar.
    apply_block_size = min(triton.next_power_of_2(spatial_size), 16384)

    with torch_device_fn.device(input.device):
        philox_seed, philox_offset = philox_backend_seed_offset(n_features)
        _fad_mask_kernel[(triton.cdiv(n_features, mask_block_size),)](
            feature_mask,
            n_features,
            p,
            philox_seed,
            philox_offset,
            BLOCK_SIZE=mask_block_size,
        )
        _fad_apply_kernel[
            (n_features, triton.cdiv(spatial_size, apply_block_size))
        ](
            input,
            feature_mask,
            output,
            spatial_size,
            scale,
            shift,
            dropped_value,
            BLOCK_SPATIAL=apply_block_size,
        )

    if original_input.is_contiguous():
        return output
    result = torch.empty_like(original_input)
    result.copy_(output)
    return result
