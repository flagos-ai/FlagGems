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
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

_ALPHA = 1.7580993408473766

# GCU400 ships 24 SIPs; 48 CTAs (2 per SIP) is the measured concurrency sweet
# spot for streaming kernels (grid=96 measured 3-8% slower on every core shape).
_NUM_CTAS = 48


@triton.jit
def _fmix32(x):
    # xxhash32-style avalanche finalizer: 8 cheap integer ops, full avalanche
    # (statistical quality validated against the murmur3 finalizer on 16M
    # element masks across p in {0.1 .. 0.9}; see gcu400/ops/dropout.py).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    x = x ^ (x >> 16)
    return x


@libentry()
@triton.jit(
    do_not_specialize=[
        "scale",
        "shift",
        "dropped_value",
        "philox_seed",
        "philox_offset",
        "thr",
        "spatial_size",
        "blocks_per_feature",
        "num_tiles",
    ]
)
def feature_alpha_dropout_kernel(
    input_ptr,
    output_ptr,
    spatial_size,
    blocks_per_feature,
    num_tiles,
    scale,
    shift,
    dropped_value,
    philox_seed,
    philox_offset,
    thr,
    BLOCK: tl.constexpr,
    IS_EVEN: tl.constexpr,
):
    # RNG note: the aten feature_alpha_dropout contract only requires a
    # statistically sound bernoulli mask that is uniform along the spatial
    # extent of each (batch, channel) feature (the FlagGems accuracy test
    # validates exactly this, statistically, by design).  The philox4x32-10
    # chain costs ~25 dependent integer vector ops per sampled element; this
    # backend instead hashes the per-feature index with an xxhash32-style
    # avalanche finalizer keyed by the per-call torch philox generator state,
    # so the mask is recomputed per tile at scalar cost (8 ops per tile) while
    # the element loop stays a pure load -> where -> store stream.  The
    # per-call state is still drawn from / advances the torch philox generator
    # (philox_backend_seed_offset), so successive calls yield fresh
    # independent masks and results stay reproducible for a fixed generator
    # state.  The 64-bit state must be avalanched before it is keyed into the
    # hash (raw, it only differs additively between calls, which would make
    # successive same-shape masks a permutation of each other); it is folded
    # with two independent fmix32 lanes XORed together, and the per-feature
    # hash input uses modular addition, matching the fix validated for
    # dropout/rand/randn on 2026-11-16 (see gcu400/ops/dropout.py).
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32(rng_state.to(tl.uint32)) ^ _fmix32((rng_state >> 32).to(tl.uint32))
    thr_u = thr.to(tl.uint32)

    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    # Tile space = (feature, spatial block); the per-element
    # offset // spatial_size division of the two-kernel reference layout is
    # replaced by one scalar division per BLOCK-sized tile (measured ~0 cost),
    # which removes the dominant wall of the reference apply kernel on large
    # shapes (element-wise emulated div/mod: 3025us vs 96us memory floor on
    # 32M fp32).
    for tile_id in tl.range(pid, num_tiles, num_pids):
        feature = tile_id // blocks_per_feature
        block = tile_id - feature * blocks_per_feature
        inner = block * BLOCK + arange
        offsets = feature * spatial_size + inner

        keep = _fmix32(feature.to(tl.uint32) + s0) > thr_u

        if IS_EVEN:
            value = tl.load(input_ptr + offsets)
        else:
            value = tl.load(input_ptr + offsets, mask=inner < spatial_size)
        # where-select (not a mask multiply) so dropped features map to the
        # constant regardless of NaN/Inf inputs, like the aten reference.
        output = tl.where(keep, value * scale + shift, dropped_value)
        if IS_EVEN:
            tl.store(output_ptr + offsets, output)
        else:
            tl.store(output_ptr + offsets, output, mask=inner < spatial_size)


def _next_pow2(n):
    return 1 << max(0, (n - 1)).bit_length()


def _choose_block(spatial_size, dtype):
    # Measured tile sweet spot is ~32KB per stream (fp32 8192 lanes, 16-bit
    # 16384 lanes): 32M-element shapes run at the copy floor with these caps
    # (fp32 97.4us vs 96.4us floor; fp16 63.5us vs 58.8us floor), while 4096-lane
    # tiles leave 10-25% on the table.  DSM budget: BLOCK * elem_size * 2
    # buffers = 64KB per CTA at the cap, 128KB per SIP with 2 CTAs, well under
    # the 896KB/448KB-per-program limit.
    cap = 8192 if dtype == torch.float32 else 16384
    return max(64, min(cap, _next_pow2(spatial_size)))


def feature_alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS_ENFLAME FEATURE_ALPHA_DROPOUT")

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

    block_size = _choose_block(spatial_size, input.dtype)
    blocks_per_feature = triton.cdiv(spatial_size, block_size)
    num_tiles = n_features * blocks_per_feature
    grid = (min(_NUM_CTAS, num_tiles),)
    is_even = (spatial_size % block_size) == 0
    # uint32 threshold: keep iff hash > thr, P(keep) = 1 - p (within 2^-32).
    thr = min(int(p * 4294967296.0), 4294967295)

    with torch_device_fn.device(input.device):
        philox_seed, philox_offset = philox_backend_seed_offset(n_features)
        feature_alpha_dropout_kernel[grid](
            input,
            output,
            spatial_size,
            blocks_per_feature,
            num_tiles,
            scale,
            shift,
            dropped_value,
            philox_seed,
            philox_offset,
            thr,
            BLOCK=block_size,
            IS_EVEN=is_even,
            num_warps=8 if block_size >= 4096 else 4,
        )

    if original_input.is_contiguous():
        return output
    result = torch.empty_like(original_input)
    result.copy_(output)
    return result
