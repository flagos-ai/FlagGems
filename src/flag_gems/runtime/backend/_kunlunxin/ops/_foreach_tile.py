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

"""Tiling plan and launch knobs for the Kunlunxin ``_foreach_*`` fast paths.

Sizing a per-tensor launch is not specific to one operator family, so the plan,
its environment knobs and the output allocation live here instead of being
repeated inside every kernel module.

The per-tensor plan is:

* **one tile per CTA** -- while one program streams its tile the next one
  computes, so loads and stores overlap; a single huge tile per program serialises
  the two.
* **at most ``GEMS_XPU_FOREACH_CTAS`` programs**, capped by the tile count.
* **small inputs are split until they span ``GEMS_XPU_FOREACH_MIN_TILES``
  programs** -- a tensor that fits in one tile runs on one program and cannot
  overlap anything.

Every knob is read once at import time, so set the environment before the
process starts.
"""

import os

import torch
import triton

# Only these float dtypes are taken over: integers have ATen promotion and
# wraparound rules, complex has its own kernels, so both fall back to generic.
FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# One launch processes at most this many equal-sized tensors in the 16-slot
# kernels; longer lists fall back to the per-tensor path.
K16_SLOTS = 16
# The 16-way unrolled single launch only pays off while launch cost dominates;
# past ~2^19 elements it loses to the big-tile path (PR #7051 crossover).  Raise
# this knob to re-test the "one launch for large shapes" hypothesis.
K16_MAX_NUMEL = int(os.environ.get("GEMS_XPU_FOREACH_K16_MAX", "0") or 0) or (
    1 << 19
)

# Defaults come from the 2026-10-25 sweep on 16 x [4096,4096] (dtype equal
# weight): tile=131072 with one tile per CTA measured 0.876 against 0.798 for the
# old 2M-tile / 8-CTA arrangement, and the 16-slot single launch for large shapes
# measured only 0.555.
TILE_DEFAULT = 131072
CTAS_DEFAULT = 128
TILE_CAP = int(os.environ.get("GEMS_XPU_FOREACH_TILE", "0") or 0) or TILE_DEFAULT
CTAS_CAP = int(os.environ.get("GEMS_XPU_FOREACH_CTAS", "0") or 0) or CTAS_DEFAULT
# A tensor whose whole range fits in one tile runs on a single program, so every
# load waits for the store that follows it.  Split until the range spans
# MIN_TILES programs; stop at MIN_TILE because smaller shapes are launch-bound.
MIN_TILES = int(os.environ.get("GEMS_XPU_FOREACH_MIN_TILES", "0") or 0) or 2
MIN_TILE = 16384
# XTriton DMA batching knobs.  Swept on 2026-10-25: 16 combinations
# (UNROLL 4/8/16/32 x BUF 4096/8192/16384/32768) all landed in 1.999-2.019 ms on
# 16 x [4096,4096], a 0.99% spread, so this axis is not the bottleneck (the cost
# is per-element instruction issue, not DMA batching).  Knobs kept for re-tests.
UNROLL = int(os.environ.get("GEMS_XPU_FOREACH_UNROLL", "0") or 0) or 8
BUF = int(os.environ.get("GEMS_XPU_FOREACH_BUF", "0") or 0) or 4096


def pick_block(n):
    """``(block, even)`` for the 16-slot kernels: largest power of two dividing n."""
    for block in (8192, 4096, 2048, 1024):
        if n % block == 0:
            return block, True
    return 1024, False


def num_warps_for(tile_size):
    """Warp count that matches the per-program tile size."""
    return 16 if tile_size >= 4096 else (8 if tile_size >= 2048 else 4)


def big_tile_plan(n):
    """``(num_ctas, tiles_per_cta, tile_size, even)`` for the per-tensor path.

    ``even`` requires BOTH that ``n`` is tile-aligned and that the tile count is
    an exact multiple of the CTA count -- otherwise a program would address a
    tile past the buffer, and the unmasked body has no guard (a runtime
    ``if tile_id < num_tiles`` does not lower on this backend).
    """
    tile_size = min(TILE_CAP, triton.next_power_of_2(n))
    # Keep small tensors spread over more than one program: with a single tile
    # the whole tensor runs on one CTA and loads never overlap stores.
    while tile_size > MIN_TILE and triton.cdiv(n, tile_size) < MIN_TILES:
        tile_size //= 2
    num_tiles = max(1, triton.cdiv(n, tile_size))

    want = min(CTAS_CAP, num_tiles)
    num_ctas = want
    while num_ctas > 1 and num_tiles % num_ctas:
        num_ctas -= 1
    even = n % tile_size == 0
    if num_ctas * 2 < want:
        # The largest divisor is too small (e.g. a prime tile count): prefer
        # concurrency and let the masked body absorb the stray tile ids.
        num_ctas = want
        even = False
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    return num_ctas, tiles_per_cta, tile_size, even


def alloc_dsts(tensors, out_dtype):
    """Outputs for the out-of-place path, from one allocation when possible.

    ``[torch.empty_like(t) for t in tensors]`` costs one allocator round trip per
    tensor; on 16 tensors that is ~50 us of host time, which showed up as the
    entire small-shape latency (0.0686 ms vs 0.0148 ms in-place).  A single flat
    buffer sliced into views has the same layout because every tensor here is
    contiguous, and it keeps the fast path allocation-free per tensor.
    """
    if all(t.is_contiguous() for t in tensors):
        n = tensors[0].numel()
        buf = torch.empty(
            len(tensors) * n, dtype=out_dtype, device=tensors[0].device
        )
        return [buf[i * n : (i + 1) * n].view(t.shape) for i, t in enumerate(tensors)]
    return [torch.empty_like(t, dtype=out_dtype) for t in tensors]
