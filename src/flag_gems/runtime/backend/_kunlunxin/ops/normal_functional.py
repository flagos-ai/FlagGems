# Copyright 2026, The FlagOS Contributors.
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
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

from .randn import pair_uniform_to_normal

logger = logging.getLogger(__name__)

UNROLL = 4


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "mean", "std"])
def _normal_fused_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    mean,
    std,
    BLOCK: tl.constexpr,
    FULL: tl.constexpr,
):
    # Single-pass normal_functional: draw the standard-normal sample with the
    # same Philox layout as the generic implementation and apply the affine
    # (val * std + mean) transform in the same kernel, so out-of-place normal
    # costs one launch instead of a randn launch + a separate pointwise
    # transform launch + a second full read/write of the buffer.
    # When FULL is true the grid covers the buffer exactly, so plain unmasked
    # stores are emitted; otherwise the four stores carry a boundary mask.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O, 5)
    u0 = uint_to_uniform_float(r0)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u3 = uint_to_uniform_float(r3)
    n0, n1 = pair_uniform_to_normal(u0, u1)
    n2, n3 = pair_uniform_to_normal(u2, u3)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    if FULL:
        tl.store(out_ptr + off_0, n0 * std + mean)
        tl.store(out_ptr + off_1, n1 * std + mean)
        tl.store(out_ptr + off_2, n2 * std + mean)
        tl.store(out_ptr + off_3, n3 * std + mean)
    else:
        tl.store(out_ptr + off_0, n0 * std + mean, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1 * std + mean, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2 * std + mean, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3 * std + mean, mask=off_3 < N)


def normal_functional(self, mean=0.0, std=1.0, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN NORMAL_FUNCTIONAL")
    shape = self.shape
    device = self.device

    # Out-of-place sibling of normal_: draw standard-normal samples with the
    # philox RNG and apply the affine (val * std + mean) transform in a single
    # fused kernel -- one launch, one write pass, instead of a randn launch
    # followed by a separate pointwise transform that re-reads and re-writes
    # the whole buffer.
    out = torch.empty(shape, device=device, dtype=self.dtype)
    N = volume(shape)
    if N == 0:
        # Empty tensor: nothing to generate; skip the launch (the philox
        # increment would be 0 anyway, so the generator state is unaffected).
        return out
    # Bounded, shape-derived BLOCK plus a plain integer-tuple grid: the launch
    # grid must not be a grid lambda because on the xpubin backend the grid is
    # part of the compile cache key, which would recompile the kernel on every
    # call (fixed multi-hundred-ms latency regardless of shape).
    cluster_num = 12
    BLOCK = min(
        triton.next_power_of_2(triton.cdiv(N, cluster_num * UNROLL)),
        1024,
    )
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    FULL = N % (BLOCK * UNROLL) == 0
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        _normal_fused_kernel[grid](
            out, N, philox_seed, philox_offset, mean, std, BLOCK, FULL
        )

    return out
