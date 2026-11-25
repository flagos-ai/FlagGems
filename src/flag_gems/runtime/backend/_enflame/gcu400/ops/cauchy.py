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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs). The generic implementation launches
# cdiv(N, BLOCK * UNROLL) CTAs which exceeds the 65535 grid.x hardware
# limit for large inputs (2**28 elements -> 65536 CTAs ->
# OutOfResources). Striding from a fixed grid keeps every launch within
# the limit regardless of input size.
_GRID_SIZE = 48

PI = tl.constexpr(3.14159265358979323846)


@triton.jit
def uniform_to_cauchy(u, median, sigma):
    # Transform uniform [0, 1) to Cauchy using inverse CDF
    # X = median + sigma * tan(PI * (u - 0.5))
    # Clamp u to avoid tan(±PI/2) which is undefined
    u = tl.maximum(1.0e-7, u)
    u = tl.minimum(1.0 - 1.0e-7, u)
    # tan(x) = sin(x) / cos(x). NOTE: measured on GCU400, the libdevice
    # tan variant (tl_extra_shim.tan) is ~4x SLOWER than sin/cos here
    # (21.5ms vs 5.2ms on 2**28 fp16) -- the iluvatar-specific tuning in
    # the generic implementation does not carry over to GCU.
    angle = PI * (u - 0.5)
    return median + sigma * (tl.sin(angle) / tl.cos(angle))


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "median", "sigma"])
def cauchy_kernel(
    out_ptr,
    N,
    median,
    sigma,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Fixed Grid + grid-stride over the UNROLL=4 tile space. The Philox
    # counter is derived from the GLOBAL tile id, so each output element
    # keeps exactly the random stream it had under the large-grid launch
    # (reproducibility with a fixed seed is preserved).
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    philox_c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    num_tile = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for tile_id in tl.range(
        ext.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        philox_c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        # tl.range's induction variable is int64 on GCU; the Philox counter
        # must stay 32-bit (64-bit c0 selects the slow bitcast path in
        # tl.philox and fails to compile), so cast explicitly.
        i4 = tile_id.to(tl.int32) * BLOCK + tl.arange(0, BLOCK)
        philox_c0 += i4
        _O = philox_c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, philox_c0, philox_c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)
        c0 = uniform_to_cauchy(r0, median, sigma)
        c1 = uniform_to_cauchy(r1, median, sigma)
        c2 = uniform_to_cauchy(r2, median, sigma)
        c3 = uniform_to_cauchy(r3, median, sigma)
        off_0 = tile_id * BLOCK * 4 + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        tl.store(out_ptr + off_0, c0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, c1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, c2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, c3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4


def _launch(out: torch.Tensor, N: int, median, sigma, generator):
    # Tuned on GCU400 (sweep BLOCK x num_warps x num_stages on 2**28 fp16):
    # BLOCK=1024 + num_warps=4 + num_stages=2 is optimal (5.17ms; 256/512
    # blocks are 2-4x worse, 2048 slightly worse). The op is compute-bound
    # (Philox + sin/cos), so larger blocks do not help.
    block = 1024
    num_warps = 4
    num_stages = 2
    num_tile = triton.cdiv(N, block * UNROLL)
    grid = (min(_GRID_SIZE, num_tile), 1, 1)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(out.device):
        cauchy_kernel[grid](
            out,
            N,
            median,
            sigma,
            philox_seed,
            philox_offset,
            BLOCK=block,
            GRID_SIZE=min(_GRID_SIZE, num_tile),
            NUM_STAGES=num_stages,
            num_warps=num_warps,
        )


def cauchy_(self, median=0, sigma=1, *, generator=None):
    """
    In-place Cauchy distribution sampler.

    Fills self with elements drawn from the Cauchy distribution:
    f(x) = 1 / (π * sigma * (1 + ((x - median) / sigma)^2))

    Uses inverse transform sampling: X = median + sigma * tan(π * (U - 0.5))
    where U ~ Uniform(0, 1).
    """
    logger.debug("GEMS_ENFLAME CAUCHY_")
    N = self.numel()
    if N == 0:
        return self
    _launch(self, N, median, sigma, generator)
    return self


def cauchy(self, median=0, sigma=1, *, generator=None):
    """
    Out-of-place Cauchy distribution sampler.

    Returns a new tensor with elements drawn from the Cauchy distribution.
    """
    logger.debug("GEMS_ENFLAME CAUCHY")
    out = torch.empty_like(self)
    if out.numel() == 0:
        return out
    _launch(out, out.numel(), median, sigma, generator)
    return out
