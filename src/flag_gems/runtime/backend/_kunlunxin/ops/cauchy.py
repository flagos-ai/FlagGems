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

# Kunlunxin (XPU) implementation of cauchy_ / cauchy (random Cauchy sampler).
#
# Optimizations relative to the generic implementation
# (flag_gems/ops/cauchy.py):
#   1. Fixed configs instead of `triton.autotune(configs=6, key=["N"])`.
#      On XPU every distinct N re-benchmarks all 6 configs at the first call
#      (each run pays ~100-1000 ms; empirically (64,64) paid 408 ms,
#      (4096,4096) 1046 ms, first 10K call 609 ms) and the autotune wrapper
#      adds ~4 us of host overhead per call.  Fixed config eliminates both:
#      N <= 65536 -> BLOCK=256/num_warps=8 (measured 14.9 us vs 18.7 us for
#      the autotune path on the 10K shapes), larger N -> BLOCK=1024/8w
#      (identical to the autotune pick, within noise).  The launch grid is a
#      constant tuple, so the launch grid does not participate in the
#      compilation cache key (grid=lambda performs a full recompile per call
#      on this backend, see KERNEL_OPT_EXPERIENCE.md 2.4 #41).
#   2. `uint_to_uniform_float` rewritten without compare/select:
#      `x ^ (x >> 31)` computes |x| in two's-complement magnitude
#      (x >= 0 -> x, x < 0 -> ~x = -x - 1) with 2 integer ops instead of
#      cmp + select + neg + sub; bit-identical mod 2^32 (verified).
#   3. Store path: plain masked stores with `evict_first` (streaming output,
#      keep it out of L2), same as the generic.  We measured and REJECTED
#      clamped-index tails (`off = tl.minimum(off, N-1)` before a masked
#      store): the int min on the access path destroys the store lowering
#      (5.4x slower on every shape: (4096,4096) bf16 7.6 ms -> 42.3 ms).
#      The plain mask is honored here (verified by the guard-region probe).
#   4. The sine/cosine transform stays `tl.sin(angle) / tl.cos(angle)`:
#      a polynomial sin/cos (no externs) was measured 30-50% SLOWER on this
#      backend (655M f32 295.6 -> 334.3 ms), i.e. the P800 sinf/cosf are not
#      a soft-extern bottleneck for this kernel.
#
# NOTE on the numerical tail: same semantics as the generic (u clamped to
# [1e-7, 1-1e-7]), so values are in [-3.2e6, 3.2e6] * sigma + median; no
# NaN/inf for the shapes/dtypes exercised by the tests.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

PI = tl.constexpr(3.14159265358979323846)


@triton.jit
def _uint_to_uniform_float(x):
    # Maps u32 in [0, 2^32) to f32 in [0, 1).
    # x ^ (x >> 31) replaces the compare/select of the generic version
    # (tl.where(x < 0, -x - 1, x)) with 2 integer ops; identical mod 2^32.
    x = x.to(tl.int32, bitcast=True)
    x = x ^ (x >> 31)
    return x.to(tl.float32) * 4.6566127342e-10


@triton.jit
def _uniform_to_cauchy(u, median, sigma):
    # X = median + sigma * tan(PI * (u - 0.5)), inverse CDF of the Cauchy.
    u = tl.maximum(1.0e-7, u)
    u = tl.minimum(1.0 - 1.0e-7, u)
    # tan(x) = sin(x) / cos(x)
    angle = PI * (u - 0.5)
    return median + sigma * (tl.sin(angle) / tl.cos(angle))


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "median", "sigma"])
def _cauchy_kernel(
    out_ptr,
    N,
    median,
    sigma,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = _uint_to_uniform_float(r0)
    r1 = _uint_to_uniform_float(r1)
    r2 = _uint_to_uniform_float(r2)
    r3 = _uint_to_uniform_float(r3)
    c0 = _uniform_to_cauchy(r0, median, sigma)
    c1 = _uniform_to_cauchy(r1, median, sigma)
    c2 = _uniform_to_cauchy(r2, median, sigma)
    c3 = _uniform_to_cauchy(r3, median, sigma)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, c0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, c1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, c2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, c3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4
# Fixed configs (measured on XPU P800, harness-style do_bench):
#   N <= 65536 -> 256/8w (10K shapes: 14.9 us vs 27.9 us for 1024/8w)
#   N >  65536 -> 1024/8w (2.5M+: 1.183 ms vs 1.284 ms for 256/8w)
BLOCK_SMALL = 256
WARPS_SMALL = 8
BLOCK_LARGE = 1024
WARPS_LARGE = 8
_SMALL_N = 1 << 16


def _launch_cauchy(out, N, median, sigma, philox_seed, philox_offset):
    if N <= _SMALL_N:
        block = BLOCK_SMALL
        warps = WARPS_SMALL
    else:
        block = BLOCK_LARGE
        warps = WARPS_LARGE
    grid = (triton.cdiv(N, block * UNROLL),)
    _cauchy_kernel[grid](
        out,
        N,
        median,
        sigma,
        philox_seed,
        philox_offset,
        BLOCK=block,
        num_warps=warps,
        num_stages=2,
    )


def cauchy_(self, median=0, sigma=1, *, generator=None):
    """
    In-place Cauchy distribution sampler.

    Fills self with elements drawn from the Cauchy distribution:
    f(x) = 1 / (pi * sigma * (1 + ((x - median) / sigma)^2))

    Uses inverse transform sampling: X = median + sigma * tan(pi * (U - 0.5))
    where U ~ Uniform(0, 1).
    """
    logger.debug("GEMS_KUNLUNXIN CAUCHY_")
    shape = self.shape
    device = self.device
    N = volume(shape)
    if N == 0:
        return self
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        _launch_cauchy(self, N, float(median), float(sigma), philox_seed, philox_offset)
    return self


def cauchy(self, median=0, sigma=1, *, generator=None):
    """
    Out-of-place Cauchy distribution sampler.
    Returns a new tensor with elements drawn from the Cauchy distribution.
    """
    logger.debug("GEMS_KUNLUNXIN CAUCHY")
    out = torch.empty_like(self)
    cauchy_(out, median, sigma, generator=generator)
    return out
