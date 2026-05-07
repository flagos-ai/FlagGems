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

logger = logging.getLogger(__name__)

PI = tl.constexpr(3.14159265358979323846)


@triton.jit
def uniform_to_cauchy(u, median, sigma):
    # Transform uniform [0, 1) to Cauchy using inverse CDF
    # X = median + sigma * tan(PI * (u - 0.5))
    # Clamp u to avoid tan(±PI/2) which is undefined
    u = tl.maximum(1.0e-7, u)
    u = tl.minimum(1.0 - 1.0e-7, u)
    # tan(x) = sin(x) / cos(x)
    angle = PI * (u - 0.5)
    return median + sigma * (tl.sin(angle) / tl.cos(angle))


# @triton.heuristics(runtime.get_heuristic_config("cauchy"))
configs = [
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=configs, key=["N"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "median", "sigma"])
def cauchy_kernel(
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
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    c0 = uniform_to_cauchy(r0, median, sigma)
    c1 = uniform_to_cauchy(r1, median, sigma)
    c2 = uniform_to_cauchy(r2, median, sigma)
    c3 = uniform_to_cauchy(r3, median, sigma)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, c0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, c1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, c2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, c3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4


def cauchy_(self, median=0, sigma=1, *, generator=None):
    """
    In-place Cauchy distribution sampler.

    Fills self with elements drawn from the Cauchy distribution:
    f(x) = 1 / (π * sigma * (1 + ((x - median) / sigma)^2))

    Uses inverse transform sampling: X = median + sigma * tan(π * (U - 0.5))
    where U ~ Uniform(0, 1).
    """
    logger.debug("GEMS CAUCHY_")
    shape = self.shape
    device = self.device
    N = volume(shape)
    if N == 0:
        return self
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        cauchy_kernel[grid_fn](self, N, median, sigma, philox_seed, philox_offset)
    return self


def cauchy(self, median=0, sigma=1, *, generator=None):
    """
    Out-of-place Cauchy distribution sampler.

    Returns a new tensor with elements drawn from the Cauchy distribution.
    """
    logger.debug("GEMS CAUCHY")
    out = torch.empty_like(self)
    cauchy_(out, median, sigma, generator=generator)
    return out
