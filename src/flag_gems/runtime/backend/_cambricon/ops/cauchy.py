import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import philox as _philox

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

PI = tl.constexpr(3.14159265358979323846)
UNROLL = 4
BLOCK_SIZE = 1024


@triton.jit
def uniform_to_cauchy(u, median, sigma):
    u = tl.maximum(1.0e-7, u)
    u = tl.minimum(1.0 - 1.0e-7, u)
    angle = PI * (u - 0.5)
    return median + sigma * (tl.sin(angle) / tl.cos(angle))


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
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    i4_start = pid * BLOCK
    block_start = pid * UNROLL * BLOCK
    step = num_jobs * BLOCK * UNROLL

    for block_offset in range(block_start, N, step):
        sl = (philox_seed & 0xFFFFFFFF).to(tl.uint32)
        sh = ((philox_seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
        r = _philox(BLOCK, sl, sh, c0 + i4_start, c1, 0, 0, 10)
        r = uint_to_uniform_float(r)
        values = uniform_to_cauchy(r, median, sigma)

        off = block_offset + tl.arange(0, UNROLL * BLOCK)
        values = tl.reshape(values, [UNROLL * BLOCK], can_reorder=True)
        tl.store(out_ptr + off, values, mask=off < N, eviction_policy="evict_first")
        i4_start += num_jobs * BLOCK


def cauchy_(self, median=0, sigma=1, *, generator=None):
    logger.debug("GEMS_CAMBRICON CAUCHY_")
    N = volume(self.shape)
    if N == 0:
        return self

    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        cauchy_kernel[grid_fn](
            self,
            N,
            median,
            sigma,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK_SIZE,
            num_warps=1,
            num_stages=3,
        )
    return self


def cauchy(self, median=0, sigma=1, *, generator=None):
    logger.debug("GEMS_CAMBRICON CAUCHY")
    out = torch.empty_like(self)
    cauchy_(out, median, sigma, generator=generator)
    return out
