import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


@triton.heuristics(runtime.get_heuristic_config("uniform"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "p"])
def geometric_kernel(
    out_ptr,
    N,
    p,
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

    # Convert random uint32 to uniform float in [0, 1)
    u0 = uint_to_uniform_float(r0)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u3 = uint_to_uniform_float(r3)

    # Geometric distribution: ceil(log(u) / log(1-p))
    # where u is uniform in (0, 1). Use log1p for numerical stability.
    log1p_minus_p = tl.log(1.0 - p)
    y0 = tl.ceil(tl.log(u0) / log1p_minus_p)
    y1 = tl.ceil(tl.log(u1) / log1p_minus_p)
    y2 = tl.ceil(tl.log(u2) / log1p_minus_p)
    y3 = tl.ceil(tl.log(u3) / log1p_minus_p)

    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4


def geometric_(self, p=0.5, *, generator=None):
    logger.debug("GEMS GEOMETRIC_")
    N = volume(self.shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        geometric_kernel[grid_fn](self, N, p, philox_seed, philox_offset)
    return self


def geometric(self, p=0.5, *, generator=None):
    logger.debug("GEMS GEOMETRIC")
    out = torch.empty_like(self)
    N = volume(out.shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        geometric_kernel[grid_fn](out, N, p, philox_seed, philox_offset)
    return out
