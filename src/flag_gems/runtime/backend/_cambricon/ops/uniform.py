import logging

import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import philox as _philox

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.heuristics(runtime.get_heuristic_config("uniform"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4  # philox generate 128 random bits at a time
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
        r = uint_to_uniform_float(r) * (to - from_) + from_
        r = tl.reshape(r, [UNROLL * BLOCK], can_reorder=True)

        off = block_offset + tl.arange(0, UNROLL * BLOCK)
        tl.store(out_ptr + off, r, mask=off < N)
        i4_start += num_jobs * BLOCK


UNROLL = 4


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_CAMBRICON UNIFORM")
    N = volume(self.shape)
    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        uniform_kernel[grid_fn](
            self, N, philox_seed, philox_offset, from_, to, num_warps=1, num_stages=3
        )
    return self
