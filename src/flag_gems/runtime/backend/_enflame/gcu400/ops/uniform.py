import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

UNROLL = 4


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "N"])
def uniform_kernel_gcu400(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
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
    scale = to - from_
    r0 = uint_to_uniform_float(r0) * scale + from_
    r1 = uint_to_uniform_float(r1) * scale + from_
    r2 = uint_to_uniform_float(r2) * scale + from_
    r3 = uint_to_uniform_float(r3) * scale + from_
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N)
    tl.store(out_ptr + off_1, r1, mask=off_1 < N)
    tl.store(out_ptr + off_2, r2, mask=off_2 < N)
    tl.store(out_ptr + off_3, r3, mask=off_3 < N)


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS UNIFORM GCU400")
    N = volume(self.shape)
    dtype = self.dtype

    if N <= 16384:
        BLOCK = 256
    elif N <= 65536:
        BLOCK = 1024
    elif dtype == torch.float32:
        BLOCK = 32768
    else:
        BLOCK = 16384

    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    nw = 2 if dtype == torch.float32 else 4
    with torch_device_fn.device(self.device):
        uniform_kernel_gcu400[grid](
            self, N, philox_seed, philox_offset, from_, to,
            BLOCK=BLOCK, num_warps=nw, num_stages=1,
        )
    return self
