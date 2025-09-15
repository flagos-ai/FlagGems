import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

try:
    pair_uniform_to_normal = tl.pair_uniform_to_normal
except AttributeError:

    @triton.jit
    def pair_uniform_to_normal(u1, u2):
        """Box-Muller transform"""
        u1 = tl.maximum(1.0e-7, u1)
        th = 6.283185307179586 * u2
        r = tl.sqrt(-2.0 * tl.log(u1))
        return r * tl.cos(th), r * tl.sin(th)


device_ = device
logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@triton.heuristics(runtime.get_heuristic_config("randn"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.uint32)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    n0, n1 = pair_uniform_to_normal(r0, r1)
    n2, n3 = pair_uniform_to_normal(r2, r3)
    off_0 = ((tl.program_id(0) * BLOCK * 4).to(tl.int64) + tl.arange(0, BLOCK)).to(
        tl.int64
    )
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, n0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, n1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, n2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, n3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_MTHREADS RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out
