import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import philox as _philox

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
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


@triton.heuristics(runtime.get_heuristic_config("randn"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
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

    res = tl.empty([UNROLL, BLOCK], dtype=tl.float32)
    for block_offset in range(block_start, N, step):
        sl = (philox_seed & 0xFFFFFFFF).to(tl.uint32)
        sh = ((philox_seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
        r = _philox(BLOCK, sl, sh, c0 + i4_start, c1, 0, 0, 10)
        r = uint_to_uniform_float(r)

        res[0, :], res[1, :] = pair_uniform_to_normal(r[:, 0], r[:, 1])
        res[2, :], res[3, :] = pair_uniform_to_normal(r[:, 2], r[:, 3])

        off = block_offset + tl.arange(0, BLOCK * UNROLL)
        tl.store(
            out_ptr + off,
            tl.reshape(res, [BLOCK * UNROLL], can_reorder=True),
            mask=off < N,
        )
        i4_start += num_jobs * BLOCK


UNROLL = 4


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_CAMBRICON RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )
    philox_seed, philox_offset = philox_backend_seed_offset(N)
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](
            out, N, philox_seed, philox_offset, num_stages=3, num_warps=1
        )
    return out
