import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import philox as _philox

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.heuristics(runtime.get_heuristic_config("exponential_"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel(
    out_ptr,
    N,
    is_double: tl.constexpr,
    lambd,
    eps,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    if is_double:
        UNROLL: tl.constexpr = 2  # philox generate 128 random bits at a time
    else:
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
        r = tl.reshape(r, [UNROLL * BLOCK], can_reorder=True)
        off = block_offset + tl.arange(0, UNROLL * BLOCK)

        if is_double:
            r = r.to(tl.uint64, bitcast=True)
            f = uint_to_uniform_float(r)
        else:
            f = uint_to_uniform_float(r)
        y = transform_exponential(f, lambd, eps)
        tl.store(out_ptr + off, y, mask=off < N)
        i4_start += num_jobs * BLOCK


@triton.jit
def paste_u64(hi: tl.uint32, lo: tl.uint32):
    hi = hi.to(tl.uint64) << 32
    x = hi | lo.to(tl.uint64)
    return x


@triton.jit
def transform_exponential(u, lambd, eps):
    eps1 = -0.5 * eps
    is_min = u >= 1.0 + eps1
    log = tl.where(is_min, eps1, tl.math.log(u))
    v = -1.0 / lambd * log
    return v


def exponential_(x, lambd: float = 1.0, *, gen=None):
    logger.debug("GEMS_CAMBRICON EXPONENTIAL_")
    dtype = x.dtype
    device = x.device
    inplace = x.is_contiguous()
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    is_double = dtype in (torch.float64,)
    UNROLL = 2 if is_double else 4
    N = x.numel()
    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)
    eps = torch.finfo(dtype).eps
    x_ = x if inplace else torch.empty(x.size(), dtype=dtype, device=device)
    with torch_device_fn.device(device):
        fused_exponential_kernel[grid_fn](
            x_,
            N,
            is_double,
            lambd,
            eps,
            philox_seed,
            philox_offset,
            num_warps=1,
            num_stages=3,
        )
    if not inplace:
        x.copy_(x_)
    return x
