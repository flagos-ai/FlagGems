import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.xpu.libdevice import log2

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
# def heur_block(args):
#     if args["N"] <= 512:
#         return 512
#     else:
#         return 1024


def heur_block(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))  # CLUSTER_NUM = 12


def heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "BLOCK": heur_block,
        "num_warps": heur_num_warps,
    }
)
# @triton.heuristics(runtime.get_heuristic_config("exponential_"))
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
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    if is_double:
        d0 = uint_to_uniform_float(paste_u64(r0, r2))
        d1 = uint_to_uniform_float(paste_u64(r1, r3))
        y0 = transform_exponential(d0, lambd, eps)
        y1 = transform_exponential(d1, lambd, eps)
        UNROLL = 2
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    else:
        f0 = uint_to_uniform_float(r0)
        f1 = uint_to_uniform_float(r1)
        f2 = uint_to_uniform_float(r2)
        f3 = uint_to_uniform_float(r3)
        y0 = transform_exponential(f0, lambd, eps)
        y1 = transform_exponential(f1, lambd, eps)
        y2 = transform_exponential(f2, lambd, eps)
        y3 = transform_exponential(f3, lambd, eps)
        UNROLL = 4
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


@triton.jit
def paste_u64(hi: tl.uint32, lo: tl.uint32):
    hi = hi.to(tl.uint64) << 32
    x = hi | lo.to(tl.uint64)
    return x


@triton.jit
def transform_exponential(u, lambd, eps):
    eps1 = -0.5 * eps
    is_min = u >= 1.0 + eps1
    log = tl.where(is_min, eps1, log2(u))
    v = -1.0 / lambd * log
    return v


def exponential_(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS EXPONENTIAL_")
    dtype = x.dtype
    device = x.device
    inplace = x.is_contiguous()
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    is_double = dtype in (torch.float64,)
    UNROLL = 2 if is_double else 4
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    eps = torch.finfo(dtype).eps
    x_ = x if inplace else torch.empty(x.size(), dtype=dtype, device=device)
    with torch_device_fn.device(device):
        fused_exponential_kernel[grid_fn](
            x_, N, is_double, lambd, eps, philox_seed, philox_offset
        )
    if not inplace:
        x.copy_(x_)
    return x
