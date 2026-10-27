import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)
TWO_PI = tl.constexpr(6.283185307179586)

UNROLL = 4

_SMALL_N_THRESHOLD = 65536
_SMALL_BLOCK = tl.constexpr(256)


@triton.jit
def fast_sin_cos(x):
    """High-precision minimax sin/cos on [-pi, pi] (~1.5e-9 max error)."""
    x = x - TWO_PI * tl.floor(x / TWO_PI + 0.5)
    x2 = x * x

    s_c0 = 0.99999999999999999999
    s_c1 = -0.16666666666666666654
    s_c2 = 0.00833333333333332876
    s_c3 = -0.00019841269841269616
    s_c4 = 2.755731922398589e-6
    s_c5 = -2.505210838544172e-8
    sin_x = x * (
        s_c0 + x2 * (s_c1 + x2 * (s_c2 + x2 * (s_c3 + x2 * (s_c4 + x2 * s_c5))))
    )

    c_c0 = 1.0
    c_c1 = -0.49999999999999999983
    c_c2 = 0.04166666666666666636
    c_c3 = -0.00138888888888888742
    c_c4 = 2.4801587301587299e-5
    c_c5 = -2.755731922398581e-7
    cos_x = c_c0 + x2 * (c_c1 + x2 * (c_c2 + x2 * (c_c3 + x2 * (c_c4 + x2 * c_c5))))

    return sin_x, cos_x


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "mean", "std"])
def _log_normal_kernel_impl(
    out_ptr,
    N,
    mean,
    std,
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

    f0 = uint_to_uniform_float(r0)
    f1 = uint_to_uniform_float(r1)
    f2 = uint_to_uniform_float(r2)
    f3 = uint_to_uniform_float(r3)

    u1 = tl.maximum(1.0e-7, f0)
    th = TWO_PI * f1
    r = tl.sqrt(-2.0 * tl.math.log2(u1) * LN2)
    s, c = fast_sin_cos(th)
    n0 = r * c
    n1 = r * s

    u2 = tl.maximum(1.0e-7, f2)
    th2 = TWO_PI * f3
    r2v = tl.sqrt(-2.0 * tl.math.log2(u2) * LN2)
    s2, c2 = fast_sin_cos(th2)
    n2 = r2v * c2
    n3 = r2v * s2

    sl = std * LOG2E
    ml = mean * LOG2E
    y0 = tl.math.exp2(tl.fma(n0, sl, ml))
    y1 = tl.math.exp2(tl.fma(n1, sl, ml))
    y2 = tl.math.exp2(tl.fma(n2, sl, ml))
    y3 = tl.math.exp2(tl.fma(n3, sl, ml))

    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "mean", "std"])
def log_normal_kernel(
    out_ptr,
    N,
    mean,
    std,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    _log_normal_kernel_impl(
        out_ptr, N, mean, std, philox_seed, philox_offset, BLOCK=BLOCK
    )


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "mean", "std"])
def log_normal_kernel_small(
    out_ptr,
    N,
    mean,
    std,
    philox_seed,
    philox_offset,
):
    _log_normal_kernel_impl(
        out_ptr, N, mean, std, philox_seed, philox_offset, BLOCK=_SMALL_BLOCK
    )


def log_normal(x, mean=1.0, std=2.0, *, generator=None):
    logger.debug("GEMS_ILUVATAR LOG_NORMAL")
    dtype = x.dtype
    device = x.device
    res = torch.empty(x.shape, dtype=dtype, device=device)
    N = res.numel()

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        if N <= _SMALL_N_THRESHOLD:
            grid = (triton.cdiv(N, _SMALL_BLOCK * UNROLL),)
            log_normal_kernel_small[grid](
                res, N, mean, std, philox_seed, philox_offset, num_warps=8
            )
        else:
            grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
            log_normal_kernel[grid_fn](res, N, mean, std, philox_seed, philox_offset)
    return res
