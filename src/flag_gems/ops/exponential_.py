import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

MIN_NORMAL_F32 = 1.17549435e-38
# Largest value less than 1.0 to avoid log(1)=0 edge (though harmless)
MAX_U_F32 = 0.99999994  # nextafter(1.0, 0.0) in float32


@triton.jit
def safe_fast_log(x):
    # Construct FP32 constants matching x's dtype
    min_normal = x * 0.0 + 1.17549435e-38
    max_u = x * 0.0 + 0.99999994

    x = tl.minimum(tl.maximum(x, min_normal), max_u)

    bits = x.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) - 127
    # mantissa = (bits & 0x7FFFFF).to(tl.float32) * (1.0 / (1 << 23)) + 1.0
    mantissa = (bits & 0x7FFFFF).to(tl.float32) * (1.0 / 8388608) + 1.0

    m1 = mantissa - 1.0
    log_m = m1 * (1.0 + m1 * (-0.5 + m1 * (0.3333333333 - m1 * 0.25)))
    log_val = log_m + exponent.to(tl.float32) * 0.6931471805599453

    return log_val


# ===== Kernel with constexpr switch =====
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=32, num_stages=3),
        triton.Config({"BLOCK": 2048}, num_warps=32, num_stages=4),
    ],
    key=["N", "is_double"],
)
# @triton.heuristics(runtime.get_heuristic_config("exponential_"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel(
    out_ptr,
    N,
    is_double,
    inv_lambd,
    eps_minus,
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
        y0 = transform_exponential(d0, inv_lambd, eps_minus)
        y1 = transform_exponential(d1, inv_lambd, eps_minus)
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
        y0 = transform_exponential(f0, inv_lambd, eps_minus)
        y1 = transform_exponential(f1, inv_lambd, eps_minus)
        y2 = transform_exponential(f2, inv_lambd, eps_minus)
        y3 = transform_exponential(f3, inv_lambd, eps_minus)

        UNROLL = 4
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_last")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_last")
        tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_last")
        tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_last")


@triton.jit
def paste_u64(hi: tl.uint32, lo: tl.uint32):
    hi = hi.to(tl.uint64) << 32
    x = hi | lo.to(tl.uint64)
    return x


@triton.jit
def transform_exponential(u, inv_lambd, eps_minus):
    # eps1 = -0.5 * eps
    is_min = u >= 1.0 + eps_minus
    # log = tl.where(is_min, eps1, tl.math.log(u))
    # is_min = u >= compare_val
    log = tl.where(is_min, eps_minus, safe_fast_log(u))
    v = -inv_lambd * log
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
    eps_minus = -0.5 * eps
    inv_lambd = 1.0 / lambd
    x_ = x if inplace else torch.empty(x.size(), dtype=dtype, device=device)
    with torch_device_fn.device(device):
        fused_exponential_kernel[grid_fn](
            x_, N, is_double, inv_lambd, eps_minus, philox_seed, philox_offset
        )
    if not inplace:
        x.copy_(x_)
    return x
