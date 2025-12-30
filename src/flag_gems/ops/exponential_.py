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
    mantissa = (bits & 0x7FFFFF).to(tl.float32) * (1.0 / (1 << 23)) + 1.0

    m1 = mantissa - 1.0
    log_m = m1 * (1.0 + m1 * (-0.5 + m1 * (0.3333333333 - m1 * 0.25)))
    log_val = log_m + exponent.to(tl.float32) * 0.6931471805599453

    return log_val


# ===== Unified transform function inside kernel =====
@triton.jit
def transform_exponential_dispatch(u, inv_lambd, eps, USE_FAST_MATH: tl.constexpr):
    u = tl.maximum(u, eps)
    if USE_FAST_MATH:
        # Only valid for FP32; assume u is already FP32
        log_val = safe_fast_log(u)
    else:
        log_val = tl.math.log(u)
    return -log_val * inv_lambd


# ===== Kernel with constexpr switch =====
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=16),
        triton.Config({"BLOCK": 2048}, num_warps=16),
        triton.Config({"BLOCK": 4096}, num_warps=16),
        triton.Config({"BLOCK": 8192}, num_warps=32),
    ],
    key=["N", "is_double", "USE_FAST_MATH"],
)
@triton.jit
def fused_exponential_kernel_switch(
    out_ptr,
    N,
    is_double,
    inv_lambd,
    eps,
    philox_seed,
    philox_offset,
    USE_FAST_MATH: tl.constexpr,
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
        # For double, ignore USE_FAST_MATH (or you could add fast_log64 later)
        d0 = uint_to_uniform_float(paste_u64(r0, r2))
        d1 = uint_to_uniform_float(paste_u64(r1, r3))
        y0 = -tl.math.log(tl.maximum(d0, eps)) * inv_lambd
        y1 = -tl.math.log(tl.maximum(d1, eps)) * inv_lambd
        UNROLL = 2
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    else:
        # Promote to FP32 for computation
        f0 = uint_to_uniform_float(r0).to(tl.float32)
        f1 = uint_to_uniform_float(r1).to(tl.float32)
        f2 = uint_to_uniform_float(r2).to(tl.float32)
        f3 = uint_to_uniform_float(r3).to(tl.float32)

        eps_f32 = eps.to(tl.float32)
        inv_lambd_f32 = inv_lambd.to(tl.float32)

        y0 = transform_exponential_dispatch(f0, inv_lambd_f32, eps_f32, USE_FAST_MATH)
        y1 = transform_exponential_dispatch(f1, inv_lambd_f32, eps_f32, USE_FAST_MATH)
        y2 = transform_exponential_dispatch(f2, inv_lambd_f32, eps_f32, USE_FAST_MATH)
        y3 = transform_exponential_dispatch(f3, inv_lambd_f32, eps_f32, USE_FAST_MATH)

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


# ===== User-facing function with switch =====
def exponential_(
    x,
    lambd: float = 1.0,
    *,
    generator=None,
    use_fast_math: bool = True,  # <-- add new parameter
):
    logger.debug(f"GEMS EXPONENTIAL_ (use_fast_math={use_fast_math})")
    dtype = x.dtype
    device = x.device
    inplace = x.is_contiguous()
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    is_double = dtype in (torch.float64,)
    UNROLL = 2 if is_double else 4
    N = x.numel()

    # Grid function that captures USE_FAST_MATH as constexpr
    def grid_fn(meta):
        return (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    eps = torch.finfo(dtype).eps
    inv_lambd = 1.0 / lambd

    x_ = x if inplace else torch.empty(x.size(), dtype=dtype, device=device)

    with torch_device_fn.device(device):
        fused_exponential_kernel_switch[grid_fn](
            x_,
            N,
            is_double,
            inv_lambd,
            eps,
            philox_seed,
            philox_offset,
            USE_FAST_MATH=use_fast_math,  # <-- new parameter
        )

    if not inplace:
        x.copy_(x_)
    return x
