import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


@triton.jit
def safe_fast_log_f32(x):
    min_normal = (x * 0.0 + 1.17549435e-38).to(tl.float32)
    max_u = x * 0.0 + 0.99999994
    x = tl.minimum(tl.maximum(x, min_normal), max_u)
    bits = x.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) - 127
    mantissa = (bits & 0x7FFFFF).to(tl.float32) * (1.0 / 8388608.0) + 1.0
    m1 = mantissa - 1.0
    return (
        m1 * (1.0 + m1 * (-0.5 + m1 * (0.3333333333 - m1 * 0.25)))
        + exponent.to(tl.float32) * 0.6931471805599453
    )


@triton.jit
def safe_fast_log_f64(x):
    min_normal = x * 0.0 + 2.2250738585072014e-308
    max_u = x * 0.0 + (1.0 - 2.220446049250313e-16)
    x = tl.minimum(tl.maximum(x, min_normal), max_u)
    bits = x.to(tl.int64, bitcast=True)
    exponent = (bits >> 52) - 1023
    mantissa = (bits & 0x000FFFFFFFFFFFFF).to(tl.float64) * (
        1.0 / 4503599627370496.0
    ) + 1.0
    m1 = mantissa - 1.0
    return (
        m1 * (1.0 + m1 * (-0.5 + m1 * (0.3333333333333333 - m1 * 0.25)))
        + exponent.to(tl.float64) * 0.6931471805599453
    )


@triton.jit
def paste_u64(hi: tl.uint32, lo: tl.uint32):
    return (hi.to(tl.uint64) << 32) | lo.to(tl.uint64)


@triton.jit
def transform_exponential_f32_precise(u, inv_lambd, eps_minus):
    log = tl.where(u >= 1.0 + eps_minus, eps_minus, tl.math.log(u))
    # log = tl.log(tl.maximum(u, 1e-38))
    return -inv_lambd * log


@triton.jit
def transform_exponential_f32_fast(u, inv_lambd, eps_minus):
    log = tl.where(u >= 1.0 + eps_minus, eps_minus, safe_fast_log_f32(u))
    # log = tl.log(tl.maximum(u, 1e-38))
    return -inv_lambd * log


if device.vendor_name == "iluvatar":
    transform_exponential_f32 = transform_exponential_f32_precise
else:
    transform_exponential_f32 = transform_exponential_f32_fast


@triton.jit
def transform_exponential_f64(u, inv_lambd, eps_minus):
    log = tl.where(u >= 1.0 + eps_minus, eps_minus, safe_fast_log_f64(u))
    return -inv_lambd * log


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel_f32(
    out_ptr, N, inv_lambd, eps_minus, philox_seed, philox_offset, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)

    y0 = transform_exponential_f32(uint_to_uniform_float(r0), inv_lambd, eps_minus)
    y1 = transform_exponential_f32(uint_to_uniform_float(r1), inv_lambd, eps_minus)
    y2 = transform_exponential_f32(uint_to_uniform_float(r2), inv_lambd, eps_minus)
    y3 = transform_exponential_f32(uint_to_uniform_float(r3), inv_lambd, eps_minus)

    start = pid.to(tl.uint64) * BLOCK * 4
    off0 = start + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, y0, mask=off0 < N)
    tl.store(out_ptr + off1, y1, mask=off1 < N)
    tl.store(out_ptr + off2, y2, mask=off2 < N)
    tl.store(out_ptr + off3, y3, mask=off3 < N)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel_f64(
    out_ptr, N, inv_lambd, eps_minus, philox_seed, philox_offset, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)

    u0 = uint_to_uniform_float(paste_u64(r0, r2))
    u1 = uint_to_uniform_float(paste_u64(r1, r3))

    y0 = transform_exponential_f64(u0, inv_lambd, eps_minus)
    y1 = transform_exponential_f64(u1, inv_lambd, eps_minus)

    start = pid.to(tl.uint64) * BLOCK * 2
    off0 = start + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK

    tl.store(out_ptr + off0, y0, mask=off0 < N)
    tl.store(out_ptr + off1, y1, mask=off1 < N)


def exponential_(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS EXPONENTIAL_")

    dtype = x.dtype
    device = x.device
    inplace = x.is_contiguous()
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    N = x.numel()
    inv_lambd = 1.0 / lambd
    eps_minus = -0.5 * torch.finfo(dtype).eps

    out = x if inplace else torch.empty_like(x)

    if dtype is torch.float64:
        UNROLL = 2
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        increment = triton.cdiv(N, UNROLL)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        with torch_device_fn.device(device):
            fused_exponential_kernel_f64[grid](
                out, N, inv_lambd, eps_minus, philox_seed, philox_offset
            )
    else:
        UNROLL = 4
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        increment = triton.cdiv(N, UNROLL)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        with torch_device_fn.device(device):
            fused_exponential_kernel_f32[grid](
                out, N, inv_lambd, eps_minus, philox_seed, philox_offset
            )

    if not inplace:
        x.copy_(out)
    return x
