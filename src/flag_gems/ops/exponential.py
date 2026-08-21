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
def paste_u64(hi: tl.uint32, lo: tl.uint32):
    return (hi.to(tl.uint64) << 32) | lo.to(tl.uint64)


@triton.jit
def safe_poly_log_f32(x):
    min_normal = (x * 0.0 + tl.full((), 1.17549435e-38, tl.float32)).to(tl.float32)
    max_u = x * 0.0 + 0.99999994
    x = tl.minimum(tl.maximum(x, min_normal), max_u)
    bits = x.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) - 127
    mantissa = (bits & 0x7FFFFF).to(tl.float32) * (1.0 / 8388608.0) + 1.0
    t = mantissa - 1.0
    p = 3.0102415366e-02
    p = p * t - 1.3011859043e-01
    p = p * t + 2.8330324636e-01
    p = p * t - 4.8915625549e-01
    p = p * t + 9.9901031459e-01
    p = p * t + 2.2125781657e-05
    return p + exponent.to(tl.float32) * 0.6931471805599453


@triton.jit
def safe_poly_log7_f32(x):
    min_normal = (x * 0.0 + tl.full((), 1.17549435e-38, tl.float32)).to(tl.float32)
    max_u = x * 0.0 + 0.99999994
    x = tl.minimum(tl.maximum(x, min_normal), max_u)
    bits = x.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) - 127
    mantissa = (bits & 0x7FFFFF).to(tl.float32) * (1.0 / 8388608.0) + 1.0
    t = mantissa - 1.0
    p = 1.0118982276e-02
    p = p * t - 5.2624353622e-02
    p = p * t + 1.3076410292e-01
    p = p * t - 2.2283540252e-01
    p = p * t + 3.2697268479e-01
    p = p * t - 4.9920646516e-01
    p = p * t + 9.9995747546e-01
    p = p * t + 5.6260525100e-07
    return p + exponent.to(tl.float32) * 0.6931471805599453


@triton.jit
def transform_exponential_f32_poly5(u, inv_lambd, eps_minus):
    is_min = u >= 1.0 - 1e-4
    log = tl.where(is_min, eps_minus, safe_poly_log_f32(u))
    return -inv_lambd * log


@triton.jit
def transform_exponential_f32_poly7(u, inv_lambd, eps_minus):
    is_min = u >= 1.0 - 1e-4
    log = tl.where(is_min, eps_minus, safe_poly_log7_f32(u))
    return -inv_lambd * log


if device.vendor_name == "iluvatar":
    transform_exponential_f32 = transform_exponential_f32_poly7
else:
    transform_exponential_f32 = transform_exponential_f32_poly5


@triton.jit
def transform_exponential_f64(u, inv_lambd, eps_minus):
    u = tl.maximum(u, tl.full((), 2.2250738585072014e-308, tl.float64))
    log = tl.where(u >= 1.0 + eps_minus, eps_minus, tl.math.log(u))
    return -inv_lambd * log


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def exponential_kernel(
    out_ptr,
    N,
    is_double: tl.constexpr,
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
    pid = tl.program_id(0)
    offsets = pid * BLOCK * 2 + tl.arange(0, BLOCK)
    c0_first = c0 + offsets
    c0_second = c0_first + BLOCK
    z = c0_first * 0
    ra0, ra1, ra2, ra3 = tl.philox(philox_seed, c0_first, c1, z, z)
    rb0, rb1, rb2, rb3 = tl.philox(philox_seed, c0_second, c1, z, z)
    if is_double:
        d0 = uint_to_uniform_float(paste_u64(ra0, ra2))
        d1 = uint_to_uniform_float(paste_u64(ra1, ra3))
        d2 = uint_to_uniform_float(paste_u64(rb0, rb2))
        d3 = uint_to_uniform_float(paste_u64(rb1, rb3))
        y0 = transform_exponential_f64(d0, inv_lambd, eps_minus)
        y1 = transform_exponential_f64(d1, inv_lambd, eps_minus)
        y2 = transform_exponential_f64(d2, inv_lambd, eps_minus)
        y3 = transform_exponential_f64(d3, inv_lambd, eps_minus)
        UNROLL = 4
        start = pid.to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")
    else:
        fa0 = uint_to_uniform_float(ra0)
        fa1 = uint_to_uniform_float(ra1)
        fa2 = uint_to_uniform_float(ra2)
        fa3 = uint_to_uniform_float(ra3)
        fb0 = uint_to_uniform_float(rb0)
        fb1 = uint_to_uniform_float(rb1)
        fb2 = uint_to_uniform_float(rb2)
        fb3 = uint_to_uniform_float(rb3)
        ya0 = transform_exponential_f32(fa0, inv_lambd, eps_minus)
        ya1 = transform_exponential_f32(fa1, inv_lambd, eps_minus)
        ya2 = transform_exponential_f32(fa2, inv_lambd, eps_minus)
        ya3 = transform_exponential_f32(fa3, inv_lambd, eps_minus)
        yb0 = transform_exponential_f32(fb0, inv_lambd, eps_minus)
        yb1 = transform_exponential_f32(fb1, inv_lambd, eps_minus)
        yb2 = transform_exponential_f32(fb2, inv_lambd, eps_minus)
        yb3 = transform_exponential_f32(fb3, inv_lambd, eps_minus)
        UNROLL = 8
        start = pid.to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        off_4 = off_3 + BLOCK
        off_5 = off_4 + BLOCK
        off_6 = off_5 + BLOCK
        off_7 = off_6 + BLOCK
        tl.store(out_ptr + off_0, ya0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, ya1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, ya2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, ya3, mask=off_3 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_4, yb0, mask=off_4 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_5, yb1, mask=off_5 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_6, yb2, mask=off_6 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_7, yb3, mask=off_7 < N, eviction_policy="evict_first")


def exponential(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS EXPONENTIAL")
    dtype = x.dtype
    device = x.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    is_double = dtype in (torch.float64,)
    UNROLL = 4 if is_double else 8
    N = x.numel()
    inv_lambd = 1.0 / lambd
    eps_minus = -0.5 * torch.finfo(dtype).eps

    res = torch.empty(x.shape, dtype=dtype, device=device)

    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL // 2) + 2048
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        exponential_kernel[grid_fn](
            res, N, is_double, inv_lambd, eps_minus, philox_seed, philox_offset
        )
    return res
