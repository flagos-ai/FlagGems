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
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel_f32_unroll8(
    out_ptr, N, inv_lambd, eps_minus, philox_seed, philox_offset, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)

    c0_first = c0 + offsets * 4
    c0_second = c0_first + BLOCK * 4
    z = c0_first * 0

    r0_0, r1_0, r2_0, r3_0 = tl.philox(philox_seed, c0_first, c1, z, z)
    r0_1, r1_1, r2_1, r3_1 = tl.philox(philox_seed, c0_second, c1, z, z)

    y0_0 = transform_exponential_f32(uint_to_uniform_float(r0_0), inv_lambd, eps_minus)
    y1_0 = transform_exponential_f32(uint_to_uniform_float(r1_0), inv_lambd, eps_minus)
    y2_0 = transform_exponential_f32(uint_to_uniform_float(r2_0), inv_lambd, eps_minus)
    y3_0 = transform_exponential_f32(uint_to_uniform_float(r3_0), inv_lambd, eps_minus)

    y0_1 = transform_exponential_f32(uint_to_uniform_float(r0_1), inv_lambd, eps_minus)
    y1_1 = transform_exponential_f32(uint_to_uniform_float(r1_1), inv_lambd, eps_minus)
    y2_1 = transform_exponential_f32(uint_to_uniform_float(r2_1), inv_lambd, eps_minus)
    y3_1 = transform_exponential_f32(uint_to_uniform_float(r3_1), inv_lambd, eps_minus)

    base_off = pid.to(tl.uint64) * BLOCK * 8
    off0 = base_off + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK
    off4 = off3 + BLOCK
    off5 = off4 + BLOCK
    off6 = off5 + BLOCK
    off7 = off6 + BLOCK

    tl.store(out_ptr + off0, y0_0, mask=off0 < N)
    tl.store(out_ptr + off1, y1_0, mask=off1 < N)
    tl.store(out_ptr + off2, y2_0, mask=off2 < N)
    tl.store(out_ptr + off3, y3_0, mask=off3 < N)
    tl.store(out_ptr + off4, y0_1, mask=off4 < N)
    tl.store(out_ptr + off5, y1_1, mask=off5 < N)
    tl.store(out_ptr + off6, y2_1, mask=off6 < N)
    tl.store(out_ptr + off7, y3_1, mask=off7 < N)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
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
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_exponential_kernel_f32_small(
    out_ptr, N, inv_lambd, eps_minus, philox_seed, philox_offset, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    base_idx = pid * BLOCK * 4
    c0_i = c0 + tl.arange(0, BLOCK)
    z = c0_i * 0

    r0, r1, r2, r3 = tl.philox(philox_seed, c0_i, c1, z, z)

    y0 = transform_exponential_f32(uint_to_uniform_float(r0), inv_lambd, eps_minus)
    y1 = transform_exponential_f32(uint_to_uniform_float(r1), inv_lambd, eps_minus)
    y2 = transform_exponential_f32(uint_to_uniform_float(r2), inv_lambd, eps_minus)
    y3 = transform_exponential_f32(uint_to_uniform_float(r3), inv_lambd, eps_minus)

    off0 = base_idx + tl.arange(0, BLOCK)
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
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
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
    base_idx = pid * BLOCK * 4
    block_offset = tl.arange(0, BLOCK)
    c0_base = c0 + block_offset
    z = c0_base * 0

    r0_0, r1_0, r2_0, r3_0 = tl.philox(philox_seed, c0_base, c1, z, z)
    r0_1, r1_1, r2_1, r3_1 = tl.philox(philox_seed, c0_base + BLOCK, c1, z, z)

    u0_0 = uint_to_uniform_float(paste_u64(r0_0, r2_0))
    u1_0 = uint_to_uniform_float(paste_u64(r1_0, r3_0))
    u0_1 = uint_to_uniform_float(paste_u64(r0_1, r2_1))
    u1_1 = uint_to_uniform_float(paste_u64(r1_1, r3_1))

    y0_0 = transform_exponential_f64(u0_0, inv_lambd, eps_minus)
    y1_0 = transform_exponential_f64(u1_0, inv_lambd, eps_minus)
    y0_1 = transform_exponential_f64(u0_1, inv_lambd, eps_minus)
    y1_1 = transform_exponential_f64(u1_1, inv_lambd, eps_minus)

    off0 = base_idx + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, y0_0, mask=off0 < N)
    tl.store(out_ptr + off1, y1_0, mask=off1 < N)
    tl.store(out_ptr + off2, y0_1, mask=off2 < N)
    tl.store(out_ptr + off3, y1_1, mask=off3 < N)


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
    elif dtype in (torch.float16, torch.bfloat16) and N < 65536:
        UNROLL = 4
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        increment = triton.cdiv(N, UNROLL)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        with torch_device_fn.device(device):
            fused_exponential_kernel_f32_small[grid](
                out, N, inv_lambd, eps_minus, philox_seed, philox_offset
            )
    else:
        UNROLL = 8
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        increment = triton.cdiv(N, UNROLL)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        with torch_device_fn.device(device):
            fused_exponential_kernel_f32_unroll8[grid](
                out, N, inv_lambd, eps_minus, philox_seed, philox_offset
            )

    if not inplace:
        x.copy_(out)
    return x
