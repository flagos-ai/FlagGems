import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as device_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

UNROLL = 4
MAX_GRID = 65535
BLOCK_SIZE_SUB = 1024


@triton.jit
def high_precision_fast_sin_cos(x):
    two_pi = 6.283185307179586
    x = x - two_pi * tl.floor(x / two_pi + 0.5)
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


@triton.jit
def pair_uniform_to_normal_fast(u1, u2):
    u1 = tl.maximum(1.0e-7, u1)
    theta = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    sin_t, cos_t = high_precision_fast_sin_cos(theta)
    return r * cos_t, r * sin_t


def _compute_block_size(N):
    """Compute BLOCK_SIZE so that grid = cdiv(N, BLOCK_SIZE * 4) <= MAX_GRID."""
    min_block = triton.cdiv(N, MAX_GRID * UNROLL)
    # Round up to next multiple of BLOCK_SIZE_SUB
    block_size = triton.cdiv(min_block, BLOCK_SIZE_SUB) * BLOCK_SIZE_SUB
    return max(block_size, BLOCK_SIZE_SUB)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    # Each program handles BLOCK_SIZE * 4 output elements.
    # Within the program, iterate in sub-blocks of BLOCK_SIZE_SUB.
    base_output = pid * BLOCK_SIZE * 4
    base_philox = pid * BLOCK_SIZE

    for sub in range(BLOCK_SIZE // BLOCK_SIZE_SUB):
        i4 = tl.arange(0, BLOCK_SIZE_SUB)
        c0 = c0_base + (base_philox + sub * BLOCK_SIZE_SUB + i4).to(tl.uint32)
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)
        n0, n1 = pair_uniform_to_normal_fast(r0, r1)
        n2, n3 = pair_uniform_to_normal_fast(r2, r3)

        out_base = base_output + sub * BLOCK_SIZE_SUB * 4
        off_0 = out_base + tl.arange(0, BLOCK_SIZE_SUB)
        off_1 = off_0 + BLOCK_SIZE_SUB
        off_2 = off_1 + BLOCK_SIZE_SUB
        off_3 = off_2 + BLOCK_SIZE_SUB

        tl.store(out_ptr + off_0, n0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, n1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, n2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, n3, mask=off_3 < N, eviction_policy="evict_first")


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ASCEND RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    block_size = _compute_block_size(N)
    grid = (triton.cdiv(N, block_size * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        randn_kernel[grid](out, N, philox_seed, philox_offset, block_size, BLOCK_SIZE_SUB)
    return out
