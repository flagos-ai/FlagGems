import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume


@triton.jit
def fast_sin_cos_normalized(x, REDUCED: tl.constexpr):
    """sin/cos for x already in [-pi, pi]. No normalization needed."""
    x2 = x * x
    if REDUCED:
        sin_x = x * (
            0.99999999999999999999
            + x2 * (
                -0.16666666666666666654
                + x2 * (
                    0.00833333333333332876
                    + x2 * (
                        -0.00019841269841269616
                        + x2 * 2.755731922398589e-6
                    )
                )
            )
        )
        cos_x = (
            1.0
            + x2 * (
                -0.49999999999999999983
                + x2 * (
                    0.04166666666666666636
                    + x2 * (
                        -0.00138888888888888742
                        + x2 * 2.4801587301587299e-5
                    )
                )
            )
        )
    else:
        sin_x = x * (
            0.99999999999999999999
            + x2 * (
                -0.16666666666666666654
                + x2 * (
                    0.00833333333333332876
                    + x2 * (
                        -0.00019841269841269616
                        + x2 * (
                            2.755731922398589e-6
                            + x2 * -2.505210838544172e-8
                        )
                    )
                )
            )
        )
        cos_x = (
            1.0
            + x2 * (
                -0.49999999999999999983
                + x2 * (
                    0.04166666666666666636
                    + x2 * (
                        -0.00138888888888888742
                        + x2 * (
                            2.4801587301587299e-5
                            + x2 * -2.755731922398581e-7
                        )
                    )
                )
            )
        )
    return sin_x, cos_x


@triton.jit
def pair_uniform_to_normal(u1, u2, REDUCED: tl.constexpr):
    u1 = tl.maximum(1.0e-7, u1)
    r = tl.sqrt(-2.0 * tl.log(u1))
    x = 3.141592653589793 * (2.0 * u2 - 1.0)
    sin_t, cos_t = fast_sin_cos_normalized(x, REDUCED)
    return r * cos_t, r * sin_t


device_ = device
logger = logging.getLogger(__name__)

UNROLL = 4


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    REDUCED: tl.constexpr = False,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    n0, n1 = pair_uniform_to_normal(r0, r1, REDUCED)
    n2, n3 = pair_uniform_to_normal(r2, r3, REDUCED)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, n0, mask=off_0 < N)
    tl.store(out_ptr + off_1, n1, mask=off_1 < N)
    tl.store(out_ptr + off_2, n2, mask=off_2 < N)
    tl.store(out_ptr + off_3, n3, mask=off_3 < N)


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    if N <= 65536:
        BLOCK = 1024
    elif N <= 1048576:
        BLOCK = 4096
    else:
        BLOCK = 8192
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        reduced = (dtype != torch.float32)
        randn_kernel[grid](out, N, philox_seed, philox_offset, BLOCK=BLOCK, REDUCED=reduced, num_warps=4)
    return out
