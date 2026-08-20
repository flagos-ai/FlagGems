# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

# Poisson sampling split:
#   lambda <= INVERSE_MAX : inverse-transform (one uniform + CDF accumulation)
#   lambda  > INVERSE_MAX : normal approximation with alpha^2/6 skewness correction
INVERSE_MAX = 20.0
# INVERSE_CAP bounds the CDF loop; 48 covers lambda=20 at 2.5 sigma.
INVERSE_CAP = 48
# Fixed per-element philox stride so poisson() never reads input on the host.
PHILOX_STRIDE = 8
BLOCK_SIZE = 2048
NUM_WARPS = 8


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
def poisson_inverse(lam, lam_bound, seed, c0, c1, z, CAP: tl.constexpr):
    """Inverse-transform sampling: one uniform, accumulate CDF until s >= u."""
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
    u = uint_to_uniform_float(r0)
    p = tl.exp(-lam)
    s = p
    k = (lam * 0).to(tl.int32)
    iters = tl.minimum((lam_bound + 2.5 * tl.sqrt(lam_bound) + 2.0).to(tl.int32), CAP)
    iters = tl.maximum(iters, 1)
    for _ in tl.range(0, iters):
        active = u > s
        k = tl.where(active, k + 1, k)
        denom = tl.where(active, k.to(tl.float32), 1.0)
        p = tl.where(active, p * lam / denom, p)
        s = tl.where(active, s + p, s)
    return k.to(tl.float32)


@triton.jit
def poisson_corrected_normal(lam, seed, c0, c1, z):
    """Normal approximation with alpha^2/6 skewness correction (Box-Muller)."""
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
    u1 = tl.maximum(uint_to_uniform_float(r0), 1e-10)
    u2 = uint_to_uniform_float(r1)
    radius = tl.sqrt(-2.0 * tl.log(u1))
    _, cos_t = high_precision_fast_sin_cos(6.283185307179586 * u2)
    alpha = radius * cos_t
    y = lam + alpha * tl.sqrt(lam) + alpha * alpha / 6.0
    return tl.floor(tl.maximum(y, 0.0) + 0.5)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def poisson_kernel(
    inp_ptr,
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    STRIDE: tl.constexpr,
    IMAX: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    lam = tl.maximum(tl.load(inp_ptr + offs, mask=mask, other=0.0).to(tl.float32), 0.0)
    bmax = tl.max(lam)
    counter = philox_offset + offs.to(tl.int64) * STRIDE
    c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = c0 * 0
    if bmax <= IMAX:
        result = poisson_inverse(lam, bmax, philox_seed, c0, c1, z, CAP)
    else:
        inv = poisson_inverse(lam, IMAX * 1.0, philox_seed, c0, c1, z, CAP)
        nrm = poisson_corrected_normal(lam, philox_seed, c0, c1, z)
        result = tl.where(lam <= IMAX, inv, nrm)
    tl.store(out_ptr + offs, result, mask=mask)


def poisson(input, generator=None):
    """
    Returns a tensor of the same size as input with each element sampled
    from a Poisson distribution with rate parameter given by the corresponding
    element in input.
    """
    logger.debug("GEMS POISSON")

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {input.dtype}"

    inp = input.contiguous()
    n = volume(inp.shape)
    out = torch.empty_like(inp)
    if n == 0:
        return out

    increment = triton.cdiv(n * PHILOX_STRIDE, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    grid = triton.cdiv(n, BLOCK_SIZE)
    with torch_device_fn.device(inp.device):
        poisson_kernel[(grid,)](
            inp,
            out,
            n,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK_SIZE,
            CAP=INVERSE_CAP,
            STRIDE=PHILOX_STRIDE,
            IMAX=INVERSE_MAX,
            num_warps=NUM_WARPS,
        )

    return out
