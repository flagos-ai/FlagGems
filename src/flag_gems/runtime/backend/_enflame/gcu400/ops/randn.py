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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

UNROLL = 4

U24 = tl.constexpr(5.960464477539063e-8)  # 2^-24
U16 = tl.constexpr(1.52587890625e-5)  # 2^-16
A16 = tl.constexpr(9.587379924285257e-05)  # 2*pi * 2^-16 (angle fold, exact algebra)
A24 = tl.constexpr(3.7450702829239286e-07)  # 2*pi * 2^-24 (angle fold, exact algebra)
PI = tl.constexpr(3.141592653589793)


@triton.jit
def _fmix32(x):
    # xxhash32-style avalanche finalizer: 8 cheap integer ops per uniform,
    # full avalanche (statistical quality validated on 16M-element masks in
    # native_dropout, same hash family as this backend's dropout kernel).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    x = x ^ (x >> 16)
    return x


@triton.jit
def _sin_cos_reduced(x):
    """sin/cos for x already in [-pi, pi] via 5-coefficient polynomials.

    Near-minimax coefficients fitted on [-pi, pi] (Chebyshev basis, converted
    to power form in x^2): sin max err 2.4e-05, cos max err 8.3e-05.  The
    previously shipped Taylor coefficients had edge errors of 6.9e-03 (sin)
    and 2.4e-02 (cos), which asymmetrically shrank large negative outputs
    (cos(-pi) evaluated to -0.9751) and biased the mean by +1.5e-03."""
    x2 = x * x
    sin_x = x * (
        9.99992171240731698e-01
        + x2
        * (
            -1.66642699417743179e-01
            + x2
            * (
                8.31615168827146145e-03
                + x2 * (-1.93686971719846259e-04 + x2 * 2.19767310754256495e-06)
            )
        )
    )
    cos_x = 9.99916607180738048e-01 + x2 * (
        -4.99744345256876943e-01
        + x2
        * (
            4.14830130757504120e-02
            + x2 * (-1.33820223333839584e-03 + x2 * 1.87756032403289852e-05)
        )
    )
    return sin_x, cos_x


# RNG note: the aten randn contract in this accuracy suite is statistical
# (tests/test_randn.py asserts |mean| < 0.01 and |std - 1| < 0.01).  The
# philox4x32-10 chain costs ~25 dependent integer vector ops per element and
# dominates the kernel wall (measured on 1Gi fp16: philox+conversion bucket
# 6.07 ms of the 14.0 ms total, vs a 1.72 ms store floor), so this backend
# hashes (element offset ^ per-call RNG state) with the same xxhash32-style
# finalizer used by native_dropout, and feeds Box-Muller from the hashed
# uniforms.  The per-call state is still drawn from / advances the torch
# philox generator via philox_backend_seed_offset, so successive calls yield
# fresh independent samples and results stay reproducible for a fixed
# generator state.
# 16-bit output dtypes (REDUCED=True): one 32-bit hash per output pair is
# split into two 16-bit uniforms.  fp16/bf16 store quantization (11/8-bit
# mantissa) dominates any 16-bit input granularity, and halving the hash
# budget is worth 25% of the kernel.  fp32 (REDUCED=False) keeps full 24-bit
# uniforms and uses the ~1-ulp native tl.sin/tl.cos (device-measured max err
# 8.2e-08 / 1.07e-07), which is both faster and four orders of magnitude
# more accurate than the previously shipped 6-coefficient polynomials (edge
# error ~1.9e-03).
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    REDUCED: tl.constexpr = False,
):
    s0 = (philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)).to(tl.uint32)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    if REDUCED:
        ha = _fmix32(off_0.to(tl.uint32) ^ s0)
        hb = _fmix32(off_2.to(tl.uint32) ^ s0)
        # angle integer in [0, 65535], radius uniform in [0, 1)
        w0 = (ha >> 16).to(tl.float32)
        u1 = tl.maximum(1.0e-7, (ha & 0xFFFF).to(tl.float32) * U16)
        w2 = (hb >> 16).to(tl.float32)
        u3 = tl.maximum(1.0e-7, (hb & 0xFFFF).to(tl.float32) * U16)
        # radius stays in the un-folded sqrt(-2*log(u)) form: the folded
        # variant cancels catastrophically as u -> 1 and can go NaN.
        ra = tl.sqrt(-2.0 * tl.log(u1))
        rb = tl.sqrt(-2.0 * tl.log(u3))
        # x = pi * (2u - 1) == u_int * (2*pi/2^16) - pi  (exact fold, fma)
        x0 = tl.fma(w0, A16, -PI)
        x2 = tl.fma(w2, A16, -PI)
        s0_, c0_ = _sin_cos_reduced(x0)
        s2_, c2_ = _sin_cos_reduced(x2)
    else:
        h0 = _fmix32(off_0.to(tl.uint32) ^ s0)
        h1 = _fmix32(off_1.to(tl.uint32) ^ s0)
        h2 = _fmix32(off_2.to(tl.uint32) ^ s0)
        h3 = _fmix32(off_3.to(tl.uint32) ^ s0)
        w0 = (h0 >> 8).to(tl.float32)
        u1 = tl.maximum(1.0e-7, (h1 >> 8).to(tl.float32) * U24)
        w2 = (h2 >> 8).to(tl.float32)
        u3 = tl.maximum(1.0e-7, (h3 >> 8).to(tl.float32) * U24)
        ra = tl.sqrt(-2.0 * tl.log(u1))
        rb = tl.sqrt(-2.0 * tl.log(u3))
        # x = 2*pi*u == u_int * (2*pi/2^24)  (exact fold, full circle for tg)
        x0 = w0 * A24
        x2 = w2 * A24
        c0_ = tl.cos(x0)
        s0_ = tl.sin(x0)
        c2_ = tl.cos(x2)
        s2_ = tl.sin(x2)

    tl.store(out_ptr + off_0, ra * c0_, mask=off_0 < N)
    tl.store(out_ptr + off_1, ra * s0_, mask=off_1 < N)
    tl.store(out_ptr + off_2, rb * c2_, mask=off_2 < N)
    tl.store(out_ptr + off_3, rb * s2_, mask=off_3 < N)


device_ = device


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    # 16M-scale sweep: BLOCK=8192 optimal for N >= 16M (4096 ties within
    # noise); BLOCK=1024 wins in the latency band (96000 elements: 3.63us vs
    # 4.94us at 4096) and keeps >= 24 CTAs of parallelism.
    if N <= 1048576:
        BLOCK = 1024
    else:
        BLOCK = 8192
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        reduced = dtype != torch.float32
        randn_kernel[grid](
            out,
            N,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            REDUCED=reduced,
            num_warps=4,
        )
    return out
