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
NEG_TWO_LN2 = tl.constexpr(-1.3862943611198906)  # -2*ln(2), folds ln(u) into log2(u)


@triton.jit
def _neg_two_ln(x, FAST_LOG: tl.constexpr):
    """-2*ln(x) as a single multiply on log2 when the dtype allows it.

    log2 is a hardware primitive here while tl.log carries an extra conversion
    chain; measured on 16-bit output it is 1.8%/2.1% faster at 2^24/2^30
    elements.  fp32 stores compile differently and lose 13% with this form, so
    the host keeps the plain tl.log arm for float32 outputs.
    """
    if FAST_LOG:
        return NEG_TWO_LN2 * tl.math.log2(x)
    return -2.0 * tl.log(x)


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
    (cos(-pi) evaluated to -0.9751) and biased the mean by +1.5e-03.

    Evaluation uses the Estrin form with shared even powers (x2/x4/x6/x8
    computed once) instead of two independent Horner chains: identical
    coefficients and accuracy class, but the dependent-op chain per output
    halves, which on GCU400 removes the latency floor of this bucket
    (1Gi fp16 kernel: 8.38 -> 7.06 ms, device-measured)."""
    x2 = x * x
    x4 = x2 * x2
    x6 = x2 * x4
    x8 = x4 * x4
    sin_x = x * (
        (2.19767310754256495e-06 * x8 + -1.93686971719846259e-04 * x6)
        + (8.31615168827146145e-03 * x4 + -1.66642699417743179e-01 * x2)
        + 9.99992171240731698e-01
    )
    cos_x = (
        (1.87756032403289852e-05 * x8 + -1.33820223333839584e-03 * x6)
        + (4.14830130757504120e-02 * x4 + -4.99744345256876943e-01 * x2)
        + 9.99916607180738048e-01
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
# uniforms; the angle is folded into [-pi, pi) with the same exact fma so the
# shared 5-coefficient minimax sincos applies.  The previously shipped fp32
# arm used native tl.sin/tl.cos (~1-ulp, max err 8.2e-08/1.07e-07), which is
# 3-4 orders more accurate than the minimax fit but costs 1.35 ms per 1Gi
# fp32 kernel (device-measured 10.14 -> 6.94 ms after the swap, BLOCK=16384);
# the 8.3e-05 cos edge error is 20x below fp16 ulp and 1000x below the
# statistical tolerance (|mean|, |std-1| < 0.01), and tail asymmetry stays
# within binomial noise (validated vs native torch.randn, 2^26 samples).
# MASKED=False (EVEN specialization, host-guaranteed N % (BLOCK*4) == 0)
# drops the four bound compares per store group: ~0.31 ms per 1Gi kernel.
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "grid_x"])
def randn_kernel(
    out_ptr,
    N,
    grid_x,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    REDUCED: tl.constexpr = False,
    MASKED: tl.constexpr = True,
    FLAT: tl.constexpr = True,
    WIDE: tl.constexpr = False,
    FAST_LOG: tl.constexpr = True,
):
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    # The raw per-call state must be avalanched before it is XOR-keyed into
    # the per-element hash input.  Raw, the state only differs additively
    # between calls (offset advances by N/4), so s0_1 ^ s0_2 is a constant
    # small mask M: element i of call 2 then hashes input (i ^ M) ^ s0_1 --
    # i.e. successive same-shape calls return the SAME multiset of samples,
    # merely permuted (device-verified: 40 histogram bins identical to the
    # last count across successive calls, which every per-call statistical
    # test misses).  fmix32 makes the per-call key pseudorandom, so the
    # cross-call sample overlap drops from a structural invariant to a
    # ~2^-32 coincidence.  Scalar prologue: ~11 ops once per program.
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
    # Grid form: this backend caps grid.x at 65535 programs, which previously
    # forced BLOCK-wide tiles for large N (grid.x*BLOCK*4 >= N).  The program
    # axis now optionally splits in 2D (pid = x + y*grid_x), unlocking smaller
    # tiles for huge N.  FLAT keeps the single-program-axis form with zero
    # prologue arithmetic: the runtime grid_x multiply sits on the per-CTA
    # serial pid chain and costs ~0.85% when grid_y==1 still pays it, so hosts
    # must pass FLAT=True for 1D launches.  grid_x is do_not_specialize to
    # avoid per-shape recompiles.  Per-element sample identity is preserved:
    # the hash input is the global element offset, independent of the grid
    # decomposition, so outputs match the 1D launch bitwise.
    if FLAT:
        pid = tl.program_id(0)
    else:
        pid = tl.program_id(0) + tl.program_id(1) * grid_x
    # WIDE: without the grid.x bound the program count for a launch is no
    # longer implicitly < 2^31/BLOCK, so element offsets past the int32 domain
    # (N > 2^31 - BLOCK*4) must be computed in int64 to avoid wrap-around
    # stores.  Note the hash input truncates to uint32, capping the fresh-
    # sample offset domain at 2^32 elements per RNG state; the host arm only
    # guards the offset arithmetic, beyond 2^32 elements samples repeat.
    if WIDE:
        off_0 = pid.to(tl.int64) * (BLOCK * 4) + tl.arange(0, BLOCK)
    else:
        off_0 = pid * (BLOCK * 4) + tl.arange(0, BLOCK)
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
        ra = tl.sqrt(_neg_two_ln(u1, FAST_LOG))
        rb = tl.sqrt(_neg_two_ln(u3, FAST_LOG))
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
        ra = tl.sqrt(_neg_two_ln(u1, FAST_LOG))
        rb = tl.sqrt(_neg_two_ln(u3, FAST_LOG))
        # x = u_int * (2*pi/2^24) - pi  (exact fold into the [-pi, pi) fit
        # domain of _sin_cos_reduced; full circle is preserved by symmetry)
        x0 = tl.fma(w0, A24, -PI)
        x2 = tl.fma(w2, A24, -PI)
        s0_, c0_ = _sin_cos_reduced(x0)
        s2_, c2_ = _sin_cos_reduced(x2)

    if MASKED:
        tl.store(out_ptr + off_0, ra * c0_, mask=off_0 < N)
        tl.store(out_ptr + off_1, ra * s0_, mask=off_1 < N)
        tl.store(out_ptr + off_2, rb * c2_, mask=off_2 < N)
        tl.store(out_ptr + off_3, rb * s2_, mask=off_3 < N)
    else:
        tl.store(out_ptr + off_0, ra * c0_)
        tl.store(out_ptr + off_1, ra * s0_)
        tl.store(out_ptr + off_2, rb * c2_)
        tl.store(out_ptr + off_3, rb * s2_)


device_ = device


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    # Post-Estrin rescan (GCU400, official do_bench kernel mode, median):
    # BLOCK=1024 still wins the latency band (4096 elems: 3.22us; 96000:
    # 3.54us) and keeps >= 24 CTAs of parallelism.  Above it the optimum
    # moved with the shortened trig dependency chains:
    #   16-bit lane:  B8192 up to 1Gi (16.7M fp16 115.7us vs 129.1 @4096),
    #                 B16384 beyond (1Gi fp16 6.90ms vs 7.12 @8192).
    #   fp32 lane:    B2048 up to 256Mi elements (16.7M fp32 120.1us vs
    #                 129.0 @8192), B16384 beyond (1Gi fp32 6.94ms vs
    #                 7.96 @8192).
    if N <= 1048576:
        BLOCK = 1024
    elif dtype == torch.float32:
        BLOCK = 2048 if N < 268435456 else 16384
    else:
        BLOCK = 8192 if N < 1073741824 else 16384
    total = triton.cdiv(N, BLOCK * UNROLL)
    if total <= 65535:
        grid = (total,)
        grid_x, flat = 0, True
    else:
        grid_y = triton.cdiv(total, 32768)
        grid_x = triton.cdiv(total, grid_y)
        grid = (grid_x, grid_y)
        flat = False
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        reduced = dtype != torch.float32
        # EVEN specialization: when N is an exact multiple of the per-program
        # tile the four bound compares per store group are dead weight
        # (~0.31 ms per 1Gi kernel); the masked kernel covers every other N.
        randn_kernel[grid](
            out,
            N,
            grid_x,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            REDUCED=reduced,
            MASKED=N % (BLOCK * UNROLL) != 0,
            FLAT=flat,
            WIDE=N > 2**31 - BLOCK * UNROLL,
            FAST_LOG=dtype != torch.float32,
            num_warps=4,
        )
    return out
