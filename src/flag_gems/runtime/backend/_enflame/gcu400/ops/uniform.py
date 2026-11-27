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

import functools
import logging
import math

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

UNROLL = 4

U24 = tl.constexpr(5.960464477539063e-8)  # 2^-24
U16 = tl.constexpr(1.52587890625e-5)  # 2^-16

# Persistent (grid-stride) path parameters: a fixed number of programs loops over the entire range.
# 2026-11-17 RSI resweep (after the fmix32 RNG + integer lattice clamp structure changes, probe sweep_uniform.py,
# GCU400): B=2048/P=96 beats the earlier philox-era tuned B=4096/P=192 across the board — at the 16.7M size
# fp32 62.9→47.6us (-24.5%), fp16 43.8→35.3us (-19.4%); at the 1Gi size fp32 2951→
# 2707us (-8.3%), fp16 1973→1963us. EVEN (no masked store when N divides BLOCK*4)
# gains another -0.6% at 1Gi fp32. Small shapes still use the naive grid.
PERSIST_BLOCK = 2048
PERSIST_NUM_PROG = 96
PERSIST_NUM_WARPS = 4
# Threshold measurement: at 4M elements the naive grid is faster (34.6us vs 38.5us); from 16.7M on, persistent is faster
# (109.5us vs 122.9us), so 8M is chosen as the cutoff.
PERSIST_MIN_NUMEL = 1 << 23


@triton.jit
def _next_below(x, DT: tl.constexpr):
    """Largest ``DT`` value strictly less than ``x``.

    Used to build the exclusive upper bound of the sampling interval in the
    *output* dtype: ``uniform`` samples from ``[from, to)``, so a value that
    rounds up to ``to`` in a low-precision dtype must be stepped back one ulp.

    A zero input needs care: the bit patterns of ``+0.0`` and ``-0.0`` are the
    bottom of the positive block and the top of the negative block, so the
    usual "-1 for x >= 0, +1 otherwise" step walks *out* of the number line and
    wraps around -- ``+0.0`` (0) minus one is all-ones, and ``-0.0``
    (0x8000...0) plus one is all-ones too, both of which are NaN. The true
    predecessor of either zero is ``-smallest_subnormal``, so a zero input is
    answered directly as the negation of the smallest positive subnormal.
    """
    zero = x == 0
    if DT == tl.float32:
        bits = x.to(tl.int32, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int32)
        stepped = (bits + step).to(tl.float32, bitcast=True)
        # smallest positive subnormal: the bit pattern with a lone low bit.
        tiny = tl.full((), 1, tl.int32).to(tl.float32, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float32)
    elif DT == tl.float64:
        bits = x.to(tl.int64, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int64)
        stepped = (bits + step).to(tl.float64, bitcast=True)
        tiny = tl.full((), 1, tl.int64).to(tl.float64, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float64)
    else:
        bits = x.to(tl.int16, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int16)
        stepped = (bits + step).to(DT, bitcast=True)
        tiny = tl.full((), 1, tl.int16).to(DT, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(DT)


@triton.jit
def _fmix32(x):
    # xxhash32-style avalanche finalizer: 8 cheap integer ops per uniform,
    # full avalanche (statistical quality validated on 16M-element masks in
    # native_dropout; same hash family as this backend's dropout/randn kernels).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    x = x ^ (x >> 16)
    return x


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "N"])
def uniform_kernel_gcu400(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    scale = to - from_
    r0 = uint_to_uniform_float(r0) * scale + from_
    r1 = uint_to_uniform_float(r1) * scale + from_
    r2 = uint_to_uniform_float(r2) * scale + from_
    r3 = uint_to_uniform_float(r3) * scale + from_
    # Samples are computed in float32; rounding into a low-precision dtype can land exactly on the exclusive upper bound
    # ``to`` (especially fp16 / bf16), so clamp to ``to``'s
    # predecessor ``hi`` in the output dtype. Clamping is done entirely in the float32 domain: ``hi`` is exactly
    # representable in OUT_DT and
    # strictly less than ``to``, so take min in float32 then do the single fp32->OUT_DT conversion, keeping
    # ``[from, to)`` semantics and avoiding an extra per-element dtype conversion on 16-bit dtypes
    # (measured as the main bottleneck for fp16/bf16 large shapes).
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = to.to(OUT_DT)
    from_dt = from_.to(OUT_DT)
    hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_dt, hi, to_dt)
    if OUT_DT == tl.float64:
        hi_wide = hi.to(tl.float64)
        r0 = tl.minimum(r0.to(tl.float64), hi_wide).to(OUT_DT)
        r1 = tl.minimum(r1.to(tl.float64), hi_wide).to(OUT_DT)
        r2 = tl.minimum(r2.to(tl.float64), hi_wide).to(OUT_DT)
        r3 = tl.minimum(r3.to(tl.float64), hi_wide).to(OUT_DT)
    else:
        hi_narrow = hi.to(tl.float32)
        r0 = tl.minimum(r0, hi_narrow).to(OUT_DT)
        r1 = tl.minimum(r1, hi_narrow).to(OUT_DT)
        r2 = tl.minimum(r2, hi_narrow).to(OUT_DT)
        r3 = tl.minimum(r3, hi_narrow).to(OUT_DT)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    # When N is divisible by BLOCK*4, use an unmasked direct store: on GCU a 16-bit
    # masked store degrades to per-element writes, measured ~6% slower than unmasked.
    if EVEN:
        tl.store(out_ptr + off_0, r0)
        tl.store(out_ptr + off_1, r1)
        tl.store(out_ptr + off_2, r2)
        tl.store(out_ptr + off_3, r3)
    else:
        tl.store(out_ptr + off_0, r0, mask=off_0 < N)
        tl.store(out_ptr + off_1, r1, mask=off_1 < N)
        tl.store(out_ptr + off_2, r2, mask=off_2 < N)
        tl.store(out_ptr + off_3, r3, mask=off_3 < N)


# RNG note: the aten::uniform_ accuracy contract in this suite is statistical
# (tests/test_uniform_.py asserts the sampled values stay inside [from, to)).
# The philox4x32-10 chain costs ~15 dependent integer vector ops per element
# and dominates the kernel wall (1Gi fp16: philox+conversion bucket ≈ 5.3ms of
# the 6.07ms total vs a 0.72ms store floor), so the hash kernel below hashes
# the global element offset with the same xxhash32-style finalizer used by
# this backend's randn / native_dropout kernels. The per-call state is still
# drawn from / advances the torch philox generator via
# philox_backend_seed_offset (byte-identical increment accounting), so
# successive calls yield fresh independent samples and stay reproducible for
# a fixed generator state; the raw per-call state is avalanched into s0 first,
# otherwise two same-shape calls would return the same multiset of samples,
# merely permuted (randn precedent, device-verified). 16-bit output dtypes
# (REDUCED=True) split one 32-bit hash into a hi/lo pair of 16-bit uniforms,
# halving the hash budget; fp16/bf16 store quantization dominates any 16-bit
# input granularity. fp32 keeps one 24-bit uniform per element, matching the
# granularity of aten's CUDA uniform sampler for float. Sampling stays in
# [from, to): the integer-grid clamp described below keeps every sample
# <= hi (the output-dtype predecessor of ``to``), so the monotone final
# rounding store can never reach ``to``.
# The hash input truncates to uint32, capping the fresh-sample offset domain
# at 2^32 elements per RNG state; the philox kernel had the same per-counter
# domain. fp64 output keeps the philox kernel above: it has no fp32-domain
# clamp path there and fp64 arithmetic is unsupported on this device anyway.
#
# [from, to) exclusive upper bound, integer-grid edition: the host precomputes
# w_top = floor((safe_to - from32) / s_u), the largest sampling-grid index
# whose fp32 sample stays <= hi (the output-dtype predecessor of ``to``),
# with safe_to = hi, or midpoint(hi, to_dt) when hi < from32 (intervals
# narrower than half an output ulp; keeps the rounding-conversion result
# <= hi there as well). Clamping the integer index (instead of min-ing the
# fp32 sample against hi_narrow, as the philox kernel does) buys two things:
# 1) for common fp32 intervals (w_top == full grid, e.g. [0, 1) / [-3, 3))
#    NEED_CLAMP=False drops the per-element clamp entirely, and the kernel
#    prologue no longer computes the _next_below chain;
# 2) the resulting marginal matches the exact sampling-grid expectation to
#    the bin level (2^24-sample probe vs enumerated-grid model, all bins
#    within 4 sigma). Note the top value hi legitimately carries ~1.5x the
#    mass of an interior value (its rounding window [mid(hi, prev), to) is
#    1.5 ulp wide vs 1 ulp) -- that pile-up is the correct marginal of
#    round(U[from, to)), not a defect; GCU-native 16-bit samplers are
#    coarsely quantized (every-other-value 2x) and are not the reference.
# Exclusivity proof: v = RN32(w * s_u + from32) <= RN32(w_top * s_u + from32)
# <= hi (w_top chosen with fp64-exact arithmetic, hi representable), and the
# monotone fp32->OUT_DT rounding of a value <= hi stays <= hi < to.
@triton.jit
def _uniform_hash_tile(
    out_ptr,
    base,
    N,
    s0,
    s_u,
    from_,
    w_top,
    BLOCK: tl.constexpr,
    REDUCED: tl.constexpr,
    EVEN: tl.constexpr,
    NEED_CLAMP: tl.constexpr,
):
    off_1 = base + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    if REDUCED:
        # one hash feeds a pair of BLOCK lanes (groups 0/1 and 2/3)
        ha = _fmix32(base.to(tl.uint32) ^ s0)
        hb = _fmix32(off_2.to(tl.uint32) ^ s0)
        u0 = ha >> 16
        u1 = ha & 0xFFFF
        u2 = hb >> 16
        u3 = hb & 0xFFFF
    else:
        u0 = _fmix32(base.to(tl.uint32) ^ s0) >> 8
        u1 = _fmix32(off_1.to(tl.uint32) ^ s0) >> 8
        u2 = _fmix32(off_2.to(tl.uint32) ^ s0) >> 8
        u3 = _fmix32(off_3.to(tl.uint32) ^ s0) >> 8
    if NEED_CLAMP:
        w = w_top.to(tl.uint32)
        u0 = tl.minimum(u0, w)
        u1 = tl.minimum(u1, w)
        u2 = tl.minimum(u2, w)
        u3 = tl.minimum(u3, w)
    # The sampling range is folded into s_u = U_k * scale (one scalar multiply in the prologue, replacing the
    # per-element
    # double multiply of ``u * U_k`` then ``* scale``).
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    r0 = tl.fma(u0.to(tl.float32), s_u, from_).to(OUT_DT)
    r1 = tl.fma(u1.to(tl.float32), s_u, from_).to(OUT_DT)
    r2 = tl.fma(u2.to(tl.float32), s_u, from_).to(OUT_DT)
    r3 = tl.fma(u3.to(tl.float32), s_u, from_).to(OUT_DT)
    # When N is divisible by BLOCK*4, use an unmasked direct store: on GCU, masked stores of 16-bit dtypes
    # degrade to element-wise writes, measured about 6% slower than unmasked.
    if EVEN:
        tl.store(out_ptr + base, r0)
        tl.store(out_ptr + off_1, r1)
        tl.store(out_ptr + off_2, r2)
        tl.store(out_ptr + off_3, r3)
    else:
        tl.store(out_ptr + base, r0, mask=base < N)
        tl.store(out_ptr + off_1, r1, mask=off_1 < N)
        tl.store(out_ptr + off_2, r2, mask=off_2 < N)
        tl.store(out_ptr + off_3, r3, mask=off_3 < N)


@libentry()
@triton.jit(
    do_not_specialize=["philox_seed", "philox_offset", "from_", "to", "N", "w_top"]
)
def uniform_kernel_hash_gcu400(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    w_top,
    BLOCK: tl.constexpr,
    REDUCED: tl.constexpr,
    EVEN: tl.constexpr,
    PERSIST: tl.constexpr,
    NUM_PROG: tl.constexpr,
    NEED_CLAMP: tl.constexpr,
):
    """fmix32 hashed RNG version: one program handles 4*BLOCK elements.

    With PERSIST=False each program handles one block (plain grid); with PERSIST=True a fixed
    NUM_PROG programs grid-stride over the whole range. Each element is sampled independently
    by hashing its global offset, so values are bit-identical across both grid forms.
    """
    # Same per-call state avalanche as randn: prevents two consecutive calls with the same shape from returning the same
    # multiset of samples (only rearranged).
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
    scale = to - from_
    if REDUCED:
        s_u = scale * U16
    else:
        s_u = scale * U24
    lane = tl.arange(0, BLOCK)
    if PERSIST:
        step = NUM_PROG * BLOCK * 4
        start = tl.program_id(0) * BLOCK * 4
        while start < N:
            _uniform_hash_tile(
                out_ptr,
                start + lane,
                N,
                s0,
                s_u,
                from_,
                w_top,
                BLOCK,
                REDUCED,
                EVEN,
                NEED_CLAMP,
            )
            start += step
    else:
        _uniform_hash_tile(
            out_ptr,
            tl.program_id(0) * BLOCK * 4 + lane,
            N,
            s0,
            s_u,
            from_,
            w_top,
            BLOCK,
            REDUCED,
            EVEN,
            NEED_CLAMP,
        )


_GRID_BITS = {torch.float32: 24, torch.float16: 16, torch.bfloat16: 16}


@functools.lru_cache(maxsize=128)
def _uniform_w_top(from_, to, dtype):
    """Exclusive upper-bound index of the sampled lattice domain (host scalar side, exact fp64).

    Returns (w_top, need_clamp), where w_top is the largest lattice index w such that the
    in-kernel fp32 sample RN32(w * s_u + from32) <= hi (hi being the output-dtype predecessor of
    ``to``). When the interval is narrower than half an output ulp and hi < from32, the midpoint
    of hi and to_dt is used as a safe upper bound (rounded conversion still <= hi); if that
    midpoint is also <= from32, lattice point 0 rounds to to_dt -- the output dtype has no
    representable value in [from, to), an inherently unsatisfiable corner (the philox version
    clamps to hi with min, violating the lower bound), handled uniformly as a constant fill and
    noted in the kernel comments. Degenerate intervals (to_dt <= from_dt, including from == to)
    return (0, True) = constant fill of the rounded from32, matching the philox version.
    """
    k = _GRID_BITS[dtype]
    grid_max = (1 << k) - 1
    from32 = float(torch.tensor(from_, dtype=torch.float32))
    to32 = float(torch.tensor(to, dtype=torch.float32))
    scale32 = float(torch.tensor(to32 - from32, dtype=torch.float32))
    s_u = scale32 * (2.0**-k)
    if not (math.isfinite(scale32) and s_u > 0.0 and math.isfinite(from32)):
        # Non-finite boundary / underflow interval: filled with a constant (pathological inputs, also undefined in the
        # philox version).
        return 0, True
    to_dt_t = torch.tensor(to, dtype=dtype)
    from_dt_t = torch.tensor(from_, dtype=dtype)
    if not (float(to_dt_t) > float(from_dt_t)):
        return 0, True
    if float(to_dt_t) == 0.0:
        # Subtracting one bit from +0.0/-0.0 steps off the number line: the predecessor takes -smallest subnormal
        # (consistent with
        # the zero-value branch of kernel `_next_below`).
        hi = -float(torch.finfo(dtype).tiny)
    else:
        bits = to_dt_t.view(torch.int16 if k == 16 else torch.int32)
        hi = float((bits - 1).view(dtype))
    safe_to = hi
    if hi < from32:
        safe_to = (hi + float(to_dt_t)) / 2
    w_top = math.floor((safe_to - from32) / s_u)
    w_top = max(0, min(int(w_top), grid_max))
    return w_top, w_top < grid_max


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logger.debug("GEMS_ENFLAME UNIFORM_")
    # aten::uniform_ rejects reversed intervals; the in-place version previously lacked the same check.
    if from_ > to:
        raise RuntimeError(
            "uniform_ expects to return a [from, to) range, but found "
            f"from={from_:g} > to={to:g}"
        )
    # The kernel writes back by flat contiguous indices; non-contiguous views write a contiguous copy then copy back by
    # the original layout.
    if not self.is_contiguous():
        contiguous_out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        uniform_(contiguous_out, from_, to, generator=generator)
        self.copy_(contiguous_out)
        return self
    N = volume(self.shape)
    dtype = self.dtype

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        if dtype == torch.float64:
            # fp64 output has no fp32-domain clamp path, so it continues through the philox kernel.
            if N <= 16384:
                BLOCK = 256
            elif N <= 65536:
                BLOCK = 1024
            else:
                BLOCK = 16384
            grid = (triton.cdiv(N, BLOCK * UNROLL),)
            uniform_kernel_gcu400[grid](
                self,
                N,
                philox_seed,
                philox_offset,
                from_,
                to,
                BLOCK=BLOCK,
                EVEN=N % (BLOCK * UNROLL) == 0,
                num_warps=4,
                num_stages=1,
            )
            return self

        w_top, need_clamp = _uniform_w_top(from_, to, dtype)
        if N >= PERSIST_MIN_NUMEL:
            # Large shapes use the persistent grid-stride path: a fixed number of programs avoids an overly large grid.
            # When N is divisible by BLOCK*4, the tile stays in bounds throughout, skipping masked store.
            uniform_kernel_hash_gcu400[(PERSIST_NUM_PROG,)](
                self,
                N,
                philox_seed,
                philox_offset,
                from_,
                to,
                w_top,
                BLOCK=PERSIST_BLOCK,
                REDUCED=dtype != torch.float32,
                EVEN=N % (PERSIST_BLOCK * UNROLL) == 0,
                PERSIST=True,
                NUM_PROG=PERSIST_NUM_PROG,
                NEED_CLAMP=need_clamp,
                num_warps=PERSIST_NUM_WARPS,
                num_stages=1,
            )
            return self

        if N <= 4096:
            BLOCK = 64
        elif N <= 16384:
            BLOCK = 256
        elif N <= 65536:
            BLOCK = 1024
        elif dtype == torch.float32:
            BLOCK = 32768
        else:
            BLOCK = 16384

        grid = (triton.cdiv(N, BLOCK * UNROLL),)
        # With small tiles (BLOCK=64), fp32 is slightly better with nw=4 (2.8 vs 2.9us, probe sweep).
        nw = 2 if (dtype == torch.float32 and N > 4096) else 4
        uniform_kernel_hash_gcu400[grid](
            self,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            w_top,
            BLOCK=BLOCK,
            REDUCED=dtype != torch.float32,
            EVEN=N % (BLOCK * UNROLL) == 0,
            PERSIST=False,
            NUM_PROG=1,
            NEED_CLAMP=need_clamp,
            num_warps=nw,
            num_stages=1,
        )
    return self
