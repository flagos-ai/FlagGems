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
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

UNROLL = 4

U24 = tl.constexpr(5.960464477539063e-8)  # 2^-24
U16 = tl.constexpr(1.52587890625e-5)  # 2^-16


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


# RNG note: the aten rand_like contract in the accuracy suite is statistical
# (tests/test_rand_like.py asserts values in [0, 1]).  The philox4x32-10
# chain costs ~25 dependent integer vector ops per 4-element group and
# dominates the kernel wall on GCU400 (measured on this backend's rand/randn:
# philox bucket 4.25 of 7.0 ms at 1Gi fp16, vs a 1.74 ms store floor), so
# this backend hashes (element offset + per-call RNG state) with the same
# xxhash32-style finalizer used by native_dropout/randn/rand, and scales the
# hash to a uniform directly.  The per-call state is still drawn from /
# advances the torch philox generator via philox_backend_seed_offset, so
# successive calls yield fresh independent samples and results stay
# reproducible for a fixed generator state.
# 16-bit output dtypes (REDUCED=True): one 32-bit hash per output PAIR is
# split into two 16-bit uniforms; fp16/bf16 store quantization (11/8-bit
# mantissa) dominates the 16-bit input granularity and halving the hash
# budget is worth 25% of the kernel.  fp32 (REDUCED=False) keeps full 24-bit
# uniforms: (h >> 8) * 2^-24 matches the native torch rand float32
# granularity exactly (at::uniform_real uses a 24-bit mask), so the value
# set is the native one.
# The per-call key s0 is avalanched before use: raw, the state only differs
# additively between calls (offset advances by N/4), so successive
# same-shape calls would return the SAME multiset of samples, merely
# permuted (device-verified pathology in randn, invisible to per-call
# statistical tests).  The per-element hash is keyed by modular ADDITION,
# not XOR: with XOR the element domain [0, N) is closed under the key
# difference K = s0_call1 ^ s0_call2, so whenever K < 2^ceil(log2(N)) --
# probability N/2^32 for an avalanched key, measured 331/20000 at N=2^26,
# rising to ~25% at 1Gi -- successive calls return a permuted repeat of the
# same sample multiset (this rand_like battery caught one such pair; the
# same XOR keying ships in the rand/randn twins of this kernel).  Modular
# addition maps the domain to a shifted interval instead, which equals the
# original interval only when the keys are bit-identical, dropping the
# cross-call sample overlap to a structural zero (the same fix dropout
# ships).  Scalar prologue: ~22 ops once per program.
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "grid_x"])
def rand_like_kernel(
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
):
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    # Fold the 64-bit generator state with two independent fmix32 lanes
    # XORed together (avalanche quality identical to the single-fold form).
    s0 = _fmix32(rng_state.to(tl.uint32)) ^ _fmix32((rng_state >> 32).to(tl.uint32))
    # Grid form: this backend caps grid.x at 65535 programs.  The program
    # axis optionally splits in 2D (pid = x + y*grid_x), unlocking smaller
    # tiles for huge N.  FLAT keeps the single-program-axis form with zero
    # prologue arithmetic: the runtime grid_x multiply sits on the per-CTA
    # serial pid chain when grid_y==1 would still pay it, so hosts must pass
    # FLAT=True for 1D launches.  grid_x is do_not_specialize to avoid
    # per-shape recompiles.  Per-element sample identity is preserved: the
    # hash input is the global element offset, independent of the grid
    # decomposition, so outputs match the 1D launch bitwise.
    if FLAT:
        pid = tl.program_id(0)
    else:
        pid = tl.program_id(0) + tl.program_id(1) * grid_x
    # WIDE: without the grid.x bound the program count for a launch is no
    # longer implicitly < 2^31/BLOCK, so element offsets past the int32
    # domain (N > 2^31 - BLOCK*4) must be computed in int64 to avoid
    # wrap-around stores.  The hash input truncates to uint32, capping the
    # fresh-sample offset domain at 2^32 elements per RNG state; the host
    # arm only guards the offset arithmetic, beyond 2^32 elements samples
    # repeat.
    if WIDE:
        off_0 = pid.to(tl.int64) * (BLOCK * 4) + tl.arange(0, BLOCK)
    else:
        off_0 = pid * (BLOCK * 4) + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    if REDUCED:
        ha = _fmix32(off_0.to(tl.uint32) + s0)
        hb = _fmix32(off_2.to(tl.uint32) + s0)
        # two 16-bit uniforms per hash: [0, 1) with 2^-16 granularity,
        # max value 65535 * 2^-16 = 1 - 2^-16
        r0 = (ha >> 16).to(tl.float32) * U16
        r1 = (ha & 0xFFFF).to(tl.float32) * U16
        r2 = (hb >> 16).to(tl.float32) * U16
        r3 = (hb & 0xFFFF).to(tl.float32) * U16
        # [0, 1) half-open fix for 16-bit outputs: the 2^-16-grid maximum
        # 1 - 2^-16 rounds UP to 1.0 in the fp16/bf16 store cast (values
        # >= (1.0 + 0.99951171875)/2 for fp16,
        # >= (1.0 + 0.99609375)/2 for bf16; ~2^-12 of the mass), violating
        # the uniform_ half-open contract (native torch rand never yields
        # 1.0).  Clamp in the fp32 domain to the predecessor of 1.0 in the
        # output dtype (same hi as uniform_ applies).
        OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
        if OUT_DT == tl.float16:
            hi = 0.99951171875
        else:
            hi = 0.99609375
        r0 = tl.minimum(r0, hi)
        r1 = tl.minimum(r1, hi)
        r2 = tl.minimum(r2, hi)
        r3 = tl.minimum(r3, hi)
    else:
        r0 = (_fmix32(off_0.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24
        r1 = (_fmix32(off_1.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24
        r2 = (_fmix32(off_2.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24
        r3 = (_fmix32(off_3.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24

    if MASKED:
        tl.store(out_ptr + off_0, r0, mask=off_0 < N)
        tl.store(out_ptr + off_1, r1, mask=off_1 < N)
        tl.store(out_ptr + off_2, r2, mask=off_2 < N)
        tl.store(out_ptr + off_3, r3, mask=off_3 < N)
    else:
        tl.store(out_ptr + off_0, r0)
        tl.store(out_ptr + off_1, r1)
        tl.store(out_ptr + off_2, r2)
        tl.store(out_ptr + off_3, r3)


# Persistent (grid-stride) form for large 16-bit launches: CTA dispatch on
# GCU400 costs ~52ns/program, so a naive 1Gi fp16 launch at B16384 pays
# ~850us in dispatch alone.  A fixed 192-program grid-stride loop (the CU
# count, keeping every program busy concurrently) reclaims it: 1Gi fp16
# 2757 -> 1802 us, 3.8% above the 1736 us L0 store floor.  The hash input
# stays the GLOBAL element offset, so samples are bitwise identical to the
# naive grid at the same generator state; only the tiling changes.  B8192
# in the loop keeps the live set within the envelope that B16384 loop form
# spills (13452 us pathology, measured on the rand twin of this kernel).
# EVEN_STEP drops the per-iteration scalar full/masked branch when N is an
# exact multiple of the stride; otherwise full tiles store unmasked and
# only the tail tile of each program stores masked.
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_like_kernel_persist(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    NUM_PROG: tl.constexpr,
    REDUCED: tl.constexpr,
    WIDE: tl.constexpr,
    EVEN_STEP: tl.constexpr,
):
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32(rng_state.to(tl.uint32)) ^ _fmix32((rng_state >> 32).to(tl.uint32))
    lane = tl.arange(0, BLOCK)
    step = NUM_PROG * BLOCK * 4
    if WIDE:
        start = tl.program_id(0).to(tl.int64) * (BLOCK * 4)
    else:
        start = tl.program_id(0) * (BLOCK * 4)
    while start < N:
        if WIDE:
            off_0 = start + lane.to(tl.int64)
        else:
            off_0 = start + lane
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        if REDUCED:
            ha = _fmix32(off_0.to(tl.uint32) + s0)
            hb = _fmix32(off_2.to(tl.uint32) + s0)
            r0 = (ha >> 16).to(tl.float32) * U16
            r1 = (ha & 0xFFFF).to(tl.float32) * U16
            r2 = (hb >> 16).to(tl.float32) * U16
            r3 = (hb & 0xFFFF).to(tl.float32) * U16
            OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
            if OUT_DT == tl.float16:
                hi = 0.99951171875
            else:
                hi = 0.99609375
            r0 = tl.minimum(r0, hi)
            r1 = tl.minimum(r1, hi)
            r2 = tl.minimum(r2, hi)
            r3 = tl.minimum(r3, hi)
        else:
            r0 = (_fmix32(off_0.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24
            r1 = (_fmix32(off_1.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24
            r2 = (_fmix32(off_2.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24
            r3 = (_fmix32(off_3.to(tl.uint32) + s0) >> 8).to(tl.float32) * U24
        if EVEN_STEP:
            tl.store(out_ptr + off_0, r0)
            tl.store(out_ptr + off_1, r1)
            tl.store(out_ptr + off_2, r2)
            tl.store(out_ptr + off_3, r3)
        else:
            if start + BLOCK * 4 <= N:
                tl.store(out_ptr + off_0, r0)
                tl.store(out_ptr + off_1, r1)
                tl.store(out_ptr + off_2, r2)
                tl.store(out_ptr + off_3, r3)
            else:
                tl.store(out_ptr + off_0, r0, mask=off_0 < N)
                tl.store(out_ptr + off_1, r1, mask=off_1 < N)
                tl.store(out_ptr + off_2, r2, mask=off_2 < N)
                tl.store(out_ptr + off_3, r3, mask=off_3 < N)
        start += step


# 16-bit persistent threshold: measured crossover between the naive grid and
# the grid-stride loop (do_bench median, GCU400, on the byte-identical rand
# twin of this kernel body).  At 8Mi the naive B16384 grid still wins
# (25.6 vs 28.8 us); from 16.7M on the persistent form wins (43.3 vs 46.5 us)
# and the gap widens with N (64Mi: 129 vs 175 us, 1Gi: 1802 vs 2757 us).
# fp32 gains nothing from the loop (1Gi: 2622-2632 both forms) and keeps the
# naive grid at every size.
PERSIST_MIN_NUMEL = 1 << 24
PERSIST_BLOCK = 8192
PERSIST_NUM_PROG = 192


def rand_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ENFLAME RAND_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    reduced = dtype in (torch.float16, torch.bfloat16)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    # BLOCK/num_warps ladder (do_bench median rescan on the rand twin of
    # this kernel; nw2 wins every fp32 band, 16-bit bands want B8192-16384,
    # tiny shapes want B512/nw2):
    #   4096 elems fp32: B512/nw2 2.43 vs B1024/nw4 3.04 us
    #   96000 fp32:      B1024/nw2 3.09 vs B512/nw2 3.63 us
    #   2^20 fp32:       B8192/nw2 7.29 vs B2048/nw2 9.59 us
    #   16.7M fp32:      B16384/nw2 47.8 vs B8192/nw4 62.4 us
    #   1Gi fp32:        B32768/nw2 2632 vs B16384/nw4 3788 us
    #   2^20 fp16:       B8192/nw4 7.04 vs B2048/nw4 8.96 us
    #   8Mi fp16:        B16384/nw4 25.6 us
    #   1Gi fp16 naive:  B16384/nw4 2757 vs B8192/nw4 3650 us
    with torch_device_fn.device(x.device):
        if reduced and N >= PERSIST_MIN_NUMEL:
            grid = (min(PERSIST_NUM_PROG, triton.cdiv(N, PERSIST_BLOCK * UNROLL)),)
            rand_like_kernel_persist[grid](
                out,
                N,
                philox_seed,
                philox_offset,
                BLOCK=PERSIST_BLOCK,
                NUM_PROG=PERSIST_NUM_PROG,
                REDUCED=True,
                WIDE=N > 2**31 - PERSIST_BLOCK * UNROLL,
                EVEN_STEP=N % (PERSIST_NUM_PROG * PERSIST_BLOCK * UNROLL) == 0,
                num_warps=4,
            )
            return out

        if N <= 32768:
            BLOCK, nw = 512, 2
        elif N <= 131072:
            BLOCK, nw = 1024, 2
        elif reduced:
            BLOCK, nw = (8192, 4) if N <= 1048576 else (16384, 4)
        else:
            if N <= 16777216:
                BLOCK, nw = 8192, 2
            elif N < 268435456:
                BLOCK, nw = 16384, 2
            else:
                BLOCK, nw = 32768, 2
        total = triton.cdiv(N, BLOCK * UNROLL)
        if total <= 65535:
            grid = (total,)
            grid_x, flat = 0, True
        else:
            grid_y = triton.cdiv(total, 32768)
            grid_x = triton.cdiv(total, grid_y)
            grid = (grid_x, grid_y)
            flat = False
        # EVEN specialization: when N is an exact multiple of the per-program
        # tile the four bound compares per store group are dead weight
        # (masked stores also degrade to element-wise writes on 16-bit
        # dtypes on GCU); the masked kernel covers every other N.
        rand_like_kernel[grid](
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
            num_warps=nw,
        )
    return out
