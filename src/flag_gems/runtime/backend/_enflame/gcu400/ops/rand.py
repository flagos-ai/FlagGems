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


# RNG note: the aten rand contract in the accuracy suite is statistical
# (tests/test_rand.py asserts values in [0, 1)).  The philox4x32-10 chain
# costs ~25 dependent integer vector ops per 4-element group and dominates
# the kernel wall on GCU400 (measured on randn: philox+conversion bucket
# 6.07 ms of the 14.0 ms total at 1Gi fp16, vs a 1.72 ms store floor), so
# this backend hashes (element offset ^ per-call RNG state) with the same
# xxhash32-style finalizer used by native_dropout/randn, and scales the
# hash to a uniform directly.  The per-call state is still drawn from /
# advances the torch philox generator via philox_backend_seed_offset, so
# successive calls yield fresh independent samples and results stay
# reproducible for a fixed generator state.
# 16-bit output dtypes (REDUCED=True): one 32-bit hash per output PAIR is
# split into two 16-bit uniforms; fp16/bf16 store quantization (11/8-bit
# mantissa) dominates the 16-bit input granularity and halving the hash
# budget is worth 25% of the kernel (same trade as randn).  fp32
# (REDUCED=False) keeps full 24-bit uniforms: (h >> 8) * 2^-24 matches the
# native torch rand float32 granularity exactly (at::uniform_real uses a
# 24-bit mask), so the value set is the native one.
# The per-call key s0 is avalanched before use: raw, the state only differs
# additively between calls (offset advances by N/4), so successive
# same-shape calls would return the SAME multiset of samples, merely
# permuted (device-verified pathology in randn, invisible to per-call
# statistical tests).  fmix32 makes the per-call key pseudorandom and drops
# the cross-call sample overlap from a structural invariant to a ~2^-32
# coincidence.  Scalar prologue: ~11 ops once per program.
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "grid_x"])
def rand_kernel(
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
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
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
        ha = _fmix32(off_0.to(tl.uint32) ^ s0)
        hb = _fmix32(off_2.to(tl.uint32) ^ s0)
        # two 16-bit uniforms per hash: [0, 1) with 2^-16 granularity,
        # max value 65535 * 2^-16 = 1 - 2^-16
        r0 = (ha >> 16).to(tl.float32) * U16
        r1 = (ha & 0xFFFF).to(tl.float32) * U16
        r2 = (hb >> 16).to(tl.float32) * U16
        r3 = (hb & 0xFFFF).to(tl.float32) * U16
        # The fp32 store cast rounds the top of the domain UP to 1.0 in the
        # 16-bit output dtypes (fp16: values >= (1.0 + 0.99951171875)/2,
        # bf16: >= (1.0 + 0.99609375)/2; ~2^-12 of the mass), violating the
        # half-open [0, 1) contract.  Clamp in the fp32 domain to the
        # predecessor of 1.0 in the output dtype (exactly representable in
        # fp32), the same fix this backend's uniform_ kernel applies.
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
        r0 = (_fmix32(off_0.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24
        r1 = (_fmix32(off_1.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24
        r2 = (_fmix32(off_2.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24
        r3 = (_fmix32(off_3.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24

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
# naive grid at the same generator state (device-verified); only the tiling
# changes.  B8192 in the loop keeps the live set within the envelope that
# B16384 loop form spills (13452 us pathology, measured).  EVEN_STEP drops
# the per-iteration scalar full/masked branch when N is an exact multiple
# of the stride; otherwise full tiles store unmasked and only the tail
# tile of each program stores masked.
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_kernel_persist(
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
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
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
            ha = _fmix32(off_0.to(tl.uint32) ^ s0)
            hb = _fmix32(off_2.to(tl.uint32) ^ s0)
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
            r0 = (_fmix32(off_0.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24
            r1 = (_fmix32(off_1.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24
            r2 = (_fmix32(off_2.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24
            r3 = (_fmix32(off_3.to(tl.uint32) ^ s0) >> 8).to(tl.float32) * U24
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


device_ = device

# 16-bit persistent ladder (second-round full-band rescan, do_bench median
# w1000/r100, GCU400): the grid-stride loop beats the naive grid from 2^19
# elements up, and the sweet (P, B) shrinks with N -- a per-program live set
# of 4*BLOCK output lanes with only 24-48 concurrent programs outperforms the
# first round's P192/B8192 at every measured point (P64/B4096 and
# P96/B8192 lose everywhere; larger tiles in the loop form spill):
#   2^19: P24/B4096 4.14-4.22 us   vs naive B8192/nw4 4.73
#   2^20: P24/B4096 4.81           vs naive B8192/nw4 7.01
#   2^21: P24/B4096 6.50           vs naive B16384/nw4 9.70
#   2^22: P24/B4096 9.60-9.64      vs P48/B4096 11.16-11.26
#   2^23: P48/B4096 18.51          vs naive B16384/nw4 25.64
#   2^24: P48/B4096 31.80          vs P192/B8192 43.29 (P96 34.7)
#   2^25: P48/B4096 59.46 | 2^26: 113.84 | 2^27: 223.20 | 1Gi: 1753.5
#   (1Gi L0 store floor 1736 us; P48 lands 1% above it, P192/B8192 3.8%.)
# fp32 gains nothing from the loop (hash-bound, not dispatch-bound): 2^24
# persist P96/B8192/nw2 51.61 vs naive B16384/nw2 47.99; 1Gi persist 2793 vs
# naive B65536/nw2 2584 -- fp32 keeps the naive grid at every size.
# The hash input stays the GLOBAL element offset, so persistent samples are
# bitwise identical to the naive grid at the same generator state; only the
# tiling changes, which keeps the accuracy gate free.
PERSIST_MIN_NUMEL = 1 << 19
PERSIST_BLOCK = 4096
PERSIST_NUM_PROG = 24
PERSIST_NUM_PROG_LARGE = 48
PERSIST_LARGE_MIN_NUMEL = 1 << 22


def rand(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME RAND")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    reduced = dtype in (torch.float16, torch.bfloat16)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    # BLOCK/num_warps ladder (second full-band rescan after the hash rewrite;
    # do_bench median, GCU400).  The small naive bands are shared across
    # dtypes, then the arms diverge:
    #   <=32Ki  B512/nw2  (launch floor 2.4-2.6 us; (64,64) 2.47)
    #   <=64Ki  B1024/nw2 (2.93-3.10 us; B2048 3.10-3.26 loses)
    #   <=128Ki B2048/nw2 (fp16 3.07 vs B1024 3.57; fp32 3.69 vs 4.06)
    #   <=256Ki B4096/nw2 (fp16 4.07, fp32 4.20 vs B8192 4.80)
    #   fp32:  <=2^23 B8192/nw2 (4.84@512Ki, 11.04@2Mi, 16.71@4Mi, 27.19@8Mi)
    #          <=2^27 B16384/nw2 (47.97-48.46@16.7M, 90.31@32Mi, 173.1@64Mi)
    #          else   B65536/nw2 (654.3@256Mi, 1299.5@512Mi, 2583.7@1Gi vs
    #                 B32768 2632; nw4 3035, B131072/nw2 2578.7 WITHIN-NOISE)
    #   16-bit: persist P24/B4096 in [2^19, 2^22], P48/B4096 beyond;
    #           naive B4096/nw2 covers (128Ki, 2^19).
    with torch_device_fn.device(device):
        if reduced and N >= PERSIST_MIN_NUMEL:
            if N <= PERSIST_LARGE_MIN_NUMEL:
                nprog = PERSIST_NUM_PROG
            else:
                nprog = PERSIST_NUM_PROG_LARGE
            grid = (min(nprog, triton.cdiv(N, PERSIST_BLOCK * UNROLL)),)
            rand_kernel_persist[grid](
                out,
                N,
                philox_seed,
                philox_offset,
                BLOCK=PERSIST_BLOCK,
                NUM_PROG=nprog,
                REDUCED=True,
                WIDE=N > 2**31 - PERSIST_BLOCK * UNROLL,
                EVEN_STEP=N % (nprog * PERSIST_BLOCK * UNROLL) == 0,
                num_warps=4,
            )
            return out

        if N <= 32768:
            BLOCK, nw = 512, 2
        elif N <= 65536:
            BLOCK, nw = 1024, 2
        elif N <= 131072:
            BLOCK, nw = 2048, 2
        elif reduced or N <= 262144:
            BLOCK, nw = 4096, 2
        elif N <= 8388608:
            BLOCK, nw = 8192, 2
        elif N <= 134217728:
            BLOCK, nw = 16384, 2
        else:
            BLOCK, nw = 65536, 2
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
        rand_kernel[grid](
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
