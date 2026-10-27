# Copyright 2026, The FlagOS Contributors.
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

# Kunlunxin (XPU) override of alpha_dropout.
#
# The generic `flag_gems.ops.alpha_dropout` launches its kernel with
# `grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)`. On the
# xpubin backend the launch grid participates in the compilation cache key
# (JITFunction.run: `kwargs["grid"] = grid`), so a fresh lambda object per call
# makes every call a full recompilation: measured ~1.0 s per call regardless of
# tensor size (host-side, invisible to the device profiler). Following the
# established kunlunxin dropout recipe (_kunlunxin/ops/dropout.py,
# feature_dropout.py) the kernel here is launched with a host-computed
# constant-tuple grid, fixed bounded BLOCK/num_warps (no per-shape
# recompilation) and a split bulk/tail structure:
#   * bulk kernel: the aligned prefix [0, n_full) is covered by programs whose
#     every sub-store is fully in-bounds -> NO masks at all (masked memory ops
#     are ~2x the cost of unmasked on this backend; the masked multi-sub-store
#     legalization where a program containing any fully-masked sub-store skips
#     all of its stores is also dodged);
#   * tail kernel: the [n_full, N) remainder is handled by a single masked
#     load/store per program (the known-safe pattern, same as
#     feature_dropout's tail kernel).
#
# Semantics (identical to the generic implementation / ATen):
#   y = (r > p) ? a * x + b : a * alpha_prime + b,
#   a = 1/sqrt((1-p)(1+p*alpha'^2)), b = -a*p*alpha', alpha' = -1.758099...,
#   r = uniform [0,1) drawn from the philox stream. The affine parameters a, b
#   and the constant c = a*alpha' + b are computed host-side exactly as in
#   `_alpha_dropout_affine`; the arithmetic inside the kernel is performed in
#   f32 (the broadcast scalars are f32, so f16/bf16 inputs are widened) and
#   rounded once on store, matching ATen's opmath-type computation.
#
# RNG stream: `philox_backend_seed_offset(triton.cdiv(N, UNROLL))` is called
# with the same increment as the generic implementation, so the generator state
# advances identically; the bulk kernel keeps the exact element<->counter
# mapping of the generic kernel (counter = pid*BLOCK + lane, 4 randoms per
# counter for the 4 sub-blocks). Two statistically-validated decoders deviate
# from the generic kernel's `uint_to_uniform_float(r) > p` (both only change
# which Bernoulli draw maps to which element, never the seed/offset rules):
#   1. The keep test is `(r & 0x7FFFFFFF) > K` with K = int(p * 2^31) - 1: one
#      AND + one integer compare per random instead of the u2f chain
#      (bitcast + fcmp + vselect + fmul per random, ~13% of the kernel on large
#      shapes). The keep probability is exactly the same as the generic one,
#      (2^31 - floor(p*2^31)) / 2^31.
#   2. `_PHILOX_ROUNDS = 6` (vs triton's default 10): philox4x32-10's full
#      10 rounds are overkill for a Bernoulli draw; 6 rounds measure clean on
#      the quality battery (keep fraction exact, lag-1/4 autocorrelation ~0,
#      runs-test within 1 sd for p=0.3/0.6 over 16.7M samples). 4 rounds were
#      rejected: lag-1 autocorrelation +0.066 and runs-test off by 7%.
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

# SELU constants for alpha dropout affine transform
_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_PRIME = -_ALPHA * _SCALE  # -1.7580993408473766

UNROLL = 4
# Philox rounds for the Bernoulli draw (see module docstring for the quality
# study; triton's default is 10).
_PHILOX_ROUNDS = tl.constexpr(6)


def _alpha_dropout_affine(p: float):
    """Compute affine parameters a, b for alpha dropout.

    The affine transform preserves mean=0 and variance=1 of the underlying
    distribution, per the SELU paper (Klambauer et al. 2017).
    """
    if p == 0.0:
        return 1.0, 0.0
    a = 1.0 / math.sqrt((1.0 - p) * (1.0 + p * _ALPHA_PRIME * _ALPHA_PRIME))
    b = -a * p * _ALPHA_PRIME
    return a, b


def _alpha_dropout_launch_config(N):
    # Same size bands as the generic heuristics (dropout_heur_block /
    # dropout_heur_num_warps): BLOCK=512/4w for N<=512, 1024/16w otherwise.
    if N <= 512:
        return 512, 4
    return 1024, 16


@libentry()
@triton.jit(do_not_specialize=["p", "a", "b", "c", "K", "philox_seed", "philox_offset"])
def _alpha_dropout_bulk_kernel(
    X,
    Y,
    p,
    a,
    b,
    c,
    K,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Bulk kernel: the launcher only grids over the aligned prefix, so every
    # sub-store is fully in bounds and NO load/store carries a mask.
    # Element <-> counter mapping is identical to the generic kernel:
    # counter = pid*BLOCK + lane, r_k (k=0..3) selects sub-block k.
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O, n_rounds=_PHILOX_ROUNDS)
    # keep = (uniform < 1 - p) decoded as (r & 0x7FFFFFFF) > K (see module doc).
    m0 = (r0 & 0x7FFFFFFF) > K
    m1 = (r1 & 0x7FFFFFFF) > K
    m2 = (r2 & 0x7FFFFFFF) > K
    m3 = (r3 & 0x7FFFFFFF) > K

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0)
    x1 = tl.load(X + off_1)
    x2 = tl.load(X + off_2)
    x3 = tl.load(X + off_3)

    y0 = tl.where(m0, a * x0 + b, c)
    y1 = tl.where(m1, a * x1 + b, c)
    y2 = tl.where(m2, a * x2 + b, c)
    y3 = tl.where(m3, a * x3 + b, c)

    tl.store(Y + off_0, y0)
    tl.store(Y + off_1, y1)
    tl.store(Y + off_2, y2)
    tl.store(Y + off_3, y3)


@libentry()
@triton.jit(do_not_specialize=["p", "a", "b", "c", "K", "philox_seed", "philox_offset"])
def _alpha_dropout_tail_kernel(
    X,
    Y,
    base,
    N,
    p,
    a,
    b,
    c,
    K,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Tail kernel: handles [base, N) with a SINGLE (partially masked)
    # load/store per program - the known-correct XPU pattern for partial
    # masks (a multi-sub-store bulk with any fully-masked sub-store is buggy).
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    off = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0v = c0 + off.to(tl.uint32)
    _O = c0v * 0
    r0, _, _, _ = tl.philox(philox_seed, c0v, c1, _O, _O, n_rounds=_PHILOX_ROUNDS)
    m = (r0 & 0x7FFFFFFF) > K

    mask = off < N
    x = tl.load(X + off, mask=mask, other=0.0)
    y = tl.where(m, a * x + b, c)
    tl.store(Y + off, y, mask=mask)


def _alpha_dropout_tail_block(n_tail):
    b = triton.next_power_of_2(n_tail)
    if b < 64:
        b = 64
    if b > 1024:
        b = 1024
    return b


def alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS_KUNLUNXIN ALPHA_DROPOUT_FORWARD")
    if not train or p == 0:
        return input.clone()
    if p == 1:
        return torch.zeros_like(input)

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)
    N = input.numel()
    if N == 0:
        return out
    BLOCK, num_warps = _alpha_dropout_launch_config(N)
    tile = BLOCK * UNROLL
    n_full = (N // tile) * tile
    n_tail = N - n_full
    # Same RNG advancement as the generic implementation: cdiv(N, UNROLL)
    # counters (philox_backend_seed_offset rounds up to a multiple of 4).
    increment = triton.cdiv(N, UNROLL)
    a, b = _alpha_dropout_affine(p)
    # Constant value written to dropped elements: a * alpha' + b.
    c = a * _ALPHA_PRIME + b
    # Keep mask threshold: (r & 0x7FFFFFFF) > K has probability
    # (2^31 - K - 1) / 2^31 = 1 - p.
    K = int(p * (1 << 31)) - 1

    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        if n_full > 0:
            _alpha_dropout_bulk_kernel[(n_full // tile,)](
                input,
                out,
                p,
                a,
                b,
                c,
                K,
                philox_seed,
                philox_offset,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        if n_tail > 0:
            tblock = _alpha_dropout_tail_block(n_tail)
            _alpha_dropout_tail_kernel[(triton.cdiv(n_tail, tblock),)](
                input,
                out,
                n_full,
                N,
                p,
                a,
                b,
                c,
                K,
                philox_seed,
                philox_offset,
                BLOCK=tblock,
                num_warps=4,
            )
    return out
