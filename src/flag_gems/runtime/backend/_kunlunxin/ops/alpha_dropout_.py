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
#
import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

# SELU constants for the alpha dropout affine transform (same as the generic
# implementation and ATen).
_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_PRIME = -_ALPHA * _SCALE  # -1.7580993408473766

# The generic implementation indexes 4 elements per philox counter (one
# 4-output philox call per lane); keep the same grid structure so the random
# stream layout is unchanged.
UNROLL = 4


def _alpha_dropout_affine(p: float):
    """Compute the affine parameters a, b for alpha dropout.

    The affine transform preserves mean=0 and variance=1 of the underlying
    distribution, per the SELU paper (Klambauer et al. 2017).
    """
    if p == 0.0:
        return 1.0, 0.0
    a = 1.0 / math.sqrt((1.0 - p) * (1.0 + p * _ALPHA_PRIME * _ALPHA_PRIME))
    b = -a * p * _ALPHA_PRIME
    return a, b


@triton.jit(do_not_specialize=["p", "a", "b", "philox_seed", "philox_offset"])
def _alpha_dropout_inplace_kernel(
    X,
    N,
    p,
    a,
    b,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    # Main (block-aligned) region, fully unmasked. Element g is handled by
    # program g // (BLOCK*UNROLL), lane (g % BLOCK), sub-index (g // BLOCK) % 4
    # and uses r_(sub) of counter (c0 + (g // (BLOCK*UNROLL))*BLOCK + g % BLOCK)
    # - identical counter/sub mapping to the generic implementation. In-place
    # safety: every element is loaded and stored by exactly one lane.
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

    m0 = (r0 > p).to(tl.float32)
    m1 = (r1 > p).to(tl.float32)
    m2 = (r2 > p).to(tl.float32)
    m3 = (r3 > p).to(tl.float32)

    # alpha_prime is the value replacing dropped elements before the affine
    # transform.
    alpha_prime = -1.7580993408473766
    # y = tap + m*(a*x + b - tap) with m in {0,1}: m=1 (kept) -> a*x+b,
    # m=0 (dropped) -> tap = a*alpha' + b. This fma form avoids the
    # fcmp-derived tl.where select, which is a measured slow path on this
    # backend (10.6ms vs 3.1ms on 16M f32, see solution README).
    tap = a * alpha_prime + b

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0, eviction_policy="evict_first")
    x1 = tl.load(X + off_1, eviction_policy="evict_first")
    x2 = tl.load(X + off_2, eviction_policy="evict_first")
    x3 = tl.load(X + off_3, eviction_policy="evict_first")

    y0 = tap + m0 * (a * x0 + b - tap)
    y1 = tap + m1 * (a * x1 + b - tap)
    y2 = tap + m2 * (a * x2 + b - tap)
    y3 = tap + m3 * (a * x3 + b - tap)

    # In-place: store back to the same tensor.
    tl.store(X + off_0, y0, eviction_policy="evict_first")
    tl.store(X + off_1, y1, eviction_policy="evict_first")
    tl.store(X + off_2, y2, eviction_policy="evict_first")
    tl.store(X + off_3, y3, eviction_policy="evict_first")


@triton.jit(do_not_specialize=["BASE", "p", "a", "b", "philox_seed", "philox_offset"])
def _alpha_dropout_tail_kernel(
    X,
    BASE,
    N,
    p,
    a,
    b,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Tail (remainder) region: ONE masked load + ONE masked store per program.
    # Measured on this backend: an unmasked multi-store program silently drops
    # ALL of its stores as soon as any lane is out of bounds (masked multi-store
    # legalization defect), so the tail must be a single store.
    # Element g = BASE + off uses counter c0 + g // 4 and the (g % 4)-th output
    # of that counter, which keeps every counter within [c0, c0 + cdiv(N, 4)) -
    # the same seed/offset bounds as the generic implementation.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = BASE + off
    cnt = c0 + g // 4
    sub = g % 4
    _O = cnt * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, cnt, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    r = tl.where(sub == 0, r0, tl.where(sub == 1, r1, tl.where(sub == 2, r2, r3)))

    m = (r > p).to(tl.float32)
    x = tl.load(X + g, mask=g < N, other=0.0)
    alpha_prime = -1.7580993408473766
    # Same fma form as the main kernel: m=1 (kept) -> a*x+b, m=0 -> a*ap+b.
    tap = a * alpha_prime + b
    y = tap + m * (a * x + b - tap)
    tl.store(X + g, y, mask=g < N)


def _alpha_dropout_launch_config(N):
    if N <= 512:
        return 512, 4
    elif N <= 1024:
        return 1024, 8
    elif N <= 65536:
        return 1024, 16
    else:
        return 4096, 32


def alpha_dropout_(input, p=0.5, train=True):
    """Inplace alpha dropout (Kunlunxin/XPU optimized).

    Applies alpha dropout directly on the input tensor without allocating a
    separate output buffer. The philox stream (seed/offset rules) is identical
    to the generic implementation.
    """
    logger.debug("GEMS_KUNLUNXIN ALPHA_DROPOUT__INPLACE_FORWARD")

    if not train or p == 0:
        return input

    if p == 1.0:
        # ATen semantics verified against torch native (CPU & device): all
        # elements become the affine constant a*alpha' + b, which evaluates to
        # exactly 0.0 at p=1. (The generic implementation's
        # _alpha_dropout_affine(1.0) raises ZeroDivisionError; the functional
        # variant returns zeros_like, i.e. the same value.)
        input.fill_(0.0)
        return input

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input_contig = input.contiguous()
    is_contiguous = input.is_contiguous()

    N = input_contig.numel()
    if N == 0:
        return input

    BLOCK, num_warps = _alpha_dropout_launch_config(N)
    increment = triton.cdiv(N, UNROLL)

    a, b = _alpha_dropout_affine(p)

    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        # Constant-tuple grid (NOT a lambda): the launch grid participates in
        # the xpubin compile cache key, and a per-call lambda would force a
        # full recompile on every launch (~0.3-1s/call on this backend).
        full = (N // (BLOCK * UNROLL)) * (BLOCK * UNROLL)
        if full > 0:
            grid = (full // (BLOCK * UNROLL),)
            _alpha_dropout_inplace_kernel[grid](
                input_contig,
                full,
                p,
                a,
                b,
                philox_seed,
                philox_offset,
                BLOCK=BLOCK,
                UNROLL=UNROLL,
                num_warps=num_warps,
            )
        if full < N:
            tail = N - full
            grid_tail = (triton.cdiv(tail, 1024),)
            _alpha_dropout_tail_kernel[grid_tail](
                input_contig,
                full,
                N,
                p,
                a,
                b,
                philox_seed,
                philox_offset,
                BLOCK=1024,
                num_warps=8,
            )

    if not is_contiguous:
        # Restore the in-place result into the original (possibly non-contig)
        # storage. This is a device-side strided copy that does not move data
        # to the CPU and is required to keep the in-place semantics on
        # non-contiguous inputs; the kernels above ran on the contiguous view.
        input.copy_(input_contig)

    return input


__all__ = ["alpha_dropout_"]
