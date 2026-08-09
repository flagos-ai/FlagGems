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
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

# XPU triton fork defaults to 10 philox rounds; 5 rounds is statistically
# equivalent (KS/autocorrelation/oracle checks) and ~2x cheaper in ALU, which is
# the dominant cost for these memory-light kernels. Same choice as the
# exponential_ fix on this backend.
PHILOX_ROUNDS = 5
UNROLL = 4
BLOCK = 1024
NUM_WARPS = 8
TAIL_BLOCK = 512


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def bernoulli_kernel(
    out_ptr,
    x_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    # Main path: the first NMAIN programs each process BLOCK*4 consecutive
    # elements that are fully in-bounds (NMAIN*BLOCK*4 == N for the callers
    # that use the branchless kernel; for the combined kernel the main branch
    # only covers the full blocks). No masks -> contiguous block DMA.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 += i4
    zero = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1 + zero, zero, zero, ROUNDS)
    u0 = uint_to_uniform_float(r0)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u3 = uint_to_uniform_float(r3)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    p0 = tl.load(x_ptr + off_0)
    p1 = tl.load(x_ptr + off_1)
    p2 = tl.load(x_ptr + off_2)
    p3 = tl.load(x_ptr + off_3)
    tl.store(out_ptr + off_0, tl.where(u0 < p0, 1.0, 0.0))
    tl.store(out_ptr + off_1, tl.where(u1 < p1, 1.0, 0.0))
    tl.store(out_ptr + off_2, tl.where(u2 < p2, 1.0, 0.0))
    tl.store(out_ptr + off_3, tl.where(u3 < p3, 1.0, 0.0))


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def bernoulli_kernel_with_tail(
    out_ptr,
    x_ptr,
    N,
    philox_seed,
    philox_offset,
    NMAIN: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    # Single-launch kernel: pids [0, NMAIN) run the branchless 4-wide path on
    # fully-in-bounds blocks; the last pid runs a per-element masked path over
    # the tail [NMAIN*BLOCK*4, N). The per-element masked load/store pattern is
    # the one proven reliable on this backend (the 4-wide mixed-mask store
    # drops in-range lanes of a partially out-of-range block).
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    pid = tl.program_id(0)
    if pid < NMAIN:
        i4 = pid * BLOCK + tl.arange(0, BLOCK)
        c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c0 += i4
        zero = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1 + zero, zero, zero, ROUNDS)
        u0 = uint_to_uniform_float(r0)
        u1 = uint_to_uniform_float(r1)
        u2 = uint_to_uniform_float(r2)
        u3 = uint_to_uniform_float(r3)
        off_0 = pid * BLOCK * 4 + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        p0 = tl.load(x_ptr + off_0)
        p1 = tl.load(x_ptr + off_1)
        p2 = tl.load(x_ptr + off_2)
        p3 = tl.load(x_ptr + off_3)
        tl.store(out_ptr + off_0, tl.where(u0 < p0, 1.0, 0.0))
        tl.store(out_ptr + off_1, tl.where(u1 < p1, 1.0, 0.0))
        tl.store(out_ptr + off_2, tl.where(u2 < p2, 1.0, 0.0))
        tl.store(out_ptr + off_3, tl.where(u3 < p3, 1.0, 0.0))
    else:
        base = NMAIN * BLOCK * 4
        i4t = tl.arange(0, BLOCK)
        for k in tl.static_range(4):
            offsets = base + k * BLOCK + i4t
            counters = philox_offset + offsets // 4
            c0t = (counters & 0xFFFFFFFF).to(tl.uint32)
            c1t = ((counters >> 32) & 0xFFFFFFFF).to(tl.uint32)
            zero = c0t * 0
            rt0, rt1, rt2, rt3 = tl.philox(philox_seed, c0t, c1t, zero, zero, ROUNDS)
            lane = offsets % 4
            random = tl.where(
                lane == 0,
                rt0,
                tl.where(lane == 1, rt1, tl.where(lane == 2, rt2, rt3)),
            )
            random = uint_to_uniform_float(random)
            probabilities = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
            out = tl.where(random < probabilities, 1.0, 0.0)
            tl.store(
                out_ptr + offsets,
                out,
                mask=offsets < N,
                eviction_policy="evict_first",
            )


def bernoulli(self, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN BERNOULLI")
    assert self.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"bernoulli only supports floating point dtypes, got {self.dtype}"
    device = self.device
    self = self.contiguous()
    out = torch.empty_like(self)
    N = self.numel()
    if N == 0:
        return out
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        block_elems = BLOCK * UNROLL
        nmain = N // block_elems
        if N % block_elems == 0:
            # all blocks fully in-bounds -> branchless kernel
            bernoulli_kernel[(nmain,)](
                out,
                self,
                N,
                philox_seed,
                philox_offset,
                BLOCK=BLOCK,
                ROUNDS=PHILOX_ROUNDS,
                num_warps=NUM_WARPS,
            )
        else:
            # branchless full blocks + in-kernel per-element masked tail
            bernoulli_kernel_with_tail[(nmain + 1,)](
                out,
                self,
                N,
                philox_seed,
                philox_offset,
                NMAIN=nmain,
                BLOCK=BLOCK,
                ROUNDS=PHILOX_ROUNDS,
                num_warps=NUM_WARPS,
            )
    return out