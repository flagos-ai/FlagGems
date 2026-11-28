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
import math

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# GCU400 replication_pad2d strategy (2026-11-16 v5 perf pass on top of the
# 2026-11-11 row-group rewrite). The v1 rewrite decomposed the op into a
# row-group bulk copy (affine fast path + per-row clamped slow path) plus a
# per-lane pad gather; v5 keeps that skeleton and removes the two costs that
# dominated the mid/narrow bench shapes (same fix family as _pad_circular v4):
#   1. per-row _repl_row_src div/mod chains on slow groups: a slow group's
#      rows are now decoded incrementally from the group-start (b0, h0) --
#      hh = h0 + i, single image-wrap select, clamp(hh - plh, 0, H-1) --
#      with no per-row division (the group preamble keeps one r0 // Ho,
#      now a compile-time-constant division). With the slow path cheap,
#      larger U unlocks: the 2026-11-11 "big U rejected" verdict was slow-
#      path row-swallowing amplification, not a U limit (U8 -> U16/U32 is
#      -31..-40% on the bench shapes). Only the <=1-per-program tail group
#      keeps the div-based per-row decode.
#   2. vector integer divisions in the phase-2 pad gather and the group
#      preambles (t//PW, r//Ho, r%Ho, ... cost ~0.5-0.7us per 1024-lane
#      division): all shape-derived scalars (W, Wo, plw, PW, Ho, H, plh) are
#      now tl.constexpr, so every division is a compile-time-constant
#      division and i*W address offsets fold to immediates. R stays a
#      runtime argument so leading-dim changes reuse one compiled kernel.
# Dispatch (2026-11-16 retune, official-bench-mirroring probe, bitwise
# checked vs aten): W >= 2048 -> U4/nw4; [768, 2048) -> U8/nw4; [256, 768)
# -> U32/nw2; [32, 256) -> U32/nw1; W < 32 -> U16/nw1; U clamped to <= Ho
# (single image wrap per group, shapes with Ho < 4 take the wide kernel) and
# to the compiled tiers {4, 8, 16, 32}; small-R grids clamp U to 4 and bump
# nw to 4 (launch-floor shapes). grid stays capped at 48; the wide-row
# (W > 8192) and int64-flat fallback kernels are unchanged.


@triton.jit
def _repl_row_src_c(
    r, W: tl.constexpr, Ho: tl.constexpr, H: tl.constexpr, plh: tl.constexpr
):
    # scalar source row base (element offset) of output row r under
    # replication (clamp instead of reflection fold-back); constexpr shape
    # scalars make the div/mod compile-time-constant
    b = r // Ho
    h = r % Ho
    sh = h - plh
    sh = tl.maximum(sh, 0)
    sh = tl.minimum(sh, H - 1)
    return (b * H + sh) * W


@triton.jit
def _repl_row_src_rt(r, W, Ho, H, plh):
    # runtime-arg twin for the wide/flat fallback kernels
    b = r // Ho
    h = r % Ho
    sh = h - plh
    sh = tl.maximum(sh, 0)
    sh = tl.minimum(sh, H - 1)
    return (b * H + sh) * W


@triton.jit
def replication_pad2d_rowseg_v5_kernel(
    x_ptr,
    o_ptr,
    R,
    W: tl.constexpr,
    Wo: tl.constexpr,
    plw: tl.constexpr,
    PW: tl.constexpr,
    Ho: tl.constexpr,
    H: tl.constexpr,
    plh: tl.constexpr,
    BLOCK: tl.constexpr,
    GBLOCK: tl.constexpr,
    U: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    mb = offs < W
    Hm1 = H - 1
    HW = H * W
    GU = G * U
    for r0 in range(pid * U, R, GU):
        b0 = r0 // Ho
        h0 = r0 - b0 * Ho
        b0p1 = b0 + 1
        fast = (r0 + U <= R) & (h0 >= plh) & (h0 + U <= plh + H)
        if U == 4:
            if fast:
                s0 = (b0 * H + h0 - plh) * W
                d0 = r0 * Wo + plw
                v0 = tl.load(x_ptr + s0 + 0 * W + offs, mask=mb)
                v1 = tl.load(x_ptr + s0 + 1 * W + offs, mask=mb)
                v2 = tl.load(x_ptr + s0 + 2 * W + offs, mask=mb)
                v3 = tl.load(x_ptr + s0 + 3 * W + offs, mask=mb)
                tl.store(o_ptr + d0 + 0 * Wo + offs, v0, mask=mb)
                tl.store(o_ptr + d0 + 1 * Wo + offs, v1, mask=mb)
                tl.store(o_ptr + d0 + 2 * Wo + offs, v2, mask=mb)
                tl.store(o_ptr + d0 + 3 * Wo + offs, v3, mask=mb)
            elif r0 + U <= R:
                hh0 = h0 + 0
                w0 = hh0 >= Ho
                hi0 = tl.where(w0, hh0 - Ho, hh0)
                bi0 = tl.where(w0, b0p1, b0)
                sh0 = hi0 - plh
                sh0 = tl.maximum(sh0, 0)
                sh0 = tl.minimum(sh0, Hm1)
                v0 = tl.load(x_ptr + bi0 * HW + sh0 * W + offs, mask=mb)
                hh1 = h0 + 1
                w1 = hh1 >= Ho
                hi1 = tl.where(w1, hh1 - Ho, hh1)
                bi1 = tl.where(w1, b0p1, b0)
                sh1 = hi1 - plh
                sh1 = tl.maximum(sh1, 0)
                sh1 = tl.minimum(sh1, Hm1)
                v1 = tl.load(x_ptr + bi1 * HW + sh1 * W + offs, mask=mb)
                hh2 = h0 + 2
                w2 = hh2 >= Ho
                hi2 = tl.where(w2, hh2 - Ho, hh2)
                bi2 = tl.where(w2, b0p1, b0)
                sh2 = hi2 - plh
                sh2 = tl.maximum(sh2, 0)
                sh2 = tl.minimum(sh2, Hm1)
                v2 = tl.load(x_ptr + bi2 * HW + sh2 * W + offs, mask=mb)
                hh3 = h0 + 3
                w3 = hh3 >= Ho
                hi3 = tl.where(w3, hh3 - Ho, hh3)
                bi3 = tl.where(w3, b0p1, b0)
                sh3 = hi3 - plh
                sh3 = tl.maximum(sh3, 0)
                sh3 = tl.minimum(sh3, Hm1)
                v3 = tl.load(x_ptr + bi3 * HW + sh3 * W + offs, mask=mb)
                tl.store(o_ptr + (r0 + 0) * Wo + plw + offs, v0, mask=mb)
                tl.store(o_ptr + (r0 + 1) * Wo + plw + offs, v1, mask=mb)
                tl.store(o_ptr + (r0 + 2) * Wo + plw + offs, v2, mask=mb)
                tl.store(o_ptr + (r0 + 3) * Wo + plw + offs, v3, mask=mb)
            else:
                for i in range(0, U):
                    r_i = tl.minimum(r0 + i, R - 1)
                    v_i = tl.load(
                        x_ptr + _repl_row_src_c(r_i, W, Ho, H, plh) + offs,
                        mask=mb,
                    )
                    tl.store(o_ptr + r_i * Wo + plw + offs, v_i, mask=mb)
        elif U == 8:
            if fast:
                s0 = (b0 * H + h0 - plh) * W
                d0 = r0 * Wo + plw
                v0 = tl.load(x_ptr + s0 + 0 * W + offs, mask=mb)
                v1 = tl.load(x_ptr + s0 + 1 * W + offs, mask=mb)
                v2 = tl.load(x_ptr + s0 + 2 * W + offs, mask=mb)
                v3 = tl.load(x_ptr + s0 + 3 * W + offs, mask=mb)
                v4 = tl.load(x_ptr + s0 + 4 * W + offs, mask=mb)
                v5 = tl.load(x_ptr + s0 + 5 * W + offs, mask=mb)
                v6 = tl.load(x_ptr + s0 + 6 * W + offs, mask=mb)
                v7 = tl.load(x_ptr + s0 + 7 * W + offs, mask=mb)
                tl.store(o_ptr + d0 + 0 * Wo + offs, v0, mask=mb)
                tl.store(o_ptr + d0 + 1 * Wo + offs, v1, mask=mb)
                tl.store(o_ptr + d0 + 2 * Wo + offs, v2, mask=mb)
                tl.store(o_ptr + d0 + 3 * Wo + offs, v3, mask=mb)
                tl.store(o_ptr + d0 + 4 * Wo + offs, v4, mask=mb)
                tl.store(o_ptr + d0 + 5 * Wo + offs, v5, mask=mb)
                tl.store(o_ptr + d0 + 6 * Wo + offs, v6, mask=mb)
                tl.store(o_ptr + d0 + 7 * Wo + offs, v7, mask=mb)
            elif r0 + U <= R:
                hh0 = h0 + 0
                w0 = hh0 >= Ho
                hi0 = tl.where(w0, hh0 - Ho, hh0)
                bi0 = tl.where(w0, b0p1, b0)
                sh0 = hi0 - plh
                sh0 = tl.maximum(sh0, 0)
                sh0 = tl.minimum(sh0, Hm1)
                v0 = tl.load(x_ptr + bi0 * HW + sh0 * W + offs, mask=mb)
                hh1 = h0 + 1
                w1 = hh1 >= Ho
                hi1 = tl.where(w1, hh1 - Ho, hh1)
                bi1 = tl.where(w1, b0p1, b0)
                sh1 = hi1 - plh
                sh1 = tl.maximum(sh1, 0)
                sh1 = tl.minimum(sh1, Hm1)
                v1 = tl.load(x_ptr + bi1 * HW + sh1 * W + offs, mask=mb)
                hh2 = h0 + 2
                w2 = hh2 >= Ho
                hi2 = tl.where(w2, hh2 - Ho, hh2)
                bi2 = tl.where(w2, b0p1, b0)
                sh2 = hi2 - plh
                sh2 = tl.maximum(sh2, 0)
                sh2 = tl.minimum(sh2, Hm1)
                v2 = tl.load(x_ptr + bi2 * HW + sh2 * W + offs, mask=mb)
                hh3 = h0 + 3
                w3 = hh3 >= Ho
                hi3 = tl.where(w3, hh3 - Ho, hh3)
                bi3 = tl.where(w3, b0p1, b0)
                sh3 = hi3 - plh
                sh3 = tl.maximum(sh3, 0)
                sh3 = tl.minimum(sh3, Hm1)
                v3 = tl.load(x_ptr + bi3 * HW + sh3 * W + offs, mask=mb)
                hh4 = h0 + 4
                w4 = hh4 >= Ho
                hi4 = tl.where(w4, hh4 - Ho, hh4)
                bi4 = tl.where(w4, b0p1, b0)
                sh4 = hi4 - plh
                sh4 = tl.maximum(sh4, 0)
                sh4 = tl.minimum(sh4, Hm1)
                v4 = tl.load(x_ptr + bi4 * HW + sh4 * W + offs, mask=mb)
                hh5 = h0 + 5
                w5 = hh5 >= Ho
                hi5 = tl.where(w5, hh5 - Ho, hh5)
                bi5 = tl.where(w5, b0p1, b0)
                sh5 = hi5 - plh
                sh5 = tl.maximum(sh5, 0)
                sh5 = tl.minimum(sh5, Hm1)
                v5 = tl.load(x_ptr + bi5 * HW + sh5 * W + offs, mask=mb)
                hh6 = h0 + 6
                w6 = hh6 >= Ho
                hi6 = tl.where(w6, hh6 - Ho, hh6)
                bi6 = tl.where(w6, b0p1, b0)
                sh6 = hi6 - plh
                sh6 = tl.maximum(sh6, 0)
                sh6 = tl.minimum(sh6, Hm1)
                v6 = tl.load(x_ptr + bi6 * HW + sh6 * W + offs, mask=mb)
                hh7 = h0 + 7
                w7 = hh7 >= Ho
                hi7 = tl.where(w7, hh7 - Ho, hh7)
                bi7 = tl.where(w7, b0p1, b0)
                sh7 = hi7 - plh
                sh7 = tl.maximum(sh7, 0)
                sh7 = tl.minimum(sh7, Hm1)
                v7 = tl.load(x_ptr + bi7 * HW + sh7 * W + offs, mask=mb)
                tl.store(o_ptr + (r0 + 0) * Wo + plw + offs, v0, mask=mb)
                tl.store(o_ptr + (r0 + 1) * Wo + plw + offs, v1, mask=mb)
                tl.store(o_ptr + (r0 + 2) * Wo + plw + offs, v2, mask=mb)
                tl.store(o_ptr + (r0 + 3) * Wo + plw + offs, v3, mask=mb)
                tl.store(o_ptr + (r0 + 4) * Wo + plw + offs, v4, mask=mb)
                tl.store(o_ptr + (r0 + 5) * Wo + plw + offs, v5, mask=mb)
                tl.store(o_ptr + (r0 + 6) * Wo + plw + offs, v6, mask=mb)
                tl.store(o_ptr + (r0 + 7) * Wo + plw + offs, v7, mask=mb)
            else:
                for i in range(0, U):
                    r_i = tl.minimum(r0 + i, R - 1)
                    v_i = tl.load(
                        x_ptr + _repl_row_src_c(r_i, W, Ho, H, plh) + offs,
                        mask=mb,
                    )
                    tl.store(o_ptr + r_i * Wo + plw + offs, v_i, mask=mb)
        elif U == 16:
            if fast:
                s0 = (b0 * H + h0 - plh) * W
                d0 = r0 * Wo + plw
                v0 = tl.load(x_ptr + s0 + 0 * W + offs, mask=mb)
                v1 = tl.load(x_ptr + s0 + 1 * W + offs, mask=mb)
                v2 = tl.load(x_ptr + s0 + 2 * W + offs, mask=mb)
                v3 = tl.load(x_ptr + s0 + 3 * W + offs, mask=mb)
                v4 = tl.load(x_ptr + s0 + 4 * W + offs, mask=mb)
                v5 = tl.load(x_ptr + s0 + 5 * W + offs, mask=mb)
                v6 = tl.load(x_ptr + s0 + 6 * W + offs, mask=mb)
                v7 = tl.load(x_ptr + s0 + 7 * W + offs, mask=mb)
                v8 = tl.load(x_ptr + s0 + 8 * W + offs, mask=mb)
                v9 = tl.load(x_ptr + s0 + 9 * W + offs, mask=mb)
                v10 = tl.load(x_ptr + s0 + 10 * W + offs, mask=mb)
                v11 = tl.load(x_ptr + s0 + 11 * W + offs, mask=mb)
                v12 = tl.load(x_ptr + s0 + 12 * W + offs, mask=mb)
                v13 = tl.load(x_ptr + s0 + 13 * W + offs, mask=mb)
                v14 = tl.load(x_ptr + s0 + 14 * W + offs, mask=mb)
                v15 = tl.load(x_ptr + s0 + 15 * W + offs, mask=mb)
                tl.store(o_ptr + d0 + 0 * Wo + offs, v0, mask=mb)
                tl.store(o_ptr + d0 + 1 * Wo + offs, v1, mask=mb)
                tl.store(o_ptr + d0 + 2 * Wo + offs, v2, mask=mb)
                tl.store(o_ptr + d0 + 3 * Wo + offs, v3, mask=mb)
                tl.store(o_ptr + d0 + 4 * Wo + offs, v4, mask=mb)
                tl.store(o_ptr + d0 + 5 * Wo + offs, v5, mask=mb)
                tl.store(o_ptr + d0 + 6 * Wo + offs, v6, mask=mb)
                tl.store(o_ptr + d0 + 7 * Wo + offs, v7, mask=mb)
                tl.store(o_ptr + d0 + 8 * Wo + offs, v8, mask=mb)
                tl.store(o_ptr + d0 + 9 * Wo + offs, v9, mask=mb)
                tl.store(o_ptr + d0 + 10 * Wo + offs, v10, mask=mb)
                tl.store(o_ptr + d0 + 11 * Wo + offs, v11, mask=mb)
                tl.store(o_ptr + d0 + 12 * Wo + offs, v12, mask=mb)
                tl.store(o_ptr + d0 + 13 * Wo + offs, v13, mask=mb)
                tl.store(o_ptr + d0 + 14 * Wo + offs, v14, mask=mb)
                tl.store(o_ptr + d0 + 15 * Wo + offs, v15, mask=mb)
            elif r0 + U <= R:
                hh0 = h0 + 0
                w0 = hh0 >= Ho
                hi0 = tl.where(w0, hh0 - Ho, hh0)
                bi0 = tl.where(w0, b0p1, b0)
                sh0 = hi0 - plh
                sh0 = tl.maximum(sh0, 0)
                sh0 = tl.minimum(sh0, Hm1)
                v0 = tl.load(x_ptr + bi0 * HW + sh0 * W + offs, mask=mb)
                hh1 = h0 + 1
                w1 = hh1 >= Ho
                hi1 = tl.where(w1, hh1 - Ho, hh1)
                bi1 = tl.where(w1, b0p1, b0)
                sh1 = hi1 - plh
                sh1 = tl.maximum(sh1, 0)
                sh1 = tl.minimum(sh1, Hm1)
                v1 = tl.load(x_ptr + bi1 * HW + sh1 * W + offs, mask=mb)
                hh2 = h0 + 2
                w2 = hh2 >= Ho
                hi2 = tl.where(w2, hh2 - Ho, hh2)
                bi2 = tl.where(w2, b0p1, b0)
                sh2 = hi2 - plh
                sh2 = tl.maximum(sh2, 0)
                sh2 = tl.minimum(sh2, Hm1)
                v2 = tl.load(x_ptr + bi2 * HW + sh2 * W + offs, mask=mb)
                hh3 = h0 + 3
                w3 = hh3 >= Ho
                hi3 = tl.where(w3, hh3 - Ho, hh3)
                bi3 = tl.where(w3, b0p1, b0)
                sh3 = hi3 - plh
                sh3 = tl.maximum(sh3, 0)
                sh3 = tl.minimum(sh3, Hm1)
                v3 = tl.load(x_ptr + bi3 * HW + sh3 * W + offs, mask=mb)
                hh4 = h0 + 4
                w4 = hh4 >= Ho
                hi4 = tl.where(w4, hh4 - Ho, hh4)
                bi4 = tl.where(w4, b0p1, b0)
                sh4 = hi4 - plh
                sh4 = tl.maximum(sh4, 0)
                sh4 = tl.minimum(sh4, Hm1)
                v4 = tl.load(x_ptr + bi4 * HW + sh4 * W + offs, mask=mb)
                hh5 = h0 + 5
                w5 = hh5 >= Ho
                hi5 = tl.where(w5, hh5 - Ho, hh5)
                bi5 = tl.where(w5, b0p1, b0)
                sh5 = hi5 - plh
                sh5 = tl.maximum(sh5, 0)
                sh5 = tl.minimum(sh5, Hm1)
                v5 = tl.load(x_ptr + bi5 * HW + sh5 * W + offs, mask=mb)
                hh6 = h0 + 6
                w6 = hh6 >= Ho
                hi6 = tl.where(w6, hh6 - Ho, hh6)
                bi6 = tl.where(w6, b0p1, b0)
                sh6 = hi6 - plh
                sh6 = tl.maximum(sh6, 0)
                sh6 = tl.minimum(sh6, Hm1)
                v6 = tl.load(x_ptr + bi6 * HW + sh6 * W + offs, mask=mb)
                hh7 = h0 + 7
                w7 = hh7 >= Ho
                hi7 = tl.where(w7, hh7 - Ho, hh7)
                bi7 = tl.where(w7, b0p1, b0)
                sh7 = hi7 - plh
                sh7 = tl.maximum(sh7, 0)
                sh7 = tl.minimum(sh7, Hm1)
                v7 = tl.load(x_ptr + bi7 * HW + sh7 * W + offs, mask=mb)
                hh8 = h0 + 8
                w8 = hh8 >= Ho
                hi8 = tl.where(w8, hh8 - Ho, hh8)
                bi8 = tl.where(w8, b0p1, b0)
                sh8 = hi8 - plh
                sh8 = tl.maximum(sh8, 0)
                sh8 = tl.minimum(sh8, Hm1)
                v8 = tl.load(x_ptr + bi8 * HW + sh8 * W + offs, mask=mb)
                hh9 = h0 + 9
                w9 = hh9 >= Ho
                hi9 = tl.where(w9, hh9 - Ho, hh9)
                bi9 = tl.where(w9, b0p1, b0)
                sh9 = hi9 - plh
                sh9 = tl.maximum(sh9, 0)
                sh9 = tl.minimum(sh9, Hm1)
                v9 = tl.load(x_ptr + bi9 * HW + sh9 * W + offs, mask=mb)
                hh10 = h0 + 10
                w10 = hh10 >= Ho
                hi10 = tl.where(w10, hh10 - Ho, hh10)
                bi10 = tl.where(w10, b0p1, b0)
                sh10 = hi10 - plh
                sh10 = tl.maximum(sh10, 0)
                sh10 = tl.minimum(sh10, Hm1)
                v10 = tl.load(x_ptr + bi10 * HW + sh10 * W + offs, mask=mb)
                hh11 = h0 + 11
                w11 = hh11 >= Ho
                hi11 = tl.where(w11, hh11 - Ho, hh11)
                bi11 = tl.where(w11, b0p1, b0)
                sh11 = hi11 - plh
                sh11 = tl.maximum(sh11, 0)
                sh11 = tl.minimum(sh11, Hm1)
                v11 = tl.load(x_ptr + bi11 * HW + sh11 * W + offs, mask=mb)
                hh12 = h0 + 12
                w12 = hh12 >= Ho
                hi12 = tl.where(w12, hh12 - Ho, hh12)
                bi12 = tl.where(w12, b0p1, b0)
                sh12 = hi12 - plh
                sh12 = tl.maximum(sh12, 0)
                sh12 = tl.minimum(sh12, Hm1)
                v12 = tl.load(x_ptr + bi12 * HW + sh12 * W + offs, mask=mb)
                hh13 = h0 + 13
                w13 = hh13 >= Ho
                hi13 = tl.where(w13, hh13 - Ho, hh13)
                bi13 = tl.where(w13, b0p1, b0)
                sh13 = hi13 - plh
                sh13 = tl.maximum(sh13, 0)
                sh13 = tl.minimum(sh13, Hm1)
                v13 = tl.load(x_ptr + bi13 * HW + sh13 * W + offs, mask=mb)
                hh14 = h0 + 14
                w14 = hh14 >= Ho
                hi14 = tl.where(w14, hh14 - Ho, hh14)
                bi14 = tl.where(w14, b0p1, b0)
                sh14 = hi14 - plh
                sh14 = tl.maximum(sh14, 0)
                sh14 = tl.minimum(sh14, Hm1)
                v14 = tl.load(x_ptr + bi14 * HW + sh14 * W + offs, mask=mb)
                hh15 = h0 + 15
                w15 = hh15 >= Ho
                hi15 = tl.where(w15, hh15 - Ho, hh15)
                bi15 = tl.where(w15, b0p1, b0)
                sh15 = hi15 - plh
                sh15 = tl.maximum(sh15, 0)
                sh15 = tl.minimum(sh15, Hm1)
                v15 = tl.load(x_ptr + bi15 * HW + sh15 * W + offs, mask=mb)
                tl.store(o_ptr + (r0 + 0) * Wo + plw + offs, v0, mask=mb)
                tl.store(o_ptr + (r0 + 1) * Wo + plw + offs, v1, mask=mb)
                tl.store(o_ptr + (r0 + 2) * Wo + plw + offs, v2, mask=mb)
                tl.store(o_ptr + (r0 + 3) * Wo + plw + offs, v3, mask=mb)
                tl.store(o_ptr + (r0 + 4) * Wo + plw + offs, v4, mask=mb)
                tl.store(o_ptr + (r0 + 5) * Wo + plw + offs, v5, mask=mb)
                tl.store(o_ptr + (r0 + 6) * Wo + plw + offs, v6, mask=mb)
                tl.store(o_ptr + (r0 + 7) * Wo + plw + offs, v7, mask=mb)
                tl.store(o_ptr + (r0 + 8) * Wo + plw + offs, v8, mask=mb)
                tl.store(o_ptr + (r0 + 9) * Wo + plw + offs, v9, mask=mb)
                tl.store(o_ptr + (r0 + 10) * Wo + plw + offs, v10, mask=mb)
                tl.store(o_ptr + (r0 + 11) * Wo + plw + offs, v11, mask=mb)
                tl.store(o_ptr + (r0 + 12) * Wo + plw + offs, v12, mask=mb)
                tl.store(o_ptr + (r0 + 13) * Wo + plw + offs, v13, mask=mb)
                tl.store(o_ptr + (r0 + 14) * Wo + plw + offs, v14, mask=mb)
                tl.store(o_ptr + (r0 + 15) * Wo + plw + offs, v15, mask=mb)
            else:
                for i in range(0, U):
                    r_i = tl.minimum(r0 + i, R - 1)
                    v_i = tl.load(
                        x_ptr + _repl_row_src_c(r_i, W, Ho, H, plh) + offs,
                        mask=mb,
                    )
                    tl.store(o_ptr + r_i * Wo + plw + offs, v_i, mask=mb)
        else:
            if fast:
                s0 = (b0 * H + h0 - plh) * W
                d0 = r0 * Wo + plw
                v0 = tl.load(x_ptr + s0 + 0 * W + offs, mask=mb)
                v1 = tl.load(x_ptr + s0 + 1 * W + offs, mask=mb)
                v2 = tl.load(x_ptr + s0 + 2 * W + offs, mask=mb)
                v3 = tl.load(x_ptr + s0 + 3 * W + offs, mask=mb)
                v4 = tl.load(x_ptr + s0 + 4 * W + offs, mask=mb)
                v5 = tl.load(x_ptr + s0 + 5 * W + offs, mask=mb)
                v6 = tl.load(x_ptr + s0 + 6 * W + offs, mask=mb)
                v7 = tl.load(x_ptr + s0 + 7 * W + offs, mask=mb)
                v8 = tl.load(x_ptr + s0 + 8 * W + offs, mask=mb)
                v9 = tl.load(x_ptr + s0 + 9 * W + offs, mask=mb)
                v10 = tl.load(x_ptr + s0 + 10 * W + offs, mask=mb)
                v11 = tl.load(x_ptr + s0 + 11 * W + offs, mask=mb)
                v12 = tl.load(x_ptr + s0 + 12 * W + offs, mask=mb)
                v13 = tl.load(x_ptr + s0 + 13 * W + offs, mask=mb)
                v14 = tl.load(x_ptr + s0 + 14 * W + offs, mask=mb)
                v15 = tl.load(x_ptr + s0 + 15 * W + offs, mask=mb)
                v16 = tl.load(x_ptr + s0 + 16 * W + offs, mask=mb)
                v17 = tl.load(x_ptr + s0 + 17 * W + offs, mask=mb)
                v18 = tl.load(x_ptr + s0 + 18 * W + offs, mask=mb)
                v19 = tl.load(x_ptr + s0 + 19 * W + offs, mask=mb)
                v20 = tl.load(x_ptr + s0 + 20 * W + offs, mask=mb)
                v21 = tl.load(x_ptr + s0 + 21 * W + offs, mask=mb)
                v22 = tl.load(x_ptr + s0 + 22 * W + offs, mask=mb)
                v23 = tl.load(x_ptr + s0 + 23 * W + offs, mask=mb)
                v24 = tl.load(x_ptr + s0 + 24 * W + offs, mask=mb)
                v25 = tl.load(x_ptr + s0 + 25 * W + offs, mask=mb)
                v26 = tl.load(x_ptr + s0 + 26 * W + offs, mask=mb)
                v27 = tl.load(x_ptr + s0 + 27 * W + offs, mask=mb)
                v28 = tl.load(x_ptr + s0 + 28 * W + offs, mask=mb)
                v29 = tl.load(x_ptr + s0 + 29 * W + offs, mask=mb)
                v30 = tl.load(x_ptr + s0 + 30 * W + offs, mask=mb)
                v31 = tl.load(x_ptr + s0 + 31 * W + offs, mask=mb)
                tl.store(o_ptr + d0 + 0 * Wo + offs, v0, mask=mb)
                tl.store(o_ptr + d0 + 1 * Wo + offs, v1, mask=mb)
                tl.store(o_ptr + d0 + 2 * Wo + offs, v2, mask=mb)
                tl.store(o_ptr + d0 + 3 * Wo + offs, v3, mask=mb)
                tl.store(o_ptr + d0 + 4 * Wo + offs, v4, mask=mb)
                tl.store(o_ptr + d0 + 5 * Wo + offs, v5, mask=mb)
                tl.store(o_ptr + d0 + 6 * Wo + offs, v6, mask=mb)
                tl.store(o_ptr + d0 + 7 * Wo + offs, v7, mask=mb)
                tl.store(o_ptr + d0 + 8 * Wo + offs, v8, mask=mb)
                tl.store(o_ptr + d0 + 9 * Wo + offs, v9, mask=mb)
                tl.store(o_ptr + d0 + 10 * Wo + offs, v10, mask=mb)
                tl.store(o_ptr + d0 + 11 * Wo + offs, v11, mask=mb)
                tl.store(o_ptr + d0 + 12 * Wo + offs, v12, mask=mb)
                tl.store(o_ptr + d0 + 13 * Wo + offs, v13, mask=mb)
                tl.store(o_ptr + d0 + 14 * Wo + offs, v14, mask=mb)
                tl.store(o_ptr + d0 + 15 * Wo + offs, v15, mask=mb)
                tl.store(o_ptr + d0 + 16 * Wo + offs, v16, mask=mb)
                tl.store(o_ptr + d0 + 17 * Wo + offs, v17, mask=mb)
                tl.store(o_ptr + d0 + 18 * Wo + offs, v18, mask=mb)
                tl.store(o_ptr + d0 + 19 * Wo + offs, v19, mask=mb)
                tl.store(o_ptr + d0 + 20 * Wo + offs, v20, mask=mb)
                tl.store(o_ptr + d0 + 21 * Wo + offs, v21, mask=mb)
                tl.store(o_ptr + d0 + 22 * Wo + offs, v22, mask=mb)
                tl.store(o_ptr + d0 + 23 * Wo + offs, v23, mask=mb)
                tl.store(o_ptr + d0 + 24 * Wo + offs, v24, mask=mb)
                tl.store(o_ptr + d0 + 25 * Wo + offs, v25, mask=mb)
                tl.store(o_ptr + d0 + 26 * Wo + offs, v26, mask=mb)
                tl.store(o_ptr + d0 + 27 * Wo + offs, v27, mask=mb)
                tl.store(o_ptr + d0 + 28 * Wo + offs, v28, mask=mb)
                tl.store(o_ptr + d0 + 29 * Wo + offs, v29, mask=mb)
                tl.store(o_ptr + d0 + 30 * Wo + offs, v30, mask=mb)
                tl.store(o_ptr + d0 + 31 * Wo + offs, v31, mask=mb)
            elif r0 + U <= R:
                hh0 = h0 + 0
                w0 = hh0 >= Ho
                hi0 = tl.where(w0, hh0 - Ho, hh0)
                bi0 = tl.where(w0, b0p1, b0)
                sh0 = hi0 - plh
                sh0 = tl.maximum(sh0, 0)
                sh0 = tl.minimum(sh0, Hm1)
                v0 = tl.load(x_ptr + bi0 * HW + sh0 * W + offs, mask=mb)
                hh1 = h0 + 1
                w1 = hh1 >= Ho
                hi1 = tl.where(w1, hh1 - Ho, hh1)
                bi1 = tl.where(w1, b0p1, b0)
                sh1 = hi1 - plh
                sh1 = tl.maximum(sh1, 0)
                sh1 = tl.minimum(sh1, Hm1)
                v1 = tl.load(x_ptr + bi1 * HW + sh1 * W + offs, mask=mb)
                hh2 = h0 + 2
                w2 = hh2 >= Ho
                hi2 = tl.where(w2, hh2 - Ho, hh2)
                bi2 = tl.where(w2, b0p1, b0)
                sh2 = hi2 - plh
                sh2 = tl.maximum(sh2, 0)
                sh2 = tl.minimum(sh2, Hm1)
                v2 = tl.load(x_ptr + bi2 * HW + sh2 * W + offs, mask=mb)
                hh3 = h0 + 3
                w3 = hh3 >= Ho
                hi3 = tl.where(w3, hh3 - Ho, hh3)
                bi3 = tl.where(w3, b0p1, b0)
                sh3 = hi3 - plh
                sh3 = tl.maximum(sh3, 0)
                sh3 = tl.minimum(sh3, Hm1)
                v3 = tl.load(x_ptr + bi3 * HW + sh3 * W + offs, mask=mb)
                hh4 = h0 + 4
                w4 = hh4 >= Ho
                hi4 = tl.where(w4, hh4 - Ho, hh4)
                bi4 = tl.where(w4, b0p1, b0)
                sh4 = hi4 - plh
                sh4 = tl.maximum(sh4, 0)
                sh4 = tl.minimum(sh4, Hm1)
                v4 = tl.load(x_ptr + bi4 * HW + sh4 * W + offs, mask=mb)
                hh5 = h0 + 5
                w5 = hh5 >= Ho
                hi5 = tl.where(w5, hh5 - Ho, hh5)
                bi5 = tl.where(w5, b0p1, b0)
                sh5 = hi5 - plh
                sh5 = tl.maximum(sh5, 0)
                sh5 = tl.minimum(sh5, Hm1)
                v5 = tl.load(x_ptr + bi5 * HW + sh5 * W + offs, mask=mb)
                hh6 = h0 + 6
                w6 = hh6 >= Ho
                hi6 = tl.where(w6, hh6 - Ho, hh6)
                bi6 = tl.where(w6, b0p1, b0)
                sh6 = hi6 - plh
                sh6 = tl.maximum(sh6, 0)
                sh6 = tl.minimum(sh6, Hm1)
                v6 = tl.load(x_ptr + bi6 * HW + sh6 * W + offs, mask=mb)
                hh7 = h0 + 7
                w7 = hh7 >= Ho
                hi7 = tl.where(w7, hh7 - Ho, hh7)
                bi7 = tl.where(w7, b0p1, b0)
                sh7 = hi7 - plh
                sh7 = tl.maximum(sh7, 0)
                sh7 = tl.minimum(sh7, Hm1)
                v7 = tl.load(x_ptr + bi7 * HW + sh7 * W + offs, mask=mb)
                hh8 = h0 + 8
                w8 = hh8 >= Ho
                hi8 = tl.where(w8, hh8 - Ho, hh8)
                bi8 = tl.where(w8, b0p1, b0)
                sh8 = hi8 - plh
                sh8 = tl.maximum(sh8, 0)
                sh8 = tl.minimum(sh8, Hm1)
                v8 = tl.load(x_ptr + bi8 * HW + sh8 * W + offs, mask=mb)
                hh9 = h0 + 9
                w9 = hh9 >= Ho
                hi9 = tl.where(w9, hh9 - Ho, hh9)
                bi9 = tl.where(w9, b0p1, b0)
                sh9 = hi9 - plh
                sh9 = tl.maximum(sh9, 0)
                sh9 = tl.minimum(sh9, Hm1)
                v9 = tl.load(x_ptr + bi9 * HW + sh9 * W + offs, mask=mb)
                hh10 = h0 + 10
                w10 = hh10 >= Ho
                hi10 = tl.where(w10, hh10 - Ho, hh10)
                bi10 = tl.where(w10, b0p1, b0)
                sh10 = hi10 - plh
                sh10 = tl.maximum(sh10, 0)
                sh10 = tl.minimum(sh10, Hm1)
                v10 = tl.load(x_ptr + bi10 * HW + sh10 * W + offs, mask=mb)
                hh11 = h0 + 11
                w11 = hh11 >= Ho
                hi11 = tl.where(w11, hh11 - Ho, hh11)
                bi11 = tl.where(w11, b0p1, b0)
                sh11 = hi11 - plh
                sh11 = tl.maximum(sh11, 0)
                sh11 = tl.minimum(sh11, Hm1)
                v11 = tl.load(x_ptr + bi11 * HW + sh11 * W + offs, mask=mb)
                hh12 = h0 + 12
                w12 = hh12 >= Ho
                hi12 = tl.where(w12, hh12 - Ho, hh12)
                bi12 = tl.where(w12, b0p1, b0)
                sh12 = hi12 - plh
                sh12 = tl.maximum(sh12, 0)
                sh12 = tl.minimum(sh12, Hm1)
                v12 = tl.load(x_ptr + bi12 * HW + sh12 * W + offs, mask=mb)
                hh13 = h0 + 13
                w13 = hh13 >= Ho
                hi13 = tl.where(w13, hh13 - Ho, hh13)
                bi13 = tl.where(w13, b0p1, b0)
                sh13 = hi13 - plh
                sh13 = tl.maximum(sh13, 0)
                sh13 = tl.minimum(sh13, Hm1)
                v13 = tl.load(x_ptr + bi13 * HW + sh13 * W + offs, mask=mb)
                hh14 = h0 + 14
                w14 = hh14 >= Ho
                hi14 = tl.where(w14, hh14 - Ho, hh14)
                bi14 = tl.where(w14, b0p1, b0)
                sh14 = hi14 - plh
                sh14 = tl.maximum(sh14, 0)
                sh14 = tl.minimum(sh14, Hm1)
                v14 = tl.load(x_ptr + bi14 * HW + sh14 * W + offs, mask=mb)
                hh15 = h0 + 15
                w15 = hh15 >= Ho
                hi15 = tl.where(w15, hh15 - Ho, hh15)
                bi15 = tl.where(w15, b0p1, b0)
                sh15 = hi15 - plh
                sh15 = tl.maximum(sh15, 0)
                sh15 = tl.minimum(sh15, Hm1)
                v15 = tl.load(x_ptr + bi15 * HW + sh15 * W + offs, mask=mb)
                hh16 = h0 + 16
                w16 = hh16 >= Ho
                hi16 = tl.where(w16, hh16 - Ho, hh16)
                bi16 = tl.where(w16, b0p1, b0)
                sh16 = hi16 - plh
                sh16 = tl.maximum(sh16, 0)
                sh16 = tl.minimum(sh16, Hm1)
                v16 = tl.load(x_ptr + bi16 * HW + sh16 * W + offs, mask=mb)
                hh17 = h0 + 17
                w17 = hh17 >= Ho
                hi17 = tl.where(w17, hh17 - Ho, hh17)
                bi17 = tl.where(w17, b0p1, b0)
                sh17 = hi17 - plh
                sh17 = tl.maximum(sh17, 0)
                sh17 = tl.minimum(sh17, Hm1)
                v17 = tl.load(x_ptr + bi17 * HW + sh17 * W + offs, mask=mb)
                hh18 = h0 + 18
                w18 = hh18 >= Ho
                hi18 = tl.where(w18, hh18 - Ho, hh18)
                bi18 = tl.where(w18, b0p1, b0)
                sh18 = hi18 - plh
                sh18 = tl.maximum(sh18, 0)
                sh18 = tl.minimum(sh18, Hm1)
                v18 = tl.load(x_ptr + bi18 * HW + sh18 * W + offs, mask=mb)
                hh19 = h0 + 19
                w19 = hh19 >= Ho
                hi19 = tl.where(w19, hh19 - Ho, hh19)
                bi19 = tl.where(w19, b0p1, b0)
                sh19 = hi19 - plh
                sh19 = tl.maximum(sh19, 0)
                sh19 = tl.minimum(sh19, Hm1)
                v19 = tl.load(x_ptr + bi19 * HW + sh19 * W + offs, mask=mb)
                hh20 = h0 + 20
                w20 = hh20 >= Ho
                hi20 = tl.where(w20, hh20 - Ho, hh20)
                bi20 = tl.where(w20, b0p1, b0)
                sh20 = hi20 - plh
                sh20 = tl.maximum(sh20, 0)
                sh20 = tl.minimum(sh20, Hm1)
                v20 = tl.load(x_ptr + bi20 * HW + sh20 * W + offs, mask=mb)
                hh21 = h0 + 21
                w21 = hh21 >= Ho
                hi21 = tl.where(w21, hh21 - Ho, hh21)
                bi21 = tl.where(w21, b0p1, b0)
                sh21 = hi21 - plh
                sh21 = tl.maximum(sh21, 0)
                sh21 = tl.minimum(sh21, Hm1)
                v21 = tl.load(x_ptr + bi21 * HW + sh21 * W + offs, mask=mb)
                hh22 = h0 + 22
                w22 = hh22 >= Ho
                hi22 = tl.where(w22, hh22 - Ho, hh22)
                bi22 = tl.where(w22, b0p1, b0)
                sh22 = hi22 - plh
                sh22 = tl.maximum(sh22, 0)
                sh22 = tl.minimum(sh22, Hm1)
                v22 = tl.load(x_ptr + bi22 * HW + sh22 * W + offs, mask=mb)
                hh23 = h0 + 23
                w23 = hh23 >= Ho
                hi23 = tl.where(w23, hh23 - Ho, hh23)
                bi23 = tl.where(w23, b0p1, b0)
                sh23 = hi23 - plh
                sh23 = tl.maximum(sh23, 0)
                sh23 = tl.minimum(sh23, Hm1)
                v23 = tl.load(x_ptr + bi23 * HW + sh23 * W + offs, mask=mb)
                hh24 = h0 + 24
                w24 = hh24 >= Ho
                hi24 = tl.where(w24, hh24 - Ho, hh24)
                bi24 = tl.where(w24, b0p1, b0)
                sh24 = hi24 - plh
                sh24 = tl.maximum(sh24, 0)
                sh24 = tl.minimum(sh24, Hm1)
                v24 = tl.load(x_ptr + bi24 * HW + sh24 * W + offs, mask=mb)
                hh25 = h0 + 25
                w25 = hh25 >= Ho
                hi25 = tl.where(w25, hh25 - Ho, hh25)
                bi25 = tl.where(w25, b0p1, b0)
                sh25 = hi25 - plh
                sh25 = tl.maximum(sh25, 0)
                sh25 = tl.minimum(sh25, Hm1)
                v25 = tl.load(x_ptr + bi25 * HW + sh25 * W + offs, mask=mb)
                hh26 = h0 + 26
                w26 = hh26 >= Ho
                hi26 = tl.where(w26, hh26 - Ho, hh26)
                bi26 = tl.where(w26, b0p1, b0)
                sh26 = hi26 - plh
                sh26 = tl.maximum(sh26, 0)
                sh26 = tl.minimum(sh26, Hm1)
                v26 = tl.load(x_ptr + bi26 * HW + sh26 * W + offs, mask=mb)
                hh27 = h0 + 27
                w27 = hh27 >= Ho
                hi27 = tl.where(w27, hh27 - Ho, hh27)
                bi27 = tl.where(w27, b0p1, b0)
                sh27 = hi27 - plh
                sh27 = tl.maximum(sh27, 0)
                sh27 = tl.minimum(sh27, Hm1)
                v27 = tl.load(x_ptr + bi27 * HW + sh27 * W + offs, mask=mb)
                hh28 = h0 + 28
                w28 = hh28 >= Ho
                hi28 = tl.where(w28, hh28 - Ho, hh28)
                bi28 = tl.where(w28, b0p1, b0)
                sh28 = hi28 - plh
                sh28 = tl.maximum(sh28, 0)
                sh28 = tl.minimum(sh28, Hm1)
                v28 = tl.load(x_ptr + bi28 * HW + sh28 * W + offs, mask=mb)
                hh29 = h0 + 29
                w29 = hh29 >= Ho
                hi29 = tl.where(w29, hh29 - Ho, hh29)
                bi29 = tl.where(w29, b0p1, b0)
                sh29 = hi29 - plh
                sh29 = tl.maximum(sh29, 0)
                sh29 = tl.minimum(sh29, Hm1)
                v29 = tl.load(x_ptr + bi29 * HW + sh29 * W + offs, mask=mb)
                hh30 = h0 + 30
                w30 = hh30 >= Ho
                hi30 = tl.where(w30, hh30 - Ho, hh30)
                bi30 = tl.where(w30, b0p1, b0)
                sh30 = hi30 - plh
                sh30 = tl.maximum(sh30, 0)
                sh30 = tl.minimum(sh30, Hm1)
                v30 = tl.load(x_ptr + bi30 * HW + sh30 * W + offs, mask=mb)
                hh31 = h0 + 31
                w31 = hh31 >= Ho
                hi31 = tl.where(w31, hh31 - Ho, hh31)
                bi31 = tl.where(w31, b0p1, b0)
                sh31 = hi31 - plh
                sh31 = tl.maximum(sh31, 0)
                sh31 = tl.minimum(sh31, Hm1)
                v31 = tl.load(x_ptr + bi31 * HW + sh31 * W + offs, mask=mb)
                tl.store(o_ptr + (r0 + 0) * Wo + plw + offs, v0, mask=mb)
                tl.store(o_ptr + (r0 + 1) * Wo + plw + offs, v1, mask=mb)
                tl.store(o_ptr + (r0 + 2) * Wo + plw + offs, v2, mask=mb)
                tl.store(o_ptr + (r0 + 3) * Wo + plw + offs, v3, mask=mb)
                tl.store(o_ptr + (r0 + 4) * Wo + plw + offs, v4, mask=mb)
                tl.store(o_ptr + (r0 + 5) * Wo + plw + offs, v5, mask=mb)
                tl.store(o_ptr + (r0 + 6) * Wo + plw + offs, v6, mask=mb)
                tl.store(o_ptr + (r0 + 7) * Wo + plw + offs, v7, mask=mb)
                tl.store(o_ptr + (r0 + 8) * Wo + plw + offs, v8, mask=mb)
                tl.store(o_ptr + (r0 + 9) * Wo + plw + offs, v9, mask=mb)
                tl.store(o_ptr + (r0 + 10) * Wo + plw + offs, v10, mask=mb)
                tl.store(o_ptr + (r0 + 11) * Wo + plw + offs, v11, mask=mb)
                tl.store(o_ptr + (r0 + 12) * Wo + plw + offs, v12, mask=mb)
                tl.store(o_ptr + (r0 + 13) * Wo + plw + offs, v13, mask=mb)
                tl.store(o_ptr + (r0 + 14) * Wo + plw + offs, v14, mask=mb)
                tl.store(o_ptr + (r0 + 15) * Wo + plw + offs, v15, mask=mb)
                tl.store(o_ptr + (r0 + 16) * Wo + plw + offs, v16, mask=mb)
                tl.store(o_ptr + (r0 + 17) * Wo + plw + offs, v17, mask=mb)
                tl.store(o_ptr + (r0 + 18) * Wo + plw + offs, v18, mask=mb)
                tl.store(o_ptr + (r0 + 19) * Wo + plw + offs, v19, mask=mb)
                tl.store(o_ptr + (r0 + 20) * Wo + plw + offs, v20, mask=mb)
                tl.store(o_ptr + (r0 + 21) * Wo + plw + offs, v21, mask=mb)
                tl.store(o_ptr + (r0 + 22) * Wo + plw + offs, v22, mask=mb)
                tl.store(o_ptr + (r0 + 23) * Wo + plw + offs, v23, mask=mb)
                tl.store(o_ptr + (r0 + 24) * Wo + plw + offs, v24, mask=mb)
                tl.store(o_ptr + (r0 + 25) * Wo + plw + offs, v25, mask=mb)
                tl.store(o_ptr + (r0 + 26) * Wo + plw + offs, v26, mask=mb)
                tl.store(o_ptr + (r0 + 27) * Wo + plw + offs, v27, mask=mb)
                tl.store(o_ptr + (r0 + 28) * Wo + plw + offs, v28, mask=mb)
                tl.store(o_ptr + (r0 + 29) * Wo + plw + offs, v29, mask=mb)
                tl.store(o_ptr + (r0 + 30) * Wo + plw + offs, v30, mask=mb)
                tl.store(o_ptr + (r0 + 31) * Wo + plw + offs, v31, mask=mb)
            else:
                for i in range(0, U):
                    r_i = tl.minimum(r0 + i, R - 1)
                    v_i = tl.load(
                        x_ptr + _repl_row_src_c(r_i, W, Ho, H, plh) + offs,
                        mask=mb,
                    )
                    tl.store(o_ptr + r_i * Wo + plw + offs, v_i, mask=mb)
    # ---- phase 2: left/right replicate lanes via per-lane gather ----
    total = R * PW
    if PW > 0:
        goffs = tl.arange(0, GBLOCK)
        for t0 in range(pid * GBLOCK, total, G * GBLOCK):
            t = t0 + goffs
            m = t < total
            row = t // PW
            rowc = tl.minimum(row, R - 1)
            j = t - row * PW
            is_left = j < plw
            # left lane j: out col j, src col 0 (replicate first element)
            # right lane j: out col W + j, src col W-1 (replicate last)
            dj = tl.where(is_left, j, W + j)
            sj = tl.where(is_left, 0, W - 1)
            srow = _repl_row_src_c(rowc, W, Ho, H, plh)
            tl.store(
                o_ptr + rowc * Wo + dj,
                tl.load(x_ptr + srow + sj, mask=m),
                mask=m,
            )


@triton.jit
def replication_pad2d_wide_kernel(
    x_ptr,
    o_ptr,
    R,
    W,
    Wo,
    plw,
    Ho,
    H,
    plh,
    BLOCK: tl.constexpr,
    FPAD: tl.constexpr,
):
    # general path for very wide rows (W > 8192): per-row chunked interior
    # copy plus chunked border broadcast; correctness-first, off the hot
    # path.
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    prw = Wo - W - plw
    CB = (W + BLOCK - 1) // BLOCK
    CL = (plw + FPAD - 1) // FPAD
    CR = (prw + FPAD - 1) // FPAD
    offs_b = tl.arange(0, BLOCK)
    offs_f = tl.arange(0, FPAD)
    for r in range(pid, R, G):
        srow = _repl_row_src_rt(r, W, Ho, H, plh)
        drow = r * Wo
        for c in range(0, CB):
            ob = c * BLOCK + offs_b
            m = ob < W
            tl.store(
                o_ptr + drow + plw + ob,
                tl.load(x_ptr + srow + ob, mask=m),
                mask=m,
            )
        if (Wo - W) > 0:
            vl = tl.load(x_ptr + srow)
            vr = tl.load(x_ptr + srow + W - 1)
            for c in range(0, CL):
                ol = c * FPAD + offs_f
                ml = ol < plw
                tl.store(o_ptr + drow + ol, vl, mask=ml)
            for c in range(0, CR):
                orr = c * FPAD + offs_f
                mr = orr < prw
                tl.store(o_ptr + drow + W + plw + orr, vr, mask=mr)


@triton.jit
def replication_pad2d_flat_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    PAD_LEFT,
    PAD_TOP,
    TOTAL_ELEMS,
    BLOCK_SIZE: tl.constexpr,
):
    # int64-safe fallback for outputs beyond int32 index range
    # (>= 2^31 elements): grid-strided flat decode.
    pid = tl.program_id(axis=0)
    nprog = tl.num_programs(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(pid * BLOCK_SIZE, TOTAL_ELEMS, nprog * BLOCK_SIZE):
        offs64 = (base + offs).to(tl.int64)
        mask = offs64 < TOTAL_ELEMS
        OW_i64 = tl.full([1], OW, dtype=tl.int64)
        OH_i64 = tl.full([1], OH, dtype=tl.int64)
        C_i64 = tl.full([1], C, dtype=tl.int64)
        W_i64 = tl.full([1], W, dtype=tl.int64)
        H_i64 = tl.full([1], H, dtype=tl.int64)
        PAD_LEFT_i64 = tl.full([1], PAD_LEFT, dtype=tl.int64)
        PAD_TOP_i64 = tl.full([1], PAD_TOP, dtype=tl.int64)

        ow = offs64 % OW_i64
        tmp = offs64 // OW_i64
        oh = tmp % OH_i64
        tmp = tmp // OH_i64
        c = tmp % C_i64
        n = tmp // C_i64

        ih = oh - PAD_TOP_i64
        iw = ow - PAD_LEFT_i64

        zero = tl.full([1], 0, dtype=tl.int64)
        Hm1 = H_i64 - 1
        Wm1 = W_i64 - 1

        ih = tl.maximum(zero, tl.minimum(Hm1, ih))
        iw = tl.maximum(zero, tl.minimum(Wm1, iw))

        in_index = ((n * C_i64 + c) * H_i64 + ih) * W_i64 + iw
        x = tl.load(in_ptr + in_index, mask=mask)
        tl.store(out_ptr + offs64, x, mask=mask)


def _next_pow2(n, cap):
    p = 1
    while p < n and p < cap:
        p *= 2
    return p


def launch_replication_pad2d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of 4 integers: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Validate padding values
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("replication_pad2d does not support negative padding")

    # Validate input
    if input.dim() != 3 and input.dim() != 4:
        raise ValueError(
            "replication_pad2d expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )
    if input.device.type != flag_gems.device:
        raise ValueError(f"input must be a {flag_gems.device} tensor")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    if H_in <= 0 or W_in <= 0:
        raise ValueError(
            "Input height and width must be greater than 0 for replication padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        if out.device.type != flag_gems.device:
            raise ValueError(f"out must be a {flag_gems.device} tensor")
        expected_shape = (*leading_shape, H_out, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"Provided out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError("Input and out must have the same dtype")
        if out.device != x.device:
            raise ValueError("Input and out must be on the same device")
        if not out.is_contiguous():
            raise ValueError("Only contiguous out tensors are supported")

    R = B * H_out
    PW = W_out - W_in
    pad_right_w = PW - pad_left
    total_out = out.numel()
    if total_out == 0:
        return out

    def _launch():
        if total_out >= 2**31 - 4096:
            # int64-safe flat fallback (off any realistic path)
            if input.dim() == 4:
                kernel_N, kernel_C = int(x.shape[0]), int(x.shape[1])
            else:
                kernel_N, kernel_C = 1, B
            grid = (min(48, triton.cdiv(total_out, 1024)),)
            replication_pad2d_flat_kernel[grid](
                x,
                out,
                kernel_N,
                kernel_C,
                H_in,
                W_in,
                H_out,
                W_out,
                pad_left,
                pad_top,
                total_out,
                BLOCK_SIZE=1024,
                num_warps=4,
            )
        elif W_in > 8192 or H_out < 4:
            # very wide rows / degenerate Ho: chunked general path
            FPAD = max(16, _next_pow2(max(pad_left, pad_right_w, 1), 1024))
            grid = (min(48, max(1, R)),)
            replication_pad2d_wide_kernel[grid](
                x,
                out,
                R,
                W_in,
                W_out,
                pad_left,
                H_out,
                H_in,
                pad_top,
                BLOCK=8192,
                FPAD=FPAD,
                num_warps=4,
            )
        else:
            # rows per group: wide rows are bandwidth-bound (small groups),
            # narrow rows are latency-bound (large groups hide the ~420ns
            # per-group latency). num_warps and the U tier are coupled.
            # 2026-11-15 retune (interleaved A/B, min-of-3 rounds, same-process
            # aten baseline): the [17, 768) band moves from U=20/nw=1 to
            # U=8/nw=2 — official bench geomean 1.0694 -> 1.1097 (+3.8%), and a
            # dedicated W sweep (17/32/128/256/512/700) 0.9421 -> 0.9759
            # (+3.6%); only the very narrow W=17 probe regressed (-19%, 12us
            # case dominated by launch overhead). U=4/nw=4 (-35%) and
            # U=20/nw=1 on W<=16 (-5%) were both rejected.
            # Small R clamp: when fewer groups than programs would result,
            # prefer more programs with smaller groups.
            if W_in >= 2048:
                U, nw = 4, 4
            elif W_in >= 768:
                U, nw = 8, 4
            elif W_in >= 256:
                U, nw = 32, 2
            elif W_in >= 32:
                U, nw = 32, 1
            else:
                U, nw = 16, 1
            # single image wrap per group requires U <= Ho; snap down to the
            # compiled tiers
            U = min(U, H_out)
            if U >= 32:
                U = 32
            elif U >= 16:
                U = 16
            elif U >= 8:
                U = 8
            else:
                U = 4
            # Small R clamp: when fewer groups than programs would result,
            # prefer more programs with smaller groups.
            n_min = (R + 47) // 48
            if n_min <= 4:
                U = min(U, 4)
                if nw < 4:
                    nw = 4
            BLOCK = _next_pow2(W_in, 8192)
            total_pad = R * PW
            grid = (min(48, max(1, (R + U - 1) // U + (total_pad + 1023) // 1024)),)
            replication_pad2d_rowseg_v5_kernel[grid](
                x,
                out,
                R,
                W=W_in,
                Wo=W_out,
                plw=pad_left,
                PW=PW,
                Ho=H_out,
                H=H_in,
                plh=pad_top,
                BLOCK=BLOCK,
                GBLOCK=1024,
                U=U,
                num_warps=nw,
            )

    # cold compiles hit the documented node-level toolchain flakiness
    # (P02 family): a fresh-compile subprocess can die before producing a
    # cache entry; retrying the launch re-enters compilation and succeeds
    # (reflection_pad2d/_pad_circular precedent). Zero cost on the warm
    # path.
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                _launch()
            break
        except Exception:
            if attempt == 2:
                raise
    return out


def replication_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D")
    return launch_replication_pad2d(input, padding, out=None)


def replication_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_OUT")
    return launch_replication_pad2d(input, padding, out=out)
