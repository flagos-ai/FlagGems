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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.tensor_wrapper import StridedBuffer

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# GCU400 flip strategy (2026-11-11 perf rewrite):
# The generic implementation hands a negative-stride StridedBuffer to the
# pointwise_dynamic codegen.  Negative strides trip `FlagOfNotUseDMA`, so the
# copy runs through the notDMA tile heuristic and a multi-dim
# `make_block_ptr` with negative constexpr strides plus boundary_check on
# every dim.  On mixed-sign layouts (flip of outer dims, innermost kept --
# e.g. (64,512,512) dims (0,1)) that lowering scalarises into a per-element
# gather: 17.2 ms / 1315 ms on the two 3D core shapes (0.006x / 0.005x vs
# torch native), which is what dragged the flip geomean to 0.628x on libra.
#
# This rewrite keeps one kernel launch and never builds a negative-stride
# block pointer.  The host folds the flip into *signed strides + a base
# offset* (the mirrored view), simplifies dims (drop size-1, merge
# contiguous same-sign neighbours), and dispatches on the simplified shape:
#   * reversal: all dims merged into one stride -1 run (1D flip, or a full
#     flip of a contiguous tensor) -> one flat descending block-ptr copy,
#     the exact construct the generated rank-1 kernel already runs fast
#     (2.5 TB/s fp32); boundary_check only when N is not tile-divisible.
#   * rowseg: innermost merged dim is an unflipped stride-1 run (W) and the
#     outer dims are the mirrored part -> rows are copied as flat
#     `src_row + tl.arange` pairs, U rows per group with all loads issued
#     before all stores (reflection_pad2d row-group form; hides the ~450ns
#     per-op latency on narrow rows).  Row sources are affine in the merged
#     outer index for the common merged cases (e.g. (64,512,512) dims(0,1)
#     collapses to one outer dim of stride -512).
#   * gather: everything else with a non-reversed tail (odd widths,
#     non-dense/dilated inputs, permuted-dense outputs) -> flat 1D walk in
#     output memory order (linear stores), per-lane `%`/`//` decode with
#     signed strides.  Correctness net; index decode uses the rem form
#     (division-decomposed remainders mis-lower, see compiler_traps).
#   * pointwise fallback: reversed-tail multi-dim cases (flip of the
#     innermost dim only, where the old bptr rank-2/rank-1 path with inner
#     stride -1 is already serviceable: 0.47x-1.2x), rank > 5, and bool.
#   * 8-byte dtypes (int64/fp64) never touch an 8-byte pointer: GCU400
#     mis-lowers i64-pointer loads whose offsets are not purely
#     program-id-derived (scalar offsets apply twice -- 8B pointer
#     arithmetic over the default 4B/element physical storage) and the
#     negative-stride block pointer returns garbage for 8-byte types.
#     They run the same kernels over a same-shape int32 word view (set_
#     over the same storage; the numel-doubled word view is falsified,
#     see perf_playbook int64 payload 双视图归一).
# Grids are capped at 48 (one program per SPC) with grid-stride loops; the
# 2D-tile forms are deliberately avoided (HC_PAD/LANES unmasked 2D tile
# silent-corruption family).

_INT32_LIMIT = 2**31 - 1
_MAX_ROW_WIDTH = 8192
_GATHER_BLOCK = 2048
_NUM_SPC = 48


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


@triton.jit
def _flip_reverse_kernel(
    x_ptr,
    o_ptr,
    N,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # out[i] = x[N-1-i]: descending block pointer anchored at the last
    # element (the generated rank-1 kernel gets this base from the
    # pre-offset StridedBuffer), linear ascending store.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(N, BLOCK)
    base = x_ptr + (N - 1)
    for tile in range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        src = tl.make_block_ptr(
            base, (N,), (-1,), (tile * BLOCK,), (BLOCK,), order=(0,)
        )
        if EVEN:
            v = tl.load(src)
            tl.store(o_ptr + offs, v)
        else:
            v = tl.load(src, boundary_check=(0,))
            tl.store(o_ptr + offs, v, mask=offs < N)


@triton.jit
def _flip_row_src(r, s0, s1, s2, s3, t0, t1, t2, t3, in_off):
    # element offset of output row r's source row: decode the merged outer
    # index (rem form) and apply the signed strides + mirrored base.
    i3 = r % s3
    r = r // s3
    i2 = r % s2
    r = r // s2
    i1 = r % s1
    r = r // s1
    i0 = r % s0
    return in_off + i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3


@triton.jit
def _flip_rowseg_kernel(
    x_ptr,
    o_ptr,
    R,
    W,
    s0,
    s1,
    s2,
    s3,
    t0,
    t1,
    t2,
    t3,
    in_off,
    BLOCK: tl.constexpr,
    U: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Copy R rows of W elements; row r loads from the mirrored source row
    # (scalar base) and stores contiguously.  U rows per group with all
    # loads issued before all stores (hides the ~450ns per-op latency that
    # serializes narrow rows -- U=4 -> 16 gains 2.0-2.2x on the 3D core
    # shapes).  Surplus rows of the last group are clamped onto the last
    # real row: the duplicate store rewrites identical data.  NOTE: the
    # group bodies stay hand-split -- batching them through a
    # tuple-returning jit helper mis-compiles non-deterministically on
    # this backend once more than one group call is inlined.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    mb = offs < W
    GU = num_ctas * U
    r_last = R - 1
    for r0 in range(pid * U, R, GU):
        r1 = tl.minimum(r0 + 1, r_last)
        r2 = tl.minimum(r0 + 2, r_last)
        r3 = tl.minimum(r0 + 3, r_last)
        s_0 = _flip_row_src(r0, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
        s_1 = _flip_row_src(r1, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
        s_2 = _flip_row_src(r2, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
        s_3 = _flip_row_src(r3, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
        if EVEN:
            if U == 1:
                v0 = tl.load(x_ptr + s_0 + offs)
                tl.store(o_ptr + r0 * W + offs, v0)
            elif U == 4:
                v0 = tl.load(x_ptr + s_0 + offs)
                v1 = tl.load(x_ptr + s_1 + offs)
                v2 = tl.load(x_ptr + s_2 + offs)
                v3 = tl.load(x_ptr + s_3 + offs)
                tl.store(o_ptr + r0 * W + offs, v0)
                tl.store(o_ptr + r1 * W + offs, v1)
                tl.store(o_ptr + r2 * W + offs, v2)
                tl.store(o_ptr + r3 * W + offs, v3)
            elif U == 8:
                r4 = tl.minimum(r0 + 4, r_last)
                r5 = tl.minimum(r0 + 5, r_last)
                r6 = tl.minimum(r0 + 6, r_last)
                r7 = tl.minimum(r0 + 7, r_last)
                s_4 = _flip_row_src(r4, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_5 = _flip_row_src(r5, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_6 = _flip_row_src(r6, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_7 = _flip_row_src(r7, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                v0 = tl.load(x_ptr + s_0 + offs)
                v1 = tl.load(x_ptr + s_1 + offs)
                v2 = tl.load(x_ptr + s_2 + offs)
                v3 = tl.load(x_ptr + s_3 + offs)
                v4 = tl.load(x_ptr + s_4 + offs)
                v5 = tl.load(x_ptr + s_5 + offs)
                v6 = tl.load(x_ptr + s_6 + offs)
                v7 = tl.load(x_ptr + s_7 + offs)
                tl.store(o_ptr + r0 * W + offs, v0)
                tl.store(o_ptr + r1 * W + offs, v1)
                tl.store(o_ptr + r2 * W + offs, v2)
                tl.store(o_ptr + r3 * W + offs, v3)
                tl.store(o_ptr + r4 * W + offs, v4)
                tl.store(o_ptr + r5 * W + offs, v5)
                tl.store(o_ptr + r6 * W + offs, v6)
                tl.store(o_ptr + r7 * W + offs, v7)
            else:
                r4 = tl.minimum(r0 + 4, r_last)
                r5 = tl.minimum(r0 + 5, r_last)
                r6 = tl.minimum(r0 + 6, r_last)
                r7 = tl.minimum(r0 + 7, r_last)
                r8 = tl.minimum(r0 + 8, r_last)
                r9 = tl.minimum(r0 + 9, r_last)
                r10 = tl.minimum(r0 + 10, r_last)
                r11 = tl.minimum(r0 + 11, r_last)
                r12 = tl.minimum(r0 + 12, r_last)
                r13 = tl.minimum(r0 + 13, r_last)
                r14 = tl.minimum(r0 + 14, r_last)
                r15 = tl.minimum(r0 + 15, r_last)
                s_4 = _flip_row_src(r4, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_5 = _flip_row_src(r5, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_6 = _flip_row_src(r6, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_7 = _flip_row_src(r7, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_8 = _flip_row_src(r8, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_9 = _flip_row_src(r9, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_10 = _flip_row_src(r10, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_11 = _flip_row_src(r11, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_12 = _flip_row_src(r12, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_13 = _flip_row_src(r13, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_14 = _flip_row_src(r14, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_15 = _flip_row_src(r15, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                v0 = tl.load(x_ptr + s_0 + offs)
                v1 = tl.load(x_ptr + s_1 + offs)
                v2 = tl.load(x_ptr + s_2 + offs)
                v3 = tl.load(x_ptr + s_3 + offs)
                v4 = tl.load(x_ptr + s_4 + offs)
                v5 = tl.load(x_ptr + s_5 + offs)
                v6 = tl.load(x_ptr + s_6 + offs)
                v7 = tl.load(x_ptr + s_7 + offs)
                v8 = tl.load(x_ptr + s_8 + offs)
                v9 = tl.load(x_ptr + s_9 + offs)
                v10 = tl.load(x_ptr + s_10 + offs)
                v11 = tl.load(x_ptr + s_11 + offs)
                v12 = tl.load(x_ptr + s_12 + offs)
                v13 = tl.load(x_ptr + s_13 + offs)
                v14 = tl.load(x_ptr + s_14 + offs)
                v15 = tl.load(x_ptr + s_15 + offs)
                tl.store(o_ptr + r0 * W + offs, v0)
                tl.store(o_ptr + r1 * W + offs, v1)
                tl.store(o_ptr + r2 * W + offs, v2)
                tl.store(o_ptr + r3 * W + offs, v3)
                tl.store(o_ptr + r4 * W + offs, v4)
                tl.store(o_ptr + r5 * W + offs, v5)
                tl.store(o_ptr + r6 * W + offs, v6)
                tl.store(o_ptr + r7 * W + offs, v7)
                tl.store(o_ptr + r8 * W + offs, v8)
                tl.store(o_ptr + r9 * W + offs, v9)
                tl.store(o_ptr + r10 * W + offs, v10)
                tl.store(o_ptr + r11 * W + offs, v11)
                tl.store(o_ptr + r12 * W + offs, v12)
                tl.store(o_ptr + r13 * W + offs, v13)
                tl.store(o_ptr + r14 * W + offs, v14)
                tl.store(o_ptr + r15 * W + offs, v15)
        else:
            if U == 1:
                v0 = tl.load(x_ptr + s_0 + offs, mask=mb)
                tl.store(o_ptr + r0 * W + offs, v0, mask=mb)
            elif U == 4:
                v0 = tl.load(x_ptr + s_0 + offs, mask=mb)
                v1 = tl.load(x_ptr + s_1 + offs, mask=mb)
                v2 = tl.load(x_ptr + s_2 + offs, mask=mb)
                v3 = tl.load(x_ptr + s_3 + offs, mask=mb)
                tl.store(o_ptr + r0 * W + offs, v0, mask=mb)
                tl.store(o_ptr + r1 * W + offs, v1, mask=mb)
                tl.store(o_ptr + r2 * W + offs, v2, mask=mb)
                tl.store(o_ptr + r3 * W + offs, v3, mask=mb)
            elif U == 8:
                r4 = tl.minimum(r0 + 4, r_last)
                r5 = tl.minimum(r0 + 5, r_last)
                r6 = tl.minimum(r0 + 6, r_last)
                r7 = tl.minimum(r0 + 7, r_last)
                s_4 = _flip_row_src(r4, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_5 = _flip_row_src(r5, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_6 = _flip_row_src(r6, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_7 = _flip_row_src(r7, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                v0 = tl.load(x_ptr + s_0 + offs, mask=mb)
                v1 = tl.load(x_ptr + s_1 + offs, mask=mb)
                v2 = tl.load(x_ptr + s_2 + offs, mask=mb)
                v3 = tl.load(x_ptr + s_3 + offs, mask=mb)
                v4 = tl.load(x_ptr + s_4 + offs, mask=mb)
                v5 = tl.load(x_ptr + s_5 + offs, mask=mb)
                v6 = tl.load(x_ptr + s_6 + offs, mask=mb)
                v7 = tl.load(x_ptr + s_7 + offs, mask=mb)
                tl.store(o_ptr + r0 * W + offs, v0, mask=mb)
                tl.store(o_ptr + r1 * W + offs, v1, mask=mb)
                tl.store(o_ptr + r2 * W + offs, v2, mask=mb)
                tl.store(o_ptr + r3 * W + offs, v3, mask=mb)
                tl.store(o_ptr + r4 * W + offs, v4, mask=mb)
                tl.store(o_ptr + r5 * W + offs, v5, mask=mb)
                tl.store(o_ptr + r6 * W + offs, v6, mask=mb)
                tl.store(o_ptr + r7 * W + offs, v7, mask=mb)
            else:
                r4 = tl.minimum(r0 + 4, r_last)
                r5 = tl.minimum(r0 + 5, r_last)
                r6 = tl.minimum(r0 + 6, r_last)
                r7 = tl.minimum(r0 + 7, r_last)
                r8 = tl.minimum(r0 + 8, r_last)
                r9 = tl.minimum(r0 + 9, r_last)
                r10 = tl.minimum(r0 + 10, r_last)
                r11 = tl.minimum(r0 + 11, r_last)
                r12 = tl.minimum(r0 + 12, r_last)
                r13 = tl.minimum(r0 + 13, r_last)
                r14 = tl.minimum(r0 + 14, r_last)
                r15 = tl.minimum(r0 + 15, r_last)
                s_4 = _flip_row_src(r4, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_5 = _flip_row_src(r5, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_6 = _flip_row_src(r6, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_7 = _flip_row_src(r7, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_8 = _flip_row_src(r8, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_9 = _flip_row_src(r9, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_10 = _flip_row_src(r10, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_11 = _flip_row_src(r11, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_12 = _flip_row_src(r12, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_13 = _flip_row_src(r13, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_14 = _flip_row_src(r14, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                s_15 = _flip_row_src(r15, s0, s1, s2, s3, t0, t1, t2, t3, in_off)
                v0 = tl.load(x_ptr + s_0 + offs, mask=mb)
                v1 = tl.load(x_ptr + s_1 + offs, mask=mb)
                v2 = tl.load(x_ptr + s_2 + offs, mask=mb)
                v3 = tl.load(x_ptr + s_3 + offs, mask=mb)
                v4 = tl.load(x_ptr + s_4 + offs, mask=mb)
                v5 = tl.load(x_ptr + s_5 + offs, mask=mb)
                v6 = tl.load(x_ptr + s_6 + offs, mask=mb)
                v7 = tl.load(x_ptr + s_7 + offs, mask=mb)
                v8 = tl.load(x_ptr + s_8 + offs, mask=mb)
                v9 = tl.load(x_ptr + s_9 + offs, mask=mb)
                v10 = tl.load(x_ptr + s_10 + offs, mask=mb)
                v11 = tl.load(x_ptr + s_11 + offs, mask=mb)
                v12 = tl.load(x_ptr + s_12 + offs, mask=mb)
                v13 = tl.load(x_ptr + s_13 + offs, mask=mb)
                v14 = tl.load(x_ptr + s_14 + offs, mask=mb)
                v15 = tl.load(x_ptr + s_15 + offs, mask=mb)
                tl.store(o_ptr + r0 * W + offs, v0, mask=mb)
                tl.store(o_ptr + r1 * W + offs, v1, mask=mb)
                tl.store(o_ptr + r2 * W + offs, v2, mask=mb)
                tl.store(o_ptr + r3 * W + offs, v3, mask=mb)
                tl.store(o_ptr + r4 * W + offs, v4, mask=mb)
                tl.store(o_ptr + r5 * W + offs, v5, mask=mb)
                tl.store(o_ptr + r6 * W + offs, v6, mask=mb)
                tl.store(o_ptr + r7 * W + offs, v7, mask=mb)
                tl.store(o_ptr + r8 * W + offs, v8, mask=mb)
                tl.store(o_ptr + r9 * W + offs, v9, mask=mb)
                tl.store(o_ptr + r10 * W + offs, v10, mask=mb)
                tl.store(o_ptr + r11 * W + offs, v11, mask=mb)
                tl.store(o_ptr + r12 * W + offs, v12, mask=mb)
                tl.store(o_ptr + r13 * W + offs, v13, mask=mb)
                tl.store(o_ptr + r14 * W + offs, v14, mask=mb)
                tl.store(o_ptr + r15 * W + offs, v15, mask=mb)


@triton.jit
def _flip_gather_kernel(
    x_ptr,
    o_ptr,
    N,
    s0,
    s1,
    s2,
    s3,
    s4,
    t0,
    t1,
    t2,
    t3,
    t4,
    in_off,
    BLOCK: tl.constexpr,
):
    # Universal fallback for non-reversed tails: walk the output linearly
    # (memory order), decode each element's multi-index (rem form) and load
    # from the mirrored source address.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for b0 in range(pid * BLOCK, N, num_ctas * BLOCK):
        offs = b0 + tl.arange(0, BLOCK)
        m = offs < N
        i4 = offs % s4
        tmp = offs // s4
        i3 = tmp % s3
        tmp = tmp // s3
        i2 = tmp % s2
        tmp = tmp // s2
        i1 = tmp % s1
        tmp = tmp // s1
        i0 = tmp % s0
        src = in_off + i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4
        v = tl.load(x_ptr + src, mask=m)
        tl.store(o_ptr + offs, v, mask=m)


def _simplify_dims(shape, strides):
    # Drop size-1 dims (no position consumed) and merge contiguous
    # same-sign neighbours (t_prev == t_cur * s_cur); flip merges across
    # same-sign chains, so mirrored outer dims collapse into one affine
    # stride whenever the layout allows.
    dims = [(s, t) for s, t in zip(shape, strides) if s != 1]
    if not dims:
        dims = [(1, strides[0] if strides else 0)]
    merged = [list(dims[0])]
    for size, stride in dims[1:]:
        prev_size, prev_stride = merged[-1]
        if prev_stride == stride * size:
            merged[-1] = [prev_size * size, stride]
        else:
            merged.append([size, stride])
    sizes = [m[0] for m in merged]
    t_strides = [m[1] for m in merged]
    return sizes, t_strides


def _pad_outer(sizes, t_strides, count):
    padded_s = list(sizes) + [1] * (count - len(sizes))
    padded_t = list(t_strides) + [0] * (count - len(t_strides))
    return padded_s, padded_t


def _launch_rowseg(A, out, sizes, t_strides, in_off, numel):
    W = sizes[-1]
    R = numel // W
    outer_s, outer_t = _pad_outer(sizes[:-1], t_strides[:-1], 4)
    BLOCK = triton.next_power_of_2(W)
    if BLOCK > _MAX_ROW_WIDTH:
        return False
    EVEN = W % BLOCK == 0
    # Row-group size (sweep on the 3D core shapes: U=4 -> 16 gains
    # 2.0-2.2x; U=32 adds only ~13% on one 2-byte shape and the flat
    # hand-split bodies stop at 16).  DSM guard keeps U * W * itemsize
    # within 128KB; tiny row counts shrink U so the grid still spreads
    # over the SPCs.
    U = 16
    while U > 1 and U * W * A.element_size() > 128 * 1024:
        U //= 2
    while U > 1 and triton.cdiv(R, U) < _NUM_SPC:
        U = 1 if U == 4 else U // 2
    grid = (min(_NUM_SPC, triton.cdiv(R, U)),)
    with torch_device_fn.device(A.device):
        _flip_rowseg_kernel[grid](
            A,
            out,
            R,
            W,
            outer_s[0],
            outer_s[1],
            outer_s[2],
            outer_s[3],
            outer_t[0],
            outer_t[1],
            outer_t[2],
            outer_t[3],
            in_off,
            BLOCK=BLOCK,
            U=U,
            EVEN=EVEN,
            num_warps=4,
        )
    return True


def _launch_gather(A, out, sizes, t_strides, in_off, numel):
    padded_s, padded_t = _pad_outer(sizes, t_strides, 5)
    grid = (min(_NUM_SPC, triton.cdiv(numel, _GATHER_BLOCK)),)
    with torch_device_fn.device(A.device):
        _flip_gather_kernel[grid](
            A,
            out,
            numel,
            padded_s[0],
            padded_s[1],
            padded_s[2],
            padded_s[3],
            padded_s[4],
            padded_t[0],
            padded_t[1],
            padded_t[2],
            padded_t[3],
            padded_t[4],
            in_off,
            BLOCK=_GATHER_BLOCK,
            num_warps=4,
        )


def _flip_pointwise_fallback(A, strides, offset):
    out = torch.empty_like(A)
    flipped_A = StridedBuffer(A, strides=strides, offset=offset)
    overload = copy_func.instantiate(A.ndim)
    overload(flipped_A, out0=out)
    return out


def _int32_word_view(A):
    # int64/f64 tensors are physically 4B/element on GCU unless
    # TORCH_GCU_ENABLE_INT64_AND_UINT64 turns on real 8B storage; a
    # same-shape / same-stride / same-offset int32 set_ view aliases those
    # words 1:1 (the numel-doubled "word view" is a falsified form -- see
    # perf_playbook int64 payload 双视图归一).  flip is a pure permutation,
    # so moving the physical words moves the elements exactly.
    view = torch.empty(0, dtype=torch.int32, device=A.device)
    view.set_(
        A.untyped_storage(),
        A.storage_offset(),
        tuple(A.shape),
        tuple(A.stride()),
    )
    return view


def _signed_view(A, flip_flags):
    # strides negated on the flipped dims plus the mirrored base offset:
    # element (i_0..i_{k-1}) of the flipped view sits at
    # offset + sum(i_d * t_d) of A's storage.
    strides = list(A.stride())
    offset = 0
    for i, flipped in enumerate(flip_flags):
        if flipped and A.size(i) > 1 and A.stride(i) != 0:
            offset += strides[i] * (A.shape[i] - 1)
            strides[i] = -strides[i]
    return strides, offset


def _try_fast(A, out, flip_flags):
    # returns True when one of the GCU fast arms handled the copy
    numel = A.numel()
    if numel > _INT32_LIMIT:
        return False
    strides, offset = _signed_view(A, flip_flags)
    sizes, t_strides = _simplify_dims(A.shape, strides)
    rank = len(sizes)

    if rank == 1 and t_strides[0] == -1 and out.is_contiguous():
        # full reversal of a contiguous run; BLOCK sweep showed no signal
        # for 2-byte at 8192/16384/32768 and a regression for 4-byte above
        # 8192, so one block size serves both.
        block = 8192
        if numel < block:
            block = triton.next_power_of_2(numel)
        grid = (min(_NUM_SPC, triton.cdiv(numel, block)),)
        with torch_device_fn.device(A.device):
            _flip_reverse_kernel[grid](
                A,
                out,
                numel,
                BLOCK=block,
                EVEN=numel % block == 0,
                num_warps=4,
            )
        return True
    if (
        rank >= 2
        and t_strides[-1] == 1
        and out.is_contiguous()
        and rank <= 5
        and _launch_rowseg(A, out, sizes, t_strides, offset, numel)
    ):
        return True
    if rank <= 5:
        if out.is_contiguous():
            _launch_gather(A, out, sizes, t_strides, offset, numel)
            return True
        # permuted-dense output: walk it in memory order so stores stay
        # linear; rebuild the decode in that order.
        if A.dim() <= 5:
            perm = sorted(range(A.dim()), key=lambda d: (-out.stride(d), d))
            perm_sizes = [A.shape[d] for d in perm]
            perm_t = [strides[d] for d in perm]
            _launch_gather(A, out, perm_sizes, perm_t, offset, numel)
            return True
    return False


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME FLIP")
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        assert not flip_dims_b[
            dim
        ], "dim {} appears multiple times in the list of dims".format(dim)
        flip_dims_b[dim] = True
    n = 0
    for flipped in flip_dims_b:
        if flipped:
            n += 1
    if n == 0 or A.numel() <= 1:
        return A.clone()

    if A.device.type != "gcu":
        strides, offset = _signed_view(A, flip_dims_b)
        return _flip_pointwise_fallback(A, strides, offset)

    if A.element_size() == 8:
        # i64/f64 pointers mis-lower on GCU400 whenever the pointer offset
        # is not purely program-id-derived (measured: scalar offsets apply
        # twice -- 8B-element pointer arithmetic over 4B-physical storage),
        # and the negative-stride block pointer returns garbage for 8-byte
        # types (fp64 additionally crashes the compiler with an mlir
        # VectorType cast assert).  With the default 4B/element physical
        # storage the same-shape int32 word view aliases the tensor 1:1 and
        # flip's permutation runs on the int32 kernels directly.
        if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1":
            strides, offset = _signed_view(A, flip_dims_b)
            return _flip_pointwise_fallback(A, strides, offset)
        # same-shape word views are only safe at storage_offset 0
        # (offset > 0 slices land at half the byte offset -- compiler_traps
        # "fake-int64 切片/字视图" entry), so keep those on the bptr path.
        if A.storage_offset() != 0:
            strides, offset = _signed_view(A, flip_dims_b)
            return _flip_pointwise_fallback(A, strides, offset)
        try:
            words = _int32_word_view(A)
        except (RuntimeError, ValueError):
            strides, offset = _signed_view(A, flip_dims_b)
            return _flip_pointwise_fallback(A, strides, offset)
        out = torch.empty_like(A)
        out_words = _int32_word_view(out)
        if _try_fast(words, out_words, flip_dims_b):
            return out
        strides, offset = _signed_view(A, flip_dims_b)
        return _flip_pointwise_fallback(A, strides, offset)

    if A.dtype == torch.bool:
        # the bptr codegen carries a bool-specific workaround; keep bools
        # on that path instead of the plain-load kernels.
        strides, offset = _signed_view(A, flip_dims_b)
        return _flip_pointwise_fallback(A, strides, offset)

    out = torch.empty_like(A)
    if _try_fast(A, out, flip_dims_b):
        return out
    strides, offset = _signed_view(A, flip_dims_b)
    return _flip_pointwise_fallback(A, strides, offset)
