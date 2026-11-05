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

# Kunlunxin (XPU) specialization of aten::_upsample_bicubic2d_aa_backward.
#
# !!! STATUS 2026-08-31: CANDIDATE, **NOT WIRED AND NOT PERFORMANCE-VALIDATED**.
# This module is deliberately *not* imported from
# `runtime/backend/_kunlunxin/ops/__init__.py`, so `SpecOpRegistrar` cannot see
# it and the operator still runs the generic implementation.  Do not wire it in
# before the open items in
# `harness/solution/performance/upsample_bicubic2d_aa_backward_xpu3_20260831.md`
# are closed:
#   * `_elem_cap()` (the ~64 KiB live-vector budget guard) is derived from a
#     single fault observation and has not been swept.  The un-guarded version
#     wedged XPU 3 with `reason[4] load/store operation exceed memory size`
#     inside `_aa_w_gather_kernel` at BLK=4096 x 14 taps.
#   * no A/B latency measurement exists yet, so it is unknown whether this is
#     faster than the generic implementation on the official matrix.
# What *is* validated (XPU 3, 2026-08-31) is numerical correctness: 21 shapes
# against an ATen CPU float64 oracle, worst relative error 7.5e-7 (same order as
# the generic implementation), plus three levels of out-of-bounds-write canaries
# clean on 15 shapes x 3 pad sizes.
#
# Why the generic implementation is slow here (measured, XPU3 2026-08-31):
#   * Its fused kernel re-evaluates the Keys-cubic antialias filter inside the
#     kernel for every tap, costing MAX_OW * (KW + MAX_OH * (KH + 1)) predicated
#     ALU ops per input element -- 414 for the 2x benchmark shapes.  Predicated
#     selects are the most expensive primitive on this backend.
#   * Both of its paths launch one program per (nc, row) with a tile of only
#     min(next_pow2(W_in), 256) lanes, so the large-NC benchmark shapes launch
#     10-21 million programs whose tiles are 128-256 bytes wide, and the
#     per-program cost on this backend is set by the tile byte width.
#
# This specialization instead
#   * precomputes both separable, already-normalized weight axes on the host in
#     an *input-centric* layout T[j, i] / start[i], where tap j of input index i
#     is output index start[i] + j and start[i] + MAXJ - 1 is always a legal
#     output index.  Every kernel access is therefore in bounds by construction:
#     no mask, no `other=`, no runtime clamp anywhere.
#   * pre-tiles those tables to the kernel block width so the weight loads are
#     contiguous, and
#   * uses a single flat 1D tile of BLK >= 64 contiguous destination elements
#     per program.  A 2D tile whose row pitch is W_in < 64 is *not* usable on
#     this backend: a vector store always writes 64 contiguous elements, so a
#     narrow-pitch 2D store clobbers the following rows (measured: relative
#     error up to 2.2e3 for W_in in {4, 8, 16, 32}, exact for W_in >= 64).
#
# Correctness anchors: the table layout is bit-exact against the dense weight
# matrix, and the whole separable model matches ATen CPU float64 to <= 1.2e-15
# over 20 shapes including H_out/W_out == 1, size-1 input dims, both
# align_corners settings and non-integer scales.

import logging
import math
from collections import OrderedDict

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Elements per program.  Two live float32 vectors (accumulator + freshly loaded
# tile) must stay well inside this backend's ~64 KiB live-vector budget.
_TILE_ELEMS = 4096
# A vector store touches 64 contiguous elements, so a narrower block is unsafe.
_MIN_BLK = 64
# Widest inner row we accept (power of two).
_MAX_ROW_W = 4096
# static_range unroll depth cap for the tap loop.
_MAX_TAPS = 32
# Cap on the pre-tiled weight table size (float32 elements).
_MAX_TABLE_ELEMS = 16 * 1024 * 1024
# Live-vector byte budget for a fully unrolled tap loop; see `_elem_cap`.
# Swept upward on XPU3 2026-08-31 over the union of the benchmark and accuracy
# shape matrices (every distinct (BLK, taps) pair, both kernels, fp32 + fp16,
# checked against the ATen CPU float64 reference, with a pure-ATen canary before
# and after each step): 32 / 48 / 64 / 96 / 128 / 192 KiB all PASS.  The known
# fault point is BLK=4096 x 14 taps == 256 KiB in this accounting, so the fault
# threshold lies in (192, 256] KiB.  Production is set one step below the last
# swept-good value, i.e. 128 KiB, leaving a 2x margin.
_LIVE_VEC_BUDGET_BYTES = 128 * 1024

_TABLE_CACHE_MAX = 256
_axis_cache = OrderedDict()
_tiled_w_cache = OrderedDict()
_tiled_h_cache = OrderedDict()


@triton.jit
def _aa_w_gather_kernel(
    src_ptr,
    dst_ptr,
    twg_ptr,
    twsg_ptr,
    SRC_W: tl.constexpr,
    LOG2X: tl.constexpr,
    G: tl.constexpr,
    BLK: tl.constexpr,
    MAXJ: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.arange(0, BLK)
    r = (pid * G + (t >> LOG2X)).to(tl.int64)
    ows = tl.load(twsg_ptr + t)
    base = r * SRC_W + ows.to(tl.int64)
    acc = tl.zeros([BLK], dtype=tl.float32)
    for j in tl.static_range(MAXJ):
        w = tl.load(twg_ptr + j * BLK + t)
        g = tl.load(src_ptr + base + j)
        acc += w * g.to(tl.float32)
    tl.store(dst_ptr + pid.to(tl.int64) * BLK + t, acc)


@triton.jit
def _aa_h_select_kernel(
    src_ptr,
    dst_ptr,
    thg_ptr,
    thsg_ptr,
    H_OUT: tl.constexpr,
    X: tl.constexpr,
    NB: tl.constexpr,
    M: tl.constexpr,
    LOG2PH: tl.constexpr,
    BLK: tl.constexpr,
    MAXJ: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.arange(0, BLK)
    ib = pid % NB
    nc = (pid // NB) * M + (t >> LOG2PH)
    ohs = tl.load(thsg_ptr + ib * BLK + t)
    iw = t & (X - 1)
    base = (nc.to(tl.int64) * H_OUT + ohs.to(tl.int64)) * X + iw.to(tl.int64)
    acc = tl.zeros([BLK], dtype=tl.float32)
    for j in tl.static_range(MAXJ):
        w = tl.load(thg_ptr + (j * NB + ib) * BLK + t)
        b = tl.load(src_ptr + base + j * X)
        acc += w * b
    tl.store(
        dst_ptr + pid.to(tl.int64) * BLK + t,
        acc.to(dst_ptr.dtype.element_ty),
    )


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _cubic_aa_filter_scalar(x):
    x = abs(float(x))
    if x < 1.0:
        return (1.5 * x - 2.5) * x * x + 1.0
    if x < 2.0:
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    return 0.0


def _compute_scale(input_size, output_size, align_corners, scale=None):
    if align_corners:
        return float(input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    return (
        (1.0 / scale)
        if (scale is not None and scale > 0)
        else float(input_size) / output_size
    )


def _build_axis_tables(in_size, out_size, align_corners, scale_arg):
    scale = _compute_scale(in_size, out_size, align_corners, scale_arg)
    support = 2.0 * scale if scale >= 1.0 else 2.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0

    contrib = [[] for _ in range(in_size)]
    for oi in range(out_size):
        center = scale * (oi + 0.5)
        xmin = max(int(math.floor(center - support + 0.5)), 0)
        xmax = min(int(math.floor(center + support + 0.5)), in_size)
        if xmax <= xmin:
            continue
        vals = [
            _cubic_aa_filter_scalar((ii - center + 0.5) * invscale)
            for ii in range(xmin, xmax)
        ]
        total = 0.0
        for v in vals:
            total += v
        if total == 0.0:
            continue
        inv_total = 1.0 / total
        for ii, v in zip(range(xmin, xmax), vals):
            if v != 0.0:
                contrib[ii].append((oi, v * inv_total))

    maxj = 1
    for row in contrib:
        if row:
            span = row[-1][0] - row[0][0] + 1
            if span > maxj:
                maxj = span
    maxj = min(maxj, out_size)

    hi = max(out_size - maxj, 0)
    starts = [0] * in_size
    table = [[0.0] * in_size for _ in range(maxj)]
    for ii in range(in_size):
        row = contrib[ii]
        if not row:
            continue
        base = min(max(row[0][0], 0), hi)
        starts[ii] = base
        for oi, w in row:
            j = oi - base
            if 0 <= j < maxj:
                table[j][ii] = w
    return table, starts, maxj


def _cache_put(cache, key, value):
    if len(cache) >= _TABLE_CACHE_MAX:
        cache.popitem(last=False)
    cache[key] = value
    return value


def _get_axis_tables(in_size, out_size, align_corners, scale_arg):
    key = (
        int(in_size),
        int(out_size),
        bool(align_corners),
        None if scale_arg is None else float(scale_arg),
    )
    cached = _axis_cache.get(key)
    if cached is not None:
        _axis_cache.move_to_end(key)
        return cached
    table, starts, maxj = _build_axis_tables(
        in_size, out_size, align_corners, scale_arg
    )
    value = (
        torch.tensor(table, dtype=torch.float32).reshape(maxj, in_size),
        torch.tensor(starts, dtype=torch.int32),
        maxj,
    )
    return _cache_put(_axis_cache, key, value)


def _get_tiled_w(in_size, out_size, align_corners, scale_arg, blk, device):
    key = (
        int(in_size),
        int(out_size),
        bool(align_corners),
        None if scale_arg is None else float(scale_arg),
        int(blk),
        str(device),
    )
    cached = _tiled_w_cache.get(key)
    if cached is not None:
        _tiled_w_cache.move_to_end(key)
        return cached
    tw, tws, maxj = _get_axis_tables(in_size, out_size, align_corners, scale_arg)
    idx = torch.arange(blk, dtype=torch.int64) & (in_size - 1)
    value = (
        tw[:, idx].contiguous().to(device=device),
        tws[idx].contiguous().to(device=device),
        maxj,
    )
    return _cache_put(_tiled_w_cache, key, value)


def _get_tiled_h(in_size, out_size, align_corners, scale_arg, blk, log2x, device):
    key = (
        int(in_size),
        int(out_size),
        bool(align_corners),
        None if scale_arg is None else float(scale_arg),
        int(blk),
        int(log2x),
        str(device),
    )
    cached = _tiled_h_cache.get(key)
    if cached is not None:
        _tiled_h_cache.move_to_end(key)
        return cached
    th, ths, maxj = _get_axis_tables(in_size, out_size, align_corners, scale_arg)
    group = blk >> log2x
    nb = max(in_size // group, 1)
    sub = torch.arange(blk, dtype=torch.int64) >> log2x
    idx = torch.arange(nb, dtype=torch.int64).view(-1, 1) * group + sub.view(1, -1)
    idx = idx % in_size
    value = (
        th[:, idx].contiguous().to(device=device),
        ths[idx].contiguous().to(device=device),
        maxj,
        nb,
    )
    return _cache_put(_tiled_h_cache, key, value)


def _pick_group_w(rows, row_w, elem_cap):
    cap = max(1, elem_cap // row_w)
    group = 1
    while group * 2 <= cap and rows % (group * 2) == 0:
        group *= 2
    return group


def _pick_group_h(rows, row_w, h_in, elem_cap):
    """Largest power-of-two row group that tiles `rows` exactly and keeps the
    per-tile input-row pattern independent of the program id."""
    cap = max(1, elem_cap // row_w)
    best = 1
    group = 1
    while group <= cap:
        aligned = (h_in % group == 0) or (_is_pow2(h_in) and group % h_in == 0)
        if rows % group == 0 and aligned:
            best = group
        group *= 2
    return best


def _elem_cap(taps):
    """Cap the block so the fully unrolled tap loop stays inside the live-vector
    budget of this backend.

    Measured on XPU3 2026-08-31: BLK=1024 with 8 taps (32 KiB of tap vectors)
    is fine, while BLK=4096 with 14 taps (224 KiB) makes
    `_aa_w_gather_kernel` raise `reason[4] load/store operation exceed memory
    size` even though every address is provably in bounds, and that wedges the
    card.  `_LIVE_VEC_BUDGET_BYTES` is the swept safe value.
    """
    budget_elems = _LIVE_VEC_BUDGET_BYTES // 4
    return max(_MIN_BLK, min(_TILE_ELEMS, budget_elems // (taps + 2)))


def _num_warps(blk):
    if blk >= 1024:
        return 8
    if blk >= 256:
        return 4
    return 1


def _generic_fallback(*args):
    from flag_gems.ops.upsample_bicubic2d_aa_backward import (
        _upsample_bicubic2d_aa_backward as _generic,
    )

    return _generic(*args)


def _upsample_bicubic2d_aa_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_BICUBIC2D_AA_BACKWARD")
    N, C, H_in, W_in = input_size
    H_out, W_out = output_size

    assert grad_output.shape == (N, C, H_out, W_out), (
        f"grad_output shape {grad_output.shape} != "
        f"expected ({N}, {C}, {H_out}, {W_out})"
    )

    NC = N * C
    if NC == 0 or H_in == 0 or W_in == 0 or H_out == 0 or W_out == 0:
        return grad_output.new_zeros(input_size)

    args = (grad_output, output_size, input_size, align_corners, scales_h, scales_w)

    # Fast-path eligibility is pure metadata only (shapes and powers of two);
    # it never dispatches an operator.
    if not _is_pow2(W_in) or W_in > _MAX_ROW_W:
        return _generic_fallback(*args)

    rows_w = NC * H_out
    rows_h = NC * H_in

    _, _, taps_w = _get_axis_tables(W_in, W_out, align_corners, scales_w)
    _, _, taps_h = _get_axis_tables(H_in, H_out, align_corners, scales_h)
    if taps_w > _MAX_TAPS or taps_h > _MAX_TAPS:
        return _generic_fallback(*args)

    group_w = _pick_group_w(rows_w, W_in, _elem_cap(taps_w))
    group_h = _pick_group_h(rows_h, W_in, H_in, _elem_cap(taps_h))
    blk_w = group_w * W_in
    blk_h = group_h * W_in
    if blk_w < _MIN_BLK or blk_h < _MIN_BLK:
        return _generic_fallback(*args)
    if blk_w > _elem_cap(taps_w) or blk_h > _elem_cap(taps_h):
        return _generic_fallback(*args)
    if taps_h * max(H_in, group_h) * W_in > _MAX_TABLE_ELEMS:
        return _generic_fallback(*args)
    if taps_w * blk_w > _MAX_TABLE_ELEMS:
        return _generic_fallback(*args)

    device = grad_output.device
    log2x = W_in.bit_length() - 1
    if group_h <= H_in:
        nb_mult = 1
        log2ph = blk_h.bit_length() - 1
    else:
        nb_mult = group_h // H_in
        log2ph = (H_in.bit_length() - 1) + log2x

    twg, twsg, taps_w = _get_tiled_w(
        W_in, W_out, align_corners, scales_w, blk_w, device
    )
    thg, thsg, taps_h, nb = _get_tiled_h(
        H_in, H_out, align_corners, scales_h, blk_h, log2x, device
    )

    grad_out_flat = grad_output.contiguous()

    buf = torch.empty(rows_w * W_in, dtype=torch.float32, device=device)
    _aa_w_gather_kernel[(rows_w // group_w,)](
        grad_out_flat,
        buf,
        twg,
        twsg,
        SRC_W=W_out,
        LOG2X=log2x,
        G=group_w,
        BLK=blk_w,
        MAXJ=taps_w,
        num_warps=_num_warps(blk_w),
    )

    grad_in = torch.empty(rows_h * W_in, dtype=grad_output.dtype, device=device)
    _aa_h_select_kernel[(rows_h // group_h,)](
        buf,
        grad_in,
        thg,
        thsg,
        H_OUT=H_out,
        X=W_in,
        NB=nb,
        M=nb_mult,
        LOG2PH=log2ph,
        BLK=blk_h,
        MAXJ=taps_h,
        num_warps=_num_warps(blk_h),
    )

    return grad_in.reshape(N, C, H_in, W_in)
