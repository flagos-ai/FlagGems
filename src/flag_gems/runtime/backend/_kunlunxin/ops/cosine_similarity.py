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

"""Kunlunxin cosine_similarity.

Reduction strategy
------------------
The generic kernel builds a [BLOCK_M, BLOCK_D] 2-D tile and reduces along
axis 1. On XPU every ``tl.sum`` reduction carries a large fixed cost, so the
generic layout ends up at 10-20 GB/s and it miscompiles outright when the tile
is wider than D (masked 2-D loads).

Measured on XPU3, the fast shape for a row reduction is the same one
``ops/dot.py`` uses: a [128, YBLOCK] tile staged into LM through
``tle.gpu.copy``, reduced with ``tl.sum(axis=1)``. That reaches 0.3-0.5 TB/s,
2x the native ATen chain, so this file is built around it:

* ``D == 1``          - one flat elementwise pass (one-lane rows cannot be
                        amortised: per-row programs measure ~90x slower).
* ``D <= 4096``       - one TLE row-reduce per program, result written directly.
* ``D > 4096``        - the flat data is viewed as (N*MID, cols) segments with
                        ``cols`` chosen near a target program count, then a
                        cheap merge kernel sums the segments of a row.
* small inputs / no TLE / fp64 / D without a suitable tile-aligned factor -
                        the scalar piece reducer below (1-D scalar reductions,
                        several rows per program) is the portable fallback.

Three scalar reductions per row (dot, |x1|^2, |x2|^2) are accumulated in
float32 and combined once per row as
``dot / (max(sqrt(n1), eps) * max(sqrt(n2), eps))``.

``XBLOCK`` is pinned to 128: smaller tiles silently produce wrong values in
the XPU TMA path (verified: dot error ~130% at XBLOCK=16 on a 16-row tile).
Rows past N are clamped by the descriptor, so padding to 128 is free.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

# TLE row-reduce geometry: [XBLOCK rows, YBLOCK cols] LM tile.
_TLE_XBLOCK = 128
_TLE_YBLOCK = 128
_TLE_COLS_MAX = 4096  # widest flat segment
_TLE_COLS_MIN = 128  # narrowest flat segment (== the tile height)
_TLE_MIN_ELEMS = 1 << 15
_PROG_TARGET = 128  # tile blocks to aim for when choosing the segment width
_MIN_PROGS = 8  # a narrower grid leaves the device idle
_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}

_ROWS = 8  # scalar fallback: rows per program when many short rows
_MULTI_MIN_N = 1024
_BLOCK_D = 2048
_D1_BLOCK = 1024

_MERGE_VEC_MAX = 64  # segment count below which the merge stays vectorised
_MERGE_ROWS = 256
_MERGE_BLOCK = 4096  # max segments folded per merge loop iteration
_MID_BLOCK = 2048


def _tle_available():
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()

# Latched True when this toolchain cannot compile the tle row kernel. flagtree XPU
# fails with "failed to legalize operation 'tt.make_range' that was explicitly marked
# illegal" on its 2D local_ptr range (the same latch sum/prod/logsumexp use), so the
# first failing call is retried on the scalar path and every later call skips tle.
_TLE_BROKEN = False


@triton.jit(
    do_not_specialize=["D", "MID_STRIDE"],
    do_not_specialize_on_alignment=["x1_desc", "x2_desc", "out_desc"],
)
def _cos_tle_kernel(
    x1_desc,
    x2_desc,
    out_desc,
    D,
    MID_STRIDE,
    eps,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
    FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * XBLOCK
    x1_lmem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=IN_DTYPE, scope=tle.gpu.lmem)
    x2_lmem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=IN_DTYPE, scope=tle.gpu.lmem)
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    x1_ptrs = tle.gpu.local_ptr(x1_lmem, (row_ids, col_ids))
    x2_ptrs = tle.gpu.local_ptr(x2_lmem, (row_ids, col_ids))
    dot = tl.zeros([XBLOCK], ACC_DTYPE)
    n1 = tl.zeros([XBLOCK], ACC_DTYPE)
    n2 = tl.zeros([XBLOCK], ACC_DTYPE)
    for coff in tl.range(0, D, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > D:
                # The DMA clamps the tail read, so stale LM lanes must be zeroed.
                tl.store(x1_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
                tl.store(x2_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(x1_desc, x1_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        tle.gpu.copy(x2_desc, x2_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        a = tl.load(x1_ptrs).to(ACC_DTYPE)
        b = tl.load(x2_ptrs).to(ACC_DTYPE)
        dot += tl.sum(a * b, axis=1)
        n1 += tl.sum(a * a, axis=1)
        n2 += tl.sum(b * b, axis=1)
    # TMA epilogue: plain per-program stores cost ~1us each on XPU3, the LM
    # staging copy is free in comparison.
    if FINAL:
        o_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, scope=tle.gpu.lmem)
        o_ptrs = tle.gpu.local_ptr(o_lmem, (tl.arange(0, XBLOCK),))
        o = dot / (tl.maximum(tl.sqrt(n1), eps) * tl.maximum(tl.sqrt(n2), eps))
        tl.store(o_ptrs, o.to(OUT_DTYPE))
        tle.gpu.copy(o_lmem, out_desc, [XBLOCK], [row_off])
    else:
        d_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, scope=tle.gpu.lmem)
        a_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, scope=tle.gpu.lmem)
        b_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, scope=tle.gpu.lmem)
        d_ptrs = tle.gpu.local_ptr(d_lmem, (tl.arange(0, XBLOCK),))
        a_ptrs = tle.gpu.local_ptr(a_lmem, (tl.arange(0, XBLOCK),))
        b_ptrs = tle.gpu.local_ptr(b_lmem, (tl.arange(0, XBLOCK),))
        tl.store(d_ptrs, dot.to(OUT_DTYPE))
        tl.store(a_ptrs, n1.to(OUT_DTYPE))
        tl.store(b_ptrs, n2.to(OUT_DTYPE))
        tle.gpu.copy(d_lmem, out_desc, [XBLOCK], [row_off])
        tle.gpu.copy(a_lmem, out_desc, [XBLOCK], [MID_STRIDE + row_off])
        tle.gpu.copy(b_lmem, out_desc, [XBLOCK], [2 * MID_STRIDE + row_off])


@triton.jit
def _cos_fold_kernel(
    mid_ptr,
    out_ptr,
    MID_STRIDE,
    STAT_STRIDE,
    NBLK: tl.constexpr,
    MID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Parallel first hop of the merge: one program per (row, BLOCK segments).
    r = tl.program_id(0)
    gid = tl.program_id(1)
    base = mid_ptr + r * MID
    idx = gid * BLOCK + tl.arange(0, BLOCK)
    wt = idx < MID
    idx = tl.minimum(idx, MID - 1)
    dot = tl.sum(tl.where(wt, tl.load(base + idx), 0.0))
    n1 = tl.sum(tl.where(wt, tl.load(base + MID_STRIDE + idx), 0.0))
    n2 = tl.sum(tl.where(wt, tl.load(base + 2 * MID_STRIDE + idx), 0.0))
    d = out_ptr + r * NBLK + gid
    tl.store(d, dot)
    tl.store(d + STAT_STRIDE, n1)
    tl.store(d + 2 * STAT_STRIDE, n2)


@triton.jit
def _cos_merge_vec_kernel(
    mid_ptr,
    out_ptr,
    N,
    MID_STRIDE,
    eps,
    ROWS: tl.constexpr,
    MID: tl.constexpr,
    MP2: tl.constexpr,
):
    # Few wide segments per row: fold them as one [ROWS, MP2] tile.
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < N
    rc = tl.minimum(rows, N - 1)
    j = tl.arange(0, MP2)
    # A computed mask on tl.load is ignored on XPU3 (masked lanes still
    # contribute), so read in-range and weight with tl.where instead.
    offs = rc[:, None] * MID + tl.minimum(j, MID - 1)[None, :]
    wt = (j < MID)[None, :]
    dot = tl.sum(tl.where(wt, tl.load(mid_ptr + offs), 0.0), axis=1)
    n1 = tl.sum(tl.where(wt, tl.load(mid_ptr + MID_STRIDE + offs), 0.0), axis=1)
    n2 = tl.sum(tl.where(wt, tl.load(mid_ptr + 2 * MID_STRIDE + offs), 0.0), axis=1)
    o = dot / (tl.maximum(tl.sqrt(n1), eps) * tl.maximum(tl.sqrt(n2), eps))
    tl.store(out_ptr + rows, o.to(out_ptr.type.element_ty), rmask)


@triton.jit
def _cos_merge_row_kernel(
    mid_ptr,
    out_ptr,
    eps,
    MID_STRIDE,
    MID: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    # Many narrow segments per row: one program per row, folded in blocks.
    r = tl.program_id(0)
    off = tl.arange(0, BLOCK_MID)
    base = mid_ptr + r * MID
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for j0 in tl.range(0, MID, BLOCK_MID):
        idx = j0 + off
        wt = idx < MID
        idx = tl.minimum(idx, MID - 1)
        dot += tl.sum(tl.where(wt, tl.load(base + idx), 0.0))
        n1 += tl.sum(tl.where(wt, tl.load(base + MID_STRIDE + idx), 0.0))
        n2 += tl.sum(tl.where(wt, tl.load(base + 2 * MID_STRIDE + idx), 0.0))
    o = dot / (tl.maximum(tl.sqrt(n1), eps) * tl.maximum(tl.sqrt(n2), eps))
    tl.store(out_ptr + r, o.to(out_ptr.type.element_ty))


@triton.jit
def _cos_finalize(dot, n1, n2, eps):
    return dot / (tl.maximum(tl.sqrt(n1), eps) * tl.maximum(tl.sqrt(n2), eps))


@triton.jit
def _cos_piece_stats(
    x1_ptr,
    x2_ptr,
    base,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    """dot, |x1|^2, |x2|^2 of the row slice at ``base``.

    NP uniform tiles of width S plus NSCALAR trailing scalars (uniform widths
    keep the live-tile SRAM footprint down; mixed widths blow uni_sram).
    """
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in tl.static_range(NP):
        off = base + i * S + tl.arange(0, S)
        a = tl.load(x1_ptr + off).to(acc_dtype)
        b = tl.load(x2_ptr + off).to(acc_dtype)
        dot += tl.sum(a * b)
        n1 += tl.sum(a * a)
        n2 += tl.sum(b * b)
    if NSCALAR > 0:
        for j in tl.range(NSCALAR):
            sa = tl.load(x1_ptr + base + NP * S + j).to(acc_dtype)
            sb = tl.load(x2_ptr + base + NP * S + j).to(acc_dtype)
            dot += sa * sb
            n1 += sa * sa
            n2 += sb * sb
    return dot, n1, n2


@libentry()
@triton.jit
def _cos_small_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    D,
    eps,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    dot, n1, n2 = _cos_piece_stats(x1_ptr, x2_ptr, pid * D, S, NP, NSCALAR)
    tl.store(out_ptr + pid, _cos_finalize(dot, n1, n2, eps).to(out_ptr.type.element_ty))


@libentry()
@triton.jit
def _cos_small_multi_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    D,
    eps,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    for r in tl.static_range(ROWS):
        row = pid * ROWS + r
        if row < N:
            dot, n1, n2 = _cos_piece_stats(x1_ptr, x2_ptr, row * D, S, NP, NSCALAR)
            tl.store(
                out_ptr + row,
                _cos_finalize(dot, n1, n2, eps).to(out_ptr.type.element_ty),
            )


@libentry()
@triton.jit
def _cos_d1_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    safe = tl.minimum(off, N - 1)
    a = tl.load(x1_ptr + safe).to(tl.float32)
    b = tl.load(x2_ptr + safe).to(tl.float32)
    o = (a * b) / (tl.maximum(tl.abs(a), eps) * tl.maximum(tl.abs(b), eps))
    tl.store(out_ptr + off, o.to(out_ptr.type.element_ty), mask=off < N)


@libentry()
@triton.jit
def _cos_tail_kernel(
    x1_ptr,
    x2_ptr,
    tail_ptr,
    D,
    T,
    N,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    # Partial stats of the (D % nchunk) columns at the end of each row.
    pid = tl.program_id(0)
    dot, n1, n2 = _cos_piece_stats(x1_ptr, x2_ptr, pid * D + D - T, S, NP, NSCALAR)
    tl.store(tail_ptr + pid, dot)
    tl.store(tail_ptr + N + pid, n1)
    tl.store(tail_ptr + 2 * N + pid, n2)


@libentry()
@triton.jit
def _cos_chunk_kernel(
    x1_ptr,
    x2_ptr,
    mid_ptr,
    D,
    MID_STRIDE,
    BLOCK: tl.constexpr,
):
    # Fallback split-K partials (no TLE): one program per (row, chunk).
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    base = pid_n * D + pid_c * BLOCK
    a = tl.load(x1_ptr + base + tl.arange(0, BLOCK)).to(acc_dtype)
    b = tl.load(x2_ptr + base + tl.arange(0, BLOCK)).to(acc_dtype)
    row = mid_ptr + pid_n * (3 * MID_STRIDE)
    tl.store(row + pid_c, tl.sum(a * b))
    tl.store(row + MID_STRIDE + pid_c, tl.sum(a * a))
    tl.store(row + 2 * MID_STRIDE + pid_c, tl.sum(b * b))


@libentry()
@triton.jit
def _cos_mid_reduce_kernel(
    mid_ptr,
    out_ptr,
    STRIDE_IN,
    STRIDE_OUT,
    BLOCK: tl.constexpr,
):
    # Fold BLOCK chunk partials into one (fallback, row-major layout).
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    off = pid_c * BLOCK + tl.arange(0, BLOCK)
    src = mid_ptr + pid_n * (3 * STRIDE_IN)
    dst = out_ptr + pid_n * (3 * STRIDE_OUT)
    for s in tl.static_range(3):
        tl.store(
            dst + s * STRIDE_OUT + pid_c,
            tl.sum(tl.load(src + s * STRIDE_IN + off).to(out_ptr.type.element_ty)),
        )


@libentry()
@triton.jit
def _cos_final_kernel(
    mid_ptr,
    tail_ptr,
    out_ptr,
    N,
    eps,
    STRIDE,
    HAS_TAIL: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    # Fallback merge of the chunk partials (row-major layout).
    pid = tl.program_id(0)
    acc_dtype = tl.float64 if mid_ptr.type.element_ty == tl.float64 else tl.float32
    off = tl.arange(0, BLOCK_MID)
    src = mid_ptr + pid * (3 * STRIDE)
    dot = tl.sum(tl.load(src + off).to(acc_dtype))
    n1 = tl.sum(tl.load(src + STRIDE + off).to(acc_dtype))
    n2 = tl.sum(tl.load(src + 2 * STRIDE + off).to(acc_dtype))
    if HAS_TAIL:
        dot += tl.load(tail_ptr + pid)
        n1 += tl.load(tail_ptr + N + pid)
        n2 += tl.load(tail_ptr + 2 * N + pid)
    tl.store(out_ptr + pid, _cos_finalize(dot, n1, n2, eps).to(out_ptr.type.element_ty))


def _piece_args(t):
    """Uniform-width piece decomposition of a row of length ``t``.

    Returns (S, NP, NSCALAR): NP tiles of uniform width S (power of two, <=
    512) plus NSCALAR trailing lanes handled by a scalar loop.
    """
    if t <= 0:
        return 0, 0, 0
    best = (0, 0)
    S = 512
    while S > 0:
        n = t // S
        if n > 0:
            cov = min(n, 8) * S
            if cov > best[0]:
                best = (cov, S)
        S //= 2
    _, S = best
    np = min(t // S, 8)
    return S, np, t - np * S


def _pick_cols(N, D):
    """Flat-segment width for the TLE reduce, or 0 when none is usable.

    Total tile work is fixed (N*D / (XBLOCK*YBLOCK)); the width only trades the
    number of merge partials per row (``D // cols``) against the program count
    (``N*D // (XBLOCK*cols)``). Measured optima keep at least a handful of
    programs alive and fold every row inside a single merge block, so aim for
    ``_PROG_TARGET`` programs and widen up to that one-block merge while the
    grid stays busy. When D is too large for a one-block merge the merge cost
    per partial dominates and the program target alone wins.
    A width is usable only when it divides D (otherwise the flat view would
    overrun the row) and is a whole number of tile rows (otherwise the column
    tail would need stale LM tile lanes zeroed).
    """
    tiles = (N * D) // (_TLE_XBLOCK * _PROG_TARGET)
    if D <= _TLE_COLS_MAX * _TLE_YBLOCK:
        one_block = D // _TLE_YBLOCK
        keep_busy = (N * D) // (_TLE_XBLOCK * _MIN_PROGS)
        tiles = max(tiles, min(one_block, keep_busy))
    need = max(_TLE_COLS_MIN, min(tiles, _TLE_COLS_MAX))
    best = 0
    for cols in range(_TLE_COLS_MIN, _TLE_COLS_MAX + 1, _TLE_YBLOCK):
        if D % cols:
            continue
        if best == 0 or abs(cols - need) < abs(best - need):
            best = cols
    return best


def _launch_tle(x1, x2, N, D, eps):
    """Returns the flat (N,) output."""
    tdt = _TLE_TL_DTYPE[x1.dtype]
    xb = _TLE_XBLOCK
    if D <= _TLE_COLS_MAX:
        yb = min(_TLE_YBLOCK, triton.next_power_of_2(D))
        nblk = triton.cdiv(N, xb)
        out = torch.empty((nblk * xb,), dtype=x1.dtype, device=x1.device)
        _cos_tle_kernel[(nblk,)](
            TensorDescriptor.from_tensor(x1, block_shape=[xb, yb]),
            TensorDescriptor.from_tensor(x2, block_shape=[xb, yb]),
            TensorDescriptor.from_tensor(out, block_shape=[xb]),
            D,
            0,
            eps,
            XBLOCK=xb,
            YBLOCK=yb,
            IN_DTYPE=tdt,
            ACC_DTYPE=tl.float32,
            OUT_DTYPE=tdt,
            NEED_ZERO=(D % yb != 0),
            FINAL=True,
        )
        return out[:N]

    cols = _pick_cols(N, D)
    MID = D // cols
    rows = N * MID
    nblk = triton.cdiv(rows, xb)
    npad = nblk * xb
    mid = torch.empty((3 * npad,), dtype=torch.float32, device=x1.device)
    _cos_tle_kernel[(nblk,)](
        TensorDescriptor.from_tensor(
            x1.view(rows, cols), block_shape=[xb, _TLE_YBLOCK]
        ),
        TensorDescriptor.from_tensor(
            x2.view(rows, cols), block_shape=[xb, _TLE_YBLOCK]
        ),
        TensorDescriptor.from_tensor(mid, block_shape=[xb]),
        cols,
        npad,
        eps,
        XBLOCK=xb,
        YBLOCK=_TLE_YBLOCK,
        IN_DTYPE=tdt,
        ACC_DTYPE=tl.float32,
        OUT_DTYPE=tl.float32,
        NEED_ZERO=False,
        FINAL=False,
    )

    # Huge segment counts would be streamed by a single program in the final
    # merge, so fold them down in parallel first.
    cur, cur_stride, cur_n = mid, npad, MID
    while cur_n > _MERGE_BLOCK:
        g = triton.cdiv(cur_n, _MERGE_BLOCK)
        stat_stride = N * g
        folded = torch.empty((3 * stat_stride,), dtype=torch.float32, device=x1.device)
        _cos_fold_kernel[(N, g)](
            cur,
            folded,
            cur_stride,
            stat_stride,
            NBLK=g,
            MID=cur_n,
            BLOCK=_MERGE_BLOCK,
        )
        cur, cur_stride, cur_n = folded, stat_stride, g

    out = torch.empty((N,), dtype=x1.dtype, device=x1.device)
    if cur_n <= _MERGE_VEC_MAX:
        _cos_merge_vec_kernel[(triton.cdiv(N, _MERGE_ROWS),)](
            cur,
            out,
            N,
            cur_stride,
            eps,
            ROWS=_MERGE_ROWS,
            MID=cur_n,
            MP2=triton.next_power_of_2(cur_n),
        )
    else:
        _cos_merge_row_kernel[(N,)](
            cur,
            out,
            eps,
            cur_stride,
            MID=cur_n,
            BLOCK_MID=min(_MERGE_BLOCK, triton.next_power_of_2(cur_n)),
        )
    return out


def _launch_scalar(x1, x2, out, N, D, eps):
    if D <= _BLOCK_D:
        S, NP, NSC = _piece_args(D)
        if N >= _MULTI_MIN_N:
            _cos_small_multi_kernel[(triton.cdiv(N, _ROWS),)](
                x1, x2, out, N, D, eps, S=S, NP=NP, NSCALAR=NSC, ROWS=_ROWS
            )
        else:
            _cos_small_kernel[(N,)](x1, x2, out, N, D, eps, S=S, NP=NP, NSCALAR=NSC)
        return

    chunk_block = 4096 if D >= 2 * _BLOCK_D else _BLOCK_D
    MID = D // chunk_block
    T = D - MID * chunk_block
    P = MID + (1 if T > 0 else 0)
    stride = max(2, triton.next_power_of_2(P))
    mdtype = torch.float64 if x1.dtype == torch.float64 else torch.float32
    mid = torch.zeros((N * 3 * stride,), device=x1.device, dtype=mdtype)
    _cos_chunk_kernel[(N, MID)](x1, x2, mid, D, stride, BLOCK=chunk_block)
    tail = None
    if T > 0:
        PS, PNP, PNSC = _piece_args(T)
        tail = torch.zeros((3 * N,), device=x1.device, dtype=mdtype)
        _cos_tail_kernel[(N,)](x1, x2, tail, D, T, N, S=PS, NP=PNP, NSCALAR=PNSC)
    cur_mid, cur_stride, cur_n = mid, stride, P
    while triton.next_power_of_2(cur_n) > _MID_BLOCK:
        g = triton.cdiv(cur_n, _MID_BLOCK)
        nstride = triton.next_power_of_2(g)
        cur_out = torch.zeros((N * 3 * nstride,), device=x1.device, dtype=mdtype)
        _cos_mid_reduce_kernel[(N, g)](
            cur_mid,
            cur_out,
            STRIDE_IN=cur_stride,
            STRIDE_OUT=nstride,
            BLOCK=_MID_BLOCK,
        )
        cur_mid, cur_stride, cur_n = cur_out, nstride, g
    _cos_final_kernel[(N,)](
        cur_mid,
        mid if tail is None else tail,
        out,
        N,
        eps,
        cur_stride,
        HAS_TAIL=tail is not None,
        BLOCK_MID=triton.next_power_of_2(cur_n),
    )


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    global _TLE_BROKEN
    logger.debug("GEMS_KUNLUNXIN COSINE_SIMILARITY")
    assert x1.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"cosine_similarity: unsupported dtype {x1.dtype}"
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    ndim = x1.ndim
    dim = dim % ndim if ndim > 0 else 0
    if dim != ndim - 1:
        x1 = torch.movedim(x1, dim, -1)
        x2 = torch.movedim(x2, dim, -1)
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    D = x1.shape[-1]
    out_shape = x1.shape[:-1]
    if D == 0:
        # Empty reduction dim: numerator and both norms vanish, and
        # max(0, eps)**2 > 0, so the result is zero everywhere.
        return torch.zeros(out_shape, dtype=x1.dtype, device=x1.device)

    N = x1.numel() // D
    seg_cols = D if D <= _TLE_COLS_MAX else _pick_cols(N, D)
    use_tle = (
        _TLE_AVAILABLE
        and not _TLE_BROKEN
        and x1.dtype in _TLE_TL_DTYPE
        and D >= 64
        and (D * x1.element_size()) % 16 == 0  # TMA needs 16B-aligned row strides
        and x1.numel() >= _TLE_MIN_ELEMS
        and seg_cols > 0
    )
    with torch_device_fn.device(x1.device):
        if D == 1:
            out = torch.empty((N,), dtype=x1.dtype, device=x1.device)
            _cos_d1_kernel[(triton.cdiv(N, _D1_BLOCK),)](
                x1, x2, out, N, eps, BLOCK=_D1_BLOCK
            )
        elif use_tle:
            try:
                out = _launch_tle(x1.reshape(N, D), x2.reshape(N, D), N, D, eps)
            except Exception as exc:  # noqa: BLE001
                # Compile gap (tt.make_range on the 2D local_ptr range is not
                # legalizable on older flagtree builds): retry the scalar path.
                _TLE_BROKEN = True
                logger.debug(
                    "GEMS_KUNLUNXIN COSINE_SIMILARITY tle unavailable (%s)", exc
                )
                out = torch.empty((N,), dtype=x1.dtype, device=x1.device)
                _launch_scalar(x1.reshape(N, D), x2.reshape(N, D), out, N, D, eps)
        else:
            out = torch.empty((N,), dtype=x1.dtype, device=x1.device)
            _launch_scalar(x1.reshape(N, D), x2.reshape(N, D), out, N, D, eps)

    return out.reshape(out_shape)
