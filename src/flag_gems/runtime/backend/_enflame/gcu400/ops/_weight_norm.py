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

"""Fused Triton implementation of ``aten::_weight_norm`` (weight normalization).

Semantics reproduced from the reference/aten path:
    n_j = sqrt(sum over every axis except ``dim`` of v**2)
    out = v * (g_j / n_j)
with no epsilon and no clamping, so zero norms yield NaN exactly like the
reference, and ``g`` is read at its per-slab stride (a flattened n-shaped
tensor, e.g. (S,1,1) / (1,1,K)).

The wrapper only performs metadata reads, grid/scalar computation, tensor
allocation and JIT launches.  All numerical work happens in the Triton kernels.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _p2(x):
    """Smallest power of two >= x (x >= 1)."""
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# gcu400 per-program DSM limit enforced at fatbin generation (max_dsm option).
_GCU_DSM_BYTES = 917504

# gcu400 SIP count; used as the minimum number of programs a kernel should see.
_NUM_SIPS = 24

# dim=0 split-reduction path: engaged once the reduced axis is long enough that
# one program per slab block would serialise the whole operator.  _FIRST_SPLIT_TILE
# is the element count of one reduction tile (measured DSM optimum).
_FIRST_SPLIT_TILE = 65536
_FIRST_SPLIT_MIN_Q = 8192
_FIRST_SPLIT_BLOCK_Q = 4096

# Widest contiguous column block the last-axis path uses; wider blocks keep the
# row-major reads coalesced once K is large (measured on (10000, 65536)).
_LAST_MAX_BLOCK_K = 1024
# K at which the wide column block starts to pay off.  Below it the square
# [256, 256] tile wins: (4096, 4096) 144 us wide-256 vs 203 us with 1024, while
# (10000, 65536) needs 2197 us with 1024 against 4953 us with 256.
_LAST_WIDE_K = 16384
# Minimum program count for the wide-block ladder before BLOCK_K is halved.
_LAST_MIN_PART_PROGRAMS = 128
# Element budget of the single-pass last-axis tile (see _dsm_tile_cap: both
# fp32 and fp16 resolve to 65536 elements) and the minimum program count that
# still beats the two-pass kernel.
_LAST_SINGLE_MAX_ELEMS = 65536
_LAST_SINGLE_MIN_PROGRAMS = 16
# Program count the last-axis part kernel aims for; the row chunk count is
# derived from it so that a narrow K (which yields few column blocks) still
# fills the card with work.
_LAST_SPLIT_PROGRAMS = 64
# Smallest M (row count) that still engages the last-axis split reduction.
# The split path costs one extra scratch allocation (~25 us on gcu400) plus two
# extra launches, so below this size it loses to the single-pass kernel even
# though that kernel only launches K / 64 programs.
_LAST_SPLIT_MIN_M = 2048


def _last_block_k(K, nchunk):
    """Column-block width for the last-axis split kernels."""
    if K < _LAST_WIDE_K:
        return min(_p2(K), 256)
    bk = min(_p2(K), _LAST_MAX_BLOCK_K)
    while bk > 256 and nchunk * triton.cdiv(K, bk) < _LAST_MIN_PART_PROGRAMS:
        bk //= 2
    return bk


def _p2floor(x):
    """Largest power of two <= x (x >= 1)."""
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


def _dsm_tile_cap(bytes_per_elem):
    """Max tile elements that still fit the gcu400 DSM budget.

    The enflame compiler keeps each ``tl.load`` tile in a 3-deep pipeline and
    materializes an fp32 working copy of sub-32-bit loads, so a tile costs far
    more DSM than its dtype size alone.  Measured on gcu410 (fp32 / fp16):
    _wn_fwd_first_kernel 8 / 4 B/elem, _wn_bwd_first_kernel 12 / 6 B/elem
    (x + w tiles), _wn_fwd_last_part/scale 12 / 8 B/elem; the old 256K-element
    tiles required up to 3.1 MB and failed compilation.  10% headroom covers
    the per-row/column vectors and reduction temporaries.
    """
    return _p2floor(_GCU_DSM_BYTES * 9 // 10 // bytes_per_elem)


# --------------------------------------------------------------------------
# forward: reduction over every axis except ``dim`` (dim == 0 -> "first")
# --------------------------------------------------------------------------
@triton.jit
def _wn_fwd_first_kernel(
    v_ptr,
    g_ptr,
    out_ptr,
    norm_ptr,
    S,
    Q: tl.constexpr,
    sN: tl.constexpr,
    sQ: tl.constexpr,
    gs: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    SINGLE: tl.constexpr,
    STORE_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    q = tl.arange(0, BLOCK_Q)
    if SINGLE:
        mask = rmask[:, None] & (q < Q)[None, :]
        off = rows[:, None] * sN + q[None, :] * sQ
        x = tl.load(v_ptr + off, mask=mask, other=0.0).to(tl.float32)
        n = tl.sqrt(tl.sum(x * x, axis=1))
        gv = tl.load(g_ptr + rows * gs, mask=rmask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + rows[:, None] * Q + q[None, :], x * (gv / n)[:, None], mask=mask
        )
        if STORE_NORM:
            tl.store(norm_ptr + rows, n, mask=rmask)
    else:
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        for base in range(0, Q, BLOCK_Q):
            qo = base + q
            mask = rmask[:, None] & (qo < Q)[None, :]
            x = tl.load(
                v_ptr + rows[:, None] * sN + qo[None, :] * sQ, mask=mask, other=0.0
            ).to(tl.float32)
            acc += tl.sum(x * x, axis=1)
        n = tl.sqrt(acc)
        gv = tl.load(g_ptr + rows * gs, mask=rmask, other=0.0).to(tl.float32)
        d = gv / n
        if STORE_NORM:
            tl.store(norm_ptr + rows, n, mask=rmask)
        for base in range(0, Q, BLOCK_Q):
            qo = base + q
            mask = rmask[:, None] & (qo < Q)[None, :]
            x = tl.load(
                v_ptr + rows[:, None] * sN + qo[None, :] * sQ, mask=mask, other=0.0
            ).to(tl.float32)
            tl.store(
                out_ptr + rows[:, None] * Q + qo[None, :], x * d[:, None], mask=mask
            )


@triton.jit
def _wn_fwd_first_part_kernel(
    v_ptr,
    part_ptr,
    S,
    Q,
    sN: tl.constexpr,
    sQ: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NS: tl.constexpr,
):
    """Partial |v|^2 of one (slab block, reduced-axis chunk) tile."""
    pid = tl.program_id(0)
    sblk = pid % NS
    qblk = pid // NS
    rows = sblk * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    cols = qblk * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = rmask[:, None] & (cols < Q)[None, :]
    x = tl.load(
        v_ptr + rows[:, None] * sN + cols[None, :] * sQ, mask=mask, other=0.0
    ).to(tl.float32)
    acc = tl.sum(x * x, axis=1)
    tl.store(part_ptr + qblk * S + rows, acc, mask=rmask)


@triton.jit
def _wn_fwd_first_finish_kernel(
    part_ptr,
    g_ptr,
    d_ptr,
    norm_ptr,
    S,
    NQ: tl.constexpr,
    gs: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STORE_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    ci = tl.arange(0, BLOCK_C)
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for base in range(0, NQ, BLOCK_C):
        co = base + ci
        m = (co < NQ)[:, None] & rmask[None, :]
        p = tl.load(part_ptr + co[:, None] * S + rows[None, :], mask=m, other=0.0)
        acc += tl.sum(p, axis=0)
    n = tl.sqrt(acc)
    gv = tl.load(g_ptr + rows * gs, mask=rmask, other=0.0).to(tl.float32)
    tl.store(d_ptr + rows, gv / n, mask=rmask)
    if STORE_NORM:
        tl.store(norm_ptr + rows, n, mask=rmask)


@triton.jit
def _wn_fwd_first_scale_kernel(
    v_ptr,
    d_ptr,
    out_ptr,
    S,
    Q,
    sN: tl.constexpr,
    sQ: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NS: tl.constexpr,
):
    pid = tl.program_id(0)
    sblk = pid % NS
    qblk = pid // NS
    rows = sblk * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    cols = qblk * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = rmask[:, None] & (cols < Q)[None, :]
    x = tl.load(
        v_ptr + rows[:, None] * sN + cols[None, :] * sQ, mask=mask, other=0.0
    ).to(tl.float32)
    d = tl.load(d_ptr + rows, mask=rmask, other=0.0)
    tl.store(out_ptr + rows[:, None] * Q + cols[None, :], x * d[:, None], mask=mask)


@triton.jit
def _wn_fwd_last_kernel(
    v_ptr,
    g_ptr,
    out_ptr,
    norm_ptr,
    M,
    K: tl.constexpr,
    sP: tl.constexpr,
    sN: tl.constexpr,
    gs: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SINGLE: tl.constexpr,
    STORE_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    cmask = cols < K
    mv = tl.arange(0, BLOCK_M)
    if SINGLE:
        mask = (mv[:, None] < M) & cmask[None, :]
        off = mv[:, None] * sP + cols[None, :] * sN
        x = tl.load(v_ptr + off, mask=mask, other=0.0).to(tl.float32)
        n = tl.sqrt(tl.sum(x * x, axis=0))
        gv = tl.load(g_ptr + cols * gs, mask=cmask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + mv[:, None] * K + cols[None, :], x * (gv / n)[None, :], mask=mask
        )
        if STORE_NORM:
            tl.store(norm_ptr + cols, n, mask=cmask)
    else:
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for base in range(0, M, BLOCK_M):
            mo = base + mv
            mask = (mo[:, None] < M) & cmask[None, :]
            x = tl.load(
                v_ptr + mo[:, None] * sP + cols[None, :] * sN, mask=mask, other=0.0
            ).to(tl.float32)
            acc += tl.sum(x * x, axis=0)
        n = tl.sqrt(acc)
        gv = tl.load(g_ptr + cols * gs, mask=cmask, other=0.0).to(tl.float32)
        d = gv / n
        if STORE_NORM:
            tl.store(norm_ptr + cols, n, mask=cmask)
        for base in range(0, M, BLOCK_M):
            mo = base + mv
            mask = (mo[:, None] < M) & cmask[None, :]
            x = tl.load(
                v_ptr + mo[:, None] * sP + cols[None, :] * sN, mask=mask, other=0.0
            ).to(tl.float32)
            tl.store(
                out_ptr + mo[:, None] * K + cols[None, :], x * d[None, :], mask=mask
            )


@triton.jit
def _wn_fwd_last_part_kernel(
    v_ptr,
    part_ptr,
    M,
    K: tl.constexpr,
    sP: tl.constexpr,
    sN: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NK: tl.constexpr,
):
    """Partial |v|^2 over a contiguous row chunk of one column block.

    The grid is flattened to 1D (``pc * NK + pk``) because the GCU launcher caps
    ``grid.y`` at 255 and the column-block axis reaches 256 for K = 65536.
    """
    pid = tl.program_id(0)
    pc = pid // NK
    pk = pid % NK
    cols = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    cmask = cols < K
    mv = tl.arange(0, BLOCK_M)
    base0 = pc * CHUNK
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for base in range(0, CHUNK, BLOCK_M):
        mo = base0 + base + mv
        mask = (mo[:, None] < M) & cmask[None, :]
        x = tl.load(
            v_ptr + mo[:, None] * sP + cols[None, :] * sN, mask=mask, other=0.0
        ).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(part_ptr + pc * K + cols, acc, mask=cmask)


@triton.jit
def _wn_fwd_last_finish_kernel(
    part_ptr,
    g_ptr,
    d_ptr,
    norm_ptr,
    NCHUNK: tl.constexpr,
    K: tl.constexpr,
    gs: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STORE_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    cmask = cols < K
    ci = tl.arange(0, BLOCK_C)
    mask = (ci < NCHUNK)[:, None] & cmask[None, :]
    p = tl.load(part_ptr + ci[:, None] * K + cols[None, :], mask=mask, other=0.0)
    n = tl.sqrt(tl.sum(p, axis=0))
    gv = tl.load(g_ptr + cols * gs, mask=cmask, other=0.0).to(tl.float32)
    tl.store(d_ptr + cols, gv / n, mask=cmask)
    if STORE_NORM:
        tl.store(norm_ptr + cols, n, mask=cmask)


@triton.jit
def _wn_fwd_last_scale_kernel(
    v_ptr,
    d_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    sP: tl.constexpr,
    sN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NK: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = pid // NK
    pk = pid % NK
    cols = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    cmask = cols < K
    mo = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (mo[:, None] < M) & cmask[None, :]
    x = tl.load(v_ptr + mo[:, None] * sP + cols[None, :] * sN, mask=mask, other=0.0).to(
        tl.float32
    )
    d = tl.load(d_ptr + cols, mask=cmask, other=0.0)
    tl.store(out_ptr + mo[:, None] * K + cols[None, :], x * d[None, :], mask=mask)


@triton.jit
def _wn_fwd_mid_kernel(
    v_ptr,
    g_ptr,
    out_ptr,
    norm_ptr,
    P,
    N,
    Q,
    sP,
    sN,
    sQ,
    gs,
    BLOCK_T: tl.constexpr,
    STORE_NORM: tl.constexpr,
):
    n = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    total = P * Q
    acc = 0.0
    for base in range(0, total, BLOCK_T):
        to = base + t
        mask = to < total
        p = to // Q
        q = to % Q
        x = tl.load(v_ptr + p * sP + n * sN + q * sQ, mask=mask, other=0.0).to(
            tl.float32
        )
        acc = acc + tl.sum(x * x)
    norm = tl.sqrt(acc)
    gv = tl.load(g_ptr + n * gs).to(tl.float32)
    d = gv / norm
    if STORE_NORM:
        tl.store(norm_ptr + n, norm)
    for base in range(0, total, BLOCK_T):
        to = base + t
        mask = to < total
        p = to // Q
        q = to % Q
        x = tl.load(v_ptr + p * sP + n * sN + q * sQ, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(out_ptr + (p * N + n) * Q + q, x * d, mask=mask)


# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------
@triton.jit
def _wn_bwd_first_kernel(
    v_ptr,
    w_ptr,
    g_ptr,
    n_ptr,
    vg_ptr,
    gg_ptr,
    S,
    Q,
    sN,
    sQ,
    wsN,
    wsQ,
    gs,
    ggs,
    BLOCK_S: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    q = tl.arange(0, BLOCK_Q)
    n = tl.load(n_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
    inv = 1.0 / n
    inv3 = inv * inv * inv
    gv = tl.load(g_ptr + rows * gs, mask=rmask, other=0.0).to(tl.float32)
    vb = rows[:, None] * sN
    wb = rows[:, None] * wsN
    if SINGLE:
        mask = rmask[:, None] & (q < Q)[None, :]
        x = tl.load(v_ptr + vb + q[None, :] * sQ, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + wb + q[None, :] * wsQ, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(x * w, axis=1)
        val = gv[:, None] * (w * inv[:, None] - x * (dot * inv3)[:, None])
        tl.store(vg_ptr + rows[:, None] * Q + q[None, :], val, mask=mask)
        tl.store(gg_ptr + rows * ggs, dot * inv, mask=rmask)
    else:
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        for base in range(0, Q, BLOCK_Q):
            qo = base + q
            mask = rmask[:, None] & (qo < Q)[None, :]
            x = tl.load(v_ptr + vb + qo[None, :] * sQ, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(w_ptr + wb + qo[None, :] * wsQ, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += tl.sum(x * w, axis=1)
        for base in range(0, Q, BLOCK_Q):
            qo = base + q
            mask = rmask[:, None] & (qo < Q)[None, :]
            x = tl.load(v_ptr + vb + qo[None, :] * sQ, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(w_ptr + wb + qo[None, :] * wsQ, mask=mask, other=0.0).to(
                tl.float32
            )
            val = gv[:, None] * (w * inv[:, None] - x * (acc * inv3)[:, None])
            tl.store(vg_ptr + rows[:, None] * Q + qo[None, :], val, mask=mask)
        tl.store(gg_ptr + rows * ggs, acc * inv, mask=rmask)


@triton.jit
def _wn_bwd_last_kernel(
    v_ptr,
    w_ptr,
    g_ptr,
    n_ptr,
    vg_ptr,
    gg_ptr,
    M,
    K,
    sP,
    sN,
    wsP,
    wsN,
    gs,
    ggs,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    cmask = cols < K
    mv = tl.arange(0, BLOCK_M)
    n = tl.load(n_ptr + cols, mask=cmask, other=1.0).to(tl.float32)
    inv = 1.0 / n
    inv3 = inv * inv * inv
    gv = tl.load(g_ptr + cols * gs, mask=cmask, other=0.0).to(tl.float32)
    if SINGLE:
        mask = (mv[:, None] < M) & cmask[None, :]
        x = tl.load(
            v_ptr + mv[:, None] * sP + cols[None, :] * sN, mask=mask, other=0.0
        ).to(tl.float32)
        w = tl.load(
            w_ptr + mv[:, None] * wsP + cols[None, :] * wsN, mask=mask, other=0.0
        ).to(tl.float32)
        dot = tl.sum(x * w, axis=0)
        val = gv[None, :] * (w * inv[None, :] - x * (dot * inv3)[None, :])
        tl.store(vg_ptr + mv[:, None] * K + cols[None, :], val, mask=mask)
        tl.store(gg_ptr + cols * ggs, dot * inv, mask=cmask)
    else:
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for base in range(0, M, BLOCK_M):
            mo = base + mv
            mask = (mo[:, None] < M) & cmask[None, :]
            x = tl.load(
                v_ptr + mo[:, None] * sP + cols[None, :] * sN, mask=mask, other=0.0
            ).to(tl.float32)
            w = tl.load(
                w_ptr + mo[:, None] * wsP + cols[None, :] * wsN, mask=mask, other=0.0
            ).to(tl.float32)
            acc += tl.sum(x * w, axis=0)
        for base in range(0, M, BLOCK_M):
            mo = base + mv
            mask = (mo[:, None] < M) & cmask[None, :]
            x = tl.load(
                v_ptr + mo[:, None] * sP + cols[None, :] * sN, mask=mask, other=0.0
            ).to(tl.float32)
            w = tl.load(
                w_ptr + mo[:, None] * wsP + cols[None, :] * wsN, mask=mask, other=0.0
            ).to(tl.float32)
            val = gv[None, :] * (w * inv[None, :] - x * (acc * inv3)[None, :])
            tl.store(vg_ptr + mo[:, None] * K + cols[None, :], val, mask=mask)
        tl.store(gg_ptr + cols * ggs, acc * inv, mask=cmask)


@triton.jit
def _wn_bwd_mid_kernel(
    v_ptr,
    w_ptr,
    g_ptr,
    n_ptr,
    vg_ptr,
    gg_ptr,
    P,
    N,
    Q,
    sP,
    sN,
    sQ,
    wsP,
    wsN,
    wsQ,
    gs,
    ggs,
    BLOCK_T: tl.constexpr,
):
    n = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    total = P * Q
    norm = tl.load(n_ptr + n).to(tl.float32)
    inv = 1.0 / norm
    inv3 = inv * inv * inv
    gv = tl.load(g_ptr + n * gs).to(tl.float32)
    acc = 0.0
    for base in range(0, total, BLOCK_T):
        to = base + t
        mask = to < total
        p = to // Q
        q = to % Q
        x = tl.load(v_ptr + p * sP + n * sN + q * sQ, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + p * wsP + n * wsN + q * wsQ, mask=mask, other=0.0).to(
            tl.float32
        )
        acc = acc + tl.sum(x * w)
    for base in range(0, total, BLOCK_T):
        to = base + t
        mask = to < total
        p = to // Q
        q = to % Q
        x = tl.load(v_ptr + p * sP + n * sN + q * sQ, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + p * wsP + n * wsN + q * wsQ, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(
            vg_ptr + (p * N + n) * Q + q, gv * (w * inv - x * (acc * inv3)), mask=mask
        )
    tl.store(gg_ptr + n * ggs, acc * inv)


# --------------------------------------------------------------------------
# layout helpers (three-group "slab / reduce / inner" stride model)
# --------------------------------------------------------------------------
def _layout_first(v):
    """v viewed as (S, Q): stride of the slab axis and of the inner (Q) axis."""
    rank = v.dim()
    sN = v.stride(0)
    if rank == 1:
        return sN, 1, True
    sQ = v.stride(rank - 1)
    for i in range(1, rank - 1):
        if v.stride(i) != v.stride(i + 1) * v.shape[i + 1]:
            return sN, sQ, False
    return sN, sQ, True


def _layout_last(v):
    """v viewed as (M, K): stride of the leading (M) axis and the last axis."""
    rank = v.dim()
    sN = v.stride(rank - 1)
    if rank == 1:
        return 1, sN, True
    sP = v.stride(rank - 2)
    for i in range(0, rank - 2):
        if v.stride(i) != v.stride(i + 1) * v.shape[i + 1]:
            return sP, sN, False
    return sP, sN, True


def _layout_mid(v, d):
    rank = v.dim()
    P = 1
    for i in range(0, d):
        P *= v.shape[i]
    Q = 1
    for i in range(d + 1, rank):
        Q *= v.shape[i]
    sN = v.stride(d)
    if d == 0:
        sP = 0
        ok_p = True
    else:
        sP = v.stride(d - 1)
        ok_p = True
        for i in range(0, d - 1):
            if v.stride(i) != v.stride(i + 1) * v.shape[i + 1]:
                ok_p = False
                break
    if d == rank - 1:
        sQ = 1
        ok_q = True
    else:
        sQ = v.stride(rank - 1)
        ok_q = True
        for i in range(d + 1, rank - 1):
            if v.stride(i) != v.stride(i + 1) * v.shape[i + 1]:
                ok_q = False
                break
    return P, Q, sP, sN, sQ, (ok_p and ok_q)


def _split_scratch(device, n_part, n_vec, need_norm):
    """Single fp32 buffer for the split-reduction scratch tensors.

    Every extra ``torch.empty`` on gcu400 costs ~25 us of host time, which is
    visible on small shapes, so the partial-sum buffer, the per-slice scale and
    the optional norm share one allocation and are exposed as views.
    """
    total = n_part + n_vec + (n_vec if need_norm else 0)
    buf = torch.empty(total, dtype=torch.float32, device=device)
    part = buf[:n_part]
    dsc = buf[n_part : n_part + n_vec]
    norm = buf[n_part + n_vec :] if need_norm else None
    return part, dsc, norm


def _first_split_cfg(S, Q):
    """Column-chunk width and slab-block height for the dim=0 split path.

    A 64K-element tile is the measured optimum on (64, 512, 512): 100.9 us with
    BLOCK_Q=4096 x BLOCK_S=16 versus 186 us for 8192 x 16 and 404.8 us for
    aten.  BLOCK_Q only shrinks when the grid would not fill the SMs.
    """
    bqs = _FIRST_SPLIT_BLOCK_Q
    while True:
        bss = max(1, min(_p2(S), max(1, _FIRST_SPLIT_TILE // bqs)))
        ns = triton.cdiv(S, bss)
        nq = triton.cdiv(Q, bqs)
        if ns * nq >= _NUM_SIPS or bqs <= 1024:
            return bqs, bss, ns, nq
        bqs //= 2


def _first_split_ok(S, Q):
    """The split path is a flat 1D grid, so keep the program count under 65535."""
    return S * triton.cdiv(Q, _FIRST_SPLIT_BLOCK_Q) <= 65535


def _first_cfg(S, Q, esz=4, bwd=False):
    # Per-program launch cost dominates this op on Enflame: (4096, 256) fp16
    # costs 0.49 ms with 512 programs of 8 rows but 0.033 ms with 4 programs of
    # 1024 rows, so aim for ~256K elements per program -- but the gcu400 DSM
    # budget comes first: tiles beyond ~64K elements overflow the 917504 B
    # per-program DSM limit and fail compilation (see _dsm_tile_cap).
    bpe = (12 if bwd else 8) if esz >= 4 else (6 if bwd else 4)
    cap = _dsm_tile_cap(bpe)
    if Q <= 1024:
        bq = _p2(Q)
        bs = max(1, min(262144 // bq, _p2(S), 1024, cap // bq))
        return bs, bq, True
    bq = 512
    bs = max(1, min(65536 // bq, _p2(S), 128, cap // bq))
    # One program per slab block serialises the op when the reduced axis is
    # long: (64, 512, 512) dim=0 needed 2017 us with a single program while
    # (4096, 4096) dim=0 with 32 programs needs 85 us for the same element
    # count. Split the slab axis further until every SIP has work.
    # Round the S split *down*: rounding up (the old _p2) launched 16 programs
    # for S=1024 on a 24-SIP card.
    bs = max(1, min(bs, _p2floor(triton.cdiv(S, _NUM_SIPS))))
    return bs, bq, False


def _last_cfg(M, K):
    kfull = _p2(K)
    # Whole-reduction tile: when every row still fits one 64K-element tile the
    # kernel needs a single load/store pass instead of two reads.  Guarded by a
    # 64-column floor (narrower strips are far slower than the saved pass) and
    # by a program floor of _LAST_SINGLE_MIN_PROGRAMS.
    bm_full = _p2(M)
    if bm_full <= _LAST_SINGLE_MAX_ELEMS:
        bk_full = min(kfull, _LAST_SINGLE_MAX_ELEMS // bm_full)
        if bk_full >= 64 and triton.cdiv(K, bk_full) >= _LAST_SINGLE_MIN_PROGRAMS:
            return bm_full, bk_full, True
    elems = M * min(kfull, 64)
    # One tile covering the whole reduction keeps the tiny shapes to a single
    # load/store pass; anything bigger needs the chunked second pass.
    if M <= 2048 and elems <= 8192:
        bm = _p2(M)
        bk = max(1, min(kfull, max(1, 8192 // bm)))
        return bm, bk, True
    bk = min(kfull, 64)
    # 8K-element tiles measured 30.7 us on (1024, 1024) against 52.2 us for the
    # historical 4K tiles (aten 44.6 us).  A 256-row tile cuts that to 22.3 us.
    # Narrower column blocks (bk=32) lose 4% because a 32-column fp16 strip is
    # only 64 B per row.
    bm = max(1, min(16384 // bk, 256))
    return bm, bk, False




def _last_split_cfg(M, K, esz=4):
    """Chunking for the split-reduction last-axis forward (large M only).

    Measured best on (4096, 256): 8 row chunks of 512 with 256-row part tiles
    and 512-row scale tiles -> 0.058 ms (aten 0.062).  Small parts (the old
    BLOCK_M=8) cost 4-8x because every program then walks the row axis serially.
    On (10000, 65536) a 1024-wide contiguous column block is 2.3x faster than
    256 (2197 us vs 4953 us), so BLOCK_K follows K up to _LAST_MAX_BLOCK_K while
    the tile stays inside the gcu400 DSM budget (see _dsm_tile_cap).
    """
    cap = _dsm_tile_cap(12 if esz >= 4 else 8)
    nk0 = triton.cdiv(K, max(1, min(_p2(K), 256)))
    want = max(8, triton.cdiv(_LAST_SPLIT_PROGRAMS, nk0))
    chunk = max(8, _p2(triton.cdiv(M, want)))
    nchunk = triton.cdiv(M, chunk)
    bk = _last_block_k(K, nchunk)
    # Keep at least 1 KB of contiguous columns per row: a 256-column fp16 strip
    # is only 512 B and reaches 693 GB/s on (4096, 4096) where the fp32 strip
    # (1024 B) reaches 1.13 TB/s for the same tile shape.
    strip_bytes = 2048 if esz >= 4 else 1024
    bk = max(bk, min(_p2(K), max(256, strip_bytes // max(1, esz))))
    bm = min(chunk, 256, max(1, cap // bk))
    return bk, bm, chunk, nchunk


# --------------------------------------------------------------------------
# host wrappers
# --------------------------------------------------------------------------
def _forward(v, g, dim, need_norm):
    shape = v.shape
    rank = len(shape)
    d = dim % rank
    numel = v.numel()
    out = torch.empty(shape, dtype=v.dtype, device=v.device)
    if numel == 0:
        norm = (
            torch.empty(0, dtype=torch.float32, device=v.device) if need_norm else None
        )
        return out, norm
    gs = g.stride(d) if g.numel() > 1 else 0
    if d == 0:
        S = shape[0]
        Q = numel // S
        sN, sQ, ok = _layout_first(v)
        if not ok or S == 0:
            raise NotImplementedError("_weight_norm: unsupported stride layout")
        bs, bq, single = _first_cfg(S, Q, v.element_size())
        norm = None
        if Q >= _FIRST_SPLIT_MIN_Q and _first_split_ok(S, Q):
            # Long reduced axis: one program per slab block would walk the whole
            # inner axis serially, so split the sum over column chunks too.
            bqs, bss, ns, nq = _first_split_cfg(S, Q)
            # One allocation for the whole fp32 scratch (partial sums, scale,
            # optional norm): each torch.empty costs ~25 us on this backend,
            # while a slice view is nearly free.
            part, dsc, norm_s = _split_scratch(v.device, nq * S, S, need_norm)
            if need_norm:
                norm = norm_s
            _wn_fwd_first_part_kernel[(ns * nq,)](
                v,
                part,
                S,
                Q,
                sN,
                sQ,
                BLOCK_S=bss,
                BLOCK_Q=bqs,
                NS=ns,
            )
            bsf = min(_p2(S), 256)
            _wn_fwd_first_finish_kernel[(triton.cdiv(S, bsf),)](
                part,
                g,
                dsc,
                norm if need_norm else dsc,
                S,
                NQ=nq,
                gs=gs,
                BLOCK_S=bsf,
                BLOCK_C=min(64, _p2(nq)),
                STORE_NORM=need_norm,
            )
            _wn_fwd_first_scale_kernel[(ns * nq,)](
                v,
                dsc,
                out,
                S,
                Q,
                sN,
                sQ,
                BLOCK_S=bss,
                BLOCK_Q=bqs,
                NS=ns,
            )
            return out, norm
        if need_norm:
            norm = torch.empty(S, dtype=torch.float32, device=v.device)
        _wn_fwd_first_kernel[(triton.cdiv(S, bs),)](
            v,
            g,
            out,
            norm if need_norm else out,
            S,
            Q,
            sN,
            sQ,
            gs,
            BLOCK_S=bs,
            BLOCK_Q=bq,
            SINGLE=single,
            STORE_NORM=need_norm,
        )
        return out, norm
    if d == rank - 1:
        K = shape[rank - 1]
        M = numel // K
        sP, sN, ok = _layout_last(v)
        if not ok or K == 0:
            raise NotImplementedError("_weight_norm: unsupported stride layout")
        norm = None
        if M >= _LAST_SPLIT_MIN_M and K >= 64:
            esz = v.element_size()
            # Split the reduction over contiguous row chunks so that every
            # kernel works on fully contiguous [BLOCK_M, K] tiles; the tuning
            # column strips only reach a few percent of the memory rate.
            # Scale tiles must respect the gcu400 DSM budget: the historical
            # [512, 256] tile required 1572864 B (fp32) and overflowed.
            cap = _dsm_tile_cap(12 if esz >= 4 else 8)
            bk, bm, chunk, nchunk = _last_split_cfg(M, K, esz)
            # The scale kernel streams the same row-major data as the part
            # kernel, and the measured optimum matched its BLOCK_K on both
            # (4096, 4096) and (10000, 65536).
            bks = bk
            bms = min(_p2(M), 512, max(1, cap // bks))
            bkf = min(_p2(K), 256)
            part, dsc, norm_s = _split_scratch(v.device, nchunk * K, K, need_norm)
            if need_norm:
                norm = norm_s
            nk = triton.cdiv(K, bk)
            _wn_fwd_last_part_kernel[(nchunk * nk,)](
                v,
                part,
                M,
                K=K,
                sP=sP,
                sN=sN,
                CHUNK=chunk,
                BLOCK_M=bm,
                BLOCK_K=bk,
                NK=nk,
            )
            _wn_fwd_last_finish_kernel[(triton.cdiv(K, bkf),)](
                part,
                g,
                dsc,
                norm if need_norm else dsc,
                NCHUNK=nchunk,
                K=K,
                gs=gs,
                BLOCK_C=_p2(nchunk),
                BLOCK_K=bkf,
                STORE_NORM=need_norm,
            )
            nk = triton.cdiv(K, bks)
            nm = triton.cdiv(M, bms)
            _wn_fwd_last_scale_kernel[(nm * nk,)](
                v,
                dsc,
                out,
                M,
                K=K,
                sP=sP,
                sN=sN,
                BLOCK_M=bms,
                BLOCK_K=bks,
                NK=nk,
            )
            return out, norm
        if need_norm:
            norm = torch.empty(K, dtype=torch.float32, device=v.device)
        bm, bk, single = _last_cfg(M, K)
        # tiny single-tile shapes are latency bound: 1 warp beats 2 and 4
        # (0.020 vs 0.059 / 0.074 ms on (4, 256, 3) dim=2, fp16)
        _wn_fwd_last_kernel[(triton.cdiv(K, bk),)](
            v,
            g,
            out,
            norm if need_norm else out,
            M,
            K,
            sP,
            sN,
            gs,
            BLOCK_M=bm,
            BLOCK_K=bk,
            SINGLE=single,
            STORE_NORM=need_norm,
            num_warps=1 if (single and M <= 2048) else 4,
        )
        return out, norm
    P, Q, sP, sN, sQ, ok = _layout_mid(v, d)
    if not ok:
        raise NotImplementedError("_weight_norm: unsupported stride layout")
    N = shape[d]
    norm = torch.empty(N, dtype=torch.float32, device=v.device) if need_norm else None
    bt = 1024
    _wn_fwd_mid_kernel[(N,)](
        v,
        g,
        out,
        norm if need_norm else out,
        P,
        N,
        Q,
        sP,
        sN,
        sQ,
        gs,
        BLOCK_T=bt,
        STORE_NORM=need_norm,
    )
    return out, norm


def _backward(v, g, norm, grad, dim):
    shape = v.shape
    rank = len(shape)
    d = dim % rank
    numel = v.numel()
    vg = torch.empty(shape, dtype=v.dtype, device=v.device)
    gg = torch.empty(g.shape, dtype=g.dtype, device=g.device)
    if numel == 0:
        return vg, gg
    gs = g.stride(d) if g.numel() > 1 else 0
    ggs = 1
    for i in range(d + 1, rank):
        ggs *= g.shape[i]
    if d == 0:
        S = shape[0]
        Q = numel // S
        sN, sQ, ok = _layout_first(v)
        wsN, wsQ, okw = _layout_first(grad)
        if not (ok and okw) or S == 0:
            raise NotImplementedError("_weight_norm: unsupported stride layout")
        bs, bq, single = _first_cfg(S, Q, v.element_size(), bwd=True)
        _wn_bwd_first_kernel[(triton.cdiv(S, bs),)](
            v,
            grad,
            g,
            norm,
            vg,
            gg,
            S,
            Q,
            sN,
            sQ,
            wsN,
            wsQ,
            gs,
            ggs,
            BLOCK_S=bs,
            BLOCK_Q=bq,
            SINGLE=single,
        )
        return vg, gg
    if d == rank - 1:
        K = shape[rank - 1]
        M = numel // K
        sP, sN, ok = _layout_last(v)
        wsP, wsN, okw = _layout_last(grad)
        if not (ok and okw) or K == 0:
            raise NotImplementedError("_weight_norm: unsupported stride layout")
        bm, bk, single = _last_cfg(M, K)
        _wn_bwd_last_kernel[(triton.cdiv(K, bk),)](
            v,
            grad,
            g,
            norm,
            vg,
            gg,
            M,
            K,
            sP,
            sN,
            wsP,
            wsN,
            gs,
            ggs,
            BLOCK_M=bm,
            BLOCK_K=bk,
            SINGLE=single,
        )
        return vg, gg
    P, Q, sP, sN, sQ, ok = _layout_mid(v, d)
    _, _, wsP, wsN, wsQ, okw = _layout_mid(grad, d)
    N = shape[d]
    if not (ok and okw):
        raise NotImplementedError("_weight_norm: unsupported stride layout")
    _wn_bwd_mid_kernel[(N,)](
        v,
        grad,
        g,
        norm,
        vg,
        gg,
        P,
        N,
        Q,
        sP,
        sN,
        sQ,
        wsP,
        wsN,
        wsQ,
        gs,
        ggs,
        BLOCK_T=1024,
    )
    return vg, gg


class _WeightNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim):
        try:
            need = bool(ctx.needs_input_grad[0] or ctx.needs_input_grad[1])
        except Exception:
            need = True
        out, norm = _forward(v, g, dim, need)
        if need:
            ctx.dim = dim
            ctx.save_for_backward(v, g, norm)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, g, norm = ctx.saved_tensors
        vg, gg = _backward(v, g, norm, grad_out, ctx.dim)
        return vg, gg, None


def _weight_norm(v: torch.Tensor, g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """ATen entry point for the fused weight-normalization path."""
    logger.debug("GEMS_ENFLAME _WEIGHT_NORM")
    if v.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"_weight_norm does not support {v.dtype}")
    return _WeightNormFunction.apply(v, g, dim)
