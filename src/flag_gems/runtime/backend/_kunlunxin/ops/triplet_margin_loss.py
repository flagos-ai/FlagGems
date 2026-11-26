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
"""Kunlunxin implementation of aten::triplet_margin_loss.

The feature axis of one sample is walked by one program, which keeps a
one-dimensional accumulator per distance and reduces it along axis 0 exactly
once, outside every loop, writing either the per-sample loss or one distance
partial per (sample, column chunk). A second kernel merges the column
partials into per-sample losses when the feature axis was split, and a final
kernel folds the per-sample losses into the scalar a mean/sum reduction needs.

The reduction is kept outside the loops on purpose: a reduce nested in a
loop over samples is not legalized by this backend, while a single trailing
reduce over a one-dimensional accumulator is.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.triplet_margin_loss import _lead_groups, _mode_of
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)

# Columns per loop iteration. The whole 3-operand working set stays in
# registers at this width.
_BC_MAX = 2048
# Samples per program when the column partials are merged. Kept small: a
# k-lane masked store is not lowered above this width.
_RB_MAX = 64
# Column-partial programs the split aims for when the row count is low.
_TARGET_PROGS = 1024
# Partials per loop iteration in the final fold.
_FOLD_BLOCK = 2048
_MAX_LEAD = 4
# Below this sample count the 2-D tile form is the cheaper shape.
_SMALL_R = 128
# From this sample count on, samples are folded into one program.
_RPP_MIN_ROWS = 256


def _cdiv(x, y):
    return (x + y - 1) // y


def _broadcast_shape(*shapes):
    """Broadcast leading shapes without a torch compute call."""
    ndim = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(ndim):
        dim = 1
        for s in shapes:
            j = i - (ndim - len(s))
            v = s[j] if j >= 0 else 1
            if v == 1:
                continue
            if dim == 1:
                dim = v
            elif dim != v:
                raise RuntimeError(
                    "triplet_margin_loss: shapes cannot be broadcast together"
                )
        out.append(dim)
    return tuple(out)


def _fold_block(rows):
    """Fold tile sized to the sample count: a wide tile is wasted work here."""
    return max(64, min(_FOLD_BLOCK, triton.next_power_of_2(max(rows, 1))))


_NOS = [
    "R",
    "C",
    "chunk",
    "GC",
    "eps",
    "pp",
    "margin",
]


@triton.jit
def _acc_step(acc, d, m, MODE: tl.constexpr, pp, MASKED: tl.constexpr):
    if MASKED:
        if MODE == 2:
            out = tl.where(m, d * d, 0.0)
        elif MODE == 1:
            out = tl.where(m, tl.abs(d), 0.0)
        elif MODE == 0:
            out = tl.where(m, (d != 0.0).to(tl.float32), 0.0)
        elif MODE == 3:
            out = tl.where(m, tl.abs(d), -float("inf"))
        elif MODE == 4:
            out = tl.where(m, tl.abs(d), float("inf"))
        else:
            out = tl.where(m, tl.exp2(pp * tl.log2(tl.abs(d))), 0.0)
    else:
        # Every lane of the tile is in range, so the per-element select that
        # carries the predicate is pure overhead: it lowers to a compare plus
        # a select for all 3 * BC values of the tile.
        if MODE == 2:
            out = d * d
        elif MODE == 1:
            out = tl.abs(d)
        elif MODE == 0:
            out = (d != 0.0).to(tl.float32)
        elif MODE == 3:
            out = tl.abs(d)
        elif MODE == 4:
            out = tl.abs(d)
        else:
            out = tl.exp2(pp * tl.log2(tl.abs(d)))
    if MODE == 3:
        return tl.maximum(acc, out)
    if MODE == 4:
        return tl.minimum(acc, out)
    return acc + out


@triton.jit
def _dist_from_sum(s, pp, MODE: tl.constexpr):
    if MODE == 2:
        d = tl.sqrt(s)
    elif MODE == 5:
        d = tl.exp2(tl.log2(s) / pp)
    else:
        d = s
    return d


@triton.jit(do_not_specialize=["R", "eps", "margin"])
def _tm_small_kernel(
    a_ptr,
    p_ptr,
    n_ptr,
    o_ptr,
    R,
    C,
    eps,
    margin,
    RB: tl.constexpr,
    BC: tl.constexpr,
    SWAP: tl.constexpr,
):
    """Several samples per program as a dense 2-D tile.

    Only used when the feature axis fits one tile and the sample axis divides
    the tile height: a reduce along the tile's own last axis is a single
    reduction step for the whole tile, which is what the tiny shapes need
    (they are program-count bound, and the reduce is the per-program cost).
    """
    pid = tl.program_id(0)
    r = pid * RB + tl.arange(0, RB)
    c = tl.arange(0, BC)
    off = r[:, None] * C + c[None, :]
    av = tl.load(a_ptr + off).to(tl.float32)
    pv = tl.load(p_ptr + off).to(tl.float32)
    nv = tl.load(n_ptr + off).to(tl.float32)
    dap = av - pv + eps
    dan = av - nv + eps
    s_ap = tl.sum(dap * dap, axis=1)
    s_an = tl.sum(dan * dan, axis=1)
    if SWAP:
        dpn = pv - nv + eps
        s_pn = tl.sum(dpn * dpn, axis=1)
        d_an = tl.minimum(tl.sqrt(s_an), tl.sqrt(s_pn))
    else:
        d_an = tl.sqrt(s_an)
    loss = tl.maximum(margin + tl.sqrt(s_ap) - d_an, 0.0)
    tl.store(o_ptr + r, loss.to(o_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["C", "chunk", "GC", "eps", "pp", "margin"])
def _tm_tiny_kernel(
    a_ptr,
    p_ptr,
    n_ptr,
    o_ptr,
    part_ptr,
    R,
    C,
    chunk,
    GC,
    eps,
    pp,
    margin,
    g0s,
    g1s,
    g2s,
    g3s,
    g0a,
    g1a,
    g2a,
    g3a,
    g0p,
    g1p,
    g2p,
    g3p,
    g0n,
    g1n,
    g2n,
    g3n,
    MODE: tl.constexpr,
    SWAP: tl.constexpr,
    COMBINE: tl.constexpr,
):
    """Scalar form for a short feature axis: one lane walks one sample.

    For a short feature axis the scalar loop is both cheap and unambiguous,
    and it keeps the three-distance form (swap) on the same straightforward
    path instead of a vector reduction over a partial lane set.
    """
    pid = tl.program_id(0)
    r = pid // GC
    gc = pid % GC
    c_start = gc * chunk
    c_end = tl.minimum(c_start + chunk, C)

    rem = r
    off_a = 0
    off_p = 0
    off_n = 0
    idx = rem % g0s
    rem = rem // g0s
    off_a += idx * g0a
    off_p += idx * g0p
    off_n += idx * g0n
    idx = rem % g1s
    rem = rem // g1s
    off_a += idx * g1a
    off_p += idx * g1p
    off_n += idx * g1n
    idx = rem % g2s
    rem = rem // g2s
    off_a += idx * g2a
    off_p += idx * g2p
    off_n += idx * g2n
    idx = rem % g3s
    off_a += idx * g3a
    off_p += idx * g3p
    off_n += idx * g3n

    if MODE == 3:
        s_ap = -float("inf")
        s_an = -float("inf")
        s_pn = -float("inf")
    elif MODE == 4:
        s_ap = float("inf")
        s_an = float("inf")
        s_pn = float("inf")
    else:
        s_ap = 0.0
        s_an = 0.0
        s_pn = 0.0
    for c in range(c_start, c_end):
        av = tl.load(a_ptr + off_a + c).to(tl.float32)
        pv = tl.load(p_ptr + off_p + c).to(tl.float32)
        nv = tl.load(n_ptr + off_n + c).to(tl.float32)
        s_ap = _acc_step(s_ap, av - pv + eps, 0, MODE, pp, False)
        s_an = _acc_step(s_an, av - nv + eps, 0, MODE, pp, False)
        if SWAP:
            s_pn = _acc_step(s_pn, pv - nv + eps, 0, MODE, pp, False)
    if COMBINE:
        pos = (r * GC + gc) * 3
        tl.store(part_ptr + pos, s_ap)
        tl.store(part_ptr + pos + 1, s_an)
        if SWAP:
            tl.store(part_ptr + pos + 2, s_pn)
    else:
        d_ap = _dist_from_sum(s_ap, pp, MODE)
        d_an = _dist_from_sum(s_an, pp, MODE)
        if SWAP:
            d_an = tl.minimum(d_an, _dist_from_sum(s_pn, pp, MODE))
        loss = tl.maximum(margin + d_ap - d_an, 0.0)
        tl.store(o_ptr + r, loss.to(o_ptr.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=_NOS)
def _tm_rows_kernel(
    a_ptr,
    p_ptr,
    n_ptr,
    o_ptr,
    part_ptr,
    R,
    C,
    chunk,
    GC,
    eps,
    pp,
    margin,
    g0s,
    g1s,
    g2s,
    g3s,
    g0a,
    g1a,
    g2a,
    g3a,
    g0p,
    g1p,
    g2p,
    g3p,
    g0n,
    g1n,
    g2n,
    g3n,
    MODE: tl.constexpr,
    SWAP: tl.constexpr,
    BC: tl.constexpr,
    COMBINE: tl.constexpr,
    ALIGN: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid // GC
    gc = pid % GC
    c_start = gc * chunk
    c_end = tl.minimum(c_start + chunk, C)

    # Flatten the leading axes into one sample index and recover each source
    # offset from the stride groups: consecutive leading axes that share a
    # stride pattern collapse into one group.
    rem = r
    off_a = 0
    off_p = 0
    off_n = 0
    idx = rem % g0s
    rem = rem // g0s
    off_a += idx * g0a
    off_p += idx * g0p
    off_n += idx * g0n
    idx = rem % g1s
    rem = rem // g1s
    off_a += idx * g1a
    off_p += idx * g1p
    off_n += idx * g1n
    idx = rem % g2s
    rem = rem // g2s
    off_a += idx * g2a
    off_p += idx * g2p
    off_n += idx * g2n
    idx = rem % g3s
    off_a += idx * g3a
    off_p += idx * g3p
    off_n += idx * g3n

    cols = tl.arange(0, BC)
    if MODE == 4:
        acc_ap = tl.full([BC], float("inf"), tl.float32)
        acc_an = tl.full([BC], float("inf"), tl.float32)
    elif MODE == 3:
        acc_ap = tl.full([BC], -float("inf"), tl.float32)
        acc_an = tl.full([BC], -float("inf"), tl.float32)
    else:
        acc_ap = tl.zeros([BC], tl.float32)
        acc_an = tl.zeros([BC], tl.float32)
    if SWAP:
        if MODE == 4:
            acc_pn = tl.full([BC], float("inf"), tl.float32)
        elif MODE == 3:
            acc_pn = tl.full([BC], -float("inf"), tl.float32)
        else:
            acc_pn = tl.zeros([BC], tl.float32)

    for c0 in range(c_start, c_end, BC):
        cc = c0 + cols
        m = cc < c_end
        if ALIGN:
            av = tl.load(a_ptr + off_a + cc).to(tl.float32)
            pv = tl.load(p_ptr + off_p + cc).to(tl.float32)
            nv = tl.load(n_ptr + off_n + cc).to(tl.float32)
        else:
            av = tl.load(a_ptr + off_a + cc, mask=m, other=0.0).to(tl.float32)
            pv = tl.load(p_ptr + off_p + cc, mask=m, other=0.0).to(tl.float32)
            nv = tl.load(n_ptr + off_n + cc, mask=m, other=0.0).to(tl.float32)
        acc_ap = _acc_step(acc_ap, av - pv + eps, m, MODE, pp, not ALIGN)
        acc_an = _acc_step(acc_an, av - nv + eps, m, MODE, pp, not ALIGN)
        if SWAP:
            acc_pn = _acc_step(acc_pn, pv - nv + eps, m, MODE, pp, not ALIGN)

    if MODE == 3:
        s_ap = tl.max(acc_ap, 0)
        s_an = tl.max(acc_an, 0)
    elif MODE == 4:
        s_ap = tl.min(acc_ap, 0)
        s_an = tl.min(acc_an, 0)
    else:
        s_ap = tl.sum(acc_ap, 0)
        s_an = tl.sum(acc_an, 0)
    if SWAP:
        if MODE == 3:
            s_pn = tl.max(acc_pn, 0)
        elif MODE == 4:
            s_pn = tl.min(acc_pn, 0)
        else:
            s_pn = tl.sum(acc_pn, 0)
    else:
        s_pn = 0.0

    if COMBINE:
        pos = (r * GC + gc) * 3
        tl.store(part_ptr + pos, s_ap)
        tl.store(part_ptr + pos + 1, s_an)
        if SWAP:
            tl.store(part_ptr + pos + 2, s_pn)
    else:
        d_ap = _dist_from_sum(s_ap, pp, MODE)
        d_an = _dist_from_sum(s_an, pp, MODE)
        if SWAP:
            d_an = tl.minimum(d_an, _dist_from_sum(s_pn, pp, MODE))
        loss = tl.maximum(margin + d_ap - d_an, 0.0)
        tl.store(o_ptr + r, loss.to(o_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["C", "chunk", "eps", "pp", "margin"])
def _tm_rows_multi_kernel(
    a_ptr,
    p_ptr,
    n_ptr,
    o_ptr,
    C,
    chunk,
    eps,
    pp,
    margin,
    g0s,
    g1s,
    g2s,
    g3s,
    g0a,
    g1a,
    g2a,
    g3a,
    g0p,
    g1p,
    g2p,
    g3p,
    g0n,
    g1n,
    g2n,
    g3n,
    MODE: tl.constexpr,
    SWAP: tl.constexpr,
    BC: tl.constexpr,
    RPP: tl.constexpr,
    NC: tl.constexpr,
):
    """``RPP`` consecutive samples per program, one sample at a time.

    The sample loop is a runtime loop on purpose: the per-program cost (launch
    plus prologue) dominates when a sample is only one tile wide, so several
    samples share one program while each keeps its own accumulator.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BC)
    for j in range(0, RPP):
        r = pid * RPP + j
        rem = r
        off_a = 0
        off_p = 0
        off_n = 0
        idx = rem % g0s
        rem = rem // g0s
        off_a += idx * g0a
        off_p += idx * g0p
        off_n += idx * g0n
        idx = rem % g1s
        rem = rem // g1s
        off_a += idx * g1a
        off_p += idx * g1p
        off_n += idx * g1n
        idx = rem % g2s
        rem = rem // g2s
        off_a += idx * g2a
        off_p += idx * g2p
        off_n += idx * g2n
        idx = rem % g3s
        off_a += idx * g3a
        off_p += idx * g3p
        off_n += idx * g3n

        if MODE == 4:
            acc_ap = tl.full([BC], float("inf"), tl.float32)
            acc_an = tl.full([BC], float("inf"), tl.float32)
        elif MODE == 3:
            acc_ap = tl.full([BC], -float("inf"), tl.float32)
            acc_an = tl.full([BC], -float("inf"), tl.float32)
        else:
            acc_ap = tl.zeros([BC], tl.float32)
            acc_an = tl.zeros([BC], tl.float32)
        if SWAP:
            if MODE == 4:
                acc_pn = tl.full([BC], float("inf"), tl.float32)
            elif MODE == 3:
                acc_pn = tl.full([BC], -float("inf"), tl.float32)
            else:
                acc_pn = tl.zeros([BC], tl.float32)

        for k in tl.static_range(NC):
            cc = k * BC + cols
            av = tl.load(a_ptr + off_a + cc).to(tl.float32)
            pv = tl.load(p_ptr + off_p + cc).to(tl.float32)
            nv = tl.load(n_ptr + off_n + cc).to(tl.float32)
            acc_ap = _acc_step(acc_ap, av - pv + eps, cc < 0, MODE, pp, False)
            acc_an = _acc_step(acc_an, av - nv + eps, cc < 0, MODE, pp, False)
            if SWAP:
                acc_pn = _acc_step(acc_pn, pv - nv + eps, cc < 0, MODE, pp, False)

        if MODE == 3:
            s_ap = tl.max(acc_ap, 0)
            s_an = tl.max(acc_an, 0)
            s_pn = tl.max(acc_pn, 0) if SWAP else 0.0
        elif MODE == 4:
            s_ap = tl.min(acc_ap, 0)
            s_an = tl.min(acc_an, 0)
            s_pn = tl.min(acc_pn, 0) if SWAP else 0.0
        else:
            s_ap = tl.sum(acc_ap, 0)
            s_an = tl.sum(acc_an, 0)
            s_pn = tl.sum(acc_pn, 0) if SWAP else 0.0

        d_ap = _dist_from_sum(s_ap, pp, MODE)
        d_an = _dist_from_sum(s_an, pp, MODE)
        if SWAP:
            d_an = tl.minimum(d_an, _dist_from_sum(s_pn, pp, MODE))
        loss = tl.maximum(margin + d_ap - d_an, 0.0)
        tl.store(o_ptr + r, loss.to(o_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["R", "GC", "margin", "pp", "scale"])
def _tm_combine_kernel(
    part_ptr,
    o_ptr,
    R,
    GC,
    margin,
    pp,
    scale,
    MODE: tl.constexpr,
    SWAP: tl.constexpr,
    RB: tl.constexpr,
    FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * RB + tl.arange(0, RB)
    rmask = r < R
    r_safe = tl.minimum(r, R - 1)

    if MODE == 4:
        acc_ap = tl.full([RB], float("inf"), tl.float32)
        acc_an = tl.full([RB], float("inf"), tl.float32)
        acc_pn = tl.full([RB], float("inf"), tl.float32)
    elif MODE == 3:
        acc_ap = tl.full([RB], -float("inf"), tl.float32)
        acc_an = tl.full([RB], -float("inf"), tl.float32)
        acc_pn = tl.full([RB], -float("inf"), tl.float32)
    else:
        acc_ap = tl.zeros([RB], tl.float32)
        acc_an = tl.zeros([RB], tl.float32)
        acc_pn = tl.zeros([RB], tl.float32)

    for gc in range(0, GC):
        base = (r_safe * GC + gc) * 3
        ap = tl.load(part_ptr + base).to(tl.float32)
        an = tl.load(part_ptr + base + 1).to(tl.float32)
        if MODE == 3:
            acc_ap = tl.maximum(acc_ap, tl.where(rmask, ap, -float("inf")))
            acc_an = tl.maximum(acc_an, tl.where(rmask, an, -float("inf")))
        elif MODE == 4:
            acc_ap = tl.minimum(acc_ap, tl.where(rmask, ap, float("inf")))
            acc_an = tl.minimum(acc_an, tl.where(rmask, an, float("inf")))
        else:
            acc_ap += tl.where(rmask, ap, 0.0)
            acc_an += tl.where(rmask, an, 0.0)
        if SWAP:
            pn = tl.load(part_ptr + base + 2).to(tl.float32)
            if MODE == 3:
                acc_pn = tl.maximum(acc_pn, tl.where(rmask, pn, -float("inf")))
            elif MODE == 4:
                acc_pn = tl.minimum(acc_pn, tl.where(rmask, pn, float("inf")))
            else:
                acc_pn += tl.where(rmask, pn, 0.0)

    d_ap = _dist_from_sum(acc_ap, pp, MODE)
    d_an = _dist_from_sum(acc_an, pp, MODE)
    if SWAP:
        d_an = tl.minimum(d_an, _dist_from_sum(acc_pn, pp, MODE))
    loss = tl.maximum(margin + d_ap - d_an, 0.0)
    dt = o_ptr.dtype.element_ty
    if FINAL:
        tl.store(o_ptr, (tl.sum(tl.where(rmask, loss, 0.0)) * scale).to(dt))
    else:
        tl.store(o_ptr + r, loss.to(dt), mask=rmask)


@triton.jit(do_not_specialize=["n", "scale"])
def _tm_fold_kernel(sums_ptr, o_ptr, n, scale, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], tl.float32)
    cols = tl.arange(0, BLOCK)
    for i0 in range(0, n, BLOCK):
        ii = i0 + cols
        acc += tl.where(ii < n, tl.load(sums_ptr + ii, mask=ii < n, other=0.0), 0.0)
    tl.store(o_ptr, (tl.sum(acc, 0) * scale).to(o_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Row-block path
#
# Each program streams a [XBLOCK, YBLOCK] tile of the three inputs once and
# reduces every row block in place, then writes the per-sample loss. A second
# small kernel folds those losses into the requested reduction. The pointer
# form above remains the fallback for the cases listed in the caller.
# ---------------------------------------------------------------------------
_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}
_TLE_NPROG = 32
# Latched when the toolchain cannot compile the tle row kernel: the call falls
# through to the pointer path, and later calls skip the attempt.
_TLE_BROKEN = False
_TLE_FOLD_BLOCK = 1024
_TLE_MIN_ROWS = 256
_TLE_MIN_COLS = 16


def _tml_tle_geom(R, C):
    """Rows/columns per program: keep the tile at 32K elements."""
    xb = 128 if R <= 8192 else 256
    yb = min(triton.next_power_of_2(C), 32768 // xb)
    return xb, max(16, yb)


@triton.jit(
    do_not_specialize=["N", "eps", "margin"],
    do_not_specialize_on_alignment=["a_desc", "p_desc", "n_desc", "o_desc"],
)
def _tml_tle_rows_kernel(
    a_desc,
    p_desc,
    n_desc,
    o_desc,
    N,
    eps,
    margin,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * XBLOCK
    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    b_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem)
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    b_ptrs = tle.gpu.local_ptr(b_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))
    acc_ap = tl.zeros([XBLOCK], tl.float32)
    acc_an = tl.zeros([XBLOCK], tl.float32)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
                tl.store(b_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        tle.gpu.copy(p_desc, b_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        av = tl.load(a_ptrs).to(tl.float32)
        pv = tl.load(b_ptrs).to(tl.float32)
        d = av - pv + eps
        acc_ap += tl.sum(d * d, axis=1)
        tle.gpu.copy(n_desc, b_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        av = tl.load(a_ptrs).to(tl.float32)
        nv = tl.load(b_ptrs).to(tl.float32)
        d = av - nv + eps
        acc_an += tl.sum(d * d, axis=1)
    loss = tl.maximum(margin + tl.sqrt(acc_ap) - tl.sqrt(acc_an), 0.0)
    tl.store(c_ptrs, loss)
    tle.gpu.copy(c_lmem, o_desc, [XBLOCK], [row_off])


@triton.jit(do_not_specialize=["R"])
def _tml_tle_sum_partial(
    loss, part, R, BLOCK: tl.constexpr, NPROG: tl.constexpr, ALIGN: tl.constexpr
):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], tl.float32)
    nb = tl.cdiv(R, BLOCK)
    for b in range(pid, nb, NPROG):
        i = b * BLOCK + tl.arange(0, BLOCK)
        if ALIGN:
            acc += tl.load(loss + i)
        else:
            m = i < R
            acc += tl.load(loss + i, mask=m, other=0.0)
    tl.store(part + pid, tl.sum(acc, 0))


@triton.jit(do_not_specialize=["scale"])
def _tml_tle_final_fold(part, out, scale, N: tl.constexpr):
    i = tl.arange(0, N)
    m = i < N
    tl.store(
        out,
        (tl.sum(tl.load(part + i, mask=m, other=0.0), 0) * scale).to(
            out.dtype.element_ty
        ),
    )


def _tml_tle_run(a, b, c, loss, out, R, C, eps, margin, scale):
    xb, yb = _tml_tle_geom(R, C)
    ad = TensorDescriptor.from_tensor(a.view(R, C), block_shape=[xb, yb])
    bd = TensorDescriptor.from_tensor(b.view(R, C), block_shape=[xb, yb])
    nd = TensorDescriptor.from_tensor(c.view(R, C), block_shape=[xb, yb])
    od = TensorDescriptor.from_tensor(loss, block_shape=[xb])
    part = torch.empty((_TLE_NPROG,), dtype=torch.float32, device=a.device)
    grid = _cdiv(R, xb)
    _tml_tle_rows_kernel[(grid,)](
        ad,
        bd,
        nd,
        od,
        C,
        eps,
        margin,
        XBLOCK=xb,
        YBLOCK=yb,
        IN_DTYPE=_TLE_TL_DTYPE[a.dtype],
        NEED_ZERO=(C % yb) != 0,
        num_warps=4,
    )
    _tml_tle_sum_partial[(_TLE_NPROG,)](
        loss,
        part,
        R,
        BLOCK=_TLE_FOLD_BLOCK,
        NPROG=_TLE_NPROG,
        ALIGN=(R % _TLE_FOLD_BLOCK) == 0,
        num_warps=4,
    )
    _tml_tle_final_fold[(1,)](part, out, scale, N=_TLE_NPROG, num_warps=4)


def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
    p=2.0,
    eps=1e-6,
    swap=False,
    reduction="mean",
):
    logger.debug("GEMS_KUNLUNXIN TRIPLET_MARGIN_LOSS")
    global _TLE_BROKEN

    if isinstance(reduction, str):
        red = {"none": 0, "mean": 1, "sum": 2}[reduction.lower()]
    elif reduction is None:
        red = 0
    else:
        red = int(reduction)
    if red not in (0, 1, 2):
        raise ValueError("triplet_margin_loss: invalid reduction")

    shape = _broadcast_shape(
        tuple(anchor.shape), tuple(positive.shape), tuple(negative.shape)
    )
    out_dtype = torch.promote_types(
        torch.promote_types(anchor.dtype, positive.dtype), negative.dtype
    )
    a = anchor if tuple(anchor.shape) == shape else anchor.expand(shape)
    b = positive if tuple(positive.shape) == shape else positive.expand(shape)
    c = negative if tuple(negative.shape) == shape else negative.expand(shape)
    dev = anchor.device
    # The backend does not take float64 pointers, so a float64 result is
    # computed in float32 and widened on the host.
    res_dtype = out_dtype
    if out_dtype == torch.float64:
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        c = c.to(torch.float32)
        out_dtype = torch.float32

    if shape:
        C = shape[-1]
        R = 1
        for s in shape[:-1]:
            R *= s
        out_shape = tuple(shape[:-1])
    else:
        C = 1
        R = 1
        out_shape = ()

    pval = float(p)
    mode = _mode_of(pval)
    scale = (1.0 / R) if red == 1 and R > 0 else 1.0

    if red == 0:
        out = torch.empty(out_shape, dtype=out_dtype, device=dev)
        rows = out.reshape(-1) if R > 0 else None
    else:
        out = torch.empty((), dtype=out_dtype, device=dev)
        rows = torch.empty((R,), dtype=torch.float32, device=dev) if R > 0 else None
    if res_dtype != out_dtype:
        result = torch.empty(out_shape if red == 0 else (), dtype=res_dtype, device=dev)
    else:
        result = out

    if R <= 0:
        return result
    if C <= 0:
        # An empty feature axis makes every distance zero, so ATen returns
        # clamp_min(margin, 0) for each sample.
        loss = float(margin) if margin > 0 else 0.0
        result.fill_(loss * R if red == 2 else loss)
        return result

    groups = _lead_groups([a, b, c], shape)
    if len(groups) > _MAX_LEAD:
        raise RuntimeError(
            "triplet_margin_loss: unsupported stride pattern "
            f"({len(groups)} leading stride groups)"
        )
    gs = [1] * _MAX_LEAD
    ga = [0] * _MAX_LEAD
    gp = [0] * _MAX_LEAD
    gn = [0] * _MAX_LEAD
    for k, (size, strides) in enumerate(groups):
        gs[k] = int(size)
        ga[k] = int(strides[0])
        gp[k] = int(strides[1])
        gn[k] = int(strides[2])

    if (
        _HAS_TLE
        and not _TLE_BROKEN
        and red in (1, 2)
        and mode == 2
        and not swap
        and R >= _TLE_MIN_ROWS
        and C >= _TLE_MIN_COLS
        and out_dtype in _TLE_TL_DTYPE
        and tuple(anchor.shape) == shape
        and tuple(positive.shape) == shape
        and tuple(negative.shape) == shape
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
    ):
        try:
            with torch_device_fn.device(dev):
                _tml_tle_run(
                    a,
                    b,
                    c,
                    rows,
                    out,
                    R,
                    C,
                    float(eps),
                    float(margin),
                    (1.0 / R) if red == 1 else 1.0,
                )
            if result is not out:
                result.copy_(out)
            return result
        except Exception:  # noqa: BLE001
            _TLE_BROKEN = True

    bc = min(triton.next_power_of_2(C), _BC_MAX)
    # Split the feature axis only while there are few samples: one program
    # already keeps the pipeline busy once there is a row per program.
    gc = 1
    if R < _TARGET_PROGS:
        gc = min(_cdiv(_TARGET_PROGS, R), max(1, _cdiv(C, bc)))
    chunk = _cdiv(_cdiv(C, gc), bc) * bc
    gc = _cdiv(C, chunk)

    # Distance kernel: one program per (sample, column chunk).
    if gc == 1 and red == 0:
        dst = rows
        combine = False
        part = rows
    else:
        combine = gc > 1
        dst = rows
        part = (
            torch.empty((R * gc * 3,), dtype=torch.float32, device=dev)
            if combine
            else rows
        )

    swap_i = bool(swap)
    n_progs = R * gc
    # Few samples with a tile-sized feature axis: the 2-D form reduces a whole
    # tile at once, which is cheaper than one reduce per sample here.
    rb = 2 if R % 2 == 0 else 1
    if (
        mode == 2
        and gc == 1
        and C == bc
        and C >= 32
        and not (swap_i and bc < 256)
        and R <= _SMALL_R
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
    ):
        with torch_device_fn.device(dev):
            _tm_small_kernel[(R // rb,)](
                a,
                b,
                c,
                dst,
                R,
                C,
                float(eps),
                float(margin),
                RB=rb,
                BC=bc,
                SWAP=swap_i,
                num_warps=4,
            )
            if red != 0:
                _tm_fold_kernel[(1,)](
                    rows,
                    out,
                    R,
                    scale,
                    BLOCK=_fold_block(R),
                    num_warps=4,
                )
        if result is not out:
            result.copy_(out)
        return result
    rpp = 8 if (R % 8 == 0 and R >= _RPP_MIN_ROWS) else 1
    if gc == 1 and rpp > 1 and (C % bc) == 0 and not (swap_i and bc < 256):
        with torch_device_fn.device(dev):
            _tm_rows_multi_kernel[(R // rpp,)](
                a,
                b,
                c,
                dst,
                C,
                chunk,
                float(eps),
                pval,
                float(margin),
                gs[0],
                gs[1],
                gs[2],
                gs[3],
                ga[0],
                ga[1],
                ga[2],
                ga[3],
                gp[0],
                gp[1],
                gp[2],
                gp[3],
                gn[0],
                gn[1],
                gn[2],
                gn[3],
                MODE=mode,
                SWAP=swap_i,
                BC=bc,
                RPP=rpp,
                NC=_cdiv(C, bc),
                num_warps=4,
            )
            if red != 0:
                _tm_fold_kernel[(1,)](
                    rows,
                    out,
                    R,
                    scale,
                    BLOCK=_fold_block(R),
                    num_warps=4,
                )
        if result is not out:
            result.copy_(out)
        return result
    nw = 1 if bc < 32 else 4
    with torch_device_fn.device(dev):
        if bc < 32 or (swap_i and bc < 256):
            _tm_tiny_kernel[(n_progs,)](
                a,
                b,
                c,
                dst,
                part,
                R,
                C,
                chunk,
                gc,
                float(eps),
                pval,
                float(margin),
                gs[0],
                gs[1],
                gs[2],
                gs[3],
                ga[0],
                ga[1],
                ga[2],
                ga[3],
                gp[0],
                gp[1],
                gp[2],
                gp[3],
                gn[0],
                gn[1],
                gn[2],
                gn[3],
                MODE=mode,
                SWAP=swap_i,
                COMBINE=combine,
                num_warps=1,
            )
        else:
            _tm_rows_kernel[(n_progs,)](
                a,
                b,
                c,
                dst,
                part,
                R,
                C,
                chunk,
                gc,
                float(eps),
                pval,
                float(margin),
                gs[0],
                gs[1],
                gs[2],
                gs[3],
                ga[0],
                ga[1],
                ga[2],
                ga[3],
                gp[0],
                gp[1],
                gp[2],
                gp[3],
                gn[0],
                gn[1],
                gn[2],
                gn[3],
                MODE=mode,
                SWAP=swap_i,
                BC=bc,
                COMBINE=combine,
                ALIGN=(C % chunk) == 0,
                num_warps=nw,
            )
        if combine:
            rb = max(1, min(_RB_MAX, triton.next_power_of_2(R)))
            n_comb = _cdiv(R, rb)
            if red == 0:
                _tm_combine_kernel[(n_comb,)](
                    part,
                    out,
                    R,
                    gc,
                    float(margin),
                    pval,
                    scale,
                    MODE=mode,
                    SWAP=swap_i,
                    RB=rb,
                    FINAL=False,
                    num_warps=4,
                )
            elif n_comb == 1:
                _tm_combine_kernel[(1,)](
                    part,
                    out,
                    R,
                    gc,
                    float(margin),
                    pval,
                    scale,
                    MODE=mode,
                    SWAP=swap_i,
                    RB=rb,
                    FINAL=True,
                    num_warps=4,
                )
            else:
                _tm_combine_kernel[(n_comb,)](
                    part,
                    rows,
                    R,
                    gc,
                    float(margin),
                    pval,
                    1.0,
                    MODE=mode,
                    SWAP=swap_i,
                    RB=rb,
                    FINAL=False,
                    num_warps=4,
                )
        if red != 0 and (not combine or n_comb > 1):
            _tm_fold_kernel[(1,)](
                rows,
                out,
                R,
                scale,
                BLOCK=_fold_block(R),
                num_warps=4,
            )

    if result is not out:
        result.copy_(out)
    return result
