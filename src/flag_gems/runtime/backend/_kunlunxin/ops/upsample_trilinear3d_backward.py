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
"""Kunlunxin (XPU3) override for ``aten::upsample_trilinear3d_backward``.

The upstream generic kernel scatters each output voxel into its eight input
corners with ``tl.atomic_add``.  On XPU3 that path is both numerically wrong in
the official test matrix (lost/duplicated updates -> O(1) absolute errors on
~0.2 % of the elements) and unsafe for the benchmark shapes (the benchmark
process aborts around the first large shape, consistent with the masked-atomic
out-of-bounds hazard that is documented for this backend).

This override computes the exact same result without atomics.  The trilinear
weight is separable,

    grad_in[nc, i, j, k] = sum_{o,p,q} grad_out[nc, o, p, q]
                           * wd(o, i) * wh(p, j) * ww(q, k)

so the backward is evaluated as three deterministic gather passes, one per
axis (W -> H -> D) over fp32 intermediates.  Each pass is a "reduce one axis"
kernel whose per-lane unrolled window is ``floor(2/scale) + 3`` positions
(<= 11 for the official matrix; 7 for the exact-2x benchmark matrix).  A single fused 3-D gather is *not* usable on
this backend: unrolling ``Kd*Kh*Kw`` (up to 120 loads) per lane overflows the
local-memory budget (``Failed to tune buffer size`` / ``uni_sram``), and a
dynamic nested loop trips ``TritonXPUUnrollControl``.

The per-axis weight is reproduced the way ATen computes it
(``area_pixel_compute_source_index`` + ``i0/i1`` lambda split, including the
``input_index0 == input_index1`` top-edge duplicate).  The weight formula was
verified against ``torch.ops.aten.upsample_trilinear3d_backward`` on CPU to
float32 round-off for align_corners in {True, False} and up/down/mixed scales.

All indices are int32; every gather address is clamped so that rejected
candidates and tail lanes always touch in-bounds memory (masked loads are not
honoured on this backend).  Outputs are over-allocated to a whole tile and tail
lanes recompute element 0 (harmless duplicate write).
"""

import logging
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK = 1024


@libentry()
@triton.jit
def _axis_reduce_kernel(
    src,
    dst,
    total,
    IN_SIZE,
    OUT_SIZE,
    OUTER_STR_IN,
    AXIS_STR_IN,
    INNER,
    S,
    B,
    K: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One lane per output element; gather over the window of one source axis.

    ``src`` is logically ``(outer, IN_SIZE, inner)`` and ``dst`` is
    ``(outer, OUT_SIZE, inner)``.  Both are contiguous, so the only strides
    needed are ``OUTER_STR_IN`` (outer stride of ``src``) and ``AXIS_STR_IN``.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    k = offsets % INNER
    t = offsets // INNER
    o = t % OUT_SIZE
    outer = t // OUT_SIZE

    # Tail lanes decode to garbage; force them onto element 0 so that a dropped
    # store mask only duplicates lane 0's value.
    k = tl.where(mask, k, 0)
    o = tl.where(mask, o, 0)
    outer = tl.where(mask, outer, 0)

    inv = tl.where(S == 0.0, 0.0, 1.0 / S)
    # ``- 1`` keeps a one-position margin on the low side: the analytic inverse
    # is exact in real arithmetic only, so an fp32 round-up must not clip the
    # lowest contributing source position.
    lo = tl.floor((o.to(tl.float32) - 1.0 - B) * inv).to(tl.int32) - 1
    lo = tl.maximum(lo, 0)

    base_in = outer * OUTER_STR_IN + k
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(K):
        i = lo + dd
        ok = i < IN_SIZE
        ic = tl.minimum(i, IN_SIZE - 1)
        if ALIGN:
            s = ic.to(tl.float32) * S
        else:
            s = (ic.to(tl.float32) + 0.5) * S - 0.5
        s = tl.maximum(s, 0.0)
        # ``OUT_SIZE`` is a runtime scalar in general, but Triton turns the
        # integer 1 into a constexpr (``equal_to_1`` specialization), so keep
        # the bound in floating point instead of calling ``.to`` on it.
        s = tl.minimum(s, (OUT_SIZE - 1) * 1.0)
        i0 = s.to(tl.int32)
        i1 = tl.minimum(i0 + 1, OUT_SIZE - 1)
        lam = s - i0.to(tl.float32)
        w = tl.where(o == i0, 1.0 - lam, 0.0) + tl.where(o == i1, lam, 0.0)
        w = tl.where(ok, w, 0.0)
        value = tl.load(src + base_in + ic * AXIS_STR_IN)
        acc += value.to(tl.float32) * w

    tl.store(dst + offsets, acc.to(dst.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Exact-2x fast path (benchmark matrix).
#
# When the grad_input axis is *exactly* half of the grad_output axis and the
# transform is the ATen non-align mapping, ``src(q) = q/2 - 1/4`` is an affine
# half-rate map, so along one axis the backward is the fixed 4-tap transpose
# filter
#
#     g[k] = 0.25*s[2k-1] + 0.75*s[2k] + 0.75*s[2k+1] + 0.25*s[2k+2]
#
# with every out-of-range tap clamped (verified against ATen CPU for interior
# and both edges, align_corners=False).  Two structural kernels exploit this:
#
#  * ``_w2x_i64_kernel`` (fp32 only) reads the W axis as packed ``int64`` words
#    (two neighbouring fp32 per lane).  For an output column ``k`` it needs the
#    words ``k-1, k, k+1`` -- three *unit-stride contiguous* loads -- instead
#    of the per-lane scalar gather of the generic pass; the even/odd halves are
#    rebuilt with bitcasts.  This is the W (inner==1) axis, the worst gather.
#  * ``_ax4_kernel`` evaluates the same 4-tap filter for a generic axis whose
#    inner block is contiguous (H and D), keeping the per-tap loads as long
#    contiguous runs.
# ---------------------------------------------------------------------------

_FAST_BLOCK = 4096
_FAST_WARPS = 4


@libentry()
@triton.jit
def _w2x_i64_kernel(
    src64,
    dst,
    total,
    IW: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
):
    """W axis for exact 2x (fp32): 3 contiguous packed-word loads per lane.

    ``CLAMP`` is ``False`` only when the launch covers exactly ``total`` (no
    tail): an address clamp makes the load offset data-dependent and blocks the
    block-DMA lowering on this backend (~1.5x slower), so the common
    tile-divisible case keeps fully affine offsets.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        off = tl.minimum(offsets, total - 1)
    else:
        off = offsets
    k = off % IW
    outer = off // IW
    base = outer * IW

    w0 = tl.load(src64 + base + tl.maximum(k - 1, 0))
    w1 = tl.load(src64 + base + k)
    w2 = tl.load(src64 + base + tl.minimum(k + 1, IW - 1))

    lo = (w1 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi = ((w1 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    o_km1 = ((w0 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    e_kp1 = (w2 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    # Boundary clamp: the left tap of k==0 and the right tap of k==IW-1 wrap
    # onto the same (already clamped) source element, not the neighbouring pair.
    o_km1 = tl.where(k == 0, lo, o_km1)
    e_kp1 = tl.where(k == IW - 1, hi, e_kp1)

    g = 0.25 * o_km1 + 0.75 * lo + 0.75 * hi + 0.25 * e_kp1
    # The destination is over-allocated to the tile size and every load address
    # stays in bounds, so the store is deliberately unmasked (a masked store
    # blocks the block-DMA lowering); tail lanes rewrite the padding.
    tl.store(dst + offsets, g)


@libentry()
@triton.jit
def _ax4_i64_kernel(
    src64,
    dst64,
    total,
    ONAXIS: tl.constexpr,
    INNER2: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
):
    """align_corners=False 2x 4-tap axis reduce, two inner columns per lane.

    Same layout/packing as ``_ax5_i64_kernel`` but with the constant
    ``0.25/0.75/0.75/0.25`` weights, so no weight table is loaded.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        off = tl.minimum(offsets, total - 1)
    else:
        off = offsets
    p = off % INNER2
    t = off // INNER2
    o = t % ONAXIS
    outer = t // ONAXIS
    base = outer * (2 * ONAXIS * SW) + p
    o2 = 2 * ONAXIS

    v0 = tl.load(src64 + base + tl.maximum(2 * o - 1, 0) * SW)
    v1 = tl.load(src64 + base + (2 * o) * SW)
    v2 = tl.load(src64 + base + tl.minimum(2 * o + 1, o2 - 1) * SW)
    v3 = tl.load(src64 + base + tl.minimum(2 * o + 2, o2 - 1) * SW)

    lo = 0.25 * (v0 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi = 0.25 * ((v0 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    lo += 0.75 * (v1 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += 0.75 * ((v1 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    lo += 0.75 * (v2 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += 0.75 * ((v2 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    lo += 0.25 * (v3 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += 0.25 * ((v3 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)

    out = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    out |= hi.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    tl.store(dst64 + offsets, out.to(tl.int64, bitcast=True))


@libentry()
@triton.jit
def _ax4_kernel(
    src,
    dst,
    total,
    ONAXIS: tl.constexpr,
    INNER: tl.constexpr,
    SAXIS_IN: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
):
    """Exact-2x 4-tap reduce along one axis of a contiguous (outer, 2N, inner)."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        off = tl.minimum(offsets, total - 1)
    else:
        off = offsets
    inner = off % INNER
    t = off // INNER
    o = t % ONAXIS
    outer = t // ONAXIS
    base = outer * (2 * ONAXIS * SAXIS_IN) + inner

    o2 = 2 * ONAXIS
    i0 = tl.maximum(2 * o - 1, 0)
    i1 = 2 * o
    i2 = tl.minimum(2 * o + 1, o2 - 1)
    i3 = tl.minimum(2 * o + 2, o2 - 1)
    v = 0.25 * tl.load(src + base + i0 * SAXIS_IN)
    v += 0.75 * tl.load(src + base + i1 * SAXIS_IN)
    v += 0.75 * tl.load(src + base + i2 * SAXIS_IN)
    v += 0.25 * tl.load(src + base + i3 * SAXIS_IN)
    # Unmasked store: dst is over-allocated to the tile size and all loads are
    # clamped in bounds, so tail lanes land inside the padding.
    tl.store(dst + offsets, v.to(dst.dtype.element_ty))



# ---------------------------------------------------------------------------
# align_corners=True exact-2x fast path.
#
# With ``align_corners=True`` the source scale is ``S = (in-1)/(out-1)`` which
# for an exact 2x axis (out == 2*in) equals ``0.5 - 1/(2*(2N-1))``, i.e. *not*
# 0.5, so the 4-tap constant filter above does not apply: the per-lane weights
# vary with the lane.  The weights are still cheap to tabulate:
#
#   grad_in[o] = sum_q grad_out[q] * tri(o - q*S),  tri(x) = max(0, 1-|x|)
#
# Lane ``o`` only receives from ``q`` with ``floor(q*S) in {o-1, o}``, i.e.
# ``q in ((o-1)/S, (o+1)/S]`` -- a window of width ``2/S <= 5`` for out >= 4.
# Starting the window at ``2o-1`` (5 taps ``q = 2o-1 .. 2o+3``) covers every
# non-zero tap for all out_axis >= 4 (verified exhaustively for out in 4..1024
# against the full weight matrix; out_axis < 4 keeps the generic gather pass).
#
# A host-cached 5-tap weight table (one tiny kernel launch per unique
# (in_axis, out_axis)) turns the main kernel into a pure table-driven 5-tap
# merge: no per-lane scale, floor or weight arithmetic.  Combined with the same
# int64 packing as the 2x path (2 fp32 per word) this brings the worst axis (W,
# inner == 1) from ~12.5 ms to ~1.4 ms and the H/D axes to ~0.5/0.4 ms on the
# benchmark's largest shape.  The tail-block CLAMP / over-allocated unmasked
# store strategy and the flat 1-D layout are reused unchanged.
# ---------------------------------------------------------------------------

_TAB5 = 5
_TAB_CACHE = {}


@libentry()
@triton.jit
def _wt5_kernel(tab, S, IN, OUT, total, K: tl.constexpr, BLOCK: tl.constexpr):
    """Build the ``(K, OUT)`` weight table for an align_corners=True 2x axis."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < total
    j = idx // OUT
    o = idx % OUT
    q = 2 * o - 1 + j
    sf = q.to(tl.float32) * S
    sf = tl.maximum(sf, 0.0)
    sf = tl.minimum(sf, (OUT - 1) * 1.0)
    i0 = sf.to(tl.int32)
    i1 = tl.minimum(i0 + 1, OUT - 1)
    lam = sf - i0.to(tl.float32)
    w = tl.where(o == i0, 1.0 - lam, 0.0) + tl.where(o == i1, lam, 0.0)
    w = tl.where((q >= 0) & (q < IN), w, 0.0)
    tl.store(tab + idx, tl.where(m, w, 0.0), mask=m)


def _weight_table5(in_axis: int, out_axis: int, dev) -> torch.Tensor:
    """Cached ``(5, out_axis)`` fp32 weight table (constant launch-time data)."""
    key = (str(dev), in_axis, out_axis)
    tab = _TAB_CACHE.get(key)
    if tab is not None:
        return tab
    total = _TAB5 * out_axis
    buf = torch.empty(
        triton.cdiv(total, _BLOCK) * _BLOCK, device=dev, dtype=torch.float32
    )
    with torch_device_fn.device(dev):
        _wt5_kernel[(triton.cdiv(total, _BLOCK),)](
            buf,
            (float(out_axis) - 1.0) / (float(in_axis) - 1.0),
            in_axis,
            out_axis,
            total,
            K=_TAB5,
            BLOCK=_BLOCK,
            num_warps=1,
        )
    tab = buf[:total].view(_TAB5, out_axis)
    _TAB_CACHE[key] = tab
    return tab


@libentry()
@triton.jit
def _w5_i64_kernel(
    src64,
    dst,
    tab,
    total,
    IW: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
):
    """W axis, align_corners=True 2x: 3 packed-word loads + 5 table taps."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        off = tl.minimum(offsets, total - 1)
    else:
        off = offsets
    k = off % IW
    outer = off // IW
    base = outer * IW

    w0 = tl.load(src64 + base + tl.maximum(k - 1, 0))
    w1 = tl.load(src64 + base + k)
    w2 = tl.load(src64 + base + tl.minimum(k + 1, IW - 1))
    # taps q = 2k-1 .. 2k+3: w0.hi | w1.lo w1.hi | w2.lo w2.hi
    t0 = ((w0 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    t1 = (w1 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    t2 = ((w1 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    t3 = (w2 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    t4 = ((w2 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)

    g = t0 * tl.load(tab + k)
    g += t1 * tl.load(tab + IW + k)
    g += t2 * tl.load(tab + 2 * IW + k)
    g += t3 * tl.load(tab + 3 * IW + k)
    g += t4 * tl.load(tab + 4 * IW + k)
    # Unmasked store: dst is over-allocated to the tile size and every load is
    # clamped in bounds, so tail lanes only rewrite padding.
    tl.store(dst + offsets, g)


@libentry()
@triton.jit
def _ax5_i64_kernel(
    src64,
    dst64,
    tab,
    total,
    ONAXIS: tl.constexpr,
    INNER2: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
):
    """align_corners=True 2x axis reduce, fp32 with two inner columns per lane.

    ``src64``/``dst64`` are the fp32 buffers viewed as int64 (two neighbouring
    inner columns per word), so each lane produces two outputs and halves the
    per-lane tap-load count.  ``SW = inner // 2`` is the row stride in words.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        off = tl.minimum(offsets, total - 1)
    else:
        off = offsets
    p = off % INNER2
    t = off // INNER2
    o = t % ONAXIS
    outer = t // ONAXIS
    base = outer * (2 * ONAXIS * SW) + p
    o2 = 2 * ONAXIS

    v0 = tl.load(src64 + base + tl.maximum(2 * o - 1, 0) * SW)
    v1 = tl.load(src64 + base + (2 * o) * SW)
    v2 = tl.load(src64 + base + (2 * o + 1) * SW)
    v3 = tl.load(src64 + base + tl.minimum(2 * o + 2, o2 - 1) * SW)
    v4 = tl.load(src64 + base + tl.minimum(2 * o + 3, o2 - 1) * SW)

    lo = tl.zeros((BLOCK,), dtype=tl.float32)
    hi = tl.zeros((BLOCK,), dtype=tl.float32)
    w = tl.load(tab + o)
    lo += w * (v0 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += w * ((v0 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    w = tl.load(tab + ONAXIS + o)
    lo += w * (v1 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += w * ((v1 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    w = tl.load(tab + 2 * ONAXIS + o)
    lo += w * (v2 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += w * ((v2 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    w = tl.load(tab + 3 * ONAXIS + o)
    lo += w * (v3 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += w * ((v3 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    w = tl.load(tab + 4 * ONAXIS + o)
    lo += w * (v4 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    hi += w * ((v4 >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)

    out = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    out |= hi.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    tl.store(dst64 + offsets, out.to(tl.int64, bitcast=True))


@libentry()
@triton.jit
def _ax5_kernel(
    src,
    dst,
    tab,
    total,
    ONAXIS: tl.constexpr,
    INNER: tl.constexpr,
    SAXIS_IN: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
):
    """align_corners=True 2x 5-tap reduce along one axis (unpacked fp32)."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CLAMP:
        off = tl.minimum(offsets, total - 1)
    else:
        off = offsets
    inner = off % INNER
    t = off // INNER
    o = t % ONAXIS
    outer = t // ONAXIS
    base = outer * (2 * ONAXIS * SAXIS_IN) + inner
    o2 = 2 * ONAXIS

    a = tl.load(tab + o) * tl.load(
        src + base + tl.maximum(2 * o - 1, 0) * SAXIS_IN
    )
    a += tl.load(tab + ONAXIS + o) * tl.load(src + base + (2 * o) * SAXIS_IN)
    a += tl.load(tab + 2 * ONAXIS + o) * tl.load(src + base + (2 * o + 1) * SAXIS_IN)
    a += tl.load(tab + 3 * ONAXIS + o) * tl.load(
        src + base + tl.minimum(2 * o + 2, o2 - 1) * SAXIS_IN
    )
    a += tl.load(tab + 4 * ONAXIS + o) * tl.load(
        src + base + tl.minimum(2 * o + 3, o2 - 1) * SAXIS_IN
    )
    tl.store(dst + offsets, a.to(dst.dtype.element_ty))


def _scale_and_bias(in_size: int, out_size: int, align_corners: bool, scale):
    """Match ATen ``compute_scales_value<float>`` / ``compute_source_index``."""
    if align_corners:
        s = (float(in_size) - 1.0) / (float(out_size) - 1.0) if out_size > 1 else 0.0
        b = 0.0
    else:
        if scale is not None and scale > 0:
            s = 1.0 / float(scale)
        else:
            s = float(in_size) / float(out_size)
        b = 0.5 * s - 0.5
    return s, b


def _candidate_count(scale_units: float) -> int:
    """Contributing window per lane.

    The triangular basis is centred on the source coordinate with one unit of
    support on each side, so the contributing interval has length ``2/scale``
    in source-position units (at most ``floor(2/scale) + 1`` integers), plus one
    slot for the fp32 boundary slip and one for the ``- 1`` low margin.
    """
    if scale_units <= 0.0:
        return 1
    return int(2.0 / scale_units) + 3


def _axis_pass(
    src: torch.Tensor,
    outer: int,
    in_size: int,
    out_size: int,
    inner: int,
    out_dtype: torch.dtype,
    scale,
    align_corners: bool,
) -> torch.Tensor:
    """Reduce one axis of ``src`` (logical ``(outer, in_size, inner)``).

    ``in_size`` is the grad_output axis (source of the gather) and ``out_size``
    the grad_input axis (lane/destination axis, i.e. ATen's "input_size").
    """
    # ATen's source scale maps grad_output -> grad_input, so it is parameterised
    # as (input_size, output_size) = (grad_input, grad_output) here.
    s, b = _scale_and_bias(out_size, in_size, align_corners, scale)
    # ``s == 0`` (align_corners=True with a grad_input axis of size 1) maps
    # *every* grad_output position onto that single position with weight 1, so
    # the required window is the whole axis rather than a local band.
    taps = in_size if s == 0.0 else _candidate_count(s)
    total = outer * out_size * inner
    buf = torch.empty(
        triton.cdiv(total, _BLOCK) * _BLOCK,
        device=src.device,
        dtype=out_dtype,
    )
    if total == 0:
        return buf[:0].view(outer, out_size, inner)
    dst = buf[:total]
    grid = (triton.cdiv(total, _BLOCK),)
    with torch_device_fn.device(src.device):
        _axis_reduce_kernel[grid](
            src,
            dst,
            total,
            in_size,
            out_size,
            in_size * inner,
            inner,
            inner,
            s,
            b,
            K=taps,
            ALIGN=bool(align_corners),
            BLOCK=_BLOCK,
            num_warps=1,
            buffer_size_limit=2048,
        )
    return buf[:total].view(outer, out_size, inner)


def _fast_ax4(
    src: torch.Tensor,
    outer: int,
    out_axis: int,
    inner: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Exact-2x 4-tap reduce along one axis (src axis is ``2*out_axis``)."""
    total = outer * out_axis * inner
    buf = torch.empty(
        triton.cdiv(total, _FAST_BLOCK) * _FAST_BLOCK,
        device=src.device,
        dtype=out_dtype,
    )
    if total == 0:
        return buf[:0].view(outer, out_axis, inner)
    clamp = (total % _FAST_BLOCK) != 0
    grid = (total // _FAST_BLOCK if not clamp else triton.cdiv(total, _FAST_BLOCK),)
    with torch_device_fn.device(src.device):
        _ax4_kernel[grid](
            src,
            buf,
            total,
            ONAXIS=out_axis,
            INNER=inner,
            SAXIS_IN=inner,
            BLOCK=_FAST_BLOCK,
            CLAMP=clamp,
            num_warps=_FAST_WARPS,
        )
    return buf[:total].view(outer, out_axis, inner)


def _fast_ax4_i64(
    src: torch.Tensor,
    outer: int,
    out_axis: int,
    inner: int,
) -> torch.Tensor:
    """Exact-2x 4-tap reduce, fp32 via packed int64 words (2 columns per lane).

    Halves the number of tap loads relative to ``_fast_ax4`` (one word load
    feeds two neighbouring inner columns), which on this backend is a direct
    win because the cost model bills DMA operations, not bytes.
    """
    dev = src.device
    words = outer * out_axis * (inner // 2)
    pad = triton.cdiv(words, _FAST_BLOCK) * _FAST_BLOCK
    buf = torch.empty(2 * pad, device=dev, dtype=torch.float32)
    clamp = (words % _FAST_BLOCK) != 0
    with torch_device_fn.device(dev):
        _ax4_i64_kernel[(pad // _FAST_BLOCK,)](
            src.reshape(-1).view(torch.int64),
            buf.view(torch.int64),
            words,
            ONAXIS=out_axis,
            INNER2=inner // 2,
            SW=inner // 2,
            BLOCK=_FAST_BLOCK,
            CLAMP=clamp,
            num_warps=_FAST_WARPS,
        )
    return buf[: outer * out_axis * inner].view(outer, out_axis, inner)


def _align_ax5(
    src: torch.Tensor,
    outer: int,
    out_axis: int,
    inner: int,
    out_dtype: torch.dtype,
    tab: torch.Tensor,
) -> torch.Tensor:
    """One align_corners=True 2x axis pass (src axis is ``2*out_axis``)."""
    total = outer * out_axis * inner
    if total == 0:
        return torch.empty(
            (outer, out_axis, inner), device=src.device, dtype=out_dtype
        )
    dev = src.device
    if out_dtype == torch.float32 and inner % 2 == 0:
        # Two inner columns per lane through packed int64 words.
        words = outer * out_axis * (inner // 2)
        pad = triton.cdiv(words, _FAST_BLOCK) * _FAST_BLOCK
        buf = torch.empty(2 * pad, device=dev, dtype=out_dtype)
        clamp = (words % _FAST_BLOCK) != 0
        with torch_device_fn.device(dev):
            _ax5_i64_kernel[(pad // _FAST_BLOCK,)](
                src.reshape(-1).view(torch.int64),
                buf.view(torch.int64),
                tab,
                words,
                ONAXIS=out_axis,
                INNER2=inner // 2,
                SW=inner // 2,
                BLOCK=_FAST_BLOCK,
                CLAMP=clamp,
                num_warps=_FAST_WARPS,
            )
        return buf[:total].view(outer, out_axis, inner)
    pad = triton.cdiv(total, _FAST_BLOCK) * _FAST_BLOCK
    buf = torch.empty(pad, device=dev, dtype=out_dtype)
    clamp = (total % _FAST_BLOCK) != 0
    with torch_device_fn.device(dev):
        _ax5_kernel[(pad // _FAST_BLOCK,)](
            src,
            buf,
            tab,
            total,
            ONAXIS=out_axis,
            INNER=inner,
            SAXIS_IN=inner,
            BLOCK=_FAST_BLOCK,
            CLAMP=clamp,
            num_warps=_FAST_WARPS,
        )
    return buf[:total].view(outer, out_axis, inner)


def _fast_align2x(src, nc, ID, IH, IW, OD, OH, OW, out_dtype):
    """align_corners=True exact-2x pipeline: W (packed) -> H -> D."""
    dev = src.device
    rows = nc * OD * OH
    flat = src.reshape(rows, OW)
    tabw = _weight_table5(OW, IW, dev)
    total = rows * IW
    pad = triton.cdiv(total, _FAST_BLOCK) * _FAST_BLOCK
    buf = torch.empty(pad, device=dev, dtype=torch.float32)
    clamp = (total % _FAST_BLOCK) != 0
    with torch_device_fn.device(dev):
        if out_dtype == torch.float32:
            # int64 packing needs a 4-byte element (2 fp32 per word).
            _w5_i64_kernel[(pad // _FAST_BLOCK,)](
                flat.view(torch.int64),
                buf,
                tabw,
                total,
                IW=IW,
                BLOCK=_FAST_BLOCK,
                CLAMP=clamp,
                num_warps=_FAST_WARPS,
            )
        else:
            _ax5_kernel[(pad // _FAST_BLOCK,)](
                flat,
                buf,
                tabw,
                total,
                ONAXIS=IW,
                INNER=1,
                SAXIS_IN=1,
                BLOCK=_FAST_BLOCK,
                CLAMP=clamp,
                num_warps=_FAST_WARPS,
            )
    t1 = buf[:total].view(rows, IW)
    # H: (nc*OD, OH, IW) -> (nc*OD, IH, IW)
    t2 = _align_ax5(t1, nc * OD, IH, IW, torch.float32, _weight_table5(OH, IH, dev))
    # D: (nc, OD, IH*IW) -> (nc, ID, IH*IW)
    out = _align_ax5(
        t2, nc, ID, IH * IW, out_dtype, _weight_table5(OD, ID, dev)
    )
    return out.view(nc, ID, IH, IW)


def upsample_trilinear3d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_TRILINEAR3D_BACKWARD")
    assert grad_output.device.type == device.name
    assert grad_output.ndim == 5, "The ndim of grad_output must be 5"

    OD, OH, OW = (int(v) for v in output_size)
    N, C, ID, IH, IW = (int(v) for v in input_size)
    assert grad_output.shape == (N, C, OD, OH, OW), (
        f"grad_output shape {grad_output.shape} does not match "
        f"expected (N={N}, C={C}, OD={OD}, OH={OH}, OW={OW})"
    )

    nc = N * C
    grad_output = grad_output.contiguous()
    if nc * ID * IH * IW == 0 or nc * OD * OH * OW == 0:
        # No elements to reduce; an empty allocation is a valid result.
        return torch.empty(
            (N, C, ID, IH, IW), device=grad_output.device, dtype=grad_output.dtype
        )

    src = grad_output.view(nc, OD, OH, OW)

    def _is_2x(scale, grad_in, grad_out):
        return grad_out == 2 * grad_in and (scale is None or scale == 2.0)

    exact2x = (
        not align_corners
        and _is_2x(scales_d, ID, OD)
        and _is_2x(scales_h, IH, OH)
        and _is_2x(scales_w, IW, OW)
    )
    if exact2x:
        rows = nc * OD * OH
        if grad_output.dtype == torch.float32:
            # int64 packing needs a 4-byte element (2 fp32 per word).
            src64 = src.reshape(rows, OW).view(torch.int64)
            total = rows * IW
            buf = torch.empty(
                triton.cdiv(total, _FAST_BLOCK) * _FAST_BLOCK,
                device=src.device,
                dtype=torch.float32,
            )
            clamp = (total % _FAST_BLOCK) != 0
            grid = (
                (total // _FAST_BLOCK,)
                if not clamp
                else (triton.cdiv(total, _FAST_BLOCK),)
            )
            with torch_device_fn.device(src.device):
                _w2x_i64_kernel[grid](
                    src64,
                    buf,
                    total,
                    IW=IW,
                    BLOCK=_FAST_BLOCK,
                    CLAMP=clamp,
                    num_warps=_FAST_WARPS,
                )
            t1 = buf[:total].view(rows, IW)
        else:
            t1 = _fast_ax4(src.reshape(rows, OW), rows, IW, 1, torch.float32)
        fp32 = grad_output.dtype == torch.float32
        # H: (nc*OD, OH, IW) -> (nc*OD, IH, IW)
        if fp32 and IW % 2 == 0:
            t2 = _fast_ax4_i64(t1, nc * OD, IH, IW)
        else:
            t2 = _fast_ax4(t1, nc * OD, IH, IW, torch.float32)
        # D: (nc, OD, IH*IW) -> (nc, ID, IH*IW)
        if fp32 and (IH * IW) % 2 == 0:
            out = _fast_ax4_i64(t2, nc, ID, IH * IW)
        else:
            out = _fast_ax4(t2, nc, ID, IH * IW, grad_output.dtype)
        return out.view(N, C, ID, IH, IW)

    # align_corners=True still has a per-lane varying weight, but for an exact
    # 2x axis it only spans a 5-tap window.  A cached per-lane weight table plus
    # the same int64 packing makes the three gather passes table-driven.  The
    # 5-tap window is only proven for out_axis >= 4 (see the kernel notes);
    # smaller axes keep the generic gather pass.
    align2x = (
        align_corners
        and ID >= 4
        and IH >= 4
        and IW >= 4
        and _is_2x(scales_d, ID, OD)
        and _is_2x(scales_h, IH, OH)
        and _is_2x(scales_w, IW, OW)
    )
    if align2x:
        out = _fast_align2x(
            src, nc, ID, IH, IW, OD, OH, OW, grad_output.dtype
        )
        return out.view(N, C, ID, IH, IW)

    # W: (nc*OD*OH, OW, 1) -> (nc*OD*OH, IW, 1)
    t1 = _axis_pass(
        src, nc * OD * OH, OW, IW, 1, torch.float32, scales_w, align_corners
    )
    # H: (nc*OD, OH, IW) -> (nc*OD, IH, IW)
    t2 = _axis_pass(
        t1, nc * OD, OH, IH, IW, torch.float32, scales_h, align_corners
    )
    # D: (nc, OD, IH*IW) -> (nc, ID, IH*IW)
    out = _axis_pass(
        t2, nc, OD, ID, IH * IW, grad_output.dtype, scales_d, align_corners
    )
    return out.view(N, C, ID, IH, IW)
