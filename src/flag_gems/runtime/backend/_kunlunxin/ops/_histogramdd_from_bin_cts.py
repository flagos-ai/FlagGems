# Copyright 2026 The FlagOS Contributors
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

"""Kunlunxin (XPU3) override for ``aten::_histogramdd_from_bin_cts``.

The generic KernelGen implementation scatters one ``tl.atomic_add`` per point
into the histogram.  ``tl.atomic_add`` does not provide a correct
read-modify-write on this backend (see
``wiki/patterns/xpu3-atomic-add-misaccumulates.md``): every point ends up in
bin 0.  The generic kernel also fails to legalize for this backend's tile
shapes.

This override keeps the exact ATen binning semantics but replaces the atomic
scatter with the atomic-free two-pass scheme already validated for
``histogramdd`` (``wiki/techniques/xpu3-histogramdd-index-count-split.md``):

1. ``range``/``edges`` setup on the device (no host ``.tolist()`` round trip),
2. an *index* pass that derives every point's row-major linear bin index once
   (``N * D`` lanes, ``-1`` for out-of-range/padding),
3. a bin-major *count* pass that scans the int32 index array
   (``num_bins * N`` cheap compares, exactly one writer per output bin).

Bin placement still searches the materialised ``torch.linspace`` edges the way
ATen does (``histogramdd`` builds the edges with linspace and searches them),
because a closed-form ``floor`` disagrees with ATen for points that sit exactly
on an edge.  The edge materialisation below reproduces ``torch.linspace``
bit-for-bit, which is what the exact-boundary tests pin down.
"""

import builtins
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Points per program in the index pass, points per inner iteration in the
# bin-major count pass, and points/reduction width in the auto-range pass.
# Fixed constexpr tiles (instead of next_power_of_2(D)-sized tiles) keep every
# reduction inside the verified lane window and keep the 2-D range tile wide in
# its inner dimension (narrow 2-D tiles are silently miscompiled here).
# The index pass amortises its per-dimension uniform edge loads over the whole
# program tile, so it wants a wide tile (measured at N=1e5: 501us at 1024 lanes
# vs 105-149us at 4096-8192); the widest tile that still wins is dimension
# dependent (D<=2: 8192, D=3: 4096).
INDEX_TILE_WIDE = 8192
INDEX_TILE = 4096
COUNT_TILE = 4096
RANGE_TILE = 1024
# Per-dimension edge slabs and the per-dimension volume reduction use a
# >= 64-lane tile; anything narrower than one warp is miscompiled.
MIN_DIM_TILE = 64
# Deepest static unroll of the bin-major count chunk loop. The int32 variant
# stops compiling past 64 chunks ("Failed to tune buffer size": the unrolled
# body exceeds the local-memory budget), so larger point clouds take the
# dynamic loop instead.
COUNT_UNROLL_MAX = 64
# Widest single tile used to normalise the histogram for ``density=True``. A
# 8192-lane tile keeps every in-tree bin count (<= 8192 bins) in one pass, which
# avoids a masked-store-in-unrolled-loop hazard (see the host comment).
DENSITY_TILE = 8192


@libentry()
@triton.jit
def histogramdd_from_bin_cts_range_kernel(
    input_ptr,  # (D, N) contiguous input (transposed by the caller)
    lohi_ptr,  # (2 * D,) out: [lo0, hi0, lo1, hi1, ...]
    N,
    D: tl.constexpr,
    DPAD: tl.constexpr,  # next_pow2(D)
    ACC_DTYPE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Per-dimension min/max with ATen's degenerate-range rule, on the device.

    A 2-D ``(DPAD, TILE)`` tile whose rows are the transposed input's
    per-dimension rows (stride-1 block DMA) yields every dimension's min/max in
    one ``axis=1`` reduce.  A dimension whose min equals max is widened to
    ``(min - 0.5, max + 0.5)``; a dimension holding a NaN gets a NaN range (as
    ``aminmax`` does).

    The min/max reduce ignores NaN operands, so NaN is tracked in a separate
    float accumulator and folded back in as a NaN range.  The host derives the
    "range is not finite" error from these (lo, hi) values, so no separate
    non-finite reduction is needed.
    """
    j = tl.arange(0, DPAD)
    i = tl.arange(0, TILE)
    in_dim = j < D
    safe_j = tl.minimum(j, D - 1)
    row = input_ptr + safe_j[:, None] * N

    acc_min = tl.full((DPAD, TILE), float("inf"), dtype=ACC_DTYPE)
    acc_max = tl.full((DPAD, TILE), float("-inf"), dtype=ACC_DTYPE)
    # A 1.0 flag kept in a float accumulator: an axis=1 float max is the
    # reduction shape that legalizes reliably in this 2-D tile band.
    acc_nan = tl.zeros((DPAD, TILE), dtype=ACC_DTYPE)

    nchunks = tl.cdiv(N, TILE)
    for c in range(0, nchunks):
        cur = c * TILE + i
        # The tail is handled with an explicit address clamp plus a lane mask:
        # this backend ignores the ``other`` fill of a masked load.
        safe_i = tl.minimum(cur, N - 1)
        vals = tl.load(row + safe_i[None, :]).to(ACC_DTYPE)
        keep = (cur < N)[None, :] & in_dim[:, None]
        acc_nan = tl.maximum(acc_nan, tl.where(keep & (vals != vals), 1.0, 0.0))
        acc_min = tl.minimum(acc_min, tl.where(keep, vals, float("inf")))
        acc_max = tl.maximum(acc_max, tl.where(keep, vals, float("-inf")))

    mn = tl.min(acc_min, axis=1)
    mx = tl.max(acc_max, axis=1)
    has_nan = tl.max(acc_nan, axis=1) > 0
    # Covariant with ``aminmax``: a NaN column gets a NaN range.
    mn = tl.where(has_nan, float("nan"), mn)
    mx = tl.where(has_nan, float("nan"), mx)
    same = mn == mx
    lo = tl.where(same, mn - 0.5, mn)
    hi = tl.where(same, mx + 0.5, mx)
    # The store address is clamped to the last real dimension: a masked store is
    # not reliably predicated on this backend, and a padded row would otherwise
    # write past the (2 * D,) buffer. A clamped padded row duplicates the last
    # real dimension's value, so an unpredicated write is harmless.
    tl.store(lohi_ptr + 2 * safe_j, lo, mask=in_dim)
    tl.store(lohi_ptr + 2 * safe_j + 1, hi, mask=in_dim)


@libentry()
@triton.jit
def histogramdd_from_bin_cts_edges_kernel(
    lohi_ptr,  # (2 * D,) [lo0, hi0, lo1, hi1, ...]
    edges_key_ptr,  # (D * EDGE_TILE,) out: per-dimension edge keys (int32)
    lohi_key_ptr,  # (2 * D,) out: per-dimension [lo, hi] keys (int32)
    D: tl.constexpr,
    BINS: tl.constexpr,  # per-dimension bin counts
    EDGE_TILE: tl.constexpr,  # >= 64, power of two, >= max(BINS) + 1
    ACC_DTYPE: tl.constexpr,
):
    """Materialise the per-dimension bin edges the index pass searches.

    Reproduces ``torch.linspace`` bit-for-bit, which is how ATen builds these:
    interpolate from the nearer endpoint (accumulating from the left throughout
    puts edge 6 of [0, 1]/10 bins at 0.6000000000000001, misplacing a point at
    exactly 0.6), round ``step`` to the storage dtype, run the multiply-add in
    fp64 and narrow only on the store.

    Each edge is stored as a monotone int32 key of its float bits, and so are
    ``lo``/``hi``.  The index pass compares those keys instead of the floats
    because this backend flushes denormals to zero, which makes a denormal
    coordinate compare equal to a 0.0 edge and misplace it.  The key is
    ``bits >= 0 ? bits ^ INT_MIN : ~bits`` (a monotone map from IEEE floats to
    signed int32) with ``-0.0`` canonicalised to ``+0.0``, so key order is IEEE
    order including equality.
    """
    e = tl.arange(0, EDGE_TILE)
    min_int = tl.zeros((EDGE_TILE,), dtype=tl.int32) - 2147483647 - 1
    zero = tl.zeros((EDGE_TILE,), dtype=tl.int32)
    for d in tl.static_range(D):
        nb = BINS[d]
        left = tl.load(lohi_ptr + 2 * d)
        right = tl.load(lohi_ptr + 2 * d + 1)
        steps = nb + 1
        half = steps // 2
        l_out = left.to(ACC_DTYPE)
        r_out = right.to(ACC_DTYPE)
        if ACC_DTYPE == tl.float32:
            # div_rn rather than `/`: Triton lowers plain fp32 division to a
            # fast reciprocal that is not correctly rounded, and it shifts
            # `step` by an ulp on many configurations.
            step = tl.math.div_rn(r_out - l_out, nb * 1.0).to(tl.float64)
        else:
            step = ((r_out - l_out) / nb).to(tl.float64)
        l64 = l_out.to(tl.float64)
        r64 = r_out.to(tl.float64)
        fi = e.to(tl.float64)
        rev = (steps - 1 - e).to(tl.float64)
        val = tl.where(e < half, step * fi + l64, r64 - step * rev).to(tl.float32)
        bits = val.to(tl.int32, bitcast=True)
        bits = tl.where(bits == min_int, zero, bits)
        key = tl.where(bits >= 0, bits, ~bits ^ min_int)
        # Only edges 0..nb are ever read, so the slab needs no padding value.
        tl.store(edges_key_ptr + d * EDGE_TILE + e, key, mask=e <= nb)

        lbits = left.to(tl.float32).to(tl.int32, bitcast=True)
        lbits = tl.where(lbits == min_int, zero, lbits)
        lkey = tl.where(lbits >= 0, lbits, ~lbits ^ min_int)
        rbits = right.to(tl.float32).to(tl.int32, bitcast=True)
        rbits = tl.where(rbits == min_int, zero, rbits)
        rkey = tl.where(rbits >= 0, rbits, ~rbits ^ min_int)
        tl.store(lohi_key_ptr + 2 * d, tl.max(lkey, axis=0))
        tl.store(lohi_key_ptr + 2 * d + 1, tl.max(rkey, axis=0))


@libentry()
@triton.jit
def histogramdd_from_bin_cts_index_kernel(
    input_ptr,  # (D, N) contiguous input (transposed by the caller)
    edges_key_ptr,  # (D * EDGE_TILE,) per-dimension edge keys
    lohi_key_ptr,  # (2 * D,) per-dimension [lo, hi] keys
    idx_ptr,  # (nchunks * TILE,) int32 row-major linear bin (or -1) per point
    N,
    D: tl.constexpr,
    BINS: tl.constexpr,
    STRIDES: tl.constexpr,
    EDGE_TILE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Compute every point's row-major linear bin index exactly once.

    Bin placement is ``count(edges <= coord) - 1`` clamped to ``[0, nb - 1]``:
    ATen materialises the ``torch.linspace`` edges and searches them, and this
    reproduces that exactly (a closed-form ``floor`` misplaces points that sit
    on an edge).  One dimension only has ``nb + 1`` edges, so they are read as
    *uniform* scalar loads in a statically unrolled loop -- one broadcast load
    per edge for the whole program, not a per-lane gather.  A per-lane binary
    search over the same edge slab does not lower on this backend (a vector
    ``addptr`` offset is rejected by ``TritonXPUUnrollControl``, and both
    ``tl.range`` and dynamic-bound loops are rejected as well), so the unrolled
    uniform-edge form is the one that compiles, and it is exact.

    Comparisons use the int32 keys produced by the edges kernel, not float
    compares, because this backend flushes denormals to zero (a denormal
    coordinate would otherwise compare equal to a 0.0 edge). Points outside
    ``[lo, hi]`` and padding lanes are tagged ``-1``; a NaN key is larger than
    any finite key, so NaN is excluded by the ``<= hi`` test.
    """
    pid = ext.program_id(0)
    offs = pid * TILE + tl.arange(0, TILE)
    safe = tl.minimum(offs, N - 1)
    # Integer lane arithmetic stays int32 with explicitly typed constants:
    # mixing a Python int into a tensor op emits a mixed-width op that this
    # backend's legalize rejects.
    zero = tl.zeros((TILE,), dtype=tl.int32)
    one = tl.full((TILE,), 1, dtype=tl.int32)
    min_int = tl.zeros((TILE,), dtype=tl.int32) - 2147483647 - 1

    linear = tl.zeros((TILE,), dtype=tl.int32)
    in_range = tl.full((TILE,), 1, dtype=tl.int32)
    for d in tl.static_range(D):
        val = tl.load(input_ptr + d * N + safe).to(tl.float32)
        nb_vec = tl.full((TILE,), BINS[d], dtype=tl.int32)
        stride_vec = tl.full((TILE,), STRIDES[d], dtype=tl.int32)
        bits = val.to(tl.int32, bitcast=True)
        # ``-0.0`` is canonicalised to ``+0.0`` so key equality matches IEEE
        # equality (a point at -0.0 belongs to the bin a point at +0.0 does).
        bits = tl.where(bits == min_int, zero, bits)
        vkey = tl.where(bits >= 0, bits, ~bits ^ min_int)
        lkey = tl.load(lohi_key_ptr + 2 * d)
        rkey = tl.load(lohi_key_ptr + 2 * d + 1)
        in_range = in_range & ((vkey >= lkey) & (vkey <= rkey)).to(tl.int32)

        row = edges_key_ptr + d * EDGE_TILE
        cnt = tl.zeros((TILE,), dtype=tl.int32)
        for j in tl.static_range(BINS[d] + 1):
            cnt += (vkey >= tl.load(row + j)).to(tl.int32)
        # ``count(edges <= coord) - 1``: the rightmost bin is inclusive of its
        # right edge, so a point exactly on ``right`` counts nb + 1 edges and
        # folds back into the last bin.
        b = tl.minimum(cnt - one, nb_vec - one)
        b = tl.maximum(b, zero)
        linear = linear + b * stride_vec

    # Padding lanes (offs >= N) are tagged -1 as well, so the counting pass can
    # scan the whole padded index buffer with neither mask nor clamp.
    linear = tl.where((offs < N) & (in_range != 0), linear, -1)
    tl.store(idx_ptr + offs, linear)


@libentry()
@triton.jit
def histogramdd_from_bin_cts_count_kernel(
    idx_ptr,  # (NCH * TILE,) int32 linear bin per point, -1 = drop
    weight_ptr,  # (N,) per-point weights, or unused when HAS_WEIGHT=False
    hist_ptr,  # (num_bins,) histogram output (one writer per bin)
    N,
    NCH: tl.constexpr,  # point-tile count
    UNROLL: tl.constexpr,  # statically unroll the chunk loop?
    HAS_WEIGHT: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    TILE: tl.constexpr,
):
    """One program per output bin: count the points whose index is ``b``.

    Bin-major ownership means no cross-program atomic is issued and every output
    element has exactly one writer, so the histogram buffer only needs an
    allocation, not a zero fill.  The scan is over int32 indices instead of D
    float coordinates, so it pays no per-dimension scalar loads and no bin
    arithmetic.

    The chunk loop is statically unrolled when ``UNROLL`` is set: with a dynamic
    bound the loop runs at ~1 lane/cycle (a measured 2.7x penalty at
    ``num_bins * N`` lanes), while the unrolled form lets the backend vectorise
    the load/compare/accumulate chain.  A very deep unroll exhausts the local
    memory budget (the int32 variant stops compiling past 64 chunks), so the
    host asks for the dynamic loop beyond that.
    """
    b = ext.program_id(0)
    offs = tl.arange(0, TILE)
    if HAS_WEIGHT:
        acc = tl.zeros((TILE,), dtype=ACC_DTYPE)
        if UNROLL:
            for c in tl.static_range(NCH):
                pos = c * TILE + offs
                cur = tl.load(idx_ptr + pos)
                # A mask keeps the weight address affine; an index clamp would
                # make it a per-lane gather (measured 4x slower for the same
                # lanes). Padding lanes never match any bin (their index is
                # -1), so their weight value is irrelevant.
                w = tl.load(weight_ptr + pos, mask=pos < N, other=0.0).to(ACC_DTYPE)
                acc += tl.where(cur == b, w, 0.0)
        else:
            for c in range(0, NCH):
                pos = c * TILE + offs
                cur = tl.load(idx_ptr + pos)
                w = tl.load(weight_ptr + pos, mask=pos < N, other=0.0).to(ACC_DTYPE)
                acc += tl.where(cur == b, w, 0.0)
        tl.store(hist_ptr + b, tl.sum(acc, axis=0))
    else:
        acc_i = tl.zeros((TILE,), dtype=tl.int32)
        if UNROLL:
            for c in tl.static_range(NCH):
                acc_i += (tl.load(idx_ptr + c * TILE + offs) == b).to(tl.int32)
        else:
            for c in range(0, NCH):
                acc_i += (tl.load(idx_ptr + c * TILE + offs) == b).to(tl.int32)
        tl.store(hist_ptr + b, tl.sum(acc_i, axis=0).to(ACC_DTYPE))


@libentry()
@triton.jit
def histogramdd_from_bin_cts_density_kernel(
    hist_ptr,  # (num_bins,) counts, normalised in place
    lohi_ptr,  # (2 * D,) per-dimension [lo, hi] pairs
    num_bins,
    NCH: tl.constexpr,  # bin-tile count
    UNROLL: tl.constexpr,
    D: tl.constexpr,
    BINS: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Density normalisation in place: count / (total * bin_volume).

    Uniform bins make the bin volume a single scalar (the product of the
    per-dimension bin widths).  The static per-dimension loop keeps the product
    a chain of scalars rather than a narrow-vector reduce (which this backend
    rejects at these widths), and the division is deliberately unguarded: ATen
    divides straight through, so a zero total yields inf/nan and those must
    propagate.
    """
    offs = tl.arange(0, TILE)
    acc = tl.zeros((TILE,), dtype=ACC_DTYPE)
    if UNROLL:
        for c in tl.static_range(NCH):
            cur = c * TILE + offs
            vals = tl.load(hist_ptr + tl.minimum(cur, num_bins - 1)).to(ACC_DTYPE)
            acc += tl.where(cur < num_bins, vals, 0.0)
    else:
        for c in range(0, NCH):
            cur = c * TILE + offs
            vals = tl.load(hist_ptr + tl.minimum(cur, num_bins - 1)).to(ACC_DTYPE)
            acc += tl.where(cur < num_bins, vals, 0.0)
    total = tl.sum(acc, axis=0)

    volume = tl.load(lohi_ptr).to(ACC_DTYPE) * 0.0 + 1.0
    for d in tl.static_range(D):
        lo = tl.load(lohi_ptr + 2 * d).to(ACC_DTYPE)
        hi = tl.load(lohi_ptr + 2 * d + 1).to(ACC_DTYPE)
        volume = volume * ((hi - lo) / (BINS[d] * 1.0))

    scale = 1.0 / (total * volume)
    for c in tl.static_range(NCH):
        cur = c * TILE + offs
        tail = cur < num_bins
        # Masked store on purpose: clamping the address would let several lanes
        # rewrite the last bin from its already normalised value.
        vals = tl.load(hist_ptr + tl.minimum(cur, num_bins - 1)).to(ACC_DTYPE)
        tl.store(hist_ptr + cur, vals * scale, mask=tail)


def _format_bound(value):
    """Format a range bound the way ATen's range messages do (``%g``)."""
    return "%g" % value


def _resolve_range(inp, range_, D, acc_dtype):
    """Return a (2 * D,) device tensor of per-dimension [left, right] pairs.

    Mirrors ``torch.histogramdd``: an explicit ``range`` is validated (finite
    bounds first, then ``min <= max``) and a degenerate dimension is widened to
    ``(min - 0.5, max + 0.5)``; with ``range=None`` the per-dimension min/max
    comes from the data on the device, with a NaN/inf value making the range
    non-finite and raising ATen's error.
    """
    device = inp.device
    if range_ is not None:
        rng = [float(v) for v in range_]
        if len(rng) != 2 * D:
            raise RuntimeError(
                f"torch.histogramdd: for a {D}-dimensional histogram range should "
                f"have {2 * D} elements, but got {len(rng)}"
            )
        for d in builtins.range(D):
            lo, hi = rng[2 * d], rng[2 * d + 1]
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise RuntimeError(
                    f"torch.histogramdd: dimension {d}'s range "
                    f"[{_format_bound(lo)}, {_format_bound(hi)}] is not finite"
                )
            if lo > hi:
                raise RuntimeError(
                    f"torch.histogramdd: min should not exceed max, but got "
                    f"min {_format_bound(lo)} max {_format_bound(hi)} for "
                    f"dimension {d}"
                )
        for d in builtins.range(D):
            if rng[2 * d] == rng[2 * d + 1]:
                rng[2 * d] -= 0.5
                rng[2 * d + 1] += 0.5
        return torch.tensor(rng, dtype=acc_dtype, device=device)

    M = inp.shape[1]
    if M == 0:
        # No data to reduce over; the empty auto-range collapses to (0, 1).
        return torch.tensor([0.0, 1.0] * D, dtype=acc_dtype, device=device)

    dpad = triton.next_power_of_2(D)
    lohi = torch.empty(2 * D, dtype=acc_dtype, device=device)
    with torch_device_fn.device(device):
        histogramdd_from_bin_cts_range_kernel[(1,)](
            inp,
            lohi,
            M,
            D=D,
            DPAD=dpad,
            ACC_DTYPE=tl.float64 if acc_dtype == torch.float64 else tl.float32,
            TILE=RANGE_TILE,
        )
    # The degenerate-range expansion leaves a NaN range NaN and an infinite
    # bound infinite, so the ATen "range is not finite" check reads straight off
    # the device-computed bounds (one host read of 2 * D scalars).
    bounds = lohi.tolist()
    for d in builtins.range(D):
        lo, hi = bounds[2 * d], bounds[2 * d + 1]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise RuntimeError(
                f"torch.histogramdd: dimension {d}'s range "
                f"[{_format_bound(lo)}, {_format_bound(hi)}] is not finite"
            )
    return lohi


def _histogramdd_from_bin_cts_impl(
    self, bins, *, range=None, weight=None, density=False, out=None
):
    """Shared implementation for the base and ``.out`` ATen variants."""
    logger.debug("GEMS_KUNLUNXIN _HISTOGRAMDD_FROM_BIN_CTS")

    if self.ndim < 2:
        raise RuntimeError(
            "torch.histogramdd: input tensor should have at least 2 dimensions"
        )
    if self.shape[-1] != len(bins):
        raise RuntimeError(
            "histogramdd: The size of bins must be equal to the innermost "
            "dimension of the input."
        )

    # torch._histogramdd_from_bin_cts is only implemented for floating types.
    if not self.is_floating_point():
        raise NotImplementedError('"histogramdd" not implemented for this dtype')

    if weight is not None and weight.dtype != self.dtype:
        raise RuntimeError(
            "torch.histogramdd: if weight tensor is provided, input tensor and "
            "weight tensor should have the same dtype, but got "
            f"input({self.dtype}), and weight({weight.dtype})"
        )

    D = self.shape[-1]
    bins_list = [int(b) for b in bins]
    out_shape = tuple(bins_list)

    # Flatten all leading dims to (M, D) and make contiguous.
    inp = self.reshape(-1, D).contiguous()
    M = inp.shape[0]
    out_dtype = self.dtype

    # Accumulate in the input's dtype: float32 for float32 inputs and float64
    # for float64 inputs, so double-precision inputs keep their precision.
    acc_dtype = self.dtype
    acc_triton_dtype = tl.float64 if acc_dtype == torch.float64 else tl.float32

    # Allocate / validate the output buffer. ATen resizes a mis-shaped out
    # tensor in place and keeps the caller's object identity, but rejects a
    # dtype mismatch outright.
    if out is not None:
        if out.dtype != out_dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {out_dtype}, but got "
                f"{out.dtype} instead"
            )
        if out.device != self.device:
            raise RuntimeError(
                f"Expected out tensor to have device {self.device}, but got "
                f"{out.device} instead"
            )
        if tuple(out.shape) != out_shape:
            out.resize_(out_shape)
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=self.device)

    if out.numel() == 0:
        # No bins to fill; zero the output and return.
        out.zero_()
        return out

    num_bins = 1
    for b in bins_list:
        num_bins *= b
    density_chunks = triton.cdiv(
        num_bins, min(DENSITY_TILE, max(MIN_DIM_TILE, triton.next_power_of_2(num_bins)))
    )

    bins_key = tuple(bins_list)
    strides = [1] * D
    for d in builtins.range(D - 2, -1, -1):
        strides[d] = strides[d + 1] * bins_list[d + 1]

    edge_tile = max(MIN_DIM_TILE, triton.next_power_of_2(max(bins_list) + 1))

    # For the plain histogram, write straight into the output buffer: out's
    # dtype is validated to equal the input dtype, which is also the
    # accumulator dtype, and every bin has exactly one writer. A
    # non-contiguous out cannot be viewed as flat, so use a temporary.
    hist_dense = out if out.is_contiguous() else torch.empty_like(out)
    hist_ptr = hist_dense.view(num_bins)

    with torch_device_fn.device(self.device):
        # Per-dimension coordinates become contiguous, so the range and index
        # passes read stride-1 block DMA instead of a strided gather.
        input_t = inp.t().contiguous() if M > 0 else inp
        lohi = _resolve_range(input_t, range, D, acc_dtype)

        # The index pass compares int32 keys of the float edges (see the edges
        # kernel for why a float compare is wrong on this backend), so the edge
        # slabs and the (lo, hi) pairs are materialised as keys.
        edges_key = torch.empty(D * edge_tile, dtype=torch.int32, device=self.device)
        lohi_key = torch.empty(2 * D, dtype=torch.int32, device=self.device)
        histogramdd_from_bin_cts_edges_kernel[(1,)](
            lohi,
            edges_key,
            lohi_key,
            D=D,
            BINS=bins_key,
            EDGE_TILE=edge_tile,
            ACC_DTYPE=acc_triton_dtype,
        )

        if M == 0:
            # No point can land in any bin; density still has to divide the
            # zero counts by a zero total (ATen yields NaN there).
            hist_dense.zero_()
        else:
            # The counting pass scans the index scratch without a mask, so it is
            # padded to a whole number of count tiles and the padding lanes are
            # tagged -1 by the index pass. The count tile shrinks with the point
            # count: a fixed 4096-lane tile would make a many-bin/small-point
            # histogram scan mostly padding.
            count_tile = min(COUNT_TILE, max(MIN_DIM_TILE, triton.next_power_of_2(M)))
            index_cap = INDEX_TILE_WIDE if D <= 2 else INDEX_TILE
            index_tile = min(index_cap, max(MIN_DIM_TILE, triton.next_power_of_2(M)))
            unit = max(count_tile, index_tile)
            padded = triton.cdiv(M, unit) * unit
            count_chunks = padded // count_tile
            idx = torch.empty(padded, dtype=torch.int32, device=self.device)
            histogramdd_from_bin_cts_index_kernel[(padded // index_tile,)](
                input_t,
                edges_key,
                lohi_key,
                idx,
                M,
                D=D,
                BINS=bins_key,
                STRIDES=tuple(strides),
                EDGE_TILE=edge_tile,
                ACC_DTYPE=acc_triton_dtype,
                TILE=index_tile,
            )
            has_weight = weight is not None
            if has_weight:
                weight = weight.reshape(-1).contiguous()
                if weight.numel() != M:
                    raise RuntimeError(
                        "_histogramdd_from_bin_cts: weight must have the same "
                        "number of rows as the input"
                    )
                weight_ptr = weight
            else:
                # Triton needs a valid pointer slot even when HAS_WEIGHT=False.
                weight_ptr = torch.empty(1, dtype=inp.dtype, device=self.device)
            histogramdd_from_bin_cts_count_kernel[(num_bins,)](
                idx,
                weight_ptr,
                hist_ptr,
                M,
                NCH=count_chunks,
                # The fp32 (weighted) accumulate body does not tolerate a deep
                # static unroll as well as the int32 one (measured 3x slower at
                # 25 chunks), so only the unweighted path unrolls.
                UNROLL=(not has_weight) and count_chunks <= COUNT_UNROLL_MAX,
                HAS_WEIGHT=has_weight,
                ACC_DTYPE=acc_triton_dtype,
                TILE=count_tile,
            )

        if density:
            # The scaling pass re-reads the counts and writes them back with a
            # tail mask. A masked store inside a *statically unrolled* chunk loop
            # reuses the last chunk's mask for every earlier chunk on this
            # backend (bins near the end of chunk 0 stayed unscaled and were
            # left out of the total), so a multi-chunk tail is only safe with
            # the dynamic loop; covering the histogram with one tile avoids the
            # hazard entirely.
            dens_tile = min(DENSITY_TILE, max(MIN_DIM_TILE, triton.next_power_of_2(num_bins)))
            histogramdd_from_bin_cts_density_kernel[(1,)](
                hist_ptr,
                lohi,
                num_bins,
                NCH=density_chunks,
                UNROLL=density_chunks == 1,
                D=D,
                BINS=bins_key,
                ACC_DTYPE=acc_triton_dtype,
                TILE=dens_tile,
            )

    if hist_dense is not out:
        out.copy_(hist_dense)
    return out


def _histogramdd_from_bin_cts(self, bins, *, range=None, weight=None, density=False):
    """aten::_histogramdd_from_bin_cts(Tensor self, int[] bins, *, float[]? range=None,
    Tensor? weight=None, bool density=False) -> Tensor
    """
    return _histogramdd_from_bin_cts_impl(
        self, bins, range=range, weight=weight, density=density, out=None
    )


def _histogramdd_from_bin_cts_out(
    self, bins, *, range=None, weight=None, density=False, out
):
    """aten::_histogramdd_from_bin_cts.out(Tensor self, int[] bins, *, float[]? range=None,
    Tensor? weight=None, bool density=False, Tensor(a!) out) -> Tensor(a!)
    """
    return _histogramdd_from_bin_cts_impl(
        self, bins, range=range, weight=weight, density=density, out=out
    )
