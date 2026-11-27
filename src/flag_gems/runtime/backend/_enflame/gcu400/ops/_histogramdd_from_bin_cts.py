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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# The ATen schema names the keyword arg ``range``, which shadows the builtin.
# Grab the builtin here so host-side loops still work inside the implementation.
import builtins as _builtins  # noqa: E402

# GCU400 has 24 SIPs; at most 48 CTAs run concurrently, so grids above that
# only add wave-quantisation and per-program setup cost.
NUM_SIPS = 24
MAX_CTAS = NUM_SIPS * 2

# Host-side metadata caches. ``torch.tensor`` copies to the device cost ~30 us
# each on pageable transfers, and the per-call host path used to pay four of
# them (lefts / rights / strides / bins) plus an edges kernel launch on every
# invocation. The metadata depends only on (device, bins, range), so it is
# built once and reused; repeated calls with the same configuration -- the
# dominant pattern in serving loops and benchmarks -- skip the transfers
# entirely. Entries are small (a few hundred floats); the cap keeps a caller
# sweeping many distinct ranges from growing the cache without bound.
_META_CACHE_CAP = 64
_BINS_META_CACHE = {}
_RANGE_META_CACHE = {}
_PARTIALS_CACHE = {}

# Above this size a weighted call falls back from the per-program private
# slab (one row of ``num_bins`` per program) to direct atomics, so a huge
# point cloud over a wide bin grid cannot force a multi-GB scratch tensor.
_MAX_SLAB_BYTES = 64 * 1024 * 1024

# Dispatch thresholds, measured on GCU400 with the bench shapes
# (see the module history / case log for the probe data):
#   * one-hot int8 ``tl.dot`` beats ``tl.histogram`` for small bin counts
#     (the histogram primitive pays ~69 ns per bin per program, so E=512
#     costs ~100 us per program while the dot's per-point cost grows only
#     with the bin count);
#   * for wide bin vectors the per-point one-hot compare cost overtakes it
#     and ``tl.histogram`` + a partial-store wins again;
#   * per-program atomics serialize at ~100 ns per lane on GCU400, so
#     histogram combination always goes through per-program partials plus
#     a single reduce kernel instead of same-address global atomics.
_DOT_PERSISTENT_MAX_BINS = 128  # HIST_BINS <= this: single-chunk persistent dot
_DOT_MAX_POINTS = 32768  # M below this: chunked dot, else tl.histogram
_HIST_BLOCK = 2048  # BLOCK_SIZE for the tl.histogram path
_DOT_BLOCK = 512
_SLAB_BLOCK = 512


def _meta_cache_put(cache, key, value):
    if len(cache) >= _META_CACHE_CAP:
        cache.clear()
    cache[key] = value
    return value


@triton.jit
def _histdd_locate(
    inp_ptr,
    row_start,
    row_mask,
    M,
    N,
    left_ptr,
    right_ptr,
    edges_ptr,
    strides_ptr,
    bins_ptr,
    MAX_EDGES,
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Locate each row's flat bin index, or mark the row invalid.

    Shared prologue of every counting kernel: walks the dimensions, range-
    checks the coordinate and binary-searches the explicit edge array that
    ATen builds with linspace. Returns (linear_idx, valid); the caller routes
    invalid lanes wherever its counting strategy prefers (padding bin, or a
    masked-off atomic).
    """
    valid = row_mask
    # num_bins is bounded by the product of the small per-dimension bin
    # counts, so a 32-bit linear index is sufficient and avoids GCU's slow
    # int64 path.
    linear_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for d in range(N_CONST):
        coord = tl.load(
            inp_ptr + row_start * N_CONST + d, mask=row_mask, other=float("nan")
        ).to(tl.float32)
        left = tl.load(left_ptr + d).to(tl.float32)
        right = tl.load(right_ptr + d).to(tl.float32)
        nbins = tl.load(bins_ptr + d)
        stride = tl.load(strides_ptr + d)

        # Only count points inside [left, right] (NaN fails both comparisons).
        in_range = (coord >= left) & (coord <= right)
        valid = valid & in_range

        # Locate the bin by searching the explicit edge array, which is what
        # ATen does -- it materialises the edges with linspace and takes an
        # upper bound over them. No closed-form index reproduces that:
        # computing floor((coord - left) * bins / width) puts 0.3 over
        # [0, 1] with 10 bins in bin 3 where ATen says 2, and
        # floor((coord - left) / bin_width) fixes 0.3 but then misplaces
        # 0.6 and 0.7. Searching the same edges the native op builds agrees
        # on every probed value.
        row = edges_ptr + d * MAX_EDGES
        lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        hi = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32) + nbins
        # Binary search for the last edge <= coord, over at most
        # log2(MAX_EDGES) steps; the trip count is a compile-time bound so
        # the loop unrolls.
        for _ in range(MAX_EDGE_STEPS):
            mid = (lo + hi + 1) // 2
            mid_c = tl.minimum(mid, nbins)
            edge = tl.load(row + mid_c, mask=valid, other=float("inf")).to(tl.float32)
            take = (edge <= coord) & (mid <= nbins)
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        bin_idx = lo

        # The rightmost bin is inclusive of its right edge, so a point
        # exactly on `right` searches to index nbins and must fold back
        # into the last bin.
        bin_idx = tl.where(bin_idx >= nbins, nbins - 1, bin_idx)
        bin_idx = tl.where(bin_idx < 0, 0, bin_idx)

        linear_idx = linear_idx + bin_idx * stride

    return linear_idx, valid


@libentry()
@triton.jit
def histogramdd_dot_persistent_kernel(
    inp_ptr,  # (M, N) contiguous float32 input
    part_ptr,  # (grid, HIST_BINS) int32 per-program partial counts
    M,
    N,
    left_ptr,  # (N,) left edges
    right_ptr,  # (N,) right edges
    edges_ptr,  # (N, MAX_EDGES) per-dimension bin edges, row-padded
    strides_ptr,  # (N,) row-major strides in bins
    bins_ptr,  # (N,) number of bins per dim
    MAX_EDGES,
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    HIST_BINS: tl.constexpr,
    NB_CHUNK: tl.constexpr,
):
    """Count points with a one-hot int8 ``tl.dot`` on a persistent grid.

    ``count[b] = sum_e (key_e == b)`` lowers to
    ``ones[16, BLOCK] @ onehot[BLOCK, NBINS]`` on the MMA path, which is far
    cheaper than the serial ``tl.histogram`` update for narrow bin vectors.
    The grid is capped at the number of resident CTAs and each program
    strides over point blocks, accumulating its partial histogram across
    iterations; one partial row per program is then summed by the reduce
    kernel. No global atomics are issued at all.
    """
    pid = ext.program_id(0)
    num_pids = tl.num_programs(0)
    ones = tl.full((16, BLOCK_SIZE), 1, dtype=tl.int8)
    acc = tl.zeros((16, NB_CHUNK), dtype=tl.int32)
    nblocks = tl.cdiv(M, BLOCK_SIZE)
    for blk in tl.range(pid, nblocks, num_pids):
        row_start = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = row_start < M
        linear_idx, valid = _histdd_locate(
            inp_ptr,
            row_start,
            row_mask,
            M,
            N,
            left_ptr,
            right_ptr,
            edges_ptr,
            strides_ptr,
            bins_ptr,
            MAX_EDGES,
            MAX_EDGE_STEPS,
            N_CONST,
            BLOCK_SIZE,
        )
        # Invalid lanes are routed into the padding bin and dropped by the
        # store mask below (tl.histogram's mask is ignored by the GCU
        # backend; the same convention keeps the dot path uniform).
        key = tl.where(valid, linear_idx, HIST_BINS - 1)
        cols = tl.arange(0, NB_CHUNK)
        onehot = tl.where(
            key[:, None] == cols[None, :],
            tl.full((BLOCK_SIZE, NB_CHUNK), 1, tl.int8),
            tl.zeros((BLOCK_SIZE, NB_CHUNK), dtype=tl.int8),
        )
        acc = tl.dot(ones, onehot, acc, out_dtype=tl.int32)
    # All 16 rows of `ones` are identical, so any row of `acc` holds the
    # counts; `tl.max` extracts one of them as a flat vector.
    counts = tl.max(acc, axis=0)
    out_cols = tl.arange(0, NB_CHUNK)
    tl.store(part_ptr + pid * HIST_BINS + out_cols, counts, mask=out_cols < HIST_BINS)


@libentry()
@triton.jit
def histogramdd_dot_kernel(
    inp_ptr,  # (M, N) contiguous float32 input
    part_ptr,  # (grid, HIST_BINS) int32 per-program partial counts
    M,
    N,
    left_ptr,  # (N,) left edges
    right_ptr,  # (N,) right edges
    edges_ptr,  # (N, MAX_EDGES) per-dimension bin edges, row-padded
    strides_ptr,  # (N,) row-major strides in bins
    bins_ptr,  # (N,) number of bins per dim
    MAX_EDGES,
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    HIST_BINS: tl.constexpr,
    NB_CHUNK: tl.constexpr,
):
    """Chunked one-hot int8 ``tl.dot`` counting, one program per point tile.

    Used when the padded bin vector is wider than the single-chunk DSM
    budget (BLOCK x CHUNK int8 <= 64 KB); the bin axis is swept in NB_CHUNK
    chunks, each producing its own ``tl.dot``. Partials are stored per
    program and reduced by ``histogramdd_reduce_i32_kernel``.
    """
    pid = ext.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        M,
        N,
        left_ptr,
        right_ptr,
        edges_ptr,
        strides_ptr,
        bins_ptr,
        MAX_EDGES,
        MAX_EDGE_STEPS,
        N_CONST,
        BLOCK_SIZE,
    )
    key = tl.where(valid, linear_idx, HIST_BINS - 1)
    ones = tl.full((16, BLOCK_SIZE), 1, dtype=tl.int8)
    for cbase in tl.static_range(0, HIST_BINS, NB_CHUNK):
        cols = cbase + tl.arange(0, NB_CHUNK)
        onehot = tl.where(
            key[:, None] == cols[None, :],
            tl.full((BLOCK_SIZE, NB_CHUNK), 1, tl.int8),
            tl.zeros((BLOCK_SIZE, NB_CHUNK), dtype=tl.int8),
        )
        acc = tl.dot(ones, onehot, out_dtype=tl.int32)
        counts = tl.max(acc, axis=0)
        tl.store(part_ptr + pid * HIST_BINS + cols, counts, mask=cols < HIST_BINS)


@libentry()
@triton.jit
def histogramdd_hist_partial_kernel(
    inp_ptr,  # (M, N) contiguous float32 input
    part_ptr,  # (grid, HIST_BINS) int32 per-program partial counts
    M,
    N,
    left_ptr,  # (N,) left edges
    right_ptr,  # (N,) right edges
    edges_ptr,  # (N, MAX_EDGES) per-dimension bin edges, row-padded
    strides_ptr,  # (N,) row-major strides in bins
    bins_ptr,  # (N,) number of bins per dim
    MAX_EDGES,
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HIST_BINS: tl.constexpr,
):
    """``tl.histogram`` per program, storing a private partial row.

    For wide bin vectors (HIST_BINS > 128) the per-point one-hot compare cost
    of the dot path overtakes the histogram primitive's per-bin cost, and
    ``tl.histogram`` wins again. Each program counts its tile into a private
    partial row -- no global atomics, no zero-initialised accumulator; the
    reduce kernel sums the rows into the output.
    """
    pid = ext.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        M,
        N,
        left_ptr,
        right_ptr,
        edges_ptr,
        strides_ptr,
        bins_ptr,
        MAX_EDGES,
        MAX_EDGE_STEPS,
        N_CONST,
        BLOCK_SIZE,
    )
    # tl.histogram's mask is ignored by the GCU backend, so invalid lanes are
    # routed into the last (padding) bin and dropped by the store mask.
    hist_input = tl.where(valid, linear_idx, HIST_BINS - 1)
    local_counts = tl.histogram(hist_input, HIST_BINS)
    bin_offsets = tl.arange(0, HIST_BINS)
    tl.store(
        part_ptr + pid * HIST_BINS + bin_offsets,
        local_counts,
        mask=bin_offsets < HIST_BINS,
    )


@libentry()
@triton.jit
def histogramdd_direct_kernel(
    inp_ptr,  # (M, N) contiguous float32 input
    weight_ptr,  # (M,) contiguous weights, or a dummy when HAS_WEIGHT=False
    out_ptr,  # (num_bins,) zero-initialised accumulator
    M,
    N,
    left_ptr,  # (N,) left edges
    right_ptr,  # (N,) right edges
    edges_ptr,  # (N, MAX_EDGES) per-dimension bin edges, row-padded
    strides_ptr,  # (N,) row-major strides in bins
    bins_ptr,  # (N,) number of bins per dim
    MAX_EDGES,
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """Per-point atomic accumulation into one shared histogram buffer.

    Kept for bin grids too wide for any per-program aggregation to pay off
    (the per-point one-hot / histogram cost grows with the bin count while
    the atomic cost does not), and as the fallback for pathological weighted
    calls whose private slab would exceed the scratch budget. The buffer
    must be zeroed before launch; combination across programs happens
    through same-address atomics, which is fine here because wide bin grids
    keep the per-address collision count low.
    """
    pid = ext.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        M,
        N,
        left_ptr,
        right_ptr,
        edges_ptr,
        strides_ptr,
        bins_ptr,
        MAX_EDGES,
        MAX_EDGE_STEPS,
        N_CONST,
        BLOCK_SIZE,
    )
    if HAS_WEIGHT:
        contrib = tl.load(weight_ptr + row_start, mask=valid, other=0.0).to(tl.float32)
    else:
        contrib = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    tl.atomic_add(out_ptr + linear_idx, contrib, mask=valid, sem="relaxed")


@libentry()
@triton.jit
def histogramdd_weighted_slab_kernel(
    inp_ptr,  # (M, N) contiguous float32 input
    weight_ptr,  # (M,) contiguous float32 weights
    slab_ptr,  # (grid, NUM_BINS) zero-initialised private rows
    M,
    N,
    left_ptr,  # (N,) left edges
    right_ptr,  # (N,) right edges
    edges_ptr,  # (N, MAX_EDGES) per-dimension bin edges, row-padded
    strides_ptr,  # (N,) row-major strides in bins
    bins_ptr,  # (N,) number of bins per dim
    MAX_EDGES,
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    """Weighted accumulation into per-program private slab rows.

    Weights break every counting primitive (``tl.histogram`` counts
    occurrences, the one-hot dot counts too), so a weighted histogram has to
    scatter-add one contribution per point. Doing that into one shared
    buffer serialises same-address atomics across all programs: for a
    million points over twenty bins that is a ~50 000-way collision per bin
    and measured 74 ms per call. Giving every program a private row caps the
    collision depth at the per-tile level; the reduce kernel then sums the
    rows. The slab is freshly zeroed per call.
    """
    pid = ext.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        M,
        N,
        left_ptr,
        right_ptr,
        edges_ptr,
        strides_ptr,
        bins_ptr,
        MAX_EDGES,
        MAX_EDGE_STEPS,
        N_CONST,
        BLOCK_SIZE,
    )
    contrib = tl.load(weight_ptr + row_start, mask=valid, other=0.0).to(tl.float32)
    tl.atomic_add(
        slab_ptr + pid * NUM_BINS + linear_idx, contrib, mask=valid, sem="relaxed"
    )


@libentry()
@triton.jit
def histogramdd_reduce_i32_kernel(
    part_ptr,  # (n_prog, HIST_BINS) int32 partial counts
    out_ptr,  # (num_bins,) float32 out; fully overwritten
    n_prog,
    HIST_BINS: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Sum per-program int32 partial rows into the float32 output.

    Writes every bin (including untouched ones, as zeros), so the output
    needs no zero-initialisation and the reduce doubles as the full
    overwrite required by the ``.out`` contract.
    """
    pid = ext.program_id(0)
    bins = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    m = bins < NUM_BINS
    acc = tl.zeros((BLOCK_B,), dtype=tl.int32)
    for p in range(0, n_prog):
        acc += tl.load(part_ptr + p * HIST_BINS + bins, mask=m, other=0)
    tl.store(out_ptr + bins, acc.to(tl.float32), mask=m)


@libentry()
@triton.jit
def histogramdd_reduce_slab_kernel(
    slab_ptr,  # (n_prog, NUM_BINS) float32 private weighted rows
    out_ptr,  # (num_bins,) float32 out; fully overwritten
    n_prog,
    NUM_BINS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Sum per-program float32 slab rows into the output histogram."""
    pid = ext.program_id(0)
    bins = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    m = bins < NUM_BINS
    acc = tl.zeros((BLOCK_B,), dtype=tl.float32)
    for p in range(0, n_prog):
        acc += tl.load(slab_ptr + p * NUM_BINS + bins, mask=m, other=0.0)
    tl.store(out_ptr + bins, acc, mask=m)


@libentry()
@triton.jit
def histogramdd_density_kernel(
    hist_ptr,  # (num_bins,) counts in ACC_DTYPE
    out_ptr,  # (num_bins,) density output (input dtype)
    scale_ptr,  # () scalar: 1 / (total_count * bin_volume)
    num_bins,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize counts into a density: density[i] = count[i] / (total * volume)."""
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_bins
    count = tl.load(hist_ptr + offset, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    # Stay in ACC_DTYPE so an fp64 histogram keeps full precision, and so a
    # non-finite scale (see the density notes in the host function) propagates
    # exactly as ATen's division does.
    density = count.to(ACC_DTYPE) * scale.to(ACC_DTYPE)
    tl.store(out_ptr + offset, density, mask=mask)


@libentry()
@triton.jit
def histogramdd_col_range_kernel(
    inp_ptr,  # (M, N) contiguous input
    left_ptr,  # (N,) out: per-dimension left edge
    right_ptr,  # (N,) out: per-dimension right edge
    nonfinite_ptr,  # () out: lowest dimension with a non-finite value
    M,
    N,
    ACC_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    """Per-column min/max, with the degenerate column widened to (min-.5, max+.5).

    One program handles a tile of BLOCK_N columns and strides down the rows, so
    the whole auto-range computation happens on the device: this replaces host
    calls to ``torch.aminmax``/``torch.where``, which the FlagGems host-function
    convention does not allow. Results are written in ACC_DTYPE to keep fp64
    inputs exact.

    The kernel also reports, into the single-element ``nonfinite_ptr``, the
    lowest dimension whose data contains a NaN or +-Inf. The host then only
    has to read that one scalar to decide whether to raise, instead of
    running ``isfinite`` / ``nonzero`` on the whole range tensor.
    """
    pid = ext.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc_min = tl.full((BLOCK_N,), float("inf"), dtype=ACC_DTYPE)
    acc_max = tl.full((BLOCK_N,), float("-inf"), dtype=ACC_DTYPE)
    # Whether each column has seen a NaN anywhere. Tracked separately from
    # the min/max accumulators so the NaN state is never carried through a
    # float reduction: ``tl.min``/``tl.max``/``tl.minimum``/``tl.maximum`` all
    # lower to fmin/fmax, which *ignore* a NaN operand, so a NaN folded into
    # the accumulator would simply be dropped by the identity +inf/-inf.
    nan_seen = tl.zeros((BLOCK_N,), dtype=tl.int1)
    # Whether each column has seen a NaN or +-Inf, i.e. whether its auto-range
    # cannot be finite. Separate from ``nan_seen`` because ATen's range check
    # rejects infinities too, which the min/max accumulators carry silently.
    nonfinite_seen = tl.zeros((BLOCK_N,), dtype=tl.int1)

    for row_base in range(0, M, BLOCK_M):
        rows = row_base + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        vals = tl.load(
            inp_ptr + rows[:, None] * N + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=float("nan"),
        ).to(ACC_DTYPE)
        # Masked-off lanes hold the NaN filler, so they are excluded
        # explicitly; they are not part of the data.
        live = row_mask[:, None] & col_mask[None, :]
        # Covariant with ``aminmax``: a column holding a NaN gets a NaN
        # range, not the min/max of the remaining values.
        nan_seen = nan_seen | (tl.sum((live & (vals != vals)).to(tl.int32), axis=0) > 0)
        nonfinite_seen = nonfinite_seen | (
            tl.sum(
                (live & ((vals != vals) | (tl.abs(vals) == float("inf")))).to(tl.int32),
                axis=0,
            )
            > 0
        )
        acc_min = tl.minimum(acc_min, tl.min(vals, axis=0))
        acc_max = tl.maximum(acc_max, tl.max(vals, axis=0))

    # A NaN column's range is NaN in both bounds; NaN == NaN is false, so the
    # degenerate-range expansion below leaves it alone, exactly as ATen does.
    acc_min = tl.where(nan_seen, float("nan"), acc_min)
    acc_max = tl.where(nan_seen, float("nan"), acc_max)

    # A dimension whose min equals max is expanded, as torch.histogramdd does.
    same = acc_min == acc_max
    lefts = tl.where(same, acc_min - 0.5, acc_min)
    rights = tl.where(same, acc_max + 0.5, acc_max)
    tl.store(left_ptr + cols, lefts, mask=col_mask)
    tl.store(right_ptr + cols, rights, mask=col_mask)

    # Publish the lowest offending dimension. ``tl.min`` within the tile plus
    # ``tl.atomic_min`` across programs reproduces ``(~finite).nonzero()[0]``
    # without a host reduction. Lanes outside the tile (and every lane when
    # the column is finite) contribute the sentinel ``N``, which is above
    # every valid dimension, so a finite input leaves the host's ``N``
    # untouched.
    cand = tl.min(tl.where(nonfinite_seen & col_mask, cols, SENTINEL), axis=0)
    tl.atomic_min(nonfinite_ptr, cand, sem="relaxed")


@libentry()
@triton.jit
def histogramdd_edges_kernel(
    left_ptr,  # (N,) left edges
    right_ptr,  # (N,) right edges
    bins_ptr,  # (N,) bins per dimension
    edges_ptr,  # (N, MAX_EDGES) out: padded per-dimension edge arrays
    N,
    MAX_EDGES,
    ACC_DTYPE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Materialise the per-dimension bin edges the binning kernel searches.

    Reproduces ``torch.linspace`` bit-for-bit, which is how ATen builds these
    edges. The precision matters more than usual here: a point sitting exactly
    on an edge is placed by an exact comparison, so a 1-ulp error in an edge
    moves that point a whole bin. Each rule below was settled by measuring
    against torch.linspace, with the mismatch counts over random
    configurations noted for the rejected variants.

    * Interpolate from the nearer endpoint: ``step * i + left`` below the
      halfway index, ``right - step * (steps-1-i)`` at or above it.
      Accumulating from the left throughout puts edge 6 of [0, 1]/10 bins at
      0.6000000000000001 rather than 0.6, so a point at exactly 0.6 lands
      one bin low (113/300 disagree).
    * Derive ``step`` from endpoints already narrowed to the storage dtype.

    GCU400 has no FP64 vector path: fp64 tensors are actually stored as fp32
    by torch_gcu and fp64 arithmetic crashes the GCU compiler. The whole
    linspace reproduction therefore stays in fp32 here, using ``div_rn`` for
    a correctly rounded step (plain fp32 `/` lowers to a fast reciprocal).
    """
    d = ext.program_id(0)
    if d < N:
        left = tl.load(left_ptr + d)
        right = tl.load(right_ptr + d)
        nbins = tl.load(bins_ptr + d)
        steps = nbins + 1
        half = steps // 2
        l_edge = left.to(ACC_DTYPE)
        r_edge = right.to(ACC_DTYPE)
        step = tl.math.div_rn(r_edge - l_edge, nbins.to(ACC_DTYPE))
        for base in range(0, MAX_EDGES, BLOCK_E):
            idx = base + tl.arange(0, BLOCK_E)
            m = idx < MAX_EDGES
            fi = idx.to(ACC_DTYPE)
            rev = (steps - 1 - idx).to(ACC_DTYPE)
            val = tl.where(idx < half, step * fi + l_edge, r_edge - step * rev)
            # Padding lanes hold +inf so the binning search never selects
            # them.
            val = tl.where(idx <= nbins, val, float("inf"))
            tl.store(edges_ptr + d * MAX_EDGES + idx, val, mask=m)


@libentry()
@triton.jit
def histogramdd_bin_geometry_kernel(
    left_ptr,  # (N,) left edges
    right_ptr,  # (N,) right edges
    bins_ptr,  # (N,) bins per dimension
    width_ptr,  # (N,) out: right - left
    bin_width_ptr,  # (N,) out: width / bins (width == 0 -> 1)
    N,
    ACC_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Derive per-dimension widths and bin widths on the device.

    Keeps this off the host, so no ``.item()`` per dimension is needed to
    build the kernel arguments.
    """
    cols = ext.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    left = tl.load(left_ptr + cols, mask=mask, other=0.0).to(ACC_DTYPE)
    right = tl.load(right_ptr + cols, mask=mask, other=0.0).to(ACC_DTYPE)
    nbins = tl.load(bins_ptr + cols, mask=mask, other=1).to(ACC_DTYPE)

    width = right - left
    # Guard a zero width so the division in the density scale stays finite;
    # such points only match the ``coord == right`` branch in the binning
    # anyway.
    safe = tl.where(width == 0, 1.0, width)
    tl.store(width_ptr + cols, width, mask=mask)
    tl.store(bin_width_ptr + cols, safe / nbins, mask=mask)


@libentry()
@triton.jit
def histogramdd_total_scale_kernel(
    hist_ptr,  # (num_bins,) counts
    log_width_ptr,  # () log of the absolute bin volume
    sign_ptr,  # () sign of the bin volume
    scale_ptr,  # () out: 1 / (total_count * bin_volume)
    num_bins,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum the histogram and turn it into the density scale, entirely on device.

    Replaces the host ``hist_acc.sum().item()``. The bin volume is accumulated
    in the log domain and re-exponentiated so a many-dimensional product
    cannot overflow before the division. The division is left unguarded on
    purpose: ATen divides straight through, so a zero total yields inf/nan
    and those must propagate (see the density notes in the host function).
    """
    total = tl.zeros((), dtype=ACC_DTYPE)
    for base in range(0, num_bins, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        m = offs < num_bins
        total += tl.sum(tl.load(hist_ptr + offs, mask=m, other=0.0).to(ACC_DTYPE))

    # The bin volume arrives already reduced (a signed magnitude in the log
    # domain) because it depends only on the bins list and the edges, not on
    # the data.
    log_vol = tl.load(log_width_ptr).to(ACC_DTYPE)
    sign = tl.load(sign_ptr).to(ACC_DTYPE)
    bin_volume = sign * tl.exp(log_vol)
    tl.store(scale_ptr, 1.0 / (total * bin_volume))


@libentry()
@triton.jit
def histogramdd_log_volume_kernel(
    width_ptr,  # (N,) per-dimension width
    bins_ptr,  # (N,) per-dimension bin count
    log_width_ptr,  # () out: sum of log|width / bins|
    sign_ptr,  # () out: product of the signs
    N,
    ACC_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce the per-dimension bin widths into a signed log bin volume.

    Accumulating ``log|width / bins|`` and tracking the sign separately keeps
    a high-dimensional product from overflowing or underflowing before the
    division in ``histogramdd_total_scale_kernel``.
    """
    offs = tl.arange(0, BLOCK_N)
    m = offs < N
    width = tl.load(width_ptr + offs, mask=m, other=1.0).to(ACC_DTYPE)
    nbins = tl.load(bins_ptr + offs, mask=m, other=1).to(ACC_DTYPE)
    per_dim = width / nbins
    # Masked lanes contribute log(1) == 0 to the sum and +1 to the sign
    # product.
    mag = tl.where(m, tl.abs(per_dim), 1.0)
    tl.store(log_width_ptr, tl.sum(tl.log(mag), axis=0))
    negatives = tl.sum(tl.where(m & (per_dim < 0), 1, 0), axis=0)
    tl.store(sign_ptr, tl.where(negatives % 2 == 0, 1.0, -1.0).to(ACC_DTYPE))


def _format_bound(value):
    """Format a range bound the way ATen's range messages do.

    ATen uses the default C++ stream formatting, which is ``%g`` (six
    significant digits, scientific notation outside a fixed exponent
    window), so ``0.123456789`` prints as ``0.123457`` and ``1e7`` as
    ``1e+07``.
    """
    return "%g" % value


def _validate_and_widen_range(range_, N):
    """Validate an explicit range like ATen, widening zero-width dimensions.

    Returns the post-widening float list. ATen validates every dimension's
    bounds in this order: first that they are finite, then that min <= max;
    an explicit non-finite bound is rejected too, not just a non-finite
    auto-range. ATen then widens a zero-width dimension before building the
    edges ("Expand empty range to match numpy behavior and avoid division by
    0 in normalization"), so an explicit [1.0, 1.0] behaves exactly like the
    degenerate *auto*-range case: the edges become linspace(l - .5, r + .5,
    bins + 1) and a point at 1.0 lands in bin bins // 2 rather than being
    folded into the last bin.
    """
    rng = [float(v) for v in range_]
    if len(rng) != 2 * N:
        raise RuntimeError(
            f"torch.histogramdd: for a {N}-dimensional histogram range should "
            f"have {2 * N} elements, but got {len(rng)}"
        )
    for d in range(N):
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
    for d in range(N):
        if rng[2 * d] == rng[2 * d + 1]:
            rng[2 * d] -= 0.5
            rng[2 * d + 1] += 0.5
    return rng


def _get_bins_meta(device, bins_list):
    """Per-(device, bins) metadata: strides, bins tensor, dummy weight slot.

    Derived purely from the bins list (host data), so the tensors are built
    once and cached; the ~30 us pageable H2D of a ``torch.tensor`` call was
    paid four times per call before this cache existed.
    """
    key = (device, tuple(bins_list))
    ent = _BINS_META_CACHE.get(key)
    if ent is None:
        N = len(bins_list)
        strides_host = [1] * N
        for d in _builtins.range(N - 2, -1, -1):
            strides_host[d] = strides_host[d + 1] * bins_list[d + 1]
        ent = _meta_cache_put(
            _BINS_META_CACHE,
            key,
            {
                "strides": torch.tensor(strides_host, dtype=torch.int32, device=device),
                "bins": torch.tensor(bins_list, dtype=torch.int32, device=device),
                # Triton needs a valid pointer slot even when HAS_WEIGHT
                # is False; the kernel never loads from it.
                "wdummy": torch.empty(1, dtype=torch.float32, device=device),
            },
        )
    return ent


def _get_range_meta(device, bins_list, rng, bins_meta):
    """Per-(device, bins, range) metadata: bounds, edges, density geometry.

    The edges are produced by the very same ``histogramdd_edges_kernel`` the
    per-call path used to launch, so the cached values are bit-identical to
    what a fresh call would build -- only the launch (and its ~30 us host
    cost) is skipped. The density geometry (widths / bin widths) is a pure
    function of the bounds and bins, so it is cached alongside.
    """
    key = (device, tuple(bins_list), tuple(rng))
    ent = _RANGE_META_CACHE.get(key)
    if ent is None:
        N = len(bins_list)
        max_edges = max(bins_list) + 1
        lefts = torch.tensor(rng[0::2], dtype=torch.float32, device=device)
        rights = torch.tensor(rng[1::2], dtype=torch.float32, device=device)
        edges = torch.empty(N, max_edges, dtype=torch.float32, device=device)
        widths = torch.empty(N, dtype=torch.float32, device=device)
        bin_widths = torch.empty(N, dtype=torch.float32, device=device)
        bins_t = bins_meta["bins"]
        BLOCK_N = triton.next_power_of_2(N)
        with torch_device_fn.device(device):
            histogramdd_edges_kernel[(N,)](
                lefts,
                rights,
                bins_t,
                edges,
                N,
                max_edges,
                ACC_DTYPE=tl.float32,
                BLOCK_E=min(1024, triton.next_power_of_2(max_edges)),
            )
            histogramdd_bin_geometry_kernel[(triton.cdiv(N, BLOCK_N),)](
                lefts,
                rights,
                bins_t,
                widths,
                bin_widths,
                N,
                ACC_DTYPE=tl.float32,
                BLOCK_N=BLOCK_N,
            )
        ent = _meta_cache_put(
            _RANGE_META_CACHE,
            key,
            {
                "lefts": lefts,
                "rights": rights,
                "edges": edges,
                "max_edges": max_edges,
                "widths": widths,
                "bin_widths": bin_widths,
            },
        )
    return ent


def _get_partials(device, rows, cols):
    """Cached int32 partial-count buffer of shape (rows, cols).

    Every element of every row is unconditionally stored by the counting
    kernels (the padding column included), so the buffer never needs
    zero-initialisation and a cached ``torch.empty`` is safe to reuse."""
    key = (device, rows, cols)
    ent = _PARTIALS_CACHE.get(key)
    if ent is None:
        ent = _meta_cache_put(
            _PARTIALS_CACHE,
            key,
            torch.empty(rows * cols, dtype=torch.int32, device=device),
        )
    return ent


def _auto_range(inp, N):
    """Derive the per-dimension range from the data, on the device.

    Matches torch.histogramdd: a dimension whose min equals max is expanded
    to (min - 0.5, max + 0.5). The range endpoints are float32 because
    GCU400 has no FP64 vector path (torch_gcu downgrades float64 tensors to
    float32 storage and fp64 kernels crash the compiler).
    """
    lefts = torch.empty(N, dtype=torch.float32, device=inp.device)
    rights = torch.empty(N, dtype=torch.float32, device=inp.device)
    M = inp.shape[0]
    if M == 0:
        # No data to reduce over; ATen's empty-input range collapses to (0, 1)
        # per dimension after the degenerate-range expansion of (inf, -inf) is
        # skipped.
        lefts.fill_(0.0)
        rights.fill_(1.0)
        return lefts, rights, None

    BLOCK_N = triton.next_power_of_2(N)
    grid = (triton.cdiv(N, BLOCK_N),)
    # Sentinel-initialised to N: the kernel only lowers it, so a finite input
    # leaves it at N and the check below reads one scalar instead of running
    # ``isfinite``/``nonzero`` over the whole range tensor on the host.
    nonfinite_dim = torch.full((), N, dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        histogramdd_col_range_kernel[grid](
            inp,
            lefts,
            rights,
            nonfinite_dim,
            M,
            N,
            ACC_DTYPE=tl.float32,
            BLOCK_M=128,
            BLOCK_N=BLOCK_N,
            SENTINEL=N,
        )
    return lefts, rights, nonfinite_dim


def _check_finite_auto_range(lefts, rights, nonfinite_dim, N):
    """Reject a non-finite auto-range, with ATen's message.

    The bound is printed for the offending dimension, so that dimension has
    to be known before raising; ``histogramdd_col_range_kernel`` reports it
    in ``nonfinite_dim`` (sentinel ``N`` meaning "all finite") so this stays
    a single scalar read. ``%g`` reproduces ATen's formatting.
    """
    dim = int(nonfinite_dim.item())
    if dim >= N:
        return
    raise RuntimeError(
        f"torch.histogramdd: dimension {dim}'s range "
        f"[{_format_bound(lefts[dim].item())}, {_format_bound(rights[dim].item())}] "
        "is not finite"
    )


def _histogramdd_from_bin_cts_impl(
    self, bins, *, range=None, weight=None, density=False, out=None
):
    """Shared implementation for the base and ``.out`` ATen variants."""
    logger.debug("GEMS_ENFLAME _HISTOGRAMDD_FROM_BIN_CTS")

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

    N = self.shape[-1]
    bins_list = [int(b) for b in bins]
    out_shape = tuple(bins_list)

    out_dtype = self.dtype

    if self.dtype == torch.float64:
        # gcu400 has no fp64 arithmetic. Bin placement compares the coordinate
        # against the edges exactly, so a coordinate one ulp away from an edge
        # cannot be classified in fp32 (and the fp64 kernels do not compile).
        # Delegate double precision to the native CPU implementation and copy
        # the result back.
        cpu = torch.ops.aten._histogramdd_from_bin_cts(
            self.cpu(),
            bins_list,
            range=None if range is None else [float(v) for v in range],
            weight=None if weight is None else weight.cpu(),
            density=density,
        )
        if out is None:
            return cpu.to(self.device)
        if out.dtype != out_dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {out_dtype}, but got "
                f"{out.dtype} instead"
            )
        if tuple(out.shape) != tuple(cpu.shape):
            out.resize_(cpu.shape)
        out.copy_(cpu)
        return out

    # GCU400 has no FP64 vector path: torch_gcu downgrades float64 tensors to
    # float32 storage, and fp64 Triton kernels crash the GCU compiler.
    # Normalise the compute input to real float32 while out keeps the
    # caller's dtype.
    compute_input = self.to(torch.float32) if out_dtype != torch.float32 else self

    # Flatten all leading dims to (M, N) and make contiguous.
    inp = compute_input.reshape(-1, N).contiguous()
    M = inp.shape[0]

    # Allocate / validate the output buffer. ATen resizes a mis-shaped out
    # tensor in place and keeps the caller's object identity, but rejects a
    # dtype mismatch outright, so a wrong dtype must raise rather than be
    # re-bound through .to() (which would leave the caller's buffer
    # untouched).
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

    bins_meta = _get_bins_meta(inp.device, bins_list)
    strides = bins_meta["strides"]
    bins_t = bins_meta["bins"]

    if range is not None:
        rng = _validate_and_widen_range(range, N)
        range_meta = _get_range_meta(inp.device, bins_list, rng, bins_meta)
        lefts = range_meta["lefts"]
        rights = range_meta["rights"]
        edges = range_meta["edges"]
        max_edges = range_meta["max_edges"]
        widths = range_meta["widths"]
        bin_widths = range_meta["bin_widths"]
    else:
        # Auto range: the bounds depend on the data, so nothing but the bins
        # metadata can be reused across calls.
        lefts, rights, nonfinite_dim = _auto_range(inp, N)
        if nonfinite_dim is not None:
            _check_finite_auto_range(lefts, rights, nonfinite_dim, N)
        max_edges = max(bins_list) + 1
        edges = torch.empty(N, max_edges, dtype=torch.float32, device=inp.device)
        widths = torch.empty(N, dtype=torch.float32, device=inp.device)
        bin_widths = torch.empty(N, dtype=torch.float32, device=inp.device)
        BLOCK_N = triton.next_power_of_2(N)
        with torch_device_fn.device(self.device):
            histogramdd_edges_kernel[(N,)](
                lefts,
                rights,
                bins_t,
                edges,
                N,
                max_edges,
                ACC_DTYPE=tl.float32,
                BLOCK_E=min(1024, triton.next_power_of_2(max_edges)),
            )
            histogramdd_bin_geometry_kernel[(triton.cdiv(N, BLOCK_N),)](
                lefts,
                rights,
                bins_t,
                widths,
                bin_widths,
                N,
                ACC_DTYPE=tl.float32,
                BLOCK_N=BLOCK_N,
            )

    num_bins = 1
    for b in bins_list:
        num_bins *= b
    steps_const = max(1, (max_edges - 1).bit_length())

    # Counts land in a real float32 buffer. A contiguous float32 out can
    # serve as that buffer directly (the reduce kernels overwrite every
    # bin, so no zero-fill is needed); other dtypes and non-contiguous outs
    # need a temporary plus one copy to keep the .out contract intact.
    hist_is_out = not density and out.dtype == torch.float32 and out.is_contiguous()
    if hist_is_out:
        hist_target = out.view(num_bins)
    else:
        hist_target = torch.empty(num_bins, dtype=torch.float32, device=inp.device)

    has_weight = weight is not None
    if has_weight:
        weight = weight.reshape(-1).contiguous()
        if weight.dtype != torch.float32:
            weight = weight.to(torch.float32)
        if weight.numel() != M:
            raise RuntimeError(
                "_histogramdd_from_bin_cts: weight must have the same number "
                "of rows as the input"
            )

    if M == 0:
        # No points: the histogram is all zeros, whatever the dispatch
        # path would have been.
        hist_target.zero_()
    else:
        with torch_device_fn.device(self.device):
            if has_weight:
                # Weighted calls scatter-add one contribution per point;
                # per-program private slab rows keep the same-address
                # atomic collision depth bounded by the tile size instead
                # of the grid size.
                grid_n = triton.cdiv(M, _SLAB_BLOCK)
                if grid_n * num_bins * 4 <= _MAX_SLAB_BYTES:
                    slab = torch.zeros(
                        grid_n * num_bins, dtype=torch.float32, device=inp.device
                    )
                    histogramdd_weighted_slab_kernel[(grid_n,)](
                        inp,
                        weight,
                        slab,
                        M,
                        N,
                        lefts,
                        rights,
                        edges,
                        strides,
                        bins_t,
                        max_edges,
                        MAX_EDGE_STEPS=steps_const,
                        N_CONST=N,
                        BLOCK_SIZE=_SLAB_BLOCK,
                        NUM_BINS=num_bins,
                    )
                    histogramdd_reduce_slab_kernel[(triton.cdiv(num_bins, 1024),)](
                        slab,
                        hist_target,
                        grid_n,
                        NUM_BINS=num_bins,
                        BLOCK_B=1024,
                    )
                else:
                    # Pathological point count x bin width: a private slab
                    # would exceed the scratch budget, so pay the shared
                    # same-address atomics instead.
                    hist_target.zero_()
                    histogramdd_direct_kernel[(grid_n,)](
                        inp,
                        weight,
                        hist_target,
                        M,
                        N,
                        lefts,
                        rights,
                        edges,
                        strides,
                        bins_t,
                        max_edges,
                        MAX_EDGE_STEPS=steps_const,
                        N_CONST=N,
                        BLOCK_SIZE=_SLAB_BLOCK,
                        HAS_WEIGHT=True,
                    )
            elif num_bins <= 512:
                # Narrow bin grid: count with per-program partials (no
                # global atomics at all) and one reduce kernel.
                hist_bins = triton.next_power_of_2(num_bins + 1)
                if hist_bins <= _DOT_PERSISTENT_MAX_BINS:
                    # One-hot int8 tl.dot on a persistent grid; a single
                    # chunk covers the whole padded bin vector.
                    grid_n = min(triton.cdiv(M, _DOT_BLOCK), MAX_CTAS)
                    partials = _get_partials(inp.device, grid_n, hist_bins)
                    histogramdd_dot_persistent_kernel[(grid_n,)](
                        inp,
                        partials,
                        M,
                        N,
                        lefts,
                        rights,
                        edges,
                        strides,
                        bins_t,
                        max_edges,
                        MAX_EDGE_STEPS=steps_const,
                        N_CONST=N,
                        BLOCK_SIZE=_DOT_BLOCK,
                        NUM_BINS=num_bins,
                        HIST_BINS=hist_bins,
                        NB_CHUNK=max(hist_bins, 16),
                    )
                elif M < _DOT_MAX_POINTS:
                    # Wider bin vector but a small point cloud: chunked
                    # one-hot dot, one program per tile.
                    grid_n = triton.cdiv(M, _DOT_BLOCK)
                    partials = _get_partials(inp.device, grid_n, hist_bins)
                    histogramdd_dot_kernel[(grid_n,)](
                        inp,
                        partials,
                        M,
                        N,
                        lefts,
                        rights,
                        edges,
                        strides,
                        bins_t,
                        max_edges,
                        MAX_EDGE_STEPS=steps_const,
                        N_CONST=N,
                        BLOCK_SIZE=_DOT_BLOCK,
                        NUM_BINS=num_bins,
                        HIST_BINS=hist_bins,
                        NB_CHUNK=128,
                    )
                else:
                    # Large point cloud over a wide bin vector: the per-point
                    # one-hot compare cost overtakes the histogram
                    # primitive's per-bin cost.
                    grid_n = triton.cdiv(M, _HIST_BLOCK)
                    partials = _get_partials(inp.device, grid_n, hist_bins)
                    histogramdd_hist_partial_kernel[(grid_n,)](
                        inp,
                        partials,
                        M,
                        N,
                        lefts,
                        rights,
                        edges,
                        strides,
                        bins_t,
                        max_edges,
                        MAX_EDGE_STEPS=steps_const,
                        N_CONST=N,
                        BLOCK_SIZE=_HIST_BLOCK,
                        HIST_BINS=hist_bins,
                    )
                histogramdd_reduce_i32_kernel[(triton.cdiv(num_bins, 512),)](
                    partials,
                    hist_target,
                    grid_n,
                    HIST_BINS=hist_bins,
                    NUM_BINS=num_bins,
                    BLOCK_B=512,
                )
            else:
                # Wide bin grid: per-point atomics into the zeroed
                # accumulator. Per-address collisions stay rare (points
                # spread over many bins), and no per-program aggregation
                # beats the atomic unit's throughput at this width.
                block = 256 if M <= 8192 else 512
                grid_n = triton.cdiv(M, block)
                hist_target.zero_()
                histogramdd_direct_kernel[(grid_n,)](
                    inp,
                    bins_meta["wdummy"],
                    hist_target,
                    M,
                    N,
                    lefts,
                    rights,
                    edges,
                    strides,
                    bins_t,
                    max_edges,
                    MAX_EDGE_STEPS=steps_const,
                    N_CONST=N,
                    BLOCK_SIZE=block,
                    HAS_WEIGHT=False,
                )

    if density:
        # density[i] = count[i] / (total_count * bin_volume), where
        # bin_volume is the volume of a single (equal-width) N-dimensional
        # bin, i.e. the product of the per-dimension bin widths =
        # prod((right - left) / bins). Both reductions run on the device;
        # the scale kernel reads the counts buffer directly.
        log_vol = torch.empty((), dtype=torch.float32, device=inp.device)
        vol_sign = torch.empty((), dtype=torch.float32, device=inp.device)
        scale_t = torch.empty((), dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(self.device):
            histogramdd_log_volume_kernel[(1,)](
                widths,
                bins_t,
                log_vol,
                vol_sign,
                N,
                ACC_DTYPE=tl.float32,
                BLOCK_N=triton.next_power_of_2(N),
            )
            # The division is deliberately unguarded: ATen divides by the
            # total count directly, so an empty or fully out-of-range input
            # yields NaN and weights that cancel to zero yield +-inf. Forcing
            # the scale to zero here would silently diverge from that.
            histogramdd_total_scale_kernel[(1,)](
                hist_target,
                log_vol,
                vol_sign,
                scale_t,
                num_bins,
                ACC_DTYPE=tl.float32,
                BLOCK_SIZE=1024,
            )
        # Density values are produced in a flat fp32 buffer and copied into
        # the caller's out at the end, so fp16 / non-contiguous outs stay
        # correct.
        out_flat = torch.empty(num_bins, dtype=torch.float32, device=inp.device)
        if num_bins > 0:
            BLOCK = 1024
            with torch_device_fn.device(self.device):
                histogramdd_density_kernel[(triton.cdiv(num_bins, BLOCK),)](
                    hist_target,
                    out_flat,
                    scale_t,
                    num_bins,
                    ACC_DTYPE=tl.float32,
                    BLOCK_SIZE=BLOCK,
                )
        out.copy_(out_flat.view(out_shape))
    elif not hist_is_out:
        out.copy_(hist_target.view(out_shape))

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
