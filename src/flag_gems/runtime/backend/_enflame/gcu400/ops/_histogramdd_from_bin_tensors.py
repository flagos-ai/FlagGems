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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 has 24 SIPs; at most 48 CTAs run concurrently, so grids above that
# only add wave-quantisation and per-program setup cost.
NUM_SIPS = 24
MAX_CTAS = NUM_SIPS * 2

# Host-side metadata caches. A ``torch.tensor`` copy to the device costs ~30 us
# on a pageable transfer, and the per-call host path used to pay one geometry
# upload plus a ``torch.cat`` of the edge tensors (another kernel) on every
# invocation. The packed edges and the geometry depend only on the bins
# configuration, so they are built once and reused; repeated calls with the
# same bins tensors -- the dominant pattern in serving loops and benchmarks --
# skip the transfers entirely. Entries pin strong references to the source
# bins tensors so the (data_ptr, _version) key cannot collide with a
# different live tensor reusing a freed allocation.
_META_CACHE_CAP = 64
_BINS_META_CACHE = {}
_PARTIALS_CACHE = {}
_SLAB_CACHE = {}
_SCALAR_CACHE = {}

# Above this size a weighted call falls back from the per-program private
# slab (one row of ``num_bins`` per program) to direct atomics, so a huge
# point cloud over a wide bin grid cannot force a multi-GB scratch tensor.
_MAX_SLAB_BYTES = 64 * 1024 * 1024

# Dispatch thresholds, measured on GCU400 (probe data in the module history /
# case log; the one-hot dot lanes cost ~1 ps each while atomics serialise at
# ~100 ns per lane, so the dot wins until the padded bin vector gets wide):
#   * padded bins <= 128: persistent-grid dot, one chunk, no atomics at all;
#   * padded bins <= 2048 and M < 65536: chunked dot over 128-wide bin slabs;
#   * wider bin vectors: per-point atomics (collision depth stays low once the
#     grid has thousands of cells).
_DOT_PERSISTENT_MAX_BINS = 128
_DOT_MAX_BINS = 2048
_DOT_CHUNK_MAX_POINTS = (
    8192  # above this the slab's atomic floor beats the per-chunk dot cost
)
_DOT_BLOCK = 512
_WDOT_BLOCK = 256  # the f32 one-hot tile is 4x the int8 one; halve the rows
_DOT_CHUNK_BLOCK = 256  # 256 x 256 int8 one-hot tile = 64 KB DSM budget
_WDOT_CHUNK_BLOCK = 256  # 256 x 128 f32 one-hot tile = 128 KB DSM budget
_DOT_NB_CHUNK = 256
_WDOT_NB_CHUNK = 128  # probed: 256-wide f32 chunks at BLOCK=128 regress 58->71 us
_DIRECT_BLOCK_SMALL = 256
_SLAB_BLOCK = 512


def _meta_cache_put(cache, key, value):
    if len(cache) >= _META_CACHE_CAP:
        cache.clear()
    cache[key] = value
    return value


@triton.jit
def _histdd_pack_edges_kernel(
    src_ptr,  # per-dimension edge tensor (any float dtype, contiguous)
    dst_ptr,  # flat f32 packed edges
    off,  # start offset of this dimension inside dst
    count,  # number of edges in this dimension
    BLOCK: tl.constexpr,
):
    """Copy one dimension's edges into the flat f32 pack, casting on the fly.

    fp16/bf16 -> fp32 is exact, so bin placement compares the same values the
    (fp32-widened) reference compares.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < count
    v = tl.load(src_ptr + idx, mask=m, other=0.0).to(tl.float32)
    tl.store(dst_ptr + off + idx, v, mask=m)


@triton.jit
def _histdd_locate(
    inp_ptr,  # (M, N) contiguous input, any float dtype
    row_start,
    row_mask,
    edges_ptr,  # flat f32 concatenated per-dimension edges
    geom_ptr,  # i32 (3, N): edge offsets / bins per dim / row-major strides
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Locate each row's flat bin index, or mark the row invalid.

    Shared prologue of every counting kernel. For every dimension the
    coordinate is range-checked against the first/top edge and the bin is
    found by binary search over that dimension's slice of the packed edge
    array: ``lo`` converges to the last edge index ``e`` with
    ``edges[e] <= coord``, which is exactly the bin index for the
    left-inclusive / rightmost-right-inclusive ATen rule (a point on the top
    edge searches to ``nbins`` and folds back into the last bin). NaN
    coordinates fail both range comparisons, like the reference.

    Probe note: a uniform "slab-compare" form (broadcast the whole edge
    vector, count ``edges <= coord``) was tried for narrow edge vectors and
    gave no measurable win in the dot kernels (17.99 vs 18.82 us, within
    noise) while mis-compiling to 5x slowdowns in the atomic kernels at
    BLOCK=256, so the search form is the one kept everywhere.
    """
    valid = row_mask
    # num_bins is the product of small per-dimension bin counts, so a 32-bit
    # linear index is sufficient and avoids GCU's slow int64 path.
    linear_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for d in tl.static_range(N_CONST):
        coord = tl.load(
            inp_ptr + row_start * N_CONST + d, mask=row_mask, other=float("nan")
        ).to(tl.float32)
        nbins = tl.load(geom_ptr + N_CONST + d)
        stride = tl.load(geom_ptr + 2 * N_CONST + d)
        off = tl.load(geom_ptr + d)
        row = edges_ptr + off

        first = tl.load(row).to(tl.float32)
        top = tl.load(row + nbins).to(tl.float32)
        in_range = (coord >= first) & (coord <= top)
        valid = valid & in_range

        lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        hi = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32) + nbins
        # Binary search for the last edge <= coord; the trip count is a
        # compile-time bound sized for the widest dimension (extra steps are
        # no-ops once the interval collapses: `mid <= nbins` gates the top
        # and a converged probe never moves lo).
        for _ in tl.static_range(MAX_EDGE_STEPS):
            mid = (lo + hi + 1) // 2
            mid_c = tl.minimum(mid, nbins)
            edge = tl.load(row + mid_c, mask=valid, other=float("inf")).to(tl.float32)
            take = (edge <= coord) & (mid <= nbins)
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        bin_idx = tl.where(lo >= nbins, nbins - 1, lo)
        linear_idx = linear_idx + bin_idx * stride

    return linear_idx, valid


@libentry()
@triton.jit
def histogramdd_dot_persistent_kernel(
    inp_ptr,  # (M, N) contiguous input
    part_ptr,  # (grid, HIST_BINS + 1) int32 per-program partials + total
    M,
    edges_ptr,  # flat f32 concatenated edges
    geom_ptr,  # i32 (3, N) geometry
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    HIST_BINS: tl.constexpr,
):
    """Count points with a one-hot int8 ``tl.dot`` on a persistent grid.

    ``count[b] = sum_e (key_e == b)`` lowers to
    ``ones[16, BLOCK] @ onehot[BLOCK, NBINS]`` on the MMA path, which is far
    cheaper than either the serial ``tl.histogram`` update or per-point
    atomics for narrow bin vectors. The grid is capped at the number of
    resident CTAs and each program strides over point blocks, accumulating
    its partial histogram across iterations; one partial row per program is
    then summed by the reduce kernel. No global atomics are issued at all.
    The row's last column carries the program's valid-point total so the
    density normalisation needs no extra launch.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    ones = tl.full((16, BLOCK_SIZE), 1, dtype=tl.int8)
    acc = tl.zeros((16, HIST_BINS), dtype=tl.int32)
    total = tl.zeros((), dtype=tl.int32)
    nblocks = tl.cdiv(M, BLOCK_SIZE)
    for blk in tl.range(pid, nblocks, num_pids):
        row_start = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = row_start < M
        linear_idx, valid = _histdd_locate(
            inp_ptr,
            row_start,
            row_mask,
            edges_ptr,
            geom_ptr,
            MAX_EDGE_STEPS,
            N_CONST,
            BLOCK_SIZE,
        )
        # Invalid lanes are routed into the padding bin and dropped by the
        # store mask below (tl.histogram's mask is ignored by the GCU
        # backend; the same convention keeps the dot path uniform).
        key = tl.where(valid, linear_idx, HIST_BINS - 1)
        cols = tl.arange(0, HIST_BINS)
        onehot = tl.where(
            key[:, None] == cols[None, :],
            tl.full((BLOCK_SIZE, HIST_BINS), 1, tl.int8),
            tl.zeros((BLOCK_SIZE, HIST_BINS), dtype=tl.int8),
        )
        acc = tl.dot(ones, onehot, acc, out_dtype=tl.int32)
        total += tl.sum(valid.to(tl.int32), axis=0)
    # All 16 rows of `ones` are identical, so any row of `acc` holds the
    # counts; `tl.max` extracts one of them as a flat vector (counts are
    # non-negative).
    counts = tl.max(acc, axis=0)
    out_cols = tl.arange(0, HIST_BINS)
    tl.store(part_ptr + pid * (HIST_BINS + 1) + out_cols, counts)
    tl.store(part_ptr + pid * (HIST_BINS + 1) + HIST_BINS, total)


@libentry()
@triton.jit
def histogramdd_dot_kernel(
    inp_ptr,  # (M, N) contiguous input
    part_ptr,  # (grid, HIST_BINS + 1) int32 per-program partials + total
    M,
    edges_ptr,  # flat f32 concatenated edges
    geom_ptr,  # i32 (3, N) geometry
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    HIST_BINS: tl.constexpr,
    NB_CHUNK: tl.constexpr,
):
    """Chunked one-hot int8 ``tl.dot`` counting, one program per point tile.

    Used when the padded bin vector is wider than a single chunk fits in the
    DSM budget (BLOCK x CHUNK int8 <= 64 KB); the bin axis is swept in
    NB_CHUNK-wide slabs, each producing its own ``tl.dot``. Partials are
    stored per program and reduced by ``histogramdd_reduce_kernel``. The
    sweep is a dynamic loop on purpose: a static unroll keeps every chunk's
    one-hot tile live at once, which the GCU register allocator turns into
    spills (measured 86 us for an 8-chunk sweep before this change).
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        edges_ptr,
        geom_ptr,
        MAX_EDGE_STEPS,
        N_CONST,
        BLOCK_SIZE,
    )
    key = tl.where(valid, linear_idx, HIST_BINS - 1)
    ones = tl.full((16, BLOCK_SIZE), 1, dtype=tl.int8)
    for cbase in range(0, HIST_BINS, NB_CHUNK):
        cols = cbase + tl.arange(0, NB_CHUNK)
        onehot = tl.where(
            key[:, None] == cols[None, :],
            tl.full((BLOCK_SIZE, NB_CHUNK), 1, tl.int8),
            tl.zeros((BLOCK_SIZE, NB_CHUNK), dtype=tl.int8),
        )
        acc = tl.dot(ones, onehot, out_dtype=tl.int32)
        counts = tl.max(acc, axis=0)
        tl.store(
            part_ptr + pid * (HIST_BINS + 1) + cols,
            counts,
            mask=cols < HIST_BINS,
        )
    tl.store(
        part_ptr + pid * (HIST_BINS + 1) + HIST_BINS,
        tl.sum(valid.to(tl.int32), axis=0),
    )


@libentry()
@triton.jit
def histogramdd_wdot_kernel(
    inp_ptr,  # (M, N) contiguous input
    weight_ptr,  # (M,) contiguous weights (any float dtype)
    part_ptr,  # (grid, HIST_BINS + 1) f32 per-program partials + total
    M,
    edges_ptr,  # flat f32 concatenated edges
    geom_ptr,  # i32 (3, N) geometry
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    HIST_BINS: tl.constexpr,
    NB_CHUNK: tl.constexpr,
):
    """Chunked f32 one-hot ``tl.dot`` weighted counting, one program per tile."""
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        edges_ptr,
        geom_ptr,
        MAX_EDGE_STEPS,
        N_CONST,
        BLOCK_SIZE,
    )
    w = tl.load(weight_ptr + row_start, mask=row_mask, other=0.0).to(tl.float32)
    key = tl.where(valid, linear_idx, HIST_BINS - 1)
    wb = tl.broadcast_to(tl.reshape(w, (1, BLOCK_SIZE)), (16, BLOCK_SIZE))
    for cbase in range(0, HIST_BINS, NB_CHUNK):
        cols = cbase + tl.arange(0, NB_CHUNK)
        onehot = tl.where(
            key[:, None] == cols[None, :],
            tl.full((BLOCK_SIZE, NB_CHUNK), 1.0, tl.float32),
            tl.zeros((BLOCK_SIZE, NB_CHUNK), dtype=tl.float32),
        )
        acc = tl.dot(wb, onehot, out_dtype=tl.float32)
        counts = tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, acc, 0.0), axis=0)
        tl.store(
            part_ptr + pid * (HIST_BINS + 1) + cols,
            counts,
            mask=cols < HIST_BINS,
        )
    tl.store(
        part_ptr + pid * (HIST_BINS + 1) + HIST_BINS,
        tl.sum(tl.where(valid, w, 0.0), axis=0),
    )


@libentry()
@triton.jit
def histogramdd_direct_kernel(
    inp_ptr,  # (M, N) contiguous input
    weight_ptr,  # (M,) weights, or a dummy when HAS_WEIGHT=False
    out_ptr,  # (num_bins,) zero-initialised accumulator
    M,
    edges_ptr,  # flat f32 concatenated edges
    geom_ptr,  # i32 (3, N) geometry
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """Per-point atomic accumulation into one shared histogram buffer.

    Kept for bin grids too wide for any per-program aggregation to pay off:
    the one-hot compare cost grows with the bin count while the atomic cost
    does not, and thousands of cells keep the per-address collision count
    low. The buffer must be zeroed before launch.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        edges_ptr,
        geom_ptr,
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
def histogramdd_slab_kernel(
    inp_ptr,  # (M, N) contiguous input
    weight_ptr,  # (M,) contiguous weights (any float dtype)
    slab_ptr,  # (grid, num_bins) zero-initialised private rows
    M,
    edges_ptr,  # flat f32 concatenated edges
    geom_ptr,  # i32 (3, N) geometry
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """Per-point accumulation into per-program private slab rows.

    A shared buffer serialises same-address atomics across all programs --
    for Gaussian data over a narrow-ish bin grid the hottest cell chains
    thousands of collisions (measured 513 us for 16K points over 1000 bins
    through the direct kernel, vs 70 us here). A private row per program caps
    the collision depth at the tile level; the reduce kernel then sums the
    rows. Weights scatter their contribution; the unweighted form
    (HAS_WEIGHT=False) counts 1.0 in f32, exact up to 2^24 points per
    program. The slab is freshly zeroed per call.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_start < M
    linear_idx, valid = _histdd_locate(
        inp_ptr,
        row_start,
        row_mask,
        edges_ptr,
        geom_ptr,
        MAX_EDGE_STEPS,
        N_CONST,
        BLOCK_SIZE,
    )
    if HAS_WEIGHT:
        contrib = tl.load(weight_ptr + row_start, mask=valid, other=0.0).to(tl.float32)
    else:
        contrib = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    tl.atomic_add(
        slab_ptr + pid * NUM_BINS + linear_idx, contrib, mask=valid, sem="relaxed"
    )


@libentry()
@triton.jit
def histogramdd_reduce_kernel(
    part_ptr,  # (n_prog, W) partials; W = HIST_BINS + 1 (dot) or num_bins (slab)
    out_ptr,  # (num_bins,) output, fully overwritten
    total_ptr,  # () valid-point total; written when HAS_TOTAL
    n_prog,
    NUM_BINS: tl.constexpr,
    W: tl.constexpr,
    HAS_TOTAL: tl.constexpr,
    PART_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Sum per-program partial rows into the output histogram.

    Writes every bin (including untouched ones, as zeros), so the output
    needs no zero-initialisation and the reduce doubles as the full
    overwrite required by the ``.out`` contract. Accumulation stays in the
    partials' native dtype (int32 counts sum exactly; fp32 partials keep the
    weighted path's precision) and is cast once at the store. When the rows
    carry a valid-point total in their last column (the dot kernels always
    store one), the last program also publishes it, which keeps the density
    path at three launches.
    """
    pid = tl.program_id(0)
    bins = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    m = bins < NUM_BINS
    acc = tl.zeros((BLOCK_B,), dtype=PART_DTYPE)
    for p in range(0, n_prog):
        acc += tl.load(part_ptr + p * W + bins, mask=m, other=0)
    tl.store(out_ptr + bins, acc.to(OUT_DTYPE), mask=m)
    if HAS_TOTAL:
        if pid == tl.num_programs(0) - 1:
            tot = tl.zeros((), dtype=PART_DTYPE)
            for p in range(0, n_prog):
                tot += tl.load(part_ptr + p * W + (W - 1))
            tl.store(total_ptr, tot.to(tl.float32))


@libentry()
@triton.jit
def histogramdd_total_kernel(
    hist_ptr,  # (num_bins,) counts
    total_ptr,  # () out
    num_bins,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum the histogram on device for the atomic paths (no partials row)."""
    total = tl.zeros((), dtype=tl.float32)
    for base in range(0, num_bins, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        m = offs < num_bins
        total += tl.sum(tl.load(hist_ptr + offs, mask=m, other=0.0).to(tl.float32))
    tl.store(total_ptr, total)


@libentry()
@triton.jit
def histogramdd_density_kernel(
    hist_ptr,  # (num_bins,) counts in f32
    out_ptr,  # (num_bins,) density output (input dtype)
    total_ptr,  # () valid-point total
    edges_ptr,  # flat f32 concatenated edges
    geom_ptr,  # i32 (3, N) geometry
    num_bins,
    N_CONST: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """density[i] = hist[i] / total / volume(i), with no guard on the total.

    ATen divides through unconditionally, so an empty or fully out-of-range
    input yields NaN, weights that cancel yield +-inf, and a net-negative
    total keeps the sign of each bin. The per-bin volume is computed inline
    (the bins are non-uniform for this op, so unlike the equal-width
    from_bin_cts sibling it cannot be folded into one scalar): each flat bin
    index is decomposed back into per-dimension indices with the row-major
    strides and the adjacent edge widths are multiplied.
    """
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < num_bins
    h = tl.load(hist_ptr + offs, mask=m, other=0.0).to(tl.float32)
    total = tl.load(total_ptr).to(tl.float32)
    vol = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    for d in tl.static_range(N_CONST):
        stride = tl.load(geom_ptr + 2 * N_CONST + d)
        size = tl.load(geom_ptr + N_CONST + d)
        off = tl.load(geom_ptr + d)
        idx_d = (offs // stride) % size
        lo = tl.load(edges_ptr + off + idx_d, mask=m, other=0.0).to(tl.float32)
        hi = tl.load(edges_ptr + off + idx_d + 1, mask=m, other=1.0).to(tl.float32)
        vol = vol * (hi - lo)
    res = h / total / vol
    tl.store(out_ptr + offs, res.to(OUT_DTYPE), mask=m)


def _validate(self, bins, weight):
    """Validate inputs with ATen's error types and wording.

    Uses exceptions rather than ``assert`` so the checks survive ``python -O``
    and so callers see the same messages the native op produces.
    """
    D = len(bins)
    if self.ndim < 2:
        raise RuntimeError(
            "torch.histogramdd: input tensor should have at least 2 dimensions"
        )
    if self.shape[-1] != D:
        raise RuntimeError(
            f"torch.histogramdd: expected {self.shape[-1]} sequences of bin edges "
            f"for a {self.shape[-1]}-dimensional histogram but got {D}"
        )
    if not self.is_floating_point():
        raise RuntimeError(
            "torch.histogramdd: input tensor and bins tensors should have the same "
            "dtype, but got input with dtype " + str(self.dtype)
        )
    for d, b in enumerate(bins):
        if b.dtype != self.dtype:
            raise RuntimeError(
                "torch.histogramdd: input tensor and bins tensors should have the "
                f"same dtype, but got input {self.dtype} and bins[{d}] {b.dtype}"
            )
        if b.numel() < 2:
            raise RuntimeError(
                f"torch.histogram(): bins must be > 0, but got {b.numel() - 1} "
                f"for dimension {d}"
            )
    if weight is not None:
        # ATen requires an exact dtype match and rejects anything else; the
        # previous ``pass`` branch silently accepted a mismatched weight.
        if weight.dtype != self.dtype:
            raise RuntimeError(
                "torch.histogramdd: if weight tensor is provided, input tensor and "
                "weight tensor should have the same dtype, but got "
                f"input({self.dtype}), and weight({weight.dtype})"
            )
        if weight.numel() != self.numel() // self.shape[-1]:
            raise RuntimeError(
                "torch.histogramdd: if weight tensor is provided it should have "
                "the same shape as the input tensor excluding its innermost "
                "dimension"
            )


def _get_bins_meta(device, dtype, bins):
    """Per-(device, dtype, bins-tensors) metadata: packed edges + geometry.

    The packed f32 edge array and the i32 geometry (edge offsets / bins per
    dim / row-major strides) depend only on the bins configuration, so they
    are built once and reused: a ``torch.tensor`` pageable H2D costs ~30 us
    and the pack kernels a few launches, none of which should repeat per
    call. The key carries (data_ptr, _version, numel) per dimension and the
    entry strong-references the source tensors, so an in-place mutation of
    any edge tensor bumps its version counter and forces a repack, while a
    freed-and-reused allocation cannot alias a live entry.
    """
    key = (device, dtype, tuple((b.data_ptr(), b._version, b.numel()) for b in bins))
    ent = _BINS_META_CACHE.get(key)
    if ent is None:
        D = len(bins)
        bin_counts = [b.numel() for b in bins]
        nbins = [bc - 1 for bc in bin_counts]
        total_edges = sum(bin_counts)
        offs = [0] * D
        for d in range(1, D):
            offs[d] = offs[d - 1] + bin_counts[d - 1]
        edges = torch.empty(total_edges, dtype=torch.float32, device=device)
        with torch_device_fn.device(device):
            for d, b in enumerate(bins):
                src = b.detach().reshape(-1)
                if not src.is_contiguous():
                    src = src.contiguous()
                _histdd_pack_edges_kernel[(triton.cdiv(bin_counts[d], 1024),)](
                    src, edges, offs[d], bin_counts[d], BLOCK=1024
                )
        strides = [1] * D
        for d in range(D - 2, -1, -1):
            strides[d] = strides[d + 1] * nbins[d + 1]
        geom = torch.tensor([offs, nbins, strides], dtype=torch.int32, device=device)
        num_bins = 1
        for b in nbins:
            num_bins *= b
        ent = _meta_cache_put(
            _BINS_META_CACHE,
            key,
            {
                "edges": edges,
                "geom": geom,
                "nbins": nbins,
                "num_bins": num_bins,
                "steps": max(1, max(nbins).bit_length()),
                "bins_ref": list(bins),
            },
        )
    return ent


def _get_partials(device, rows, cols, dtype):
    """Cached partial-count buffer of shape (rows, cols).

    Every element of every row is unconditionally stored by the counting
    kernels (the padding bin and the total column included), so the buffer
    never needs zero-initialisation and a cached ``torch.empty`` is safe to
    reuse.
    """
    key = (device, rows, cols, dtype)
    ent = _PARTIALS_CACHE.get(key)
    if ent is None:
        ent = _meta_cache_put(
            _PARTIALS_CACHE,
            key,
            torch.empty(rows * cols, dtype=dtype, device=device),
        )
    return ent


def _get_slab(device, rows, cols):
    """Cached weighted-slab buffer; zeroed by the caller before each launch."""
    key = (device, rows, cols)
    ent = _SLAB_CACHE.get(key)
    if ent is None:
        ent = _meta_cache_put(
            _SLAB_CACHE,
            key,
            torch.empty(rows * cols, dtype=torch.float32, device=device),
        )
    return ent


def _get_scalar(device, dtype):
    """Cached one-element scratch (totals, dummy weight pointer slot)."""
    key = (device, dtype)
    ent = _SCALAR_CACHE.get(key)
    if ent is None:
        ent = _meta_cache_put(
            _SCALAR_CACHE, key, torch.empty(1, dtype=dtype, device=device)
        )
    return ent


def _run_histogram(self, bins, weight, density, out):
    """Shared driver for the base and ``.out`` variants."""
    _validate(self, bins, weight)
    D = len(bins)

    if self.dtype == torch.float64:
        # gcu400 has no fp64 arithmetic: the Triton pipeline cannot lower the
        # fp64 comparisons/atomics this kernel needs (the compiler crashes), and
        # computing in fp32 moves values that sit near a bin edge across it.
        # Double precision is therefore handled by the native CPU implementation
        # and copied back; the inputs are small point clouds in practice.
        cpu = torch.ops.aten._histogramdd_from_bin_tensors(
            self.cpu(),
            [b.cpu() for b in bins],
            weight=None if weight is None else weight.cpu(),
            density=density,
        )
        if out is None:
            return cpu.to(self.device)
        if out.dtype != self.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {self.dtype}, "
                f"but got {out.dtype} instead"
            )
        if tuple(out.shape) != tuple(cpu.shape):
            out.resize_(cpu.shape)
        out.copy_(cpu)
        return out

    inp = self.reshape(-1, D).contiguous()
    M = inp.shape[0]

    bin_counts = [b.numel() for b in bins]
    out_shape = tuple(bc - 1 for bc in bin_counts)
    out_dtype = self.dtype

    # Validate / resize the caller's output before computing, matching ATen: a
    # mis-shaped output is resized in place and the same object is returned, but
    # a dtype or device mismatch raises rather than being silently re-bound.
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

    meta = _get_bins_meta(self.device, self.dtype, bins)
    edges = meta["edges"]
    geom = meta["geom"]
    num_bins = meta["num_bins"]
    steps = meta["steps"]

    # fp16/bf16 accumulate in fp32: the atomic/reduce ordering differs from
    # the reference's serial loop, so a low-precision accumulator would drift
    # beyond the reference's own rounding error. The cast happens inside the
    # kernels (loads widen on the fly), so no host-side cast kernel is ever
    # launched for low-precision inputs or weights.
    out_triton = (
        tl.float32
        if out_dtype == torch.float32
        else (tl.float16 if out_dtype == torch.float16 else tl.bfloat16)
    )

    hist_bins = triton.next_power_of_2(num_bins + 1)
    has_weight = weight is not None
    if has_weight:
        w = weight.reshape(-1).contiguous()
    else:
        # Triton needs a valid pointer slot even when the kernel never loads
        # from it.
        w = _get_scalar(self.device, torch.float32)

    # Dispatch: the one-hot dot paths issue no atomics at all and win for
    # narrow bin vectors; per-point atomics only pay off once the bin grid is
    # wide enough to keep per-address collisions rare. Mid-wide grids with a
    # large point cloud go through the private-slab kernel: a shared
    # accumulator chains same-address atomics on the hot cells (Gaussian
    # data over 1000 bins measured 513 us direct vs 70 us slab at 16K
    # points), while the slab caps the collision depth at one tile.
    slab_bytes = triton.cdiv(M, _SLAB_BLOCK) * num_bins * 4
    if has_weight:
        if hist_bins <= _DOT_PERSISTENT_MAX_BINS:
            path = "wdot_persist"
        elif hist_bins <= _DOT_MAX_BINS and M <= _DOT_CHUNK_MAX_POINTS:
            path = "wdot_chunk"
        elif slab_bytes <= _MAX_SLAB_BYTES:
            path = "slab"
        else:
            path = "direct_w"
    else:
        if hist_bins <= _DOT_PERSISTENT_MAX_BINS:
            path = "dot_persist"
        elif hist_bins <= _DOT_MAX_BINS and M <= _DOT_CHUNK_MAX_POINTS:
            path = "dot_chunk"
        elif hist_bins > _DOT_MAX_BINS:
            # Very wide grids keep per-address collisions rare enough for the
            # two-launch direct form (no slab zero + reduce).
            path = "direct"
        elif slab_bytes <= _MAX_SLAB_BYTES:
            path = "slab"
        else:
            path = "direct"

    # The atomic paths need a zeroed f32 accumulator; density needs the counts
    # in f32 for its total/normalisation; everything else can be written
    # straight into the caller's out with a single cast at the reduce store.
    atomic_path = path in ("direct", "direct_w")
    if atomic_path or density:
        hist_is_out = out.is_contiguous() and out_dtype == torch.float32
    else:
        hist_is_out = out.is_contiguous()
    if hist_is_out:
        hist_target = out.view(num_bins)
    else:
        hist_target = torch.empty(num_bins, dtype=torch.float32, device=self.device)

    total = _get_scalar(self.device, torch.float32)

    with torch_device_fn.device(self.device):
        if M == 0:
            # No points: the histogram is all zeros whatever the dispatch
            # path would have been.
            hist_target.zero_()
        elif path == "dot_persist":
            grid_n = min(triton.cdiv(M, _DOT_BLOCK), MAX_CTAS)
            partials = _get_partials(self.device, grid_n, hist_bins + 1, torch.int32)
            histogramdd_dot_persistent_kernel[(grid_n,)](
                inp,
                partials,
                M,
                edges,
                geom,
                MAX_EDGE_STEPS=steps,
                N_CONST=D,
                BLOCK_SIZE=_DOT_BLOCK,
                NUM_BINS=num_bins,
                HIST_BINS=hist_bins,
            )
            histogramdd_reduce_kernel[(triton.cdiv(num_bins, 512),)](
                partials,
                hist_target,
                total,
                grid_n,
                NUM_BINS=num_bins,
                W=hist_bins + 1,
                HAS_TOTAL=True,
                PART_DTYPE=tl.int32,
                OUT_DTYPE=(out_triton if hist_is_out else tl.float32),
                BLOCK_B=512,
            )
        elif path == "wdot_persist":
            grid_n = min(triton.cdiv(M, _WDOT_BLOCK), MAX_CTAS)
            partials = _get_partials(self.device, grid_n, hist_bins + 1, torch.float32)
            histogramdd_wdot_persistent_kernel[(grid_n,)](
                inp,
                w,
                partials,
                M,
                edges,
                geom,
                MAX_EDGE_STEPS=steps,
                N_CONST=D,
                BLOCK_SIZE=_WDOT_BLOCK,
                NUM_BINS=num_bins,
                HIST_BINS=hist_bins,
            )
            histogramdd_reduce_kernel[(triton.cdiv(num_bins, 512),)](
                partials,
                hist_target,
                total,
                grid_n,
                NUM_BINS=num_bins,
                W=hist_bins + 1,
                HAS_TOTAL=True,
                PART_DTYPE=tl.float32,
                OUT_DTYPE=(out_triton if hist_is_out else tl.float32),
                BLOCK_B=512,
            )
        elif path == "dot_chunk":
            grid_n = triton.cdiv(M, _DOT_CHUNK_BLOCK)
            partials = _get_partials(self.device, grid_n, hist_bins + 1, torch.int32)
            histogramdd_dot_kernel[(grid_n,)](
                inp,
                partials,
                M,
                edges,
                geom,
                MAX_EDGE_STEPS=steps,
                N_CONST=D,
                BLOCK_SIZE=_DOT_CHUNK_BLOCK,
                NUM_BINS=num_bins,
                HIST_BINS=hist_bins,
                NB_CHUNK=_DOT_NB_CHUNK,
            )
            histogramdd_reduce_kernel[(triton.cdiv(num_bins, 512),)](
                partials,
                hist_target,
                total,
                grid_n,
                NUM_BINS=num_bins,
                W=hist_bins + 1,
                HAS_TOTAL=True,
                PART_DTYPE=tl.int32,
                OUT_DTYPE=(out_triton if hist_is_out else tl.float32),
                BLOCK_B=512,
            )
        elif path == "wdot_chunk":
            grid_n = triton.cdiv(M, _WDOT_CHUNK_BLOCK)
            partials = _get_partials(self.device, grid_n, hist_bins + 1, torch.float32)
            histogramdd_wdot_kernel[(grid_n,)](
                inp,
                w,
                partials,
                M,
                edges,
                geom,
                MAX_EDGE_STEPS=steps,
                N_CONST=D,
                BLOCK_SIZE=_WDOT_CHUNK_BLOCK,
                NUM_BINS=num_bins,
                HIST_BINS=hist_bins,
                NB_CHUNK=_WDOT_NB_CHUNK,
            )
            histogramdd_reduce_kernel[(triton.cdiv(num_bins, 512),)](
                partials,
                hist_target,
                total,
                grid_n,
                NUM_BINS=num_bins,
                W=hist_bins + 1,
                HAS_TOTAL=True,
                PART_DTYPE=tl.float32,
                OUT_DTYPE=(out_triton if hist_is_out else tl.float32),
                BLOCK_B=512,
            )
        elif path == "slab":
            grid_n = triton.cdiv(M, _SLAB_BLOCK)
            slab = _get_slab(self.device, grid_n, num_bins)
            slab.zero_()
            histogramdd_slab_kernel[(grid_n,)](
                inp,
                w,
                slab,
                M,
                edges,
                geom,
                MAX_EDGE_STEPS=steps,
                N_CONST=D,
                BLOCK_SIZE=_SLAB_BLOCK,
                NUM_BINS=num_bins,
                HAS_WEIGHT=has_weight,
            )
            histogramdd_reduce_kernel[(triton.cdiv(num_bins, 1024),)](
                slab,
                hist_target,
                total,
                grid_n,
                NUM_BINS=num_bins,
                W=num_bins,
                HAS_TOTAL=False,
                PART_DTYPE=tl.float32,
                OUT_DTYPE=(out_triton if hist_is_out else tl.float32),
                BLOCK_B=1024,
            )
        else:  # direct / direct_w
            block = _DIRECT_BLOCK_SMALL if M <= 8192 else 512
            grid_n = triton.cdiv(M, block)
            hist_target.zero_()
            histogramdd_direct_kernel[(grid_n,)](
                inp,
                w,
                hist_target,
                M,
                edges,
                geom,
                MAX_EDGE_STEPS=steps,
                N_CONST=D,
                BLOCK_SIZE=block,
                HAS_WEIGHT=(path == "direct_w"),
            )

        wrote_out = hist_is_out
        if density:
            if atomic_path or path == "slab" or M == 0:
                # The atomic/slab/empty paths have no partials row carrying
                # the total, so reduce it from the counts themselves.
                histogramdd_total_kernel[(1,)](
                    hist_target, total, num_bins, BLOCK_SIZE=1024
                )
            if out.is_contiguous():
                dens_out = out.view(num_bins)
                dens_dtype = out_triton
            else:
                # Non-contiguous out: normalise in place in the f32 buffer
                # and copy once at the end.
                dens_out = hist_target
                dens_dtype = tl.float32
            histogramdd_density_kernel[(triton.cdiv(num_bins, 1024),)](
                hist_target,
                dens_out,
                total,
                edges,
                geom,
                num_bins,
                N_CONST=D,
                OUT_DTYPE=dens_dtype,
                BLOCK_SIZE=1024,
            )
            if out.is_contiguous():
                wrote_out = True

        if not wrote_out:
            out.copy_(hist_target.view(out_shape))
    return out


def _histogramdd_from_bin_tensors(self, bins, *, weight=None, density=False):
    """FlagGems implementation of ``aten::_histogramdd_from_bin_tensors``.

    Args:
        self (Tensor): input with innermost dimension ``D``; all leading
            dimensions are flattened to ``M`` rows of ``D``-dimensional points.
        bins (sequence of Tensor): ``D`` strictly-increasing 1D edge tensors,
            with ``len(bins[d]) - 1`` bins per dimension.
        weight (Tensor, optional): weights of shape ``self.shape[:-1]``.
        density (bool): if True, normalise by total weight and bin volume.

    Returns:
        Tensor: the histogram, shaped ``(len(bins[0])-1, ..., len(bins[D-1])-1)``.
    """
    logger.debug("GEMS_ENFLAME _HISTOGRAMDD_FROM_BIN_TENSORS")
    return _run_histogram(self, bins, weight, density, None)


def _histogramdd_from_bin_tensors_out(
    self, bins, *, weight=None, density=False, out=None
):
    """Out variant of :func:`_histogramdd_from_bin_tensors`."""
    logger.debug("GEMS_ENFLAME _HISTOGRAMDD_FROM_BIN_TENSORS_OUT")
    if out is None:
        raise RuntimeError(
            "_histogramdd_from_bin_tensors.out: 'out' argument is required"
        )
    return _run_histogram(self, bins, weight, density, out)


@libentry()
@triton.jit
def histogramdd_wdot_persistent_kernel(
    inp_ptr,  # (M, N) contiguous input
    weight_ptr,  # (M,) contiguous weights (any float dtype)
    part_ptr,  # (grid, HIST_BINS + 1) f32 per-program partials + total
    M,
    edges_ptr,  # flat f32 concatenated edges
    geom_ptr,  # i32 (3, N) geometry
    MAX_EDGE_STEPS: tl.constexpr,
    N_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    HIST_BINS: tl.constexpr,
):
    """Weighted counting with an f32 one-hot ``tl.dot`` on a persistent grid.

    ``weighted_count[b] = sum_e w_e * (key_e == b)`` is the matmul
    ``w_bcast[16, BLOCK] @ onehot[BLOCK, NBINS]``; the GCU400 f32 dot is a
    true fp32 FMA path (probed bit-exact against serial fp32 sums within one
    ulp), which removes every per-point atomic from the weighted path. The
    f32 one-hot tile is four times the int8 one, so the point block is
    halved to keep the DSM budget.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    acc = tl.zeros((16, HIST_BINS), dtype=tl.float32)
    total = tl.zeros((), dtype=tl.float32)
    nblocks = tl.cdiv(M, BLOCK_SIZE)
    for blk in tl.range(pid, nblocks, num_pids):
        row_start = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = row_start < M
        linear_idx, valid = _histdd_locate(
            inp_ptr,
            row_start,
            row_mask,
            edges_ptr,
            geom_ptr,
            MAX_EDGE_STEPS,
            N_CONST,
            BLOCK_SIZE,
        )
        w = tl.load(weight_ptr + row_start, mask=row_mask, other=0.0).to(tl.float32)
        key = tl.where(valid, linear_idx, HIST_BINS - 1)
        cols = tl.arange(0, HIST_BINS)
        onehot = tl.where(
            key[:, None] == cols[None, :],
            tl.full((BLOCK_SIZE, HIST_BINS), 1.0, tl.float32),
            tl.zeros((BLOCK_SIZE, HIST_BINS), dtype=tl.float32),
        )
        wb = tl.broadcast_to(tl.reshape(w, (1, BLOCK_SIZE)), (16, BLOCK_SIZE))
        acc = tl.dot(wb, onehot, acc, out_dtype=tl.float32)
        # Weights can be negative, so the total is a masked sum, not a max.
        total += tl.sum(tl.where(valid, w, 0.0), axis=0)
    # Row selection instead of tl.max: signed accumulators must not be
    # reduced with a maximum.
    counts = tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, acc, 0.0), axis=0)
    out_cols = tl.arange(0, HIST_BINS)
    tl.store(part_ptr + pid * (HIST_BINS + 1) + out_cols, counts)
    tl.store(part_ptr + pid * (HIST_BINS + 1) + HIST_BINS, total)
