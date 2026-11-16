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

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Points processed per program by the index kernel, coordinates scanned per inner
# iteration by the setup reduction, and points scanned per program by the count
# kernel.  Fixed constexpr tiles (instead of ``next_power_of_2(N)``) keep every
# reduction inside the verified 64..8192 lane window and let ``N`` exceed a tile.
# Measured on P800/XPU3 at N=4096: the index kernel is fastest at 1024 (it wants
# program-count parallelism), the bin-major count kernel at 4096 (it wants the
# fewest chunk iterations), so they use different tiles and the index scratch is
# padded to a multiple of the count tile.
SETUP_TILE = 1024
INDEX_TILE = 1024
COUNT_TILE = 4096


@libentry()
@triton.jit
def histogramdd_setup_kernel(
    input_ptr,  # (D, N) contiguous float input (transposed by the caller)
    lo_ptr,  # (DPAD,) per-dimension lower edge (device)
    hi_ptr,  # (DPAD,) per-dimension upper edge (device)
    edges_ptr,  # (DPAD * MAX_EDGES,) padded per-dimension edge slabs
    N,
    D: tl.constexpr,
    DPAD: tl.constexpr,  # next_pow2(D)
    BINS: tl.constexpr,  # tuple, per-dimension bin counts
    MAX_EDGES: tl.constexpr,  # edge slab width, >= 64 (narrow 2D tiles scramble)
    TILE: tl.constexpr,
):
    """Derive ``(lo, hi)`` per dimension from the data and emit the bin edges.

    The reduction runs as a 2D ``(DPAD, TILE)`` tile whose rows are the
    transposed input's per-dimension rows (stride-1 block DMA), so one
    ``axis=1`` reduce yields every dimension's min/max at once -- a serial
    per-dimension loop was measured 2x slower.  ``DPAD``/``TILE`` are powers of
    two so the axis-1 reduction stays in the verified ``%8 == 0`` width window.

    The tail is handled with an explicit address clamp plus a lane mask (this
    backend ignores the ``other`` fill of a masked load and would read
    neighbouring memory), and padded rows clamp back to the last real dimension.

    Writing ``lo``/``hi``/``edges`` on the device removes the two
    ``device -> host`` ``.tolist()`` round trips (and the per-dimension
    ``torch.linspace`` launches) the previous implementation paid for.
    """
    j = tl.arange(0, DPAD)
    i = tl.arange(0, TILE)
    safe_j = tl.minimum(j, D - 1)
    row = input_ptr + safe_j[:, None] * N

    acc_min = tl.full((DPAD, TILE), float("inf"), dtype=tl.float32)
    acc_max = tl.full((DPAD, TILE), float("-inf"), dtype=tl.float32)
    nchunks = tl.cdiv(N, TILE)
    for c in range(0, nchunks):
        cur = c * TILE + i
        safe_i = tl.minimum(cur, N - 1)
        vals = tl.load(row + safe_i[None, :]).to(tl.float32)
        keep = (cur < N)[None, :]
        acc_min = tl.minimum(acc_min, tl.where(keep, vals, float("inf")))
        acc_max = tl.maximum(acc_max, tl.where(keep, vals, float("-inf")))
    mn = tl.min(acc_min, axis=1)
    mx = tl.max(acc_max, axis=1)

    # Degenerate range (all values equal) folds into [v-0.5, v+0.5], matching
    # ``torch.histogramdd``.
    lo = tl.where(mn == mx, mn - 0.5, mn)
    hi = tl.where(mn == mx, mn + 0.5, mx)
    tl.store(lo_ptr + j, lo)
    tl.store(hi_ptr + j, hi)

    # Per-dimension bin count as a vector: DPAD is constexpr, so a static
    # where-chain can pick ``BINS[d]`` without a device-side bins tensor.
    nb = tl.zeros((DPAD,), dtype=tl.float32) + BINS[0]
    for dd in tl.static_range(1, D):
        nb = tl.where(j == dd, BINS[dd] * 1.0, nb)

    e = tl.arange(0, MAX_EDGES)
    safe_e = tl.minimum(e[None, :], nb[:, None].to(tl.int32))
    step = (hi - lo) / nb
    edge = lo[:, None] + safe_e.to(tl.float32) * step[:, None]
    tl.store(edges_ptr + j[:, None] * MAX_EDGES + e[None, :], edge)


@libentry()
@triton.jit
def histogramdd_setup_range_kernel(
    lohi_ptr,  # (2 * D,) [lo0, hi0, lo1, hi1, ...] already degenerate-adjusted
    lo_ptr,  # (D,) per-dimension lower edge (device)
    hi_ptr,  # (D,) per-dimension upper edge (device)
    edges_ptr,  # (D * MAX_EDGES,) padded per-dimension edge slabs
    D: tl.constexpr,
    BINS: tl.constexpr,
    MAX_EDGES: tl.constexpr,
):
    """Explicit-range variant of :func:`histogramdd_setup_kernel`.

    ``lo``/``hi`` come from the caller's ``range`` (or the ``[0, 1]`` default for
    an empty input), so only the edge slabs have to be materialised.
    """
    idx = tl.arange(0, MAX_EDGES)
    for d in tl.static_range(D):
        lo = tl.load(lohi_ptr + 2 * d).to(tl.float32)
        hi = tl.load(lohi_ptr + 2 * d + 1).to(tl.float32)
        tl.store(lo_ptr + d, lo)
        tl.store(hi_ptr + d, hi)

        nb = BINS[d]
        step = (hi - lo) / nb
        safe_i = tl.minimum(idx, nb)
        edge = lo + safe_i.to(tl.float32) * step
        tl.store(edges_ptr + d * MAX_EDGES + idx, edge)


@libentry()
@triton.jit
def histogramdd_index_kernel(
    input_ptr,  # (D, N) contiguous float input (transposed by the caller)
    lo_ptr,  # (D,) per-dimension lower edge (device)
    hi_ptr,  # (D,) per-dimension upper edge (device)
    idx_ptr,  # (nchunks * TILE,) int32 row-major linear bin (or -1) per point
    N,
    D: tl.constexpr,
    BINS: tl.constexpr,
    STRIDES: tl.constexpr,
    TILE: tl.constexpr,
):
    """Compute every point's row-major linear bin index exactly once.

    The previous design re-derived all ``D`` coordinates inside *every*
    output-bin program (``num_bins * N * D`` lane evaluations, each paying a
    soft-extern ``tl.floor``).  Folding the bin computation into one pass costs
    ``N * D`` lanes, and the fallback for out-of-range / NaN points is ``-1`` so
    the counting pass never matches them.
    """
    pid = ext.program_id(0)
    offs = pid * TILE + tl.arange(0, TILE)
    safe = tl.minimum(offs, N - 1)

    linear = tl.zeros((TILE,), dtype=tl.int32)
    in_range = tl.full((TILE,), 1, dtype=tl.int32)
    for d in tl.static_range(D):
        val = tl.load(input_ptr + d * N + safe).to(tl.float32)
        lo = tl.load(lo_ptr + d).to(tl.float32)
        hi = tl.load(hi_ptr + d).to(tl.float32)
        nb = BINS[d]

        # NaN fails both comparisons and is therefore excluded here; values
        # exactly at the upper edge stay in the last bin.
        in_range = in_range & ((val >= lo) & (val <= hi)).to(tl.int32)

        # ``floor((val - lo) / width)`` for in-range values: they are
        # non-negative, so the int cast truncates exactly like ``floor`` -- this
        # avoids the soft-extern ``tl.floor`` (3-40x slower per lane on XPU3).
        inv_width = nb / (hi - lo)
        bin_idx = tl.minimum(((val - lo) * inv_width).to(tl.int32), nb - 1)
        linear = linear + bin_idx * STRIDES[d]

    # Padding lanes (offs >= N) are also tagged -1 so the counting pass can scan
    # the whole padded tile without a mask.
    linear = tl.where((offs < N) & (in_range != 0), linear, -1)
    tl.store(idx_ptr + offs, linear)


@libentry()
@triton.jit
def histogramdd_count_kernel(
    idx_ptr,  # (nchunks * TILE,) int32 linear bin per point, -1 = drop
    hist_ptr,  # (num_bins,) histogram accumulator
    N,
    TILE: tl.constexpr,
):
    """One program per output bin: count the points whose linear index is ``b``.

    Bin-major ownership means no cross-program atomic is issued (``tl.atomic_add``
    is unreliable on this XPU backend) and every output element has exactly one
    writer, so the histogram buffer only needs an allocation, not a zero fill.
    The scan is over int32 indices (one load per point) instead of ``D`` float
    coordinates, and it drops the per-dimension scalar loads the old bin-major
    kernel paid once per program.
    """
    b = ext.program_id(0)
    offs = tl.arange(0, TILE)
    acc = tl.zeros((TILE,), dtype=tl.int32)
    nchunks = tl.cdiv(N, TILE)
    for c in range(0, nchunks):
        # The index buffer is padded to ``nchunks * TILE`` and the padding is
        # pre-tagged -1, so the scan needs neither an address clamp nor a mask
        # (clamping would re-read a real index and double count the tail).
        acc += (tl.load(idx_ptr + c * TILE + offs) == b).to(tl.int32)
    tl.store(hist_ptr + b, tl.sum(acc, axis=0).to(tl.float32))


@libentry()
@triton.jit
def histogramdd_density_kernel(
    hist_ptr,  # (num_bins,) counts, normalised in place
    lo_ptr,  # (D,) per-dimension lower edge (device)
    hi_ptr,  # (D,) per-dimension upper edge (device)
    num_bins,
    D: tl.constexpr,
    BINS: tl.constexpr,
    TILE: tl.constexpr,
):
    """Density normalisation in place, without any host round trip.

    ``torch``-side ``hist.sum() / hist / bin_volumes`` would launch a reduction,
    force a host sync for the zero-total branch and run one multiply per
    dimension.  Uniform bins make the volume a single scalar (the product of the
    per-dimension widths), so one program can fold total + volume into one scale.
    """
    offs = tl.arange(0, TILE)
    nchunks = tl.cdiv(num_bins, TILE)

    acc = tl.zeros((TILE,), dtype=tl.float32)
    for c in range(0, nchunks):
        cur = c * TILE + offs
        safe = tl.minimum(cur, num_bins - 1)
        vals = tl.load(hist_ptr + safe).to(tl.float32)
        acc += tl.where(cur < num_bins, vals, 0.0)
    total = tl.sum(acc, axis=0)

    volume = 1.0
    for d in tl.static_range(D):
        lo = tl.load(lo_ptr + d).to(tl.float32)
        hi = tl.load(hi_ptr + d).to(tl.float32)
        volume = volume * ((hi - lo) / BINS[d])

    # Empty histogram keeps ``torch`` behaviour (0 / volume == 0).
    scale = tl.where(total > 0, 1.0 / (total * volume), 1.0 / volume)
    for c in range(0, nchunks):
        cur = c * TILE + offs
        tail = cur < num_bins
        # Masked store on purpose: clamping the address would let several lanes
        # rewrite the last bin from its *already normalised* value (this backend
        # walks wide tiles in sequential lane groups, so the re-read sees the
        # scaled result and the last bin ends up normalised twice).
        vals = tl.load(hist_ptr + tl.minimum(cur, num_bins - 1)).to(tl.float32)
        tl.store(hist_ptr + cur, vals * scale, mask=tail)


def _normalise_bins(bins, D):
    """``torch.histogramdd`` accepts an int, a sequence or a tensor of bin counts."""
    if isinstance(bins, int):
        return [bins] * D
    if torch.is_tensor(bins):
        return [int(b) for b in bins.reshape(-1).tolist()]
    return [int(b) for b in bins]


def _normalise_range(range, D):
    """Flatten ``range`` to ``[lo0, hi0, lo1, hi1, ...]`` with degenerate widening.

    Accepts the flat ``2 * D`` form, a sequence of ``(min, max)`` pairs and device
    tensors, mirroring ``torch.histogramdd``.
    """
    if torch.is_tensor(range):
        flat = [float(v) for v in range.reshape(-1).tolist()]
    else:
        flat = []
        for entry in range:
            if isinstance(entry, (list, tuple)):
                flat.extend(float(v) for v in entry)
            else:
                flat.append(float(entry))

    if len(flat) != 2 * D:
        raise ValueError(
            f"range must provide {2 * D} values (lo/hi per dim), got {flat}"
        )

    out = []
    for d in builtins.range(D):
        dim_min = flat[2 * d]
        dim_max = flat[2 * d + 1]
        if dim_min == dim_max:
            dim_min -= 0.5
            dim_max += 0.5
        out.extend([dim_min, dim_max])
    return out


def histogramdd(input, bins, range=None, weight=None, density=False):
    """Compute the D-dimensional histogram of a (N, D) input tensor.

    Functional override of the generic KernelGen ``histogramdd`` with identical
    edge / density semantics, but the accumulation is atomic-free and split into
    three passes:

    1. :func:`histogramdd_setup_kernel` reduces each dimension's contiguous row to
       ``(lo, hi)`` and writes the bin edges on the device (no host round trip).
    2. :func:`histogramdd_index_kernel` computes every point's linear bin index
       once (``N * D`` lane evaluations).
    3. :func:`histogramdd_count_kernel` owns one output bin per program and counts
       matching indices (``num_bins * N`` int32 compares).

    The old bin-major kernel re-derived all coordinates inside every bin program
    (``num_bins * N * D`` lanes, each with a soft-extern ``tl.floor``) and paid
    ``4 * D`` scalar loads per program.
    """
    logger.debug("GEMS_KUNLUNXIN HISTOGRAMDD")

    if input.dtype not in [torch.float32, torch.float64]:
        raise NotImplementedError(
            f"histogramdd not implemented for dtype {input.dtype}. "
            "Only float32 and float64 are supported."
        )

    if input.dim() != 2:
        raise ValueError(
            f"histogramdd requires 2D input (N, D), got shape {input.shape}"
        )

    N, D = input.shape
    if D < 2:
        raise ValueError(f"histogramdd requires at least 2 dimensions, got D={D}")

    bins = _normalise_bins(bins, D)
    if len(bins) != D:
        raise ValueError(f"bins length {len(bins)} must match input dimensions {D}")
    if any(b <= 0 for b in bins):
        raise ValueError(f"each bin count must be positive, got {bins}")

    if weight is not None:
        if weight.shape != (N,):
            raise ValueError(f"weight must have shape ({N},), got {weight.shape}")
        if weight.dtype != input.dtype:
            raise ValueError(
                f"weight dtype {weight.dtype} must match input dtype {input.dtype}"
            )
        raise NotImplementedError("weighted histogramdd not yet implemented")

    input = input.contiguous()
    device = input.device

    # Row-major strides: stride[d] = prod(bins[d+1:]), last dim fastest.
    strides = [1] * D
    for d in builtins.range(D - 2, -1, -1):
        strides[d] = strides[d + 1] * bins[d + 1]

    num_bins = 1
    for b in bins:
        num_bins *= b

    max_edges = triton.next_power_of_2(max(bins) + 1)
    # The edge slab is written with a 2D ``(DPAD, MAX_EDGES)`` store; a tile
    # narrower than one warp (<64 lanes) is silently miscompiled by this backend
    # (measured: scattered values for MAX_EDGES=8), so the slab is padded.
    edge_stride = max(64, max_edges)
    bins_key = tuple(bins)
    dpad = triton.next_power_of_2(D)

    lo = torch.empty(dpad, dtype=torch.float32, device=device)
    hi = torch.empty(dpad, dtype=torch.float32, device=device)
    edges = torch.empty(dpad * edge_stride, dtype=input.dtype, device=device)
    hist = torch.empty(tuple(bins), dtype=input.dtype, device=device)

    with torch_device_fn.device(device):
        # Per-dimension coordinates become contiguous, so both the reduction and
        # the binning pass read stride-1 block DMA instead of a strided gather (a
        # large strided gather hangs this backend's NOC).
        input_t = input.t().contiguous() if N > 0 else input
        if range is None and N > 0:
            histogramdd_setup_kernel[(1,)](
                input_t,
                lo,
                hi,
                edges,
                N,
                D=D,
                DPAD=dpad,
                BINS=bins_key,
                MAX_EDGES=edge_stride,
                TILE=SETUP_TILE,
            )
            need_index = True
        else:
            lohi_host = [0.0, 1.0] * D if range is None else _normalise_range(range, D)
            lohi = torch.tensor(lohi_host, dtype=torch.float32, device=device)
            histogramdd_setup_range_kernel[(1,)](
                lohi,
                lo,
                hi,
                edges,
                D=D,
                BINS=bins_key,
                MAX_EDGES=edge_stride,
            )
            need_index = N > 0

        if need_index:
            # The index scratch is padded to a multiple of COUNT_TILE; the extra
            # programs clamp-load and tag their lanes -1, so the counting pass can
            # scan it unmasked.
            padded = triton.cdiv(N, COUNT_TILE) * COUNT_TILE
            idx = torch.empty(padded, dtype=torch.int32, device=device)
            histogramdd_index_kernel[(padded // INDEX_TILE,)](
                input_t,
                lo,
                hi,
                idx,
                N,
                D=D,
                BINS=bins_key,
                STRIDES=tuple(strides),
                TILE=INDEX_TILE,
            )
            histogramdd_count_kernel[(num_bins,)](idx, hist, N, TILE=COUNT_TILE)
            if density:
                histogramdd_density_kernel[(1,)](
                    hist, lo, hi, num_bins, D=D, BINS=bins_key, TILE=SETUP_TILE
                )
        else:
            # No point can land in any bin.
            hist = torch.zeros(tuple(bins), dtype=input.dtype, device=device)

    bin_edges = [
        edges[d * edge_stride : d * edge_stride + bins[d] + 1]
        for d in builtins.range(D)
    ]
    return hist, bin_edges
