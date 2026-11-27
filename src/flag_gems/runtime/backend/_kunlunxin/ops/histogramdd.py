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

"""Kunlunxin (XPU3) ``histogramdd`` override: zero-aten, launch-optimised.

The pipeline is four Triton kernels; the host path contains no aten compute
(``torch.empty`` for allocation only, plus view slicing of the returned edge
buffer):

1. ``_histdd_setup_kernel`` ``grid=(D,)``: reads dimension ``d`` of the native
   ``(N, D)`` input with an *affine* stride-``D`` load, writes that row into a
   contiguous ``(D, NPAD)`` scratch, reduces ``min``/``max`` over the row,
   applies the degenerate-range fold and writes ``lo``/``hi``.  This replaces
   ``input.t().contiguous()`` (aten transpose) *and* the separate range
   kernel: one launch instead of two plus an aten op.
2. ``_histdd_index_kernel`` ``grid=(NPAD/TILE,)``: per-point row-major linear
   bin index from the contiguous scratch; out-of-range / NaN / padding -> -1.
   Program 0 also writes every ``bin_edges`` row with statically unrolled
   scalar stores (``BINS`` is a constexpr tuple, so every store offset is a
   Python int -- a *tensor* pointer offset is rejected by
   ``TritonXPUUnrollControl`` on this backend).
3. ``_histdd_count_kernel`` ``grid=(num_bins,)``: bin-major, atomic-free count
   over the int32 index array; exactly one writer per output bin
   (``tl.atomic_add`` mis-accumulates here, see
   ``pattern-xpu3-atomic-add-misaccumulates``).
4. ``_histdd_density_kernel`` ``grid=(1,)`` (``density=True`` only): in-kernel
   total + uniform-volume scale, so the host never issues ``hist.sum()``
   (which would be an aten reduction *and* a device->host sync).

``_histdd_edges_kernel`` is the ``N == 0`` path: there is nothing to transpose
or count, and the edges follow from the (already constant-folded) range, so
only the edge write and the zero fill are launched.

Why the launch path matters: ``do_bench`` queues a 256 MB L2 flush before each
timed rep, so once the whole host path costs less than that flush the measured
latency collapses to device time.  The benchmark shapes are 4096 points, i.e.
everything is launch bound, therefore launches go through the compiled
kernel's packed-metadata runner (~6-10 us) instead of ``JITFunction.run``
(~12-14 us).

Backend traps honoured here (all measured, see the KernelOpt wiki):
* masked loads ignore ``other`` -> clamp the address and mask with ``tl.where``;
* no per-lane gather at large BLOCK (NOC hang) -> only affine loads;
* no runtime-D loop with an ``axis=0`` reduction (silently NaN);
* no cross-program vector store into a padded buffer (row offset lands wrong)
  -> edges are written by a single program with scalar stores;
* ``tl.floor`` is a soft extern (~0.3 ns/lane) -> the int cast truncates
  identically for the in-range, non-negative ``(val - lo) * inv_width``.
"""

import builtins
import contextlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Point tile of the index pass: one program per 1024 points keeps the
# program count (parallelism) high without inflating the per-program tail.
_INDEX_TILE = 1024
# Point tile of the count pass: 4096 lanes measured best (8192 regressed).
_COUNT_TILE = 4096
# Point tile (and upper bound) of the setup pass.  BLOCK >= 32768 bare
# launches are unreliable on this backend, so a larger N is chunked.
_SETUP_TILE = 4096
# Static unroll depth limit of the count loop (measured: 64 ok, 128 fails to
# tune its buffer size).
_MAX_STATIC_NCH = 64
# Pointer alignment assumed by `warmup()` (it builds MockTensors with
# data_ptr == 0); a misaligned launch takes the plain Triton path instead.
_ALIGN = 16
_NULL_CTX = contextlib.nullcontext()

_PLAN_CACHE = {}
_LAUNCH_CACHE = {}


@triton.jit
def _histdd_setup_kernel(
    in_ptr,
    t_ptr,
    lo_hi_ptr,
    N,
    sd0,
    sd1,
    NCH_RT,
    D: tl.constexpr,
    TILE: tl.constexpr,
    SINGLE: tl.constexpr,
    TAIL: tl.constexpr,
    CONTIG: tl.constexpr,
    GIVEN: tl.constexpr,
    LO_T: tl.constexpr,
    HI_T: tl.constexpr,
):
    """One program per dimension: transpose a row, then reduce it to (lo, hi).

    ``CONTIG`` selects the affine ``offs * D`` address (constant stride, the
    fast block-DMA form) used by contiguous inputs; otherwise the runtime
    strides are used.  ``GIVEN`` takes the range from the constexpr tuples
    instead of the reduction (explicit ``range=`` and the ``N == 0`` case,
    both already folded by the caller).
    """
    d = ext.program_id(0)
    offs = tl.arange(0, TILE)
    out_row = t_ptr + d * N

    if SINGLE:
        # A single tile covers every point.  When the point count is not
        # tile-aligned the address must be clamped (masked loads read
        # neighbouring memory for the masked lanes on this backend) -- but the
        # clamp makes the address non-affine, which this backend lowers to a
        # per-lane gather (~10 ns/element, 5-8x the block-DMA cost).  So the
        # tile-aligned case takes the unmasked, affine path instead.
        if TAIL:
            safe = tl.maximum(tl.minimum(offs, N - 1), 0)
            m = offs < N
            if CONTIG:
                src = in_ptr + d + safe * D
            else:
                src = in_ptr + d * sd1 + safe * sd0
            vraw = tl.load(src)
            tl.store(out_row + offs, vraw, mask=m)
            v = tl.where(m, vraw.to(tl.float32), float("inf"))
            mn = tl.min(v, axis=0)
            v = tl.where(m, vraw.to(tl.float32), float("-inf"))
            mx = tl.max(v, axis=0)
        else:
            if CONTIG:
                src = in_ptr + d + offs * D
            else:
                src = in_ptr + d * sd1 + offs * sd0
            vraw = tl.load(src)
            tl.store(out_row + offs, vraw)
            mn = tl.min(vraw.to(tl.float32), axis=0)
            mx = tl.max(vraw.to(tl.float32), axis=0)
    else:
        # Chunked with *vector* accumulators; the only reduction is the final
        # one (a loop-carried scalar reduce does not lower on this backend).
        acc_min = tl.full((TILE,), float("inf"), tl.float32)
        acc_max = tl.full((TILE,), float("-inf"), tl.float32)
        for c in range(0, NCH_RT):
            pos = c * TILE + offs
            safe = tl.maximum(tl.minimum(pos, N - 1), 0)
            if CONTIG:
                src = in_ptr + d + safe * D
            else:
                src = in_ptr + d * sd1 + safe * sd0
            vraw = tl.load(src)
            mm = pos < N
            tl.store(out_row + pos, vraw, mask=mm)
            v = vraw.to(tl.float32)
            acc_min = tl.minimum(acc_min, tl.where(mm, v, float("inf")))
            acc_max = tl.maximum(acc_max, tl.where(mm, v, float("-inf")))
        mn = tl.min(acc_min, axis=0)
        mx = tl.max(acc_max, axis=0)

    if GIVEN:
        # `d` is a runtime program id, so this dimension's range is selected
        # with masked arithmetic over the constexpr tuples (a Python tuple
        # cannot be indexed by a runtime value).  D >= 2, so the unrolled loop
        # below never degenerates into a single-iteration constexpr loop (that
        # form mislowers on this backend).
        zero = (d * 0).to(tl.float32)
        lo = zero
        hi = zero
        for j in tl.static_range(D):
            hit = (d == j).to(tl.float32)
            lo = lo + hit * LO_T[j]
            hi = hi + hit * HI_T[j]
    else:
        lo = mn
        hi = mx

    # Degenerate range (all values equal) folds into [v - 0.5, v + 0.5].
    deg = lo == hi
    lo = tl.where(deg, lo - 0.5, lo)
    hi = tl.where(deg, hi + 0.5, hi)
    tl.store(lo_hi_ptr + 2 * d, lo)
    tl.store(lo_hi_ptr + 2 * d + 1, hi)


@triton.jit
def _histdd_index_kernel(
    t_ptr,
    lo_hi_ptr,
    idx_ptr,
    N,
    D: tl.constexpr,
    BINS: tl.constexpr,
    STRIDES: tl.constexpr,
    TILE: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Linear row-major bin index per point.

    The store covers the whole padded index buffer unmasked (padding writes
    -1), so the count pass needs neither a mask nor an address clamp.  A
    tile-aligned point count (``TAIL == False``) keeps the load address
    affine -- clamping it turns the block DMA into a per-lane gather.

    The ``bin_edges`` write must NOT live here: guarded by ``if pid == 0`` it
    measured 128/190/254 us at D=2/3/4 (vs 8/10/13 us for the same body
    without it), i.e. ~60 us of extra cost per dimension.  The same scalar
    stores in the ungated single-program ``_histdd_edges_kernel`` cost ~1 us,
    so edges get their own launch.
    """
    pid = ext.program_id(0)
    offs = pid * TILE + tl.arange(0, TILE)
    if TAIL:
        safe = tl.maximum(tl.minimum(offs, N - 1), 0)
        ok = (offs < N).to(tl.int32)
    else:
        safe = offs
        ok = tl.full((TILE,), 1, tl.int32)

    lin = tl.zeros((TILE,), dtype=tl.int32)
    for d in tl.static_range(D):
        v = tl.load(t_ptr + d * N + safe).to(tl.float32)
        lo = tl.load(lo_hi_ptr + 2 * d).to(tl.float32)
        hi = tl.load(lo_hi_ptr + 2 * d + 1).to(tl.float32)
        nb = BINS[d]
        # Truncating cast == floor for the in-range, non-negative
        # (v - lo) / bin_width; the clamp folds the top edge into the last bin
        # and keeps out-of-range lanes from going negative.  The division (not
        # a reciprocal multiply) mirrors the previous, accuracy-verified
        # formula so boundary points land in the same bin as before.
        b = ((v - lo) / ((hi - lo) / nb)).to(tl.int32)
        b = tl.minimum(b, nb - 1)
        # (v >= lo) & (v <= hi) also drops NaN (it fails both comparisons).
        inr = (v >= lo) & (v <= hi)
        ok = ok & inr.to(tl.int32)
        lin = lin + tl.where(inr, b * STRIDES[d], 0)
    tl.store(idx_ptr + offs, tl.where(ok > 0, lin, -1))


@triton.jit
def _histdd_count_kernel(
    idx_ptr,
    hist_ptr,
    NCH_RT,
    NCH: tl.constexpr,
    TILE: tl.constexpr,
    STATIC: tl.constexpr,
):
    """Bin-major count: one program per output bin, one writer per bin.

    ``STATIC`` unrolls the chunk loop (the backend then vectorises the
    load/compare/accumulate chain); it is used while NCH stays below the
    measured unroll-depth limit, otherwise the dynamic form is emitted.
    ``NCH == 0`` (``N == 0``) leaves the accumulator untouched and writes 0.
    """
    b = ext.program_id(0)
    offs = tl.arange(0, TILE)
    acc = tl.zeros((TILE,), dtype=tl.int32)
    if NCH == 1:
        # The common (single-tile) case is emitted straight-line: a
        # single-iteration constexpr loop mislowers on this backend.
        acc = acc + (tl.load(idx_ptr + offs) == b).to(tl.int32)
    elif NCH > 1:
        if STATIC:
            for c in tl.static_range(NCH):
                acc = acc + (tl.load(idx_ptr + c * TILE + offs) == b).to(tl.int32)
        else:
            for c in range(0, NCH_RT):
                acc = acc + (tl.load(idx_ptr + c * TILE + offs) == b).to(tl.int32)
    tl.store(hist_ptr + b, tl.sum(acc, axis=0).to(tl.float32))


@triton.jit
def _histdd_density_kernel(
    hist_ptr,
    lo_hi_ptr,
    NB,
    D: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    FROM_DEVICE: tl.constexpr,
    LO_T: tl.constexpr,
    HI_T: tl.constexpr,
):
    """In-place density normalisation: ``hist / total / prod_d(width_d)``.

    ``FROM_DEVICE == False`` (``N == 0``) folds the host-side range into
    compile-time constants, so that path never touches the ``lo_hi`` buffer.
    """
    offs = tl.arange(0, BLOCK)
    safe = tl.minimum(offs, NB - 1)
    m = offs < NB
    v = tl.load(hist_ptr + safe).to(tl.float32)
    v = tl.where(m, v, 0.0)
    tot = tl.sum(v, axis=0)

    vol = 1.0
    for d in tl.static_range(D):
        if FROM_DEVICE:
            lo = tl.load(lo_hi_ptr + 2 * d).to(tl.float32)
            hi = tl.load(lo_hi_ptr + 2 * d + 1).to(tl.float32)
        else:
            lo = LO_T[d]
            hi = HI_T[d]
        vol = vol * ((hi - lo) / BINS[d])

    # total_count == 0 keeps the plain volume normalisation (ATen semantics).
    scale = tl.where(tot > 0, 1.0 / (tot * vol), 1.0 / vol)
    tl.store(hist_ptr + offs, tl.where(m, v * scale, v), mask=m)


@triton.jit
def _histdd_edges_kernel(
    edges_ptr,
    lo_hi_ptr,
    D: tl.constexpr,
    BINS: tl.constexpr,
    EOFF: tl.constexpr,
    EPAD: tl.constexpr,
    FROM_DEVICE: tl.constexpr,
    LO_T: tl.constexpr,
    HI_T: tl.constexpr,
):
    """Write ``bin_edges`` with ONE vector store (``EPAD >= 64`` lanes).

    Scalar stores are the trap here: each one measured ~8.9 us on this backend
    (128/190/254 us for the 14/21/28 scalar accesses of D=2/3/4, a perfect
    linear fit), no matter whether they sit in the index kernel behind
    ``if pid == 0`` or in their own single-program kernel.  So the whole edge
    slab is materialised as one per-lane vector instead:

    * lane -> (dim, k) through a static ``where`` chain over the constexpr
      tuples (all vector ops, no scalar access);
    * per-lane ``lo``/``hi`` come from one small gather
      (``FROM_DEVICE``) or from the host-side tuples (``N == 0``, folded to
      compile-time constants);
    * one ``tl.store`` over ``EPAD`` lanes, unmasked, addresses contiguous.

    The buffer is ``EPAD`` wide and only the first ``TOT_E`` entries are
    exposed, so the padding lanes may write anything.
    """
    lane = tl.arange(0, EPAD)
    dim = tl.zeros((EPAD,), dtype=tl.int32)
    base = tl.zeros((EPAD,), dtype=tl.int32)
    # Typed constant vector (an untyped literal would be materialised with a
    # different width and rejected by the legalize pass).
    nb = tl.zeros((EPAD,), dtype=tl.int32) + BINS[D - 1]
    for j in tl.static_range(D):
        width = BINS[j] + 1
        hit = (lane >= EOFF[j]) & (lane < EOFF[j] + width)
        dim = dim + tl.where(hit, j, 0)
        base = base + tl.where(hit, EOFF[j], 0)
        nb = tl.where(hit, BINS[j], nb)
    k = lane - base

    if FROM_DEVICE:
        lo = tl.load(lo_hi_ptr + 2 * dim).to(tl.float32)
        hi = tl.load(lo_hi_ptr + 2 * dim + 1).to(tl.float32)
    else:
        lo = tl.zeros((EPAD,), dtype=tl.float32)
        hi = tl.zeros((EPAD,), dtype=tl.float32)
        for j in tl.static_range(D):
            hit = (dim == j).to(tl.float32)
            lo = lo + hit * LO_T[j]
            hi = hi + hit * HI_T[j]

    w = (hi - lo) / nb.to(tl.float32)
    tl.store(edges_ptr + lane, lo + k.to(tl.float32) * w)


def _current_stream():
    """Current device stream handle for the packed-metadata runner."""
    return torch_device_fn.current_stream().cuda_stream


def _aligned(tensors):
    for t in tensors:
        if t.data_ptr() % _ALIGN:
            return False
    return True


def _launch(kernel, cache_key, grid, tensors, scalars, constexprs, stream):
    """Launch through the cached packed-metadata runner.

    ``kernel.warmup`` builds the specialised binary once; every later call only
    pushes ``(grid, stream, fn, packed_metadata, *raw_args)``, which measures
    ~2x cheaper than ``JITFunction.run`` on this stack.  Misaligned pointers
    (the warmup binary assumes 16 B) and any warmup failure fall back to the
    regular Triton launch.
    """
    if _aligned(tensors):
        kern = _LAUNCH_CACHE.get(cache_key)
        if kern is None:
            try:
                kern = kernel.warmup(*tensors, *scalars, grid=grid, **constexprs)
            except Exception:  # pragma: no cover - never break the op
                logger.debug("histogramdd packed launch unavailable", exc_info=True)
                kern = None
            if kern is not None:
                _LAUNCH_CACHE[cache_key] = kern
        if kern is not None:
            kern.run(
                grid[0],
                1,
                1,
                stream,
                kern.function,
                kern.packed_metadata,
                None,
                None,
                None,
                *[t.data_ptr() for t in tensors],
                *scalars,
            )
            return
    kernel[grid](*tensors, *scalars, **constexprs)


def _build_plan(N, D, bins, dtype, device_index, given, lo_t, hi_t, sd0, sd1):
    """All host-side derived values for one (shape, bins, layout) key."""
    eoff = []
    tot_e = 0
    nb_total = 1
    for b in bins:
        eoff.append(tot_e)
        tot_e += b + 1
        nb_total *= b

    # Row-major strides: stride[d] = prod(bins[d+1:]), last dim fastest.
    strides = [1] * D
    for d in builtins.range(D - 2, -1, -1):
        strides[d] = strides[d + 1] * bins[d + 1]

    # Padded point count: a multiple of both tiles so neither pass needs a
    # tail branch (the padding lanes get index -1).
    npad = 0 if N == 0 else triton.cdiv(N, _COUNT_TILE) * _COUNT_TILE
    count_tile = _COUNT_TILE
    nch = npad // count_tile
    static = nch <= _MAX_STATIC_NCH
    while not static and count_tile < 16384:
        count_tile *= 2
        nch = npad // count_tile
        static = nch <= _MAX_STATIC_NCH

    if N <= _SETUP_TILE:
        setup_tile = max(64, triton.next_power_of_2(max(N, 1)))
        setup_single = True
        setup_nch = 0
    else:
        setup_tile = _SETUP_TILE
        setup_single = False
        setup_nch = triton.cdiv(N, _SETUP_TILE)

    return {
        "bins": bins,
        "strides": tuple(strides),
        "eoff": tuple(eoff),
        "tot_e": tot_e,
        # Edge slab width: at least 64 lanes (narrower vector stores are not
        # reliable on this backend) and never below the real edge count, so
        # the edges kernel can issue one unmasked vector store.
        "epad": max(64, triton.next_power_of_2(max(tot_e, 1))),
        "nb_total": nb_total,
        "npad": npad,
        "count_tile": count_tile,
        "nch": nch,
        "static": static,
        "setup_tile": setup_tile,
        "setup_single": setup_single,
        "setup_nch": setup_nch,
        "setup_tail": N != setup_tile,
        "index_tail": N != npad,
        "index_tile": _INDEX_TILE,
        "n_index_prog": max(1, npad // _INDEX_TILE),
        "given": given,
        "lo_t": lo_t,
        "hi_t": hi_t,
        "contig": sd1 == 1 and sd0 == D,
        "sd0": sd0,
        "sd1": sd1,
        "density_block": max(64, triton.next_power_of_2(max(nb_total, 1))),
    }


def histogramdd(input, bins, range=None, weight=None, density=False):
    """Compute the D-dimensional histogram of an (N, D) input tensor.

    Functional override of the generic KernelGen ``histogramdd``: identical
    edge semantics and density normalisation, but the whole pipeline runs on
    our own Triton kernels (no aten compute, no host round trips).
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

    if len(bins) != D:
        raise ValueError(f"bins length {len(bins)} must match input dimensions {D}")

    if weight is not None:
        if weight.shape != (N,):
            raise ValueError(f"weight must have shape ({N},), got {weight.shape}")
        if weight.dtype != input.dtype:
            raise ValueError(
                f"weight dtype {weight.dtype} must match input dtype {input.dtype}"
            )
        raise NotImplementedError("weighted histogramdd not yet implemented")

    bins_host = [int(b) for b in bins]
    for b in bins_host:
        if b <= 0:
            raise ValueError(f"bins must be positive, got {b}")

    device = input.device
    dtype = input.dtype

    if range is not None:
        # The degenerate fold is pure float arithmetic, so it is applied here
        # once instead of being re-derived per program on the device.
        lo_list = []
        hi_list = []
        for d in builtins.range(D):
            lo = float(range[2 * d])
            hi = float(range[2 * d + 1])
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            lo_list.append(lo)
            hi_list.append(hi)
        given = True
        lo_t = tuple(lo_list)
        hi_t = tuple(hi_list)
    elif N == 0:
        # ATen's empty-input default range.
        given = True
        lo_t = tuple([0.0] * D)
        hi_t = tuple([1.0] * D)
    else:
        given = False
        lo_t = hi_t = ()

    sd0 = input.stride(0)
    sd1 = input.stride(1)
    device_index = device.index

    key = (
        N,
        D,
        tuple(bins_host),
        dtype,
        device_index,
        given,
        lo_t,
        hi_t,
        sd0,
        sd1,
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_plan(
            N, D, tuple(bins_host), dtype, device_index, given, lo_t, hi_t, sd0, sd1
        )
        _PLAN_CACHE[key] = plan

    npad = plan["npad"]
    t_scratch = torch.empty((D, npad), dtype=dtype, device=device)
    lo_hi = torch.empty(2 * D, dtype=torch.float32, device=device)
    idx = torch.empty(npad, dtype=torch.int32, device=device)
    # EPAD wide: the edges kernel stores a full vector, the padding lanes land
    # inside this allocation and are never exposed.
    edges = torch.empty(plan["epad"], dtype=dtype, device=device)
    hist = torch.empty(tuple(bins_host), dtype=dtype, device=device)

    # Only enter the device guard when the caller's current device differs
    # (entering it unconditionally costs ~8 us per call on this stack).
    ctx = _NULL_CTX
    if device_index is not None and device_index != torch_device_fn.current_device():
        ctx = torch_device_fn.device(device)

    with ctx:
        stream = _current_stream()

        if N == 0:
            _launch(
                _histdd_edges_kernel,
                (key, "edges"),
                (1,),
                [edges, lo_hi],
                [],
                {
                    "D": D,
                    "BINS": plan["bins"],
                    "EOFF": plan["eoff"],
                    "EPAD": plan["epad"],
                    "FROM_DEVICE": False,
                    "LO_T": plan["lo_t"],
                    "HI_T": plan["hi_t"],
                },
                stream,
            )
        else:
            _launch(
                _histdd_setup_kernel,
                (key, "setup"),
                (D,),
                [input, t_scratch, lo_hi],
                [N, sd0, sd1, plan["setup_nch"]],
                {
                    "D": D,
                    "TILE": plan["setup_tile"],
                    "SINGLE": plan["setup_single"],
                    "TAIL": plan["setup_tail"],
                    "CONTIG": plan["contig"],
                    "GIVEN": plan["given"],
                    "LO_T": plan["lo_t"],
                    "HI_T": plan["hi_t"],
                },
                stream,
            )

            # Edges go in their own launch, and as ONE vector store: the
            # equivalent scalar stores cost ~8.9 us each on this backend
            # (125/187/248 us at D=2/3/4), wherever they live.
            _launch(
                _histdd_edges_kernel,
                (key, "edges"),
                (1,),
                [edges, lo_hi],
                [],
                {
                    "D": D,
                    "BINS": plan["bins"],
                    "EOFF": plan["eoff"],
                    "EPAD": plan["epad"],
                    "FROM_DEVICE": True,
                    "LO_T": (),
                    "HI_T": (),
                },
                stream,
            )

            _launch(
                _histdd_index_kernel,
                (key, "index"),
                (plan["n_index_prog"],),
                [t_scratch, lo_hi, idx],
                [N],
                {
                    "D": D,
                    "BINS": plan["bins"],
                    "STRIDES": plan["strides"],
                    "TILE": plan["index_tile"],
                    "TAIL": plan["index_tail"],
                },
                stream,
            )

        _launch(
            _histdd_count_kernel,
            (key, "count"),
            (plan["nb_total"],),
            [idx, hist],
            [plan["nch"]],
            {
                "NCH": plan["nch"],
                "TILE": plan["count_tile"],
                "STATIC": plan["static"],
            },
            stream,
        )

        if density:
            _launch(
                _histdd_density_kernel,
                (key, "density"),
                (1,),
                [hist, lo_hi],
                [plan["nb_total"]],
                {
                    "D": D,
                    "BINS": plan["bins"],
                    "BLOCK": plan["density_block"],
                    "FROM_DEVICE": bool(N),
                    "LO_T": plan["lo_t"],
                    "HI_T": plan["hi_t"],
                },
                stream,
            )

    # Contiguous views into the flat edge buffer (no copy, no kernel).  The
    # buffer is EPAD wide; only the first `tot_e` entries are exposed.
    bin_edges = [
        edges[plan["eoff"][d] : plan["eoff"][d] + bins_host[d] + 1]
        for d in builtins.range(D)
    ]
    return hist, bin_edges
