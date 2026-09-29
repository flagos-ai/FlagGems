import logging
import math
import os
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy
from .sort import radix_sort_low_mem

logger = logging.getLogger(__name__)

ModeResult = namedtuple("mode", ["values", "indices"])

# ---------------------------------------------------------------------------
# Big-N histogram mode (tle.raw).  fp16 / bf16 / int16 all have only 65536
# distinct bit patterns, so mode = argmax of a sortable-key histogram (O(N)
# single pass), which breaks the O(N log^2 N) bitonic / shared radix_sort
# ceiling.  Two-pass half-histogram (32768 int32 buckets/pass, lock-free SM
# atomics) since 65536*4B = 256KB > 128KB cluster SM budget.  Beats both the
# shipped radix path (40-640ms) and torch CPU-fallback (0.1-40ms) on every
# N>1 shape measured.  fp32/int32 (2^32 buckets infeasible) keep the radix
# path.
# ---------------------------------------------------------------------------
_HAS_RAW_HIST = False
try:
    import triton.experimental.tle as _tle_ext
    import triton.experimental.tle.language as _tle

    _RAW_DIR = os.path.dirname(os.path.abspath(__file__))
    # Single shippable object: all 9 mode payload entries (f16/bf16/i16 hist +
    # i32/f32 lm/lm2/radix) are relocatably linked (ld.lld -r) into one mode.o.
    # In object= mode the loader resolves each stub by its entry-symbol name;
    # stub name must equal the C++ entry symbol and the signature must match the
    # C++ ABI (the merge-time signature check is skipped for object=).  Rebuilt
    # via docs/xpu3/how_to_pack_payload/pack_payload.py + ld.lld -r after any
    # edit to the .xpu sources (object= cache key is the .o digest).
    _MODE_OBJ = os.path.join(
        os.path.dirname(_RAW_DIR), "payload", "obj", "mode.o"
    )

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_f16_hist(inp, oval, oidx, rows, N, nclusters): ...

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_bf16_hist(inp, oval, oidx, rows, N, nclusters): ...

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_i16_hist(inp, oval, oidx, rows, N, nclusters): ...

    @triton.jit(do_not_specialize=["rows", "N", "nclusters"])
    def _mode_hist_f16_w(Inp, Oval, Oidx, rows, N, nclusters):
        _tle.raw.call(mode_f16_hist, (Inp, Oval, Oidx, rows, N, nclusters))

    @triton.jit(do_not_specialize=["rows", "N", "nclusters"])
    def _mode_hist_bf16_w(Inp, Oval, Oidx, rows, N, nclusters):
        _tle.raw.call(mode_bf16_hist, (Inp, Oval, Oidx, rows, N, nclusters))

    @triton.jit(do_not_specialize=["rows", "N", "nclusters"])
    def _mode_hist_i16_w(Inp, Oval, Oidx, rows, N, nclusters):
        _tle.raw.call(mode_i16_hist, (Inp, Oval, Oidx, rows, N, nclusters))

    _RAW_HIST = {
        torch.float16: _mode_hist_f16_w,
        torch.bfloat16: _mode_hist_bf16_w,
        torch.int16: _mode_hist_i16_w,
    }
    _HAS_RAW_HIST = True
except Exception as _e:  # pragma: no cover - defensive
    logger.debug("mode histogram raw payload unavailable: %s", _e)
    _RAW_HIST = {}


# ---------------------------------------------------------------------------
# fp32 / int32 mode (tle.raw).  2^32 buckets rule out a direct histogram, so
# these are sorted on-device with LSD radix (8-bit x 4 pass).  Two tiers:
#   * N<=512  -> per-core in-LM radix (one row per core, NO global scatter,
#     O(N)).  Avoids the per-element LM2GM scatter-bandwidth wall entirely.
#   * N>512 & few rows (M<=32) -> cluster-cooperative radix (one row per
#     cluster, 64 cores share the scatter ~1/64 each).  Only wins for large-N
#     few-row shapes; per-element scatter is catastrophic for many-row shapes,
#     which stay on the shared radix_sort path.
# Both beat the CPU-fallback torch baseline on the shapes they route to.
# ---------------------------------------------------------------------------
_HAS_RAW_RADIX = False
try:
    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_i32_lm(inp, oval, oidx, rows, N, ncores): ...

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_f32_lm(inp, oval, oidx, rows, N, ncores): ...

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_i32_lm2(inp, oval, oidx, rows, N, ncores): ...

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_f32_lm2(inp, oval, oidx, rows, N, ncores): ...

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_i32_radix(inp, oval, oidx, bufA, bufB, rows, N, nclusters): ...

    @_tle_ext.raw.dialect("xpu3", object=_MODE_OBJ, arch=3)
    def mode_f32_radix(inp, oval, oidx, bufA, bufB, rows, N, nclusters): ...

    @triton.jit(do_not_specialize=["rows", "N", "ncores"])
    def _mode_lm_i32_w(Inp, Oval, Oidx, rows, N, ncores):
        _tle.raw.call(mode_i32_lm, (Inp, Oval, Oidx, rows, N, ncores))

    @triton.jit(do_not_specialize=["rows", "N", "ncores"])
    def _mode_lm_f32_w(Inp, Oval, Oidx, rows, N, ncores):
        _tle.raw.call(mode_f32_lm, (Inp, Oval, Oidx, rows, N, ncores))

    @triton.jit(do_not_specialize=["rows", "N", "ncores"])
    def _mode_lm2_i32_w(Inp, Oval, Oidx, rows, N, ncores):
        _tle.raw.call(mode_i32_lm2, (Inp, Oval, Oidx, rows, N, ncores))

    @triton.jit(do_not_specialize=["rows", "N", "ncores"])
    def _mode_lm2_f32_w(Inp, Oval, Oidx, rows, N, ncores):
        _tle.raw.call(mode_f32_lm2, (Inp, Oval, Oidx, rows, N, ncores))

    @triton.jit(do_not_specialize=["rows", "N", "nclusters"])
    def _mode_radix_i32_w(Inp, Oval, Oidx, BufA, BufB, rows, N, nclusters):
        _tle.raw.call(
            mode_i32_radix, (Inp, Oval, Oidx, BufA, BufB, rows, N, nclusters)
        )

    @triton.jit(do_not_specialize=["rows", "N", "nclusters"])
    def _mode_radix_f32_w(Inp, Oval, Oidx, BufA, BufB, rows, N, nclusters):
        _tle.raw.call(
            mode_f32_radix, (Inp, Oval, Oidx, BufA, BufB, rows, N, nclusters)
        )

    _RAW_LM = {torch.int32: _mode_lm_i32_w, torch.float32: _mode_lm_f32_w}
    _RAW_LM2 = {torch.int32: _mode_lm2_i32_w, torch.float32: _mode_lm2_f32_w}
    _RAW_CLUSTER = {
        torch.int32: _mode_radix_i32_w,
        torch.float32: _mode_radix_f32_w,
    }
    _HAS_RAW_RADIX = True
except Exception as _e:  # pragma: no cover - defensive
    logger.debug("mode radix raw payload unavailable: %s", _e)
    _RAW_LM = {}
    _RAW_LM2 = {}
    _RAW_CLUSTER = {}

# Tier routing thresholds.
_MODE_LM_MAX_N = 512  # per-core in-LM radix caps at N<=512 (LM budget).
# 2-chunk per-core in-LM radix + merge extends the LM path to N<=1024 by
# sorting two 512-halves separately (sa+sb+tmp+cnt=7168B < 8KB) and
# two-pointer merge-scanning for the mode; no global scatter.
_MODE_LM2_MAX_N = 1024
_MODE_CLUSTER_MAX_M = 32  # cluster radix (few-row) uses g=min(8,M).
# Many-row shapes still win with cluster radix when N is large enough to
# amortize per-row cluster overhead (probe: (1024,65536) 2294ms->281ms=8x,
# (4096,4096) 556ms->316ms; (1024,1024) N=1024 no help -> stays shared).
_MODE_CLUSTER_MANYROW_MIN_N = 4096


def _mode_lm_run(rows, flat_values, flat_indices, M, N, wrapper):
    """Per-core in-LM radix sort (N<=512).  Writes mode value bits + int32
    index, then widens the index to int64.  Returns True on success."""
    g = min(12 * 64, max(1, M))
    oval = torch.empty(M, dtype=rows.dtype, device=rows.device)
    oidx32 = torch.empty(M, dtype=torch.int32, device=rows.device)
    grid = min(12, (M + 63) // 64)
    with torch_device_fn.device(rows.device):
        wrapper[(grid,)](rows, oval, oidx32, M, N, g)
    flat_values.copy_(oval)
    flat_indices.copy_(oidx32.to(torch.int64))
    return True


def _mode_lm2_run(rows, flat_values, flat_indices, M, N, wrapper):
    """2-chunk per-core in-LM radix + merge (512<N<=1024).  Same launch shape
    as _mode_lm_run; the kernel sorts two 512-halves in LM and merge-scans for
    the mode with no global scatter.  Returns True on success."""
    g = min(12 * 64, max(1, M))
    oval = torch.empty(M, dtype=rows.dtype, device=rows.device)
    oidx32 = torch.empty(M, dtype=torch.int32, device=rows.device)
    grid = min(12, (M + 63) // 64)
    with torch_device_fn.device(rows.device):
        wrapper[(grid,)](rows, oval, oidx32, M, N, g)
    flat_values.copy_(oval)
    flat_indices.copy_(oidx32.to(torch.int64))
    return True


def _mode_cluster_run(rows, flat_values, flat_indices, M, N, wrapper):
    """Cluster-cooperative radix sort (large-N).  Few-row (M<=32) spreads over
    g=min(8,M) clusters; many-row large-N uses g=12 (probe: g=12 beats g=8 on
    (1024,65536), 281ms vs 390ms).  Allocates M*N ping-pong scratch buffers.
    Returns True on success."""
    if M > _MODE_CLUSTER_MAX_M:
        g = 12
    else:
        g = min(8, max(1, M))
    oval = torch.empty(M, dtype=rows.dtype, device=rows.device)
    oidx32 = torch.empty(M, dtype=torch.int32, device=rows.device)
    bufA = torch.empty(M * N, dtype=rows.dtype, device=rows.device)
    bufB = torch.empty(M * N, dtype=rows.dtype, device=rows.device)
    with torch_device_fn.device(rows.device):
        wrapper[(g,)](rows, oval, oidx32, bufA, bufB, M, N, g)
    flat_values.copy_(oval)
    flat_indices.copy_(oidx32.to(torch.int64))
    return True


def _mode_hist_run(rows, flat_values, flat_indices, M, N, wrapper):
    """Run the 2-pass half-histogram tle.raw kernel.  `rows` is a contiguous
    (M, N) view.  Writes mode values (as native dtype bits) into flat_values
    and an int32 occurrence index into a scratch buffer, then widens indices
    to int64 (flat_indices).  Returns True on success."""
    g = min(12, max(1, M))
    oidx32 = torch.empty(M, dtype=torch.int32, device=rows.device)
    with torch_device_fn.device(rows.device):
        wrapper[(g,)](rows, flat_values, oidx32, M, N, g)
    flat_indices.copy_(oidx32.to(torch.int64))
    return True



@libentry()
@triton.jit(do_not_specialize=["columns"])
def _mode_sorted_rows_kernel(
    sorted_values,
    sorted_indices,
    output_values,
    output_indices,
    columns,
):
    row = tl.program_id(0)
    row_offset = row * columns
    current_value = tl.load(sorted_values + row_offset)
    current_index = tl.load(sorted_indices + row_offset)
    best_value = current_value
    best_index = current_index
    current_count = 1
    best_count = 1

    column = 1
    while column < columns:
        value = tl.load(sorted_values + row_offset + column)
        index = tl.load(sorted_indices + row_offset + column)
        same_value = value == current_value
        current_count = tl.where(same_value, current_count + 1, 1)
        current_value = tl.where(same_value, current_value, value)
        current_index = index
        better = current_count > best_count
        best_count = tl.where(better, current_count, best_count)
        best_value = tl.where(better, current_value, best_value)
        best_index = tl.where(better, current_index, best_index)
        column += 1

    tl.store(output_values + row, best_value)
    tl.store(output_indices + row, best_index)


@libentry()
@triton.jit
def _mode_fill_first(x_ptr, out_v_ptr, out_i_ptr, RS: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    v = tl.load(x_ptr + pid * RS)
    tl.store(out_v_ptr + pid, v)
    tl.store(out_i_ptr + pid, 0)


# ---------------------------------------------------------------------------
# Vectorized parallel run-length scan.
#
# The default `_mode_sorted_rows_kernel` scans each sorted row with a scalar
# `tl.load(ptr + scalar_column)` inside a data-dependent while loop.  On XPU
# such a scalar-pointer load never vectorizes (`getVectorSize` returns 1 for a
# non-tensor pointer) -> every element is a 2-byte serialized GM2LM, N times.
#
# Instead we split each sorted row into NC contiguous BLOCK-wide chunks
# (grid=(M, NC)) and summarise each chunk's run structure with *block* loads
# `tl.load(ptr + tl.arange(0, BLOCK))` -> a coalesced 16-byte/core GM2LM plus a
# single `tl.associative_scan` (which lowers on XPU only outside loops).  A
# cheap per-row merge (grid=(M,)) stitches the chunk summaries.  The chunk
# kernel emits only int32 run positions/lengths (keeping per-program register
# pressure low enough for BLOCK=256); the merge scalar-loads the actual
# values/indices at those positions (only ~NC scalar loads per row).
# Bit-exact with the serial scan: smallest value at max count, index = last
# original index of that value's run (guaranteed by the stable radix sort).
# ---------------------------------------------------------------------------

_MODE_BLOCK = 256


@triton.jit
def _amax_combine(a, b):
    return tl.maximum(a, b)


@libentry()
@triton.jit(do_not_specialize=["N", "C"])
def _mode_vchunk_kernel(
    sv,
    o_hl,
    o_he,
    o_tl,
    o_ts,
    o_tll,
    o_bc,
    o_bp,
    N,
    C,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    base = row * N
    cstart = c * C
    j = tl.arange(0, BLOCK)
    gj = cstart + j
    valid = (j < C) & (gj < N)
    L = tl.sum(valid.to(tl.int32))

    v = tl.load(sv + base + gj, mask=valid, other=0)
    prev_off = tl.maximum(base + gj - 1, 0)
    vprev = tl.load(sv + prev_off, mask=valid & (j >= 1), other=0)

    boundary = (j == 0) | (v != vprev)
    cand = tl.where(boundary, j, -1)
    startpos = tl.associative_scan(cand, axis=0, combine_fn=_amax_combine)
    runlen = tl.where(valid, j - startpos + 1, -1)

    next_off = tl.minimum(base + gj + 1, N * tl.num_programs(0) - 1)
    vnext = tl.load(sv + next_off, mask=valid & (gj + 1 < N), other=0)
    is_last = j == (L - 1)
    run_end = valid & (is_last | (v != vnext))

    he = tl.max(tl.where(valid & (startpos == 0), j, -1))
    tsp = tl.max(tl.where(is_last, startpos, -1))
    interior = run_end & (startpos > 0) & (~is_last)
    bc = tl.max(tl.where(interior, runlen, 0))
    wp = tl.min(tl.where(interior & (runlen == bc) & (bc > 0), j, N))

    off = row * tl.num_programs(1) + c
    tl.store(o_hl + off, he + 1)
    tl.store(o_he + off, cstart + he)
    tl.store(o_tl + off, tl.where(L > 0, (L - 1) - tsp + 1, 0))
    tl.store(o_ts + off, cstart + tsp)
    tl.store(o_tll + off, cstart + L - 1)
    tl.store(o_bc + off, bc)
    tl.store(o_bp + off, tl.where(wp < C, cstart + wp, -1))


@libentry()
@triton.jit(do_not_specialize=["N", "C"])
def _mode_merge_kernel(
    sv,
    si,
    i_hl,
    i_he,
    i_tl,
    i_ts,
    i_tll,
    i_bc,
    i_bp,
    out_v,
    out_i,
    N,
    C,
    NC: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N
    rb = row * NC

    he0 = tl.load(i_he + rb)
    best_c = 0
    best_v = tl.load(sv + base)
    best_l = tl.load(si + base + he0)

    carry_act = 0
    carry_v = best_v
    carry_len = 0
    carry_last = best_l

    for c in range(NC):
        s = c * C
        csize = tl.minimum(C, N - s)
        off = rb + c
        hl = tl.load(i_hl + off)
        he = tl.load(i_he + off)
        tln = tl.load(i_tl + off)
        ts = tl.load(i_ts + off)
        tll = tl.load(i_tll + off)
        bc = tl.load(i_bc + off)
        bp = tl.load(i_bp + off)
        hv = tl.load(sv + base + s)
        hli = tl.load(si + base + he)
        tv = tl.load(sv + base + ts)
        tli = tl.load(si + base + tll)
        bp_safe = tl.maximum(bp, 0)
        bv = tl.load(sv + base + bp_safe)
        bli = tl.load(si + base + bp_safe)

        single = hl == csize
        compat = (carry_act == 1) & (carry_v == hv)

        # event 1: close carry as-is (carry active, incompatible head)
        g1 = (carry_act == 1) & (compat == 0)
        c1 = tl.where(g1, carry_len, 0)
        t1 = (c1 > 0) & ((c1 > best_c) | ((c1 == best_c) & (carry_v < best_v)))
        best_c = tl.where(t1, c1, best_c)
        best_v = tl.where(t1, carry_v, best_v)
        best_l = tl.where(t1, carry_last, best_l)

        # event 2: close carry extended by compatible head (multi-run chunk)
        g2 = compat & (single == 0)
        c2 = tl.where(g2, carry_len + hl, 0)
        t2 = (c2 > 0) & ((c2 > best_c) | ((c2 == best_c) & (carry_v < best_v)))
        best_c = tl.where(t2, c2, best_c)
        best_v = tl.where(t2, carry_v, best_v)
        best_l = tl.where(t2, hli, best_l)

        # event 3: head is a complete run (incompatible, multi-run chunk)
        g3 = (compat == 0) & (single == 0)
        c3 = tl.where(g3, hl, 0)
        t3 = (c3 > 0) & ((c3 > best_c) | ((c3 == best_c) & (hv < best_v)))
        best_c = tl.where(t3, c3, best_c)
        best_v = tl.where(t3, hv, best_v)
        best_l = tl.where(t3, hli, best_l)

        # event 4: interior best (multi-run chunk)
        c4 = tl.where(single == 0, bc, 0)
        t4 = (c4 > 0) & ((c4 > best_c) | ((c4 == best_c) & (bv < best_v)))
        best_c = tl.where(t4, c4, best_c)
        best_v = tl.where(t4, bv, best_v)
        best_l = tl.where(t4, bli, best_l)

        # update carry (always ends open)
        cs = compat & single
        carry_v = tl.where(cs, carry_v, tl.where(single, hv, tv))
        carry_len = tl.where(cs, carry_len + hl, tl.where(single, hl, tln))
        carry_last = tl.where(cs, hli, tl.where(single, hli, tli))
        carry_act = 1

    tf = (carry_len > 0) & (
        (carry_len > best_c) | ((carry_len == best_c) & (carry_v < best_v))
    )
    best_c = tl.where(tf, carry_len, best_c)
    best_v = tl.where(tf, carry_v, best_v)
    best_l = tl.where(tf, carry_last, best_l)

    tl.store(out_v + row, best_v)
    tl.store(out_i + row, best_l)


def _mode_scan_vectorized(sorted_v, sorted_i, flat_values, flat_indices, M, N):
    """Vectorized chunked run-length scan.  Returns True if it ran, else False
    (caller falls back to the serial per-row kernel)."""
    C = _MODE_BLOCK
    NC = (N + C - 1) // C

    dev = sorted_v.device

    def _i32():
        return torch.empty((M, NC), dtype=torch.int32, device=dev)

    o_hl = _i32()
    o_he = _i32()
    o_tl = _i32()
    o_ts = _i32()
    o_tll = _i32()
    o_bc = _i32()
    o_bp = _i32()

    with torch_device_fn.device(dev):
        _mode_vchunk_kernel[(M, NC)](
            sorted_v, o_hl, o_he, o_tl, o_ts, o_tll, o_bc, o_bp, N, C, BLOCK=C
        )
        _mode_merge_kernel[(M,)](
            sorted_v,
            sorted_i,
            o_hl,
            o_he,
            o_tl,
            o_ts,
            o_tll,
            o_bc,
            o_bp,
            flat_values,
            flat_indices,
            N,
            C,
            NC=NC,
        )
    return True


def _mode_radix_fallback(rows, flat_values, flat_indices, M, N, device):
    """Shared radix_sort_low_mem + vectorized RLE scan path (the pre-existing
    default).  Used for many-row large-N fp32/int32 shapes and as the fallback
    when a tle.raw tier fails."""
    sorted_v, sorted_i = radix_sort_low_mem(rows, 4, False)
    if sorted_v.dtype == torch.bfloat16:
        sorted_v = sorted_v.to(torch.float32)
    with torch_device_fn.device(device):
        if not _mode_scan_vectorized(
            sorted_v, sorted_i, flat_values, flat_indices, M, N
        ):
            _mode_sorted_rows_kernel[(M,)](
                sorted_v, sorted_i, flat_values, flat_indices, N
            )


def _normalize_dim(dim, ndim):
    if ndim == 0:
        if dim in (0, -1):
            return 0
    elif -ndim <= dim < ndim:
        return dim % ndim
    raise IndexError(
        f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
    )


def _mode_impl(inp, dim, keepdim):
    if inp.ndim == 0:
        values = inp.clone()
        indices = torch.zeros((), dtype=torch.long, device=inp.device)
        return ModeResult(values=values, indices=indices)

    dim = _normalize_dim(dim, inp.ndim)
    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1

    if N == 0:
        if M != 0:
            raise IndexError(
                f"mode(): Expected reduction dim {dim} to have non-zero size."
            )
        values = torch.empty(keepdim_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(keepdim_shape, dtype=torch.long, device=inp.device)
        if not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return ModeResult(values=values, indices=indices)

    values = torch.empty(keepdim_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(keepdim_shape, dtype=torch.long, device=inp.device)

    if M == 0:
        if not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return ModeResult(values=values, indices=indices)

    flat_values = values.reshape(M)
    flat_indices = indices.reshape(M)

    if dim != inp.ndim - 1:
        view = torch.movedim(inp, dim, -1)
        rows = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
        if not tle_copy(view, rows):
            torch.ops.aten._copy_from(view, rows, False)
    else:
        rows = inp.reshape(M, N)

    if N == 1:
        with torch_device_fn.device(inp.device):
            _mode_fill_first[(M,)](rows, flat_values, flat_indices, RS=N, N=N)
    elif _HAS_RAW_HIST and inp.dtype in _RAW_HIST and rows.is_contiguous():
        # fp16/bf16/int16: O(N) histogram mode (breaks the sort ceiling).
        try:
            _mode_hist_run(
                rows, flat_values, flat_indices, M, N, _RAW_HIST[inp.dtype]
            )
        except Exception as e:  # pragma: no cover - fall back to radix path
            logger.debug("mode histogram fell back to radix path: %s", e)
            _mode_radix_fallback(
                rows, flat_values, flat_indices, M, N, inp.device
            )
    elif (
        _HAS_RAW_RADIX
        and inp.dtype in _RAW_LM
        and rows.is_contiguous()
        and N <= _MODE_LM_MAX_N
    ):
        # fp32/int32, N<=512: per-core in-LM radix (no global scatter).
        try:
            _mode_lm_run(
                rows, flat_values, flat_indices, M, N, _RAW_LM[inp.dtype]
            )
        except Exception as e:  # pragma: no cover
            logger.debug("mode LM radix fell back: %s", e)
            _mode_radix_fallback(
                rows, flat_values, flat_indices, M, N, inp.device
            )
    elif (
        _HAS_RAW_RADIX
        and inp.dtype in _RAW_LM2
        and rows.is_contiguous()
        and _MODE_LM_MAX_N < N <= _MODE_LM2_MAX_N
    ):
        # fp32/int32, 512<N<=1024: 2-chunk per-core in-LM radix + merge (no
        # global scatter; sorts two 512-halves in LM and merge-scans).
        try:
            _mode_lm2_run(
                rows, flat_values, flat_indices, M, N, _RAW_LM2[inp.dtype]
            )
        except Exception as e:  # pragma: no cover
            logger.debug("mode LM2 radix fell back: %s", e)
            _mode_radix_fallback(
                rows, flat_values, flat_indices, M, N, inp.device
            )
    elif (
        _HAS_RAW_RADIX
        and inp.dtype in _RAW_CLUSTER
        and rows.is_contiguous()
        and (M <= _MODE_CLUSTER_MAX_M or N >= _MODE_CLUSTER_MANYROW_MIN_N)
    ):
        # fp32/int32, large-N: cluster-cooperative radix.  Few-row (M<=32) or
        # many-row with N>=4096 (per-element scatter split over 64 cores beats
        # the shared radix_sort global-scatter bandwidth wall for big N).
        try:
            _mode_cluster_run(
                rows, flat_values, flat_indices, M, N, _RAW_CLUSTER[inp.dtype]
            )
        except Exception as e:  # pragma: no cover
            logger.debug("mode cluster radix fell back: %s", e)
            _mode_radix_fallback(
                rows, flat_values, flat_indices, M, N, inp.device
            )
    else:
        _mode_radix_fallback(rows, flat_values, flat_indices, M, N, inp.device)

    if not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return ModeResult(values=values, indices=indices)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MODE")
    return _mode_impl(inp, dim, keepdim)
