import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2026-10-12 (attempt_09) — fixed-overhead + sparse-path rewrite
#
# Measured on XPU6 (do_bench median, same-process A/B):
#   * `_device_int_tensor` used to launch ONE `_metadata_kernel[(1,)]` per value
#     => ndim launches (~12.4us each) purely to build a tiny shape/stride tensor.
#     Now it is a single host->device copy.
#   * The old `_is_dense` path ran a FULL `cumsum` over the whole flattened input
#     just to obtain the count (832us @2.56M elems); the count kernels below need
#     no global scan (block counts + one tiny exclusive scan).
#   * The old sparse path materialised `inp != 0` with a separate torch `ne`
#     kernel and then recomputed a global `cumsum` prefix.  Both passes are gone:
#     the compaction kernel loads the raw input once and derives the in-block
#     rank with `tl.cumsum`, using the block prefix produced by the count pass.
#   * `inp != 0` (torch `ne`) on a very large tensor hits a compiler stack
#     overflow ("Failed to tune buffer size.") on this backend, which made
#     `flag_gems.nonzero` crash outright for 655M-element fp16 inputs.  The mask
#     is now produced inside our own kernels, so that path is gone.
# ---------------------------------------------------------------------------

_DENSE_TILE_CAP = 8192

# Shared contiguous-chunk layout for the count / scan / compaction kernels.
_NZ_TILE = 4096
_NZ_GRID_CAP = 4096
_NZ_K_CAP = 256
_DELIN_TILE = 2048

try:  # `tle.raw` payloads are only available on the XPU triton fork.
    import triton.experimental.tle as _tle
except Exception:  # pragma: no cover - non-XPU triton
    _tle = None

# ---------------------------------------------------------------------------
# attempt_10: tle.raw LM-compaction payload.
#
# The whole cost of the previous design was *discrete* global writes: both the
# sparse compaction (`ws[rank] = flat`) and the de-linearisation
# (`out[r*NDIM+D] = coord`, stride NDIM) lower to per-element scalar global
# stores, which are ~18x slower than a contiguous DMA on this backend.
#
# This payload keeps every global access contiguous:
#   * one GM2LM per input tile,
#   * the in-block rank + the de-linearised coordinates are produced *in LM*,
#   * one contiguous LM2GM per OBR output rows (rows of `out` are adjacent in
#     memory, so a worker owning a contiguous flat range owns a contiguous
#     row range of `out`).
# The per-worker output row base comes from a block-level exclusive scan on the
# host side (same structure as masked_scatter's `base[]`).
# ---------------------------------------------------------------------------

_TLE_GRID = 8  # 8 clusters; GRID=16/24 silently drops data on this backend.
_TLE_NW = 512  # host-chosen worker count, independent of core_num()
# (element bytes, is_float) -> entry-symbol stem.  One `.xpu` per stem lives in
# ../payload/src/ and the merged object ../payload/obj/nonzero.o holds every
# entry symbol; `_tle_stub` / `_tle_count_stub` resolve them by name at runtime.
# The table also fixes which (element size, floatness) pairs the payload covers.
_TLE_SYM = {
    (1, False): "1i",
    (2, False): "2i",
    (2, True): "2f",
    (4, False): "4i",
    (4, True): "4f",
    (8, False): "8i",
    (8, True): "8f",
}
# (element bytes, is_float) -> (C type, unsigned C type, nonzero bit mask)
_TLE_CTYPE = {
    (1, False): ("char", "unsigned char", "0xFFu"),
    (2, False): ("short", "unsigned short", "0xFFFFu"),
    (2, True): ("short", "unsigned short", "0x7FFFu"),
    (4, False): ("int", "unsigned int", "0xFFFFFFFFu"),
    (4, True): ("int", "unsigned int", "0x7FFFFFFFu"),
    (8, False): ("long", "unsigned long", "0xFFFFFFFFFFFFFFFFUL"),
    (8, True): ("long", "unsigned long", "0x7FFFFFFFFFFFFFFFUL"),
}

_TLE_CACHE = {}


# # `_nz_count_kernel`'s tail mask is unreliable on this backend: for a partial
# last block the `(w != 0) & (cols < n)` predicate is *not* applied (measured
# on (200,300) int16: partial[last] == 128 with only 96 valid lanes, i.e. the
# full tile is counted, and OOB lanes read live heap data).  The count is
# therefore produced by a `tle.raw` payload with exact C bounds instead
# (payload/src/nz_cnt_*.xpu -> payload/obj/nonzero.o, entry `nz_cnt_<tag>`).
def _tle_payload_obj():
    """Absolute path of the packed, secrecy-hardened payload object.

    The count and compaction entry symbols all live in this one merged object
    (per-entry `.o` from docs/xpu3/how_to_pack_payload/pack_payload.py, joined
    with `ld.lld -r`); in object= mode the loader resolves each stub by its
    entry-symbol name, so one .o holding every symbol works like the old
    per-kernel files.  Re-pack after editing any payload/src/*.xpu: the
    compile-cache key is the .o content digest, so a stale .o silently keeps
    the old code.
    """
    import os as _os

    return _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "payload",
        "obj",
        "nonzero.o",
    )


@libentry()
@triton.jit
def _nz_scan16_kernel(pcount, prefix, total, NW: tl.constexpr):
    """Exclusive scan over the payload's stride-16 padded per-worker counts.

    Deliberately mask-free: `pcount` always has exactly NW populated slots.
    """
    i = tl.arange(0, NW)
    c = tl.load(pcount + i * 16)
    tl.store(prefix + i, tl.cumsum(c, axis=0) - c)
    tl.store(total, tl.sum(c, axis=0).to(tl.int64))


def _tle_stub(key):
    """Build (and memoise) the `tle.raw` compaction kernel for one variant."""
    if key in _TLE_CACHE:
        return _TLE_CACHE[key]
    esize, is_float, ndim = key
    name = "nz_pack_%s_d%d" % (_TLE_SYM[(esize, is_float)], ndim)
    # Stub name == entry symbol, stub signature == the C++ ABI.  In object=
    # mode the merge-time signature check is skipped (the payload isn't in IR),
    # so a mismatched stub fails at runtime, not at compile -- keep them in
    # lockstep with the C++ in ../payload/src/.
    ns = {"tle": _tle, "triton": triton, "_OBJ": _tle_payload_obj()}
    exec(
        "@tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
        "def %s(out, inp, prefix, meta):\n"
        "    ...\n" % name,
        ns,
    )
    callee = ns[name]

    @triton.jit
    def _k(OUT, INP, PFX, META):
        _tle.raw.call(callee, (OUT, INP, PFX, META))

    _TLE_CACHE[key] = _k
    return _k


def _tle_count_stub(key):
    """Build (and memoise) the `tle.raw` per-worker count kernel."""
    ck = ("count",) + key
    if ck in _TLE_CACHE:
        return _TLE_CACHE[ck]
    esize, is_float = key
    name = "nz_cnt_%s" % _TLE_SYM[(esize, is_float)]
    ns = {"tle": _tle, "triton": triton, "_OBJ": _tle_payload_obj()}
    exec(
        "@tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
        "def %s(pcount, inp, meta):\n"
        "    ...\n" % name,
        ns,
    )
    callee = ns[name]

    @triton.jit
    def _k(PC, INP, META):
        _tle.raw.call(callee, (PC, INP, META))

    _TLE_CACHE[ck] = _k
    return _k


def nonzero_kernel_heur_block_size(args):
    return min(triton.next_power_of_2(triton.cdiv(args["n_elements"], 12)), 4096)


def _dense_block_size(n):
    if n <= 2048:
        return triton.next_power_of_2(n)
    return 2048


def _nz_plan(n_elements):
    """Pick a (TILE, K, GRID, CHUNK) contiguous-chunk layout.

    Every program owns ``CHUNK = TILE * K`` *consecutive* elements, so the
    per-program counts form a usable exclusive-scan prefix for compaction.
    """
    tile = _NZ_TILE
    k = 1
    while triton.cdiv(n_elements, tile * k) > _NZ_GRID_CAP and k < _NZ_K_CAP:
        k *= 2
    chunk = tile * k
    grid = triton.cdiv(n_elements, chunk)
    return tile, k, grid, chunk


@libentry()
@triton.jit
def _nz_count_kernel(inp, partial, n_elements, TILE: tl.constexpr, K: tl.constexpr):
    pid = ext.program_id(0)
    base = pid * (TILE * K)
    acc = tl.zeros([TILE], dtype=tl.int32)
    for k in range(K):
        cols = base + k * TILE + tl.arange(0, TILE)
        m = cols < n_elements
        w = tl.load(inp + cols, mask=m, other=0)
        acc += ((w != 0) & m).to(tl.int32)
    tl.store(partial + pid, tl.sum(acc, axis=0))


@libentry()
@triton.jit
def _nz_scan_kernel(partial, prefix, total, nblk: tl.constexpr, PBLOCK: tl.constexpr):
    i = tl.arange(0, PBLOCK)
    m = i < nblk
    c = tl.load(partial + i, mask=m, other=0)
    tl.store(prefix + i, tl.cumsum(c, axis=0) - c, mask=m)
    tl.store(total, tl.sum(c, axis=0).to(tl.int64))


@libentry()
@triton.jit
def _nz_write_kernel(
    inp, prefix, ws, n_elements, size, TILE: tl.constexpr, K: tl.constexpr
):
    """Compact the flat linear indices of the non-zero lanes into ``ws``.

    The store is *unmasked*: non-selected lanes are redirected to a coalesced
    dummy region ``[size, size + CHUNK)`` of ``ws`` (a masked / strided scatter
    is catastrophically slow on this backend).
    """
    pid = ext.program_id(0)
    base = pid * (TILE * K)
    p = tl.load(prefix + pid)
    run = tl.zeros([1], dtype=tl.int32)
    for k in range(K):
        cols = base + k * TILE + tl.arange(0, TILE)
        m = cols < n_elements
        w = tl.load(inp + cols, mask=m, other=0)
        nz = (w != 0) & m
        loc = tl.cumsum(nz.to(tl.int32), axis=0) - 1
        rank = p + run + loc
        dest = tl.where(nz, rank, size + (cols - base)).to(tl.int64)
        tl.store(ws + dest, cols.to(tl.int64))
        run = run + tl.sum(nz.to(tl.int32), axis=0, keep_dims=True)


@libentry()
@triton.jit
def _nz_delin_dim_kernel(
    ws, out, nz, T, SH, D: tl.constexpr, NDIM: tl.constexpr, BLOCK: tl.constexpr
):
    """Dense de-linearisation for one dimension: out[r, D] = (ws[r] // T) % SH.

    One launch per dimension on purpose: a scalar `tl.load(strides + d)` inside
    a `for d in range(ndim)` loop mis-lowers on this backend (the loaded value is
    folded onto the d=0 one, so every dimension but the first stores garbage),
    while the same arithmetic with *scalar kernel arguments* is exact.
    """
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < nz
    lin = tl.load(ws + r, mask=m, other=0)
    tl.store(out + r * NDIM + D, (lin // T) % SH, mask=m)


@libentry()
@triton.jit
def nonzero_dense_flat_kernel(
    out,
    n_out,
    strides,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    j = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = j < n_out
    i = j // ndim
    d = j % ndim
    stride_d = tl.load(strides + d)
    shape_d = tl.load(shape + d)
    coord = (i // stride_d) % shape_d
    tl.store(out + j, coord, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
def nonzero_dense_flat_args_kernel(
    out,
    n_out,
    ndim: tl.constexpr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    j = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = j < n_out
    i = j // ndim
    d = j % ndim
    stride_d = tl.where(
        d == 0,
        s1 * s2 * s3 * s4 * s5 * s6 * s7,
        tl.where(
            d == 1,
            s2 * s3 * s4 * s5 * s6 * s7,
            tl.where(
                d == 2,
                s3 * s4 * s5 * s6 * s7,
                tl.where(
                    d == 3,
                    s4 * s5 * s6 * s7,
                    tl.where(
                        d == 4,
                        s5 * s6 * s7,
                        tl.where(d == 5, s6 * s7, tl.where(d == 6, s7, 1)),
                    ),
                ),
            ),
        ),
    )
    shape_d = tl.where(
        d == 0,
        s0,
        tl.where(
            d == 1,
            s1,
            tl.where(
                d == 2,
                s2,
                tl.where(
                    d == 3,
                    s3,
                    tl.where(
                        d == 4,
                        s4,
                        tl.where(d == 5, s5, tl.where(d == 6, s6, s7)),
                    ),
                ),
            ),
        ),
    )
    coord = (i // stride_d) % shape_d
    tl.store(out + j, coord, mask=mask)


def _nz_mask_source(inp, n_elements):
    """Flat tensor whose `!= 0` mask matches ``inp != 0`` (complex-aware)."""
    view = inp.view(n_elements)
    if inp.dtype.is_complex:
        real = torch.view_as_real(view)
        return (real[..., 0] != 0) | (real[..., 1] != 0)
    return view


def _count_plan(inp, n_elements):
    """Return ``(num_nonzeros, plan)`` where plan feeds the compaction pass.

    One count kernel + one tiny scan kernel + a single device->host sync.
    """
    src = _nz_mask_source(inp, n_elements)
    tile, k, grid, chunk = _nz_plan(n_elements)
    partial = torch.empty(grid, dtype=torch.int32, device=inp.device)
    prefix = torch.empty(grid, dtype=torch.int32, device=inp.device)
    total = torch.empty((), dtype=torch.int64, device=inp.device)
    pblock = triton.next_power_of_2(grid)
    with torch_device_fn.device(inp.device):
        _nz_count_kernel[(grid,)](src, partial, n_elements, TILE=tile, K=k)
        _nz_scan_kernel[(1,)](partial, prefix, total, nblk=grid, PBLOCK=pblock)
    return int(total.item()), (tile, k, grid, chunk, prefix)


def _count_nonzero(inp, n_elements):
    return _count_plan(inp, n_elements)[0]


def _unbind_views(out):
    """``unbind(out, dim=1)`` as zero-copy ``as_strided`` views.

    ``unbind`` has no device kernel on XPU and falls back to the ATen
    composite implementation (forbidden); ``torch.as_strided`` is metadata-only
    and unregistered, so it never re-dispatches. ``out`` is row-major
    [N, ndim], so column i is the view shape (N,), stride (ndim,), offset i.
    """
    n, ndim = out.shape
    return [torch.as_strided(out, (n,), (ndim,), storage_offset=i) for i in range(ndim)]


_TLE_VIEW_DTYPE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


_TLE_OBJ_OK = None  # None = not yet probed


def _tle_obj_supported():
    """One-time probe: does the installed triton support the `tle.raw` object=
    loader?  That loader only exists on newer tle builds; on older environments
    building the stub raises (e.g. "RawJITFunction.__init__() got an unexpected
    keyword argument 'object'").  Guard the whole declaration so an unsupported
    tle leaves the tle path disabled and the generic kernels run instead -- a
    failure here must never crash the op."""
    global _TLE_OBJ_OK
    if _TLE_OBJ_OK is None:
        try:
            _tle_stub((4, True, 1))
            _TLE_OBJ_OK = True
        except Exception as _e:  # noqa: BLE001 - any failure means "not usable"
            logger.debug("GEMS_KUNLUNXIN NONZERO tle path deferred: %r", _e)
            _TLE_OBJ_OK = False
    return _TLE_OBJ_OK


def _tle_chunk(n_elements):
    ch = max(64, triton.next_power_of_2(triton.cdiv(n_elements, _TLE_NW)))
    return ch


def _tle_usable(src, inp_ndim, n_elements):
    return (
        _tle is not None
        and _tle_obj_supported()
        and 1 <= inp_ndim <= 8
        and n_elements < 2**31
        and n_elements * inp_ndim < 2**31
        and src.element_size() in _TLE_VIEW_DTYPE
        and not src.dtype.is_complex
    )


def _tle_fast_usable(inp, inp_ndim, n_elements):
    """`_tle_usable` on the raw input, i.e. without materialising a flat view.

    Complex inputs are excluded on purpose: their non-zero predicate is
    ``real != 0 | imag != 0``, which needs the ``_nz_mask_source`` rewrite.
    """
    return (
        _tle is not None
        and _tle_obj_supported()
        and 1 <= inp_ndim <= 8
        and n_elements < 2**31
        and n_elements * inp_ndim < 2**31
        and not inp.dtype.is_complex
        and inp.element_size() in _TLE_VIEW_DTYPE
    )


def _tle_prepare(src, shape, n_elements):
    ch = _tle_chunk(n_elements)
    ndim = len(shape)
    md = [n_elements, ch, ndim] + list(shape) + [1] * (8 - ndim) + [0] * 5
    meta = torch.tensor(md, dtype=torch.int64, device=src.device)
    esize = src.element_size()
    view = src.view(_TLE_VIEW_DTYPE[esize])
    return meta, view, (esize, src.is_floating_point())


def _tle_count(view, meta, key, device):
    """Per-worker counts + exclusive output-row base, on the payload partition."""
    pcount, prefix, total = _tle_scratch(device)
    with torch_device_fn.device(device):
        _tle_count_stub(key)[(_TLE_GRID,)](pcount, view, meta)
        _nz_scan16_kernel[(1,)](pcount, prefix, total, NW=_TLE_NW)
    return int(total[0].item()), prefix


# ---------------------------------------------------------------------------
# attempt_11: the whole remaining cost of the small shapes is *fixed* host-side
# dispatch, measured per piece on XPU6 (same-process do_bench, us/call):
#   torch.tensor(meta) H2D 19.7 | dev.copy_(host) H2D 11.1 | empty launch 10.0
#   nz_cnt launch 11.3 | scan launch 18.5 | total.item() 16.2 | alloc 3.9 each
# So the three one-off `torch.empty` scratch buffers (32KB pcount + 2KB prefix
# + 8B total) cost ~11.6us per call and the 128-byte meta upload ~20us, i.e.
# ~20% of the 154us floor, for values that almost never change between calls.
# Both are now memoised: the scratch is per device (its contents are fully
# rewritten by every count pass -- all NW slots are always stored) and the meta
# tensor is keyed by (device, n, chunk, shape) so a repeated call on the same
# shape uploads nothing at all.
# ---------------------------------------------------------------------------
_TLE_SCRATCH = {}
_TLE_META_CACHE = {}
_TLE_META_CACHE_CAP = 64
# The payload's tail flush is `LM2GM(ob, out + wbase*NDIM, cnt*NDIM*8)`, i.e. a
# sub-64-byte DMA whenever `cnt*NDIM*8 % 64 != 0`.  The DMA engine completes
# such a write at its own granularity, so the last worker can touch up to a few
# dozen bytes *past* the final output row.  Until attempt_11 that overrun
# happened to land on the (freshly allocated, already dead) count scratch; with
# a reused scratch it landed on live tensors instead and corrupted the *input*
# of the next op -- `(33,1023) bool` returned the right rows while a following
# `x.cpu()` saw extra non-zeros.  The output buffer is therefore over-allocated
# by a 512-byte guard and returned as a view of the exact [nnz, ndim] block.
_TLE_OUT_GUARD = 64  # int64 slots

# attempt_12 lever C: for small inputs, over-allocate `out` to the worst case
# [n_elements, ndim] (nnz <= n_elements always) so the host-side `total.item()`
# device sync can be *deferred past the pack launch*.  Baseline order is
# count -> scan -> item() SYNC -> alloc -> pack, where the mid-pipeline sync
# forces the host to wait for the count+scan device work before it can even
# dispatch pack (attempt_11 measured ~16.4us for the sync plus ~44us of
# serialisation residual because `.item()` breaks the launch pipeline).  With a
# pre-sized buffer the order becomes count -> scan -> alloc -> pack -> item()
# SYNC -> narrow, i.e. all three launches are issued back-to-back and the only
# sync lands at the very end.  Gated by bytes so the large device-bound shapes
# (e.g. (10000,256)=41MB, (10000,65536)) keep the exact-size path.
_TLE_OVERALLOC_BYTES = 4 * 1024 * 1024


def _tle_alloc_out(nnz, ndim, device):
    return torch.empty(nnz, ndim, dtype=torch.int64, device=device)


def _tle_scratch(device):
    """Count scratch.  **Freshly allocated on purpose** -- see note above.

    Reusing one persistent scratch per device is 11.6us/call cheaper, and it is
    exactly what attempt_11 tried first, but it makes the payload's write
    overrun land on *live* tensors: `(33,1023) bool` then returns the correct
    rows while the input it was given comes back with ~770 extra non-zero
    bytes (reproduced with `NZ11_FRESH_SCRATCH=0`).  The three allocations are
    what currently absorbs that overrun, so they stay until the payload's tail
    `LM2GM` is made 64-byte aligned.
    """
    idx = device.index if device.index is not None else 0
    s = _TLE_SCRATCH.get(idx)
    if s is None:
        s = (
            torch.empty(_TLE_NW * 16 + 1024, dtype=torch.int32, device=device),
            torch.empty(_TLE_NW + 1024, dtype=torch.int32, device=device),
            torch.empty(512, dtype=torch.int64, device=device),
        )
        _TLE_SCRATCH[idx] = s
    return s


def _tle_meta(device, shape, n_elements, ch):
    idx = device.index if device.index is not None else 0
    key = (idx, n_elements, ch, shape)
    meta = _TLE_META_CACHE.get(key)
    if meta is None:
        ndim = len(shape)
        md = [n_elements, ch, ndim] + list(shape) + [1] * (8 - ndim) + [0] * 5
        meta = torch.tensor(md, dtype=torch.int64, device=device)
        if len(_TLE_META_CACHE) >= _TLE_META_CACHE_CAP:
            _TLE_META_CACHE.clear()
        _TLE_META_CACHE[key] = meta
    return meta


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS_KUNLUNXIN NONZERO")

    inp_ndim = inp.ndim
    n_elements = inp.numel()

    if n_elements == 0:
        out = torch.empty(
            (0, inp_ndim) if inp_ndim else (0, 0), dtype=torch.int64, device=inp.device
        )
        if as_tuple:
            return _unbind_views(out) if inp_ndim else ()
        return out

    inp = inp.contiguous()

    # Fast path: hand-written `tle.raw` payload kernels.  These rely on tle
    # features that older/other tle builds may not expose (the object=
    # loader); such a mismatch can raise at stub-build or compile time
    # instead of returning False.  Guard the whole attempt so any failure
    # cleanly falls through to the generic Triton kernels below (a full
    # recompute of `out`, so a partial fast-path write is harmless).
    try:
        if _tle_fast_usable(inp, inp_ndim, n_elements):
            esize = inp.element_size()
            ch = _tle_chunk(n_elements)
            meta = _tle_meta(inp.device, tuple(inp.shape), n_elements, ch)
            view = inp.view(_TLE_VIEW_DTYPE[esize])
            key = (esize, inp.is_floating_point())
            pcount, prefix, total = _tle_scratch(inp.device)
            if n_elements * inp_ndim * 8 <= _TLE_OVERALLOC_BYTES:
                # Lever C: pre-size `out` to the worst case and defer the sync.
                with torch_device_fn.device(inp.device):
                    _tle_count_stub(key)[(_TLE_GRID,)](pcount, view, meta)
                    _nz_scan16_kernel[(1,)](pcount, prefix, total, NW=_TLE_NW)
                    out_full = torch.empty(
                        n_elements, inp_ndim, dtype=torch.int64, device=inp.device
                    )
                    _tle_stub(key + (inp_ndim,))[(_TLE_GRID,)](
                        out_full, view, prefix, meta
                    )
                    nnz = int(total[0].item())
                out = out_full[:nnz]
                return _unbind_views(out) if as_tuple else out
            with torch_device_fn.device(inp.device):
                _tle_count_stub(key)[(_TLE_GRID,)](pcount, view, meta)
                _nz_scan16_kernel[(1,)](pcount, prefix, total, NW=_TLE_NW)
                nnz = int(total[0].item())
                out = _tle_alloc_out(nnz, inp_ndim, inp.device)
                if nnz:
                    _tle_stub(key + (inp_ndim,))[(_TLE_GRID,)](out, view, prefix, meta)
            return _unbind_views(out) if as_tuple else out

        src = _nz_mask_source(inp, n_elements)
        if _tle_usable(src, inp_ndim, n_elements):
            meta, view, key = _tle_prepare(src, inp.shape, n_elements)
            nnz, prefix = _tle_count(view, meta, key, inp.device)
            out = torch.empty(nnz, inp_ndim, dtype=torch.int64, device=inp.device)
            if nnz:
                kern = _tle_stub(key + (inp_ndim,))
                with torch_device_fn.device(inp.device):
                    kern[(_TLE_GRID,)](out, view, prefix, meta)
            return _unbind_views(out) if as_tuple else out

    except Exception as _e:  # pragma: no cover - fall through to the generic kernels
        logger.debug("GEMS_KUNLUNXIN NONZERO tle path deferred: %r", _e)

    num_nonzeros, plan = _count_plan(inp, n_elements)

    if inp_ndim >= 1 and num_nonzeros == n_elements and num_nonzeros * inp_ndim < 2**31:
        return _dense_result(inp, num_nonzeros, as_tuple)

    return _sparse_result(inp, inp_ndim, n_elements, num_nonzeros, as_tuple, plan=plan)


@libentry()
@triton.jit
def _nz_dense_coord_dim_kernel(
    out, nz, T, SH, D: tl.constexpr, NDIM: tl.constexpr, BLOCK: tl.constexpr
):
    """out[r, D] = (r // T) % SH for the dense case (row r == flat index r)."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < nz
    tl.store(out + r * NDIM + D, (r // T) % SH, mask=m)


def _dense_result(inp, num_nonzeros, as_tuple):
    inp_ndim = inp.ndim
    n_out = num_nonzeros * inp_ndim
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
    if n_out > 0:
        if inp_ndim <= 8:
            strides = [1] * inp_ndim
            for i in range(inp_ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * inp.shape[i + 1]
            block = min(
                _DENSE_TILE_CAP,
                max(64, triton.next_power_of_2(min(num_nonzeros, _DENSE_TILE_CAP))),
            )
            grid = (triton.cdiv(num_nonzeros, block),)
            # The coordinates are derived from an explicit flat-index buffer read
            # back from memory: computing `(pid*BLOCK + arange) // stride` directly
            # mis-lowers on this backend (the `pid*BLOCK` addend is dropped from the
            # division operand, so every program but the first emits wrong rows).
            # This is a pre-existing defect (the original `nonzero_dense_flat_args`
            # kernel has it too); feeding the index through memory is exact.
            ws = torch.arange(num_nonzeros, dtype=torch.int64, device=inp.device)
            with torch_device_fn.device(inp.device):
                for d in range(inp_ndim):
                    _nz_delin_dim_kernel[grid](
                        ws,
                        out,
                        num_nonzeros,
                        strides[d],
                        inp.shape[d],
                        D=d,
                        NDIM=inp_ndim,
                        BLOCK=block,
                    )
        else:
            strides_t = _row_major_strides(inp.shape, inp.device)
            shape_t = _device_int_tensor(inp.shape, torch.int64, inp.device)
            block = _dense_block_size(n_out)
            grid = (triton.cdiv(n_out, block),)
            with torch_device_fn.device(inp.device):
                nonzero_dense_flat_kernel[grid](
                    out,
                    n_out,
                    strides_t,
                    shape_t,
                    inp_ndim,
                    block,
                    isCloseUnrollControl=True,
                    is_use_mask_zero=True,
                )
    if as_tuple:
        return _unbind_views(out)
    return out


def _sparse_result(inp, inp_ndim, n_elements, num_nonzeros, as_tuple=False, plan=None):
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
    if num_nonzeros == 0:
        return _unbind_views(out) if as_tuple else out

    if plan is None:
        _, plan = _count_plan(inp, n_elements)
    tile, k, grid, chunk, prefix = plan

    src = _nz_mask_source(inp, n_elements)
    ws = torch.empty(num_nonzeros + chunk, dtype=torch.int64, device=inp.device)

    strides = [1] * inp_ndim
    for i in range(inp_ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * inp.shape[i + 1]
    shape = list(inp.shape)
    block = min(
        _DELIN_TILE,
        max(64, triton.next_power_of_2(min(num_nonzeros, _DELIN_TILE))),
    )
    dgrid = (triton.cdiv(num_nonzeros, block),)
    with torch_device_fn.device(inp.device):
        _nz_write_kernel[(grid,)](
            src, prefix, ws, n_elements, num_nonzeros, TILE=tile, K=k
        )
        for d in range(inp_ndim):
            _nz_delin_dim_kernel[dgrid](
                ws,
                out,
                num_nonzeros,
                strides[d],
                shape[d],
                D=d,
                NDIM=inp_ndim,
                BLOCK=block,
            )

    if as_tuple:
        return _unbind_views(out)
    return out


def _device_int_tensor(values, dtype, device):
    """Single host->device copy (was one kernel launch per element)."""
    return torch.tensor(list(values), dtype=dtype, device=device)


def _row_major_strides(shape, device):
    ndim = len(shape)
    strides = [1] * ndim
    for k in range(ndim - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    return _device_int_tensor(strides, torch.int64, device)
