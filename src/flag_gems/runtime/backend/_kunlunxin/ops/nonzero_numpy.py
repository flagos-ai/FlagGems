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
import os

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .cumsum import cumsum
from .nonzero import (
    _count_nonzero,
    _dense_result,
    _sparse_result,
    _unbind_views,
    nonzero,
)

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))

# Block width of the compaction chain below. 8192 is the documented safe
# `tl.sum` / `tl.cumsum` tile on this backend (the value the closed nonzero
# counter and the closed masked_scatter compaction both use).
_NZNP_BLOCK = 8192
_NZNP_WARPS = 16
# Block width of the raw cluster payload below. Larger than _NZNP_BLOCK so each
# core's GM2LM/LM2GM transfer is bigger (small transfers run far below the DMA
# bandwidth). Must be a power-of-two multiple of _NZNP_BLOCK (the count pass
# granularity) and small enough that ibuf + obuf fit the per-core LM budget.
_NZNP_RAW_BLOCK = 16384
# When the raw block grid is at most this, the payload derives each block's
# output base itself from the device count array (`nz_pack_scan`) instead of
# taking a host-computed `bases` vector. The host->device copy of that vector
# costs a fixed amount even for a handful of entries, which dominates every mid
# shape; the in-payload scan is one coalesced pass over `bid * 2` entries.
# The cap is deliberately conservative: the scan is redone per program, so its
# total volume grows as blocks^2, which outruns the copy for very large grids.
_NZNP_SCAN_MAX_BLOCKS = 512

if _TLE_OK:
    try:
        # Every entry lives in the precompiled device object
        # `payload/obj/nz_pack.o` (built from payload/src/nz_pack.xpu + the shared
        # compaction primitives by payload/gen_payloads.py). It ships as machine
        # code -- the C++ source is not in the tree -- and each triton wrapper below
        # calls exactly one entry, so a compiled kernel links only the entry it uses.
        _NZ_OBJ = os.path.join(os.path.dirname(_HERE), "payload", "obj", "nz_pack.o")

        @tle.raw.dialect("xpu3", object=_NZ_OBJ, arch=3)
        def nz_pack(in_, out, bases, n, bid, D1, esz, sign_mask): ...

        @triton.jit(do_not_specialize=["n", "D1", "esz", "sign_mask"])
        def nz_pack_kernel(In, Out, Bases, n, D1, esz, sign_mask):
            pid = tl.program_id(0)
            tle.raw.call(nz_pack, (In, Out, Bases, n, pid, D1, esz, sign_mask))

        @tle.raw.dialect("xpu3", object=_NZ_OBJ, arch=3)
        def nz_pack_scan(in_, out, counts, n, bid, D1, esz, sign_mask, per_raw): ...

        @triton.jit(do_not_specialize=["n", "D1", "esz", "sign_mask", "per_raw"])
        def nz_pack_scan_kernel(In, Out, Counts, n, D1, esz, sign_mask, per_raw):
            pid = tl.program_id(0)
            tle.raw.call(
                nz_pack_scan, (In, Out, Counts, n, pid, D1, esz, sign_mask, per_raw)
            )

        @tle.raw.dialect("xpu3", object=_NZ_OBJ, arch=3)
        def nz_pack_whole(in_, out, total_out, n, D1, esz, sign_mask): ...

        @triton.jit(do_not_specialize=["n", "D1", "esz", "sign_mask"])
        def nz_pack_whole_kernel(In, Out, TotalOut, n, D1, esz, sign_mask):
            tle.raw.call(nz_pack_whole, (In, Out, TotalOut, n, D1, esz, sign_mask))

        @tle.raw.dialect("xpu3", object=_NZ_OBJ, arch=3)
        def nz_dense(out, n, bid, D1): ...

        @triton.jit(do_not_specialize=["n", "D1"])
        def nz_dense_kernel(Out, n, D1):
            pid = tl.program_id(0)
            tle.raw.call(nz_dense, (Out, n, pid, D1))

    except Exception:  # pragma: no cover - triton without object= support
        _TLE_OK = False


@libentry()
@triton.jit
def _nznp_count_full_kernel(inp, counts, BLOCK: tl.constexpr):
    # Per-block nonzero count over FULL blocks only: the offsets are affine and
    # every lane is live, so this pass carries no mask at all (a masked tail
    # load feeding a reduction is the documented silent-error pattern here).
    pid = ext.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(inp + cols)
    tl.store(counts + pid, tl.sum((w != 0).to(tl.int32), axis=0).to(tl.int64))


@libentry()
@triton.jit(do_not_specialize=["n_elements", "n_main", "slot"])
def _nznp_count_tail_kernel(inp, counts, n_elements, n_main, slot, BLOCK: tl.constexpr):
    # Remainder block (< BLOCK elements). Offsets are clamped and the
    # out-of-range lanes are zeroed by an integer ok-multiplier, so the
    # reduction never consumes an untrusted masked `other` value.
    cols = n_main + tl.arange(0, BLOCK)
    last = n_elements - 1
    cclamp = tl.minimum(cols, last)
    ok = (cols < n_elements).to(tl.int32)
    w = tl.load(inp + cclamp)
    tl.store(counts + slot, tl.sum((w != 0).to(tl.int32) * ok, axis=0).to(tl.int64))


@libentry()
@triton.jit(do_not_specialize=["stride_d", "size_d"])
def _nznp_clean_dim_kernel(
    ids,
    bases,
    outd,
    stride_d,
    size_d,
    BLOCK: tl.constexpr,
):
    # CLEAN blocks (no zero at all): the block's BLOCK source elements map onto
    # the contiguous slot run [base, base + BLOCK), so the slot index is
    # `scalar + arange` and the coordinate store is an affine, mask-free int64
    # write. The input is not read at all; the coordinate only depends on the
    # flat source index. Which blocks are clean is decided on the host from the
    # per-block counts, so no data-dependent branch is needed here (an in-kernel
    # `if cnt == BLOCK` around the two store shapes fails to lower on this
    # backend: `OutOfResources: uni_sram`).
    #
    # Two backend-specific constraints shape this kernel:
    #   * ONE LAUNCH PER DIM. A `for d in range(ndim)` loop with the store inside
    #     it does not lower in a kernel of this shape (uni_sram validation
    #     failure at every block size / warp count, whether the dim size comes
    #     from a global load, a list or an if-chain over scalar args).
    #   * SINGLE-TERM STORE ADDRESS. `outd + base + lanes` runs at full DMA
    #     bandwidth, but adding one more runtime scalar to the address expression
    #     (`out + dim_off + base + lanes`, i.e. indexing the [ndim, N] buffer
    #     inside the kernel) collapses it to a scalar-store cliff. The per-dim
    #     row is therefore passed in as its own pointer (`out.select(0, d)`).
    pid = ext.program_id(0)
    blk = tl.load(ids + pid)
    base = tl.load(bases + pid)
    lanes = tl.arange(0, BLOCK)
    idx = blk * BLOCK + lanes
    coord = (idx // stride_d) % size_d
    tl.store(outd + base + lanes.to(tl.int64), coord.to(tl.int64))


@libentry()
@triton.jit(do_not_specialize=["total", "stride_d", "size_d"])
def _nznp_dirty_dim_kernel(
    inp,
    ids,
    bases,
    outd,
    total,
    stride_d,
    size_d,
    BLOCK: tl.constexpr,
):
    # DIRTY full blocks (at least one zero): in-block rank from a 1-D
    # `tl.cumsum` of the nonzero mask on top of the block's host-computed base,
    # then a compacting scatter. No global prefix sum and no `inp != 0` bool
    # copy are materialized. Same one-launch-per-dim / single-term-address
    # discipline as the clean kernel.
    #
    # Inactive lanes are redirected to a per-lane scratch slot in
    # [total, total + BLOCK) instead of relying on masked-store semantics: with
    # `r = base + cumsum - 1` every trailing inactive lane addresses the LAST
    # valid slot, so a store mask that is not honoured (the documented
    # TRITONXPU_STORE_MASK_SIM hazard) silently clobbers it. The scratch slots
    # are unique per lane, so the redirect can never alias a live slot.
    pid = ext.program_id(0)
    blk = tl.load(ids + pid)
    base = tl.load(bases + pid)
    lanes = tl.arange(0, BLOCK)
    cols = blk * BLOCK + lanes
    w = tl.load(inp + cols)
    nz = w != 0
    nzi = nz.to(tl.int32)
    r = (base + tl.cumsum(nzi, axis=0) - 1).to(tl.int64)
    r = tl.where(nz, r, total.to(tl.int64) + lanes.to(tl.int64))
    coord = (cols // stride_d) % size_d
    tl.store(outd + r, coord.to(tl.int64), mask=nz)


@libentry()
@triton.jit(
    do_not_specialize=[
        "total",
        "n_elements",
        "n_main",
        "base",
        "stride_d",
        "size_d",
    ]
)
def _nznp_tail_dim_kernel(
    inp,
    outd,
    total,
    n_elements,
    n_main,
    base,
    stride_d,
    size_d,
    BLOCK: tl.constexpr,
):
    # Remainder block of the compaction (always treated as dirty). Same clamp +
    # ok discipline as the tail counter, same per-lane scratch redirect as the
    # dirty full-block kernel.
    lanes = tl.arange(0, BLOCK)
    cols = n_main + lanes
    last = n_elements - 1
    cclamp = tl.minimum(cols, last)
    ok = cols < n_elements
    w = tl.load(inp + cclamp)
    nz = (w != 0) & ok
    nzi = nz.to(tl.int32)
    r = (base + tl.cumsum(nzi, axis=0) - 1).to(tl.int64)
    r = tl.where(nz, r, total.to(tl.int64) + lanes.to(tl.int64))
    coord = (cols // stride_d) % size_d
    tl.store(outd + r, coord.to(tl.int64), mask=nz)


@libentry()
@triton.jit(do_not_specialize=["total"])
def _nznp_bool_pack_full_kernel(
    inp, prefix, out, total, D1: tl.constexpr, BLOCK: tl.constexpr
):
    # bool 2-D compaction with a single packed u64 store per nonzero: both dim
    # coordinates (each < 2**31) are packed into one int64 (hi32=dim0, lo32=dim1)
    # and scattered once, halving the gather-store count of the two-store path.
    # Inactive lanes are redirected to the [total, total + BLOCK) scratch (the
    # store mask is not honoured on this backend), same as _nznp_dirty_dim_kernel.
    pid = ext.program_id(0)
    lanes = tl.arange(0, BLOCK)
    cols = pid * BLOCK + lanes
    w = tl.load(inp + cols)
    oo = tl.load(prefix + cols) - 1
    nz = w
    r = tl.where(nz, oo.to(tl.int64), total.to(tl.int64) + lanes.to(tl.int64))
    c0 = cols // D1
    c1 = cols - c0 * D1
    packed = (c0.to(tl.int64) << 32) | c1.to(tl.int64)
    tl.store(out + r, packed, mask=nz)


@libentry()
@triton.jit(do_not_specialize=["total", "n_elements", "n_main"])
def _nznp_bool_pack_tail_kernel(
    inp,
    prefix,
    out,
    total,
    n_elements,
    n_main,
    D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Remainder block of the bool packed compaction: same clamp + ok discipline
    # as _nznp_tail_dim_kernel, same per-lane scratch redirect.
    lanes = tl.arange(0, BLOCK)
    cols = n_main + lanes
    last = n_elements - 1
    cclamp = tl.minimum(cols, last)
    ok = cols < n_elements
    w = tl.load(inp + cclamp)
    oo = tl.load(prefix + cclamp) - 1
    nz = w & ok
    r = tl.where(nz, oo.to(tl.int64), total.to(tl.int64) + lanes.to(tl.int64))
    c0 = cols // D1
    c1 = cols - c0 * D1
    packed = (c0.to(tl.int64) << 32) | c1.to(tl.int64)
    tl.store(out + r, packed, mask=nz)


@libentry()
@triton.jit
def _nznp_bool_unpack_kernel(packed, out0, out1, total, BLOCK: tl.constexpr):
    # Affine unpack of the packed u64 into two contiguous per-dim index tensors:
    # hi32 -> dim0, lo32 -> dim1. Purely affine loads/stores (the mask is on
    # contiguous offsets), so it stays on the fast block-DMA path. A single
    # interleaved [total, 2] buffer would force stride-2 stores (much slower).
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(packed + offs, mask=mask)
    tl.store(out0 + offs, v >> 32, mask=mask)
    tl.store(out1 + offs, v & 0xFFFFFFFF, mask=mask)


def _nznp_bool_sparse(inp, n_elements, total):
    """bool 2-D path: one packed u64 gather-store per nonzero instead of the
    two int64 stores the generic scatter performs, then an affine unpack into
    the two per-dim index tensors.

    The bool mask is dense in the benchmark, so every 8192-block is dirty
    and the generic `_sparse_result` (full prefix-sum + two-store scatter) is the
    dominant cost. Packing halves the gather-store count; the prefix is the
    usual closed int64 `cumsum`.
    """
    dev = inp.device
    block = _NZNP_BLOCK
    flat = inp.view(-1).contiguous()
    n_full = n_elements // block
    rem = n_elements - n_full * block
    with torch_device_fn.device(dev):
        prefix = cumsum(flat, dim=0)
        buf = torch.empty(total + block, dtype=torch.int64, device=dev)
        if n_full > 0:
            _nznp_bool_pack_full_kernel[(n_full,)](
                flat,
                prefix,
                buf,
                total,
                D1=inp.shape[1],
                BLOCK=block,
                num_warps=_NZNP_WARPS,
            )
        if rem > 0:
            _nznp_bool_pack_tail_kernel[(1,)](
                flat,
                prefix,
                buf,
                total,
                n_elements,
                n_full * block,
                D1=inp.shape[1],
                BLOCK=block,
                num_warps=_NZNP_WARPS,
            )
        out0 = torch.empty(total, dtype=torch.int64, device=dev)
        out1 = torch.empty(total, dtype=torch.int64, device=dev)
        if total > 0:
            _nznp_bool_unpack_kernel[(triton.cdiv(total, block),)](
                buf,
                out0,
                out1,
                total,
                BLOCK=block,
                num_warps=_NZNP_WARPS,
            )
    return [out0, out1]


# dtypes handled by the raw cluster payload: (element bytes, nonzero-test mask).
# Floating dtypes mask the sign bit so -0.0 counts as zero, matching x != 0.
_RAW_DTYPES = {
    torch.bool: (1, 0xFF),
    torch.int16: (2, 0xFFFF),
    torch.float16: (2, 0x7FFF),
    torch.bfloat16: (2, 0x7FFF),
    torch.int32: (4, -1),  # 0xFFFFFFFF as signed int32 (payload param is int)
    torch.float32: (4, 0x7FFFFFFF),
}


def _nznp_raw(inp, n_elements, counts_h, counts, total):
    """2-D path via a raw cluster payload: cub-style block compaction, any
    density, for 1/2/4-byte dtypes.

    Each program compacts one `_NZNP_RAW_BLOCK`-block by scattering into on-chip
    memory (`__shared__` cross-core scan + `__local__` per-core compact) and
    writing out with one contiguous LM2GM per core, so the global write is
    affine instead of the triton gather store. Returns None when the raw path
    does not apply, so callers fall back.

    `counts`/`counts_h` are the same per-`_NZNP_BLOCK` count vector, still on the
    device and copied to the host. When the block grid is small enough for the
    payload to scan itself (`_NZNP_SCAN_MAX_BLOCKS`), the device copy is handed
    straight to the kernel and no host-side base vector is built or copied over;
    above that cap the exclusive scan runs on the host as before.
    """
    info = _RAW_DTYPES.get(inp.dtype)
    if info is None or not _TLE_OK or total == 0:
        return None
    esz, sign_mask = info
    dev = inp.device
    block = _NZNP_RAW_BLOCK
    flat = inp.view(-1).contiguous()
    flat_u8 = flat.view(torch.uint8)
    n_blocks = (n_elements + block - 1) // block

    # The count pass above runs at _NZNP_BLOCK granularity, so one raw block
    # spans `per_raw` count entries (a power-of-two multiple).
    per_raw = block // _NZNP_BLOCK

    out = torch.empty(total, 2, dtype=torch.int64, device=dev)
    if n_blocks <= _NZNP_SCAN_MAX_BLOCKS:
        with torch_device_fn.device(dev):
            nz_pack_scan_kernel[(n_blocks,)](
                flat_u8,
                out,
                counts,
                n_elements,
                inp.shape[1],
                esz,
                sign_mask,
                per_raw,
            )
        return _unbind_views(out)

    if per_raw == 1:
        counts_raw = counts_h
    else:
        # Pair the _NZNP_BLOCK counts up into raw-block counts. Done in numpy:
        # torch CPU reductions are pathological in this environment, and this
        # runs on every raw call, so it would otherwise dominate large shapes.
        arr = counts_h.numpy()
        pad = (per_raw - arr.shape[0] % per_raw) % per_raw
        if pad:
            arr = np.concatenate([arr, np.zeros(pad, dtype=arr.dtype)])
        counts_raw = torch.from_numpy(arr.reshape(-1, per_raw).sum(axis=1))

    bases_h = torch.empty_like(counts_raw)
    bases_h[0] = 0
    if n_blocks > 1:
        torch.cumsum(counts_raw[: n_blocks - 1], dim=0, out=bases_h[1:])
    bases = bases_h.to(torch.int64).to(dev)

    with torch_device_fn.device(dev):
        nz_pack_kernel[(n_blocks,)](
            flat_u8,
            out,
            bases,
            n_elements,
            inp.shape[1],
            esz,
            sign_mask,
        )
    return _unbind_views(out)


def _nznp_dense_raw(inp, n_elements, total):
    """2-D dense path via a raw payload: every element is nonzero, so each
    output row is just the flat index's row/col. No input read, no count/scan.
    Returns None when the raw path does not apply, so callers fall back to
    `_dense_result`.
    """
    if not _TLE_OK or inp.ndim != 2:
        return None
    dev = inp.device
    block = _NZNP_RAW_BLOCK
    n_blocks = (n_elements + block - 1) // block
    out = torch.empty(total, 2, dtype=torch.int64, device=dev)
    with torch_device_fn.device(dev):
        nz_dense_kernel[(n_blocks,)](out, n_elements, inp.shape[1])
    return _unbind_views(out)


def _nznp_raw_small(inp, n_elements):
    """Single-cluster small-input 2-D path: one fused kernel counts + compacts
    and writes the total to a device scalar, so the whole count pass (extra
    launch + host reduction) is skipped and the host syncs exactly once.

    Only for numel that fits one cluster block (<= _NZNP_RAW_BLOCK) and the
    dtypes `_RAW_DTYPES` covers. Returns None otherwise, so callers fall back.
    """
    info = _RAW_DTYPES.get(inp.dtype)
    if info is None or not _TLE_OK or n_elements == 0:
        return None
    if n_elements > _NZNP_RAW_BLOCK or inp.ndim != 2:
        return None
    esz, sign_mask = info
    dev = inp.device
    flat_u8 = inp.view(-1).contiguous().view(torch.uint8)
    out = torch.empty(n_elements, 2, dtype=torch.int64, device=dev)
    total_dev = torch.empty(1, dtype=torch.int64, device=dev)
    with torch_device_fn.device(dev):
        nz_pack_whole_kernel[(1,)](
            flat_u8,
            out,
            total_dev,
            n_elements,
            inp.shape[1],
            esz,
            sign_mask,
        )
    total = int(total_dev.item())
    if total == 0:
        return [torch.empty(0, dtype=torch.int64, device=dev) for _ in range(2)]
    return _unbind_views(out[:total])


def _nznp_block_counts(flat, n_elements):
    """Per-block nonzero counts (one read pass) brought back to the host.

    The same pass yields the exact total (so the separate two-phase counter is
    not needed), every block's output base, and the clean/dirty classification
    the compaction kernels are launched from.

    Returns the counts twice: `counts_h` on the host (the total and the routing
    decisions need it) and `counts` still on the device (the raw payload reads
    it directly when it derives its own bases).
    """
    dev = flat.device
    block = _NZNP_BLOCK
    n_full = n_elements // block
    rem = n_elements - n_full * block
    n_blocks = n_full + (1 if rem else 0)
    counts = torch.empty(n_blocks, dtype=torch.int64, device=dev)
    with torch_device_fn.device(dev):
        if n_full > 0:
            _nznp_count_full_kernel[(n_full,)](
                flat, counts, BLOCK=block, num_warps=_NZNP_WARPS
            )
        if rem > 0:
            _nznp_count_tail_kernel[(1,)](
                flat,
                counts,
                n_elements,
                n_full * block,
                n_full,
                BLOCK=block,
                num_warps=_NZNP_WARPS,
            )
    return n_full, rem, counts.cpu(), counts


def _nznp_compact(inp, n_elements, n_full, rem, counts_h, total):
    """Dim-major compaction; returns ndim contiguous 1-D index views."""
    ndim = inp.ndim
    dev = inp.device
    block = _NZNP_BLOCK
    row_stride = total + block
    out = torch.empty(ndim, row_stride, dtype=torch.int64, device=dev)
    flat = inp.view(-1)

    bases_h = torch.empty_like(counts_h)
    bases_h[0] = 0
    if counts_h.numel() > 1:
        torch.cumsum(counts_h[:-1], dim=0, out=bases_h[1:])

    if n_full > 0:
        full_counts = counts_h[:n_full]
        clean_h = (full_counts == block).nonzero().reshape(-1)
        dirty_h = (full_counts != block).nonzero().reshape(-1)
    else:
        clean_h = counts_h.new_empty(0)
        dirty_h = counts_h.new_empty(0)

    # row-major source strides, so coord_d = (i // stride_d) % size_d
    src_strides = [1] * ndim
    for k in range(ndim - 2, -1, -1):
        src_strides[k] = src_strides[k + 1] * inp.shape[k + 1]

    clean_ids = clean_bases = None
    if clean_h.numel() > 0:
        clean_ids = clean_h.to(torch.int32).to(dev)
        clean_bases = bases_h[clean_h].to(dev)
    dirty_ids = dirty_bases = None
    if dirty_h.numel() > 0:
        dirty_ids = dirty_h.to(torch.int32).to(dev)
        dirty_bases = bases_h[dirty_h].to(dev)
    tail_base = int(bases_h[n_full].item()) if rem > 0 else 0

    with torch_device_fn.device(dev):
        for d in range(ndim):
            outd = out.select(0, d)
            if clean_ids is not None:
                _nznp_clean_dim_kernel[(clean_ids.numel(),)](
                    clean_ids,
                    clean_bases,
                    outd,
                    src_strides[d],
                    inp.shape[d],
                    BLOCK=block,
                    num_warps=_NZNP_WARPS,
                )
            if dirty_ids is not None:
                _nznp_dirty_dim_kernel[(dirty_ids.numel(),)](
                    flat,
                    dirty_ids,
                    dirty_bases,
                    outd,
                    total,
                    src_strides[d],
                    inp.shape[d],
                    BLOCK=block,
                    num_warps=_NZNP_WARPS,
                )
            if rem > 0:
                _nznp_tail_dim_kernel[(1,)](
                    flat,
                    outd,
                    total,
                    n_elements,
                    n_full * block,
                    tail_base,
                    src_strides[d],
                    inp.shape[d],
                    BLOCK=block,
                    num_warps=_NZNP_WARPS,
                )
    # Dim-major [ndim, row_stride] output, row d is the view [d, :total]:
    # ``out[:, :total].unbind(0)`` would fall back to ATen (forbidden), so
    # build the same rows as zero-copy ``as_strided`` views (metadata-only,
    # not registered -> no re-dispatch).
    return [
        torch.as_strided(out, (total,), (1,), storage_offset=d * row_stride)
        for d in range(ndim)
    ]


def nonzero_numpy(inp):
    """
    Returns a tuple of 1D tensors, one for each dimension of the input,
    containing the indices of the non-zero elements in that dimension.

    This is equivalent to torch.nonzero(...) / numpy.nonzero() semantics and
    matches the ATen op `nonzero_numpy` (Tensor[] of per-dim index vectors).

    Backend chain (all Kunlunxin/XPU Triton, no fallback):

    * one mask-free read pass produces the per-block nonzero counts; they give
      the exact total, every block's output base and the clean/dirty split.
    * `total == numel` (dense: `randn` in fp32, wide-range `randint`) reuses the
      closed `_dense_result` args kernel -- purely affine int64 stores.
    * NEAR-DENSE inputs, i.e. the zeros dirty at most a quarter of the blocks
      (fp16/bf16 `randn` rounds a handful of samples of a 655M-element tensor to
      exactly 0; full-range int16 `randint` hits 0 about once per 65536) go
      through a dim-major block compaction. A clean block writes its whole slot
      run with mask-free affine stores and never reads the input, so the run
      costs dense-path time instead of full-scatter time.
    * genuinely sparse inputs (a dense bool mask) would dirty every block,
      where the in-block rank scan does not pay for itself, so they keep the
      previously closed prefix-sum scatter -- with the total taken from the
      counts above, so no second counting pass is needed.
    * 0-dim scalars are special-cased as in ATen: treated as a 1-element 1-D
      tensor (index [0] when nonzero, empty index when zero).

    `ndim > 8` and outputs at/over the int32 index ceiling keep the previously
    closed chain (`_count_nonzero` + `_dense_result` / `_sparse_result`); the
    closed `nonzero` small-input branch is still never entered, because its
    fallback dense kernel (`nonzero_dense_flat_kernel`, per-lane masked
    metadata loads) fails to compile on this backend at BLOCK_SIZE >= 128.
    """
    logger.debug("GEMS_KUNLUNXIN NONZERO_NUMPY")

    if inp.ndim == 0:
        inp = inp.reshape(-1)

    inp_ndim = inp.ndim
    n_elements = inp.numel()

    if n_elements == 0:
        # ATen: empty input -> ndim empty 1-D index tensors.
        out = torch.empty(0, inp_ndim, dtype=torch.int64, device=inp.device)
        return _unbind_views(out)

    inp = inp.contiguous()

    if inp_ndim <= 8 and n_elements < 2**31:
        # Small inputs skip the count pass entirely: one fused kernel counts +
        # compacts and reports the total, so only one host sync is paid.
        small = _nznp_raw_small(inp, n_elements)
        if small is not None:
            return small
        n_full, rem, counts_h, counts = _nznp_block_counts(inp.view(-1), n_elements)
        total = int(
            counts_h.numpy().sum()
        )  # numpy: torch CPU .sum() is pathological here
        if total * inp_ndim < 2**31:
            if total == n_elements:
                dense_raw = _nznp_dense_raw(inp, n_elements, total)
                if dense_raw is not None:
                    return dense_raw
                return list(_dense_result(inp, total, True))
            if inp_ndim == 2:
                raw = _nznp_raw(inp, n_elements, counts_h, counts, total)
                if raw is not None:
                    return raw
                if inp.dtype == torch.bool:
                    return _nznp_bool_sparse(inp, n_elements, total)
            n_dirty = int((counts_h[:n_full].numpy() != _NZNP_BLOCK).sum())
            if 4 * n_dirty <= n_full:
                return _nznp_compact(inp, n_elements, n_full, rem, counts_h, total)
            return list(_sparse_result(inp, inp_ndim, n_elements, total, True))

    # Outputs at/over the int32 index ceiling, or ndim > 8: previously closed
    # chain (exact two-phase count + dense args kernel / prefix-sum scatter).
    if n_elements >= 8192 and inp.dtype != torch.bool:
        return list(nonzero(inp, as_tuple=True))
    num_nonzeros = _count_nonzero(inp, n_elements)
    if inp_ndim >= 1 and num_nonzeros == n_elements and num_nonzeros * inp_ndim < 2**31:
        return list(_dense_result(inp, num_nonzeros, True))
    return list(_sparse_result(inp, inp_ndim, n_elements, num_nonzeros, True))
