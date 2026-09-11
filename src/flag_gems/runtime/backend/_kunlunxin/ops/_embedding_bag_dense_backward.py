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
#
# Kunlunxin (TritonXPU) specialization of aten::_embedding_bag_dense_backward.
#
# Why this override exists
# ------------------------
# The generic implementation (src/flag_gems/ops/_embedding_bag_dense_backward.py)
# launches one program per (sample, D-block) and merges the per-sample row
# contributions with ``tl.atomic_add``.  On TritonXPU that is both
#
#   * WRONG: ``tl.atomic_add`` silently drops updates whenever two programs hit
#     the same address.  Because an embedding table row is normally referenced by
#     several samples, this is the common case, not a corner case.  A CPU float64
#     oracle shows 36/56 scenarios failing on HEAD, while the very same oracle is
#     clean as soon as every index is unique.
#   * SLOW: ``tl.atomic_add`` is globally serialised at a measured ~180 ns per
#     element on this backend, so HEAD costs ~180 ns * num_samples * D
#     (e.g. 191 ms for num_samples=4096, D=256).
#
# The override therefore removes atomics entirely.  It builds a CSR-like
# "which samples belong to which weight row" structure with three cheap index
# passes (count -> exclusive scan -> stable rank + permutation scatter) and then
# has ONE program own ONE output row, accumulating in registers and issuing a
# single store.  Every global access is either a stride-1 tile off a *scalar*
# base (provable stride-1 => block DMA) or an in-bounds unmasked gather.
#
# Backend rules honoured here (see harness/meta notes):
#   * no ``tl.atomic_add`` at all;
#   * no ``other=`` on any load - out-of-range lanes are clamped to a legal
#     address and gated afterwards with ``tl.where``;
#   * every store is either full-width (>= 64 lanes, no mask) into an
#     over-allocated buffer, or a discrete scatter to provably unique targets;
#   * only 2D ``axis=1`` reductions, and only outside of any nested loop nest
#     that carries the tile;
#   * all tile widths are powers of two;
#   * the fast-path eligibility test is pure metadata (shape/stride/dtype), so
#     no gems operator is dispatched before the decision is taken.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 2D probe tiles: 128 * 128 = 16384 elements (>= the 8192 minimum that keeps 2D
# tiles out of the small-tile mis-lowering window, and both dims are powers of 2).
# The row tile also has to be >= 64 lanes because every vector store on this
# backend touches 64 contiguous elements regardless of the requested length.
_ROWS_TILE = 128
_SAMP_TILE = 128
# The rank pass uses a [_RANK_TILE, _SAMP_TILE] tile; the two extents must differ
# (see _ebdb_rank_kernel) and _SAMP_TILE must be a multiple of _RANK_TILE.
_RANK_TILE = 64
# Single-program exclusive scan window.  tl.cumsum is only trusted up to 8192
# lanes on this backend, hence the num_weights ceiling for the fast path.
_MAX_SCAN = 8192


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def _ebdb_count_kernel(
    indices_ptr,
    tile_count_ptr,
    num_samples,
    num_weights,
    padding_idx,
    NWP: tl.constexpr,
    BW: tl.constexpr,
    NS: tl.constexpr,
):
    # grid = (n_sample_tiles, n_row_blocks); no loop, so the [BW, 1] store never
    # coexists with the [BW, NS] tile inside a loop body (that combination fails
    # to lower: "'triton_xpu.convert_layout' op requires the same shape").
    tid = tl.program_id(0)
    rid = tl.program_id(1)
    rows = (rid * BW + tl.arange(0, BW))[:, None]
    sp = (tid * NS + tl.arange(0, NS))[None, :]
    live = sp < num_samples
    sp_safe = tl.where(live, sp, num_samples - 1)
    iv = tl.load(indices_ptr + sp_safe).to(tl.int32)
    good = live & (iv != padding_idx) & (iv >= 0) & (iv < num_weights)
    hit = (iv == rows) & good
    cnt = tl.sum(tl.where(hit, 1.0, 0.0), axis=1, keep_dims=True)
    tl.store(tile_count_ptr + tid * NWP + rows, cnt.to(tl.int32))


@libentry()
@triton.jit
def _ebdb_tile_scan_kernel(
    tile_count_ptr,
    prefix_ptr,
    counts_ptr,
    n_tiles,
    NWP: tl.constexpr,
    BW: tl.constexpr,
):
    # Pure 1D kernel: exclusive scan of the per-sample-tile counts along the tile
    # axis, plus the per-row total.  All accesses are contiguous BW-wide tiles.
    rid = tl.program_id(0)
    rows = rid * BW + tl.arange(0, BW)
    acc = tl.zeros([BW], dtype=tl.int32)
    for t in range(n_tiles):
        c = tl.load(tile_count_ptr + t * NWP + rows)
        tl.store(prefix_ptr + t * NWP + rows, acc)
        acc += c
    tl.store(counts_ptr + rows, acc)


@libentry()
@triton.jit
def _ebdb_scan_kernel(counts_ptr, start_ptr, TILE: tl.constexpr):
    off = tl.arange(0, TILE)
    c = tl.load(counts_ptr + off).to(tl.float32)
    inclusive = tl.cumsum(c, axis=0)
    tl.store(start_ptr + off, (inclusive - c).to(tl.int32))


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def _ebdb_rank_kernel(
    indices_ptr,
    start_ptr,
    prefix_ptr,
    sorted_ptr,
    num_samples,
    num_weights,
    padding_idx,
    NWP: tl.constexpr,
    TS: tl.constexpr,
    NS: tl.constexpr,
):
    # One program per TS-sample block.  The cross-tile part of the stable rank
    # comes from prefix_ptr (already produced by the count pass), so only the
    # TS x NS intra-tile comparison is needed here - O(num_samples * NS) in total
    # instead of O(num_samples^2).
    #
    # TS must differ from NS: when both sides of the outer product come from a
    # tl.arange of the *same* length the backend tries to give one value two
    # layouts and dies with "'triton_xpu.convert_layout' op requires the same
    # shape for all operands and results".  NS == 2 * TS keeps the [TS, NS] tile
    # at 8192 elements and keeps every TS-block inside a single count tile.
    pid = tl.program_id(0)
    s = (pid * TS + tl.arange(0, TS))[:, None]
    live = s < num_samples
    s_safe = tl.where(live, s, num_samples - 1)
    iv = tl.load(indices_ptr + s_safe).to(tl.int32)
    good = live & (iv != padding_idx) & (iv >= 0) & (iv < num_weights)
    tile = pid // (NS // TS)
    sp = (tile * NS + tl.arange(0, NS))[None, :]
    live2 = sp < num_samples
    sp_safe = tl.where(live2, sp, num_samples - 1)
    jv = tl.load(indices_ptr + sp_safe).to(tl.int32)
    earlier = (jv == iv) & (sp < s) & live2
    rank = tl.sum(tl.where(earlier, 1.0, 0.0), axis=1, keep_dims=True)
    row_safe = tl.where(good, iv, 0)
    base = tl.load(start_ptr + row_safe) + tl.load(prefix_ptr + tile * NWP + row_safe)
    pos = base + rank.to(tl.int32)
    # Inactive / padded / out-of-range lanes are redirected to a per-lane unique
    # scratch slot so that the scatter needs no mask and never aliases.
    dst = tl.where(good, pos, num_samples + s)
    tl.store(sorted_ptr + dst, s.to(tl.int32))


@libentry()
@triton.jit
def _ebdb_gather_row_kernel(
    grad_ptr,
    o2b_ptr,
    bag_ptr,
    psw_ptr,
    start_ptr,
    counts_ptr,
    sorted_ptr,
    out_ptr,
    MODE_MEAN: tl.constexpr,
    HAS_PSW: tl.constexpr,
    SGBF: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BD + tl.arange(0, BD)
    start = tl.load(start_ptr + row)
    cnt = tl.load(counts_ptr + row)
    freq = 1.0
    if SGBF:
        if cnt > 1:
            freq = 1.0 / cnt.to(tl.float32)
    acc = tl.zeros([BD], dtype=tl.float32)
    for k in range(cnt):
        sid = tl.load(sorted_ptr + start + k)
        bag = tl.load(o2b_ptr + sid).to(tl.int32)
        scale = freq
        if MODE_MEAN:
            bsz = tl.load(bag_ptr + bag).to(tl.float32)
            if bsz != 0.0:
                scale = scale / bsz
        if HAS_PSW:
            scale = scale * tl.load(psw_ptr + sid).to(tl.float32)
        g = tl.load(grad_ptr + bag * D + cols)
        acc += g.to(tl.float32) * scale
    tl.store(out_ptr + row * D + cols, acc.to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit
def _ebdb_gather_flat_kernel(
    grad_ptr,
    o2b_ptr,
    bag_ptr,
    psw_ptr,
    start_ptr,
    counts_ptr,
    sorted_ptr,
    out_ptr,
    total,
    MODE_MEAN: tl.constexpr,
    HAS_PSW: tl.constexpr,
    SGBF: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    inb = flat < total
    flat_safe = tl.where(inb, flat, 0)
    row = flat_safe // D
    col = flat_safe % D
    start = tl.load(start_ptr + row)
    cnt = tl.load(counts_ptr + row)
    cnt = tl.where(inb, cnt, 0)
    freq = tl.full([BLOCK], 1.0, tl.float32)
    if SGBF:
        denom = tl.where(cnt > 1, cnt.to(tl.float32), 1.0)
        freq = 1.0 / denom
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    k_max = tl.max(cnt)
    for k in range(k_max):
        act = k < cnt
        j = tl.where(act, start + k, 0)
        sid = tl.load(sorted_ptr + j)
        sid = tl.where(act, sid, 0)
        bag = tl.load(o2b_ptr + sid).to(tl.int32)
        bag = tl.where(act, bag, 0)
        scale = freq
        if MODE_MEAN:
            bsz = tl.load(bag_ptr + bag).to(tl.float32)
            scale = scale / tl.where(bsz != 0.0, bsz, 1.0)
        if HAS_PSW:
            scale = scale * tl.load(psw_ptr + sid).to(tl.float32)
        g = tl.load(grad_ptr + bag * D + col)
        acc += tl.where(act, g.to(tl.float32) * scale, 0.0)
    tl.store(out_ptr + flat, acc.to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit
def _ebdb_max_kernel(
    grad_ptr,
    max_idx_ptr,
    out_ptr,
    total,
    num_bags,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    inb = flat < total
    flat_safe = tl.where(inb, flat, 0)
    row = flat_safe // D
    col = flat_safe % D
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for b in range(num_bags):
        mv = tl.load(max_idx_ptr + b * D + col).to(tl.int32)
        gv = tl.load(grad_ptr + b * D + col).to(tl.float32)
        acc += tl.where(inb & (mv == row), gv, 0.0)
    tl.store(out_ptr + flat, acc.to(out_ptr.dtype.element_ty))


def _pow2_block(width, lo=64, hi=128):
    """Largest power of two in [lo, hi] that divides ``width`` (None if none).

    ``hi`` is deliberately 128: BD=256 makes ``_ebdb_gather_row_kernel`` fail to
    lower (``OutOfResources: uni_sram ... Required: 0, Hardware limit: 0``, i.e.
    the "this is not a resource problem" signature), while 64 and 128 compile and
    run.  This matches the known non-monotonic TritonXPU tile-width envelope.
    """
    blk = min(hi, triton.next_power_of_2(width))
    while blk >= lo:
        if width % blk == 0:
            return blk
        blk //= 2
    return None


def _fast_path_kind(
    grad,
    indices,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    mode,
    per_sample_weights,
):
    """Pure-metadata eligibility gate.

    Dispatches no operator at all: only shapes, strides, dtypes and contiguity
    are inspected.  Returning None means "let the generic implementation run".
    """
    if grad.ndim != 2 or not grad.is_contiguous():
        return None
    if int(num_weights) < 1:
        return None
    num_bags, dim = grad.shape
    n_samples = indices.numel()
    if n_samples < 1 or dim < 1 or num_bags < 1:
        return None
    if indices.dtype not in (torch.int32, torch.int64):
        return None
    if not indices.is_contiguous():
        return None
    if mode == 2:
        if maximum_indices is None or maximum_indices.ndim != 2:
            return None
        if tuple(maximum_indices.shape) != (num_bags, dim):
            return None
        if not maximum_indices.is_contiguous():
            return None
        # nw * nb * D work; keep the (matrix-external) MAX path off huge sizes.
        if int(num_weights) * num_bags * dim > (1 << 27):
            return None
        return "max"
    if mode not in (0, 1):
        return None
    if int(num_weights) + 1 > _MAX_SCAN:
        return None
    if offset2bag.numel() != n_samples or not offset2bag.is_contiguous():
        return None
    if bag_size.numel() != num_bags or not bag_size.is_contiguous():
        return None
    if per_sample_weights is not None:
        if per_sample_weights.numel() != n_samples:
            return None
        if not per_sample_weights.is_contiguous():
            return None
    return "csr"


def _build_csr(indices, n_samples, num_weights, padding_idx):
    """count -> exclusive scan -> stable rank + permutation scatter (no atomics)."""
    device = indices.device
    n_rows_pad = max(_ROWS_TILE, triton.next_power_of_2(num_weights))
    n_samp_tiles = triton.cdiv(n_samples, _SAMP_TILE)
    counts = torch.empty(n_rows_pad, dtype=torch.int32, device=device)
    start = torch.empty(n_rows_pad, dtype=torch.int32, device=device)
    tile_count = torch.empty(
        n_samp_tiles * n_rows_pad, dtype=torch.int32, device=device
    )
    prefix = torch.empty(n_samp_tiles * n_rows_pad, dtype=torch.int32, device=device)
    n_row_blocks = n_rows_pad // _ROWS_TILE
    _ebdb_count_kernel[(n_samp_tiles, n_row_blocks)](
        indices,
        tile_count,
        n_samples,
        num_weights,
        padding_idx,
        NWP=n_rows_pad,
        BW=_ROWS_TILE,
        NS=_SAMP_TILE,
    )
    _ebdb_tile_scan_kernel[(n_row_blocks,)](
        tile_count,
        prefix,
        counts,
        n_samp_tiles,
        NWP=n_rows_pad,
        BW=_ROWS_TILE,
    )
    _ebdb_scan_kernel[(1,)](counts, start, TILE=n_rows_pad)
    n_rank_blocks = triton.cdiv(n_samples, _RANK_TILE)
    # tail slots [n_samples, n_samples + n_rank_blocks * _RANK_TILE) are per-lane
    # unique scratch for padded / out-of-range samples: the scatter needs no mask.
    order = torch.empty(
        n_samples + n_rank_blocks * _RANK_TILE, dtype=torch.int32, device=device
    )
    _ebdb_rank_kernel[(n_rank_blocks,)](
        indices,
        start,
        prefix,
        order,
        n_samples,
        num_weights,
        padding_idx,
        NWP=n_rows_pad,
        TS=_RANK_TILE,
        NS=_SAMP_TILE,
    )
    return counts, start, order


def _generic_impl(*args):
    # The generic FlagGems Triton implementation; used only as the structural
    # fall-back for shapes outside the fast-path envelope.  It is still a
    # FlagGems XPU Triton kernel - not a CPU / ATen / composite fall-back.
    from flag_gems.ops._embedding_bag_dense_backward import (
        _embedding_bag_dense_backward as _generic,
    )

    return _generic(*args)


def _embedding_bag_dense_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    maximum_indices: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _EMBEDDING_BAG_DENSE_BACKWARD")

    kind = _fast_path_kind(
        grad,
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        int(mode),
        per_sample_weights,
    )
    if kind is None:
        return _generic_impl(
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            scale_grad_by_freq,
            mode,
            per_sample_weights,
            padding_idx,
        )

    device = grad.device
    num_bags, dim = grad.shape
    num_weights = int(num_weights)
    total = num_weights * dim

    if kind == "max":
        block = max(64, min(1024, triton.next_power_of_2(dim)))
        n_blocks = triton.cdiv(total, block)
        buf = torch.empty(n_blocks * block, dtype=grad.dtype, device=device)
        _ebdb_max_kernel[(n_blocks,)](
            grad,
            maximum_indices,
            buf,
            total,
            num_bags,
            D=dim,
            BLOCK=block,
        )
        return buf[:total].view(num_weights, dim)

    n_samples = indices.numel()
    pad = int(padding_idx)
    counts, start, order = _build_csr(indices, n_samples, num_weights, pad)

    mode_mean = int(mode) == 1
    has_psw = per_sample_weights is not None
    psw = per_sample_weights if has_psw else indices  # dummy pointer when unused
    sgbf = bool(scale_grad_by_freq)

    block_d = _pow2_block(dim)
    if block_d is not None:
        # One program owns one output row: the grad tile base ``bag * D`` is a
        # scalar, the columns are a plain tl.arange, so the DMA is provably
        # stride-1 and every store is a full BD >= 64 lane write with no mask.
        out = torch.empty((num_weights, dim), dtype=grad.dtype, device=device)
        grid = (num_weights, dim // block_d)
        _ebdb_gather_row_kernel[grid](
            grad,
            offset2bag,
            bag_size,
            psw,
            start,
            counts,
            order,
            out,
            MODE_MEAN=mode_mean,
            HAS_PSW=has_psw,
            SGBF=sgbf,
            D=dim,
            BD=block_d,
        )
        return out

    block = max(64, min(1024, triton.next_power_of_2(dim)))
    n_blocks = triton.cdiv(total, block)
    buf = torch.empty(n_blocks * block, dtype=grad.dtype, device=device)
    _ebdb_gather_flat_kernel[(n_blocks,)](
        grad,
        offset2bag,
        bag_size,
        psw,
        start,
        counts,
        order,
        buf,
        total,
        MODE_MEAN=mode_mean,
        HAS_PSW=has_psw,
        SGBF=sgbf,
        D=dim,
        BLOCK=block,
    )
    return buf[:total].view(num_weights, dim)
