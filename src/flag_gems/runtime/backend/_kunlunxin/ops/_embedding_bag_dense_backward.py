import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_ROWS_TILE = 128
_SAMP_TILE = 128
_RANK_TILE = 64
_GATHER_BLOCK = 256
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
def _ebdb_tile_prefix_kernel(
    tile_count_ptr,
    prefix_ptr,
    n_tiles,
    NWP: tl.constexpr,
    BW: tl.constexpr,
):
    """Exclusive per-tile prefix; ``tid == n_tiles`` yields the row total.

    ``n_tiles`` is passed as a scalar kernel argument instead of being staged in
    device memory, which drops a ``torch.full`` host call (and its fill kernel)
    per invocation.  The extra ``tid == n_tiles`` program writes the inclusive
    total, so the former ``_ebdb_tile_total_kernel`` launch is gone as well.
    """
    tid = tl.program_id(0)
    rid = tl.program_id(1)
    rows = rid * BW + tl.arange(0, BW)
    acc = tl.zeros([BW], dtype=tl.int32)
    for u in range(n_tiles):
        c = tl.load(tile_count_ptr + (u * NWP + rows))
        acc += tl.where(u < tid, c, 0)
    tl.store(prefix_ptr + (tid * NWP + rows), acc)


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
    dst = tl.where(good, pos, num_samples + s)
    tl.store(sorted_ptr + dst, s.to(tl.int32))


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
    MODE_MEAN: tl.constexpr,
    HAS_PSW: tl.constexpr,
    SGBF: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per ``BLOCK`` contiguous output elements.

    The output buffer is padded to a whole number of ``BLOCK`` tiles and the
    CSR tables are sized to cover every padded row, so this kernel needs no
    bounds mask at all: dropping the ``flat < total`` predicate is a measured
    win (``tl.where``/masked paths are ~2x more expensive on XPU3) and padded
    rows simply carry ``count == 0`` (no loop iterations, zero store).
    """
    pid = tl.program_id(0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    row = flat // D
    col = flat % D
    start = tl.load(start_ptr + row)
    cnt = tl.load(counts_ptr + row)
    freq = tl.full([BLOCK], 1.0, tl.float32)
    if SGBF:
        freq = 1.0 / tl.where(cnt > 1, cnt.to(tl.float32), 1.0)
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


def _build_csr(indices, n_samples, num_weights, padding_idx, n_rows_pad):
    """count -> exclusive scan -> stable rank + permutation scatter (no atomics).

    No atomics anywhere: every slot of the permutation is written by exactly one
    sample and every output row is reduced by exactly one program.  Note that a
    *loop-carried* ``tl.sum`` accumulator is unusable here (XPU3 either fails to
    lower it or silently mis-counts), so the per-tile counts are materialised
    first and only reduce-free loops accumulate afterwards.
    """
    device = indices.device
    n_samp_tiles = triton.cdiv(n_samples, _SAMP_TILE)
    # prefix carries one extra tile: prefix[n_samp_tiles] *is* the row total,
    # so `counts` is a view of it and no separate total pass is needed.
    prefix = torch.empty(
        (n_samp_tiles + 1) * n_rows_pad, dtype=torch.int32, device=device
    )
    counts = prefix[n_samp_tiles * n_rows_pad :]
    start = torch.empty(n_rows_pad, dtype=torch.int32, device=device)
    tile_count = torch.empty(
        n_samp_tiles * n_rows_pad, dtype=torch.int32, device=device
    )
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
    _ebdb_tile_prefix_kernel[(n_samp_tiles + 1, n_row_blocks)](
        tile_count,
        prefix,
        n_samp_tiles,
        NWP=n_rows_pad,
        BW=_ROWS_TILE,
    )
    _ebdb_scan_kernel[(1,)](counts, start, TILE=n_rows_pad)
    n_rank_blocks = triton.cdiv(n_samples, _RANK_TILE)
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
    block = _GATHER_BLOCK
    n_blocks = triton.cdiv(total, block)
    n_rows_pad = max(_ROWS_TILE, triton.next_power_of_2(num_weights))
    # The gather kernel is launched without a tail mask, so its padded flat tail
    # can address rows up to ``rows_touched``; the CSR tables must cover them
    # (those rows get count == 0 from the index comparison).
    rows_touched = triton.cdiv(n_blocks * block, dim)
    if rows_touched > n_rows_pad:
        n_rows_pad = triton.next_power_of_2(rows_touched)
    if n_rows_pad > _MAX_SCAN:
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
    counts, start, order = _build_csr(indices, n_samples, num_weights, pad, n_rows_pad)

    mode_mean = int(mode) == 1
    has_psw = per_sample_weights is not None
    psw = per_sample_weights if has_psw else indices
    sgbf = bool(scale_grad_by_freq)

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
        MODE_MEAN=mode_mean,
        HAS_PSW=has_psw,
        SGBF=sgbf,
        D=dim,
        BLOCK=block,
    )
    return buf[:total].view(num_weights, dim)
