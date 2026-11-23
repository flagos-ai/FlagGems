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

import contextlib
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

# Stand-in for torch_device_fn.device(device) when the requested device is
# already the current one (see the caller).
_NULL_CTX = contextlib.nullcontext()

try:  # triton-ascend native vector extensions (sort); absent on other backends
    import triton.language.extra.cann.extension as al
except ImportError:  # pragma: no cover
    al = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ascend masked_scatter_backward
#
#   Semantics: out = cat([grad[mask], zeros]).view(sizes), i.e.
#   out[j] = grad[idx[j]] for j < k, out[j] = 0 for k <= j < numel, where
#   idx = sorted True positions of mask and k = #True.
#
#   triton-ascend has no vscatter (a scattered store costs ~29x this whole
#   pipeline), so the output is computed by EXPAND -- for each output position
#   find its source lane, rather than placing each source element.  tl.gather
#   is the only cross-lane movement the backend exposes.
#
#     1. _count_kernel: per-block True count + the scan's tail sentinel.
#     2. an exclusive scan of those counts (nb+1 elements, so the tail is k).
#     3. _expand_sort_kernel: sort the block's lanes so output position
#        r < cnt IS the r-th True lane, then one tl.gather + a dense prefix
#        store -- no scattered stores, hence density-insensitive.
#
#   Landmine (worked around): tl.gather over a vector DERIVED from the grouped
#   scan's reshape/transpose/cumsum chain inside a search dependency loop
#   crashes the vector core (0x31), so the search back-end round-trips its rank
#   array through HBM.  No CANN extension mem_op has a compiled lowering, and
#   num_warps/multibuffer are inert here, so BLOCK_SIZE is the only tuning knob.
#
#   The benchmark's mask is randn(...) < 0.3, i.e. density Phi(0.3) ~0.62.
# ---------------------------------------------------------------------------

# BLOCK_SIZE ~ _BLOCK_SCALE * sqrt(N), clamped to [_MIN, _MAX]: a larger block
# pays more per lane (the expand grows with log2(BLOCK)), a smaller one pays
# more CTA fixed cost (~2200 cycles each), so the optimum grows like sqrt(N).
_BLOCK_SCALE = 8
_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 4096
# expand/count block ceiling: sort keys are fp16 and lane indices must stay
# < 2048 (fp16 exact-integer range); 8192 overflows the UB.
_MAX_EXPAND_BLOCK = 2048
# Below this many blocks the expand derives its own offsets from the raw counts
# (FUSED) instead of reading a scanned array, skipping the scan launch (2
# launches instead of 3).  The derivation is O(n_blocks) per CTA.
_FUSED_MAX_BLOCKS = 2048
# SELF_COUNT ceiling: n_blocks * N is the mask traffic the self-counting expand
# re-reads across all CTAs -- at (256,256) ~20 us of device time, against the
# ~73 us of host time a saved launch is worth.
_SELF_COUNT_MAX_ELEMS = 4 << 20
_MAX_SCAN_BLOCK = 16384  # single-launch grouped scan upper bound (int32)
_MAX_SCAN_GROUP = 128
_MAX_SCAN_HS = 4096  # Hillis-Steele fp32 scan bound: prefix sums exact below 2**24

# Expand back-end.  "sort" recovers the source lanes from the mask itself --
# key = lane for True lanes, a large sentinel for False ones, handed to the
# native vector sort (cann extension sort -> hivm.hir.vsort) -- so output
# position r < cnt IS the r-th True lane and the binary search vanishes.
# Measured @N=1M, cy/elem: sort 3.37-3.88 vs search 5.00-6.84 over BLOCK
# 512-4096, and the sort is flat in both BLOCK and density.  Keys MUST be
# fp16/fp32: int keys pass the Python-side allowed_types check in vec_ops.sort
# but hivm.hir.vsort rejects them at verification.  "search" is the
# binary-lifting path, kept for A/B measurement.
_EXPAND_MODE = "sort"


@triton.jit
def _grouped_excl_scan(values, BLOCK_SIZE: tl.constexpr, SCAN_GROUP_SIZE: tl.constexpr):
    """Exclusive scan via a 2-D grouped scan: reshape -> transpose -> cumsum over
    the short axis -> transpose back -> add the exclusive group prefix.
    ~3x cheaper than a plain tl.cumsum over BLOCK_SIZE lanes on Ascend.
    """
    num_groups: tl.constexpr = BLOCK_SIZE // SCAN_GROUP_SIZE
    grouped = tl.reshape(values, (num_groups, SCAN_GROUP_SIZE))
    transposed = tl.trans(grouped, (1, 0))
    within_group = tl.cumsum(transposed, axis=0) - transposed
    within_group = tl.trans(within_group, (1, 0))
    group_counts = tl.sum(grouped, axis=1)
    group_offsets = tl.cumsum(group_counts, axis=0) - group_counts
    return tl.reshape(within_group + group_offsets[:, None], (BLOCK_SIZE,))


@triton.jit
def _scan_f16_gather(
    values,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    """As _grouped_excl_scan, in fp16 (any intermediate <= 2048 is exact), which
    halves the UB traffic of the two transposes; the offset cumsum becomes
    log2(num_groups) gather steps.  Also returns the inclusive group-count scan.
    """
    num_groups: tl.constexpr = BLOCK_SIZE // SCAN_GROUP_SIZE
    LOG_GROUPS: tl.constexpr = num_groups.bit_length() - 1
    grouped = tl.reshape(values, (num_groups, SCAN_GROUP_SIZE))
    transposed = tl.trans(grouped, (1, 0))
    within_group = tl.cumsum(transposed, axis=0) - transposed
    within_group = tl.trans(within_group, (1, 0))
    group_counts = tl.sum(grouped, axis=1)
    # fp16 is exact only up to 2048, so a 4096 tile accumulates the group
    # offsets (bounded by the tile's True count) in fp32, exact below 2**24.
    if BLOCK_SIZE > 2048:
        acc = group_counts.to(tl.float32)
    else:
        acc = group_counts
    for k in tl.static_range(LOG_GROUPS):
        idx = tl.arange(0, num_groups) - (1 << k)
        shifted = tl.gather(acc, tl.maximum(idx, 0), axis=0)
        shifted = tl.where(tl.arange(0, num_groups) >= (1 << k), shifted, 0)
        acc = acc + shifted
    group_offsets = acc - group_counts  # exclusive
    return (tl.reshape(within_group + group_offsets[:, None], (BLOCK_SIZE,)), acc)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _count_kernel(
    mask_ptr,
    counts_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-CTA True count only: the sort path recovers source lanes from the mask
    directly, so no rank array (or its HBM round-trip) is needed.

    The last CTA also writes the sentinel 0 into counts[n_blocks], which the
    exclusive scan needs to return n_blocks+1 prefixes (offsets[n_blocks] == k).
    Writing it here lets counts come from torch.empty instead of torch.zeros --
    the zeroing memset would be an extra device kernel plus ~60 us of dispatch.
    """
    pid = ext.program_id(axis=0)
    n_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    tl.store(counts_ptr + pid, tl.sum(mask_val, axis=0))
    if pid == n_blocks - 1:
        tl.store(counts_ptr + n_blocks, 0)


@libentry()
@triton.jit(do_not_specialize=["N", "numel"])
def _count_rank_kernel(
    mask_ptr,
    counts_ptr,
    inc_ptr,
    out_ptr,
    N,
    numel,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    """Per-CTA True count + dense zero-fill + dense inclusive-rank store.

    inc[lane] = (#True before lane) + mask[lane], monotone -- the binary-search
    array of _expand_kernel, stored int16 (exact for BLOCK_SIZE <= 4096).  The
    zero-fill of out[0:numel) is folded in; the expand overwrites the prefix.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    rank, _ = _scan_f16_gather(mask_val.to(tl.float16), BLOCK_SIZE, SCAN_GROUP_SIZE)
    inc = (rank.to(tl.int32) + mask_val).to(tl.int16)
    tl.store(inc_ptr + offsets, inc, mask=m)
    tl.store(counts_ptr + pid, tl.sum(mask_val, axis=0))

    # Dense zero-fill of out[0:numel) — covered by grid = cdiv(max(N, numel), BLOCK)
    z = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + z, 0.0, mask=(z < numel))


@libentry()
@triton.jit(do_not_specialize=["N", "numel"])
def _expand_kernel(
    grad16_ptr,
    inc_ptr,
    offsets_ptr,
    out_ptr,
    N,
    numel,
    BLOCK_SIZE: tl.constexpr,
    WIDE_ELEM: tl.constexpr,
):
    """Expand block b into the dense output prefix [off[b], off[b+1]).

    For output position r < cnt the source lane is the first lane whose
    inclusive rank is >= r+1, found by log2(BLOCK_SIZE) rounds of binary lifting
    over inc via tl.gather.  The store is a dense prefix write, so the kernel is
    insensitive to mask density.

    inc is int16 read through its fp16 view as raw bits, padded past N with
    65535 to stay monotone, and must come from HBM (see the module landmine).
    For 4-byte elements (WIDE_ELEM) grad is read as fp16 halfword pairs and
    recombined in-register: a direct fp32 vector load runs at only ~40 GB/s here.
    """
    LOG_REST: tl.constexpr = BLOCK_SIZE.bit_length() - 1
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    cnt = tl.load(offsets_ptr + pid + 1) - tl.load(offsets_ptr + pid)
    out_off = tl.load(offsets_ptr + pid)
    # int16 ranks via the fp16 view (raw bits); 65535 -> fp16 65504, a
    # monotone pad >= any target <= BLOCK_SIZE
    inc_bits = tl.load(inc_ptr + offsets, mask=m, other=65535)
    inc = inc_bits.to(tl.int16, bitcast=True).to(tl.float32)

    r_i = tl.arange(0, BLOCK_SIZE)
    target = r_i.to(tl.float32) + 1.0  # r+1; r < cnt => target <= cnt
    # Binary lifting for "last lane with inc < r+1", +1: exactly log2(BLOCK_SIZE)
    # rounds of 3 vector ops + 1 gather, against 1 + log2() for the two-pointer
    # search it replaces -- 15.25 -> 11.42 us per 4096-lane block, bit-identical.
    # `cand` stays in [0, BLOCK_SIZE) by construction (pos starts at -1 and only
    # gains accepted powers), so no per-round min() is needed; inc is monotone,
    # so r >= cnt converges on a wrong lane that the store mask discards.
    pos = tl.full((BLOCK_SIZE,), -1, tl.int32)
    for k in tl.static_range(LOG_REST):
        cand = pos + (1 << (LOG_REST - 1 - k))
        v = tl.gather(inc, cand, axis=0)
        pos = tl.where(v < target, cand, pos)
    lane = tl.minimum(pos + 1, BLOCK_SIZE - 1)
    mst = (r_i < cnt) & ((out_off + r_i).to(tl.int64) < numel)
    if WIDE_ELEM:
        offsets16 = pid * (2 * BLOCK_SIZE) + tl.arange(0, 2 * BLOCK_SIZE)
        g16 = tl.load(grad16_ptr + offsets16, mask=offsets16 < 2 * N, other=0)
        glo = tl.gather(g16, lane * 2, axis=0)
        ghi = tl.gather(g16, lane * 2 + 1, axis=0)
        # Recombine the halfword pair into the fp32 bit pattern in-register
        # (the int32 casts sign-extend, so mask each to 16 bits first).
        lo32 = glo.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        hi32 = (ghi.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) << 16
        tl.store(out_ptr + out_off + r_i, lo32 | hi32, mask=mst)
    else:
        g16 = tl.load(grad16_ptr + offsets, mask=m, other=0)
        gv = tl.gather(g16, lane, axis=0)
        tl.store(out_ptr + out_off + r_i, gv, mask=mst)


@libentry()
@triton.jit(do_not_specialize=["N", "numel"])
def _expand_sort_kernel(
    grad16_ptr,
    mask_ptr,
    offsets_ptr,
    out_ptr,
    N,
    numel,
    BLOCK_SIZE: tl.constexpr,
    WIDE_ELEM: tl.constexpr,
    DO_TAIL: tl.constexpr,
    FUSED: tl.constexpr,
    SELF_COUNT: tl.constexpr,
    NB_POW2: tl.constexpr,
):
    """Expand block b into the dense output prefix [off[b], off[b+1]) by sort.

    key = lane for True lanes, 32768 for False ones: after the native sort,
    position r < cnt is exactly the r-th True lane, so the binary search becomes
    one sort.  fp16 keys because the BLOCK is capped at 2048 (every lane index
    exact) and the narrower key sorts ~1.4x cheaper.  No ties inside [0, cnt):
    True lanes are distinct and the sentinel only lands at positions >= cnt.

    DO_TAIL folds the tail zero-fill out[k:numel) in here (disjoint write
    ranges, hence race-free), saving a launch -- valid only when the CTAs
    together span the tail: numel <= n_blocks * BLOCK_SIZE.

    SELF_COUNT also drops the count launch, leaving a single kernel, at the
    price of every CTA scanning all N mask elements (gated by
    _SELF_COUNT_MAX_ELEMS).  offsets_ptr is then unused and gets out_ptr.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    n_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    if SELF_COUNT:
        # One pass over the block grid gives the prefix before this CTA and the
        # grand total (k, for DO_TAIL); its own count comes from the block load
        # further down.  The accumulators are BLOCK-wide vectors reduced once at
        # the end: a tl.sum per iteration costs 25 us on (256,256) against 15 us
        # here, so the reduction, not the load latency, dominates.  fp32 is
        # exact here (partial sums stay < 2**24).
        acc_all = tl.zeros([BLOCK_SIZE], tl.float32)
        acc_pre = tl.zeros([BLOCK_SIZE], tl.float32)
        for b in range(0, n_blocks):
            boff = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            bv = tl.load(mask_ptr + boff, mask=boff < N, other=0).to(tl.float32)
            acc_all += bv
            acc_pre += tl.where(b < pid, bv, 0.0)
        k = tl.sum(acc_all, axis=0).to(tl.int32)
        out_off = tl.sum(acc_pre, axis=0).to(tl.int32)
    elif FUSED:
        # No scan kernel: this CTA derives its count, prefix and the grand total
        # k from the raw per-block counts itself.  The latter two are tree
        # reductions over NB_POW2 lanes (O(n_blocks) per CTA, hence
        # _FUSED_MAX_BLOCKS) but remove the _exclusive_scan launch (~84 us host).
        idx = tl.arange(0, NB_POW2)
        c = tl.load(offsets_ptr + idx, mask=idx < n_blocks, other=0)
        k = tl.sum(c, axis=0)
        out_off = tl.sum(tl.where(idx < pid, c, 0), axis=0)
        cnt = tl.load(offsets_ptr + pid)
    else:
        cnt = tl.load(offsets_ptr + pid + 1) - tl.load(offsets_ptr + pid)
        out_off = tl.load(offsets_ptr + pid)
        if DO_TAIL:
            # k = offsets[n_blocks], the scan's tail element.
            k = tl.load(offsets_ptr + n_blocks).to(tl.int64)

    if DO_TAIL:
        tail = k.to(tl.int64) + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + tail, 0.0, mask=tail < numel)

    r_i = tl.arange(0, BLOCK_SIZE)
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    if SELF_COUNT:
        # This CTA's own block count, already in hand: reusing this load saves a
        # second BLOCK-wide read of the block and a second reduction.
        cnt = tl.sum(mask_val, axis=0)
    key = tl.where(mask_val != 0, r_i.to(tl.float16), 32768.0)
    sorted_key = al.sort(key, dim=-1)
    lane = tl.minimum(sorted_key.to(tl.int32), BLOCK_SIZE - 1)

    mst = (r_i < cnt) & ((out_off + r_i).to(tl.int64) < numel)
    if WIDE_ELEM:
        offsets16 = pid * (2 * BLOCK_SIZE) + tl.arange(0, 2 * BLOCK_SIZE)
        g16 = tl.load(grad16_ptr + offsets16, mask=offsets16 < 2 * N, other=0)
        glo = tl.gather(g16, lane * 2, axis=0)
        ghi = tl.gather(g16, lane * 2 + 1, axis=0)
        lo32 = glo.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        hi32 = (ghi.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) << 16
        tl.store(out_ptr + out_off + r_i, lo32 | hi32, mask=mst)
    else:
        g16 = tl.load(grad16_ptr + offsets, mask=m, other=0)
        gv = tl.gather(g16, lane, axis=0)
        tl.store(out_ptr + out_off + r_i, gv, mask=mst)


@libentry()
@triton.jit(do_not_specialize=["numel"])
def _tail_zero_kernel(
    out_ptr,
    offsets_ptr,
    n_blocks,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero-fill the output tail out[k:numel), k = offsets[n_blocks]; the expand
    kernels already wrote out[0:k).  The grid is over-provisioned to
    cdiv(numel, BLOCK_SIZE) and the mask discards the excess, so the host never
    needs k.
    """
    k = tl.load(offsets_ptr + n_blocks).to(tl.int64)
    pid = ext.program_id(axis=0)
    offs = k + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, 0.0, mask=(offs < numel))


# ---------------------------------------------------------------------------
# device-side exclusive scan over block counts (int32, grouped 2-D scan)
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _scan_kernel(
    counts_ptr,
    part_sums_ptr,
    n_elem,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    m = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    cumsums = _grouped_excl_scan(counts, BLOCK_SIZE, SCAN_GROUP_SIZE)
    tl.store(part_sums_ptr + offsets, cumsums, mask=m)


@libentry()
@triton.jit
def _scan_hs_kernel(
    counts_ptr,
    part_sums_ptr,
    n_elem,
    BLOCK_SIZE: tl.constexpr,
):
    """Exclusive scan of a small count vector via a Hillis-Steele gather tree.

    Only for n_elem <= 4096, where the fp32 accumulator is exact (prefixes stay
    below 2**24) and a log2(BLOCK_SIZE)-round gather tree beats the cumsum-based
    _grouped_excl_scan, whose lowering carries a large fixed Ascend cost.
    """
    LOG: tl.constexpr = BLOCK_SIZE.bit_length() - 1
    offsets = tl.arange(0, BLOCK_SIZE)
    m = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=m, other=0).to(tl.float32)
    acc = counts
    for k in tl.static_range(LOG):
        idx = offsets - (1 << k)
        shifted = tl.gather(acc, tl.maximum(idx, 0), axis=0)
        shifted = tl.where(offsets >= (1 << k), shifted, 0)
        acc = acc + shifted
    tl.store(part_sums_ptr + offsets, (acc - counts).to(tl.int32), mask=m)


@libentry()
@triton.jit
def _chunk_scan_kernel(
    counts_ptr,
    part_sums_ptr,
    chunk_totals_ptr,
    n_elem,
    CHUNK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem
    counts = tl.load(counts_ptr + offsets, mask=m, other=0)
    cumsums = _grouped_excl_scan(counts, CHUNK_SIZE, SCAN_GROUP_SIZE)
    tl.store(part_sums_ptr + offsets, cumsums, mask=m)
    tl.store(chunk_totals_ptr + pid, tl.sum(counts, axis=0))


@libentry()
@triton.jit
def _add_offsets_kernel(
    part_sums_ptr,
    chunk_offsets_ptr,
    n_elem,
    CHUNK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    m = offsets < n_elem
    val = tl.load(part_sums_ptr + offsets, mask=m, other=0)
    tl.store(part_sums_ptr + offsets, val + tl.load(chunk_offsets_ptr + pid), mask=m)


def _exclusive_scan(arr, n_elems, device):
    part_sums = torch.empty(n_elems, dtype=torch.int32, device=device)
    if n_elems <= _MAX_SCAN_HS:
        scan_block = triton.next_power_of_2(n_elems)
        _scan_hs_kernel[(1,)](
            arr,
            part_sums,
            n_elems,
            BLOCK_SIZE=scan_block,
        )
    elif n_elems <= _MAX_SCAN_BLOCK:
        scan_block = triton.next_power_of_2(n_elems)
        _scan_kernel[(1,)](
            arr,
            part_sums,
            n_elems,
            BLOCK_SIZE=scan_block,
            SCAN_GROUP_SIZE=min(_MAX_SCAN_GROUP, scan_block),
        )
    else:
        # Chunked scan for very many counts.  CHUNK_SIZE=4096: larger chunks
        # (8192 with SG=64, 16384) overflow the UB -- the per-chunk total
        # reduction needs extra local buffer.
        n_chunks = triton.cdiv(n_elems, 4096)
        chunk_totals = torch.empty(n_chunks, dtype=torch.int32, device=device)
        _chunk_scan_kernel[(n_chunks,)](
            arr,
            part_sums,
            chunk_totals,
            n_elems,
            CHUNK_SIZE=4096,
            SCAN_GROUP_SIZE=_MAX_SCAN_GROUP,
        )
        chunk_offsets = torch.empty(n_chunks, dtype=torch.int32, device=device)
        scan_block2 = triton.next_power_of_2(n_chunks)
        _scan_kernel[(1,)](
            chunk_totals,
            chunk_offsets,
            n_chunks,
            BLOCK_SIZE=scan_block2,
            SCAN_GROUP_SIZE=min(_MAX_SCAN_GROUP, scan_block2),
        )
        _add_offsets_kernel[(n_chunks,)](
            part_sums,
            chunk_offsets,
            n_elems,
            CHUNK_SIZE=4096,
        )
    return part_sums


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------


def masked_scatter_backward(grad_output, mask, sizes):
    logger.debug("GEMS_ASCEND MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()
    device = grad_output.device

    if N == 0:
        return torch.zeros(numel, dtype=grad_output.dtype, device=device).view(sizes)

    # ~8*sqrt(N) lanes per block, rounded up to a power of two (int arithmetic:
    # triton.next_power_of_2 costs ~3us of host time, and the bounds are powers
    # of two, so rounding up and clamping commute).
    blk = 1 << (max(1, int(_BLOCK_SCALE * math.sqrt(N))) - 1).bit_length()
    BLOCK_SIZE = min(max(blk, _MIN_BLOCK_SIZE), _MAX_BLOCK_SIZE)
    wide = grad_output.dtype.itemsize == 4
    out = torch.empty(numel, dtype=grad_output.dtype, device=device)
    mask_flat = mask.ravel()

    # torch_npu's device context manager round-trips _npuSetDevice + _lazy_init
    # on every enter (11-22us of host time) but is a no-op when `device` is
    # already current, so only take it then.
    if device.index is not None and torch_device_fn.current_device() != device.index:
        ctx = torch_device_fn.device(device)
    else:
        ctx = _NULL_CTX
    with ctx:
        # count -> device scan -> expand -> tail-zero-fill, all Triton, no torch
        # compute ops and no stream syncs: launches on the current stream are
        # ordered, so each kernel sees the previous one's stores.
        exp_block = min(BLOCK_SIZE, _MAX_EXPAND_BLOCK)
        # int arithmetic instead of triton.cdiv (~3us each); the divisor is a
        # positive power of two.
        n_blocks = (N + exp_block - 1) // exp_block
        # The sort path zero-fills only the tail out[k:numel) and folds it into
        # the expand when the block grid spans the tail; the search path's
        # _count_rank kernel still zero-fills out[0:numel) densely, so its grid
        # covers max(N, numel).
        use_sort = _EXPAND_MODE == "sort" and al is not None
        # DO_TAIL needs the CTAs to span out[k:numel): numel <= n_blocks*exp_block
        # (true whenever `sizes` is the mask's own shape).  Given that the expand
        # can also work out the block counts itself, so no count kernel runs at
        # all.  The two launch-removing modes are mutually exclusive: FUSED is
        # the fallback for block counts too large for the redundant scan.
        merge_tail = use_sort and numel <= n_blocks * exp_block
        self_count = merge_tail and n_blocks * N <= _SELF_COUNT_MAX_ELEMS
        fused = merge_tail and not self_count and n_blocks <= _FUSED_MAX_BLOCKS
        if self_count:
            # No counts buffer exists; offsets_ptr is unused by the kernel.
            offsets = out
        elif use_sort:
            # all n_blocks+1 slots are written, so torch.empty suffices here
            counts = torch.empty(n_blocks + 1, dtype=torch.int32, device=device)
            _count_kernel[(n_blocks,)](
                mask_flat,
                counts,
                N,
                BLOCK_SIZE=exp_block,
                multibuffer=False,
            )
            offsets = counts if fused else _exclusive_scan(counts, n_blocks + 1, device)
        else:
            grid_c = (max(N, numel) + exp_block - 1) // exp_block
            # counts[grid_c] must stay 0: blocks beyond n_blocks never write it
            # and the scan needs it as the sentinel.
            counts = torch.zeros(grid_c + 1, dtype=torch.int32, device=device)
            inc = torch.empty(N, dtype=torch.int16, device=device)
            _count_rank_kernel[(grid_c,)](
                mask.ravel(),
                counts,
                inc,
                out,
                N,
                numel,
                BLOCK_SIZE=exp_block,
                SCAN_GROUP_SIZE=32,
                multibuffer=False,
            )
            offsets = _exclusive_scan(counts, n_blocks + 1, device)
        # The out arg must be an fp16 view, never bf16: a bf16 arg would make
        # the compiler CONVERT fp16->bf16 on store, corrupting the bits.
        if use_sort:
            _expand_sort_kernel[(n_blocks,)](
                grad_output.ravel().view(torch.float16),
                mask_flat,
                offsets,
                out.view(torch.int32) if wide else out.view(torch.float16),
                N,
                numel,
                BLOCK_SIZE=exp_block,
                WIDE_ELEM=wide,
                DO_TAIL=merge_tail,
                FUSED=fused,
                SELF_COUNT=self_count,
                NB_POW2=1 << (max(n_blocks, 1) - 1).bit_length(),
                num_warps=4,
            )
            if not merge_tail:
                # only the tail is left: k = offsets[n_blocks]
                _tail_zero_kernel[((numel + exp_block - 1) // exp_block,)](
                    out,
                    offsets,
                    n_blocks,
                    numel,
                    BLOCK_SIZE=exp_block,
                    multibuffer=False,
                )
        else:
            _expand_kernel[(n_blocks,)](
                grad_output.ravel().view(torch.float16),
                inc.view(torch.float16),
                offsets,
                out.view(torch.int32) if wide else out.view(torch.float16),
                N,
                numel,
                BLOCK_SIZE=exp_block,
                WIDE_ELEM=wide,
                num_warps=4,
            )

    return out.view(sizes)
