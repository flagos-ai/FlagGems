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
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ascend masked_scatter_backward
#
#   Semantics: out = cat([grad[mask], zeros]).view(sizes)
#     i.e.  out[j] = grad[idx[j]] for j < k, out[j] = 0 for k <= j < numel,
#     where idx = sorted True positions of mask and k = #True.
#
#   Measured on 910B4 (triton-ascend 3.x, CANN 9.0):
#     - dense masked store (zero-fill):       HBM bandwidth (~600 GB/s)
#     - scattered masked store:               ~2.5 ns per active lane +
#                                             ~0.3 ns per lane slot — 92.4%
#                                             aiv_mte2 in msprof; effective
#                                             bandwidth ~5 GB/s vs ~1.6 TB/s
#                                             dense.  There is no vscatter
#                                             in triton-ascend.
#     - tl.gather (cross-lane UB permutation): ~0.01 ns/lane — ~350x
#                                             cheaper than the scatter store
#     - tl.cumsum over BLOCK lanes:           very expensive on Ascend
#       (13.6 ms @16M for BLOCK=1024) — replaced by a grouped 2-D scan
#       (reshape (num_groups, SCAN_GROUP_SIZE), trans, cumsum over the
#       short axis, trans back), ~3x cheaper (same trick as nonzero_static)
#     - kernel launch ~57 us device time; cross-kernel dependencies are
#       ordered by same-stream launches — no syncs needed (removed
#       stream syncs cost ~0.13 ms EACH at 16M)
#
#   The large-N path therefore computes the EXPAND (inverse compaction) on
#   the output side instead of compacting with scattered stores:
#
#     1. _count_rank_kernel:  per-block True count + dense zero-fill of the
#        whole output + dense materialization of the block's INCLUSIVE rank
#        inc[lane] = (#True strictly before lane) + mask[lane].  inc is
#        monotone non-decreasing with values in [0, BLOCK], so the output
#        position r < cnt maps to the first lane with inc >= r+1 — found by
#        log2(BLOCK) rounds of binary search in step 3.  The rank scan runs
#        per _COUNT_TILE-lane tile in fp16 (half the UB traffic of int32;
#        every per-tile intermediate stays <= 2048, i.e. exact) with a
#        Hillis-Steele gather tree for the group offsets; the tile prefix
#        is added in int32 and inc is stored int16.  1.03 -> 0.50 ms @16M.
#     2. device-side exclusive scan of the counts (nb+1 elements so the
#        tail element equals k).
#     3. _expand_kernel:  for output r, binary-search inc via tl.gather
#        (the only cross-lane movement triton-ascend exposes, and by far
#        the cheapest), then one tl.gather of grad at the found lane, then
#        a dense prefix store.  No scattered stores at all: the kernel is
#        density-insensitive.
#
#   inc is stored as int16 (integers up to 32767 exact), read through its
#   fp16 view as raw bit patterns and searched in fp32 — the expand BLOCK
#   is capped at _MAX_EXPAND_BLOCK = 4096 (8192 overflows the UB).
#
#   4-byte grad traffic rides the fp16 view instead of fp32 loads:
#   bishengir's fp32 vector-load path runs at ~40 GB/s (1.7 ms @16M for
#   64 MB) while fp16 loads hit full MTE bandwidth; the two gathered
#   halfwords are recombined in-register into the fp32 bit pattern
#   (fp16->int16 bitcast per element, then lo | hi<<16) and written as one
#   contiguous int32 store through the int32 view of the fp32 output — no
#   host transpose copy, and zero extra traffic beyond the 2x grad read.
#   Measured and REJECTED alternatives: fp16->fp32 vector bitcast (the
#   frontend refuses unequal element widths); the interleaved stride-2
#   halfword store pair (blows the UB in bishengir — 2.6 Mbits for the
#   stores alone); a joined 4096-wide masked store (silently drops whole
#   blocks, never root-caused); the (2, numel) two-row layout plus host
#   transpose copy (correct but ~0.2 ms of extra traffic).
#   (A 2-D compare to precompute search anchors was measured and REJECTED:
#   the compiler's global-liveness tiling gives it ~3 ms, far more than
#   the 4 search rounds it saves.)
#
#   Compiler landmine (worked around): tl.gather over a vector DERIVED from
#   the grouped scan's reshape/transpose/cumsum chain inside a binary-search
#   dependency loop crashes the vector core at runtime (npuSynchronizeDevice
#   error, 0x31 vector core exception).  Gather over loaded or where-derived
#   vectors is fine; the dependency loop alone is fine.  Hence inc must be
#   round-tripped through HBM (written in kernel 1, loaded in kernel 3).
#   Materializing costs ~64 MB of extra traffic @16M (~0.08 ms) and
#   additionally saves the grouped scan from running twice.
#
#   n_blocks <= 128 (and numel <= N): single kernel — each program re-reads
#   earlier mask blocks (dense vector loads, torch-written -> no sync
#   needed) for its exclusive offset, scatters its block, and zero-fills its
#   own slice of the disjoint tail [total, numel).  Small N is launch-bound
#   anyway, and the reread path avoids the extra launches.  No scf.if
#   regions: ttadapter cannot compile a kernel that combines a scattered
#   store fed by the grouped scan's transpose/reshape chain with an if
#   region containing stores.
#
#   Note on benchmark numbers: the op benchmark builds its mask as
#   randn(...) < 0.3, whose TRUE density is Phi(0.3) ~= 0.62, not 0.3.
#   The old scatter path was linear in the density (36.7 ms @0.62);
#   the expand path is density-insensitive (3.3 ms for any density).
#
#   Hardware-accelerated alternatives measured and REJECTED on this stack:
#     - tl.sort  (bitonic, hypercube reshape): UB overflow — 5.3 Mbits
#       required at BLOCK=128 vs 1.5 Mbits available.
#     - index_put / gather_out_to_ub (CANN extension mem_ops): implemented
#       only in the interpreter; no lowering in the compiled path
#       ("failed to legalize unresolved materialization").
#     - torch_npu.npu_scatter: single-op GE build fails (ArgMaxGrad
#       fusion pass error); torch_npu.scatter_update segfaults;
#       torch_npu.npu_scatter_nd_update is correct but 489 ms @16M
#       (~13x slower than the triton scatter — no vscatter there).
#     - warps {2,4,8} x SCAN_GROUP {64,128} x int32/int64 store mask on the
#       old scatter kernel: all within noise (36.2-37.8 ms).
# ---------------------------------------------------------------------------

_MIN_BLOCK_SIZE = 128
_MAX_BLOCK_SIZE = 4096
_MAX_EXPAND_BLOCK = 4096  # expand/count block ceiling; 8192 overflows the UB
_COUNT_TILE = 2048  # count kernel scans per-TILE in fp16 (exact <= 2048)
_MAX_SCAN_BLOCK = 16384  # single-launch grouped scan upper bound (int32)
_MAX_SCAN_GROUP = 128
_MAX_REREAD_BLOCKS = 128
_REREAD_CHUNK = 4096  # (must be restated literally inside jit kernels)
_TARGET_BLOCKS = 40  # aim for ~#cores blocks; BLOCK_SIZE capped at 4096


@triton.jit
def _grouped_excl_scan(values, BLOCK_SIZE: tl.constexpr, SCAN_GROUP_SIZE: tl.constexpr):
    """Exclusive scan of a BLOCK_SIZE vector via a 2-D grouped scan.

    reshape -> (num_groups, SCAN_GROUP_SIZE), transpose, cumsum over the
    short axis (SCAN_GROUP_SIZE in parallel lanes), transpose back, add the
    exclusive group prefix.  ~3x cheaper than a plain tl.cumsum over
    BLOCK_SIZE lanes on Ascend.
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
    LOG_GROUPS: tl.constexpr,
):
    """fp16 exclusive scan: 2-D grouped scan + Hillis-Steele group offsets.

    Same 2-D structure as _grouped_excl_scan, but all values are fp16
    (group sums <= SCAN_GROUP_SIZE are exact, as is any intermediate <=
    2048), which halves the UB traffic of the two transposes, and the
    num_groups-element offset cumsum is replaced by log2(num_groups) steps
    of 32-lane tl.gather.  Measured 2x faster than the int32 version in the
    count kernel (1.03 ms -> 0.50 ms @16M).

    Also returns acc = the inclusive scan of the group counts, in case the
    caller needs it.
    """
    num_groups: tl.constexpr = BLOCK_SIZE // SCAN_GROUP_SIZE
    grouped = tl.reshape(values, (num_groups, SCAN_GROUP_SIZE))
    transposed = tl.trans(grouped, (1, 0))
    within_group = tl.cumsum(transposed, axis=0) - transposed
    within_group = tl.trans(within_group, (1, 0))
    group_counts = tl.sum(grouped, axis=1)
    # inclusive scan of the num_groups counts via log2(num_groups) gather
    # steps (Hillis-Steele); fp16 gather source — values stay <= 2048
    acc = group_counts
    for k in tl.static_range(LOG_GROUPS):
        idx = tl.arange(0, num_groups) - (1 << k)
        shifted = tl.gather(acc, tl.maximum(idx, 0), axis=0)
        shifted = tl.where(tl.arange(0, num_groups) >= (1 << k), shifted, 0)
        acc = acc + shifted
    group_offsets = acc - group_counts  # exclusive
    return (tl.reshape(within_group + group_offsets[:, None], (BLOCK_SIZE,)), acc)


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
    LOG_GROUPS: tl.constexpr,
):
    """Per-CTA True count + dense zero-fill + dense inclusive-rank store.

    inc[lane] = (#True strictly before lane) + mask[lane] within this
    BLOCK_SIZE block is monotone non-decreasing — the binary-search array
    of the expand kernel.  Stored int16: exact for BLOCK_SIZE <= 4096
    (fp16's exact range ends at 2048, hence int16).  The scan runs per
    _COUNT_TILE-lane tile in fp16 (_scan_f16_gather) — half the UB traffic
    of int32, and every per-tile intermediate stays <= 2048, i.e. exact;
    the tile prefix accumulates in int32 and is added in int32 before the
    int16 store.

    The zero-fill is done here rather than in a separate launch, and
    densely — the expand kernel later overwrites the selected prefix, so it
    never reads the zeros (no read-visibility dependency).
    """
    pid = ext.program_id(axis=0)
    TILE: tl.constexpr = 2048
    NTILES: tl.constexpr = BLOCK_SIZE // TILE
    prefix = tl.cast(0, tl.int32)
    for t in tl.static_range(NTILES):
        offsets = pid * BLOCK_SIZE + t * TILE + tl.arange(0, TILE)
        m = offsets < N
        mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
        mask16 = mask_val.to(tl.float16)
        rank16, e16 = _scan_f16_gather(mask16, TILE, SCAN_GROUP_SIZE, LOG_GROUPS)
        inc = (prefix + rank16.to(tl.int32) + mask_val).to(tl.int16)
        tl.store(inc_ptr + offsets, inc, mask=m)
        prefix += tl.sum(mask_val, axis=0)
    tl.store(counts_ptr + pid, prefix)

    # Dense zero-fill of out[0:numel) — covered by grid = cdiv(max(N, numel), BLOCK)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, 0.0, mask=(offsets < numel))


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
    LOG_REST: tl.constexpr,
    WIDE_ELEM: tl.constexpr,
):
    """Expand block b into the dense output prefix [off[b], off[b+1]).

    For output position r < cnt the source lane is the first lane whose
    inclusive rank is >= r+1: log2(BLOCK_SIZE) rounds of binary search
    over inc via tl.gather (the only cross-lane movement triton-ascend
    exposes), round 0 being a 0-d scalar load (mid is the same for every
    lane), then the grad gather at the found lane.  The store is a dense
    prefix write, so the kernel is insensitive to mask density.

    inc is int16 (block-local ranks, exact up to 32767) read through its
    fp16 view as raw bit patterns and widened to fp32 (int16->fp32 is
    exact); lanes beyond N get the padding bit pattern 65535 (decodes as
    fp16 65504) to keep the array monotone.  The search runs in fp32 —
    required for BLOCK_SIZE > 2048 where fp16's exact-integer range ends.

    For 4-byte elements (WIDE_ELEM) all grad/output traffic runs in fp16
    halfword pairs: grad is read through its fp16 view (2*BLOCK halfwords)
    and the two gathered halfwords are recombined into the fp32 bit pattern
    in-register (fp16->int16 bitcast per element, then lo | hi<<16) and
    stored as ONE contiguous int32 store through the int32 view of the fp32
    output — no host transpose.  A direct fp32 vector load of grad takes
    ~1.7 ms @16M (bishengir generates a ~40 GB/s access path for it),
    while the fp16-view load runs at full MTE bandwidth (~0.1 ms).  The
    halfword split/recombine is pure bit reinterpretation — no precision
    loss.  (Measured rejects: fp16->fp32 vector bitcast is refused by the
    frontend — element widths must match; the interleaved stride-2
    halfword store pair blows the UB; a joined 4096-wide masked store
    silently drops blocks; the (2, numel) two-row layout plus host
    transpose copy works but costs ~0.2 ms extra traffic.)  2-byte
    elements load/store one halfword per element directly.

    inc must come from HBM (materialized by _count_rank_kernel): tl.gather
    over a grouped-scan-derived vector inside a search dependency loop
    crashes the vector core at runtime (ttadapter bug; see the module
    docstring).  The grad gathers use lane*2 indices inside the same
    dependency loop, but their source is a plain load — safe.
    """
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
    # Round 0: mid identical for all lanes — 0-d scalar load, no gather.
    # Masked: a tail block shorter than mid0+1 would read past inc;
    # other=65535 keeps the padded array monotone (v0 >= any target).
    mid0 = BLOCK_SIZE // 2
    v0 = tl.load(
        inc_ptr + pid * BLOCK_SIZE + mid0, mask=pid * BLOCK_SIZE + mid0 < N, other=65535
    )
    v0 = v0.to(tl.int16, bitcast=True).to(tl.float32)
    lo = tl.where(v0 >= target, 0, mid0 + 1)
    hi = tl.where(v0 >= target, mid0, BLOCK_SIZE - 1)
    for _ in tl.static_range(LOG_REST):
        mid = (lo + hi) >> 1
        mid_safe = tl.minimum(mid, BLOCK_SIZE - 1)
        vmid = tl.gather(inc, mid_safe, axis=0)
        ge = vmid >= target
        lo = tl.where(ge, lo, mid + 1)
        hi = tl.where(ge, mid, hi)
    lane = tl.minimum(lo, BLOCK_SIZE - 1)
    # r >= cnt converges at a wrong lane; the store mask below discards it.
    mst = (r_i < cnt) & ((out_off + r_i).to(tl.int64) < numel)
    if WIDE_ELEM:
        offsets16 = pid * (2 * BLOCK_SIZE) + tl.arange(0, 2 * BLOCK_SIZE)
        g16 = tl.load(grad16_ptr + offsets16, mask=offsets16 < 2 * N, other=0)
        glo = tl.gather(g16, lane * 2, axis=0)
        ghi = tl.gather(g16, lane * 2 + 1, axis=0)
        # Recombine the halfword pair into the fp32 bit pattern in-register
        # (fp16->int16 bitcast per element — the int32 casts sign-extend,
        # so mask first — then lo | hi<<16) and store through the int32
        # view of the fp32 output: one contiguous store, no host transpose.
        lo32 = glo.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        hi32 = (ghi.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) << 16
        tl.store(out_ptr + out_off + r_i, lo32 | hi32, mask=mst)
    else:
        g16 = tl.load(grad16_ptr + offsets, mask=m, other=0)
        gv = tl.gather(g16, lane, axis=0)
        tl.store(out_ptr + out_off + r_i, gv, mask=mst)


@libentry()
@triton.jit(do_not_specialize=["N", "numel", "NB"])
def _scatter_reread_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    N,
    numel,
    NB,
    BLOCK_SIZE: tl.constexpr,
    SCAN_GROUP_SIZE: tl.constexpr,
):
    """Single-launch multi-CTA scatter + zero-fill, no control-flow regions.

    Program p computes its exclusive offset = sum(mask[0 : p*BLOCK)) with
    dense vector chunk loads (reads only torch-written data, so no scan
    kernel and no sync are needed), scatters its block, and zero-fills its
    own slice of the output [pid*BLOCK, (pid+1)*BLOCK) intersected with
    [total, numel) — that region is disjoint from every scatter write
    (which lands in [0, total)), so there is no cross-program race.

    Every program also sums the whole mask to obtain `total` (grid <= 128,
    N <= ~0.5M here, so the per-program re-read is cheap).

    Note: the scatter store's address derives from the grouped scan's
    transpose/reshape chain.  triton-ascend's ttadapter fails to compile
    any kernel that combines such a scattered store with an scf.if region
    containing stores (e.g. a `if pid == NB-1:` tail block) — all simple
    reformulations were tried and fail.  Hence this kernel deliberately
    avoids all if/loop control flow around the zero-fill: slices, bounds
    and masks are computed arithmetically (empty loops simply don't run).
    """
    pid = ext.program_id(axis=0)
    CHUNK: tl.constexpr = 4096  # dense chunk for offset re-read
    total_prev = pid * BLOCK_SIZE
    off = tl.cast(0, tl.int32)
    n_chunks = total_prev // CHUNK
    for i in range(n_chunks):
        offs = i * CHUNK + tl.arange(0, CHUNK)
        off += tl.sum(
            tl.load(mask_ptr + offs, mask=offs < N, other=0).to(tl.int32), axis=0
        )
    rem_start = n_chunks * CHUNK
    rem = total_prev - rem_start
    if rem > 0:
        offs = rem_start + tl.arange(0, CHUNK)
        off += tl.sum(
            tl.load(mask_ptr + offs, mask=(offs < N) & (offs < total_prev), other=0).to(
                tl.int32
            ),
            axis=0,
        )

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=m, other=0).to(tl.int32)
    grad_val = tl.load(grad_ptr + offsets, mask=m, other=0)
    rank = _grouped_excl_scan(mask_val, BLOCK_SIZE, SCAN_GROUP_SIZE)
    pos = off + rank
    tl.store(
        out_ptr + pos,
        grad_val,
        mask=m & (mask_val == 1) & (pos.to(tl.int64) < numel),
    )

    # total = sum of the whole mask (every program computes the same value)
    total = tl.cast(0, tl.int32)
    n_tc = N // CHUNK
    for i in range(n_tc):
        offs = i * CHUNK + tl.arange(0, CHUNK)
        total += tl.sum(
            tl.load(mask_ptr + offs, mask=offs < N, other=0).to(tl.int32),
            axis=0,
        )
    trem = N - n_tc * CHUNK
    if trem > 0:
        offs = n_tc * CHUNK + tl.arange(0, CHUNK)
        total += tl.sum(
            tl.load(mask_ptr + offs, mask=offs < N, other=0).to(tl.int32),
            axis=0,
        )

    # Zero-fill this program's output slice [pid*BLOCK, (pid+1)*BLOCK)
    # intersected with [total, numel); disjoint from all scatter writes.
    lo = tl.maximum(pid * BLOCK_SIZE, total)
    hi = tl.minimum((pid + 1) * BLOCK_SIZE, numel)
    n_zc = (hi - lo) // CHUNK
    for j in range(n_zc):
        zoffs = lo + j * CHUNK + tl.arange(0, CHUNK)
        tl.store(out_ptr + zoffs, 0.0, mask=zoffs < hi)
    zrem = lo + n_zc * CHUNK
    if zrem < hi:
        zoffs = zrem + tl.arange(0, CHUNK)
        tl.store(out_ptr + zoffs, 0.0, mask=(zoffs < hi) & (zoffs >= lo))


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
    if n_elems <= _MAX_SCAN_BLOCK:
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
        # (8192 with SG=64, 16384) overflow the UB in bishengir because the
        # per-chunk total reduction needs extra local buffer.
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

    BLOCK_SIZE = bracket_next_power_of_2(
        triton.cdiv(N, _TARGET_BLOCKS),
        _MIN_BLOCK_SIZE,
        _MAX_BLOCK_SIZE,
    )
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    scan_group = min(_MAX_SCAN_GROUP, BLOCK_SIZE)
    wide = grad_output.dtype.itemsize == 4
    out = torch.empty(numel, dtype=grad_output.dtype, device=device)

    with torch_device_fn.device(device):
        if n_blocks <= _MAX_REREAD_BLOCKS and numel <= N:
            # Single launch: each program re-reads earlier mask blocks for its
            # exclusive offset, scatters its block, and zero-fills its own
            # slice of the disjoint tail [total, numel).  numel > N falls to
            # the scan path: the reread kernel's grid would have to cover
            # numel programs, each re-summing the whole mask.  Writes `out`
            # in its native dtype (no fp16-view tricks) and returns early.
            grid_r = max(n_blocks, triton.cdiv(numel, BLOCK_SIZE))
            _scatter_reread_kernel[(grid_r,)](
                grad_output.ravel(),
                mask.ravel(),
                out,
                N,
                numel,
                n_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
                SCAN_GROUP_SIZE=scan_group,
                num_warps=4,
            )
            return out.view(sizes)
        # count(+zero-fill+rank materialization) -> device scan -> expand.
        # All Triton, no torch compute ops, and no stream syncs: launches on
        # the current stream are ordered, so each kernel sees the previous
        # one's stores (the syncs this code used to take cost ~0.13 ms each
        # at 16M; no other op in this backend syncs between kernels).
        exp_block = min(BLOCK_SIZE, _MAX_EXPAND_BLOCK)
        n_blocks = triton.cdiv(N, exp_block)
        # The count kernel zero-fills out[0:numel) densely (fp32 zeros =
        # int32 zeros, so the WIDE int32 store sees a zeroed tail).  Grid
        # covers the larger of N (counting) and numel (zero-fill) so the
        # whole output is zeroed densely even when numel > N.
        grid_c = triton.cdiv(max(N, numel), exp_block)
        # counts[grid_c] is never written (blocks beyond n_blocks have
        # mask==0) and stays 0, so scanning n_blocks+1 elements yields
        # offsets[0..n_blocks] with the tail element equal to k.
        counts = torch.zeros(grid_c + 1, dtype=torch.int32, device=device)
        # int16 block-local ranks (exact <= 4096; the expand kernel reads
        # them via the fp16 view as raw bit patterns)
        inc = torch.empty(N, dtype=torch.int16, device=device)
        assert exp_block >= _COUNT_TILE and exp_block % _COUNT_TILE == 0
        _count_rank_kernel[(grid_c,)](
            mask.ravel(),
            counts,
            inc,
            out,
            N,
            numel,
            BLOCK_SIZE=exp_block,
            SCAN_GROUP_SIZE=32,
            LOG_GROUPS=int(math.log2(_COUNT_TILE // 32)),
            multibuffer=False,
        )
        # same-stream ordering: the scan sees the counts, the expand sees
        # the offsets
        offsets = _exclusive_scan(counts, n_blocks + 1, device)
        # Store pointers: WIDE writes the recombined int32 halfword pairs
        # through the int32 view of the fp32 out; narrow stores fp16
        # halfword bit patterns, and a bf16 tensor arg would make the
        # compiler CONVERT fp16->bf16 on store, corrupting the bits —
        # hence the fp16 view (the arg dtype must match the store type).
        _expand_kernel[(n_blocks,)](
            grad_output.ravel().view(torch.float16),
            inc.view(torch.float16),
            offsets,
            out.view(torch.int32) if wide else out.view(torch.float16),
            N,
            numel,
            BLOCK_SIZE=exp_block,
            LOG_REST=int(math.log2(exp_block)),
            WIDE_ELEM=wide,
            num_warps=4,
        )

    return out.view(sizes)
