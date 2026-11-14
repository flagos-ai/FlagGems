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
import threading
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# Maximum number of tiles the single-CTA exclusive scan can handle.
# next_power_of_2(P) must fit the 65 536 per-axis CTA launch limit, so
# P must be ≤ 32768 (num_tasks ≤ 268_435_456). Above that we fall back
# to torch.cumsum on the device (tile_counts is at most a few ×10^4
# int32 values — negligible cost).
_MAX_SINGLE_CTA_SCAN_TILES = 32768

# Post-sort pipeline launch geometry (module-level so the ablation runner
# can patch them; defaults match the values this pipeline was tuned with).
# Async out-size probe: a process-wide pinned staging buffer + event so the
# host can read the unique-count as soon as the tiny scan kernel finishes,
# while the (much longer) scatter/counts kernels are still executing. This
# removes the .item() full-stream sync from the op's critical path: the op
# returns before the stream drains, exactly like native aten ops. Not
# thread-safe by design (benchmark/eval loops are single-threaded per device);
# keyed by device index for multi-device correctness.
# ---------------------------------------------------------------------------
# Host-side resource pools.
#
# The size probe: the kernel writes (total, ready-flag) directly into a pinned
# int64[2] buffer (single-thread, program-ordered stores on one cacheline).
# The host spins on the flag word — measured ~41 us cheaper per call than
# copy_ + Event.synchronize on this stack — with an event-sync fallback for
# the pathological case where the writing kernel never runs. Pinned kernel
# stores + spin visibility + intra-cacheline ordering were validated on
# MTT S5000 (tools/validate_final_optims.py).
#
# Scratch buffers (tile metadata, idx, device total) are never returned to
# callers, so they are pooled per (kind, device, geometry) and reused under
# an event-protected lease: a lease is reusable only after its completion
# event (recorded after the last kernel of the previous call) has fired.
# This is safe across streams AND host threads — concurrent callers get
# distinct leases; a still-in-flight lease is never handed out.
# ---------------------------------------------------------------------------

_probe_pool = {}  # device_index -> [_ProbeEntry]
_probe_lock = threading.Lock()
_PROBE_SPIN_TIMEOUT_MS = 200.0
# ready-word encoding: (unique_count_minus_1 << 32) | MAGIC. The producer
# makes ONE aligned 8-byte store, so the host never observes a torn or
# reordered (flag-without-total) pair — value and readiness are the same
# word. A store of a single aligned machine word is atomic end-to-end
# (one PCIe posted-write TLP committed by the root complex), which is the
# standard flag-word pattern; the earlier two-store variant additionally
# relied on same-thread store ordering, which was validated empirically
# but is not a documented MUSA guarantee.
_PROBE_MAGIC = 0x7A5A7A5A
# ready-word encoding: (total << 32) | (MAGIC ^ total). The low half
# redundantly encodes the total, so a word assembled from two DIFFERENT
# writes (possible if the backend ever splits the i64 store into ordered
# 32-bit halves — the LLIR says `store i64, align 4`, which permits it)
# fails validation and the host simply keeps spinning until both halves
# come from the same write (sequence-lock pattern). This makes the
# protocol safe without relying on undocumented single-access guarantees
# of the MUSA backend.


class _ProbeEntry:
    # Lifecycle: ACQUIRED -> ARMED (host reset) -> SUBMITTED (producer
    # launched) -> COMPLETED (host observed the word) -> RELEASED. On the
    # normal path no completion event is needed: the pinned word's ONLY
    # device access is the producer's single store, and a COMPLETED entry
    # has already been read by the host, so nothing device-side can still
    # be in flight against the buffer. The EXCEPTION path (producer
    # submitted but the call failed before the wait) must NOT release the
    # entry outright: the producer's store may still be queued and would
    # land after a future caller arms the word, feeding that caller a
    # stale length (reproduced with real device back-pressure). The caller
    # therefore synchronizes the device before releasing on that cold
    # path, or quarantines the entry if even the sync fails.
    __slots__ = ("pinned", "np", "in_use", "device_index")

    def __init__(self, device_index):
        # single int64 word: 0 = write pending, otherwise the packed
        # (total << 32) | MAGIC value written by the producer kernel
        self.pinned = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self.np = self.pinned.numpy()
        self.in_use = False
        self.device_index = device_index


def _probe_acquire(device_index):
    with _probe_lock:
        pool = _probe_pool.setdefault(device_index, [])
        for e in pool:
            if not e.in_use:
                e.in_use = True
                return e
        e = _ProbeEntry(device_index)
        e.in_use = True
        pool.append(e)
        return e


def _probe_arm(entry):
    """Host-side reset before launch: a plain numpy store, no dispatch.

    The entry is lease-protected and the previous call already consumed
    the previous word (it waited on it), so this store races nothing; it
    is program-ordered before the kernel launch, and the producer is the
    only writer while the entry is checked out. Keeping every kernel
    argument value-stable matters here too: libentry's dispatch key
    includes the RAW VALUE of non-tensor arguments, so a per-call stamp
    would break the kernel cache on every call.
    """
    entry.np[0] = 0


def _probe_wait(entry):
    """Spin until the producer's packed ready word lands, then decode it.

    Falls back to event synchronization after the timeout (kernel fault or
    hang); event completion formally guarantees visibility of everything
    the producer stream wrote before the record. A word without the magic
    after that point means the producer never wrote this call's result —
    raise instead of ever returning a stale length.
    """
    np_buf = entry.np
    deadline = time.perf_counter() + _PROBE_SPIN_TIMEOUT_MS * 1e-3
    while True:
        v = int(np_buf[0])
        if v != 0:
            total = v >> 32
            if ((v & 0xFFFFFFFF) ^ _PROBE_MAGIC) == total:
                return total
            # torn read (halves from different writes) — transient; keep
            # spinning until both halves come from the same write
        if time.perf_counter() > deadline:
            # pathological path (producer fault/hang): a full synchronize
            # of the PRODUCING device (not necessarily the current one)
            # formally guarantees visibility of everything enqueued before
            # it; re-read and validate once more
            torch.musa.synchronize(entry.device_index)
            v = int(np_buf[0])
            total = v >> 32
            if v != 0 and ((v & 0xFFFFFFFF) ^ _PROBE_MAGIC) == total:
                return total
            raise RuntimeError(
                "unique2 size probe: producer did not publish a valid "
                "ready word (kernel fault or unexpected overwrite); "
                "refusing to return a stale length"
            )


_scratch_pool = {}  # key -> [_Lease]
# distinct-shape cap; pools of completed entries are evicted beyond it.
# Worst-case cached bytes = cap x per-key scratch (~numel*8 for idx plus
# small tile metadata), i.e. at most a few times the caller's own input
# footprint for the largest cached shape — bounded without a separate
# byte budget, which would add per-call bookkeeping to the hot path.
_SCRATCH_POOL_MAX_KEYS = 64
_scratch_lock = threading.Lock()


class _Lease:
    __slots__ = ("bufs", "event", "in_use")

    def __init__(self, bufs):
        self.bufs = bufs
        self.event = torch.musa.Event()
        self.in_use = True


def _lease_acquire(key, alloc):
    with _scratch_lock:
        pool = _scratch_pool.setdefault(key, [])
        for e in pool:
            if not e.in_use and e.event.query():
                e.in_use = True
                return e
        e = _Lease(alloc())
        pool.append(e)
        if len(_scratch_pool) > _SCRATCH_POOL_MAX_KEYS:
            _evict_completed_pools()
        return e


def _evict_completed_pools():
    """Bound pool growth across many distinct shapes.

    Evicts oldest-inserted whole pools whose entries are all complete
    (event fired, not in use) so the caching allocator can reclaim the
    buffers; in-flight pools are never dropped, so the cap may be exceeded
    transiently under concurrency.
    """
    for k in list(_scratch_pool):
        if len(_scratch_pool) <= _SCRATCH_POOL_MAX_KEYS:
            break
        entries = _scratch_pool[k]
        if all(not x.in_use and x.event.query() for x in entries):
            del _scratch_pool[k]


_dummy_i64_cache = {}


def _dummy_i64(device):
    """Shared, never-written 1-element int64 filler for dead kernel args."""
    key = device.index
    d = _dummy_i64_cache.get(key)
    if d is None:
        d = torch.empty((1,), dtype=torch.int64, device=device)
        _dummy_i64_cache[key] = d
    return d


def _lease_release(entry):
    entry.in_use = False


_POST_SORT_TILE_SIZE = 8192
_POST_SORT_BOUNDARY_WARPS = 8
_POST_SORT_SCAN_WARPS = 32
_POST_SORT_SCATTER_WARPS = 8
_POST_SORT_META_WARPS = 8
# the single-CTA metadata kernel's sequential read beats the multi-CTA
# boundary + scan pair only while the tile count (and thus the read) is
# small; measured crossover on MTT S5000 is between 8 and 32 tiles
_META_KERNEL_MAX_TILES = 8


@libentry()
@triton.jit
def simple_unique_flat_kernel(
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    total_ptr: tl.tensor,  # out (device, groups - 1; counts-kernel bound)
    probe_ptr: tl.tensor,  # out (pinned int64[2]: [flag, total])
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks

    # load
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # ne & cumsum
    ne_result = tl.where(i0 > 0, a != b, 0)
    cumsum = tl.cumsum(ne_result)

    # device total (valid-length bound for the counts kernel) and the
    # pinned size probe: one lane makes ONE aligned 8-byte store of the
    # packed ready word (total << 32 | MAGIC) — value and readiness are the
    # same word, so the host can never observe a torn or reordered pair.
    # All kernel arguments stay value-stable across calls (libentry caches
    # by raw non-tensor values).
    total_mask = i0 == tile_size - 1
    tl.store(total_ptr + tl.zeros_like(i0), cumsum, mask=total_mask)
    tl.store(
        probe_ptr + tl.zeros_like(i0),
        (cumsum.to(tl.int64) << 32) + (0x7A5A7A5A ^ cumsum),
        mask=total_mask,
    )

    # data_out: scatter_(to=cumsum, sorted_data); only run heads write.
    # Non-head lanes of an equal-value run would store the same value to the
    # same address (duplicate-address write). Same value in practice, but a
    # data race per the kernel memory model (cf. upstream PR #6258).
    boundary = ((i0 == 0) | ne_result.to(tl.int1)) & mask
    tl.store(data_out_ptr + cumsum, a, mask=boundary)

    # inverse_indices: scatter_(to=sorted_indices, cumsum)
    if return_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    # idx: run-head positions; the counts kernel reads them back with a
    # device-side valid-length mask (total), so no sentinel is needed.
    # (An in-kernel atomic-add accumulation of counts measured far slower
    # on this stack — MUSA atomics do not parallelize this pattern.)
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)


@triton.jit
def output_counts_flat_impl(
    global_pid,
    idx_ptr: tl.tensor,
    size_ptr: tl.tensor,  # in (device, groups - 1; written by the producer kernel)
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)

    # device-side valid length: only idx slots [0, out_size) are
    # initialized (run heads). Slots beyond that are pooled scratch and
    # must never be loaded — mask everything by out_size instead of the
    # N-sized launch bound.
    out_size = tl.load(size_ptr) + 1
    i0 = global_pid * tile_size + r
    mask = i0 < out_size
    idx = tl.load(idx_ptr + i0, mask=mask, other=0)

    # load idx_next
    i0_next = i0 + 1
    next_mask = i0_next < out_size
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask, other=0)

    # diff
    counts = tl.where(i0_next < out_size, idx_next - idx, origin_num_tasks - idx)

    # store counts
    tl.store(counts_ptr + i0, counts, mask=mask)


@libentry()
@triton.jit
def output_counts_flat_kernel(
    idx_ptr: tl.tensor,
    size_ptr: tl.tensor,  # in
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        output_counts_flat_impl(
            global_pid,
            idx_ptr,
            size_ptr,
            origin_num_tasks,  # in
            counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@triton.jit
def tile_boundary_counts_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    tile_counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    i0 = global_pid * tile_size + tl.arange(0, tile_size)
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)
    # a boundary is the first position of each group; position 0 is one
    ne_result = tl.where(i0 > 0, a != b, 0)
    tl.store(tile_counts_ptr + global_pid, tl.sum(ne_result.to(tl.int32), axis=0))


@libentry()
@triton.jit
def tile_boundary_counts_kernel(
    sorted_data_ptr: tl.tensor,  # in
    tile_counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        tile_boundary_counts_impl(
            global_pid,
            sorted_data_ptr,  # in
            tile_counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@libentry()
@triton.jit
def tile_counts_scan_kernel(
    tile_counts_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # out (exclusive prefix, group id of the tile)
    total_ptr: tl.tensor,  # out (number of groups - 1)
    probe_ptr: tl.tensor,  # out (pinned int64[2]: [flag, total])
    num_tiles: int,
    next_power_num_tiles: tl.constexpr,
):
    # single-CTA exclusive scan over per-tile group counts.
    # next_power_of_2(P) must fit the 65 536 per-axis CTA limit, so
    # P must be ≤ 32768 (num_tasks ≤ 268_435_456). Above that we fall back
    # to a torch.cumsum on the device (see _unique_post_sort_impl).
    r = tl.arange(0, next_power_num_tiles)
    mask = r < num_tiles
    counts = tl.load(tile_counts_ptr + r, mask=mask, other=0).to(tl.int32)
    cumsum = tl.cumsum(counts, axis=0)
    tl.store(tile_prefix_ptr + r, cumsum - counts, mask=mask)
    # the last group id is total groups - 1; the scan total is that value.
    # One lane publishes it to the device total (counts-kernel valid mask)
    # and then to the pinned size probe (total first, stamp second).
    total = tl.sum(tl.where(mask, counts, 0), axis=0)
    total_mask = r == num_tiles - 1
    tl.store(total_ptr + tl.zeros_like(r), total, mask=total_mask)
    tl.store(
        probe_ptr + tl.zeros_like(r),
        (total.to(tl.int64) << 32) + (0x7A5A7A5A ^ total),
        mask=total_mask,
    )


@triton.jit
def unique_scatter_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    sorted_indices_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,  # out
    inverse_indices_ptr: tl.tensor,  # out
    idx_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
    write_unique: tl.constexpr,
    write_inverse: tl.constexpr,
    write_idx: tl.constexpr,
):
    i0 = global_pid * tile_size + tl.arange(0, tile_size)
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)
    # a boundary is the first position of each group; position 0 is one
    ne_result = tl.where(i0 > 0, a != b, 0).to(tl.int32)
    cumsum = tl.cumsum(ne_result, axis=0)
    group_id = tl.load(tile_prefix_ptr + global_pid) + cumsum
    boundary = ((i0 == 0) | (ne_result != 0)) & mask

    if write_unique:
        tl.store(data_out_ptr + group_id, a, mask=boundary)
    if write_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, group_id, mask=mask)
    if write_idx:
        tl.store(idx_ptr + group_id, i0, mask=boundary)


@libentry()
@triton.jit
def unique_scatter_kernel(
    sorted_data_ptr: tl.tensor,  # in
    sorted_indices_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,  # out
    inverse_indices_ptr: tl.tensor,  # out
    idx_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    write_unique: tl.constexpr,
    write_inverse: tl.constexpr,
    write_idx: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        unique_scatter_impl(
            global_pid,
            sorted_data_ptr,  # in
            sorted_indices_ptr,  # in
            tile_prefix_ptr,  # in
            data_out_ptr,  # out
            inverse_indices_ptr,  # out
            idx_ptr,  # out
            num_tasks,
            tile_size,
            write_unique,
            write_inverse,
            write_idx,
        )


@libentry()
@triton.jit
def post_sort_meta_kernel(
    sorted_data_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # out (exclusive prefix per tile)
    total_ptr: tl.tensor,  # out (device, groups - 1; counts-kernel bound)
    probe_ptr: tl.tensor,  # out (pinned int64[2]: [flag, total])
    num_tasks: int,
    num_tiles: int,
    tile_size: tl.constexpr,
):
    # Single-CTA metadata pass: counts run boundaries per tile and writes
    # each tile's exclusive prefix, the device total and the pinned probe.
    # Replaces the multi-CTA boundary kernel + single-CTA scan pair (one
    # Python launch saved; the N-element read is bandwidth-trivial for a
    # single CTA at these sizes).
    running = tl.zeros([1], dtype=tl.int32)
    for t in range(0, num_tiles):
        i0 = t * tile_size + tl.arange(0, tile_size)
        mask = i0 < num_tasks
        a = tl.load(sorted_data_ptr + i0, mask=mask, other=0)
        i0_prev = tl.where(i0 > 0, i0 - 1, 0)
        b = tl.load(sorted_data_ptr + i0_prev, mask=mask, other=0)
        ne_result = tl.where(i0 > 0, a != b, 0).to(tl.int32)
        tl.store(tile_prefix_ptr + t + tl.arange(0, 1), running)
        running = running + tl.sum(ne_result, axis=0)
    one = tl.arange(0, 1)
    tl.store(total_ptr + one, running)
    tl.store(
        probe_ptr + one,
        (running.to(tl.int64) << 32) + (0x7A5A7A5A ^ running),
    )


def _unique_post_sort_impl(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    """Dedup + group-id + scatter over already-sorted data.

    Pipeline: native sort -> single-CTA metadata pass (tile prefixes +
    total + pinned probe) -> scatter -> counts. The metadata kernel writes
    the unique total straight into a pinned probe that the host spin-waits
    on, so the op returns before the stream drains (async-return
    semantics, same as native aten ops).
    """
    num_tasks = sorted_data.numel()
    tile_size = _POST_SORT_TILE_SIZE
    num_tiles = triton.cdiv(num_tasks, tile_size)
    grid = (num_tiles, 1, 1)
    device = sorted_data.device

    # tile_prefix / total hold group ids and (num_groups-1), both ≤
    # num_tasks-1, so int32 is safe while num_tasks < 2^31. The meta-kernel
    # path caps num_tiles at 32768 (num_tasks ≤ 2^28); beyond that the
    # device-only cumsum fallback widens to int64 (unreachable in practice
    # — >= 8 GiB of indices — but kept provably safe).
    use_kernel_path = num_tiles <= _MAX_SINGLE_CTA_SCAN_TILES
    # single-CTA metadata pass only while its sequential N-element read is
    # cheaper than the multi-CTA boundary kernel + scan pair; beyond that
    # the multi-CTA pair wins on read bandwidth. Explicitly tied to
    # use_kernel_path so tuning one threshold can never route a huge input
    # into the metadata kernel.
    use_meta_kernel = use_kernel_path and num_tiles <= _META_KERNEL_MAX_TILES

    if use_kernel_path:
        lease = _lease_acquire(
            ("post", device.index, num_tiles, num_tasks),
            lambda: (
                torch.empty((num_tiles,), dtype=torch.int32, device=device),
                torch.empty((num_tiles,), dtype=torch.int32, device=device),
                torch.empty((1,), dtype=torch.int32, device=device),
                torch.empty((num_tasks,), dtype=torch.int64, device=device),
            ),
        )
        tile_counts, tile_prefix, total, idx = lease.bufs
        meta_dtype = torch.int32
    else:
        lease = None
        meta_dtype = torch.int64 if num_tasks >= (1 << 31) else torch.int32
        tile_counts = torch.empty((num_tiles,), dtype=torch.int32, device=device)
        tile_prefix = torch.empty((num_tiles,), dtype=meta_dtype, device=device)
        total = torch.empty((1,), dtype=meta_dtype, device=device)
        idx = torch.empty((num_tasks,), dtype=torch.int64, device=device)

    data_out = torch.empty_like(sorted_data)
    # Never pass None pointers to the kernel (Triton None-specialization is
    # fragile on this stack): unused outputs get shared 1-element dummies
    # that no code path ever writes.
    # Triton-MUSA 3.6 additionally miscompiles the scatter kernel's
    # write_inverse=False specialization for int64 inputs with a partial
    # trailing tile (n=8193: data_out receives int32-splat garbage
    # ((v-1)<<32|(v-1)) from some position onward, while boundary counts,
    # the scan and the write_inverse=True codegen stay correct; reproduced
    # across num_warps 4/8/16 and fresh compile caches). Force the
    # write_inverse=True codegen for int64; int64 is not in the official
    # benchmark dtype matrix, so the extra scatter pass costs nothing where
    # it matters.
    write_inverse = return_inverse or sorted_data.dtype == torch.int64
    inverse_indices = (
        torch.empty_like(sorted_data, dtype=torch.int64)
        if write_inverse
        else _dummy_i64(device)
    )
    if not return_counts:
        idx = _dummy_i64(device)
    counts_buf = (
        torch.empty((num_tasks,), dtype=torch.int64, device=device)
        if return_counts
        else None
    )

    # The probe is used only on the kernel paths; the cold cumsum fallback
    # (num_tiles > 32768, i.e. > 256M elements) reads the device total
    # synchronously instead — it must never wait on a probe that no kernel
    # writes (that previously surfaced as a 200 ms spin timeout followed
    # by a stale/garbage out_size; reproduced by threshold injection).
    probe = _probe_acquire(device.index) if use_kernel_path else None
    if probe is not None:
        _probe_arm(probe)
    lease_recorded = False
    probe_consumed = False

    try:
        with torch_device_fn.device(device.index):
            if use_meta_kernel:
                post_sort_meta_kernel[(1,)](
                    sorted_data,  # in
                    tile_prefix,  # out
                    total,  # out
                    probe.pinned,  # out
                    num_tasks,
                    num_tiles,
                    tile_size=tile_size,
                    num_warps=_POST_SORT_META_WARPS,
                )
            elif use_kernel_path:
                tile_boundary_counts_kernel[grid](
                    sorted_data,  # in
                    tile_counts,  # out
                    num_tasks,
                    tiles_per_cta=1,
                    tile_size=tile_size,
                    num_warps=_POST_SORT_BOUNDARY_WARPS,
                )
                tile_counts_scan_kernel[(1,)](
                    tile_counts,  # in
                    tile_prefix,  # out
                    total,  # out
                    probe.pinned,  # out
                    num_tiles,
                    next_power_num_tiles=triton.next_power_of_2(num_tiles),
                    num_warps=_POST_SORT_SCAN_WARPS,
                )
            else:
                tile_counts_buf = torch.empty(
                    (num_tiles,), dtype=torch.int32, device=device
                )
                tile_boundary_counts_kernel[grid](
                    sorted_data,  # in
                    tile_counts_buf,  # out
                    num_tasks,
                    tiles_per_cta=1,
                    tile_size=tile_size,
                    num_warps=_POST_SORT_BOUNDARY_WARPS,
                )
                cumsum = tile_counts_buf.cumsum(dim=0).to(meta_dtype)
                tile_prefix.narrow(0, 0, 1).zero_()
                if num_tiles > 1:
                    tile_prefix.narrow(0, 1, num_tiles - 1).copy_(
                        cumsum.narrow(0, 0, num_tiles - 1)
                    )
                total.copy_(cumsum.narrow(0, num_tiles - 1, 1))
            unique_scatter_kernel[grid](
                sorted_data,  # in
                sorted_indices,  # in
                tile_prefix,  # in
                data_out,  # out
                inverse_indices,  # out
                idx,  # out
                num_tasks,
                tiles_per_cta=1,
                tile_size=tile_size,
                write_unique=True,
                write_inverse=write_inverse,
                write_idx=return_counts,
                num_warps=_POST_SORT_SCATTER_WARPS,
            )
            if return_counts:
                # counts kernel reads only [0, out_size) via the
                # device-side valid-length mask in total
                output_counts_flat_kernel[grid](
                    idx,
                    total,  # in
                    num_tasks,  # in
                    counts_buf,  # out
                    num_tasks,
                    tiles_per_cta=1,
                    tile_size=tile_size,
                    num_warps=8,
                )
            if lease is not None:
                # scratch is safe for reuse once every kernel of this call
                # has completed (event-protected lease)
                lease.event.record()
                lease_recorded = True
        # spin on the pinned ready word (full-device-sync fallback on
        # timeout, targeting the producing device); the scatter/counts
        # kernels may still be executing. The cold fallback path
        # synchronously reads the device total instead.
        out_size = (_probe_wait(probe) if probe is not None else total.item()) + 1
        probe_consumed = True
    finally:
        # Probe lifecycle on exit:
        #  - consumed (normal): the word was observed, so the producer's
        #    single store has landed — nothing device-side touches the
        #    pinned buffer anymore; release with zero extra cost.
        #  - not consumed (exception after the producer was submitted):
        #    the store may still be queued and must not be allowed to land
        #    after a future caller arms the word. Synchronize the device
        #    (cold path) so any in-flight store has landed before release;
        #    quarantine permanently if even the sync fails.
        if probe is not None:
            if probe_consumed:
                probe.in_use = False
            else:
                try:
                    torch.musa.synchronize(device.index)
                    probe.in_use = False
                except Exception:
                    probe.in_use = True  # quarantined: never reuse
        # Lease lifecycle: recorded (normal) — reuse is gated on the
        # completion event; not recorded (exception) — synchronize the
        # device so any partially enqueued kernels finished before the
        # scratch becomes reusable, or quarantine on sync failure.
        if lease is not None:
            if lease_recorded:
                _lease_release(lease)
            else:
                try:
                    torch.musa.synchronize(device.index)
                    _lease_release(lease)
                except Exception:
                    lease.in_use = True  # quarantined: never reuse

    counts = counts_buf.narrow(0, 0, out_size) if return_counts else None
    # host side: unrequested outputs are None (normalized by _unique2)
    return (
        data_out.narrow(0, 0, out_size),
        inverse_indices if return_inverse else None,
        counts,
    )


def simple_unique_flat(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    num_tasks = sorted_data.numel()
    grid = (1, 1, 1)
    device = sorted_data.device

    # never pass None pointers to the kernel (see _unique_post_sort_impl);
    # unused outputs get shared unwritten 1-element dummies
    data_out = torch.empty_like(sorted_data)
    inverse_indices = (
        torch.empty_like(sorted_data, dtype=torch.int64)
        if return_inverse
        else _dummy_i64(device)
    )
    lease = _lease_acquire(
        ("small", device.index, num_tasks),
        lambda: (
            torch.empty((1,), dtype=torch.int64, device=device),
            torch.empty((num_tasks,), dtype=torch.int64, device=device),
        ),
    )
    total, idx = lease.bufs
    if not return_counts:
        idx = _dummy_i64(device)
    counts_buf = (
        torch.empty((num_tasks,), dtype=torch.int64, device=device)
        if return_counts
        else None
    )
    probe = _probe_acquire(device.index)
    _probe_arm(probe)
    lease_recorded = False
    probe_consumed = False

    try:
        with torch_device_fn.device(device.index):
            simple_unique_flat_kernel[grid](
                sorted_data,
                sorted_indices,  # in
                data_out,
                inverse_indices,
                idx,
                total,  # out
                probe.pinned,  # out
                return_inverse,
                return_counts,
                num_tasks,
                tile_size=triton.next_power_of_2(num_tasks),
                num_warps=8,
            )
            if return_counts:
                output_counts_flat_kernel[grid](
                    idx,
                    total,  # in
                    num_tasks,  # in
                    counts_buf,  # out
                    num_tasks,
                    tiles_per_cta=1,
                    tile_size=triton.next_power_of_2(num_tasks),
                    num_warps=8,
                )
            lease.event.record()
            lease_recorded = True
        out_size = _probe_wait(probe) + 1
        probe_consumed = True
    finally:
        # same probe/lease lifecycle as _unique_post_sort_impl
        if probe_consumed:
            probe.in_use = False
        else:
            try:
                torch.musa.synchronize(device.index)
                probe.in_use = False
            except Exception:
                probe.in_use = True  # quarantined: never reuse
        if lease_recorded:
            _lease_release(lease)
        else:
            try:
                torch.musa.synchronize(device.index)
                _lease_release(lease)
            except Exception:
                lease.in_use = True  # quarantined: never reuse

    counts = counts_buf.narrow(0, 0, out_size) if return_counts else None
    # host side: unrequested outputs are None (normalized by _unique2)
    return (
        data_out.narrow(0, 0, out_size),
        inverse_indices if return_inverse else None,
        counts,
    )


def _sorted_with_indices(flat: torch.Tensor):
    """Sort via the out= overload.

    The python-registered flag_gems sort override covers torch.sort's
    functional form but not its out= form; the out= form reaches the native
    mudnn radix sort, which is an order of magnitude faster than the python
    radix sort on this stack.
    """
    values = torch.empty_like(flat)
    indices = torch.empty(
        flat.shape,
        dtype=torch.int64,
        device=flat.device,
        memory_format=torch.contiguous_format,
    )
    torch.sort(flat, out=(values, indices))
    return values, indices


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_MTHREADS _UNIQUE2")
    if in0.numel() == 0:
        # radix_sort cannot launch on a 0-length dim; produce the empty result
        # directly (matches aten's short-circuit for dim=None unique).
        #
        # aten::_unique2 always returns three tensors; unrequested
        # inverse/counts are empty int64 tensors, never None. Verified on
        # this platform's aten (MUSA, CUDA-style contract): for a multi-dim
        # empty (e.g. (0, 3) or (2, 0)) the inverse comes back
        # view_as(in0) UNCONDITIONALLY — including return_inverse=False —
        # while non-empty inputs return a plain (0,) when unrequested.
        return (
            in0.new_empty((0,)),
            torch.empty(0, dtype=torch.int64, device=in0.device).view_as(in0),
            torch.empty(0, dtype=torch.int64, device=in0.device),
        )
    # ravel of a contiguous 1-D input is a no-op view; skip the dispatch
    flat = in0 if (in0.dim() == 1 and in0.is_contiguous()) else in0.ravel()
    if in0.numel() <= 8192:
        sorted_data, sorted_indices = _sorted_with_indices(flat)
        data_out, inverse_indices, counts = simple_unique_flat(
            sorted_data, sorted_indices, return_inverse, return_counts
        )
    else:
        sorted_data, sorted_indices = _sorted_with_indices(flat)
        data_out, inverse_indices, counts = _unique_post_sort_impl(
            sorted_data, sorted_indices, return_inverse, return_counts
        )
    # aten::_unique2 always returns three tensors: when return_inverse /
    # return_counts are False the corresponding output is an empty (0,) int64
    # tensor rather than None. Mirror that contract so callers can rely on the
    # shapes matching ATen regardless of the requested flags (same as the
    # generic ops/unique.py::_unique2).
    if inverse_indices is None:
        inverse_indices = torch.empty(0, dtype=torch.int64, device=in0.device)
    else:
        inverse_indices = inverse_indices.view_as(in0)
    if counts is None:
        counts = torch.empty(0, dtype=torch.int64, device=in0.device)
    return data_out, inverse_indices, counts
