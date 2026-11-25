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

# GCU400 notes (knowledge/perf_playbook.md, knowledge/compiler_traps.md):
#   * i64 tl.load/tl.store/atomics are unsafe on this stack (P17), so int64
#     inputs are normalized to int32 on the host and the counts are staged in
#     an int32 buffer that is widened to int64 at the end. Do NOT write counts
#     through view(torch.int32) of the int64 output: on this stack that view is
#     element-aligned (not 4B-packed), so the classic "2 * bin" low-word
#     indexing lands every count at twice its bin index (verified: the whole
#     bincount suite failed before the int32 staging buffer was introduced).
#   * device scalars (bin upper bound) are read back via .cpu().item(): the
#     GCU `_local_scalar_dense` has no narrow-int (int32) path.
#   * the histogram is a bin-centric one-hot scan ([BLOCK_N, TILE_B]
#     orientation, tl.sum(axis=0)) for narrow bin counts: the transposed
#     orientation with a column-broadcast weight silently miscompiles, and
#     num_warps must stay at 4 (num_warps=1 is a known silent-miscompile).
#     BLOCK_N is next_pow2(min(n, 1024)): masked-out lanes still pay the
#     [BLOCK_N, TILE_B] tile cost, so a padded BLOCK_N multiplies the
#     per-program work (n=64 measured 402 -> 18us after tightening).
#   * wide bin counts (needed_size > 2048) switch to an atomic-scatter
#     arm: with many bins the scatter is contention-free and the
#     per-program atomic chain (256 elements at ~90ns/op, measured on
#     GCU400) beats the O(n * num_bins) one-hot scan.
#   * weighted counts accumulate in fp32 in every arm, including fp16/bf16
#     weights (cast to fp32 before the scatter and narrowed once at the end):
#     half-precision accumulation drifts ~0.5% against the fp32-accumulated
#     reference for concentrated bins.
#   * grids are capped/folded below 65536 programs (P19 hard limit).

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_HIST_BLOCK_N = 1024
_HIST_TILE_B = 32
_HIST_W_BLOCK_N = 256
_HIST_W_TILE_B = 64
# Upper bound on the n-axis split of the weighted one-hot arm: past this the
# per-program input segment shrinks below one BLOCK_N tile for the shapes this
# arm serves, and the extra fp32 atomics stop paying for themselves.
_HIST_W_SPLIT_MAX = 16
_MAX_BLOCK_N = 1024
_MAX_PARTIAL_BLOCK_N = 1024
_MAX_PARTIAL_PROGS = 8192
_SINGLE_CTA_MAX_N = 65536
_ATOMIC_BLOCK_N = 256
_GRID_MAX = 65535

# Small inputs run the bin-centric one-hot scan with a tight BLOCK_N
# (next_pow2(n)); the [BLOCK_N, TILE_B] tile cost is paid even for masked-out
# lanes, so a padded BLOCK_N multiplies the per-program work.
_SMALL_N_LIMIT = 256

# Above this bin count the plain / fp32-weighted paths switch from the
# O(n * num_bins) one-hot scan to the atomic-scatter arm: with many bins the
# scatter is contention-free and the per-program atomic chain (BLOCK_N
# elements at ~90ns/op, measured on GCU400) beats the one-hot scan.
_ONEHOT_NB_LIMIT = 2048

# O(n * num_bins) compare budget for the one-hot arms. Measured: beyond this
# the bin-major scan loses to the scatter arm even when the tile is split along
# n, because the estimate is an upper bound on the bin count (n=4096 measured
# 0.37ms one-hot versus 0.11ms scatter).
_ONEHOT_MAX_OPS = 1 << 24

# The one-hot arm only pays off for short inputs: with a narrow bin count the
# O(n * num_bins) scan degenerates, so longer inputs use the atomic arm.
_WEIGHTED_ONEHOT_MAX_N = 2048

# Upper size limit for the weighted estimation arm: for larger n, a single onehot pass lacks parallelism and the benefit
# turns negative.
_WEIGHTED_EST_MAX_N = 1024

# Wide-minlength regime (minlength >= 4096 guarantees a wide, contention-free
# bin range): a pre-sync fused max+count pass into an estimated-size buffer
# keeps the count work ahead of the required output-length sync, which beats
# any post-sync arm by hiding the scatter behind the sync roundtrip.
_PRESYNC_MAX_N = 2048
_PRESYNC_MIN_ML = 4096
_PRESYNC_ESTIMATE_FLOOR = 8192

# Lower bound for the estimated output length on the unweighted path: when the number of bins is at or below this value,
# the entire chain needs only one synchronization.
_EST_BINS = 4096

# Estimate the weighted path separately: a weighted one-hot tile is heavier (each bin must also be multiplied by the
# weight),
# and an estimate that is too large can actually slow down small-bin cases (measured nc=10 rose from 46us to 92us), so
# keep this value small.
_WEIGHTED_EST_BINS = 256


# Upper limit for splitting the one-hot tile along the n dimension (splits are aggregated with integer atomic_add, so
# counts remain exact).
_ONEHOT_SPLIT_MAX = 8

_INT64_ALIAS_MODE = None


def _int32_view_of_int64(input_contig):
    """Zero-copy int32 alias of a GCU int64 tensor on the packed layout."""
    global _INT64_ALIAS_MODE
    if _INT64_ALIAS_MODE is None:
        try:
            probe = torch.arange(3, dtype=torch.int64, device=input_contig.device)
            words = probe.view(torch.int32)[:3]
            _INT64_ALIAS_MODE = (
                "packed" if words.cpu().tolist() == [0, 1, 2] else "wide"
            )
        except Exception:
            _INT64_ALIAS_MODE = "wide"
    if _INT64_ALIAS_MODE == "packed":
        return input_contig.view(torch.int32)[: input_contig.numel()]
    return input_contig.to(torch.int32)


@libentry()
@triton.jit
def _bincount_max_kernel(input_ptr, out_ptr, n_elements, BLOCK_N: tl.constexpr):
    acc = tl.full((BLOCK_N,), -1, dtype=tl.int32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        vals = tl.load(input_ptr + offs, mask=offs < n_elements, other=-1).to(tl.int32)
        acc = tl.maximum(acc, vals)
    tl.store(out_ptr, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_partial_max_kernel(
    input_ptr,
    partials_ptr,
    n_elements,
    num_progs,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.full((BLOCK_N,), -1, dtype=tl.int32)
    for base in range(pid * BLOCK_N, n_elements, num_progs * BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        vals = tl.load(input_ptr + offs, mask=offs < n_elements, other=-1).to(tl.int32)
        acc = tl.maximum(acc, vals)
    tl.store(partials_ptr + pid, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_reduce_max_kernel(
    partials_ptr,
    out_ptr,
    num_partials,
    BLOCK_R: tl.constexpr,
):
    acc = tl.full((BLOCK_R,), -1, dtype=tl.int32)
    for base in range(0, num_partials, BLOCK_R):
        offs = base + tl.arange(0, BLOCK_R)
        vals = tl.load(partials_ptr + offs, mask=offs < num_partials, other=-1)
        acc = tl.maximum(acc, vals)
    tl.store(out_ptr, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_hist_weights_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.float32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        match = (vals[:, None] == bins[None, :]).to(tl.float32)
        acc += tl.sum(match * w[:, None], axis=0)
    tl.store(output_ptr + bins, acc.to(output_ptr.dtype.element_ty), mask=bin_mask)


@libentry()
@triton.jit
def _bincount_atomic_weights_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    num_bins,
    grid_progs,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    for base in range(pid * BLOCK_N, n_elements, grid_progs * BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        safe = mask & (vals >= 0) & (vals < num_bins)
        tl.atomic_add(output_ptr + vals, w, mask=safe, sem="relaxed")


@libentry()
@triton.jit
def _bincount_split_hist_weights_kernel(
    input_ptr,
    weights_ptr,
    out_ptr,
    n_elements,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    """Wide-input / narrow-bin weighted histogram, split along the n axis.

    The plain one-hot arm walks the whole input with a single program per bin
    tile, so a narrow bin count leaves almost no parallelism and a long input
    degenerates to a serial scan. Here each (bin tile, n-split) pair owns one
    segment of the input and reduces it to TILE_B partial sums that are folded
    with fp32 atomics; the atomic count is TILE_B per program instead of one
    per element, so contention stays bounded even for 10 bins and 10000
    elements. ``out`` must be pre-zeroed and fp32.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    bins = pid_b * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.float32)
    stride = SPLIT_N * BLOCK_N
    for base in range(pid_n * BLOCK_N, n_elements, stride):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        match = (vals[:, None] == bins[None, :]).to(tl.float32)
        acc += tl.sum(match * w[:, None], axis=0)
    tl.atomic_add(out_ptr + bins, acc, mask=bin_mask, sem="relaxed")


@libentry()
@triton.jit
def _bincount_onehot_weights_est_kernel(
    input_ptr,
    weights_ptr,
    out_ptr,
    max_ptr,
    n_elements,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Weighted one-hot estimate arm: one launch fills the preallocated length and yields the bin bound.

    Each bin is owned by exactly one tile, so plain stores suffice (nothing to pre-zero, and no
    repeated fp32->low-precision accumulation rounding); out-of-range elements are skipped.
    """
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.float32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_max(max_ptr, tl.max(vals, axis=0), sem="relaxed")
        match = (vals[:, None] == bins[None, :]).to(tl.float32)
        acc += tl.sum(match * w[:, None], axis=0)
    tl.store(out_ptr + bins, acc.to(out_ptr.dtype.element_ty), mask=bin_mask)


@libentry()
@triton.jit
def _bincount_atomic_i64_kernel(
    input_ptr,
    out32_ptr,
    max_ptr,
    n_elements,
    num_bins,
    BLOCK_N: tl.constexpr,
):
    """scatter counting: writes the interleaved low word of the int64 result and tracks the bin bound.

    num_bins is the estimated output length; out-of-range elements are dropped and the caller recomputes after the sync.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for base in range(pid * BLOCK_N, n_elements, num_progs * BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        tl.atomic_max(max_ptr, tl.max(vals, axis=0), sem="relaxed")
        safe = mask & (vals >= 0) & (vals < num_bins)
        tl.atomic_add(out32_ptr + vals, 1, mask=safe, sem="relaxed")


@libentry()
@triton.jit
def _bincount_onehot_i64_kernel(
    input_ptr,
    out32_ptr,
    max_ptr,
    n_elements,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    """bin-major one-hot counting: each program handles one input segment for TILE_B bins.

    The result is written as the interleaved low word of int64; the same bin across SPLIT_N is
    summed with an integer atomic_add, so the output buffer must be pre-zeroed beforehand.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    bins = pid_b * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.int32)
    stride = SPLIT_N * BLOCK_N
    for base in range(pid_n * BLOCK_N, n_elements, stride):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        tl.atomic_max(max_ptr, tl.max(vals, axis=0), sem="relaxed")
        match = (vals[:, None] == bins[None, :]).to(tl.int32)
        acc += tl.sum(match, axis=0)
    tl.atomic_add(out32_ptr + bins, acc, mask=bin_mask, sem="relaxed")


@libentry()
@triton.jit
def _bincount_fused_weights_kernel(
    input_ptr,
    weights_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    output_size,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_elements
    vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
    w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_max = tl.max(vals, axis=0)
    tl.atomic_max(max_ptr, local_max)
    safe = mask & (vals >= 0) & (vals < output_size)
    tl.atomic_add(output_ptr + vals, w, mask=safe, sem="relaxed")


def _bincount_max(input_contig, n, device):
    max_out = torch.empty(1, dtype=torch.int32, device=device)
    if n <= _SINGLE_CTA_MAX_N:
        _bincount_max_kernel[(1,)](
            input_contig,
            max_out,
            n,
            BLOCK_N=_MAX_BLOCK_N,
            num_warps=4,
        )
    else:
        num_progs = min(triton.cdiv(n, _MAX_PARTIAL_BLOCK_N), _MAX_PARTIAL_PROGS)
        partials = torch.empty(num_progs, dtype=torch.int32, device=device)
        _bincount_partial_max_kernel[(num_progs,)](
            input_contig,
            partials,
            n,
            num_progs,
            BLOCK_N=_MAX_PARTIAL_BLOCK_N,
            num_warps=4,
        )
        _bincount_reduce_max_kernel[(1,)](
            partials,
            max_out,
            num_progs,
            BLOCK_R=_MAX_PARTIAL_BLOCK_N,
            num_warps=4,
        )
    return max_out


def _scalar_int(device_scalar):
    """Safely reads a device-side integer scalar back into a Python int.

    The GCU backend's `_local_scalar_dense` does not implement narrow integer (int32) reads, so a
    direct `.item()` raises NotImplementedError; hence the copy to CPU before reading.
    """
    return int(device_scalar.reshape(-1)[0].cpu().item())


def _bin_estimate(minlength):
    """Estimated output length for the unweighted path (the larger of minlength and the empirical bound).

    If the estimate falls short the caller recomputes once against the real length (speed only).
    """
    return max(minlength, _EST_BINS)


def _unweighted_arm(pre_size, n, minlength):
    """Returns (arm, extra): arm is `atomic` / `onehot` / None (none feasible)."""
    if minlength >= _PRESYNC_MIN_ML:
        # A large minlength keeps bins sparse and scatter conflict-free
        if triton.cdiv(n, _ATOMIC_BLOCK_N) <= _GRID_MAX:
            return ("atomic", None)
        return (None, None)
    if pre_size > _ONEHOT_NB_LIMIT:
        if triton.cdiv(n, _ATOMIC_BLOCK_N) <= _GRID_MAX:
            return ("atomic", None)
        return (None, None)
    num_tiles = triton.cdiv(pre_size, _HIST_TILE_B)
    if num_tiles > _GRID_MAX or n * pre_size > _ONEHOT_MAX_OPS:
        if triton.cdiv(n, _ATOMIC_BLOCK_N) <= _GRID_MAX:
            return ("atomic", None)
        return (None, None)
    return ("onehot", num_tiles)


def _count_unweighted(input_contig, n, minlength, device):
    """Unweighted bincount: one launch does both the counting and the bin bound, saving a second kernel.

    Counts are written into an int32 buffer (the int64 `view(torch.int32)` alias is unreliable on
    this backend -- measured per-element interleaved misalignment), and the bin bound goes via
    atomic_max into an extra slot at the buffer end; one sync yields the true output length.
    """
    pre_size = _bin_estimate(minlength)
    arm, num_tiles = _unweighted_arm(pre_size, n, minlength)
    if arm is None:
        # If the estimated length is infeasible, fall back to computing from the real length (first take max, then count
        # exactly)
        needed_size = max(
            _scalar_int(_bincount_max(input_contig, n, device)) + 1, minlength
        )
        return _count_unweighted_exact(input_contig, n, minlength, needed_size, device)

    out32 = torch.zeros(pre_size + 1, dtype=torch.int32, device=device)
    max_slot = out32[pre_size : pre_size + 1]
    if arm == "atomic":
        num_progs = min(triton.cdiv(n, _ATOMIC_BLOCK_N), _GRID_MAX)
        _bincount_atomic_i64_kernel[(num_progs,)](
            input_contig,
            out32,
            max_slot,
            n,
            pre_size,
            BLOCK_N=_ATOMIC_BLOCK_N,
            num_warps=4,
        )
    else:
        block_n = (
            _HIST_BLOCK_N if n > _SMALL_N_LIMIT else max(triton.next_power_of_2(n), 1)
        )
        split_n = min(max(triton.cdiv(n, block_n), 1), _ONEHOT_SPLIT_MAX)
        _bincount_onehot_i64_kernel[(num_tiles, split_n)](
            input_contig,
            out32,
            max_slot,
            n,
            pre_size,
            TILE_B=_HIST_TILE_B,
            BLOCK_N=block_n,
            SPLIT_N=split_n,
            num_warps=4,
        )
    needed_size = max(_scalar_int(max_slot) + 1, minlength)
    if needed_size <= pre_size:
        return out32[:needed_size].to(torch.int64)
    # If the estimated length is insufficient, the bin upper bound is already known; recompute exactly once from the
    # real length
    return _count_unweighted_exact(input_contig, n, minlength, needed_size, device)


def _count_unweighted_exact(input_contig, n, minlength, needed_size, device):
    """Counts against a known true length (the deterministic path when the estimate was short)."""
    if needed_size <= 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    arm, num_tiles = _unweighted_arm(needed_size, n, 0)
    if arm == "onehot":
        # The kernel always aggregates with atomic_add (exact across splits), so the buffer must be zeroed first
        out32 = torch.zeros(needed_size, dtype=torch.int32, device=device)
        max_slot = torch.zeros(1, dtype=torch.int32, device=device)
        block_n = (
            _HIST_BLOCK_N if n > _SMALL_N_LIMIT else max(triton.next_power_of_2(n), 1)
        )
        split_n = min(max(triton.cdiv(n, block_n), 1), _ONEHOT_SPLIT_MAX)
        _bincount_onehot_i64_kernel[(num_tiles, split_n)](
            input_contig,
            out32,
            max_slot,
            n,
            needed_size,
            TILE_B=_HIST_TILE_B,
            BLOCK_N=block_n,
            SPLIT_N=split_n,
            num_warps=4,
        )
        return out32.to(torch.int64)
    out32 = torch.zeros(needed_size, dtype=torch.int32, device=device)
    max_slot = torch.zeros(1, dtype=torch.int32, device=device)
    num_progs = min(triton.cdiv(n, _ATOMIC_BLOCK_N), _GRID_MAX)
    _bincount_atomic_i64_kernel[(num_progs,)](
        input_contig,
        out32,
        max_slot,
        n,
        needed_size,
        BLOCK_N=_ATOMIC_BLOCK_N,
        num_warps=4,
    )
    return out32.to(torch.int64)


def bincount(input, weights=None, minlength=0):
    """Count the frequency of each value in an array of non-negative ints."""
    logger.debug("GEMS_ENFLAME BINCOUNT")

    assert input.dim() == 1, "input must be a 1-D tensor"
    assert minlength >= 0, "minlength must be non-negative"

    if weights is not None:
        assert weights.shape == input.shape, "weights must have the same shape as input"

    n = input.numel()

    if n == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=input.device)
        return torch.zeros(minlength, dtype=torch.int64, device=input.device)

    input_contig = input.contiguous()
    weights_contig = weights.contiguous() if weights is not None else None

    if input_contig.dtype == torch.int64:
        input_contig = _int32_view_of_int64(input_contig)

    device = input_contig.device
    weights_dtype = weights_contig.dtype if weights_contig is not None else None
    half_precision = weights_dtype in (torch.float16, torch.bfloat16)
    downcast = weights_dtype is not None and (
        weights_dtype == torch.float64 or not weights_dtype.is_floating_point
    )
    if downcast:
        # GCU400 has no fp64 arithmetic; integer and fp64 weights accumulate
        # in fp32 (the established behavior of this backend).
        weights_contig = weights_contig.to(torch.float32)

    if weights is None:
        # Unweighted path: a single launch + a single synchronization is enough to return (see _count_unweighted)
        return _count_unweighted(input_contig, n, minlength, device)

    # The output length is data-dependent (max(input) + 1 vs minlength), so a
    # single host sync is semantically required; torch's own bincount syncs
    # the same way.
    if (
        not half_precision
        and n <= _WEIGHTED_EST_MAX_N
        and minlength <= _ONEHOT_NB_LIMIT
        and n * max(minlength, _WEIGHTED_EST_BINS) <= _ONEHOT_MAX_OPS
    ):
        # Estimation arm: one launch completes "weighted counting + bin upper bound" and needs only one host
        # synchronization;
        # if the estimate is insufficient, fall back to the exact path below (needed_size is already known, saving
        # another max).
        pre_size = max(minlength, _WEIGHTED_EST_BINS)
        est_out = torch.empty(pre_size, dtype=weights_contig.dtype, device=device)
        max_slot = torch.zeros(1, dtype=torch.int32, device=device)
        # For small n, use a compact BLOCK_N: masked-off lanes still pay the tile cost
        est_block_n = (
            max(triton.next_power_of_2(n), 1)
            if n <= _SMALL_N_LIMIT
            else _HIST_W_BLOCK_N
        )
        _bincount_onehot_weights_est_kernel[(triton.cdiv(pre_size, _HIST_TILE_B),)](
            input_contig,
            weights_contig,
            est_out,
            max_slot,
            n,
            pre_size,
            TILE_B=_HIST_TILE_B,
            BLOCK_N=est_block_n,
            num_warps=4,
        )
        needed_est = max(_scalar_int(max_slot) + 1, minlength)
        if needed_est <= pre_size:
            return est_out[:needed_est]
        needed_size = needed_est
    else:
        needed_size = None

    if (
        needed_size is None
        and n <= _PRESYNC_MAX_N
        and minlength >= _PRESYNC_MIN_ML
        and not half_precision
    ):
        # Wide-minlength regime (plain / fp32-domain weights only: an fp32
        # accumulation cannot reproduce the native half-precision per-add
        # rounding when values concentrate in a few bins despite a wide
        # minlength): needed_size is guaranteed wide (hence the scatter
        # contention-free), so a pre-sync fused max+count pass into an
        # estimated-size buffer hides the count work ahead of the sync.
        pre_size = max(_PRESYNC_ESTIMATE_FLOOR, min(4 * n, 65536), minlength)
        max_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        fused_grid = (triton.cdiv(n, _ATOMIC_BLOCK_N),)
        staging = torch.zeros(pre_size, dtype=torch.float32, device=device)
        _bincount_fused_weights_kernel[fused_grid](
            input_contig,
            weights_contig,
            max_tensor,
            staging,
            n,
            pre_size,
            BLOCK_N=_ATOMIC_BLOCK_N,
            num_warps=4,
        )
        needed_size = max(_scalar_int(max_tensor) + 1, minlength)
        if needed_size <= pre_size:
            output = staging[:needed_size]
            if downcast:
                output = output.to(weights_dtype)
            return output

    if needed_size is None:
        max_out = _bincount_max(input_contig, n, device)
        needed_size = max(_scalar_int(max_out) + 1, minlength)

    if needed_size <= 0:
        return torch.empty(0, dtype=weights.dtype, device=device)

    def _atomic_route():
        # Contention-free wide-bin arm. Half-precision weights are widened so
        # the per-bin sum is accumulated in fp32 and narrowed once at the end.
        w32 = weights_contig.to(torch.float32) if half_precision else weights_contig
        stage = torch.zeros(needed_size, dtype=torch.float32, device=device)
        num_progs = min(triton.cdiv(n, _ATOMIC_BLOCK_N), _GRID_MAX)
        _bincount_atomic_weights_kernel[(num_progs,)](
            input_contig,
            w32,
            stage,
            n,
            needed_size,
            num_progs,
            BLOCK_N=_ATOMIC_BLOCK_N,
            num_warps=4,
        )
        if half_precision:
            return stage.to(weights_contig.dtype)
        return stage

    def _onehot_weighted(tight):
        # tight=True uses BLOCK_N=next_pow2(n) (small n: masked-out lanes
        # still pay the tile cost, so a padded BLOCK_N multiplies the work).
        block_n = max(triton.next_power_of_2(n), 1) if tight else _HIST_W_BLOCK_N
        num_progs = triton.cdiv(needed_size, _HIST_W_TILE_B)
        if num_progs > _GRID_MAX or n * needed_size > _ONEHOT_MAX_OPS:
            return None
        output = torch.empty(needed_size, dtype=weights_contig.dtype, device=device)
        _bincount_hist_weights_kernel[(num_progs,)](
            input_contig,
            weights_contig,
            output,
            n,
            needed_size,
            TILE_B=_HIST_W_TILE_B,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return output

    def _split_hist_route():
        """Split the input across programs for a narrow bin count.

        One-hot needs the comparison tile per element, so the plain arm scales
        with n * needed_size; splitting along n restores parallelism without
        paying one atomic per element the way the scatter arm does.
        """
        n_tiles = triton.cdiv(needed_size, _HIST_W_TILE_B)
        if n_tiles > _GRID_MAX:
            return None
        split_n = min(max(triton.cdiv(n, _HIST_W_BLOCK_N), 1), _HIST_W_SPLIT_MAX)
        stage = torch.zeros(needed_size, dtype=torch.float32, device=device)
        _bincount_split_hist_weights_kernel[(n_tiles, split_n)](
            input_contig,
            weights_contig if not half_precision else weights_contig.to(torch.float32),
            stage,
            n,
            needed_size,
            TILE_B=_HIST_W_TILE_B,
            BLOCK_N=_HIST_W_BLOCK_N,
            SPLIT_N=split_n,
            num_warps=4,
        )
        if half_precision:
            return stage.to(weights_contig.dtype)
        return stage

    if half_precision:
        # Every arm widens the weights to fp32 before the accumulate; only the
        # final narrowing reproduces the half-precision output dtype.
        if needed_size > _ONEHOT_NB_LIMIT:
            output = _atomic_route()
        elif n <= _WEIGHTED_ONEHOT_MAX_N:
            output = _onehot_weighted(tight=n <= _SMALL_N_LIMIT)
        else:
            output = _split_hist_route()
    else:
        # fp32 weights (including the fp64 / integer downcasts).
        if needed_size <= _ONEHOT_NB_LIMIT:
            if n <= _WEIGHTED_ONEHOT_MAX_N:
                output = _onehot_weighted(tight=False)
                if output is None:
                    output = _atomic_route()
            else:
                output = _split_hist_route()
        else:
            if n <= _SMALL_N_LIMIT:
                output = _onehot_weighted(tight=True)
                if output is None:
                    output = _atomic_route()
            else:
                output = _atomic_route()

    if downcast:
        output = output.to(weights_dtype)
    return output
