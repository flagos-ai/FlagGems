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
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

INTERPOLATION_METHOD = ["linear", "lower", "higher", "nearest", "midpoint"]

# Resident selection covers a whole reduction slice in registers (value-only sort,
# no indices). Above this width the per-program register footprint of the sorted
# tile outweighs the benefit and the sort-based fallback wins.
RESIDENT_M_LIMIT = 2048
RADIX_SELECT_M_MIN = RESIDENT_M_LIMIT + 1
RADIX_SELECT_MIN_SLICES = 128
RADIX_SELECT_M_LIMIT = 3072
RADIX_SELECT_MAX_Q = 1
RADIX_SELECT_BLOCK_M = 1024
RADIX_SELECT_BITS = 8
RESIDENT_MAX_Q = 16
# Target program count for the resident kernel: enough tiles to fill the device
# without fragmenting high-inner workloads into too many tiny programs.
RESIDENT_TARGET_PROGRAMS = 8192
RESIDENT_TILE_N_CAP = 16
RESIDENT_FUSED_Q_VALIDATE_MAX_PROGRAMS = 1024


def _pick_tile(block_m):
    """Rows-per-program for the resident kernel (measured optimum).

    Consecutive-row tiling (inner == 1) keeps 2 rows per program.  Inner-lane
    tiling is selected at the call site from the measured target program count.
    """
    return 2


def _use_radix_select(M, N, Q, interpolation):
    if M < RADIX_SELECT_M_MIN or M > RADIX_SELECT_M_LIMIT:
        return False
    if N < RADIX_SELECT_MIN_SLICES or Q > RADIX_SELECT_MAX_Q:
        return False
    return Q == 1 and N <= 512 and interpolation in ("lower", "higher", "nearest")


@triton.jit
def _quantile_ranks(
    q_ptr,
    M,
    Q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    interpolation: tl.constexpr,
):
    # aten rank math (fp32 product, fp32 floor/ceil):
    #   p = q * (M-1); lower = floor(p); upper = ceil(p)
    # `ceil`, not `lower + 1`, is required for integer ranks.  In particular,
    # linear quantile over [-inf, finite, +inf] must return the finite middle
    # value at q=0.5 rather than producing 0 * inf.
    qoffs = tl.arange(0, BLOCK_Q)
    qmask = qoffs < Q
    qv = tl.load(q_ptr + qoffs, mask=qmask, other=0.0)
    p = qv * (M - 1)
    q_lower = tl.floor(p).to(tl.int32)
    q_upper = tl.ceil(p).to(tl.int32)
    t = p - q_lower
    return qoffs, qmask, q_lower, q_upper, t


@triton.jit
def _store_q_status(q_ptr, status_ptr, Q: tl.constexpr, BLOCK_Q: tl.constexpr):
    qoffs = tl.arange(0, BLOCK_Q)
    qmask = qoffs < Q
    qv = tl.load(q_ptr + qoffs, mask=qmask, other=0.0)
    bad = qmask & ((qv < 0.0) | (qv > 1.0) | (qv != qv))
    tl.store(status_ptr, tl.max(bad.to(tl.int32), axis=0))


@triton.jit
def _fp32_order_key(vals, valid):
    top_mask = tl.full(vals.shape, 0x80000000, dtype=tl.uint32)
    full_mask = tl.full(vals.shape, 0xFFFFFFFF, dtype=tl.uint32)
    bits = vals.to(tl.uint32, bitcast=True)
    sign_mask = tl.where((bits & top_mask) != 0, full_mask, top_mask)
    key = bits ^ sign_mask
    return tl.where(valid, key, full_mask)


@triton.jit
def _bitonic_equal_key_perm(k, BLOCK_M: tl.constexpr):
    # MUSA sort's unstable tie movement for equal keys follows the bitonic
    # compare network.  This maps a sorted rank to the source lane for the
    # all-equal case, used only to preserve signed zero bits for all-zero rows.
    res = tl.full(k.shape, 0, dtype=tl.int32)
    xor_mask = tl.full(k.shape, 0, dtype=tl.int32)
    kk = k
    for shift in tl.static_range(10, -1, -1):
        if (1 << (shift + 1)) <= BLOCK_M:
            half = 1 << shift
            lower_half = kk < half
            res = tl.where(lower_half, res | half, res)
            kk = tl.where(lower_half, kk, kk - half)
            if shift > 0:
                xor_mask = tl.where(lower_half, xor_mask, xor_mask ^ (half >> 1))
    return res ^ xor_mask


@triton.jit
def _quantile_interpolate(
    lower_vals,
    upper_vals,
    t2,
    ql2,
    interpolation: tl.constexpr,
):
    # aten lerp semantics (bit-exact, verified against reference):
    #   d = fp32(upper - lower)
    #   t <  0.5: a + t*d        (products/sums in fp64, single final round)
    #   t >= 0.5: b - (1-t)*d
    #   midpoint: b - 0.5*d      (the t=0.5 branch)
    #   nearest : rint(p) ties-to-even on the rank
    if interpolation == "linear":
        d = (upper_vals - lower_vals).to(tl.float64)
        t64 = t2.to(tl.float64)
        a64 = lower_vals.to(tl.float64)
        b64 = upper_vals.to(tl.float64)
        outv = tl.where(
            t2 < 0.5,
            (a64 + t64 * d).to(tl.float32),
            (b64 - (1.0 - t64) * d).to(tl.float32),
        )
    elif interpolation == "lower":
        outv = lower_vals
    elif interpolation == "higher":
        outv = upper_vals
    elif interpolation == "nearest":
        lower_even = (ql2 % 2) == 0
        pick_upper = (t2 > 0.5) | ((t2 == 0.5) & (~lower_even))
        outv = tl.where(pick_upper, upper_vals, lower_vals)
    else:  # midpoint
        d = (upper_vals - lower_vals).to(tl.float64)
        b64 = upper_vals.to(tl.float64)
        outv = (b64 - 0.5 * d).to(tl.float32)
    return outv


@libentry()
@triton.jit
def quantile_resident_kernel(
    inp,
    q_ptr,
    status_ptr,
    out,
    M,
    inner,
    Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_N: tl.constexpr,
    interpolation: tl.constexpr,
    INNER_TILE: tl.constexpr,
    VALIDATE_Q: tl.constexpr,
    PRESERVE_ZERO_SIGN: tl.constexpr,
):
    # One program covers TILE_N reduction slices read directly from the native
    # [outer, M, inner] layout (strided loads, no materialization), value-only
    # sorts each slice in registers, and extracts the per-q order statistics.
    pid = tl.program_id(0)
    if VALIDATE_Q:
        _store_q_status(q_ptr, status_ptr, Q, BLOCK_Q)
    cols = tl.arange(0, BLOCK_M)
    valid = cols < M

    if INNER_TILE:
        lanes = tl.arange(0, TILE_N)
        tiles_per_row = tl.cdiv(inner, TILE_N)
        outer_id = pid // tiles_per_row
        tile_id = pid % tiles_per_row
        lane_ids = tile_id * TILE_N + lanes
        lane_mask = lane_ids < inner
        ptrs = outer_id * (M * inner) + cols[None, :] * inner + lane_ids[:, None]
        out_off = (outer_id * inner + lane_ids) * Q
        out_mask = lane_mask
    else:
        lanes = tl.arange(0, TILE_N)
        rows = pid * TILE_N + lanes
        lane_mask = rows < inner  # `inner` carries the total slice count here
        ptrs = rows[:, None] * M + cols[None, :]
        out_off = rows * Q
        out_mask = lane_mask

    m2 = valid[None, :] & lane_mask[:, None]
    row = tl.load(inp + ptrs, mask=m2, other=float("inf"))

    # NaN in a slice -> NaN result for that slice; NaNs are sorted to the tail
    # as +inf and the output is overridden below.
    nan_mask = m2 & (row != row)
    has_nan = tl.max(nan_mask.to(tl.int32), axis=1) > 0
    sortable = tl.where(nan_mask, float("inf"), row)
    ordered = tl.sort(sortable, dim=1, descending=False)
    # Zero the padding tail: masked lanes hold +inf and inf*0 = nan in the
    # one-hot extraction below.
    ordered = tl.where(cols[None, :] < M, ordered, 0.0)

    qoffs, qmask, q_lower, q_upper, t = _quantile_ranks(
        q_ptr, M, Q, BLOCK_Q, interpolation
    )

    # one-hot extraction: lower/upper order statistics per q
    # select via where (not multiply) — data may contain +-inf and inf*0 = nan
    oh_l = cols[None, :] == q_lower[:, None]
    oh_u = cols[None, :] == q_upper[:, None]
    ord3 = tl.reshape(ordered, (TILE_N, 1, BLOCK_M))
    lower_vals = tl.sum(
        tl.where(tl.reshape(oh_l, (1, BLOCK_Q, BLOCK_M)), ord3, 0.0), axis=2
    )
    upper_vals = tl.sum(
        tl.where(tl.reshape(oh_u, (1, BLOCK_Q, BLOCK_M)), ord3, 0.0), axis=2
    )

    if PRESERVE_ZERO_SIGN:
        zero_entries = m2 & (row == 0.0)
        neg_count = tl.sum((m2 & (row < 0.0)).to(tl.int32), axis=1)
        zero_count = tl.sum(zero_entries.to(tl.int32), axis=1)
        nonzero = m2 & (row != 0.0)
        all_zero = tl.max(nonzero.to(tl.int32), axis=1) == 0

        zero_prefix = tl.cumsum(zero_entries.to(tl.int32), axis=1) - zero_entries.to(
            tl.int32
        )
        zero_prefix3 = tl.reshape(zero_prefix, (TILE_N, 1, BLOCK_M))
        zero_entries3 = tl.reshape(zero_entries, (TILE_N, 1, BLOCK_M))
        cols3 = tl.reshape(cols, (1, 1, BLOCK_M))

        lower_zero_ord = q_lower[None, :] - neg_count[:, None]
        upper_zero_ord = q_upper[None, :] - neg_count[:, None]
        lower_zero_mask = (
            (lower_zero_ord >= 0)
            & (lower_zero_ord < zero_count[:, None])
            & qmask[None, :]
        )
        upper_zero_mask = (
            (upper_zero_ord >= 0)
            & (upper_zero_ord < zero_count[:, None])
            & qmask[None, :]
        )

        lower_zero_match = zero_entries3 & (
            zero_prefix3 == tl.reshape(lower_zero_ord, (TILE_N, BLOCK_Q, 1))
        )
        upper_zero_match = zero_entries3 & (
            zero_prefix3 == tl.reshape(upper_zero_ord, (TILE_N, BLOCK_Q, 1))
        )
        lower_zero_idx = tl.min(tl.where(lower_zero_match, cols3, BLOCK_M), axis=2)
        upper_zero_idx = tl.min(tl.where(upper_zero_match, cols3, BLOCK_M), axis=2)

        lower_idx = _bitonic_equal_key_perm(q_lower, BLOCK_M)
        upper_idx = _bitonic_equal_key_perm(q_upper, BLOCK_M)
        m3_lower_idx = (M - 1) - q_lower
        m3_upper_idx = (M - 1) - q_upper
        lower_idx = tl.where(M == 3, m3_lower_idx, lower_idx)
        upper_idx = tl.where(M == 3, m3_upper_idx, upper_idx)
        lower_idx = tl.where(all_zero[:, None], lower_idx[None, :], lower_zero_idx)
        upper_idx = tl.where(all_zero[:, None], upper_idx[None, :], upper_zero_idx)
        if INNER_TILE:
            lower_ptr = outer_id * (M * inner) + lower_idx * inner + lane_ids[:, None]
            upper_ptr = outer_id * (M * inner) + upper_idx * inner + lane_ids[:, None]
            zero_load_mask = lane_mask[:, None]
        else:
            lower_ptr = rows[:, None] * M + lower_idx
            upper_ptr = rows[:, None] * M + upper_idx
            zero_load_mask = lane_mask[:, None]
        lower_load_mask = zero_load_mask & lower_zero_mask
        upper_load_mask = zero_load_mask & upper_zero_mask
        zero_lower = tl.load(inp + lower_ptr, mask=lower_load_mask, other=0.0)
        zero_upper = tl.load(inp + upper_ptr, mask=upper_load_mask, other=0.0)
        lower_vals = tl.where(lower_load_mask, zero_lower, lower_vals)
        upper_vals = tl.where(upper_load_mask, zero_upper, upper_vals)

    t2 = tl.broadcast_to(t[None, :], (TILE_N, BLOCK_Q))
    ql2 = tl.broadcast_to(q_lower[None, :], (TILE_N, BLOCK_Q))
    outv = _quantile_interpolate(lower_vals, upper_vals, t2, ql2, interpolation)

    outv = tl.where(tl.reshape(has_nan, (TILE_N, 1)), float("nan"), outv)
    st_mask = qmask[None, :] & out_mask[:, None]
    tl.store(out + out_off[:, None] + qoffs[None, :], outv, mask=st_mask)


@libentry()
@triton.jit
def quantile_radix_rank_kernel(
    inp,
    q_ptr,
    rank_values,
    M,
    inner,
    Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    RADIX_BITS_: tl.constexpr,
    RANKS_PER_Q: tl.constexpr,
    interpolation: tl.constexpr,
):
    # Rank-independent radix-select: one program selects one target rank for
    # one logical output slice.  This avoids full sorting and index buffers;
    # interpolation is a separate small kernel over the N x rank_count result.
    outer_id = tl.program_id(0)
    inner_id = tl.program_id(1)
    slot = tl.program_id(2)
    offs = tl.arange(0, BLOCK_M)
    rank_count: tl.constexpr = Q * RANKS_PER_Q

    q_idx = slot // RANKS_PER_Q
    q_scaled = tl.load(q_ptr + q_idx) * (M - 1)
    lower_rank = tl.floor(q_scaled).to(tl.int32)
    upper_rank = tl.ceil(q_scaled).to(tl.int32)
    if RANKS_PER_Q == 2:
        rank = tl.where((slot % 2) == 0, lower_rank, upper_rank)
    else:
        if interpolation == "higher":
            rank = upper_rank
        elif interpolation == "nearest":
            rank = tl_extra_shim.rint(q_scaled).to(tl.int32)
        else:
            rank = lower_rank

    base = outer_id * (M * inner) + inner_id
    has_nan = tl.full((), 0, dtype=tl.int32)
    neg_count = tl.full((), 0, dtype=tl.int32)
    zero_count = tl.full((), 0, dtype=tl.int32)

    for start in tl.range(0, M, BLOCK_M):
        m = start + offs
        mask = m < M
        vals = tl.load(inp + base + m * inner, mask=mask, other=0.0)
        is_nan = mask & (vals != vals)
        valid = mask & ~is_nan
        has_nan += tl.sum(is_nan.to(tl.int32), axis=0)
        neg_count += tl.sum((valid & (vals < 0.0)).to(tl.int32), axis=0)
        zero_count += tl.sum((valid & (vals == 0.0)).to(tl.int32), axis=0)

    desired = tl.full((), 0, dtype=tl.uint32)
    desired_mask = tl.full((), 0, dtype=tl.uint32)
    radix_mask: tl.constexpr = (1 << RADIX_BITS_) - 1
    radix_tail_bits: tl.constexpr = 32 % RADIX_BITS_
    radix_tail_mask: tl.constexpr = (1 << radix_tail_bits) - 1
    radix_mask_val = tl.full((), radix_mask, dtype=tl.uint32)
    bins = tl.arange(0, 1 << RADIX_BITS_)
    k_to_find = rank + 1

    for digit_pos in tl.static_range(
        32 - RADIX_BITS_, radix_tail_bits - 1, -RADIX_BITS_
    ):
        counts = tl.zeros((1 << RADIX_BITS_,), dtype=tl.int32)
        for start in tl.range(0, M, BLOCK_M):
            m = start + offs
            mask = m < M
            vals = tl.load(inp + base + m * inner, mask=mask, other=0.0)
            valid = mask & (vals == vals)
            keys = _fp32_order_key(vals, valid)
            active = valid & ((keys & desired_mask) == desired)
            digit = ((keys >> digit_pos) & radix_mask).to(tl.int32)
            counts += tl.histogram(digit, 1 << RADIX_BITS_, mask=active).to(tl.int32)

        cumsum = tl.cumsum(counts, axis=0)
        prev = cumsum - counts
        take = (k_to_find <= cumsum) & (k_to_find > prev)
        selected_bin = tl.min(tl.where(take, bins, (1 << RADIX_BITS_) - 1), axis=0)
        counts_before = tl.max(tl.where(take, prev, 0), axis=0)
        desired = desired | (selected_bin.to(tl.uint32) << digit_pos)
        desired_mask = desired_mask | (radix_mask_val << digit_pos)
        k_to_find = k_to_find - counts_before

    if radix_tail_bits != 0:
        counts = tl.zeros((1 << RADIX_BITS_,), dtype=tl.int32)
        for start in tl.range(0, M, BLOCK_M):
            m = start + offs
            mask = m < M
            vals = tl.load(inp + base + m * inner, mask=mask, other=0.0)
            valid = mask & (vals == vals)
            keys = _fp32_order_key(vals, valid)
            active = valid & ((keys & desired_mask) == desired)
            digit = (keys & radix_tail_mask).to(tl.int32)
            counts += tl.histogram(digit, 1 << RADIX_BITS_, mask=active).to(tl.int32)

        cumsum = tl.cumsum(counts, axis=0)
        prev = cumsum - counts
        take = (k_to_find <= cumsum) & (k_to_find > prev)
        selected_bin = tl.min(tl.where(take, bins, (1 << RADIX_BITS_) - 1), axis=0)
        counts_before = tl.max(tl.where(take, prev, 0), axis=0)
        desired = desired | selected_bin.to(tl.uint32)
        desired_mask = desired_mask | tl.full((), radix_tail_mask, dtype=tl.uint32)
        k_to_find = k_to_find - counts_before

    result_idx = tl.full((), M, dtype=tl.int32)
    zero_seen = tl.full((), 0, dtype=tl.int32)
    zero_ord = rank - neg_count
    zero_selected = (zero_ord >= 0) & (zero_ord < zero_count)
    for start in tl.range(0, M, BLOCK_M):
        m = start + offs
        mask = m < M
        vals = tl.load(inp + base + m * inner, mask=mask, other=0.0)
        valid = mask & (vals == vals)
        keys = _fp32_order_key(vals, valid)
        key_idx = tl.min(tl.where(valid & (keys == desired), m, M), axis=0)

        is_zero = valid & (vals == 0.0)
        zero_prefix = tl.cumsum(is_zero.to(tl.int32), axis=0) - is_zero.to(tl.int32)
        zero_idx = tl.min(
            tl.where(is_zero & ((zero_seen + zero_prefix) == zero_ord), m, M), axis=0
        )
        result_idx = tl.minimum(result_idx, tl.where(zero_selected, zero_idx, key_idx))
        zero_seen += tl.sum(is_zero.to(tl.int32), axis=0)

    selected = tl.load(inp + base + result_idx * inner, mask=result_idx < M, other=0.0)
    selected = tl.where(has_nan != 0, float("nan"), selected)
    row = outer_id * inner + inner_id
    tl.store(rank_values + row * rank_count + slot, selected)


@libentry()
@triton.jit
def quantile_rank_interpolate_kernel(
    rank_values,
    q_ptr,
    out,
    N,
    M,
    Q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RANKS_PER_Q: tl.constexpr,
    interpolation: tl.constexpr,
):
    pid_n = tl.program_id(0)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    qoffs, qmask, q_lower, q_upper, t = _quantile_ranks(
        q_ptr, M, Q, BLOCK_Q, interpolation
    )
    mask = row_mask[:, None] & qmask[None, :]
    if RANKS_PER_Q == 2:
        lower_vals = tl.load(
            rank_values + rows[:, None] * (Q * 2) + qoffs[None, :] * 2,
            mask=mask,
            other=0.0,
        )
        upper_vals = tl.load(
            rank_values + rows[:, None] * (Q * 2) + qoffs[None, :] * 2 + 1,
            mask=mask,
            other=0.0,
        )
        t2 = tl.broadcast_to(t[None, :], (BLOCK_N, BLOCK_Q))
        ql2 = tl.broadcast_to(q_lower[None, :], (BLOCK_N, BLOCK_Q))
        outv = _quantile_interpolate(lower_vals, upper_vals, t2, ql2, interpolation)
    else:
        outv = tl.load(
            rank_values + rows[:, None] * Q + qoffs[None, :], mask=mask, other=0.0
        )
    tl.store(out + rows[:, None] * Q + qoffs[None, :], outv, mask=mask)


@libentry()
@triton.jit
def quantile_gather_kernel(
    sorted_inp,
    q_ptr,
    out,
    M,
    inner,
    Q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_N: tl.constexpr,
    interpolation: tl.constexpr,
    INNER_TILE: tl.constexpr,
):
    # Post-sort gather/interpolation over rows already sorted along the last dim.
    pid = tl.program_id(0)
    qoffs, qmask, q_lower, q_upper, t = _quantile_ranks(
        q_ptr, M, Q, BLOCK_Q, interpolation
    )

    lanes = tl.arange(0, TILE_N)
    if INNER_TILE:
        tiles_per_row = tl.cdiv(inner, TILE_N)
        outer_id = pid // tiles_per_row
        tile_id = pid % tiles_per_row
        lane_ids = tile_id * TILE_N + lanes
        lane_mask = lane_ids < inner
        base = sorted_inp + outer_id * (M * inner) + lane_ids[:, None]
        tail_ptr = sorted_inp + outer_id * (M * inner) + (M - 1) * inner + lane_ids
        out_off = (outer_id * inner + lane_ids) * Q
        row_stride = inner
    else:
        rows = pid * TILE_N + lanes
        lane_mask = rows < inner
        base = sorted_inp + rows[:, None] * M
        tail_ptr = sorted_inp + rows * M + (M - 1)
        out_off = rows * Q
        row_stride = 1

    gmask = lane_mask[:, None] & qmask[None, :]
    lower_vals = tl.load(base + q_lower[None, :] * row_stride, mask=gmask, other=0.0)
    upper_vals = tl.load(base + q_upper[None, :] * row_stride, mask=gmask, other=0.0)
    tail_vals = tl.load(tail_ptr, mask=lane_mask, other=0.0)
    has_nan = tail_vals != tail_vals

    t2 = tl.broadcast_to(t[None, :], (TILE_N, BLOCK_Q))
    ql2 = tl.broadcast_to(q_lower[None, :], (TILE_N, BLOCK_Q))
    outv = _quantile_interpolate(lower_vals, upper_vals, t2, ql2, interpolation)
    outv = tl.where(tl.reshape(has_nan, (TILE_N, 1)), float("nan"), outv)
    tl.store(out + out_off[:, None] + qoffs[None, :], outv, mask=gmask)


@libentry()
@triton.jit
def quantile_q_validate_kernel(
    q_ptr,
    status_ptr,
    Q,
    BLOCK_Q: tl.constexpr,
):
    # Single fused validity check: 0/1 status word, one launch, one host read.
    offs = tl.arange(0, BLOCK_Q)
    mask = offs < Q
    qv = tl.load(q_ptr + offs, mask=mask, other=0.0)
    bad = (qv < 0.0) | (qv > 1.0) | (qv != qv)
    any_bad = tl.max(bad.to(tl.int32), axis=0)
    tl.store(status_ptr, any_bad)


def _native_sort_rows(rows):
    """Sort rows along the last dim via the native out= overload.

    The python-registered flag_gems sort override covers torch.sort's functional
    form but not its out= form; the out= form reaches the native mudnn radix
    sort, which is an order of magnitude faster than the python radix sort on
    this stack. (Same trick as _mthreads/ops/unique.py.)
    """
    values = torch.empty_like(rows)
    indices = torch.empty(rows.shape, dtype=torch.int64, device=rows.device)
    torch.sort(rows, dim=-1, out=(values, indices))
    return values


def quantile(inp, q, dim=None, keepdim=False, interpolation="linear", out=None):
    logger.debug("GEMS_MTHREADS QUANTILE")
    assert torch.is_floating_point(inp)
    assert dim is None or isinstance(dim, int)
    assert isinstance(q, (float, torch.Tensor))
    assert interpolation in INTERPOLATION_METHOD

    if interpolation not in INTERPOLATION_METHOD:
        raise RuntimeError(
            f"quantile() interpolation must be one of {INTERPOLATION_METHOD}"
        )

    if inp.numel() == 0:
        raise RuntimeError("quantile() input tensor must be non-empty")

    if dim is None:
        inp = inp.ravel()
        dim = 0
    if dim < 0:
        dim = dim + inp.ndim

    # ---- q handling + validation (one fused kernel + one host read) ----
    # aten output-dim rule (verified against reference): the Q dimension is
    # present iff q is a tensor with ndim > 0; a float or 0-dim tensor q
    # yields the squeezed form.
    q_is_scalar = isinstance(q, float)
    squeeze_q = q_is_scalar or (isinstance(q, torch.Tensor) and q.dim() == 0)
    q_shape = ()
    q_needs_validation = not q_is_scalar
    if q_is_scalar:
        if not (0.0 <= q <= 1.0):
            raise RuntimeError("quantile() q values must be in the range [0, 1]")
        if (q == 0.0 or q == 1.0) and interpolation in (
            "lower",
            "higher",
            "nearest",
        ):
            reduce_fn = torch.amin if q == 0.0 else torch.amax
            if out is not None:
                reduce_fn(inp, dim=dim, keepdim=keepdim, out=out)
                return out
            return reduce_fn(inp, dim=dim, keepdim=keepdim)
        Q = 1
        q_t = torch.tensor([q], device=inp.device, dtype=inp.dtype)
    else:
        if q.device != inp.device or q.dtype != inp.dtype:
            q_t = q.to(device=inp.device, dtype=inp.dtype)
        else:
            q_t = q
        Q = q_t.numel()
        if Q == 0:
            raise RuntimeError("quantile() q must be non-empty")
        if q_t.dim() == 0:
            q_t = q_t.reshape(1)
        elif q_t.dim() > 1:
            raise RuntimeError("quantile() q must be a scalar or 1D tensor")
        else:
            q_shape = tuple(q_t.shape)
            q_t = q_t.reshape(Q)

    if out is not None and out.dtype != inp.dtype:
        raise RuntimeError(
            "quantile() out tensor must be same dtype as the input tensor"
        )

    # ---- logical 3D decomposition: [outer, M, inner] ----
    shape = inp.shape
    M = shape[dim]
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    inner = math.prod(shape[dim + 1 :]) if dim < inp.ndim - 1 else 1
    N = outer * inner

    result = torch.empty(
        tuple(shape[:dim]) + tuple(shape[dim + 1 :]) + (Q,),
        dtype=inp.dtype,
        device=inp.device,
    )

    contig = inp.is_contiguous()
    if M <= RESIDENT_M_LIMIT and Q <= RESIDENT_MAX_Q:
        # ---- resident selection: no materialization, no indices, one kernel ----
        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_Q = triton.next_power_of_2(max(Q, 1))
        # inner > 1: tile inner lanes (coalesced across the tile); the strided
        # program footprint is TILE_N x BLOCK_M. inner == 1: the same tiling
        # covers consecutive rows of the contiguous input.
        INNER_TILE = inner > 1
        if INNER_TILE:
            # enough programs to fill the device without widening each tile's
            # sort (measured optimum band on S5000)
            TILE_N = min(
                RESIDENT_TILE_N_CAP,
                triton.next_power_of_2(
                    max(
                        1,
                        (N + RESIDENT_TARGET_PROGRAMS - 1) // RESIDENT_TARGET_PROGRAMS,
                    )
                ),
            )
            grid = (outer * triton.cdiv(inner, TILE_N),)
            inner_arg = inner
        else:
            TILE_N = _pick_tile(BLOCK_M)
            grid = (triton.cdiv(N, TILE_N),)
            inner_arg = N
        src = inp if contig else inp.contiguous()
        fused_q_validation = q_needs_validation and (
            grid[0] <= RESIDENT_FUSED_Q_VALIDATE_MAX_PROGRAMS
        )
        status = (
            torch.empty(1, dtype=torch.int32, device=inp.device)
            if q_needs_validation
            else result.view(-1)
        )
        with torch_device_fn.device(inp.device):
            if q_needs_validation and not fused_q_validation:
                quantile_q_validate_kernel[(1,)](q_t, status, Q, BLOCK_Q=BLOCK_Q)
                if status.item() != 0:
                    raise RuntimeError(
                        "quantile() q values must be in the range [0, 1]"
                    )
            quantile_resident_kernel[grid](
                src,
                q_t,
                status,
                result.view(-1),
                M,
                inner_arg,
                Q,
                BLOCK_M=BLOCK_M,
                BLOCK_Q=BLOCK_Q,
                TILE_N=TILE_N,
                interpolation=interpolation,
                INNER_TILE=INNER_TILE,
                VALIDATE_Q=fused_q_validation,
                PRESERVE_ZERO_SIGN=(
                    BLOCK_M <= 4 or (M == BLOCK_M and BLOCK_M <= 64 and N <= 64)
                ),
                num_warps=8 if BLOCK_M >= 512 else 4,
            )
        if fused_q_validation and status.item() != 0:
            raise RuntimeError("quantile() q values must be in the range [0, 1]")
    elif inp.dtype == torch.float32 and _use_radix_select(M, N, Q, interpolation):
        # ---- large-M radix selection: no sort, no indices, no transpose ----
        src = inp if contig else inp.contiguous()
        ranks_per_q = 2 if interpolation in ("linear", "midpoint") else 1
        rank_values = torch.empty(
            (N, Q * ranks_per_q), dtype=inp.dtype, device=inp.device
        )
        BLOCK_Q = triton.next_power_of_2(max(Q, 1))
        status = torch.empty(1, dtype=torch.int32, device=inp.device)
        with torch_device_fn.device(inp.device):
            if q_needs_validation:
                quantile_q_validate_kernel[(1,)](q_t, status, Q, BLOCK_Q=BLOCK_Q)
                if status.item() != 0:
                    raise RuntimeError(
                        "quantile() q values must be in the range [0, 1]"
                    )
            quantile_radix_rank_kernel[(outer, inner, Q * ranks_per_q)](
                src,
                q_t,
                rank_values,
                M,
                inner,
                Q,
                BLOCK_M=RADIX_SELECT_BLOCK_M,
                RADIX_BITS_=RADIX_SELECT_BITS,
                RANKS_PER_Q=ranks_per_q,
                interpolation=interpolation,
                num_warps=4,
            )
            quantile_rank_interpolate_kernel[(triton.cdiv(N, 32),)](
                rank_values,
                q_t,
                result.view(-1),
                N,
                M,
                Q,
                BLOCK_Q=BLOCK_Q,
                BLOCK_N=32,
                RANKS_PER_Q=ranks_per_q,
                interpolation=interpolation,
                num_warps=4,
            )
    else:
        # ---- large-M fallback: native sort + gather/interp ----
        BLOCK_Q = triton.next_power_of_2(max(Q, 1))
        if q_needs_validation:
            status = torch.empty(1, dtype=torch.int32, device=inp.device)
            with torch_device_fn.device(inp.device):
                quantile_q_validate_kernel[(1,)](q_t, status, Q, BLOCK_Q=BLOCK_Q)
            if status.item() != 0:
                raise RuntimeError("quantile() q values must be in the range [0, 1]")
        # The sort needs rows contiguous along the reduction dim; materialize
        # only here (movedim is metadata-only; contiguous is one copy).
        if dim == inp.ndim - 1:
            rows = inp if contig else inp.contiguous()
        else:
            rows = torch.movedim(inp, dim, -1).contiguous()
        sorted_vals = _native_sort_rows(rows)
        TILE_N = 32
        grid = (triton.cdiv(N, TILE_N),)
        with torch_device_fn.device(inp.device):
            quantile_gather_kernel[grid](
                sorted_vals,
                q_t,
                result.view(-1),
                M,
                N,
                Q,
                BLOCK_Q=BLOCK_Q,
                TILE_N=TILE_N,
                interpolation=interpolation,
                INNER_TILE=False,
                num_warps=4,
            )

    # ---- output layout ----
    # result is [..., Q] (reduced dim dropped). aten: for tensor q with ndim > 0
    # the output is [Q, ...] (with the reduced dim re-inserted when keepdim);
    # for float/0-dim q the Q dimension is squeezed.
    if squeeze_q:
        output = result.squeeze(-1)
        if keepdim:
            output = output.unsqueeze(dim)
    else:
        output = result.reshape(tuple(shape[:dim]) + tuple(shape[dim + 1 :]) + q_shape)
        output = output.movedim(
            tuple(range(output.ndim - len(q_shape), output.ndim)),
            tuple(range(len(q_shape))),
        )
        if keepdim:
            output = output.unsqueeze(dim + len(q_shape))

    if out is not None:
        if tuple(out.shape) != tuple(output.shape):
            out.resize_(output.shape)
        out.copy_(output)
        return out
    return output
