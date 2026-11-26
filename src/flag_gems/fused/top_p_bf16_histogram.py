"""Exact top-p for FP32 views of BF16 logits on MetaX.

The histogram reconstructs the FP32 sequence seen by the reference radix sort
without sorting token indices. Softmax and cumulative sum use the reference
operations. Only the boundary key can need a stable tie rank.
"""

import torch
import triton
import triton.language as tl

from flag_gems.fused.top_p_pivot import _reference_top_p_

VOCAB_WIDTH = 130560
KEY_COUNT: tl.constexpr = 16384
GROUP_SIZE: tl.constexpr = 512
GROUP_COUNT: tl.constexpr = 32
TILE: tl.constexpr = 1024
# BF16 magnitude encodings for [2**-31, 2**17).
MIN_MAGNITUDE: tl.constexpr = 96 * 128
MAGNITUDE_SPAN: tl.constexpr = 48 * 128


@triton.jit
def _ordered_key(bits):
    magnitude = (bits >> 16).to(tl.int32) & 32767
    delta = magnitude - MIN_MAGNITUDE
    span = tl.full((), MAGNITUDE_SPAN, tl.int32)
    negative = (bits >> 31) != 0
    return tl.where(
        bits == 0xFF800000,
        0,
        tl.where(
            magnitude == 0, span + 1, tl.where(negative, span - delta, delta + span + 2)
        ),
    )


@triton.jit
def _from_ordered_key(key):
    span = tl.full((), MAGNITUDE_SPAN, tl.int32)
    negative = key <= span
    delta = tl.where(negative, span - key, key - (span + 2))
    upper = tl.where(negative, 32768, 0) | (delta + MIN_MAGNITUDE)
    upper = tl.where(key == 0, 0xFF80, tl.where(key == span + 1, 0, upper))
    return (upper.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _count_keys(Logits, Counts, Invalid, N: tl.constexpr, RS: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * TILE + tl.arange(0, TILE)
    x = tl.load(Logits + row * RS + j, j < N, other=0.0)
    bits = x.to(tl.uint32, bitcast=True)
    magnitude = (bits >> 16) & 32767
    supported = (magnitude >= MIN_MAGNITUDE) & (
        magnitude < MIN_MAGNITUDE + MAGNITUDE_SPAN
    )
    # -inf participates in the ordering. Refuse negative zero because the
    # ordering of signed zero keys can differ from the comparison x == pivot.
    valid = ((bits & 65535) == 0) & (bits != 0x80000000)
    valid = valid & (supported | (magnitude == 0) | (bits == 0xFF800000))
    if tl.sum(((j < N) & ~valid).to(tl.int32), 0) != 0:
        tl.atomic_or(Invalid + row, 1, sem="relaxed")
        tl.atomic_or(Invalid + tl.num_programs(0), 1, sem="relaxed")
    key = _ordered_key(tl.where(valid, bits, 0).to(tl.uint32))
    tl.atomic_add(Counts + row * KEY_COUNT + key, 1, j < N, sem="relaxed")


@triton.jit
def _prefix_counts(Counts, Groups):
    row = tl.program_id(0)
    group = tl.program_id(1)
    bins = tl.arange(0, GROUP_SIZE)
    offset = row * KEY_COUNT + group * GROUP_SIZE + bins
    count = tl.load(Counts + offset)
    tl.store(Counts + offset, tl.cumsum(count, 0))
    tl.store(Groups + row * GROUP_COUNT + group, tl.sum(count, 0))


@triton.jit
def _prefix_groups(Groups):
    row = tl.program_id(0)
    j = tl.arange(0, GROUP_COUNT)
    totals = tl.load(Groups + row * GROUP_COUNT + j)
    tl.store(Groups + row * GROUP_COUNT + j, tl.cumsum(totals, 0))


@triton.jit
def _expand_values(Counts, Groups, Sorted, N: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.program_id(1) * TILE + tl.arange(0, TILE)
    j = tl.minimum(rank, N - 1)
    coarse = tl.full((TILE,), -1, tl.int32)
    row_groups = Groups + row * GROUP_COUNT
    for n in tl.static_range(4, -1, -1):
        candidate = coarse + (1 << n)
        end = tl.load(row_groups + candidate, candidate < GROUP_COUNT, other=N)
        coarse = tl.where(end <= j, candidate, coarse)
    local_rank = j - tl.load(row_groups + coarse, coarse >= 0, other=0)
    base = (coarse + 1) * GROUP_SIZE
    low = tl.full((TILE,), -1, tl.int32)
    for n in tl.static_range(8, -1, -1):
        candidate = low + (1 << n)
        end = tl.load(
            Counts + row * KEY_COUNT + base + candidate, candidate < GROUP_SIZE, other=N
        )
        low = tl.where(end <= local_rank, candidate, low)
    value = _from_ordered_key(base + low + 1)
    tl.store(Sorted + row * N + rank, value, rank < N)


@triton.jit
def _check_cdf_prefix(Cdf, P, TileCount, TileLast, N: tl.constexpr, NT: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tile * TILE + tl.arange(0, TILE)
    cumulative = tl.load(Cdf + row * N + j, j < N - 1, other=float("inf"))
    # The locked sampler always keeps the highest-ranked token.
    remove = (j < N - 1) & (cumulative <= 1.0 - tl.load(P + row))
    tl.store(TileCount + row * NT + tile, tl.sum(remove.to(tl.int32), 0))
    tl.store(TileLast + row * NT + tile, tl.max(tl.where(remove, j, -1), 0))


@triton.jit
def _resolve_pivot(
    Sorted,
    Counts,
    Groups,
    Invalid,
    TileCount,
    TileLast,
    Pivot,
    RemoveEqual,
    N: tl.constexpr,
    NT: tl.constexpr,
    SCAN: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, SCAN)
    count = tl.load(TileCount + row * NT + i, i < NT, other=0)
    last = tl.load(TileLast + row * NT + i, i < NT, other=-1)
    last_removed = tl.max(last, 0)
    # Check the actual FP32 CDF mask rather than relying on monotonicity across
    # independently accumulated scan tiles.
    if tl.sum(count, 0) != last_removed + 1:
        tl.atomic_or(Invalid + row, 1, sem="relaxed")
        tl.atomic_or(Invalid + tl.num_programs(0), 1, sem="relaxed")
    pivot = tl.load(
        Sorted + row * N + last_removed, last_removed >= 0, other=-float("inf")
    )
    key = _ordered_key(pivot.to(tl.uint32, bitcast=True))
    group = key // GROUP_SIZE
    key_below = tl.load(
        Counts + row * KEY_COUNT + key - 1, (key % GROUP_SIZE) > 0, other=0
    )
    group_below = tl.load(Groups + row * GROUP_COUNT + group - 1, group > 0, other=0)
    tl.store(Pivot + row, pivot)
    tl.store(RemoveEqual + row, last_removed + 1 - group_below - key_below)


@triton.jit
def _count_ties(
    Logits,
    Pivot,
    Invalid,
    TileCount,
    N: tl.constexpr,
    RS: tl.constexpr,
    NT: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if tl.load(Invalid + row) == 0:
        j = tile * TILE + tl.arange(0, TILE)
        x = tl.load(Logits + row * RS + j, j < N, other=0.0)
        pivot = tl.load(Pivot + row)
        tl.store(
            TileCount + row * NT + tile,
            tl.sum(((j < N) & (x == pivot)).to(tl.int32), 0),
        )


@triton.jit
def _apply_mask(
    Logits,
    Pivot,
    RemoveEqual,
    Invalid,
    TileCount,
    N: tl.constexpr,
    RS: tl.constexpr,
    NT: tl.constexpr,
    SCAN: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if tl.load(Invalid + row) == 0:
        j = tile * TILE + tl.arange(0, TILE)
        x = tl.load(Logits + row * RS + j, j < N, other=0.0)
        pivot = tl.load(Pivot + row)
        limit = tl.load(RemoveEqual + row)
        keep = x >= pivot
        if limit > 0:
            i = tl.arange(0, SCAN)
            count = tl.load(TileCount + row * NT + i, i < tile, other=0)
            equal = (x == pivot) & (j < N)
            rank = tl.sum(count, 0) + tl.cumsum(equal.to(tl.int32), 0)
            keep = keep & (~equal | (rank > limit))
        tl.store(Logits + row * RS + j, tl.where(keep, x, -float("inf")), j < N)


def apply_top_p_bf16_histogram_(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply a pure top-p mask to the MetaX sampler's FP32 logits in place."""
    batch, width = logits.shape
    if batch == 0:
        return logits
    if torch.cuda.is_current_stream_capturing():
        return _reference_top_p_(logits, p)

    device = logits.device
    tile = TILE
    tiles = triton.cdiv(width, tile)
    groups = GROUP_COUNT
    grid = (batch, tiles)
    scan = triton.next_power_of_2(tiles)
    counts = torch.zeros((batch, KEY_COUNT), device=device, dtype=torch.int32)
    invalid = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    group_ends = torch.empty((batch, groups), device=device, dtype=torch.int32)
    sorted_values = torch.empty((batch, width), device=device, dtype=torch.float32)
    tile_count = torch.empty((batch, tiles), device=device, dtype=torch.int32)
    tile_last = torch.empty_like(tile_count)
    pivot = torch.empty(batch, device=device, dtype=torch.float32)
    remove_equal = torch.empty(batch, device=device, dtype=torch.int32)

    _count_keys[grid](logits, counts, invalid, width, logits.stride(0))
    _prefix_counts[(batch, groups)](counts, group_ends)
    _prefix_groups[(batch,)](group_ends)
    _expand_values[grid](counts, group_ends, sorted_values, width)
    cdf = sorted_values.softmax(-1)
    torch.cumsum(cdf, dim=-1, out=cdf)
    _check_cdf_prefix[grid](cdf, p, tile_count, tile_last, width, tiles)
    _resolve_pivot[(batch,)](
        sorted_values,
        counts,
        group_ends,
        invalid,
        tile_count,
        tile_last,
        pivot,
        remove_equal,
        width,
        tiles,
        scan,
    )
    _count_ties[grid](
        logits, pivot, invalid, tile_count, width, logits.stride(0), tiles
    )
    _apply_mask[grid](
        logits,
        pivot,
        remove_equal,
        invalid,
        tile_count,
        width,
        logits.stride(0),
        tiles,
        scan,
    )

    # One scalar readback. Only flagged rows need the original stable sort.
    if invalid[batch].item():
        rows = torch.nonzero(invalid[:batch], as_tuple=False).flatten()
        repaired = _reference_top_p_(
            logits.index_select(0, rows), p.index_select(0, rows)
        )
        logits.index_copy_(0, rows, repaired)
    return logits
