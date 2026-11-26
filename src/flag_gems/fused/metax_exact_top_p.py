# SPDX-License-Identifier: Apache-2.0
"""Device-side top-p for the long rows where MetaX uses a stable radix sort.

Valid BF16-lattice rows use the compact histogram. Arbitrary FP32 rows build
their permutation on device. All rows use the locked softmax and cumsum.
"""

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.fused import top_p_bf16_histogram as hist


@triton.jit
def _sort_local(
    X,
    Flag,
    Pairs,
    N: tl.constexpr,
    STRIDE,
    BLOCK: tl.constexpr,
    SIGNED_ZERO_KEYS: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Flag + row) == 0:
        return
    for tile in range(tl.program_id(1), tl.cdiv(N, BLOCK), tl.num_programs(1)):
        j = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + row * STRIDE + j, j < N, other=0.0)
        bits = x.to(tl.uint32, bitcast=True)
        # Native radix keeps signed zeros stable together. FlagGems orders their bit keys.
        if not SIGNED_ZERO_KEYS:
            bits = tl.where(x == 0.0, 0, bits).to(tl.uint32)
        key = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
        pair = (key.to(tl.uint64) << 32) | j.to(tl.uint64)
        pair = tl.where(j < N, pair, 0xFFFFFFFFFFFFFFFF).to(tl.uint64)
        ordered = tl.sort(pair, descending=False)
        tl.store(Pairs + row * N + j, ordered, j < N)


@triton.jit
def _merge_runs(
    From,
    To,
    Flag,
    N: tl.constexpr,
    RUN: tl.constexpr,
    LOG_RUN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Flag + row) == 0:
        return
    for tile in range(tl.program_id(1), tl.cdiv(N, BLOCK), tl.num_programs(1)):
        j = tile * BLOCK + tl.arange(0, BLOCK)
        base = (j // (2 * RUN)) * (2 * RUN)
        other = base + tl.where(j - base < RUN, RUN, 0)
        own = tl.load(From + row * N + j, j < N, other=0).to(tl.uint64)
        last = tl.full((BLOCK,), -1, tl.int32)
        for b in tl.static_range(LOG_RUN, -1, -1):
            probe = last + (1 << b)
            at = other + probe
            value = tl.load(From + row * N + at, (probe < RUN) & (at < N), other=0).to(
                tl.uint64
            )
            last = tl.where((probe < RUN) & (at < N) & (value < own), probe, last)
        position = base + (j % RUN) + last + 1
        tl.store(To + row * N + position, own, j < N)


@triton.jit
def _gather_sorted(
    X, Flag, Pairs, Sorted, N: tl.constexpr, STRIDE, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    if tl.load(Flag + row) == 0:
        return
    for tile in range(tl.program_id(1), tl.cdiv(N, BLOCK), tl.num_programs(1)):
        j = tile * BLOCK + tl.arange(0, BLOCK)
        pairs = tl.load(Pairs + row * N + j, j < N, other=0)
        col = (pairs.to(tl.uint64) & 0xFFFFFFFF).to(tl.int32)
        x = tl.load(X + row * STRIDE + col, j < N, other=0.0)
        tl.store(Sorted + row * N + j, x, j < N)


@triton.jit
def _bounds(
    Cdf,
    P,
    Count,
    Last,
    FirstKept,
    N: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tile * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(Cdf + row * N + j, j < N, other=0.0)
    remove = (j < N - 1) & (val <= 1.0 - tl.load(P + row))
    dst = row * NT + tile
    tl.store(Count + dst, tl.sum(remove.to(tl.int32), 0))
    tl.store(Last + dst, tl.max(tl.where(remove, j, -1), 0))
    tl.store(FirstKept + dst, tl.min(tl.where(~remove & (j < N), j, N), 0))


@triton.jit
def _classify(
    Sorted,
    Counts,
    Groups,
    Flag,
    Count,
    Last,
    FirstKept,
    Pivot,
    Low,
    RemoveEqual,
    N: tl.constexpr,
    NT: tl.constexpr,
    SCAN: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Flag + row) != 0:
        return
    i = tl.arange(0, SCAN)
    last = tl.max(tl.load(Last + row * NT + i, i < NT, other=-1), 0)
    total = tl.sum(tl.load(Count + row * NT + i, i < NT, other=0), 0)
    pivot = tl.load(Sorted + row * N + last, last >= 0, other=-float("inf"))
    tl.store(Pivot + row, pivot)
    if total != last + 1:
        first = tl.min(tl.load(FirstKept + row * NT + i, i < NT, other=N), 0)
        tl.store(Low + row, tl.load(Sorted + row * N + first))
        tl.store(Flag + row, 2)
    else:
        key = hist._ordered_key(pivot.to(tl.uint32, bitcast=True))
        group = key // 512
        local = tl.load(Counts + row * 16384 + key - 1, key % 512 != 0, other=0)
        prev = tl.load(Groups + row * 32 + group - 1, group > 0, other=0)
        tl.store(RemoveEqual + row, last + 1 - local - prev)


@triton.jit
def _repair_special(
    X,
    P,
    Cdf,
    Sorted,
    Pairs,
    Flag,
    Counts,
    Groups,
    Pivot,
    Low,
    N: tl.constexpr,
    STRIDE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    mode = tl.load(Flag + row)
    if mode == 0:
        return
    for tile in range(tl.program_id(1), tl.cdiv(N, BLOCK), tl.num_programs(1)):
        j = tile * BLOCK + tl.arange(0, BLOCK)
        threshold = 1.0 - tl.load(P + row)
        if mode == 1:
            # The packed column breaks equal-key ties in original input order.
            pair = tl.load(Pairs + row * N + j, j < N, other=0)
            col = (pair.to(tl.uint64) & 0xFFFFFFFF).to(tl.int32)
            x = tl.load(Sorted + row * N + j, j < N, other=0.0)
            cdf = tl.load(Cdf + row * N + j, j < N, other=0.0)
            remove = (j < N - 1) & (cdf <= threshold)
            tl.store(X + row * STRIDE + col, tl.where(remove, -float("inf"), x), j < N)
        else:
            # A rounded CDF can occasionally dip between adjacent scan tiles.
            # Only values between the first kept and the last removed rank need
            # a stable tie rank. Read the original row and stage writes separately.
            x = tl.load(X + row * STRIDE + j, j < N, other=0.0)
            lo = tl.load(Low + row)
            hi = tl.load(Pivot + row)
            keep = x > hi
            ambiguous = (j < N) & (x >= lo) & (x <= hi)
            keys = hist._ordered_key(x.to(tl.uint32, bitcast=True))
            key = tl.min(tl.where(ambiguous, keys, 16384), 0)
            while key < 16384:
                eq = (j < N) & (keys == key)
                value = hist._from_ordered_key(key)
                before = 0
                for prev_tile in range(tile):
                    prev_j = prev_tile * BLOCK + tl.arange(0, BLOCK)
                    prev_x = tl.load(X + row * STRIDE + prev_j)
                    before += tl.sum((prev_x == value).to(tl.int32), 0)
                group = key // 512
                a = tl.load(Counts + row * 16384 + key - 1, key % 512 != 0, other=0)
                b = tl.load(Groups + row * 32 + group - 1, group > 0, other=0)
                rank = a + b + before + tl.cumsum(eq.to(tl.int32), 0) - 1
                cdf = tl.load(Cdf + row * N + rank, eq, other=0.0)
                keep = keep | (eq & ((rank == N - 1) | ~(cdf <= threshold)))
                ambiguous = ambiguous & ~eq
                key = tl.min(tl.where(ambiguous, keys, 16384), 0)
            tl.store(Sorted + row * N + j, tl.where(keep, x, -float("inf")), j < N)


@triton.jit
def _copy_nonprefix(X, Sorted, Flag, N: tl.constexpr, STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(Flag + row) == 2:
        for tile in range(tl.program_id(1), tl.cdiv(N, BLOCK), tl.num_programs(1)):
            j = tile * BLOCK + tl.arange(0, BLOCK)
            value = tl.load(Sorted + row * N + j, j < N, other=0.0)
            tl.store(X + row * STRIDE + j, value, j < N)


def apply_exact_top_p_(logits, p):
    batch, width = logits.shape
    if batch == 0:
        return logits
    device = logits.device
    tile = 1024
    tiles = triton.cdiv(width, tile)
    scan = triton.next_power_of_2(tiles)
    grid = (batch, tiles)
    stride = logits.stride(0)
    # Conditional repairs launch a few waves even when every row uses the histogram.
    # For a rare row each worker handles multiple vocabulary tiles.
    workers = max(2, triton.cdiv(208, batch))
    repair_grid = (batch, min(workers, tiles))

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    counts = torch.zeros((batch, 16384), dtype=torch.int32, device=device)
    flags = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    groups = empty((batch, 32), torch.int32)
    sorted_x = empty((batch, width), torch.float32)
    pairs = empty((batch, width), torch.int64)
    other = torch.empty_like(pairs)
    count = empty((batch, tiles), torch.int32)
    last = torch.empty_like(count)
    first = torch.empty_like(count)
    pivot = empty(batch, torch.float32)
    low = torch.empty_like(pivot)
    remove_equal = empty(batch, torch.int32)

    registrar = getattr(flag_gems, "current_work_registrar", None)
    signed_zero_keys = registrar is not None and "sort" in registrar.get_all_keys()
    with torch.cuda.device(device):
        hist._count_keys[grid](logits, counts, flags, width, stride)
        hist._prefix_counts[(batch, 32)](counts, groups)
        hist._prefix_groups[(batch,)](groups)
        hist._expand_values[grid](counts, groups, sorted_x, width)
        run = 2048
        _sort_local[(batch, min(workers, triton.cdiv(width, run)))](
            logits, flags, pairs, width, stride, run, signed_zero_keys, num_warps=8
        )
        while run < width:
            _merge_runs[repair_grid](
                pairs, other, flags, width, run, run.bit_length() - 1, tile
            )
            pairs, other = other, pairs
            run *= 2
        _gather_sorted[repair_grid](logits, flags, pairs, sorted_x, width, stride, tile)
        cdf = sorted_x.softmax(-1)
        torch.cumsum(cdf, dim=-1, out=cdf)
        _bounds[grid](cdf, p, count, last, first, width, tiles, tile)
        _classify[(batch,)](
            sorted_x,
            counts,
            groups,
            flags,
            count,
            last,
            first,
            pivot,
            low,
            remove_equal,
            width,
            tiles,
            scan,
        )
        hist._count_ties[grid](logits, pivot, flags, count, width, stride, tiles)
        hist._apply_mask[grid](
            logits, pivot, remove_equal, flags, count, width, stride, tiles, scan
        )
        _repair_special[repair_grid](
            logits,
            p,
            cdf,
            sorted_x,
            pairs,
            flags,
            counts,
            groups,
            pivot,
            low,
            width,
            stride,
            tile,
        )
        _copy_nonprefix[repair_grid](logits, sorted_x, flags, width, stride, tile)
    return logits
