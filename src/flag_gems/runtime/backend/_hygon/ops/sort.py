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

import torch
import triton
import triton.language as tl

from flag_gems.ops.argsort import (
    _argsort_merge_entry,
    _argsort_partner,
    _argsort_row_offset,
    _argsort_tiles,
    _packed_load,
    _packed_partition,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _sort_key(values, DESC: tl.constexpr, NAN_LAST: tl.constexpr):
    bits: tl.constexpr = values.dtype.primitive_bitwidth
    uint: tl.constexpr = tl.uint64 if bits == 64 else tl.uint32
    mask = tl.full((), (1 << bits) - 1, uint)
    sign = tl.full((), 1, uint) << (bits - 1)
    if values.dtype.is_floating():
        raw = values.to(tl.uint16 if bits == 16 else uint, bitcast=True).to(uint)
        # CUB treats signed zeros as equal but orders other keys by their bits.
        raw = tl.where(values == 0, 0, raw)
        key = raw ^ tl.where((raw & sign) != 0, mask, sign)
        if NAN_LAST:
            key = tl.where(values != values, mask, key)
    else:
        key = values.to(uint) & mask
        if values.dtype.is_int_signed():
            key ^= sign
    if DESC:
        key ^= mask
    return key


@triton.jit
def _sort_pack(
    values,
    indices,
    INDEX_BITS: tl.constexpr,
    DESC: tl.constexpr,
    NAN_LAST: tl.constexpr,
):
    key = _sort_key(values, DESC, NAN_LAST)
    if values.dtype.primitive_bitwidth + INDEX_BITS <= 32:
        packed = (key << INDEX_BITS) | indices.to(tl.uint32)
    else:
        packed = (key.to(tl.uint64) << INDEX_BITS) | indices.to(tl.uint64)
    return packed


@triton.jit
def _sort_before(a, ai, b, bi, DESC: tl.constexpr, NAN_LAST: tl.constexpr):
    if a.dtype.is_floating():
        ak = _sort_key(a, DESC, NAN_LAST)
        bk = _sort_key(b, DESC, NAN_LAST)
        before = (ak < bk) | ((ak == bk) & (ai < bi))
    else:
        before = a > b if DESC else a < b
        before = before | ((a == b) & (ai < bi))
    return before


@triton.jit
def sort_copy_kernel(
    inp,
    out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    columns: tl.constexpr = triton.cdiv(N, BLOCK)
    row = (tl.program_id(0) // columns) * BLOCK + tl.arange(0, BLOCK)
    col = (tl.program_id(0) % columns) * BLOCK + tl.arange(0, BLOCK)
    source = _argsort_row_offset(row, SHAPE, STRIDES)[:, None] + col[None, :] * AXIS
    valid = (row[:, None] < ROWS) & (col[None, :] < N)
    value = tl.load(inp + source, valid, other=0)
    target = row[:, None].to(tl.int64) * N + col[None, :]
    tl.store(out + target, value, valid)


@triton.jit
def sort_rows_packed_kernel(
    inp,
    values_out,
    indices_out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    NAN_LAST: tl.constexpr,
    MINMAX: tl.constexpr,
):
    chunks: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    row = task // chunks * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    lane = tl.arange(0, BLOCK)
    col = (task % chunks) * BLOCK + lane
    base = row.to(tl.int64) * N
    valid = (row[:, None] < ROWS) & (col[None, :] < N)
    values = tl.load(inp + base[:, None] + col[None, :], valid, other=0)
    indices = tl.broadcast_to(col[None, :], (ROW_BLOCK, BLOCK))
    key = _sort_pack(values, indices, INDEX_BITS, DESC, NAN_LAST)
    key = tl.where(valid, key, ~tl.full((), 0, key.dtype))
    if BLOCK > 1:
        if MINMAX:
            for stage in tl.static_range(1, LOG_BLOCK + 1):
                for step in tl.static_range(stage - 1, -1, -1):
                    other = _argsort_partner(key, step)
                    forward = (lane & (1 << stage)) == 0
                    lower = (lane & (1 << step)) == 0
                    key = tl.where(
                        (forward == lower)[None, :],
                        tl.minimum(key, other),
                        tl.maximum(key, other),
                    )
        else:
            key = tl.sort(key, dim=1, descending=False)
    indices = (key & ((1 << INDEX_BITS) - 1)).to(tl.int32)
    values = tl.load(inp + base[:, None] + indices, valid, other=0)
    offset = base[:, None] + col[None, :]
    tl.store(values_out + offset, values, valid)
    tl.store(indices_out + offset, indices, valid)


@triton.jit
def sort_singleton_kernel(
    inp,
    values_out,
    indices_out,
    NUMEL: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL
    source = _argsort_row_offset(offsets, SHAPE, STRIDES)
    values = tl.load(inp + source, mask, other=0)
    tl.store(values_out + offsets, values, mask)
    tl.store(indices_out + offsets, tl.zeros((BLOCK,), tl.int64), mask)


@libentry()
@triton.heuristics(
    {
        "LOG_BLOCK": lambda args: args["BLOCK"].bit_length() - 1,
        "ROW_BLOCK": lambda args: max(1, 128 // args["BLOCK"]) if args["FINAL"] else 1,
        "MINMAX": lambda args: args["BLOCK"] >= 2048
        or (not args["FINAL"] and args["BLOCK"] == 1024),
    }
)
@triton.jit
def sort_rows_kernel(
    inp,
    values_out,
    indices_out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    PACKED: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    NAN_LAST: tl.constexpr,
    FINAL: tl.constexpr,
    MINMAX: tl.constexpr,
):
    chunks: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    row = task // chunks * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    lane = tl.arange(0, BLOCK)
    col = (task % chunks) * BLOCK + lane
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    valid = (row[:, None] < ROWS) & (col[None, :] < N)
    values = tl.load(inp + base[:, None] + col[None, :] * AXIS, valid, other=0)
    indices = tl.broadcast_to(col[None, :], (ROW_BLOCK, BLOCK))
    if PACKED:
        key = _sort_pack(values, indices, INDEX_BITS, DESC, NAN_LAST)
        key = tl.where(valid, key, ~tl.full((), 0, key.dtype))
        if BLOCK > 1:
            if MINMAX:
                for stage in tl.static_range(1, LOG_BLOCK + 1):
                    for step in tl.static_range(stage - 1, -1, -1):
                        other = _argsort_partner(key, step)
                        forward = (lane & (1 << stage)) == 0
                        lower = (lane & (1 << step)) == 0
                        key = tl.where(
                            (forward == lower)[None, :],
                            tl.minimum(key, other),
                            tl.maximum(key, other),
                        )
            else:
                key = tl.sort(key, dim=1, descending=False)
        if FINAL:
            indices = (key & ((1 << INDEX_BITS) - 1)).to(tl.int32)
            # Gather the original bits, retaining signed zero and NaN payloads.
            values = tl.load(inp + base[:, None] + indices * AXIS, valid, other=0)
    else:
        for stage in tl.static_range(1, LOG_BLOCK + 1):
            for step in tl.static_range(stage - 1, -1, -1):
                ov = _argsort_partner(values, step)
                oi = _argsort_partner(indices, step)
                before = _sort_before(ov, oi, values, indices, DESC, NAN_LAST)
                before = (oi < N) & (before | (indices >= N))
                forward = (col & (1 << stage)) == 0
                lower = (col & (1 << step)) == 0
                swap = tl.where((forward == lower)[None, :], before, ~before)
                values = tl.where(swap, ov, values)
                indices = tl.where(swap, oi, indices)
    if FINAL:
        offset = (
            _argsort_row_offset(row, SHAPE, OUT_STRIDES)[:, None]
            + col[None, :] * OUT_AXIS
        )
        tl.store(values_out + offset, values, valid)
        tl.store(indices_out + offset, indices, valid)
    else:
        offset = row[:, None].to(tl.int64) * N + col[None, :]
        tl.store(values_out + offset, key, valid)


@libentry()
@triton.heuristics(
    {
        "STEPS": lambda args: (args["RUN"] + 1).bit_length(),
        "LOG_BLOCK": lambda args: args["BLOCK"].bit_length() - 1,
    }
)
@triton.jit
def sort_merge_kernel(
    inp,
    keys,
    next_keys,
    values_out,
    indices_out,
    N: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    FINAL: tl.constexpr,
    DESC: tl.constexpr,
):
    blocks: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    row = task // blocks
    start = (task - row * blocks) * BLOCK
    pair = start // (2 * RUN) * (2 * RUN)
    alen = tl.minimum(RUN, N - pair)
    bs = pair + RUN
    blen = tl.maximum(0, tl.minimum(RUN, N - bs))
    diag = start - pair
    end = tl.minimum(diag + BLOCK, alen + blen)
    base = row.to(tl.int64) * N
    # A whole pair of runs needs no merge-path partition search.
    if BLOCK == 2 * RUN:
        a0 = 0
        a1 = alen
        b0 = 0
        b1 = blen
    else:
        boundary = tl.arange(0, 2)
        diagonals = tl.where(boundary == 0, diag, end)
        cuts = _packed_partition(keys, base, pair, bs, alen, blen, diagonals, STEPS)
        a0 = tl.sum(tl.where(boundary == 0, cuts, 0), 0)
        a1 = tl.sum(tl.where(boundary == 1, cuts, 0), 0)
        b0 = diag - a0
        b1 = end - a1
    na = a1 - a0
    nb = b1 - b0
    lane = tl.arange(0, BLOCK)
    isa = lane < na
    isb = lane >= BLOCK - nb
    src = tl.where(isa, pair + a0 + lane, bs + b1 - 1 - (lane - (BLOCK - nb)))
    key = _packed_load(keys, base + src, isa | isb)
    if key.dtype.primitive_bitwidth == 64:
        key = tl.bitonic_merge(key, descending=False)
    else:
        for step in tl.static_range(LOG_BLOCK - 1, -1, -1):
            other = _argsort_partner(key, step)
            lower = (lane & (1 << step)) == 0
            key = tl.where(lower, tl.minimum(key, other), tl.maximum(key, other))
    col = start + lane
    if FINAL:
        index = (key & ((1 << INDEX_BITS) - 1)).to(tl.int32)
        # Only signed-zero normalization loses bits; other values decode exactly.
        value_bits: tl.constexpr = inp.dtype.element_ty.primitive_bitwidth
        value_mask: tl.constexpr = (1 << value_bits) - 1
        sign: tl.constexpr = 1 << (value_bits - 1)
        value_key = (key >> INDEX_BITS).to(tl.uint32)
        if DESC:
            value_key ^= value_mask
        if inp.dtype.element_ty.is_floating():
            raw = value_key ^ tl.where((value_key & sign) != 0, sign, value_mask)
            raw_type: tl.constexpr = tl.uint16 if value_bits == 16 else tl.uint32
            values = raw.to(raw_type).to(inp.dtype.element_ty, bitcast=True)
            src = _argsort_row_offset(row, SHAPE, STRIDES) + index.to(tl.int64) * AXIS
            zero = tl.load(inp + src, (col < N) & (value_key == sign), other=0)
            values = tl.where(value_key == sign, zero, values)
        else:
            if inp.dtype.element_ty.is_int_signed():
                value_key ^= sign
            values = value_key.to(inp.dtype.element_ty)
        offset = (
            _argsort_row_offset(row, SHAPE, OUT_STRIDES) + col.to(tl.int64) * OUT_AXIS
        )
        tl.store(values_out + offset, values, col < N)
        tl.store(indices_out + offset, index, col < N)
    else:
        tl.store(next_keys + base + col, key, col < N)


@libentry()
@triton.jit
def sort_gather_kernel(
    inp,
    indices,
    values,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = lane // N
    col = lane % N
    offset = _argsort_row_offset(row, SHAPE, OUT_STRIDES) + col * OUT_AXIS
    index = tl.load(indices + offset, lane < NUMEL, other=0)
    src = _argsort_row_offset(row, SHAPE, STRIDES) + index * AXIS
    value = tl.load(inp + src, lane < NUMEL, other=0)
    tl.store(values + offset, value, lane < NUMEL)


@libentry()
@triton.jit
def sort_float64_keys_kernel(
    inp,
    keys,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS: tl.constexpr,
    DESC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = lane // N
    col = lane % N
    src = _argsort_row_offset(row, SHAPE, STRIDES) + col * AXIS
    value = tl.load(inp + src, lane < NUMEL, other=0)
    key = _sort_key(value, DESC, False) ^ (tl.full((), 1, tl.uint64) << 63)
    dst = _argsort_row_offset(row, SHAPE, OUT_STRIDES) + col * OUT_AXIS
    tl.store(keys + dst, key.to(tl.int64, bitcast=True), lane < NUMEL)


@libentry()
@triton.jit
def sort_radix_histogram_kernel(
    X,
    S,
    N: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    FIRST: tl.constexpr,
    DESC: tl.constexpr,
):
    T: tl.constexpr = triton.cdiv(N, B)
    HIST: tl.constexpr = triton.cdiv(M * N, 2) + M * N
    CUMULATIVE: tl.constexpr = HIST + M * T * 256
    tile = tl.program_id(0) % T
    row = tl.program_id(0) // T
    lane = tl.arange(0, B)
    offset = tile * B + lane
    if FIRST:
        ptr = X
    else:
        ptr = S.to(tl.pointer_type(X.dtype.element_ty))
    value = tl.load(ptr + row.to(tl.int64) * N + offset, offset < N, other=0)
    shift: tl.constexpr = 0 if FIRST else 8
    digit = ((_sort_key(value, DESC, False) >> shift) & 255).to(tl.int32)
    counts = tl.histogram(tl.where(offset < N, digit, 0), 256)
    bins = tl.arange(0, 256)
    counts -= tl.where(bins == 0, B - tl.minimum(B, N - tile * B), 0)
    before = tl.cumsum(counts, 0) - counts
    address = (row.to(tl.int64) * T + tile) * 256 + bins
    tl.store(S + HIST + address, counts)
    tl.store(S + CUMULATIVE + address, before)


@libentry()
@triton.jit
def sort_radix_prefix_kernel(
    S, N: tl.constexpr, M: tl.constexpr, B: tl.constexpr, BR: tl.constexpr
):
    T: tl.constexpr = triton.cdiv(N, B)
    BT: tl.constexpr = triton.next_power_of_2(T)
    HIST: tl.constexpr = triton.cdiv(M * N, 2) + M * N
    CUMULATIVE: tl.constexpr = HIST + M * T * 256
    PREFIX: tl.constexpr = CUMULATIVE + M * T * 256
    bins = (tl.program_id(0) % (256 // BR)) * BR + tl.arange(0, BR)
    row = tl.program_id(0) // (256 // BR)
    tile = tl.arange(0, BT)
    offset = (row.to(tl.int64) * T + tile[:, None]) * 256 + bins[None, :]
    counts = tl.load(S + HIST + offset, tile[:, None] < T, other=0)
    previous = tl.load(S + CUMULATIVE + offset, tile[:, None] < T, other=0)
    base = tl.sum(previous, 0)
    prefix = tl.cumsum(counts, 0) - counts + base[None, :] - previous
    tl.store(S + PREFIX + offset, prefix, tile[:, None] < T)


@libentry()
@triton.jit
def sort_radix_scatter_kernel(
    X,
    S,
    Y,
    I,
    N: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    FIRST: tl.constexpr,
    DESC: tl.constexpr,
):
    T: tl.constexpr = triton.cdiv(N, B)
    LOG_B: tl.constexpr = B.bit_length() - 1
    INDICES: tl.constexpr = triton.cdiv(M * N, 2)
    PREFIX: tl.constexpr = INDICES + M * N + 2 * M * T * 256
    tile = tl.program_id(0) % T
    row = tl.program_id(0) // T
    lane = tl.arange(0, B)
    offset = tile * B + lane
    if FIRST:
        ptr = X
    else:
        ptr = S.to(tl.pointer_type(X.dtype.element_ty))
    base = row.to(tl.int64) * N
    value = tl.load(ptr + base + offset, offset < N, other=0)
    shift: tl.constexpr = 0 if FIRST else 8
    digit = ((_sort_key(value, DESC, False) >> shift) & 255).to(tl.int32)
    digit = tl.where(offset < N, digit, 256)
    code = tl.sort((digit << LOG_B) | lane, descending=False)
    permutation = code & (B - 1)
    sorted_digit = code >> LOG_B
    source = tile * B + permutation
    sorted_value = tl.load(ptr + base + source, source < N, other=0)
    if FIRST:
        index = source
    else:
        index = tl.load(S + INDICES + base + source, source < N, other=0)
    prefix = tl.load(
        S + PREFIX + (row.to(tl.int64) * T + tile) * 256 + sorted_digit,
        sorted_digit < 256,
        other=0,
    )
    destination = base + prefix + lane
    if FIRST:
        tl.store(
            S.to(tl.pointer_type(X.dtype.element_ty)) + destination,
            sorted_value,
            sorted_digit < 256,
        )
        tl.store(S + INDICES + destination, index, sorted_digit < 256)
    else:
        tl.store(Y + destination, sorted_value, sorted_digit < 256)
        tl.store(I + destination, index, sorted_digit < 256)


def _sort_radix(x, descending=False):
    n = x.shape[-1]
    m = x.numel() // n
    block = 512
    prefix_bins = 16
    tiles = triton.cdiv(n, block)
    words = triton.cdiv(m * n, 2) + m * n + 3 * m * tiles * 256
    scratch = torch.empty((words,), device=x.device, dtype=torch.int32)
    values = torch.empty_like(x)
    indices = torch.empty(x.shape, device=x.device, dtype=torch.int64)
    grid = (tiles * m,)
    prefix_grid = (256 // prefix_bins * m,)
    for first in (True, False):
        sort_radix_histogram_kernel[grid](
            x, scratch, n, m, block, first, descending, num_warps=4
        )
        sort_radix_prefix_kernel[prefix_grid](
            scratch, n, m, block, prefix_bins, num_warps=4
        )
        sort_radix_scatter_kernel[grid](
            x, scratch, values, indices, n, m, block, first, descending, num_warps=8
        )
    return values, indices


@libentry()
@triton.jit
def sort_radix32_histogram_kernel(
    X,
    S,
    N: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    SHIFT: tl.constexpr,
    DESC: tl.constexpr,
):
    T: tl.constexpr = triton.cdiv(N, B)
    CUMULATIVE: tl.constexpr = M * T * 256
    tile = tl.program_id(0) % T
    row = tl.program_id(0) // T
    lane = tl.arange(0, B)
    offset = tile * B + lane
    value = tl.load(X + row.to(tl.int64) * N + offset, offset < N, other=0)
    digit = ((_sort_key(value, DESC, True) >> SHIFT) & 255).to(tl.int32)
    counts = tl.histogram(tl.where(offset < N, digit, 0), 256)
    bins = tl.arange(0, 256)
    counts -= tl.where(bins == 0, B - tl.minimum(B, N - tile * B), 0)
    before = tl.cumsum(counts, 0) - counts
    address = (row.to(tl.int64) * T + tile) * 256 + bins
    tl.store(S + address, counts)
    tl.store(S + CUMULATIVE + address, before)


@libentry()
@triton.jit
def sort_radix32_prefix_kernel(
    S, N: tl.constexpr, M: tl.constexpr, B: tl.constexpr, BR: tl.constexpr
):
    T: tl.constexpr = triton.cdiv(N, B)
    BT: tl.constexpr = triton.next_power_of_2(T)
    CUMULATIVE: tl.constexpr = M * T * 256
    PREFIX: tl.constexpr = CUMULATIVE + M * T * 256
    bins = (tl.program_id(0) % (256 // BR)) * BR + tl.arange(0, BR)
    row = tl.program_id(0) // (256 // BR)
    tile = tl.arange(0, BT)
    offset = (row.to(tl.int64) * T + tile[:, None]) * 256 + bins[None, :]
    counts = tl.load(S + offset, tile[:, None] < T, other=0)
    previous = tl.load(S + CUMULATIVE + offset, tile[:, None] < T, other=0)
    base = tl.sum(previous, 0)
    prefix = tl.cumsum(counts, 0) - counts + base[None, :] - previous
    tl.store(S + PREFIX + offset, prefix, tile[:, None] < T)


@libentry()
@triton.jit
def sort_radix32_scatter_kernel(
    X,
    I_in,
    S,
    Y,
    I,
    N: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    SHIFT: tl.constexpr,
    FIRST: tl.constexpr,
    DESC: tl.constexpr,
):
    T: tl.constexpr = triton.cdiv(N, B)
    LOG_B: tl.constexpr = B.bit_length() - 1
    PREFIX: tl.constexpr = 2 * M * T * 256
    tile = tl.program_id(0) % T
    row = tl.program_id(0) // T
    lane = tl.arange(0, B)
    offset = tile * B + lane
    base = row.to(tl.int64) * N
    value = tl.load(X + base + offset, offset < N, other=0)
    digit = ((_sort_key(value, DESC, True) >> SHIFT) & 255).to(tl.int32)
    digit = tl.where(offset < N, digit, 256)
    code = tl.sort((digit << LOG_B) | lane, descending=False)
    permutation = code & (B - 1)
    sorted_digit = code >> LOG_B
    source = tile * B + permutation
    sorted_value = tl.load(X + base + source, source < N, other=0)
    if FIRST:
        index = source
    else:
        index = tl.load(I_in + base + source, source < N, other=0)
    prefix = tl.load(
        S + PREFIX + (row.to(tl.int64) * T + tile) * 256 + sorted_digit,
        sorted_digit < 256,
        other=0,
    )
    destination = base + prefix + lane
    tl.store(Y + destination, sorted_value, sorted_digit < 256)
    tl.store(I + destination, index, sorted_digit < 256)


def _sort_radix32(x, descending=False):
    n = x.shape[-1]
    m = x.numel() // n
    block = 512
    prefix_bins = 16
    tiles = triton.cdiv(n, block)
    histogram = torch.empty(
        (3 * m * tiles * 256,), device=x.device, dtype=torch.int32
    )
    values_a = torch.empty_like(x)
    values_b = torch.empty_like(x)
    indices_a = torch.empty(x.shape, device=x.device, dtype=torch.int32)
    indices_b = torch.empty(x.shape, device=x.device, dtype=torch.int32)
    indices = torch.empty(x.shape, device=x.device, dtype=torch.int64)
    grid = (tiles * m,)
    prefix_grid = (256 // prefix_bins * m,)
    src_values = x
    src_indices = indices_a
    for pass_id in range(4):
        first = pass_id == 0
        final = pass_id == 3
        dst_values = values_a if pass_id % 2 == 0 else values_b
        dst_indices = (
            indices
            if final
            else (indices_a if pass_id % 2 == 0 else indices_b)
        )
        sort_radix32_histogram_kernel[grid](
            src_values,
            histogram,
            n,
            m,
            block,
            pass_id * 8,
            descending,
            num_warps=4,
        )
        sort_radix32_prefix_kernel[prefix_grid](
            histogram, n, m, block, prefix_bins, num_warps=4
        )
        sort_radix32_scatter_kernel[grid](
            src_values,
            src_indices,
            histogram,
            dst_values,
            dst_indices,
            n,
            m,
            block,
            pass_id * 8,
            first,
            descending,
            num_warps=4,
        )
        src_values, src_indices = dst_values, dst_indices
    return src_values, indices


def sort(inp, dim=-1, descending=False):
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable=True, dim=-1, descending=False):
    logger.debug("GEMS HYGON SORT")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    numel = inp.numel()
    if numel == 0:
        values = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        return values, indices
    if n == 1:
        values = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        shape = tuple(s for i, s in enumerate(inp.shape) if i != dim)
        strides = tuple(s for i, s in enumerate(inp.stride()) if i != dim)
        with torch_device_fn.device(inp.device):
            sort_singleton_kernel[triton.cdiv(numel, 256),](
                inp,
                values,
                indices,
                numel,
                shape,
                strides,
                256,
                num_warps=4,
            )
        return values, indices
    rows = inp.numel() // n
    value_bits = inp.element_size() * 8
    index_bits = max(1, (n - 1).bit_length())
    # Prefer radix for wide packed keys or a few long rows; merge dense short rows.
    use_radix32 = value_bits == 32 and n >= 128 * 1024 and rows <= 16
    use_radix = use_radix32
    if value_bits == 16 and n > 4096:
        sm_count = torch.cuda.get_device_properties(inp.device).multi_processor_count
        use_radix = (
            use_radix
            or value_bits + index_bits > 32
            or rows * 16 <= sm_count
        )
    if n > 32 and (
        inp.stride(dim) != 1
        or (use_radix and (not inp.is_contiguous() or dim != rank - 1))
    ):
        permuted = inp.movedim(dim, -1)
        contiguous = torch.empty(permuted.shape, dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            sort_copy_kernel[triton.cdiv(rows, 32) * triton.cdiv(n, 32),](
                permuted,
                contiguous,
                n,
                rows,
                tuple(permuted.shape[:-1]),
                tuple(permuted.stride()[:-1]),
                permuted.stride(-1),
                32,
            )
        values, indices = sort_stable(contiguous, stable=stable, descending=descending)
        return values.movedim(-1, dim), indices.movedim(-1, dim)
    if use_radix:
        with torch_device_fn.device(inp.device):
            if use_radix32:
                return _sort_radix32(inp, descending=descending)
            return _sort_radix(inp, descending=descending)
    values = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    packed = value_bits < 64
    # Match CUDA's comparison sort for short unstable rows and radix keys otherwise.
    nan_last = not stable and n <= 32
    if packed and inp.stride(dim) == 1 and inp.is_contiguous() and n <= 2048:
        block = triton.next_power_of_2(n)
        narrow = inp.element_size() <= 2
        warps = 1 if block <= 1024 else (2 if narrow else 8)
        row_block = max(1, 128 // block)
        grid = (triton.cdiv(rows, row_block),)
        with torch_device_fn.device(inp.device):
            sort_rows_packed_kernel[grid](
                inp,
                values,
                indices,
                N=n,
                ROWS=rows,
                BLOCK=block,
                LOG_BLOCK=block.bit_length() - 1,
                ROW_BLOCK=row_block,
                DESC=descending,
                INDEX_BITS=index_bits,
                NAN_LAST=nan_last,
                MINMAX=block >= 2048,
                num_warps=warps,
            )
        return values, indices
    shape = tuple(s for i, s in enumerate(inp.shape) if i != dim)
    strides = tuple(s for i, s in enumerate(inp.stride()) if i != dim)
    out_strides = tuple(s for i, s in enumerate(values.stride()) if i != dim)
    axis = inp.stride(dim)
    out_axis = values.stride(dim)
    with torch_device_fn.device(inp.device):
        if n <= 2048 or (n <= 4096 and not packed):
            block = triton.next_power_of_2(n)
            narrow = inp.element_size() <= 2
            warps = 1 if block <= 1024 else (2 if narrow else 8)
            grid = lambda meta: (triton.cdiv(rows, meta["ROW_BLOCK"]),)
            sort_rows_kernel[grid](
                inp,
                values,
                indices,
                N=n,
                ROWS=rows,
                SHAPE=shape,
                STRIDES=strides,
                AXIS=axis,
                OUT_STRIDES=out_strides,
                OUT_AXIS=out_axis,
                BLOCK=block,
                DESC=descending,
                PACKED=packed,
                INDEX_BITS=index_bits,
                NAN_LAST=nan_last,
                FINAL=True,
                num_warps=warps,
            )
        elif packed:
            key_dtype = torch.int32 if value_bits + index_bits <= 32 else torch.int64
            block = 512
            sm_count = torch.cuda.get_device_properties(
                inp.device
            ).multi_processor_count
            # Dense grids keep each initial run within one warp.
            dense = rows * triton.cdiv(n, block) >= 8 * sm_count
            dense_wide = dense and key_dtype == torch.int64
            block = 1024 if dense and key_dtype == torch.int32 else 512
            warps = 1 if dense else 4
            # Use smaller tiles when the merge grid cannot supply enough CTAs per SM.
            if not dense:
                merge_block = 256
            else:
                merge_block = 1024 if key_dtype == torch.int32 else 512
            # Hygon prefers a wider initial run for few long 64-bit-key rows.
            # This removes one merge pass and keeps enough CTAs resident.
            long_wide = key_dtype == torch.int64 and n > 4096
            if long_wide:
                block = min(triton.next_power_of_2(n), 2048)
                warps = 8
                merge_block = 2048
            if long_wide:
                scratch = torch.empty((rows, n), dtype=key_dtype, device=inp.device)
                merges = 0
                run = block
                while run < n:
                    merges += 1
                    run *= 2
                # Keep the output index buffer idle until the final pass, then
                # use it as the second ping-pong key buffer.
                if merges % 2:
                    keys = scratch
                    next_keys = indices
                else:
                    keys = indices
                    next_keys = scratch
            else:
                keys = torch.empty((rows, n), dtype=key_dtype, device=inp.device)
                next_keys = torch.empty_like(keys)
            if long_wide:
                _argsort_tiles[rows * triton.cdiv(n, block),](
                    inp,
                    keys,
                    indices,
                    n,
                    rows,
                    shape,
                    strides,
                    axis,
                    out_strides,
                    out_axis,
                    block,
                    block.bit_length() - 1,
                    1,
                    descending,
                    False,
                    True,
                    index_bits,
                    warps,
                    num_warps=warps,
                )
            else:
                sort_rows_kernel[rows * triton.cdiv(n, block),](
                    inp,
                    keys,
                    indices,
                    N=n,
                    ROWS=rows,
                    SHAPE=shape,
                    STRIDES=strides,
                    AXIS=axis,
                    OUT_STRIDES=out_strides,
                    OUT_AXIS=out_axis,
                    BLOCK=block,
                    DESC=descending,
                    PACKED=True,
                    INDEX_BITS=index_bits,
                    NAN_LAST=False,
                    FINAL=False,
                    num_warps=warps,
                )
            run = block
            while run < n:
                final = run * 2 >= n
                merge_tile = (
                    merge_block
                    if long_wide
                    else (1024 if dense_wide and not final else merge_block)
                )
                # With few long rows, merge the first whole run pair per CTA
                # and skip the partition search.
                if long_wide and rows <= 16 and run == block:
                    merge_tile = 2 * run
                sort_merge_kernel[rows * triton.cdiv(n, merge_tile),](
                    inp,
                    keys,
                    next_keys,
                    values,
                    indices,
                    N=n,
                    RUN=run,
                    BLOCK=merge_tile,
                    SHAPE=shape,
                    STRIDES=strides,
                    AXIS=axis,
                    OUT_STRIDES=out_strides,
                    OUT_AXIS=out_axis,
                    INDEX_BITS=index_bits,
                    FINAL=final,
                    DESC=descending,
                    num_warps=1,
                )
                keys, next_keys = next_keys, keys
                run *= 2
        else:
            if inp.dtype == torch.float64:
                keys = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
                sort_float64_keys_kernel[triton.cdiv(inp.numel(), 256),](
                    inp,
                    keys,
                    n,
                    inp.numel(),
                    shape,
                    strides,
                    axis,
                    out_strides,
                    out_axis,
                    descending,
                    256,
                )
                indices = _argsort_merge_entry(keys, dim, False)
            else:
                indices = _argsort_merge_entry(inp, dim, descending)
            sort_gather_kernel[triton.cdiv(inp.numel(), 256),](
                inp,
                indices,
                values,
                n,
                inp.numel(),
                shape,
                strides,
                axis,
                out_strides,
                out_axis,
                256,
            )
    return values, indices
