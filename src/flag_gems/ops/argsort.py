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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _byte_argsort_small(inp, out, N: tl.constexpr, B: tl.constexpr, DESC: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    v = tl.load(inp + row * N + i, i < N, other=0).to(tl.int32)
    before = v[None, :] > v[:, None] if DESC else v[None, :] < v[:, None]
    before = before | (v[None, :] == v[:, None]) & (i[None, :] < i[:, None])
    rank = tl.sum((before & (i[None, :] < N)).to(tl.int32), 1)
    tl.store(out + row * N + rank, i, i < N)


@libentry()
@triton.jit
def _byte_argsort_count(
    inp,
    counts,
    N: tl.constexpr,
    T: tl.constexpr,
    LOW: tl.constexpr,
    DESC: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) // T
    tile = tl.program_id(0) % T
    bucket = tl.program_id(1)
    value = LOW + (255 - bucket if DESC else bucket)
    i = tile * B + tl.arange(0, B)
    v = tl.load(inp + row * N + i, i < N, other=0).to(tl.int32)
    count = tl.sum(((i < N) & (v == value)).to(tl.int32), 0)
    tl.store(counts + (row * 256 + bucket) * T + tile, count)


@libentry()
@triton.jit
def _byte_argsort_prefix(counts, offsets, T: tl.constexpr, B: tl.constexpr):
    bucket = tl.program_id(0)
    i = tl.arange(0, B)
    count = tl.load(counts + bucket * T + i, i < T, other=0)
    prefix = count if T == 1 else tl.cumsum(count)
    tl.store(offsets + bucket * T + i, prefix - count, i < T)


@libentry()
@triton.jit
def _byte_argsort_bucket_prefix(counts, totals, T: tl.constexpr):
    row = tl.program_id(0)
    bucket = tl.arange(0, 256)
    total = tl.full((256,), 0, tl.int32)
    for tile in range(T):
        total += tl.load(counts + (row * 256 + bucket) * T + tile)
    tl.store(totals + row * 256 + bucket, tl.cumsum(total) - total)


@libentry()
@triton.jit
def _byte_argsort_scatter(
    inp,
    out,
    offsets,
    totals,
    N: tl.constexpr,
    T: tl.constexpr,
    LOW: tl.constexpr,
    DESC: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) // T
    tile = tl.program_id(0) % T
    bucket = tl.program_id(1)
    value = LOW + (255 - bucket if DESC else bucket)
    i = tile * B + tl.arange(0, B)
    v = tl.load(inp + row * N + i, i < N, other=0).to(tl.int32)
    match = (i < N) & (v == value)
    base = tl.load(totals + row * 256 + bucket)
    base += tl.load(offsets + (row * 256 + bucket) * T + tile)
    rank = tl.cumsum(match.to(tl.int32)) - 1
    tl.store(out + row * N + base + rank, i, match)


def _byte_argsort(inp, dim, descending):
    n = inp.shape[dim]
    dim %= inp.ndim
    if n == 1:
        return torch.zeros_like(inp, dtype=torch.int64)
    x = inp.movedim(dim, -1).contiguous()
    out = torch.empty_like(x, dtype=torch.int64)
    if x.numel() == 0:
        return out.movedim(-1, dim)
    rows = x.numel() // n
    with torch_device_fn.device(inp.device):
        if n <= 128:
            _byte_argsort_small[rows,](x, out, n, triton.next_power_of_2(n), descending)
        else:
            block = 512
            tiles = triton.cdiv(n, block)
            counts = torch.empty((rows, 256, tiles), device=x.device, dtype=torch.int32)
            offsets = torch.empty_like(counts)
            totals = torch.empty((rows, 256), device=x.device, dtype=torch.int32)
            low = torch.iinfo(x.dtype).min
            _byte_argsort_count[rows * tiles, 256](
                x, counts, n, tiles, low, descending, block
            )
            _byte_argsort_prefix[rows * 256,](
                counts, offsets, tiles, triton.next_power_of_2(tiles)
            )
            _byte_argsort_bucket_prefix[rows,](counts, totals, tiles)
            _byte_argsort_scatter[rows * tiles, 256](
                x, out, offsets, totals, n, tiles, low, descending, block
            )
    return out.movedim(-1, dim)


@triton.jit
def _argsort_before(a, ai, b, bi, DESC: tl.constexpr):
    before = a > b if DESC else a < b
    equal = a == b
    if a.dtype.is_floating():
        an = a != a
        bn = b != b
        before = before | an & ~bn if DESC else before | ~an & bn
        equal = equal | an & bn
    return before | equal & (ai < bi)


@triton.jit
def _argsort_row_offset(row, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full(row.shape, 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = row - row // SHAPE[axis] * SHAPE[axis]
        offset += coord.to(tl.int64) * STRIDES[axis]
        row = row // SHAPE[axis]
    return offset


@triton.jit
def _argsort_partner(x, STEP: tl.constexpr):
    bits: tl.constexpr = x.dtype.primitive_bitwidth
    if bits == 64:
        it: tl.constexpr = tl.uint64
    elif bits == 32:
        it: tl.constexpr = tl.uint32
    elif bits == 16:
        it: tl.constexpr = tl.uint16
    else:
        it: tl.constexpr = tl.uint8
    shape: tl.constexpr = (x.numel // (2 << STEP), 2, 1 << STEP)
    ix = tl.reshape(x.to(it, bitcast=True), shape)
    partner = (ix ^ tl.xor_sum(ix, 1, True)).to(it)
    return tl.reshape(partner, x.shape).to(x.dtype, bitcast=True)


@triton.jit
def _argsort_pack_key(values, indices, INDEX_BITS: tl.constexpr, DESC: tl.constexpr):
    VALUE_BITS: tl.constexpr = values.dtype.primitive_bitwidth
    tl.static_assert(VALUE_BITS <= 32)
    value_mask = tl.full((), (1 << VALUE_BITS) - 1, tl.uint32)
    sign_bit = tl.full((), 1 << VALUE_BITS - 1, tl.uint32)
    if values.dtype.is_floating():
        if VALUE_BITS == 16:
            bits = values.to(tl.uint16, bitcast=True).to(tl.uint32)
        else:
            bits = values.to(tl.uint32, bitcast=True)
        bits = tl.where(values == 0, tl.full((), 0, tl.uint32), bits)
        flip = tl.where(bits & sign_bit != 0, value_mask, sign_bit)
        key = bits ^ flip
        key = tl.where(values != values, value_mask, key)
    else:
        key = values.to(tl.uint32) & value_mask
        if values.dtype.is_int_signed():
            key = key ^ sign_bit
    if DESC:
        key = key ^ value_mask
    if VALUE_BITS + INDEX_BITS <= 32:
        packed = key << INDEX_BITS | indices.to(tl.uint32)
    else:
        tl.static_assert(VALUE_BITS + INDEX_BITS < 64)
        packed = key.to(tl.uint64) << INDEX_BITS | indices.to(tl.uint64)
    return packed


@libentry()
@triton.jit
def _argsort_tiles(
    inp,
    values_out,
    indices_out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    FINAL: tl.constexpr,
    PACKED: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    WARPS: tl.constexpr,
):
    chunks: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    chunk = task - task // chunks * chunks
    row = task // chunks * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    lane = tl.arange(0, BLOCK)
    col = chunk * BLOCK + lane
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(
        inp + base[:, None] + col[None, :] * AXIS_STRIDE,
        (row[:, None] < ROWS) & (col[None, :] < N),
        other=0,
    )
    indices = tl.broadcast_to(col[None, :], (ROW_BLOCK, BLOCK))
    if PACKED:
        packed = _argsort_pack_key(values, indices, INDEX_BITS, DESC)
        padding = ~tl.full((), 0, packed.dtype)
        packed = tl.where((row[:, None] < ROWS) & (col[None, :] < N), packed, padding)
        if BLOCK > 1:
            packed = tl.sort(packed, dim=1, descending=False)
        indices = (packed & (1 << INDEX_BITS) - 1).to(tl.int32)
    else:
        for stage in tl.static_range(1, LOG_BLOCK + 1):
            for step in tl.static_range(stage - 1, -1, -1):
                other_values = _argsort_partner(values, step)
                other_indices = _argsort_partner(indices, step)
                before = _argsort_before(
                    other_values, other_indices, values, indices, DESC
                )
                if N % BLOCK != 0:
                    valid = indices < N
                    other_valid = other_indices < N
                    before = other_valid & (before | ~valid)
                forward = lane & 1 << stage == 0
                lower = lane & 1 << step == 0
                swap = tl.where((forward == lower)[None, :], before, ~before)
                values = tl.where(swap, other_values, values)
                indices = tl.where(swap, other_indices, indices)
    mask = (row[:, None] < ROWS) & (col[None, :] < N)
    if FINAL:
        out_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
        offset = out_base[:, None] + col[None, :] * OUT_AXIS_STRIDE
    else:
        offset = row[:, None].to(tl.int64) * N + col[None, :]
        tl.store(values_out + offset, packed if PACKED else values, mask)
    if FINAL or not PACKED:
        tl.store(indices_out + offset, indices, mask)


@triton.jit
def _argsort_partition(
    values,
    base,
    a_start,
    b_start,
    a_len,
    b_len,
    diagonal,
    STEPS: tl.constexpr,
    DESC: tl.constexpr,
):
    low = tl.maximum(0, diagonal - b_len)
    high = tl.minimum(diagonal, a_len)
    for _ in range(STEPS):
        mid = (low + high) // 2
        j = diagonal - mid
        active = low < high
        av = tl.load(values + base + a_start + mid, active & (mid < a_len), other=0)
        bv = tl.load(values + base + b_start + j - 1, active & (j > 0), other=0)
        take_a = (
            (j > 0) & (mid < a_len) & _argsort_before(av, a_start, bv, b_start, DESC)
        )
        low = tl.where(active & take_a, mid + 1, low)
        high = tl.where(active & ~take_a, mid, high)
    return low


@libentry()
@triton.jit
def _argsort_merge(
    values_in,
    indices_in,
    values_out,
    indices_out,
    N: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    DESC: tl.constexpr,
    FINAL: tl.constexpr,
    WARPS: tl.constexpr,
):
    blocks: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    row = task // blocks
    start = (task - row * blocks) * BLOCK
    pair_start = start // (2 * RUN) * (2 * RUN)
    a_len = tl.minimum(RUN, N - pair_start)
    b_start = pair_start + RUN
    b_len = tl.maximum(0, tl.minimum(RUN, N - b_start))
    diagonal = start - pair_start
    end_diagonal = tl.minimum(diagonal + BLOCK, a_len + b_len)
    base = row.to(tl.int64) * N
    a0 = _argsort_partition(
        values_in, base, pair_start, b_start, a_len, b_len, diagonal, SEARCH_STEPS, DESC
    )
    a1 = _argsort_partition(
        values_in,
        base,
        pair_start,
        b_start,
        a_len,
        b_len,
        end_diagonal,
        SEARCH_STEPS,
        DESC,
    )
    b0 = diagonal - a0
    b1 = end_diagonal - a1
    na = a1 - a0
    nb = b1 - b0
    lane = tl.arange(0, BLOCK)
    is_a = lane < na
    is_b = lane >= BLOCK - nb
    source = tl.where(
        is_a, pair_start + a0 + lane, b_start + b1 - 1 - (lane - (BLOCK - nb))
    )
    values = tl.load(values_in + base + source, is_a | is_b, other=0)
    indices = tl.load(indices_in + base + source, is_a | is_b, other=N)
    for step in tl.static_range(LOG_BLOCK - 1, -1, -1):
        ov = _argsort_partner(values, step)
        oi = _argsort_partner(indices, step)
        before = _argsort_before(ov, oi, values, indices, DESC)
        if N % BLOCK != 0:
            valid = indices < N
            other_valid = oi < N
            before = other_valid & (before | ~valid)
        lower = lane & 1 << step == 0
        swap = tl.where(lower, before, ~before)
        values = tl.where(swap, ov, values)
        indices = tl.where(swap, oi, indices)
    col = start + lane
    if FINAL:
        out_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
        offset = out_base + col.to(tl.int64) * OUT_AXIS_STRIDE
    else:
        offset = base + col
        tl.store(values_out + offset, values, col < N)
    tl.store(indices_out + offset, indices, col < N)


@triton.jit
def _packed_load(ptr, offset, mask):
    if ptr.dtype.element_ty.primitive_bitwidth == 32:
        value = tl.load(ptr + offset, mask, other=-1).to(tl.uint32, bitcast=True)
    else:
        value = tl.load(ptr + offset, mask, other=-1).to(tl.uint64, bitcast=True)
    return value


@triton.jit
def _packed_partition(
    keys, base, a_start, b_start, a_len, b_len, diagonal, STEPS: tl.constexpr
):
    low = tl.maximum(0, diagonal - b_len)
    high = tl.minimum(diagonal, a_len)
    for _ in range(STEPS):
        mid = (low + high) // 2
        j = diagonal - mid
        active = low < high
        a = _packed_load(keys, base + a_start + mid, active & (mid < a_len))
        b = _packed_load(keys, base + b_start + j - 1, active & (j > 0))
        take = (j > 0) & (mid < a_len) & (a < b)
        low = tl.where(active & take, mid + 1, low)
        high = tl.where(active & ~take, mid, high)
    return low


@libentry()
@triton.jit
def _packed_merge(
    keys,
    next_keys,
    out,
    N: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    FINAL: tl.constexpr,
    WARPS: tl.constexpr,
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
    for step in tl.static_range(LOG_BLOCK - 1, -1, -1):
        other = _argsort_partner(key, step)
        lower = lane & 1 << step == 0
        key = tl.where(lower, tl.minimum(key, other), tl.maximum(key, other))
    col = start + lane
    if FINAL:
        index = (key & (1 << INDEX_BITS) - 1).to(tl.int32)
        offset = _argsort_row_offset(row, SHAPE, OUT_STRIDES) + col.to(tl.int64) * AXIS
        tl.store(out + offset, index, col < N)
    else:
        tl.store(next_keys + base + col, key, col < N)


def _argsort_merge_entry(inp, dim=-1, descending=False):
    """Stable indices using bounded tiles and launch-separated merge passes."""
    logger.debug("GEMS ARGSORT")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    if inp.numel() == 0:
        return out
    dim = dim % max(rank, 1)
    n = inp.shape[dim] if rank else 1
    if n >= 2**31:
        raise NotImplementedError("argsort supports fewer than 2**31 elements per row")
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    block = min(triton.next_power_of_2(n), 256)
    row_block = max(1, 128 // block)
    index_bits = (n - 1).bit_length()
    use_packed = device.vendor_name in ("nvidia",) and inp.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
    )
    if use_packed:
        block = min(triton.next_power_of_2(n), 1024)
    else:
        block = min(triton.next_power_of_2(n), 1024)
    row_block = max(1, 128 // block)
    merge_block = min(block, 1024)
    tile_warps = (
        1 if device.vendor_name == "nvidia" and use_packed and (block <= 1024) else 4
    )
    merge_warps = 4
    packed_warps = 1 if device.vendor_name == "nvidia" else 4
    pair_merge_kernel = _argsort_merge
    if inp.dtype == torch.int64 and device.vendor_name == "nvidia":
        tile_warps = 1
        merge_warps = INT64_MERGE_WARPS
    with torch_device_fn.device(inp.device):
        if n <= block:
            _argsort_tiles[triton.cdiv(rows, row_block),](
                inp,
                out,
                out,
                n,
                rows,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                block.bit_length() - 1,
                row_block,
                descending,
                True,
                use_packed,
                index_bits,
                tile_warps,
                num_warps=tile_warps,
            )
        elif use_packed:
            bits = inp.element_size() * 8 + index_bits
            key_dtype = torch.int32 if bits <= 32 else torch.int64
            keys = torch.empty((rows, n), dtype=key_dtype, device=inp.device)
            next_keys = torch.empty_like(keys)
            _argsort_tiles[rows * triton.cdiv(n, block),](
                inp,
                keys,
                out,
                n,
                rows,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                block.bit_length() - 1,
                1,
                descending,
                False,
                True,
                index_bits,
                tile_warps,
                num_warps=tile_warps,
            )
            packed_merge_kernel = _packed_merge
            run = block
            while run < n:
                final = run * 2 >= n
                packed_merge_kernel[rows * triton.cdiv(n, merge_block),](
                    keys,
                    next_keys,
                    out,
                    n,
                    run,
                    merge_block,
                    (run + 1).bit_length(),
                    merge_block.bit_length() - 1,
                    shape,
                    out_strides,
                    out_axis_stride,
                    index_bits,
                    final,
                    packed_warps,
                    num_warps=packed_warps,
                )
                keys, next_keys = (next_keys, keys)
                run *= 2
        else:
            values = torch.empty((rows, n), dtype=inp.dtype, device=inp.device)
            indices = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
            values_tmp = torch.empty_like(values)
            indices_tmp = torch.empty_like(indices)
            _argsort_tiles[rows * triton.cdiv(n, block),](
                inp,
                values,
                indices,
                n,
                rows,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                block.bit_length() - 1,
                1,
                descending,
                False,
                use_packed,
                index_bits,
                tile_warps,
                num_warps=tile_warps,
            )
            run = block
            while run < n:
                final = run * 2 >= n
                pair_merge_kernel[rows * triton.cdiv(n, merge_block),](
                    values,
                    indices,
                    values_tmp,
                    out if final else indices_tmp,
                    n,
                    run,
                    merge_block,
                    (run + 1).bit_length(),
                    merge_block.bit_length() - 1,
                    shape,
                    out_strides,
                    out_axis_stride,
                    descending,
                    final,
                    merge_warps,
                    num_warps=merge_warps,
                )
                values, values_tmp = (values_tmp, values)
                indices, indices_tmp = (indices_tmp, indices)
                run *= 2
    return out


@triton.jit
def _radix_order_key(values, DESC: tl.constexpr):
    VALUE_BITS: tl.constexpr = values.dtype.primitive_bitwidth
    if VALUE_BITS == 64:
        value_mask = ~tl.full((), 0, tl.uint64)
        sign_bit = tl.full((), 1, tl.uint64) << 63
        if values.dtype.is_floating():
            bits = values.to(tl.uint64, bitcast=True)
        else:
            bits = values.to(tl.uint64)
    else:
        value_mask = tl.full((), (1 << VALUE_BITS) - 1, tl.uint32)
        sign_bit = tl.full((), 1 << VALUE_BITS - 1, tl.uint32)
        if values.dtype.is_floating():
            if VALUE_BITS == 16:
                bits = values.to(tl.uint16, bitcast=True).to(tl.uint32)
            else:
                bits = values.to(tl.uint32, bitcast=True)
        else:
            bits = values.to(tl.uint32) & value_mask
    if values.dtype.is_floating():
        bits = tl.where(values == 0, tl.full((), 0, bits.dtype), bits)
        key = bits ^ tl.where(bits & sign_bit != 0, value_mask, sign_bit)
        key = tl.where(values != values, value_mask, key)
    elif values.dtype.is_int_signed():
        key = bits ^ sign_bit
    else:
        key = bits
    if DESC:
        key = key ^ value_mask
    return key


@libentry()
@triton.jit
def _radix_local_sort(
    inp,
    keys_current,
    indices_current,
    keys_local,
    indices_local,
    digits_local,
    counts,
    N: tl.constexpr,
    TILES: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    KEY_BITS: tl.constexpr,
    SHIFT: tl.constexpr,
    DESC: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
    BYTE_ONLY: tl.constexpr,
    WARPS: tl.constexpr,
):
    if BYTE_ONLY:
        tl.static_assert(FIRST and LAST)
    task = tl.program_id(0)
    row = task // TILES
    tile = task - row * TILES
    lane = tl.arange(0, BLOCK)
    col = tile * BLOCK + lane
    valid = col < N
    scratch_base = row.to(tl.int64) * N
    if FIRST:
        input_base = _argsort_row_offset(row, SHAPE, STRIDES)
        values = tl.load(
            inp + input_base + col.to(tl.int64) * AXIS_STRIDE, valid, other=0
        )
        keys = _radix_order_key(values, DESC)
    else:
        keys = tl.load(keys_current + scratch_base + col, valid, other=0)
        if KEY_BITS == 64:
            keys = keys.to(tl.uint64)
        else:
            keys = keys.to(tl.uint32)
    digit = (keys >> SHIFT & 255).to(tl.int32)
    histogram = tl.histogram(tl.where(valid, digit, 0), 256)
    bucket = tl.arange(0, 256)
    padding_count = BLOCK - tl.minimum(BLOCK, N - tile * BLOCK)
    histogram -= tl.where(bucket == 0, padding_count, 0)
    count_base = row.to(tl.int64) * 256 * TILES
    tl.store(counts + count_base + bucket * TILES + tile, histogram)
    code = digit.to(tl.uint32) << LOG_BLOCK | lane.to(tl.uint32)
    code = tl.where(valid, code, ~tl.full((), 0, tl.uint32))
    code = tl.sort(code, dim=0, descending=False)
    if BYTE_ONLY:
        tl.store(indices_local + scratch_base + col, code, valid)
    else:
        sorted_lane = (code & BLOCK - 1).to(tl.int32)
        source_col = tile * BLOCK + sorted_lane
        if FIRST:
            original_index = source_col
        else:
            original_index = tl.load(
                indices_current + scratch_base + source_col, valid, other=0
            )
        tl.store(indices_local + scratch_base + col, original_index, valid)
        tl.store(digits_local + scratch_base + col, code >> LOG_BLOCK, valid)
        if not LAST:
            if FIRST:
                sorted_values = tl.load(
                    inp + input_base + source_col.to(tl.int64) * AXIS_STRIDE,
                    valid,
                    other=0,
                )
                sorted_keys = _radix_order_key(sorted_values, DESC)
            else:
                sorted_keys = tl.load(
                    keys_current + scratch_base + source_col, valid, other=0
                )
            tl.store(keys_local + scratch_base + col, sorted_keys, valid)


@libentry()
@triton.jit
def _radix_prefix(
    counts,
    offsets,
    TILES: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    BUCKET_BLOCK: tl.constexpr,
    WARPS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, TILE_BLOCK)
    bucket_lane = tl.arange(0, BUCKET_BLOCK)
    base = row.to(tl.int64) * 256 * TILES
    previous_buckets = tl.full((), 0, tl.int32)
    previous_local_buckets = tl.full((TILE_BLOCK,), 0, tl.int32)
    for group in range(256 // BUCKET_BLOCK):
        bucket = group * BUCKET_BLOCK + bucket_lane
        addresses = base + bucket[:, None] * TILES + tile[None, :]
        h = tl.load(counts + addresses, tile[None, :] < TILES, other=0)
        tile_prefix = tl.cumsum(h, axis=1) - h
        totals = tl.sum(h, axis=1)
        bucket_prefix = tl.cumsum(totals, axis=0) - totals + previous_buckets
        local_prefix = tl.cumsum(h, axis=0) - h + previous_local_buckets[None, :]
        adjusted = bucket_prefix[:, None] + tile_prefix - local_prefix
        tl.store(offsets + addresses, adjusted, tile[None, :] < TILES)
        previous_buckets += tl.sum(totals, axis=0)
        previous_local_buckets += tl.sum(h, axis=0)


@libentry()
@triton.jit
def _radix_tile_prefix(
    counts, offsets, totals, TILES: tl.constexpr, TILE_BLOCK: tl.constexpr
):
    task = tl.program_id(0)
    lane = tl.arange(0, TILE_BLOCK)
    h = tl.load(counts + task.to(tl.int64) * TILES + lane, lane < TILES, other=0)
    prefix = tl.cumsum(h, axis=0) - h
    tl.store(offsets + task.to(tl.int64) * TILES + lane, prefix, lane < TILES)
    tl.store(totals + task, tl.sum(h, axis=0))


@libentry()
@triton.jit
def _radix_bucket_prefix(
    counts, offsets, totals, TILES: tl.constexpr, TILE_GROUP: tl.constexpr
):
    row = tl.program_id(0)
    tile = tl.program_id(1) * TILE_GROUP + tl.arange(0, TILE_GROUP)
    bucket = tl.arange(0, 256)
    bucket_total = tl.load(totals + row.to(tl.int64) * 256 + bucket)
    base_prefix = tl.cumsum(bucket_total, axis=0) - bucket_total
    addr = (row.to(tl.int64) * 256 + bucket[:, None]) * TILES + tile[None, :]
    h = tl.load(counts + addr, tile[None, :] < TILES, other=0)
    local_prefix = tl.cumsum(h, axis=0) - h
    tile_prefix = tl.load(offsets + addr, tile[None, :] < TILES, other=0)
    tl.store(
        offsets + addr,
        base_prefix[:, None] + tile_prefix - local_prefix,
        tile[None, :] < TILES,
    )


@libentry()
@triton.jit
def _radix_scatter(
    keys_local,
    indices_local,
    digits_local,
    offsets,
    keys_current,
    indices_current,
    out,
    N: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    LAST: tl.constexpr,
    BYTE_ONLY: tl.constexpr,
    WARPS: tl.constexpr,
):
    task = tl.program_id(0)
    row = task // TILES
    tile = task - row * TILES
    lane = tl.arange(0, BLOCK)
    col = tile * BLOCK + lane
    valid = col < N
    base = row.to(tl.int64) * N
    if BYTE_ONLY:
        tl.static_assert(LAST)
        code = tl.load(indices_local + base + col, valid, other=0).to(tl.uint32)
        digit = (code // BLOCK).to(tl.int32)
        original_index = tile * BLOCK + (code & BLOCK - 1).to(tl.int32)
    else:
        digit = tl.load(digits_local + base + col, valid, other=0).to(tl.int32)
        original_index = tl.load(indices_local + base + col, valid, other=0)
    count_base = row.to(tl.int64) * 256 * TILES
    delta = tl.load(offsets + count_base + digit * TILES + tile, valid, other=0)
    destination = delta + lane
    if LAST:
        output_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
        tl.store(
            out + output_base + destination.to(tl.int64) * OUT_AXIS_STRIDE,
            original_index,
            valid,
        )
    else:
        keys = tl.load(keys_local + base + col, valid, other=0)
        tl.store(keys_current + base + destination, keys, valid)
        tl.store(indices_current + base + destination, original_index, valid)


@libentry()
@triton.jit
def _radix_counts_only(
    inp,
    keys_current,
    counts,
    N: tl.constexpr,
    TILES: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
    DESC: tl.constexpr,
    FIRST: tl.constexpr,
    WARPS: tl.constexpr,
):
    task = tl.program_id(0)
    row = task // TILES
    tile = task - row * TILES
    col = tile * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    if FIRST:
        input_base = _argsort_row_offset(row, SHAPE, STRIDES)
        values = tl.load(
            inp + input_base + col.to(tl.int64) * AXIS_STRIDE, valid, other=0
        )
        keys = _radix_order_key(values, DESC)
    else:
        scratch_base = row.to(tl.int64) * N
        keys = tl.load(keys_current + scratch_base + col, valid, other=0).to(tl.uint32)
    digit = (keys >> SHIFT & 255).to(tl.int32)
    histogram = tl.histogram(tl.where(valid, digit, 0), 256)
    bucket = tl.arange(0, 256)
    padding_count = BLOCK - tl.minimum(BLOCK, N - tile * BLOCK)
    histogram -= tl.where(bucket == 0, padding_count, 0)
    count_base = row.to(tl.int64) * 256 * TILES
    tl.store(counts + count_base + bucket * TILES + tile, histogram)


@libentry()
@triton.jit
def _radix_fused_local_scatter(
    inp,
    keys_current,
    indices_current,
    offsets,
    out,
    N: tl.constexpr,
    TILES: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
    DESC: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(FIRST or LAST)
    tl.static_assert(BLOCK <= 1024)
    task = tl.program_id(0)
    row = task // TILES
    tile = task - row * TILES
    lane = tl.arange(0, BLOCK)
    col = tile * BLOCK + lane
    valid = col < N
    scratch_base = row.to(tl.int64) * N
    if FIRST:
        input_base = _argsort_row_offset(row, SHAPE, STRIDES)
        values = tl.load(
            inp + input_base + col.to(tl.int64) * AXIS_STRIDE, valid, other=0
        )
        keys = _radix_order_key(values, DESC)
    else:
        keys = tl.load(keys_current + scratch_base + col, valid, other=0).to(tl.uint32)
    digit = (keys >> SHIFT & 255).to(tl.uint32)
    code = digit << LOG_BLOCK | lane.to(tl.uint32)
    code = tl.where(valid, code, ~tl.full((), 0, tl.uint32))
    code = tl.sort(code, dim=0, descending=False)
    sorted_lane = (code & BLOCK - 1).to(tl.int32)
    source_col = tile * BLOCK + sorted_lane
    sorted_digit = (code >> LOG_BLOCK).to(tl.int32)
    if FIRST:
        original_index = source_col
    else:
        original_index = tl.load(
            indices_current + scratch_base + source_col, valid, other=0
        )
    count_base = row.to(tl.int64) * 256 * TILES
    delta = tl.load(offsets + count_base + sorted_digit * TILES + tile, valid, other=0)
    destination = delta + lane
    if LAST:
        output_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
        tl.store(
            out + output_base + destination.to(tl.int64) * OUT_AXIS_STRIDE,
            original_index,
            valid,
        )
    else:
        tl.static_assert(FIRST)
        sorted_values = tl.load(
            inp + input_base + source_col.to(tl.int64) * AXIS_STRIDE, valid, other=0
        )
        sorted_keys = _radix_order_key(sorted_values, DESC)
        tl.store(keys_current + scratch_base + destination, sorted_keys, valid)
        tl.store(indices_current + scratch_base + destination, original_index, valid)


def _argsort_radix(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    block = 1024
    log_block = 10
    tiles = triton.cdiv(n, block)
    tile_block = triton.next_power_of_2(tiles)
    bucket_block = min(256, 4096 // tile_block)
    value_bits = inp.element_size() * 8
    key_bits = 64 if value_bits == 64 else 32
    key_dtype = torch.int64 if key_bits == 64 else torch.int32
    passes = value_bits // 8
    byte_only = passes == 1 and inp.dtype in (torch.int8, torch.uint8)
    scratch_shape = (rows, n)
    indices_local = torch.empty(scratch_shape, dtype=torch.int32, device=inp.device)
    if byte_only:
        digits_local = indices_local
    else:
        digits_local = torch.empty(scratch_shape, dtype=torch.uint8, device=inp.device)
    counts = torch.empty((rows, 256, tiles), dtype=torch.int32, device=inp.device)
    offsets = torch.empty((rows, 256, tiles), dtype=torch.int32, device=inp.device)
    split_prefix = device.vendor_name == "nvidia" and n >= 131072 and (value_bits == 16)
    bucket_totals = (
        torch.empty((rows, 256), dtype=torch.int32, device=inp.device)
        if split_prefix
        else offsets
    )
    if passes > 1:
        keys_current = torch.empty(scratch_shape, dtype=key_dtype, device=inp.device)
        keys_local = torch.empty(scratch_shape, dtype=key_dtype, device=inp.device)
        indices_current = torch.empty(
            scratch_shape, dtype=torch.int32, device=inp.device
        )
    else:
        keys_current = indices_local
        keys_local = indices_local
        indices_current = indices_local
    warps = 4
    local_warps = 1 if device.vendor_name == "nvidia" else warps
    with torch_device_fn.device(inp.device):
        for pass_index in range(passes):
            last = pass_index + 1 == passes
            _radix_local_sort[rows * tiles,](
                inp,
                keys_current,
                indices_current,
                keys_local,
                indices_local,
                digits_local,
                counts,
                n,
                tiles,
                shape,
                strides,
                axis_stride,
                block,
                log_block,
                key_bits,
                pass_index * 8,
                descending,
                pass_index == 0,
                last,
                byte_only,
                local_warps,
                num_warps=local_warps,
            )
            if split_prefix:
                _radix_tile_prefix[rows * 256,](
                    counts, offsets, bucket_totals, tiles, tile_block, num_warps=4
                )
                _radix_bucket_prefix[rows, triton.cdiv(tiles, 16)](
                    counts, offsets, bucket_totals, tiles, 16, num_warps=4
                )
            else:
                _radix_prefix[rows,](
                    counts,
                    offsets,
                    tiles,
                    tile_block,
                    bucket_block,
                    warps,
                    num_warps=warps,
                )
            _radix_scatter[rows * tiles,](
                keys_local,
                indices_local,
                digits_local,
                offsets,
                keys_current,
                indices_current,
                out,
                n,
                tiles,
                block,
                shape,
                out_strides,
                out_axis_stride,
                last,
                byte_only,
                warps,
                num_warps=warps,
            )
    return out


def argsort(inp, dim=-1, descending=False):
    logger.debug("GEMS ARGSORT")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    n = inp.shape[dim] if rank else 1
    if (
        device.vendor_name in ("nvidia",)
        and inp.numel() > 0
        and (2048 <= n <= 262144)
        and (
            inp.dtype in (torch.int8, torch.uint8)
            or (
                device.vendor_name == "nvidia"
                and n >= 131072
                and (inp.dtype in (torch.float16, torch.bfloat16, torch.int16))
            )
        )
    ):
        return _argsort_radix(inp, dim, descending)
    return _argsort_merge_entry(inp, dim, descending)


INT64_MERGE_WARPS = 1
