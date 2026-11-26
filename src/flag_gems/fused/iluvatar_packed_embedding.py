# SPDX-License-Identifier: Apache-2.0
"""Losslessly encode BF16 input embeddings and gather directly from device storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _count_escapes(weights, counts, BASE: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(0)
    offsets = group * TILE + tl.arange(0, TILE)
    raw = tl.load(weights + offsets).to(tl.uint16, bitcast=True).to(tl.int32)
    exponent = (raw >> 7) & 255
    escape = (exponent < BASE) | (exponent >= BASE + 7)
    tl.store(counts + group, tl.sum(escape.to(tl.int32), 0))


@triton.jit
def _pack(weights, low, planes, ends, side, BASE: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, TILE)
    offsets = group * TILE + lane
    raw = tl.load(weights + offsets).to(tl.uint16, bitcast=True).to(tl.int32)
    exponent = (raw >> 7) & 255
    escape = (exponent < BASE) | (exponent >= BASE + 7)
    code = tl.where(escape, 7, exponent - BASE)
    tl.store(low + offsets, (raw & 127) | ((raw >> 8) & 128))

    # Eight consecutive exponents occupy three bytes, one per bit plane.
    byte_lanes = tl.arange(0, TILE // 8)
    for bit in tl.static_range(3):
        values = ((code >> bit) & 1) << (lane % 8)
        byte = tl.sum(values.reshape(TILE // 8, 8), 1)
        tl.store(planes + (group * 3 + bit) * (TILE // 8) + byte_lanes, byte)

    start = tl.load(ends + group - 1, group > 0, 0)
    rank = tl.cumsum(escape.to(tl.int32), 0) - 1
    tl.store(side + start + rank, exponent, escape)


@triton.jit
def _lookup(
    ids,
    low,
    planes,
    ends,
    side,
    out,
    BASE: tl.constexpr,
    ROWS: tl.constexpr,
    DIM: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.program_id(0)
    column_tile = tl.program_id(1)
    index = tl.load(ids + token)
    valid = (index >= 0) & (index < ROWS)
    tl.device_assert(valid, "embedding index outside vocabulary")
    group = tl.where(valid, index, 0).to(tl.int32) * (DIM // TILE) + column_tile
    lane = tl.arange(0, TILE)
    sign_mantissa = tl.load(low + group * TILE + lane).to(tl.int32)

    code = tl.full((TILE,), 0, tl.int32)
    for bit in tl.static_range(3):
        byte = tl.load(planes + (group * 3 + bit) * (TILE // 8) + lane // 8).to(
            tl.int32
        )
        code = code | (((byte >> (lane % 8)) & 1) << bit)
    escape = code == 7
    start = tl.load(ends + group - 1, group > 0, 0)
    rank = tl.cumsum(escape.to(tl.int32), 0) - 1
    extra = tl.load(side + start + rank, escape, 0).to(tl.int32)
    exponent = tl.where(escape, extra, code + BASE)
    bits = ((sign_mantissa & 128) << 8) | (exponent << 7) | (sign_mantissa & 127)
    value = tl.where(valid, bits, 0).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(out + token * DIM + column_tile * TILE + lane, value)


def encode(weight):
    """Pack an independent BF16 input table once, after checkpoint loading."""
    rows, dim = weight.shape
    tile = 256
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous() and dim % tile == 0
    # The sample selects the common exponent range. Every exponent outside
    # it is stored verbatim, including subnormals, infinities and NaNs.
    sample = weight[::257, ::13].contiguous().view(torch.int16).cpu().int()
    histogram = torch.bincount(((sample >> 7) & 255).flatten(), minlength=256)
    base = int(histogram.unfold(0, 7, 1).sum(1).argmax())
    groups = rows * (dim // tile)
    counts = torch.empty(groups, dtype=torch.int32, device=weight.device)
    _count_escapes[(groups,)](weight, counts, base, tile, num_warps=2)
    ends = counts.cumsum(0, dtype=torch.int32)
    side_bytes = max(1, int(ends[-1].item()))
    packed_bytes = weight.numel() + groups * 3 * (tile // 8) + ends.nbytes + side_bytes
    if packed_bytes >= weight.nbytes:
        return None
    low = torch.empty((rows, dim), dtype=torch.uint8, device=weight.device)
    planes = low.new_empty((groups, 3, tile // 8))
    side = low.new_empty(side_bytes)
    _pack[(groups,)](weight, low, planes, ends, side, base, tile, num_warps=2)
    return low, planes, ends, side, base


def lookup(indices, low, planes, ends, side, base):
    rows, dim = low.shape
    out = torch.empty((*indices.shape, dim), device=low.device, dtype=torch.bfloat16)
    if indices.numel():
        _lookup[(indices.numel(), dim // 256)](
            indices.contiguous(),
            low,
            planes,
            ends,
            side,
            out,
            base,
            rows,
            dim,
            256,
            num_warps=2,
        )
    return out
