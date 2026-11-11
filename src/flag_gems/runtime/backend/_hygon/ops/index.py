# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compact boolean prefix masks directly, without materializing coordinates."""

import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.index import index as _generic_index
from flag_gems.runtime import torch_device_fn


@triton.jit
def _gather_tiles(
    X,
    INDEX,
    OUT,
    COUNT: tl.constexpr,
    ROWS: tl.constexpr,
    SUFFIX: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    i = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    row = tl.load(INDEX + i, i < COUNT, other=0)
    tl.device_assert(
        (i >= COUNT) | ((row >= -ROWS) & (row < ROWS)), "index out of bounds"
    )
    row = tl.where(row < 0, row + ROWS, row)
    j = tl.arange(0, TILE_COLS)
    cur = j
    offset = tl.full((TILE_COLS,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, 0, -1):
        offset += cur % SHAPE[d] * STRIDE[d]
        cur = cur // SHAPE[d]
    valid = (i[:, None] < COUNT) & (j[None, :] < SUFFIX)
    value = tl.load(
        X + row[:, None] * STRIDE[0] + offset[None, :],
        valid & (row[:, None] >= 0) & (row[:, None] < ROWS),
        other=0,
    )
    tl.store(OUT + i[:, None] * SUFFIX + j[None, :], value, valid)


@triton.jit
def _gather_rows(
    X,
    INDEX,
    OUT,
    SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SUFFIX: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE
    row = tl.load(INDEX + i // SUFFIX, valid, other=0)
    tl.device_assert((~valid) | ((row >= -ROWS) & (row < ROWS)), "index out of bounds")
    row = tl.where(row < 0, row + ROWS, row)
    cur = i % SUFFIX
    offset = row * STRIDE[0]
    for d in tl.static_range(len(SHAPE) - 1, 0, -1):
        offset += cur % SHAPE[d] * STRIDE[d]
        cur = cur // SHAPE[d]
    value = tl.load(X + offset, valid & (row >= 0) & (row < ROWS), other=0)
    tl.store(OUT + i, value, valid)


@triton.jit
def _mask_counts(MASK, COUNTS, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    take = tl.load(MASK + i, i < SIZE, other=0).to(tl.int32)
    tl.store(COUNTS + tl.program_id(0), tl.sum(take, 0))


@triton.jit
def _scan_counts(COUNTS, PREFIX, TOTAL, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    count = tl.load(COUNTS + i, i < N, other=0)
    prefix = tl.cumsum(count, 0)
    tl.store(PREFIX + i, prefix - count, i < N)
    tl.store(TOTAL, tl.sum(count, 0))


@triton.jit
def _compact_mask(
    X,
    MASK,
    PREFIX,
    OUT,
    SIZE: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    SUFFIX_SHAPE: tl.constexpr,
    SUFFIX_STRIDE: tl.constexpr,
    SUFFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    take = tl.load(MASK + i, i < SIZE, other=0).to(tl.int32)
    rank = tl.cumsum(take, 0) - 1 + tl.load(PREFIX + tl.program_id(0))
    cur = i
    offset = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += cur % SHAPE[d] * STRIDE[d]
        cur = cur // SHAPE[d]
    if SUFFIX == 1:
        value = tl.load(X + offset, (i < SIZE) & (take != 0), other=0)
        tl.store(OUT + rank, value, (i < SIZE) & (take != 0))
    else:
        j = tl.arange(0, triton.next_power_of_2(SUFFIX))
        cur_j = j
        offset_j = tl.full((triton.next_power_of_2(SUFFIX),), 0, tl.int64)
        for d in tl.static_range(len(SUFFIX_SHAPE) - 1, -1, -1):
            offset_j += cur_j % SUFFIX_SHAPE[d] * SUFFIX_STRIDE[d]
            cur_j = cur_j // SUFFIX_SHAPE[d]
        valid = (i[:, None] < SIZE) & (take[:, None] != 0) & (j[None, :] < SUFFIX)
        value = tl.load(X + offset[:, None] + offset_j[None, :], valid, other=0)
        tl.store(OUT + rank[:, None] * SUFFIX + j[None, :], value, valid)


def _legacy_index(inp, indices):
    if len(indices) != 1 or indices[0] is None:
        return _generic_index(inp, indices)
    mask = indices[0]
    if (
        mask.dtype in (torch.int32, torch.int64)
        and mask.ndim == 1
        and mask.is_contiguous()
        and mask.device == inp.device
        and inp.ndim
        and inp.shape[0]
        and not inp.is_complex()
    ):
        suffix = math.prod(inp.shape[1:])
        if 0 < suffix <= 128:
            out = torch.empty(
                (mask.numel(), *inp.shape[1:]), dtype=inp.dtype, device=inp.device
            )
            if out.numel():
                with torch_device_fn.device(inp.device):
                    if suffix > 16:
                        tile_rows = (
                            1
                            if mask.numel() <= 300
                            else 2 if mask.numel() <= 2048 else 16
                        )
                        _gather_tiles[(triton.cdiv(mask.numel(), tile_rows),)](
                            inp,
                            mask,
                            out,
                            mask.numel(),
                            inp.shape[0],
                            suffix,
                            tuple(inp.shape),
                            tuple(inp.stride()),
                            tile_rows,
                            triton.next_power_of_2(suffix),
                            num_warps=1 if tile_rows <= 2 else 4,
                            debug=True,
                        )
                        return out
                    block = 256 if out.numel() <= 16384 else 1024
                    _gather_rows[(triton.cdiv(out.numel(), block),)](
                        inp,
                        mask,
                        out,
                        out.numel(),
                        inp.shape[0],
                        suffix,
                        tuple(inp.shape),
                        tuple(inp.stride()),
                        block,
                        num_warps=1 if out.numel() <= 16384 else 4,
                        debug=True,
                    )
            return out
    if (
        mask.dtype != torch.bool
        or mask.ndim == 0
        or not mask.is_contiguous()
        or mask.device != inp.device
        or inp.is_complex()
        or tuple(mask.shape) != tuple(inp.shape[: mask.ndim])
    ):
        return _generic_index(inp, indices)
    suffix_shape = tuple(inp.shape[mask.ndim :])
    suffix = math.prod(suffix_shape)
    if suffix > 16 or mask.numel() > 2**30:
        return _generic_index(inp, indices)
    size = mask.numel()
    if size == 0:
        return torch.empty((0, *suffix_shape), dtype=inp.dtype, device=inp.device)
    block = 1024
    n = triton.cdiv(size, block)
    counts = torch.empty((n,), dtype=torch.int32, device=inp.device)
    prefix = torch.empty_like(counts)
    total = torch.empty((), dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        _mask_counts[(n,)](mask, counts, size, block, num_warps=4)
        _scan_counts[(1,)](
            counts, prefix, total, n, triton.next_power_of_2(n), num_warps=4
        )
        # Only the output length crosses to the host; compaction stays on device.
        selected = total.item()
        out = torch.empty((selected, *suffix_shape), dtype=inp.dtype, device=inp.device)
        if selected and suffix:
            _compact_mask[(n,)](
                inp,
                mask,
                prefix,
                out,
                size,
                tuple(mask.shape),
                tuple(inp.stride()[: mask.ndim]),
                suffix_shape,
                tuple(inp.stride()[mask.ndim :]),
                suffix,
                block,
                num_warps=4,
            )
    return out


# Additional layouts: strided/column gathers and wide or broadcast masks.
@triton.jit
def _offset(i, SHAPE: tl.constexpr, STRIDE: tl.constexpr):
    out = tl.full(i.shape, 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        out += (i % SHAPE[d]) * STRIDE[d]
        i = i // SHAPE[d]
    return out


@triton.jit
def _axis_gather(
    X,
    I,
    O,
    SIZE: tl.constexpr,
    COUNT: tl.constexpr,
    ROWS: tl.constexpr,
    SUFFIX: tl.constexpr,
    PS: tl.constexpr,
    PST: tl.constexpr,
    SS: tl.constexpr,
    SST: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE
    row = tl.load(I + ((i // SUFFIX) % COUNT) * INDEX_STRIDE, valid, other=0)
    tl.device_assert((~valid) | ((row >= -ROWS) & (row < ROWS)), "index out of bounds")
    row = tl.where(row < 0, row + ROWS, row)
    offset = (
        _offset(i // (COUNT * SUFFIX), PS, PST)
        + row * AXIS_STRIDE
        + _offset(i % SUFFIX, SS, SST)
    )
    value = tl.load(X + offset, valid & (row >= 0) & (row < ROWS), other=0)
    tl.store(O + i, value, valid)


@triton.jit
def _counts(
    M, C, N: tl.constexpr, SH: tl.constexpr, ST: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(M + _offset(i, SH, ST), i < N, other=0).to(tl.int32)
    tl.store(C + tl.program_id(0), tl.sum(v, 0))


@triton.jit
def _prefix(C, P, T, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    c = tl.load(C + i, i < N, other=0)
    p = tl.cumsum(c, 0)
    tl.store(P + i, p - c, i < N)
    tl.store(T, tl.sum(c, 0))


@triton.jit
def _compact(
    X,
    M,
    P,
    O,
    N: tl.constexpr,
    SH: tl.constexpr,
    MST: tl.constexpr,
    XST: tl.constexpr,
    SS: tl.constexpr,
    SST: tl.constexpr,
    SUFFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    take = tl.load(M + _offset(i, SH, MST), i < N, other=0).to(tl.int32)
    rank = tl.cumsum(take, 0) - 1 + tl.load(P + tl.program_id(0))
    offset = _offset(i, SH, XST)
    if SUFFIX == 1:
        valid = (i < N) & (take != 0)
        v = tl.load(X + offset, valid, other=0)
        tl.store(O + rank, v, valid)
    else:
        j = tl.arange(0, triton.next_power_of_2(SUFFIX))
        sj = _offset(j, SS, SST)
        valid = (i[:, None] < N) & (take[:, None] != 0) & (j[None, :] < SUFFIX)
        v = tl.load(X + offset[:, None] + sj[None, :], valid, other=0)
        tl.store(O + rank[:, None] * SUFFIX + j[None, :], v, valid)


def index(x, indices):
    nonnull = [(d, v) for d, v in enumerate(indices) if v is not None]
    if len(nonnull) == 1 and len(indices) <= x.ndim:
        axis, ix = nonnull[0]
        if (
            ix.dtype in (torch.int32, torch.int64)
            and ix.ndim == 1
            and axis < x.ndim
            and x.shape[axis] > 0
            and ix.device == x.device
            and not x.is_complex()
        ):
            shape = (*x.shape[:axis], ix.numel(), *x.shape[axis + 1 :])
            output_size = math.prod(shape)
            suffix_size = math.prod(x.shape[axis + 1 :])
            if (
                ix.is_contiguous()
                and axis == 0
                and (output_size < 1048576 or suffix_size > 16)
                and not (output_size <= 16384 and suffix_size <= 16)
            ):
                return _legacy_index(x, indices)
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
            size = out.numel()
            suffix = math.prod(x.shape[axis + 1 :])
            if size:
                block, warps = (64, 1) if size <= 16384 and suffix <= 16 else (256, 4)
                with torch_device_fn.device(x.device):
                    _axis_gather[(triton.cdiv(size, block),)](
                        x,
                        ix,
                        out,
                        size,
                        ix.numel(),
                        x.shape[axis],
                        suffix,
                        tuple(x.shape[:axis]),
                        tuple(x.stride()[:axis]),
                        tuple(x.shape[axis + 1 :]),
                        tuple(x.stride()[axis + 1 :]),
                        x.stride(axis),
                        ix.stride(0),
                        block,
                        num_warps=warps,
                        debug=True,
                    )
            return out
    if len(indices) == 1 and indices[0] is not None:
        mask = indices[0]
        rank = mask.ndim
        suffix = math.prod(x.shape[rank:])
        if (
            mask.dtype == torch.bool
            and mask.device == x.device
            and rank
            and tuple(mask.shape) == tuple(x.shape[:rank])
            and suffix <= 128
            and not x.is_complex()
            and (suffix > 16 or not mask.is_contiguous())
        ):
            n = mask.numel()
            if not n:
                return torch.empty((0, *x.shape[rank:]), dtype=x.dtype, device=x.device)
            block, warps = (32, 4) if suffix > 16 else (2048, 4)
            groups = triton.cdiv(n, block)
            if groups > 8192:
                return _legacy_index(x, indices)
            counts = torch.empty((groups,), dtype=torch.int32, device=x.device)
            prefix = torch.empty_like(counts)
            total = torch.empty((), dtype=torch.int32, device=x.device)
            with torch_device_fn.device(x.device):
                _counts[(groups,)](
                    mask,
                    counts,
                    n,
                    tuple(mask.shape),
                    tuple(mask.stride()),
                    block,
                    num_warps=warps,
                )
                _prefix[(1,)](
                    counts,
                    prefix,
                    total,
                    groups,
                    triton.next_power_of_2(groups),
                    num_warps=4,
                )
                selected = total.item()
                out = torch.empty(
                    (selected, *x.shape[rank:]), dtype=x.dtype, device=x.device
                )
                if selected and suffix:
                    _compact[(groups,)](
                        x,
                        mask,
                        prefix,
                        out,
                        n,
                        tuple(mask.shape),
                        tuple(mask.stride()),
                        tuple(x.stride()[:rank]),
                        tuple(x.shape[rank:]),
                        tuple(x.stride()[rank:]),
                        suffix,
                        block,
                        num_warps=warps,
                    )
            return out
    return _legacy_index(x, indices)
