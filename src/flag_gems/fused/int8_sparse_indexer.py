# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""INT8 DSA query/cache quantization and paged MQA logits.

The cache uses the DSA page ABI: P*128 signed bytes followed by P FP32
scales. Queries and keys use symmetric INT8 with power-of-two scales and
round-to-nearest-even. Dot products accumulate in INT32; no FP8 conversion
or instruction is involved. Query scales are folded into the head weights.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INDEX_DTYPES = (torch.int32, torch.int64)


def _check_tensors(*values):
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("tensor inputs required")
    device = values[0].device
    if device.type != "cuda" or any(value.device != device for value in values):
        raise ValueError("inputs must share one GPU device")
    if any(
        value.layout != torch.strided
        or value.requires_grad
        or value.is_conj()
        or value.is_neg()
        for value in values
    ):
        raise ValueError("strided inference tensors required")


def _check_float_heads(value):
    if (
        value.dtype not in _FLOAT_DTYPES
        or value.ndim < 2
        or value.shape[-1] != 128
        or value.stride(-1) != 1
    ):
        raise ValueError("floating input requires unit-stride 128-element heads")


def _check_vector(value, size, name):
    if (
        value.ndim != 1
        or value.numel() != size
        or value.stride(0) != 1
        or value.dtype not in _INDEX_DTYPES
    ):
        raise ValueError(
            f"{name} must be a contiguous INT32/INT64 vector of length {size}"
        )


@triton.jit
def _power_of_two_scale(amax):
    # Keep the minimum on amax before division, including all-zero rows.
    raw_scale = tl.div_rn(tl.maximum(amax, 1e-10), 127.0)
    # The finite FP32 input domain makes raw_scale positive and normal.
    # Reading its mantissa avoids log2 rounding away a one-ULP excess.
    bits = raw_scale.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _quantize(X, Q, S, XS: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    x = tl.load(X + row * XS + d).to(tl.float32)
    scale = _power_of_two_scale(tl.max(tl.abs(x)))
    q = libdevice.nearbyint(tl.minimum(tl.maximum(x / scale, -127.0), 127.0))
    tl.store(Q + row * D + d, q.to(tl.int8))
    tl.store(S + row, scale)


def int8_indexer_quantize(x):
    """Quantize finite floating heads to signed INT8 and FP32 power-of-two scales.

    Leading dimensions must collapse into rows without materialization.
    """
    _check_tensors(x)
    _check_float_heads(x)
    try:
        flat = x.view(-1, 128)
    except RuntimeError as error:
        raise ValueError(
            "leading head dimensions must collapse without a copy"
        ) from error
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scales = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    if flat.shape[0]:
        _quantize[(flat.shape[0],)](flat, q, scales, flat.stride(0), 128)
    return q, scales


def _cache_views(cache, page_size):
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if (
        cache.dtype != torch.uint8
        or cache.ndim < 2
        or not cache.is_contiguous()
        or cache.storage_offset() % 4
    ):
        raise ValueError("cache must be contiguous, FP32-aligned uint8 storage")
    elements = page_size * 132
    if cache.numel() != cache.shape[0] * elements:
        raise ValueError(
            "each cache page requires 128 INT8 bytes plus one FP32 scale per token"
        )
    flat = cache.view(cache.shape[0], elements)
    return flat[:, : page_size * 128].view(torch.int8), flat[:, page_size * 128 :].view(
        torch.float32
    )


@triton.jit
def _cache_store(
    K,
    V,
    S,
    Slots,
    KS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    P: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    if slot >= 0:
        d = tl.arange(0, 128)
        k = tl.load(K + row * KS + d).to(tl.float32)
        scale = _power_of_two_scale(tl.max(tl.abs(k)))
        q = libdevice.nearbyint(tl.minimum(tl.maximum(k / scale, -127.0), 127.0))
        page, offset = slot // P, slot % P
        tl.store(V + page * VS + offset * 128 + d, q.to(tl.int8))
        tl.store(S + page * SS + offset, scale)


def int8_indexer_cache_store(k, cache, slots, page_size):
    """Quantize keys and update their selected cache slots in place.

    Nonnegative slots must be unique and within cache capacity; negative slots
    are ignored. The caller owns slot validation and write ordering.
    """
    _check_tensors(k, cache, slots)
    _check_float_heads(k)
    if k.ndim != 2:
        raise ValueError("keys must have shape [tokens, 128]")
    _check_vector(slots, slots.numel(), "slots")
    if slots.numel() > k.shape[0]:
        raise ValueError("slot mapping is longer than the key tensor")
    values, scales = _cache_views(cache, page_size)
    if slots.numel():
        _cache_store[(slots.numel(),)](
            k,
            values,
            scales,
            slots,
            k.stride(0),
            values.stride(0),
            scales.stride(0),
            page_size,
        )


@triton.jit
def _paged_logits(
    Q,
    W,
    V,
    S,
    Table,
    RowBatch,
    Lengths,
    Out,
    QT: tl.constexpr,
    QH: tl.constexpr,
    WT: tl.constexpr,
    WH: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    TS: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    MAX_K: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    keys = tl.program_id(1) * BK + tl.arange(0, BK)
    batch = tl.load(RowBatch + row)
    length = tl.load(Lengths + row)
    valid = (keys < length) & (keys < MAX_K) & (batch >= 0)
    pages = tl.load(Table + batch * TS + keys // P, valid, 0)
    valid = valid & (pages >= 0)
    d = tl.arange(0, 128)
    heads = tl.arange(0, BH)
    q = tl.load(Q + row * QT + heads[:, None] * QH + d[None, :], heads[:, None] < H, 0)
    k = tl.load(
        V + pages[None, :] * VS + (keys[None, :] % P) * 128 + d[:, None],
        valid[None, :],
        0,
    )
    dots = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
    weights = tl.load(W + row * WT + heads * WH, heads < H, 0).to(tl.float32)
    scale = tl.load(S + pages * SS + keys % P, valid, 0)
    logits = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0) * scale
    logits = tl.where(valid, logits, float("-inf"))
    tl.store(Out + row * MAX_K + keys, logits, keys < MAX_K)


def int8_indexer_paged_logits(
    q, cache, weights, block_table, row_batch, lengths, max_model_len, page_size
):
    """Compute weighted ReLU INT32 query/key dots through a packed INT8 cache.

    Query scales must already be folded into weights. Invalid batch/page IDs
    are negative; nonnegative IDs and lengths must lie within supplied storage.
    Masked positions are written as -inf, including zero-length rows.
    """
    _check_tensors(q, cache, weights, block_table, row_batch, lengths)
    if (
        q.dtype != torch.int8
        or q.ndim != 3
        or q.shape[-1] != 128
        or q.stride(-1) != 1
        or q.shape[1] <= 0
    ):
        raise ValueError(
            "queries require INT8 [tokens, heads, 128] with unit head-dimension stride"
        )
    if weights.shape != q.shape[:2] or weights.dtype != torch.float32:
        raise ValueError("weights must be FP32 [tokens, heads]")
    _check_vector(row_batch, q.shape[0], "row_batch")
    _check_vector(lengths, q.shape[0], "lengths")
    if (
        block_table.ndim != 2
        or block_table.stride(1) != 1
        or block_table.dtype not in _INDEX_DTYPES
    ):
        raise ValueError("block table requires INT32/INT64 unit-stride rows")
    values, scales = _cache_views(cache, page_size)
    if (
        type(max_model_len) is not int
        or max_model_len < 0
        or triton.cdiv(max_model_len, page_size) > block_table.shape[1]
    ):
        raise ValueError("maximum context exceeds the block table")
    out = torch.empty((q.shape[0], max_model_len), dtype=torch.float32, device=q.device)
    if q.shape[0] and max_model_len:
        _paged_logits[(q.shape[0], triton.cdiv(max_model_len, 32))](
            q,
            weights,
            values,
            scales,
            block_table,
            row_batch,
            lengths,
            out,
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            weights.stride(1),
            values.stride(0),
            scales.stride(0),
            block_table.stride(0),
            q.shape[1],
            page_size,
            max_model_len,
            triton.next_power_of_2(max(q.shape[1], 16)),
            32,
            num_warps=4,
        )
    return out


__all__ = [
    "int8_indexer_cache_store",
    "int8_indexer_paged_logits",
    "int8_indexer_quantize",
]
