# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU-only test references; never import from production code."""

import torch


def quantize_reference(x):
    assert x.device.type == "cpu" and x.shape[-1] == 128
    values = x.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    raw_scale = maximum / 127.0
    # Exact ceil(log2) from FP32 bits, independently of Triton log2/exp2.
    bits = raw_scale.view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127
    exponent = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    scales = torch.ldexp(torch.ones_like(raw_scale), exponent)
    quantized = (values / scales).clamp(-127, 127).round().to(torch.int8)
    return quantized, scales


def cache_views_reference(cache, page_size):
    assert cache.device.type == "cpu" and cache.dtype == torch.uint8
    pages = cache.reshape(cache.shape[0], page_size * 132)
    values = pages[:, : page_size * 128].view(torch.int8)
    scales = pages[:, page_size * 128 :].view(torch.float32)
    return values.reshape(cache.shape[0], page_size, 128), scales


def cache_store_reference(keys, cache, slots, page_size):
    assert keys.device.type == slots.device.type == "cpu"
    values, scales = cache_views_reference(cache, page_size)
    q, s = quantize_reference(keys[: slots.numel()])
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, offset = divmod(slot, page_size)
        values[page, offset].copy_(q[row])
        scales[page, offset] = s[row, 0]
    return cache


def paged_logits_reference(
    q, cache, weights, table, row_batch, lengths, max_len, page_size
):
    assert all(
        x.device.type == "cpu" for x in (q, cache, weights, table, row_batch, lengths)
    )
    values, scales = cache_views_reference(cache, page_size)
    result = torch.full((q.shape[0], max_len), -float("inf"), dtype=torch.float32)
    for row in range(q.shape[0]):
        batch = int(row_batch[row])
        if batch < 0:
            continue
        for column in range(min(max(int(lengths[row]), 0), max_len)):
            page = int(table[batch, column // page_size])
            if page < 0:
                continue
            offset = column % page_size
            # Signed integer dot is exact (bound: 128*127*127 < INT32_MAX).
            dot = (q[row].to(torch.int64) * values[page, offset].to(torch.int64)).sum(
                -1
            )
            weighted = dot.clamp_min(0).float() * weights[row].float()
            result[row, column] = (
                weighted.sum(dtype=torch.float32) * scales[page, offset]
            )
    return result


def fixture(page_size=16, heads=4, rows=4):
    generator = torch.Generator().manual_seed(40 + page_size + heads + rows)
    cache = torch.full((4, page_size, 1, 132), 37, dtype=torch.uint8)
    values, scales = cache_views_reference(cache, page_size)
    values.copy_(
        torch.randint(-4, 5, values.shape, generator=generator, dtype=torch.int8)
    )
    scales.fill_(0.25)
    q = torch.randint(-4, 5, (rows, heads, 128), generator=generator, dtype=torch.int8)
    weights = (
        torch.tensor([1.0, -0.5, 0.25, 0.0])
        .repeat((heads + 3) // 4)[:heads]
        .repeat(rows, 1)
    )
    table = torch.tensor([[2, 0, -1], [1, 3, 0]], dtype=torch.int32)
    row_batch = torch.tensor([1, 0, -1, 1], dtype=torch.int32)[:rows]
    lengths = torch.tensor([page_size + 1, 2 * page_size + 1, 5, 0], dtype=torch.int32)[
        :rows
    ]
    return q, cache, weights, table, row_batch, lengths, 3 * page_size, page_size
