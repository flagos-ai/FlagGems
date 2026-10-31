# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from .int8_indexer_reference import (
    cache_store_reference,
    cache_views_reference,
    fixture,
    paged_logits_reference,
    quantize_reference,
)


def test_nearest_even_signed_range_and_zero_scale():
    x = torch.zeros(3, 128)
    x[0, :9] = torch.tensor([127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0])
    x[1, :3] = torch.tensor([128, -128, 1])
    q, s = quantize_reference(x)
    assert s[:, 0].tolist() == [1.0, 2.0, 2.0**-40]
    assert q[0, :9].tolist() == [127, -127, 0, 2, 2, 0, -2, -2, 0]
    assert q[1, :3].tolist() == [64, -64, 0]
    assert torch.count_nonzero(q[2]) == 0


@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cache_page_bytes_and_slot_mutation(page_size, dtype):
    cache = torch.full((3, page_size, 1, 132), 29, dtype=torch.uint8)
    before = cache.clone()
    keys = torch.zeros(4, 133, dtype=dtype)[:, :128]
    keys[:, :4] = torch.tensor(
        [
            [127, 1.5, -127, -2.5],
            [63.5, 0.75, -63.5, -1.25],
            [128, 3, -128, -5],
            [1, 2, 3, 4],
        ],
        dtype=dtype,
    )
    slots = torch.tensor(
        [page_size - 1, page_size, -1, 2 * page_size + 1], dtype=torch.int64
    )
    cache_store_reference(keys, cache, slots, page_size)
    values, scales = cache_views_reference(cache, page_size)
    q, s = quantize_reference(keys)
    for row, slot in [(0, page_size - 1), (1, page_size), (3, 2 * page_size + 1)]:
        page, offset = divmod(slot, page_size)
        assert torch.equal(values[page, offset], q[row])
        assert scales[page, offset].item() == s[row, 0].item()
    old_values, old_scales = cache_views_reference(before, page_size)
    for page in range(3):
        for offset in range(page_size):
            if page * page_size + offset not in slots.tolist():
                assert torch.equal(values[page, offset], old_values[page, offset])
                assert torch.equal(
                    scales[page, offset].view(torch.int32),
                    old_scales[page, offset].view(torch.int32),
                )


@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("heads", [1, 4, 32])
def test_integer_dot_relu_weight_and_page_mapping(page_size, heads):
    args = fixture(page_size, heads)
    q, cache, weights, table, row_batch, lengths, max_len, _ = args
    out = paged_logits_reference(*args)
    assert out.shape == (4, max_len)
    assert torch.isneginf(out[2:]).all()
    assert torch.isneginf(out[0, page_size + 1 :]).all()
    assert torch.isneginf(out[1, 2 * page_size :]).all()  # negative page marker
    values, scales = cache_views_reference(cache, page_size)
    for row in (0, 1):
        page = int(table[int(row_batch[row]), 0])
        # Independent FP64 literal sum is exact for bounded integer fixtures.
        dots = (q[row].double() * values[page, 0].double()).sum(-1)
        expected = sum(
            max(float(dot), 0.0) * float(weight)
            for dot, weight in zip(dots, weights[row])
        ) * float(scales[page, 0])
        assert out[row, 0].item() == expected


def test_quantization_empty_and_power_of_two_thresholds():
    q, s = quantize_reference(torch.empty(0, 4, 128))
    assert q.shape == (0, 4, 128) and s.shape == (0, 4, 1)
    peaks = torch.tensor([63.5, 63.500003814697266, 127, 127.00000762939453])
    x = torch.zeros(4, 128)
    x[:, 0] = peaks
    _, s = quantize_reference(x)
    assert s[:, 0].tolist() == [0.5, 1.0, 1.0, 2.0]


def test_fp32_logarithm_precision_boundary_is_discriminating():
    raw = torch.nextafter(torch.tensor(2.0**-20), torch.tensor(float("inf")))
    x = torch.zeros(1, 128)
    x[0, 0] = raw * 127
    _, exact_scale = quantize_reference(x)
    log_scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True) / 127)))
    assert exact_scale.item() == 2.0**-19
    assert log_scale.item() == 2.0**-20
    # This proves the reference fixture is sensitive; no GPU result is inferred.
