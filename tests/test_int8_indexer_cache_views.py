# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU storage contracts for the public no-copy packed-cache helper."""

import pytest
import torch


@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("pages", [0, 1, 3])
def test_cache_views_are_aliases_with_separate_byte_and_scale_sections(
    page_size, pages
):
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_cache_views

    cache = torch.zeros((pages, page_size, 132), dtype=torch.uint8)
    values, scales = int8_indexer_cache_views(cache, page_size)
    assert values.shape == (pages, page_size * 128)
    assert scales.shape == (pages, page_size)
    assert values.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert scales.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    if pages:
        values[0, page_size * 128 - 1] = -7
        scales[0, 0] = 0.25
        flat = cache.view(pages, page_size * 132)
        assert flat[0, page_size * 128 - 1].item() == 249
        assert torch.equal(
            flat[0, page_size * 128 : page_size * 128 + 4],
            torch.tensor([0, 0, 128, 62], dtype=torch.uint8),
        )


@pytest.mark.parametrize("page_size", [0, -1, True, 1.0])
def test_cache_views_reject_invalid_page_size(page_size):
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_cache_views

    with pytest.raises(ValueError, match="positive integer"):
        int8_indexer_cache_views(
            torch.empty((1, 64, 132), dtype=torch.uint8), page_size
        )


def test_cache_views_reject_unaligned_or_noncontiguous_storage():
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_cache_views

    storage = torch.empty(64 * 132 + 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="FP32-aligned"):
        int8_indexer_cache_views(storage[1:].view(1, 64, 132), 64)
    with pytest.raises(ValueError, match="contiguous"):
        int8_indexer_cache_views(
            torch.empty((2, 64, 132), dtype=torch.uint8)[..., ::2], 64
        )
