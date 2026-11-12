# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from .int8_indexer_reference import cache_store_reference
from .int8_indexer_reference_checks import (  # noqa: F401
    test_cache_page_bytes_and_slot_mutation,
)

pytestmark = pytest.mark.int8_indexer_cache_store


@pytest.mark.int8_indexer_cache_store
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_int8_indexer_cache_state(page_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("GPU component test")
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_cache_store

    cpu_keys = (torch.arange(4 * 135).reshape(4, 135) % 127 - 63).to(dtype)[:, :128]
    cpu_cache = torch.full((3, page_size, 132), 29, dtype=torch.uint8)
    slots = torch.tensor(
        [page_size - 1, page_size, -1, 2 * page_size + 1], dtype=torch.int64
    )
    expected = cache_store_reference(cpu_keys, cpu_cache.clone(), slots, page_size)
    keys = torch.empty_strided(cpu_keys.shape, (135, 1), dtype=dtype, device="cuda")
    keys.copy_(cpu_keys)
    cache = cpu_cache.cuda()
    result = int8_indexer_cache_store(keys, cache, slots.cuda(), page_size)
    assert result is None
    assert torch.equal(cache.cpu(), expected)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("page_size", [16, 64])
@torch.inference_mode()
def test_public_cache_store_matches_same_exact_scales_and_slot_state(
    dtype_name, page_size
):
    if not torch.cuda.is_available():
        pytest.skip("GPU component test")
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_cache_store

    from .int8_indexer_exact_oracle import adversarial_tensors

    x_cpu, expected_q, expected_scales = adversarial_tensors(dtype_name)
    rows = x_cpu.shape[0]
    pages = (rows + page_size - 1) // page_size + 1
    initial = torch.full((pages, page_size, 1, 132), 53, dtype=torch.uint8)
    # Unique permuted destinations, explicit page-boundary writes and one
    # negative slot. Compare complete cache bytes, not only the written scales.
    slots = torch.arange(rows, dtype=torch.int64).flip(0)
    slots[0] = -1

    def expected_cache(q_values, scale_values):
        result = initial.clone()
        flat = result.view(pages, page_size * 132)
        values = flat[:, : page_size * 128].view(torch.int8).view(pages, page_size, 128)
        scales = flat[:, page_size * 128 :].view(torch.float32)
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                page, offset = divmod(slot, page_size)
                values[page, offset] = q_values[row]
                scales[page, offset] = scale_values[row, 0]
        return result

    k = torch.empty_strided(x_cpu.shape, (135, 1), dtype=x_cpu.dtype, device="cuda")
    k.copy_(x_cpu)
    cache, gpu_slots = initial.cuda(), slots.cuda()
    int8_indexer_cache_store(k, cache, gpu_slots, page_size)
    assert torch.equal(cache.cpu(), expected_cache(expected_q, expected_scales))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        int8_indexer_cache_store(k, cache, gpu_slots, page_size)
        with torch.cuda.graph(graph, stream=stream):
            int8_indexer_cache_store(k, cache, gpu_slots, page_size)
    torch.cuda.current_stream().wait_stream(stream)
    k.copy_(x_cpu.flip(0))
    cache.copy_(initial)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        cache.cpu(), expected_cache(expected_q.flip(0), expected_scales.flip(0))
    )
