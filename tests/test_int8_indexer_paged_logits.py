# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent INT8 primitive mathematics and mutable-input graph replay."""

import pytest
import torch

from .int8_indexer_reference import (
    cache_store_reference,
    fixture,
    paged_logits_reference,
    quantize_reference,
)
from .int8_indexer_reference_checks import (  # noqa: F401
    test_integer_dot_relu_weight_and_page_mapping,
)

pytestmark = pytest.mark.int8_indexer_paged_logits


@pytest.mark.int8_indexer_quantize
@pytest.mark.int8_indexer_cache_store
@pytest.mark.int8_indexer_paged_logits
@pytest.mark.parametrize(
    "operation, violation",
    [
        ("quantize", "dtype"),
        ("quantize", "last_stride"),
        ("quantize", "leading_view"),
        ("cache_store", "key_dtype"),
        ("cache_store", "slot_dtype"),
        ("cache_store", "slot_stride"),
        ("cache_store", "cache_dtype"),
        ("cache_store", "page_size"),
        ("paged_logits", "query_dtype"),
        ("paged_logits", "query_stride"),
        ("paged_logits", "weights_dtype"),
        ("paged_logits", "row_dtype"),
        ("paged_logits", "length_stride"),
        ("paged_logits", "table_dtype"),
        ("paged_logits", "extent"),
    ],
)
def test_int8_indexer_metadata_guards_before_launch(monkeypatch, operation, violation):
    import importlib

    if not torch.cuda.is_available():
        pytest.skip("GPU metadata test")
    module = importlib.import_module("flag_gems.fused.int8_sparse_indexer")
    calls = []

    class KernelSpy:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(module, "_quantize", KernelSpy())
    monkeypatch.setattr(module, "_cache_store", KernelSpy())
    monkeypatch.setattr(module, "_paged_logits", KernelSpy())
    device = "cuda"
    cache = torch.empty((2, 64, 132), dtype=torch.uint8, device=device)
    if operation == "quantize":
        args = [torch.empty((2, 4, 128), dtype=torch.bfloat16, device=device)]
    elif operation == "cache_store":
        args = [
            torch.empty((2, 128), dtype=torch.bfloat16, device=device),
            cache,
            torch.empty((2,), dtype=torch.int64, device=device),
            64,
        ]
    else:
        args = [
            torch.empty((2, 4, 128), dtype=torch.int8, device=device),
            cache,
            torch.empty((2, 4), dtype=torch.float32, device=device),
            torch.empty((1, 2), dtype=torch.int32, device=device),
            torch.empty((2,), dtype=torch.int32, device=device),
            torch.empty((2,), dtype=torch.int32, device=device),
            128,
            64,
        ]
    entry = getattr(module, "int8_indexer_" + operation)
    entry(*args)
    assert len(calls) == 1
    calls.clear()
    if violation in ("dtype", "key_dtype", "query_dtype"):
        args[0] = args[0].to(torch.float64)
    elif violation in ("last_stride", "query_stride"):
        args[0] = torch.empty(
            (*args[0].shape[:-1], 256), dtype=args[0].dtype, device=device
        )[..., ::2]
    elif violation == "leading_view":
        args[0] = torch.empty_strided(
            (2, 4, 128), (1024, 128, 1), dtype=args[0].dtype, device=device
        )
    elif violation == "slot_dtype":
        args[2] = args[2].float()
    elif violation == "slot_stride":
        args[2] = torch.empty(4, dtype=torch.int64, device=device)[::2]
    elif violation == "cache_dtype":
        args[1] = args[1].to(torch.int8)
    elif violation == "page_size":
        args[3] = 0
    elif violation == "weights_dtype":
        args[2] = args[2].bfloat16()
    elif violation == "row_dtype":
        args[4] = args[4].float()
    elif violation == "length_stride":
        args[5] = torch.empty(4, dtype=torch.int32, device=device)[::2]
    elif violation == "table_dtype":
        args[3] = args[3].float()
    elif violation == "extent":
        args[6] = 129
    with pytest.raises((ValueError, TypeError)):
        entry(*args)
    assert calls == []


@pytest.mark.int8_indexer_quantize
@pytest.mark.int8_indexer_cache_store
@pytest.mark.int8_indexer_paged_logits
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("heads", [1, 4, 32])
@torch.inference_mode()
def test_finite_math_cache_mutation_and_ordinary_graph(page_size, heads):
    if not torch.cuda.is_available():
        pytest.skip("GPU component test")
    # Use the real delivered public module.
    from flag_gems.fused.int8_sparse_indexer import (
        int8_indexer_cache_store,
        int8_indexer_paged_logits,
        int8_indexer_quantize,
    )

    cpu = fixture(page_size, heads)
    q_cpu, cache_cpu, weights_cpu, table_cpu, row_cpu, lengths_cpu, max_len, _ = cpu
    x_cpu = (torch.arange(4 * heads * 128).reshape(4, heads, 128) % 127 - 63).bfloat16()
    k_cpu = (torch.arange(4 * 135).reshape(4, 135) % 63 - 31).bfloat16()[:, :128]
    slots_cpu = torch.tensor(
        [page_size - 1, page_size, -1, 3 * page_size], dtype=torch.int64
    )
    x, k, slots = (
        x_cpu.cuda(),
        torch.empty_strided(k_cpu.shape, (135, 1), device="cuda", dtype=k_cpu.dtype),
        slots_cpu.cuda(),
    )
    k.copy_(k_cpu)
    q, cache, weights, table, row_batch, lengths = (v.cuda() for v in cpu[:6])
    initial_cache = cache_cpu.clone()

    def run():
        quantized, scales = int8_indexer_quantize(x)
        int8_indexer_cache_store(k, cache, slots, page_size)
        logits = int8_indexer_paged_logits(
            q, cache, weights, table, row_batch, lengths, max_len, page_size
        )
        return quantized, scales, logits

    eager = run()
    expected_cache = cache_store_reference(
        k_cpu, initial_cache.clone(), slots_cpu, page_size
    )
    expected_q, expected_scale = quantize_reference(x_cpu)
    expected_logits = paged_logits_reference(
        q_cpu,
        expected_cache,
        weights_cpu,
        table_cpu,
        row_cpu,
        lengths_cpu,
        max_len,
        page_size,
    )
    for actual, expected in zip(eager, (expected_q, expected_scale, expected_logits)):
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    assert torch.equal(cache.cpu(), expected_cache)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        run()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
    torch.cuda.current_stream().wait_stream(stream)
    # New values and known initial cache contents prove current replay writes.
    x_cpu.neg_()
    k_cpu.neg_()
    q_cpu.neg_()
    lengths_cpu[0] = 1
    x.copy_(x_cpu)
    k.copy_(k_cpu)
    q.copy_(q_cpu)
    lengths.copy_(lengths_cpu)
    cache.copy_(initial_cache)
    for output in captured:
        output.fill_(17)
    expected_cache = cache_store_reference(
        k_cpu, initial_cache.clone(), slots_cpu, page_size
    )
    expected_q, expected_scale = quantize_reference(x_cpu)
    expected_logits = paged_logits_reference(
        q_cpu,
        expected_cache,
        weights_cpu,
        table_cpu,
        row_cpu,
        lengths_cpu,
        max_len,
        page_size,
    )
    graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip(
        captured, (expected_q, expected_scale, expected_logits)
    ):
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    assert torch.equal(cache.cpu(), expected_cache)
