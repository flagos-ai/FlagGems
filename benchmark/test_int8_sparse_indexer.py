# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete primitive latency against equivalent Torch tensor decompositions.

These warm component benchmarks do not reproduce archived native timings.
"""

import pytest
import torch

from flag_gems.fused.int8_sparse_indexer import (
    int8_indexer_cache_store,
    int8_indexer_paged_logits,
    int8_indexer_quantize,
)

from . import base


def torch_quantize(x):
    xf = x.float()
    scale = torch.exp2(
        torch.ceil(torch.log2(xf.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127.0))
    )
    return (xf / scale).clamp(-127, 127).round().to(torch.int8), scale


def torch_cache_store_all_valid(k, cache, slots, page_size):
    # Benchmark fixtures use all-valid unique slots. Negative-slot numerical
    # behavior belongs to the functional tests, not a CPU-filtered timing arm.
    q, s = torch_quantize(k[: slots.numel()])
    pages = cache.view(cache.shape[0], page_size * 132)
    values = (
        pages[:, : page_size * 128]
        .view(torch.int8)
        .view(cache.shape[0], page_size, 128)
    )
    scales = pages[:, page_size * 128 :].view(torch.float32)
    physical, offset = slots // page_size, slots % page_size
    values[physical, offset] = q
    scales[physical, offset] = s[:, 0]


def torch_paged_logits(
    q, cache, weights, table, row_batch, lengths, max_len, page_size
):
    pages = cache.view(cache.shape[0], page_size * 132)
    values = (
        pages[:, : page_size * 128]
        .view(torch.int8)
        .view(cache.shape[0], page_size, 128)
    )
    scales = pages[:, page_size * 128 :].view(torch.float32)
    columns = torch.arange(max_len, device=q.device)
    selected_pages = table[
        row_batch.clamp_min(0)[:, None], columns[None, :] // page_size
    ]
    valid = (
        (row_batch[:, None] >= 0)
        & (columns[None, :] < lengths[:, None])
        & (selected_pages >= 0)
    )
    safe_pages = selected_pages.clamp_min(0).long()
    keys = values[safe_pages, columns[None, :] % page_size]
    key_scales = scales[safe_pages, columns[None, :] % page_size]
    weighted = []
    for head in range(q.shape[1]):
        # Integer elementwise product/reduction avoids unsupported CUDA INT32
        # GEMM and any TF32/FP8 precision substitution in the reference arm.
        dot = (q[:, head, None, :].int() * keys.int()).sum(-1, dtype=torch.int32)
        weighted.append(dot.clamp_min(0).float() * weights[:, head, None])
    result = torch.stack(weighted, dim=1).sum(1) * key_scales
    return result.masked_fill(~valid, -float("inf"))


class QuantizeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 32, 128), (4, 32, 128), (64, 32, 128)]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield (torch.randn(shape, dtype=dtype, device=self.device),)


class CacheStoreBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 16), (4, 64), (64, 64)]

    def get_input_iter(self, dtype):
        for rows, page_size in self.shapes:
            pages = (rows + page_size - 1) // page_size
            storage = torch.randn((rows, 135), dtype=dtype, device=self.device)
            cache = torch.zeros(
                (pages, page_size, 132), dtype=torch.uint8, device=self.device
            )
            slots = torch.arange(rows, dtype=torch.int64, device=self.device)
            yield storage[:, :128], cache, slots, page_size


class PagedLogitsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 32, 16, 64), (4, 32, 64, 256), (4, 32, 64, 1024)]

    def get_input_iter(self, dtype):
        for rows, heads, page_size, max_len in self.shapes:
            pages = (max_len + page_size - 1) // page_size
            q = torch.randint(
                -8, 9, (rows, heads, 128), dtype=torch.int8, device=self.device
            )
            cache = torch.zeros(
                (pages, page_size, 132), dtype=torch.uint8, device=self.device
            )
            view = cache.view(pages, page_size * 132)
            view[:, : page_size * 128] = torch.randint(
                0, 255, (pages, page_size * 128), dtype=torch.uint8, device=self.device
            )
            view[:, page_size * 128 :].view(torch.float32).fill_(0.25)
            weights = torch.randn(
                (rows, heads), dtype=torch.float32, device=self.device
            )
            table = torch.arange(pages, dtype=torch.int32, device=self.device).view(
                1, pages
            )
            mapping = torch.zeros(rows, dtype=torch.int32, device=self.device)
            lengths = torch.full(
                (rows,), max_len, dtype=torch.int32, device=self.device
            )
            yield q, cache, weights, table, mapping, lengths, max_len, page_size


@pytest.mark.int8_indexer_quantize
@torch.no_grad()
def test_int8_indexer_quantize():
    if not torch.cuda.is_available():
        pytest.skip("GPU component benchmark")
    bench = QuantizeBenchmark(
        op_name="int8_indexer_quantize",
        torch_op=torch_quantize,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(int8_indexer_quantize)
    bench.run()


@pytest.mark.int8_indexer_cache_store
@torch.no_grad()
def test_int8_indexer_cache_store():
    if not torch.cuda.is_available():
        pytest.skip("GPU component benchmark")
    bench = CacheStoreBenchmark(
        op_name="int8_indexer_cache_store",
        torch_op=torch_cache_store_all_valid,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(int8_indexer_cache_store)
    bench.run()


@pytest.mark.int8_indexer_paged_logits
@torch.no_grad()
def test_int8_indexer_paged_logits():
    if not torch.cuda.is_available():
        pytest.skip("GPU component benchmark")
    bench = PagedLogitsBenchmark(
        op_name="int8_indexer_paged_logits",
        torch_op=torch_paged_logits,
        dtypes=[torch.int8],
    )
    bench.set_gems(int8_indexer_paged_logits)
    bench.run()
