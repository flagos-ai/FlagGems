# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.index_put,
    pytest.mark.skipif(
        flag_gems.runtime.device.vendor_name != "metax", reason="MetaX regression"
    ),
]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_index_put_large_paged_stride_and_graph(index_dtype):
    # Qwen3.8 TP8 FP32 GDN pages: one row is 98304 values with a 102400-element stride.
    # int32 page 20972 times that stride crosses 2**31; a dense small tensor cannot test this.
    shape = (22002, 6, 128, 128)
    x = torch.empty_strided(
        shape, (102400, 16384, 128, 1), device="cuda", dtype=torch.float32
    )
    rows = [0, 20971, 20972, 22000]
    guard = [1, 20970, 20973, 22001]
    for r in guard:
        x[r].fill_(-7)
    idx = torch.tensor(rows, device="cuda", dtype=index_dtype)
    value = (
        torch.arange(len(rows), device="cuda", dtype=torch.float32)[:, None, None, None]
        .expand(4, 6, 128, 128)
        .contiguous()
    )
    flag_gems.index_put_(x, [idx], value, False)
    for i, r in enumerate(rows):
        torch.testing.assert_close(x[r], value[i], rtol=0, atol=0)
    for r in guard:
        assert torch.all(x[r] == -7)
    for _ in range(3):
        flag_gems.index_put_(x, [idx], value, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flag_gems.index_put_(x, [idx], value, False)
    idx.copy_(torch.tensor([2, 20974, 20975, 21999], device="cuda", dtype=index_dtype))
    value.add_(5)
    graph.replay()
    for i, r in enumerate([2, 20974, 20975, 21999]):
        torch.testing.assert_close(x[r], value[i], rtol=0, atol=0)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_index_put_transposed_and_accumulate(index_dtype, accumulate):
    x = torch.zeros(11, 7, device="cuda").t()
    ref = x.clone()
    idx = torch.tensor([1, 1, 5], device="cuda", dtype=index_dtype)
    value = torch.ones(3, 11, device="cuda")
    ref.index_put_((idx.to(torch.int64),), value, accumulate=accumulate)
    flag_gems.index_put_(x, [idx], value, accumulate)
    torch.testing.assert_close(x, ref, rtol=0, atol=0)
