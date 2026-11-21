# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import inspect

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.index_select,
    pytest.mark.skipif(
        flag_gems.runtime.device.vendor_name != "metax", reason="MetaX regression"
    ),
]


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64]
)
@pytest.mark.parametrize(
    "shape,dim,n", [((16, 32), 0, 17), ((4, 19), -1, 5), ((3, 4, 7), 1, 9)]
)
def test_index_select_does_not_iterate_indices(monkeypatch, dtype, shape, dim, n):
    x = (
        torch.arange(torch.tensor(shape).prod().item(), device="cuda")
        .reshape(shape)
        .to(dtype)
    )
    idx = torch.arange(n, device="cuda", dtype=torch.int64) % shape[dim]
    expected = torch.index_select(x, dim, idx)
    old_iter = torch.Tensor.__iter__

    def checked_iter(self):
        if self is idx:
            raise AssertionError("index_select must not unbind device indices on host")
        return old_iter(self)

    monkeypatch.setattr(torch.Tensor, "__iter__", checked_iter)
    assert "_metax" in inspect.getfile(flag_gems.index_select)
    actual = flag_gems.index_select(x, dim, idx)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_scalar_index_and_dynamic_graph(index_dtype):
    x = torch.arange(128, device="cuda", dtype=torch.float32).reshape(16, 8)
    scalar = torch.tensor(7, device="cuda", dtype=index_dtype)
    torch.testing.assert_close(
        flag_gems.index_select(x, 0, scalar), x[7:8], rtol=0, atol=0
    )
    idx = torch.tensor([2, 7, 12], device="cuda", dtype=index_dtype)
    for _ in range(3):
        flag_gems.index_select(x, 0, idx)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flag_gems.index_select(x, 0, idx)
    idx.copy_(torch.tensor([1, 6, 15], device="cuda", dtype=index_dtype))
    x.add_(3)
    graph.replay()
    torch.testing.assert_close(out, torch.index_select(x, 0, idx), rtol=0, atol=0)
