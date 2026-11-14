# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU parity for NaN/ties, masked lanes and both reduction stages."""
import pytest
import torch

import flag_gems


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize(
    "shape,dim,strided",
    [
        ((3, 160), -1, False),
        ((2, 65537), -1, False),
        ((2, 160, 3), 1, False),
        ((1, 4097, 17), 1, False),
        ((2, 4097), None, False),
        ((3, 321), -1, True),
    ],
)
@pytest.mark.parametrize(
    "mode", ["ties", "minimum", "random", "all_nan", "mixed_nan", "infinity"]
)
def test_argmax_valid_lane_and_nan_contract(dtype, shape, dim, strided, mode):
    if not dtype.is_floating_point and mode in ("all_nan", "mixed_nan", "infinity"):
        pytest.skip("Non-finite values require a floating dtype")
    torch.manual_seed(42)
    x = torch.randint(-20, 20, shape, dtype=torch.int64).to(dtype)
    axis = x.ndim - 1 if dim is None else dim % x.ndim
    if mode == "ties":
        x.zero_()
    elif mode == "minimum":
        x.fill_(-float("inf") if dtype.is_floating_point else torch.iinfo(dtype).min)
    elif mode == "all_nan":
        x.fill_(float("nan"))
    elif mode in ("mixed_nan", "infinity"):
        x.zero_()
        selector = [slice(None)] * x.ndim
        for position in (1, shape[axis] - 2):
            selector[axis] = position
            x[tuple(selector)] = float("nan") if mode == "mixed_nan" else float("inf")
    if strided:
        x = x[..., ::2]
    expected = torch.argmax(x, dim=dim, keepdim=True)
    gpu = x.to(device=flag_gems.device)
    if strided:
        backing = torch.empty(
            (*x.shape[:-1], x.shape[-1] * 2), dtype=dtype, device=flag_gems.device
        )
        backing[..., ::2].copy_(gpu)
        gpu = backing[..., ::2]
    actual = flag_gems.argmax(gpu, dim=dim, keepdim=True)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph contract")
def test_argmax_changing_graph_masked_last_block(dtype):
    x = torch.zeros(2, 4097, device="cuda", dtype=dtype)
    flag_gems.argmax(x, dim=None, keepdim=True)
    flag_gems.argmax(x, dim=None, keepdim=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = flag_gems.argmax(x, dim=None, keepdim=True)
    for mode in ["minimum", "last", "nan"]:
        x.fill_(-float("inf"))
        if mode == "last":
            x[-1, -1] = 10
        elif mode == "nan":
            x[0, 17] = float("nan")
        graph.replay()
        expected = torch.argmax(x.cpu(), dim=None, keepdim=True)
        assert torch.equal(output.cpu(), expected)
