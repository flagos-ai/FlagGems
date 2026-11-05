# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable ties, signed-zero bits and refreshed-input graph replay."""

import importlib

import numpy as np
import pytest
import torch


@pytest.mark.topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("width", [4096, 16384, 34880])
@torch.inference_mode()
def test_stable_k2048_numerics_and_mutable_graph(width, monkeypatch):
    import flag_gems
    from flag_gems.runtime import backend
    from flag_gems.utils.triton_version_utils import HAS_TLE

    if not HAS_TLE or flag_gems.vendor_name != "nvidia":
        pytest.skip("optional NVIDIA TLE specialization")
    # The backend loader imports vendor packages under their canonical short name.
    vendor_topk = dict(backend.get_ops(flag_gems.vendor_name))["topk"]
    module = importlib.import_module(vendor_topk.__module__)
    monkeypatch.setattr(module, "_TOPK_RADIX_STABLE_K2048_ENABLED", True)
    assert flag_gems.topk is module.topk
    rng = np.random.default_rng(width)
    cpu = rng.standard_normal((3, width), dtype=np.float32)
    cpu[0, :2500] = 0.0
    cpu[0, 1:2500:2] = -0.0
    cpu[1, :2] = [np.inf, -np.inf]
    storage = torch.empty((3, width * 2), device="cuda")
    x = storage[:, ::2]
    x.copy_(torch.from_numpy(cpu))

    def check(actual, source):
        order = np.lexsort(
            (np.broadcast_to(np.arange(width), source.shape), -source), axis=-1
        )[:, :2048]
        expected = np.take_along_axis(source, order, axis=1)
        np.testing.assert_array_equal(actual[1].cpu().numpy(), order)
        np.testing.assert_array_equal(
            actual[0].cpu().numpy().view(np.uint32), expected.view(np.uint32)
        )

    check(flag_gems.topk(x, 2048), cpu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        flag_gems.topk(x, 2048)
        with torch.cuda.graph(graph, stream=stream):
            captured = flag_gems.topk(x, 2048)
    torch.cuda.current_stream().wait_stream(stream)
    for changed in (-cpu, cpu):
        x.copy_(torch.from_numpy(changed))
        captured[0].fill_(17)
        captured[1].fill_(-17)
        graph.replay()
        torch.cuda.synchronize()
        check(captured, changed)
