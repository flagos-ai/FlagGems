# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable K2048 specialization with a complete stable TopK baseline."""

import importlib

import pytest
import torch

import flag_gems

from . import base


def reference(x, count):
    order = torch.argsort(x, dim=-1, descending=True, stable=True)[:, :count]
    return x.gather(1, order), order


class StableTopKBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(4, 4096), (256, 16384), (1024, 34816)]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield torch.randn(shape, dtype=dtype, device=flag_gems.device), 2048


@pytest.mark.topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@torch.inference_mode()
def test_topk_k2048_stable_perf(monkeypatch):
    from flag_gems.runtime import backend
    from flag_gems.utils.triton_version_utils import HAS_TLE

    if not HAS_TLE or flag_gems.vendor_name != "nvidia":
        pytest.skip("optional NVIDIA TLE specialization")
    # Resolve the same module instance used by the public operator registrar.
    vendor_topk = dict(backend.get_ops(flag_gems.vendor_name))["topk"]
    module = importlib.import_module(vendor_topk.__module__)
    monkeypatch.setattr(module, "_TOPK_RADIX_STABLE_K2048_ENABLED", True)
    assert flag_gems.topk is module.topk
    bench = StableTopKBenchmark(
        op_name="topk", torch_op=reference, dtypes=[torch.float32]
    )
    bench.set_gems(flag_gems.topk)
    bench.run()
