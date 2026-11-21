# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""MetaX index_select: native comparison plus explicit host-wrapper timing."""

import json
import statistics
import time

import pytest
import torch

import flag_gems

from . import base

pytestmark = [
    pytest.mark.index_select,
    pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX wrapper regression"
    ),
]


class IndexSelectHostBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(64, 8), (2048, 8), (8192, 8)]

    def get_input_iter(self, dtype):
        for n, width in self.shapes:
            x = torch.arange(n * width, device=self.device).reshape(n, width).to(dtype)
            index = torch.arange(n, device=self.device, dtype=torch.int64)
            yield x, 0, index


def test_metax_index_select_native_benchmark():
    bench = IndexSelectHostBenchmark(
        op_name="index_select", torch_op=torch.index_select, dtypes=[torch.float32]
    )
    bench.set_gems(flag_gems.index_select)
    bench.run()


@pytest.mark.parametrize("n", [64, 2048, 8192])
def test_metax_index_select_host_wrapper(n):
    x = torch.arange(n * 8, device="cuda", dtype=torch.float32).reshape(n, 8)
    index = torch.arange(n, device="cuda", dtype=torch.int64)

    def previous_wrapper():
        # Reproduce exactly the removed assertion's host behavior. Its generator
        # is truthy and its outer iterator creates tensor views; no values are checked.
        assert ((i >= 0 and i < x.size(0)) for i in index), "Index out of range"
        return flag_gems.index_select(x, 0, index)

    def current_wrapper():
        return flag_gems.index_select(x, 0, index)

    for _ in range(4):
        current_wrapper()
        previous_wrapper()
    values = {"previous": [], "current": []}
    for key in ["previous", "current", "current", "previous"]:
        fn = previous_wrapper if key == "previous" else current_wrapper
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(32):
            fn()
        torch.cuda.synchronize()
        values[key].append((time.perf_counter() - start) * 1000 / 32)
    result = {k: statistics.median(v) for k, v in values.items()}
    print(
        json.dumps(
            {
                "rows": n,
                "unit": "ms/wrapper including host and GPU drain",
                "timings": result,
                "speedup": result["previous"] / result["current"],
            }
        ),
        flush=True,
    )
