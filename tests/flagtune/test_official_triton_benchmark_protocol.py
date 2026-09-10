# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import triton.testing as triton_testing

from flag_gems.flagtune.runtime import _benchmark_protocol as benchmark_module


class _FakeDriver:
    def __init__(self):
        self.observed = {}

    def get_benchmarker(self):
        def benchmark(kernel_call, **kwargs):
            self.observed = dict(kwargs)
            kernel_call()
            return [2.0, 1.8, 2.2]

        return benchmark


def _driver_for(module_name):
    driver_type = type("FakeDriver", (_FakeDriver,), {"__module__": module_name})
    return driver_type()


def test_official_triton_replay_uses_fixed_retry_count(monkeypatch):
    """Do not pass FlagTree's extra keyword to official Triton 3.6."""
    active = _driver_for("triton.backends.nvidia.driver")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    observed = {}

    def official_do_bench_cudagraph(kernel_call, rep, quantiles):
        observed.update(rep=rep, quantiles=quantiles)
        kernel_call()
        return [1.0, 0.8, 1.2]

    monkeypatch.setattr(
        triton_testing,
        "do_bench_cudagraph",
        official_do_bench_cudagraph,
    )
    launches = []

    with pytest.warns(RuntimeWarning, match="fixed replay count of 10"):
        resolved = benchmark_module.resolve_benchmarker(
            "replay",
            warmup_ms=25,
            measurement_ms=100,
            n_retries=3,
        )
    result = resolved.benchmark(
        lambda: launches.append(True),
        (0.5, 0.2, 0.8),
    )

    assert result == [1.0, 0.8, 1.2]
    assert launches == [True]
    assert observed == {
        "rep": 10.0,
        "quantiles": (0.5, 0.2, 0.8),
    }
    assert resolved.protocol.n_retries == 10
    assert resolved.protocol.cache_key() == (
        "triton_cuda_graph_replay_v1",
        25,
        100,
        10,
        10.0,
    )


def test_official_triton_unsupported_replay_falls_back_to_event(monkeypatch):
    """Keep non-NVIDIA/AMD official backends usable through event timing."""
    active = _driver_for("triton.backends.example.driver")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    launches = []

    with pytest.warns(RuntimeWarning, match="falling back to event"):
        resolved = benchmark_module.resolve_benchmarker(
            "replay",
            warmup_ms=5,
            measurement_ms=20,
            n_retries=10,
        )
    result = resolved.benchmark(
        lambda: launches.append(True),
        (0.5, 0.2, 0.8),
    )

    assert result == [2.0, 1.8, 2.2]
    assert launches == [True]
    assert active.observed == {
        "warmup": 5,
        "rep": 20,
        "quantiles": (0.5, 0.2, 0.8),
    }
    assert resolved.protocol.resolved_mode is benchmark_module.BenchmarkMode.EVENT
