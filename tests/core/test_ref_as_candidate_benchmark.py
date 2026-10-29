# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run in a configured FlagGems runtime with --confcutdir=tests/core."""

import pytest

from benchmark import base, conftest
from flag_gems.testing.reference import reference_test_mode


def forbidden(*args, **kwargs):
    raise AssertionError("Gems dispatch/candidate must not run")


@pytest.fixture
def config(monkeypatch):
    config = conftest.BenchConfig()
    config.ref_as_candidate = True
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(base.flag_gems, "use_gems", forbidden)
    with reference_test_mode(True, "benchmark/test_probe.py::test_probe"):
        yield config


def test_reference_is_timed_on_both_sides(config, monkeypatch):
    calls = []
    bench = base.Benchmark(
        "example", torch_op=lambda value: calls.append(value), gems_op=forbidden
    )
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]

    def timer(op, *args, **kwargs):
        op(*args, **kwargs)
        return 1.0

    monkeypatch.setattr(bench, "get_latency", timer)
    metric = bench._measure_input((2,))
    assert calls == [2, 2]
    assert metric.candidate_source == "reference"
    assert metric.latency_base == metric.latency == metric.speedup == 1.0


def test_missing_reference_is_an_error(config):
    with pytest.raises(ValueError, match="callable torch_op reference"):
        base.Benchmark("missing", torch_op=None)


@pytest.mark.parametrize(
    "method", ["run", "get_latency", "_measure_input", "_candidate_call"]
)
def test_custom_execution_must_explicitly_support_contract(config, method):
    custom = type("Custom", (base.Benchmark,), {method: forbidden})
    with pytest.raises(ValueError, match="custom benchmark execution"):
        custom("example", torch_op=lambda args: args)


def test_candidate_timer_cannot_be_omitted(config, monkeypatch):
    bench = base.Benchmark("example", torch_op=sum)
    monkeypatch.setattr(bench, "init_user_config", lambda: None)
    bench.to_bench_metrics = ["latency_base"]
    with pytest.raises(ValueError, match="candidate latency metric"):
        bench.run()


def test_timer_must_actually_invoke_reference(config, monkeypatch):
    bench = base.Benchmark("example", torch_op=lambda value: value)
    bench.to_bench_metrics = ["latency"]
    monkeypatch.setattr(bench, "get_latency", lambda *args, **kwargs: 1.0)
    with pytest.raises(pytest.fail.Exception, match="did not invoke"):
        bench._measure_input((2,))
