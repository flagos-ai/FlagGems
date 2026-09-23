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

"""Candidate-only runner tests; inputs and device operations are host fakes.

Run with --confcutdir=tests/core in an existing FlagGems runtime environment.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _pytest.config.argparsing import Parser

from benchmark import base, conftest
from benchmark.cases import BenchmarkCaseSpec


def fail(*args, **kwargs):
    raise AssertionError("unexpected reference, timing or profiling call")


@pytest.fixture
def runner(monkeypatch):
    config = conftest.BenchConfig()
    config.preflight_only = True
    config.current_nodeid = "benchmark/test_example.py::test_example"
    events = []
    candidate = lambda value: events.append(("candidate", value))
    config.override_registry = SimpleNamespace(get_override=lambda name: candidate)
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(conftest, "Config", config)
    monkeypatch.setattr(
        base,
        "torch_device_fn",
        SimpleNamespace(synchronize=lambda: events.append("sync")),
    )
    monkeypatch.setattr(base, "profile_capture_scope", fail)
    bench = base.Benchmark("example", torch_op=fail, gems_op=fail)
    monkeypatch.setattr(bench, "init_user_config", lambda: None)
    monkeypatch.setattr(bench, "supports_cases", lambda: True)
    cases = [BenchmarkCaseSpec(f"case-{i}", i, "float32", {}) for i in range(3)]
    monkeypatch.setattr(bench, "_collect_cases", lambda: cases)
    monkeypatch.setattr(bench, "build_inputs", lambda case: (case.ordinal,))
    monkeypatch.setattr(bench, "unpack_to_args_kwargs", lambda args: (args, {}))
    monkeypatch.setattr(bench, "get_latency", fail)
    return bench, config, events


@pytest.mark.parametrize("selection,expected", [(None, [0, 1, 2]), (["case-1"], [1])])
def test_preflight_runs_candidate_once_per_selected_case(runner, selection, expected):
    bench, config, events = runner
    assert bench.run(case_ids=selection) == [f"case-{i}" for i in expected]
    assert events == [item for i in expected for item in (("candidate", i), "sync")]
    assert config.executed_case_ids == {f"case-{i}" for i in expected}
    assert all(
        r["override"] and r["count"] == 1 and r["status"] == "passed"
        for r in config.preflight_records
    )
    assert all("latency" not in r for r in config.preflight_records)


@pytest.mark.parametrize("direct_gems", [True, False])
def test_preflight_without_override_preserves_gems_dispatch(
    monkeypatch, runner, direct_gems
):
    bench, config, events = runner
    config.override_registry = None
    op = lambda value: events.append(("gems", value))
    bench.gems_op = op if direct_gems else None
    bench.torch_op = fail if direct_gems else op

    @contextmanager
    def use_gems(**kwargs):
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(base.flag_gems, "use_gems", fail if direct_gems else use_gems)
    bench.run(case_ids=["case-0"])
    assert events == (
        [("gems", 0), "sync"] if direct_gems else ["enter", ("gems", 0), "sync", "exit"]
    )
    assert config.preflight_records[0]["override"] is False


@pytest.mark.parametrize("phase", ["build", "candidate", "sync"])
def test_preflight_failure_cannot_be_recorded_as_passed(monkeypatch, runner, phase):
    bench, config, events = runner

    def broken(*args):
        raise RuntimeError("intentional " + phase)

    if phase == "build":
        monkeypatch.setattr(bench, "build_inputs", broken)
    elif phase == "candidate":
        config.override_registry.get_override = lambda name: broken
    else:
        monkeypatch.setattr(base.torch_device_fn, "synchronize", broken)
    with pytest.raises(RuntimeError, match="intentional"):
        bench.run()
    assert not config.executed_case_ids
    assert len(config.preflight_records) == 1
    assert config.preflight_records[0]["status"] == "failed"
    assert config.preflight_records[0]["count"] == (0 if phase == "build" else 1)


def test_preflight_rejects_legacy_and_empty_case_plans(monkeypatch, runner):
    bench, _, _ = runner
    monkeypatch.setattr(bench, "supports_cases", lambda: False)
    with pytest.raises(ValueError, match="does not support"):
        bench.run()
    monkeypatch.setattr(bench, "supports_cases", lambda: True)
    monkeypatch.setattr(bench, "_collect_cases", lambda: [])
    with pytest.raises(ValueError, match="no preflight cases"):
        bench.run()


def test_preflight_report_replaces_stale_data(monkeypatch, tmp_path, runner):
    bench, config, _ = runner
    output = tmp_path / "preflight.json"
    output.write_text('{"stale": true}')
    monkeypatch.setattr(conftest, "REPORT_FILE", str(output))
    config.record_json = True
    bench.run(case_ids=["case-0"])
    conftest.pytest_terminal_summary(None, 0, None)
    result = json.loads(output.read_text())
    assert result == {
        "schema_version": "flaggems.preflight/v1",
        "records": config.preflight_records,
    }


def test_unknown_or_unexecuted_case_and_all_skip_cannot_pass(runner):
    bench, config, _ = runner
    config.case_ids = ["missing"]
    bench.run()
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
    )
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    config.preflight_records = [{"status": "failed"}]
    session.exitstatus = pytest.ExitCode.OK
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    session.exitstatus = pytest.ExitCode.INTERRUPTED
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.INTERRUPTED
    config.case_ids = None
    session.exitstatus = pytest.ExitCode.OK
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


@pytest.mark.parametrize(
    "other", [["--profile-only"], ["--list-cases"], ["--query"], ["--parallel", "2"]]
)
def test_preflight_rejects_conflicting_cli_modes(monkeypatch, other):
    parser = Parser()
    conftest.pytest_addoption(parser)
    options = parser.parse(["--preflight-only", *other])
    config = SimpleNamespace(
        option=options,
        addinivalue_line=lambda *args: None,
        getini=lambda key: [],
        getoption=lambda key: getattr(options, key.lstrip("-").replace("-", "_")),
    )
    with pytest.raises(pytest.UsageError, match="--preflight-only"):
        conftest.pytest_configure(config)


def test_profile_still_uses_shared_candidate_resolution(runner, monkeypatch):
    bench, config, events = runner
    config.preflight_only = False
    config.profile_only = True
    config.case_ids = ["case-0"]
    config.profile_warmup = config.profile_iterations = 1

    @contextmanager
    def capture(**kwargs):
        events.append("capture")
        yield
        events.append("stop")

    monkeypatch.setattr(base, "profile_capture_scope", capture)
    bench.run()
    assert events == [
        ("candidate", 0),
        "sync",
        "capture",
        ("candidate", 0),
        "sync",
        "stop",
    ]
    assert config.preflight_records == []


def test_benchmark_times_reference_and_live_override_separately(monkeypatch, runner):
    from dataclasses import asdict

    bench, config, events = runner
    counts = {"flag_gems.example": 0}

    def candidate(value):
        counts["flag_gems.example"] += 1
        events.append("candidate")

    def reference(value):
        events.append("reference")

    config.override_registry = SimpleNamespace(
        get_override=lambda name: candidate, call_counts=lambda: dict(counts)
    )
    bench.torch_op = reference
    bench.gems_op = fail
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(bench, "record_shapes", lambda *args, **kwargs: ())

    def latency(op, *args, **kwargs):
        op(*args, **kwargs)
        return 2.0 if op is reference else 1.0

    monkeypatch.setattr(bench, "get_latency", latency)
    metric = bench._measure_input((1,), case_id="case-0")
    assert events == ["reference", "candidate"]
    assert asdict(metric)["candidate_source"] == "override"
    assert metric.latency == 1.0 and metric.latency_base == 2.0


def test_benchmark_cannot_claim_an_unused_override(monkeypatch, runner):
    bench, config, _ = runner
    config.override_registry.call_counts = lambda: {"flag_gems.example": 0}
    bench.to_bench_metrics = ["latency"]
    monkeypatch.setattr(bench, "record_shapes", lambda *args, **kwargs: ())
    monkeypatch.setattr(bench, "get_latency", lambda *args, **kwargs: 1.0)
    with pytest.raises(pytest.fail.Exception, match="did not invoke"):
        bench._measure_input((1,), case_id="case-0")
