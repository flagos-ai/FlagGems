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
import torch
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
    config.profile_hook = fail
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

    config.profile_hook = capture
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


@pytest.mark.parametrize("provider", [False, True])
@pytest.mark.parametrize("candidate_fails", [False, True])
def test_pytest_profile_plugin_dispatch(runner, provider, candidate_fails):
    bench, config, events = runner
    config.preflight_only = False
    config.profile_only = True
    config.case_ids = ["case-0"]
    config.profile_warmup = 1
    config.profile_iterations = 2
    manager = pytest.PytestPluginManager()
    conftest.pytest_addhooks(manager)

    class Plugin:
        @pytest.hookimpl
        @contextmanager
        def pytest_flaggems_profile_scope(self, backend, case_id):
            assert backend == base.vendor_name
            assert case_id == "case-0"
            events.append("capture")
            try:
                yield
            finally:
                events.append("stop")

    if provider:
        manager.register(Plugin())
    config.profile_hook = manager.hook.pytest_flaggems_profile_scope
    calls = []

    def candidate(value):
        calls.append(value)
        events.append(("candidate", value))
        if candidate_fails and len(calls) == 2:
            raise ValueError("candidate failed inside capture")

    config.override_registry = SimpleNamespace(get_override=lambda name: candidate)
    if candidate_fails:
        with pytest.raises(ValueError, match="inside capture"):
            bench.run()
        assert config.executed_case_ids == set()
        assert events == [
            ("candidate", 0),
            "sync",
            *(["capture"] if provider else []),
            ("candidate", 0),
            *(["stop"] if provider else []),
        ]
    else:
        bench.run()
        assert config.executed_case_ids == {"case-0"}
        assert events == [
            ("candidate", 0),
            "sync",
            *(["capture"] if provider else []),
            ("candidate", 0),
            ("candidate", 0),
            "sync",
            *(["stop"] if provider else []),
        ]


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


def _fake_profile(events):
    class FakeProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def key_averages(self):
            return events

    return lambda: FakeProfiler()


def test_profile_invokes_kernel_returns_false_for_cpu_only(monkeypatch, runner):
    from torch.autograd import DeviceType

    bench, _, _ = runner
    cpu_only = [SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=0)]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(cpu_only))
    assert bench._profile_invokes_kernel(lambda: None) is False


def test_profile_invokes_kernel_returns_true_for_device_event(monkeypatch, runner):
    from torch.autograd import DeviceType

    bench, _, _ = runner
    events = [
        SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=0),
        SimpleNamespace(device_type=DeviceType.CUDA, self_device_time_total=2.5),
    ]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(events))
    assert bench._profile_invokes_kernel(lambda: None) is True


def test_base_without_device_kernel_is_rejected(monkeypatch, runner):
    from torch.autograd import DeviceType

    bench, _, _ = runner
    cpu_only = [SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=0)]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(cpu_only))
    with pytest.raises(pytest.skip.Exception, match="did not invoke any device kernel"):
        bench._verify_base_invokes_kernel(lambda *a, **k: None, (1,), {})


def test_base_with_device_kernel_is_accepted(monkeypatch, runner):
    from torch.autograd import DeviceType

    bench, _, _ = runner
    events = [
        SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=0),
        SimpleNamespace(device_type=DeviceType.CUDA, self_device_time_total=2.5),
    ]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(events))
    bench._verify_base_invokes_kernel(lambda *a, **k: None, (1,), {})


def test_verify_base_invokes_kernel_caches_result(monkeypatch, runner):
    """After the first successful check, subsequent calls skip profiling."""
    from torch.autograd import DeviceType

    bench, _, _ = runner
    device_events = [
        SimpleNamespace(device_type=DeviceType.CUDA, self_device_time_total=1.0)
    ]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(device_events))
    # First call profiles and caches
    bench._verify_base_invokes_kernel(lambda *a, **k: None, (1,), {})
    assert bench._base_kernel_verified is True
    # Replace profiler with one that would fail — cached result should skip it
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile([]))
    bench._verify_base_invokes_kernel(lambda *a, **k: None, (1,), {})  # no error


def test_verify_base_invokes_kernel_caches_failure(monkeypatch, runner):
    """A cached False result raises on subsequent calls without re-profiling."""
    from torch.autograd import DeviceType

    bench, _, _ = runner
    cpu_only = [SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=0)]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(cpu_only))
    with pytest.raises(pytest.skip.Exception, match="did not invoke any device kernel"):
        bench._verify_base_invokes_kernel(lambda *a, **k: None, (1,), {})
    assert bench._base_kernel_verified is False
    # Second call should also raise from cache, even with a different profiler
    monkeypatch.setattr(
        base.torch.profiler,
        "profile",
        _fake_profile(
            [SimpleNamespace(device_type=DeviceType.CUDA, self_device_time_total=1.0)]
        ),
    )
    with pytest.raises(pytest.skip.Exception, match="did not invoke any device kernel"):
        bench._verify_base_invokes_kernel(lambda *a, **k: None, (1,), {})


def test_benchmark_cannot_claim_an_unused_override(monkeypatch, runner):
    bench, config, _ = runner
    config.override_registry.call_counts = lambda: {"flag_gems.example": 0}
    bench.to_bench_metrics = ["latency"]
    monkeypatch.setattr(bench, "record_shapes", lambda *args, **kwargs: ())
    monkeypatch.setattr(bench, "get_latency", lambda *args, **kwargs: 1.0)
    with pytest.raises(pytest.fail.Exception, match="did not invoke"):
        bench._measure_input((1,), case_id="case-0")


# ---------------------------------------------------------------------------
# --kernel-check tests
# ---------------------------------------------------------------------------


@pytest.fixture
def kc_runner(monkeypatch, runner):
    """Extend the preflight runner with kernel_check mode active."""
    bench, config, events = runner
    config.kernel_check = True
    config.preflight_only = False
    config.kernel_check_records = []
    # torch_op is set to `fail` by the runner fixture; replace it with a no-op
    # so _benchmark_callable can build its lambda without raising.
    bench.torch_op = lambda *a, **k: None
    return bench, config, events


def test_kernel_check_case_invokes_kernel_passes(monkeypatch, kc_runner):
    from torch.autograd import DeviceType

    bench, config, _ = kc_runner
    device_events = [
        SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=0),
        SimpleNamespace(device_type=DeviceType.CUDA, self_device_time_total=3.0),
    ]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(device_events))
    executed = bench._run_kernel_check_cases(case_ids=["case-0"])
    assert executed == ["case-0"]
    assert len(config.kernel_check_records) == 1
    rec = config.kernel_check_records[0]
    assert rec["invoked_kernel"] is True
    assert rec["status"] == "passed"
    assert "case-0" in config.executed_case_ids


def test_kernel_check_case_no_kernel_records_false(monkeypatch, kc_runner):
    from torch.autograd import DeviceType

    bench, config, _ = kc_runner
    cpu_only = [SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=0)]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(cpu_only))
    executed = bench._run_kernel_check_cases(case_ids=["case-0"])
    assert executed == ["case-0"]
    rec = config.kernel_check_records[0]
    assert rec["invoked_kernel"] is False
    assert rec["status"] == "passed"


def test_kernel_check_all_cases_run_by_default(monkeypatch, kc_runner):
    from torch.autograd import DeviceType

    bench, config, _ = kc_runner
    device_events = [
        SimpleNamespace(device_type=DeviceType.CUDA, self_device_time_total=1.0)
    ]
    monkeypatch.setattr(base.torch.profiler, "profile", _fake_profile(device_events))
    executed = bench._run_kernel_check_cases()
    assert executed == ["case-0", "case-1", "case-2"]
    assert len(config.kernel_check_records) == 3
    assert all(r["invoked_kernel"] for r in config.kernel_check_records)


def test_kernel_check_failure_records_error(monkeypatch, kc_runner):
    bench, config, _ = kc_runner

    class BrokenProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def key_averages(self):
            raise RuntimeError("profiler exploded")

    monkeypatch.setattr(base.torch.profiler, "profile", lambda: BrokenProfiler())
    with pytest.raises(RuntimeError, match="profiler exploded"):
        bench._run_kernel_check_cases(case_ids=["case-0"])
    assert config.kernel_check_records[0]["status"] == "failed"
    assert "profiler exploded" in config.kernel_check_records[0]["error"]
    assert not config.executed_case_ids


def test_kernel_check_legacy_fallback_invokes_kernel(monkeypatch, kc_runner):
    """Legacy benchmarks (input_fn mode) should be profiled via the fallback path."""
    bench, config, _ = kc_runner
    monkeypatch.setattr(bench, "supports_cases", lambda: False)
    bench.to_bench_dtypes = [torch.float32]
    # Provide a legacy input iterator that yields one input tuple.
    monkeypatch.setattr(bench, "get_input_iter", lambda dtype: iter([(42,)]))
    monkeypatch.setattr(
        bench,
        "_profile_invokes_kernel",
        staticmethod(lambda fn: True),
    )
    executed = bench._run_kernel_check_cases()
    assert len(executed) == 1
    assert executed[0].startswith("example-legacy-")
    assert config.kernel_check_records[-1]["invoked_kernel"] is True
    assert config.kernel_check_records[-1]["status"] == "passed"


def test_kernel_check_legacy_fallback_no_kernel(monkeypatch, kc_runner):
    """Legacy fallback records invoked_kernel=False when no device kernel fires."""
    bench, config, _ = kc_runner
    monkeypatch.setattr(bench, "supports_cases", lambda: False)
    bench.to_bench_dtypes = [torch.float32]
    monkeypatch.setattr(bench, "get_input_iter", lambda dtype: iter([(42,)]))
    monkeypatch.setattr(
        bench,
        "_profile_invokes_kernel",
        staticmethod(lambda fn: False),
    )
    executed = bench._run_kernel_check_cases()
    assert len(executed) == 1
    assert config.kernel_check_records[-1]["invoked_kernel"] is False
    assert config.kernel_check_records[-1]["status"] == "passed"


def test_kernel_check_rejects_empty_case_plan(monkeypatch, kc_runner):
    bench, _, _ = kc_runner
    monkeypatch.setattr(bench, "_collect_cases", lambda: [])
    with pytest.raises(ValueError, match="no kernel-check cases"):
        bench._run_kernel_check_cases()


def test_kernel_check_session_passes_if_all_operators_have_kernel(runner):
    bench, config, _ = runner
    config.preflight_only = False
    config.override_registry = None
    config.kernel_check = True
    config.kernel_check_records = [
        {"operator": "abs", "invoked_kernel": True},
        {"operator": "abs", "invoked_kernel": True},
    ]
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
    )
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.OK


@pytest.mark.parametrize(
    "operator,invoked_kernel,status,expected",
    [
        ("is_same_size", False, "passed", pytest.ExitCode.OK),
        ("is_same_size", True, "passed", pytest.ExitCode.TESTS_FAILED),
        ("is_same_size", False, "failed", pytest.ExitCode.TESTS_FAILED),
        ("abs", False, "passed", pytest.ExitCode.TESTS_FAILED),
    ],
)
def test_kernel_check_session_metadata_baseline(
    runner, operator, invoked_kernel, status, expected
):
    _, config, _ = runner
    config.preflight_only = False
    config.override_registry = None
    config.kernel_check = True
    config.kernel_check_records = [
        {"operator": operator, "invoked_kernel": invoked_kernel, "status": status}
    ]
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
    )
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == expected


def test_kernel_check_session_fails_if_any_operator_has_no_kernel(runner):
    """Even if one operator has a kernel, another without one should fail."""
    bench, config, _ = runner
    config.preflight_only = False
    config.override_registry = None
    config.kernel_check = True
    config.kernel_check_records = [
        {"operator": "abs", "invoked_kernel": True},
        {"operator": "empty", "invoked_kernel": False},
    ]
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
    )
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_kernel_check_session_fails_if_no_kernel(runner):
    bench, config, _ = runner
    config.preflight_only = False
    config.override_registry = None
    config.kernel_check = True
    config.kernel_check_records = [
        {"operator": "empty", "invoked_kernel": False},
        {"operator": "empty", "invoked_kernel": False},
    ]
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
    )
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_kernel_check_session_fails_if_no_records(runner):
    bench, config, _ = runner
    config.preflight_only = False
    config.override_registry = None
    config.kernel_check = True
    config.kernel_check_records = []
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
    )
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_kernel_check_does_not_downgrade_interrupt(runner):
    bench, config, _ = runner
    config.preflight_only = False
    config.override_registry = None
    config.kernel_check = True
    config.kernel_check_records = [{"operator": "empty", "invoked_kernel": False}]
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.INTERRUPTED,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
    )
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.INTERRUPTED


@pytest.mark.parametrize(
    "other",
    [
        ["--preflight-only"],
        ["--profile-only", "--case-id", "x"],
        ["--reference-only"],
        ["--list-cases"],
        ["--query"],
    ],
)
def test_kernel_check_rejects_conflicting_cli_modes(monkeypatch, other):
    from _pytest.config.argparsing import Parser

    parser = Parser()
    conftest.pytest_addoption(parser)
    options = parser.parse(["--kernel-check", *other])
    cfg = SimpleNamespace(
        option=options,
        addinivalue_line=lambda *args: None,
        getini=lambda key: [],
        getoption=lambda key: getattr(options, key.lstrip("-").replace("-", "_")),
        hook=SimpleNamespace(pytest_flaggems_profile_scope=None),
    )
    with pytest.raises(pytest.UsageError, match="--kernel-check"):
        conftest.pytest_configure(cfg)
