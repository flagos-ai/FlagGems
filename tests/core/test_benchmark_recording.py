# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep parameterized benchmark outcomes visible in the operator report."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

conftest = SimpleNamespace(pytest_runtest_logreport=None)


@pytest.fixture
def recording(monkeypatch):
    config = SimpleNamespace(
        reference_only=False, record_json=True, reference_records=[]
    )
    records = {}
    path = Path(__file__).resolve().parents[2] / "benchmark/conftest.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"get_reason", "_record_outcome", "pytest_runtest_logreport"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_RESULT_PRIORITY"
                for t in node.targets
            )
        )
    ]
    namespace = {"pytest": pytest, "Config": config, "TEST_RESULTS": records}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(
        conftest, "pytest_runtest_logreport", namespace["pytest_runtest_logreport"]
    )
    return config, records


def _report(outcome, when="call", nodeid="test_metadata[case]"):
    return SimpleNamespace(
        outcome=outcome,
        when=when,
        nodeid=nodeid,
        opid="underscore_assert_tensor_metadata",
        failed=outcome == "failed",
        longrepr=("test_metadata.py", 1, outcome + " reason"),
    )


@pytest.mark.parametrize("skip_when", ["setup", "call"])
@pytest.mark.parametrize("skip_first", [False, True])
def test_supported_timings_survive_partial_skips(recording, skip_when, skip_first):
    _, records = recording
    metrics = [{"latency": 1.0, "latency_base": 1.0}]
    records["underscore_assert_tensor_metadata"] = {"details": metrics}
    passed = _report("passed", nodeid="test_metadata[supported]")
    skipped = _report("skipped", skip_when, "test_metadata[device-layout]")
    for report in ([skipped, passed] if skip_first else [passed, skipped]):
        conftest.pytest_runtest_logreport(report)
    result = records[passed.opid]
    assert result["result"] == "passed"
    assert result["test_case"] == passed.nodeid
    assert result["reason"] is None
    assert result["details"] is metrics
    assert result["skipped_cases"] == [
        {"test_case": skipped.nodeid, "phase": skip_when, "reason": "skipped reason"}
    ]


@pytest.mark.parametrize("failure_when", ["setup", "call", "teardown"])
@pytest.mark.parametrize("later_outcome", ["passed", "skipped"])
def test_later_parameter_cannot_hide_failure(recording, failure_when, later_outcome):
    _, records = recording
    failed = _report("failed", failure_when, "test_metadata[failed]")
    conftest.pytest_runtest_logreport(failed)
    conftest.pytest_runtest_logreport(_report(later_outcome))
    result = records[failed.opid]
    assert result["result"] == "failed"
    assert result["test_case"] == failed.nodeid
    assert result["reason"] == "failed reason"


def test_teardown_failure_overrides_successful_call(recording):
    _, records = recording
    conftest.pytest_runtest_logreport(_report("passed"))
    conftest.pytest_runtest_logreport(_report("failed", "teardown"))
    result = records["underscore_assert_tensor_metadata"]
    assert result["result"] == "failed"
    assert result["reason"] == "failed reason"


def test_all_skipped_parameters_remain_skipped(recording):
    _, records = recording
    for nodeid in ["test_metadata[device]", "test_metadata[layout]"]:
        conftest.pytest_runtest_logreport(_report("skipped", "setup", nodeid))
    result = records["underscore_assert_tensor_metadata"]
    assert result["result"] == "skipped"
    assert len(result["skipped_cases"]) == 2


def test_reference_only_reporting_is_unchanged(recording):
    config, records = recording
    config.reference_only = True
    report = _report("failed", "teardown")
    conftest.pytest_runtest_logreport(report)
    assert records == {}
    assert config.reference_records == [
        {
            "nodeid": report.nodeid,
            "operator": report.opid,
            "status": "FAILED",
            "reason": "failed reason",
            "pytest_phase": "teardown",
        }
    ]


def test_recording_disabled_leaves_results_untouched(recording):
    config, records = recording
    config.record_json = False
    conftest.pytest_runtest_logreport(_report("failed"))
    assert records == {}
