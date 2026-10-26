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

"""Exercise the real result-recording hook without importing the GPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recording():
    path = Path(__file__).resolve().parents[2] / "benchmark/conftest.py"
    tree = ast.parse(path.read_text())
    functions = {"_record_outcome", "get_reason", "pytest_runtest_logreport"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_RESULT_PRIORITY"
                for target in node.targets
            )
        )
    ]
    # Isolate only runtime metadata; execute the hook and helpers from source.
    namespace = {
        "pytest": pytest,
        "Config": SimpleNamespace(
            reference_only=False, record_json=True, reference_records=[]
        ),
        "TEST_RESULTS": {},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def report(phase, outcome, index, op="op"):
    return SimpleNamespace(
        opid=op,
        nodeid=f"test-{index}",
        when=phase,
        outcome=outcome,
        failed=outcome == "failed",
        longrepr=f"reason-{index}",
    )


@pytest.mark.parametrize(
    "events, winner",
    [
        ([("call", "failed"), ("call", "passed")], 0),
        ([("call", "passed"), ("call", "skipped")], 0),
        ([("call", "skipped"), ("call", "passed")], 1),
        ([("call", "failed"), ("call", "skipped")], 0),
        ([("setup", "skipped"), ("call", "passed")], 1),
        ([("call", "passed"), ("setup", "skipped")], 0),
        ([("call", "failed"), ("call", "failed")], 1),
        ([("call", "passed"), ("call", "passed")], 1),
        ([("setup", "skipped"), ("call", "skipped")], 1),
    ],
)
def test_result_priority_preserves_matching_provenance(recording, events, winner):
    for index, (phase, outcome) in enumerate(events):
        recording["pytest_runtest_logreport"](report(phase, outcome, index))

    expected = events[winner][1]
    assert recording["TEST_RESULTS"]["op"] == {
        "result": expected,
        "test_case": f"test-{winner}",
        "reason": None if expected == "passed" else f"reason-{winner}",
    }


def test_operator_results_and_measurements_stay_independent(recording):
    recording["TEST_RESULTS"]["first"] = {"details": [{"latency": 1.0}]}
    hook = recording["pytest_runtest_logreport"]
    hook(report("call", "failed", 0, op="first"))
    hook(report("call", "passed", 1, op="second"))
    hook(report("call", "skipped", 2, op="first"))

    assert recording["TEST_RESULTS"]["first"] == {
        "details": [{"latency": 1.0}],
        "result": "failed",
        "test_case": "test-0",
        "reason": "reason-0",
    }
    assert recording["TEST_RESULTS"]["second"]["result"] == "passed"


@pytest.mark.parametrize("outcome", ["failed", "skipped"])
def test_reference_records_keep_their_separate_path(recording, outcome):
    recording["Config"].reference_only = True
    recording["pytest_runtest_logreport"](report("call", outcome, 0))

    assert recording["TEST_RESULTS"] == {}
    assert recording["Config"].reference_records == [
        {
            "nodeid": "test-0",
            "operator": "op",
            "status": "FAILED" if outcome == "failed" else "SKIP",
            "reason": "reason-0",
            "pytest_phase": "call",
        }
    ]


def test_recording_disabled_does_not_create_results(recording):
    recording["Config"].record_json = False
    recording["pytest_runtest_logreport"](report("call", "failed", 0))
    assert recording["TEST_RESULTS"] == {}
