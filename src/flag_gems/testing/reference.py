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

"""Explicit references for opt-in correctness and candidate timing checks.

Correctness tests explicitly pass their reference to get_candidate; benchmarks
supply their callable torch_op. Both use the same candidate selection below.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

import pytest

_MODE = ContextVar("flag_gems_reference_test_mode", default=None)


class ReferenceContractError(ValueError):
    """The selected test has not declared an executable reference contract."""

    def __init__(self, message):
        super().__init__(message)
        state = _MODE.get()
        if state is not None and state.ref_as_candidate:
            state.contract_errors.append(str(message))


@dataclass
class _Execution:
    ref_as_candidate: bool
    nodeid: str
    reference_calls: int = 0
    candidate_calls: int = 0
    reference_selected: bool = False
    completed_timings: int = 0
    benchmark_selection_empty: bool = False
    contract_errors: list = field(default_factory=list)
    reported_contract_errors: int = 0


@contextmanager
def reference_test_mode(enabled, nodeid=""):
    """Scope the common candidate choice and counters to one pytest call."""
    state = _Execution(enabled, nodeid)
    token = _MODE.set(state)
    try:
        yield state
    finally:
        _MODE.reset(token)


def reference_execution():
    """Return the current execution, for thin runner/report adapters."""
    return _MODE.get()


def select_candidate(candidate, reference, label=""):
    """Select and observe the explicit reference in both execution paths."""
    state = _MODE.get()
    if state is None or not state.ref_as_candidate:
        return candidate
    if not callable(reference):
        raise ReferenceContractError(
            f"{label or state.nodeid}: No predefined reference; explicitly "
            "define a callable reference before --ref-as-candidate."
        )
    state.reference_selected = True

    @wraps(reference)
    def invoke(*args, **kwargs):
        state.candidate_calls += 1
        state.reference_calls += 1
        return reference(*args, **kwargs)

    return invoke


def get_candidate(op_name, *, reference=None):
    """Get the current public candidate, or the explicitly supplied reference.

    Reference mode deliberately does not look up the actual operator. Ordinary
    lookup uses the package's current public attribute, including CLI overrides
    and dummy injection, rather than its underlying implementation registrar.
    """
    state = _MODE.get()
    if state is not None and state.ref_as_candidate:
        return select_candidate(None, reference, op_name)

    import flag_gems

    try:
        candidate = getattr(flag_gems, op_name)
    except AttributeError as error:
        raise ImportError(
            f"Operator '{op_name}' has no public candidate in flag_gems."
        ) from error
    return select_candidate(candidate, reference, op_name)


class _ReferenceCandidatePlugin:
    """Shared pytest lifecycle; correctness and benchmark keep their runners."""

    def __init__(self):
        self.validated = False

    @staticmethod
    def _accuracy(item):
        return item.path.parent.name != "benchmark"

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        with reference_test_mode(True, item.nodeid) as state:
            item._reference_execution = state
            yield

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        state = item._reference_execution
        # Fixture execution cannot stand in for the selected test's call.
        # Keep only contract failures sticky across the whole node lifecycle.
        state.reference_calls = 0
        state.candidate_calls = 0
        state.completed_timings = 0
        state.benchmark_selection_empty = False
        outcome = yield
        interrupted = outcome.excinfo and issubclass(
            outcome.excinfo[0],
            (KeyboardInterrupt, SystemExit, pytest.exit.Exception),
        )
        if state.contract_errors and not interrupted:
            if outcome.excinfo is None or not isinstance(
                outcome.excinfo[1], ReferenceContractError
            ):
                outcome.force_exception(
                    ReferenceContractError(state.contract_errors[0])
                )
        elif (
            outcome.excinfo is None
            and not (state.benchmark_selection_empty and state.candidate_calls == 0)
            and (
                state.candidate_calls == 0
                or (not self._accuracy(item) and state.completed_timings == 0)
            )
        ):
            outcome.force_exception(
                ReferenceContractError(
                    f"{item.nodeid}: "
                    + (
                        "No predefined reference selected; explicitly pass "
                        "a callable reference to get_candidate."
                        if self._accuracy(item) and not state.reference_selected
                        else "explicit reference contract did not complete "
                        "a candidate-side call/timing."
                    )
                )
            )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        state = getattr(item, "_reference_execution", None)
        if (
            state is not None
            and self._accuracy(item)
            and report.skipped
            and not state.reference_selected
            and not state.contract_errors
        ):
            # No declaration must not disappear into skip/xfail, including a
            # skipped fixture which prevented the test body from running.
            ReferenceContractError(
                f"{item.nodeid}: No predefined reference selected; explicitly "
                "pass a callable reference to get_candidate."
            )
        if (
            state is not None
            and len(state.contract_errors) > state.reported_contract_errors
        ):
            state.reported_contract_errors = len(state.contract_errors)
            report.outcome = "failed"
            report.longrepr = "ReferenceContractError: " + state.contract_errors[0]
            if hasattr(report, "wasxfail"):
                del report.wasxfail
        if call.when == "call" and report.passed and state.candidate_calls:
            self.validated = True
        if state is not None and (call.when == "call" or state.contract_errors):
            report.reference_candidate = {
                "candidate_source": "reference",
                "reference_calls": state.reference_calls,
                "reference_candidate_calls": state.candidate_calls,
            }
            if state.contract_errors:
                # The ordinary JSON adapter mainly records call outcomes.
                # Preserve contract failures from setup/teardown as well,
                # including later successful phases of this failed node.
                report.reference_candidate.update(
                    result="failed",
                    reason="ReferenceContractError: " + state.contract_errors[0],
                )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        yield
        if not self.validated and session.exitstatus in (0, 5):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def validate_ref_as_candidate_options(config):
    """Validate only the new mode; leave all existing combinations unchanged."""
    if not getattr(config.option, "ref_as_candidate", False):
        return False
    conflicts = [
        name
        for name in (
            "override",
            "override_config",
            "reference_only",
            "preflight_only",
            "profile_only",
            "list_cases",
            "query",
            "parallel",
            "collect_marks",
            "collectonly",
            "numprocesses",
        )
        if getattr(config.option, name, None)
    ]
    if conflicts:
        raise pytest.UsageError(
            "--ref-as-candidate cannot be combined with "
            + ", ".join(
                (
                    "--collect-only"
                    if name == "collectonly"
                    else "--" + name.replace("_", "-")
                )
                for name in conflicts
            )
        )
    if not config.pluginmanager.hasplugin("flaggems-ref-as-candidate"):
        config.pluginmanager.register(
            _ReferenceCandidatePlugin(), "flaggems-ref-as-candidate"
        )
    return True
