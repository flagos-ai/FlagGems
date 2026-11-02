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

"""A `--override` name that matches no operator must not produce a report.

When the name given to ``--override`` does not match the operator a benchmark
measures, ``DynamicOpOverride.get_override`` returns ``None`` and the harness
silently times the built-in implementation instead, while still printing
``✓ Overridden <name>``. The JSON was then written before teardown noticed that
the injected implementation had never run, so the report contained numbers that
describe something other than the requested candidate.

These tests pin the two halves of the fix: the registry reports which injected
names never ran, and the benchmark report writer refuses to publish when that
list is non-empty.
"""

import sys

import pytest

import flag_gems
from flag_gems.dynamic_registry import DynamicOpOverride

IMPL_SOURCE = """
def candidate(x):
    return x
"""


@pytest.fixture
def impl_file(tmp_path):
    path = tmp_path / "impl.py"
    path.write_text(IMPL_SOURCE)
    return str(path)


def _register(registry, impl_file, name):
    assert registry.override_from_file(name, impl_file, "candidate")
    return f"flag_gems.{name}"


def test_unused_overrides_lists_injected_but_never_called(impl_file):
    registry = DynamicOpOverride()
    _register(registry, impl_file, "NOT_AN_OPERATOR")
    try:
        assert registry.unused_overrides() == ["NOT_AN_OPERATOR"]
        # restore_all() raises for exactly this case; the read-only view must
        # agree with it.
        with pytest.raises(AssertionError, match="never invoked"):
            registry.restore_all()
    finally:
        registry.restore_all(allow_unused=True)


def test_unused_overrides_empty_once_the_candidate_runs(impl_file):
    registry = DynamicOpOverride()
    _register(registry, impl_file, "NOT_AN_OPERATOR")
    try:
        assert registry.unused_overrides() == ["NOT_AN_OPERATOR"]
        registry.get_override("NOT_AN_OPERATOR")(1)
        assert registry.call_counts().get("flag_gems.NOT_AN_OPERATOR") == 1
        assert registry.unused_overrides() == []
        registry.restore_all()  # does not raise: the candidate was invoked
    finally:
        registry.restore_all(allow_unused=True)


def test_unused_overrides_ignores_programmatic_overrides():
    # Restore semantics only flag the --override/CLI path, so the read-only
    # view must not report an in-process override() either.
    registry = DynamicOpOverride()
    registry.override("some_op", lambda x: x)
    try:
        assert registry.unused_overrides() == []
    finally:
        registry.restore_all(allow_unused=True)


class _Reporter:
    def __init__(self):
        self.lines = []

    def write_line(self, message):
        self.lines.append(message)


@pytest.fixture
def bench_conftest(monkeypatch):
    """The benchmark conftest as a module, with a fresh Config."""
    sys.modules.setdefault("flag_gems", flag_gems)
    import benchmark.conftest as conftest

    config = conftest.BenchConfig()
    monkeypatch.setattr(conftest, "Config", config)
    return conftest, config


@pytest.mark.parametrize("unused", [["NOT_AN_OPERATOR"], ["A", "B"]])
def test_report_is_withheld_when_an_override_name_did_not_match(bench_conftest, unused):
    conftest, config = bench_conftest
    config.record_json = True
    config.override_unused = unused

    reporter = _Reporter()
    conftest.pytest_terminal_summary(reporter, 0, None)

    assert len(reporter.lines) == 1
    assert "no report was written" in reporter.lines[0]
    for name in unused:
        assert f"'{name}'" in reporter.lines[0]


def test_report_writer_is_untouched_when_every_override_matched(bench_conftest):
    conftest, config = bench_conftest
    config.record_json = False  # nothing to write; just assert no early return
    config.override_unused = []

    reporter = _Reporter()
    conftest.pytest_terminal_summary(reporter, 0, None)

    assert reporter.lines == []


def test_non_timing_modes_do_not_treat_overrides_as_mistakes(bench_conftest):
    conftest, config = bench_conftest

    class _Registry:
        def unused_overrides(self):
            raise AssertionError("must not be consulted in a non-timing mode")

    config.override_registry = _Registry()
    try:
        for flag in (
            "query",
            "list_cases",
            "preflight_only",
            "reference_only",
            "ref_as_candidate",
        ):
            setattr(config, flag, True)
            assert conftest._unused_override_names() == []
            setattr(config, flag, False)
    finally:
        config.override_registry = None
