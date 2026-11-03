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

"""Explicit reference contracts through shared helpers and real pytest hooks."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract():
    # The public contract module is intentionally usable without a device
    # runtime. CLI integration below loads the normal package and conftests.
    source = ROOT / "src/flag_gems/testing/reference.py"
    spec = importlib.util.spec_from_file_location(
        "_flaggems_reference_contract_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _forbidden(*args, **kwargs):
    raise AssertionError("the original candidate was called")


def test_default_candidate_is_the_current_public_export(contract, monkeypatch):
    class FalseyCandidate:
        def __bool__(self):
            return False

        def __call__(self, value):
            return value + 1

    public = ModuleType("flag_gems")
    original, injected = _forbidden, FalseyCandidate()
    public.reference_probe = original
    monkeypatch.setitem(sys.modules, "flag_gems", public)
    assert contract.get_candidate("reference_probe", reference=sum) is original
    public.reference_probe = injected
    assert contract.get_candidate("reference_probe", reference=None) is injected
    assert contract.select_candidate(injected, sum) is injected
    with contract.reference_test_mode(False):
        assert contract.get_candidate("reference_probe") is injected
    monkeypatch.delattr(public, "reference_probe")
    with pytest.raises(ImportError):
        contract.get_candidate("reference_probe", reference=sum)


def test_reference_candidate_does_not_lookup_the_public_export(contract, monkeypatch):
    public = ModuleType("flag_gems")
    public.__getattr__ = _forbidden
    monkeypatch.setitem(sys.modules, "flag_gems", public)
    value = [1]

    def reference(values, *, increment):
        assert values is value
        values.append(increment)
        return values

    with contract.reference_test_mode(True, "explicit-node") as state:
        candidate = contract.get_candidate("absent_export", reference=reference)
        assert candidate(value, increment=2) is value
    assert value == [1, 2]
    assert (state.reference_calls, state.candidate_calls) == (1, 1)


@pytest.mark.parametrize("reference", [None, 42, "sum"])
def test_missing_reference_is_explicit_before_any_candidate_lookup(
    contract, monkeypatch, reference
):
    public = ModuleType("flag_gems")
    public.__getattr__ = _forbidden
    monkeypatch.setitem(sys.modules, "flag_gems", public)
    with contract.reference_test_mode(True, "missing-node") as state:
        with pytest.raises(
            contract.ReferenceContractError, match="No predefined reference"
        ):
            contract.get_candidate("absent_export", reference=reference)
    assert state.candidate_calls == 0
    assert len(state.contract_errors) == 1


def test_omitted_reference_and_nested_mode_restoration(contract, monkeypatch):
    public = ModuleType("flag_gems")
    public.reference_probe = _forbidden
    monkeypatch.setitem(sys.modules, "flag_gems", public)
    with contract.reference_test_mode(True):
        with pytest.raises(
            contract.ReferenceContractError, match="No predefined reference"
        ):
            contract.get_candidate("reference_probe")
        with contract.reference_test_mode(False):
            assert contract.get_candidate("reference_probe") is _forbidden
        assert contract.get_candidate("reference_probe", reference=sum)([1]) == 1
    assert contract.get_candidate("reference_probe") is _forbidden


@pytest.fixture
def project(pytester):
    # Load the actual correctness conftest and override code, but a host-only
    # package shell so these subprocess checks also run without Torch/Triton.
    preamble = f"""
import sys, types
from pathlib import Path
pkg = types.ModuleType("flag_gems")
pkg.__path__ = [{str(ROOT / "src/flag_gems")!r}]
pkg.device = "cpu"
pkg.vendor_name = "nvidia"
pkg.add = lambda a, b: -100
sys.modules["flag_gems"] = pkg
testing = types.ModuleType("flag_gems.testing")
testing.__path__ = [{str(ROOT / "src/flag_gems/testing")!r}]
sys.modules["flag_gems.testing"] = testing
"""
    pytester.makeconftest(preamble + (ROOT / "tests/conftest.py").read_text())
    pytester.makeini("[pytest]\nmarkers = good\n")
    pytester.makepyfile(test_case="""
import pytest
import flag_gems
from flag_gems.testing.reference import get_candidate

def compare(actual, expected):
    assert actual == expected

@pytest.mark.good
@pytest.mark.parametrize("value", [1, 2, 3])
def test_case(value):
    reference = lambda a, b: a + b
    candidate = get_candidate("add", reference=reference)
    compare(candidate(value, 2), reference(value, 2))
""")
    return pytester


def test_real_hooks_ref_mode_full_grid_and_call_report(project):
    result = project.runpytest_subprocess("--ref-as-candidate", "--ref", "cpu", "-q")
    result.assert_outcomes(passed=3)
    records = json.loads((project.path / "accuracy_result.json").read_text())
    assert len(records) == 3
    assert all(r["candidate_source"] == "reference" for r in records.values())
    assert all(r["reference_calls"] == 1 for r in records.values())
    assert all(r["reference_candidate_calls"] == 1 for r in records.values())


def test_real_hooks_default_and_external_overrides(project):
    project.runpytest_subprocess("-q").assert_outcomes(failed=3)
    candidate = project.makepyfile(candidate="def run(a, b): return a + b\n")
    project.runpytest_subprocess(
        "-q", "--override", f"add:{candidate}:run"
    ).assert_outcomes(passed=3)
    for suffix in ("json", "yaml"):
        config = project.path / f"override.{suffix}"
        config.write_text(
            json.dumps(
                {"overrides": {"add": {"file": str(candidate), "function": "run"}}}
            )
        )
        project.runpytest_subprocess(
            "-q", "--override-config", str(config)
        ).assert_outcomes(passed=3)


def test_missing_declaration_fails_only_when_enabled_and_selected(project):
    project.makepyfile(test_legacy="def test_legacy(): pass\n")
    project.runpytest_subprocess("-q", "test_legacy.py").assert_outcomes(passed=1)
    result = project.runpytest_subprocess("-q", "--ref-as-candidate")
    result.assert_outcomes(passed=3, failed=1)
    result.stdout.fnmatch_lines(
        ["*test_legacy.py::test_legacy*No predefined reference*"]
    )
    project.runpytest_subprocess(
        "-q", "--ref-as-candidate", "-m", "good"
    ).assert_outcomes(passed=3, deselected=1)


@pytest.mark.parametrize(
    "code",
    [
        "pass",
        "get_candidate('add', reference=lambda a, b: a+b)",
        "pytest.skip('no declaration')",
        "pytest.xfail('no declaration')",
        "get_candidate('add', reference=None)",
        "try:\n        get_candidate('add')\n    except ValueError:\n        pytest.skip('hidden')",
    ],
)
def test_missing_or_unused_contract_cannot_pass(project, code):
    project.makepyfile(
        test_bad="import pytest\nfrom flag_gems.testing.reference import get_candidate\ndef test_bad():\n    "
        + code
        + "\n"
    )
    result = project.runpytest_subprocess("-q", "--ref-as-candidate", "test_bad.py")
    result.assert_outcomes(failed=1)
    records = json.loads((project.path / "accuracy_result.json").read_text())
    assert records["test_bad.py::test_bad"]["result"] == "failed"


def test_fixture_calls_do_not_replace_body(project):
    project.makepyfile(test_fixture="""
import pytest
from flag_gems.testing.reference import get_candidate
@pytest.fixture
def candidate():
    op = get_candidate('add', reference=lambda a,b:a+b)
    op(1,2)
    return op
def test_fixture(candidate):
    pass
""")
    project.runpytest_subprocess(
        "-q", "--ref-as-candidate", "test_fixture.py"
    ).assert_outcomes(failed=1)


@pytest.mark.parametrize("phase", ["setup", "teardown"])
def test_fixture_contract_errors_are_failed_in_json(project, phase):
    setup = "get_candidate('add')" if phase == "setup" else "pass"
    teardown = "get_candidate('add')" if phase == "teardown" else "pass"
    project.makepyfile(test_phase=f"""
import pytest
from flag_gems.testing.reference import get_candidate
@pytest.fixture(autouse=True)
def fixture():
    {setup}
    yield
    {teardown}
def test_phase():
    assert get_candidate('add', reference=lambda a,b:a+b)(1,2) == 3
""")
    result = project.runpytest_subprocess("-q", "--ref-as-candidate", "test_phase.py")
    assert result.ret == 1
    data = json.loads((project.path / "accuracy_result.json").read_text())
    assert data["test_phase.py::test_phase"]["result"] == "failed"


def test_valid_reference_capability_skip_is_not_validation(project):
    project.makepyfile(test_skip="""
import pytest
from flag_gems.testing.reference import get_candidate
def test_skip():
    get_candidate('add', reference=lambda a,b:a+b)
    pytest.skip('capability')
""")
    result = project.runpytest_subprocess("-q", "--ref-as-candidate", "test_skip.py")
    result.assert_outcomes(skipped=1)
    assert result.ret == 1


@pytest.mark.parametrize("timer_calls", [0, 1, 3])
def test_actual_benchmark_methods_require_reference_invocation(contract, timer_calls):
    # Execute the production selection/measurement methods with a host timer.
    # Importing the full benchmark module requires an accelerator runtime.
    import ast
    import gc
    from contextlib import nullcontext
    from types import SimpleNamespace

    tree = ast.parse((ROOT / "benchmark/base.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Benchmark"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"__init__", "_candidate_call", "_measure_input"}
    ]
    namespace = {
        "Config": SimpleNamespace(
            ref_as_candidate=True, skip_native=False, override_registry=None
        ),
        "BenchmarkMetrics": lambda **kw: SimpleNamespace(**kw),
        "ReferenceContractError": contract.ReferenceContractError,
        "reference_execution": contract.reference_execution,
        "select_candidate": contract.select_candidate,
        "nullcontext": nullcontext,
        "gc": gc,
        "pytest": pytest,
    }
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "benchmark/base.py", "exec"),
        namespace,
    )
    benchmark_type = namespace["Benchmark"]
    with contract.reference_test_mode(True) as invalid:
        with pytest.raises(contract.ReferenceContractError, match="callable torch_op"):
            benchmark_type("missing", torch_op=None)
        assert invalid.contract_errors
    bench = object.__new__(benchmark_type)
    calls = []
    bench.op_name = "probe"
    bench.torch_op = lambda value: calls.append(value)
    bench.to_bench_metrics = ["latency_base", "latency"]
    bench.unpack_to_args_kwargs = lambda inputs: (inputs, {})
    bench.record_shapes = lambda *args, **kwargs: None
    bench._capture_external_baseline = lambda *args, **kwargs: None
    bench._apply_external_baseline = lambda *args, **kwargs: None

    def timer(op, *args, **kwargs):
        for _ in range(timer_calls):
            op(*args, **kwargs)
        return 1.0

    bench.get_latency = timer
    with contract.reference_test_mode(True) as state:
        if not timer_calls:
            with pytest.raises(pytest.fail.Exception, match="did not invoke"):
                bench._measure_input((2,))
            assert state.completed_timings == 0
        else:
            metric = bench._measure_input((2,))
            assert metric.candidate_source == "reference"
            assert state.completed_timings == 1
            assert state.candidate_calls == timer_calls
            assert calls == [2] * (2 * timer_calls)


@pytest.mark.parametrize(
    "option", ["override", "reference_only", "collectonly", "numprocesses"]
)
def test_options_are_validated_only_when_enabled(contract, option):
    from types import SimpleNamespace

    config = SimpleNamespace(
        option=SimpleNamespace(ref_as_candidate=False, **{option: True})
    )
    assert contract.validate_ref_as_candidate_options(config) is False
    config.option.ref_as_candidate = True
    with pytest.raises(pytest.UsageError, match="cannot be combined"):
        contract.validate_ref_as_candidate_options(config)
