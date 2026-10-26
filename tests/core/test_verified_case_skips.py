"""Host-only checks; no GPU/runtime import is required."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "verified_case_skips", ROOT / "benchmark/verified_case_skips.py"
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_registry_is_unique_and_conservative():
    rows = json.loads((ROOT / "benchmark/verified_case_skips.json").read_text())
    assert len(rows) == len(policy._rules()) == 2352
    assert sum(r[3] == "OOM" for r in rows) == 2348
    assert sum(r[3] == "BF16_UNSUPPORTED" for r in rows) == 4
    assert {r[0] for r in rows} == {
        "ascend",
        "hygon",
        "metax",
        "mthreads",
        "thead",
        "enflame",
        "iluvatar",
    }


def test_new_oom_rules_have_exact_recorded_scope():
    from collections import Counter

    rows = json.loads((ROOT / "benchmark/verified_case_skips.json").read_text())
    added = rows[370:791]
    assert len(added) == 421
    assert all(row[3] == "OOM" for row in added)
    assert Counter(row[0] for row in added) == {
        "ascend": 22,
        "hygon": 84,
        "metax": 80,
        "mthreads": 74,
        "thead": 53,
        "enflame": 105,
        "iluvatar": 3,
    }


def test_full_traversal_oom_rules_have_exact_recorded_scope():
    from collections import Counter

    rows = json.loads((ROOT / "benchmark/verified_case_skips.json").read_text())
    added = rows[791:]
    assert len(added) == 1561
    assert all(row[3] == "OOM" for row in added)
    assert Counter(row[0] for row in added) == {
        "ascend": 106,
        "hygon": 392,
        "metax": 376,
        "mthreads": 130,
        "thead": 109,
        "enflame": 445,
        "iluvatar": 3,
    }


def test_exact_match_only(monkeypatch):
    import hashlib

    case = SimpleNamespace(
        case_id="example", dtype="torch.float32", shape=[8], params={}
    )
    data = {"dtype": case.dtype, "shape": case.shape, "params": case.params}
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(
        policy, "_rules", lambda: {("ascend", "example", digest): "OOM"}
    )
    with pytest.raises(pytest.skip.Exception, match="Verified OOM"):
        policy.skip_verified_case("ascend", case)
    policy.skip_verified_case("nvidia", case)
    for field, value in [
        ("dtype", "torch.float16"),
        ("shape", [4]),
        ("params", {"a": 1}),
        ("case_id", "other"),
    ]:
        modified = SimpleNamespace(**vars(case))
        setattr(modified, field, value)
        policy.skip_verified_case("ascend", modified)


def test_reference_skip_continues_to_next_case():
    tree = ast.parse((ROOT / "benchmark/base.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_run_reference_cases"
    )
    handler = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.ExceptHandler)
        and n.type is not None
        and ast.unparse(n.type) == "pytest.skip.Exception"
    )
    assert any(isinstance(n, ast.Continue) for n in handler.body)
    assert not any(isinstance(n, ast.Raise) for n in ast.walk(handler))


def test_preflight_all_skipped_is_not_success():
    tree = ast.parse((ROOT / "benchmark/conftest.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "pytest_sessionfinish"
    )
    node = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "incomplete" for t in n.targets)
    )
    expression = compile(ast.Expression(node.value), "preflight_check", "eval")
    for states, expected in [
        ([], True),
        (["skipped"], True),
        (["skipped", "passed"], False),
        (["passed", "failed"], True),
    ]:
        config = SimpleNamespace(preflight_records=[{"status": s} for s in states])
        assert eval(expression, {"Config": config}) is expected


def test_preflight_single_skip_propagates_but_batch_continues():
    from contextlib import nullcontext

    tree = ast.parse((ROOT / "benchmark/base.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_run_preflight_cases"
    )
    config = SimpleNamespace(
        current_nodeid="test",
        available_case_ids=set(),
        executed_case_ids=set(),
        preflight_records=[],
    )
    namespace = {
        "pytest": pytest,
        "Config": config,
        "torch_device_fn": SimpleNamespace(synchronize=lambda: None),
    }
    method.args.args[1].annotation = None
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "preflight", "exec"),
        namespace,
    )

    def build(case):
        if case.case_id == "skip":
            pytest.skip("verified OOM")
        return ()

    bench = SimpleNamespace(
        op_name="test",
        supports_cases=lambda: True,
        _collect_cases=lambda: [SimpleNamespace(case_id=k) for k in ("skip", "run")],
        build_inputs=build,
        unpack_to_args_kwargs=lambda value: (value, {}),
        _candidate_call=lambda: (lambda: None, nullcontext(), True),
    )
    run = namespace["_run_preflight_cases"]
    with pytest.raises(pytest.skip.Exception, match="verified OOM"):
        run(bench, {"skip"})
    config.preflight_records.clear()
    assert run(bench, None) == ["run"]
    assert [r["status"] for r in config.preflight_records] == ["skipped", "passed"]
