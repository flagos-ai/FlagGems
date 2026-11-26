#!/usr/bin/env python3
r"""Run catalog, KGS inspect and dummy checks for one FlagGems operator.

Run at the FlagGems repo root with pytest and KGS available in this Python
environment. Inspect validates the committed checkout. Use a fresh report
directory outside the checkout, for example:

    python .agents/skills/flaggems-operator-test-repair/scripts/check_operator.py \
        --op linalg_lu_solve --marker linalg_lu_solve \
        --test-file tests/test_linalg_lu_solve.py \
        --bench-file benchmark/test_linalg_lu_solve.py --output /tmp/lu-solve-check

Use --baseline with an earlier run directory on the same backend to compare
core metadata and accuracy node IDs. For proven skips, pass --allowed-skips with a JSON object
mapping dummy-default, dummy-cpu or dummy-core to lists of exact skipped IDs.
This helper does not run ref-as-candidate or timed benchmarks. Accuracy node
IDs do not cover all typed correctness parameters.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from uuid import uuid4

PROBE = "kernelgen_server.evaluation.adapters.flaggems.injection_probe_plugin"
CASE_SCHEMA = "flaggems.benchmark-case-list/v2"
DUMMY_PHASES = ("dummy-default", "dummy-cpu", "dummy-core")
INSPECT = """\
import sys
from pathlib import Path
from kernelgen_client.protocol.schema import EvaluatorBinding
from kernelgen_server.evaluation.adapters.registry import create_adapter

operator, revision, output = sys.argv[1:]
binding = EvaluatorBinding(
    catalog_name="flaggems", definition=operator, source_revision=revision
)
Path(output).write_text(create_adapter(binding).inspect().model_dump_json())
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def unique_ids(rows, key):
    require(isinstance(rows, list) and rows, "No cases reported")
    ids = [row[key] for row in rows]
    require(all(isinstance(i, str) and i for i in ids), f"Invalid {key}")
    require(len(set(ids)) == len(ids), f"Duplicate {key}")
    return set(ids)


def catalog(report, operator):
    require(report["schema_version"] == CASE_SCHEMA, "Unsupported catalog schema")
    benchmarks = report["benchmarks"]
    require(isinstance(benchmarks, list) and benchmarks, "Empty benchmark catalog")
    cases = []
    for benchmark in benchmarks:
        require(
            benchmark["schema_version"] == CASE_SCHEMA
            and benchmark["op_name"] == operator
            and benchmark["level"] == "core",
            "Unexpected benchmark schema, operator or level",
        )
        require(
            benchmark["phase"] == "timing" and isinstance(benchmark["cases"], list),
            "Invalid benchmark phase or case list",
        )
        cases.extend(benchmark["cases"])
    unique_ids(cases, "case_id")
    return cases


def check_inspect(report, operator, cases):
    listed = report["case_list"]
    require(
        report["kind"] == listed["adapter_kind"] == "flaggems"
        and listed["schema_version"] == "kernelgen.case-list/v1"
        and listed["operator"] == operator
        and report["benchmark_fingerprint"] == listed["benchmark_fingerprint"],
        "Inspect returned a different adapter, operator or catalog",
    )
    keys = ("case_id", "dtype", "shape", "params")
    expected = [
        {**{key: case[key] for key in keys}, "ordinal": ordinal}
        for ordinal, case in enumerate(cases)
    ]
    actual = [
        {key: case[key] for key in (*keys, "ordinal")} for case in listed["cases"]
    ]
    require(canonical(actual) == canonical(expected), "Inspect and list-cases differ")


def check_dummy(report, phase, expected_ids, allowed_skips):
    require(
        report["schema_version"] == "kgs.injection-probe/v1"
        and report["phase"] == phase
        and report["proves"] == "candidate_invocation_only"
        and report["collection_errors"] == [],
        "Invalid dummy report or collection error",
    )
    core = phase == "benchmark"
    key = "case_id" if core else "nodeid"
    rows = report["records"]
    ids = unique_ids(rows, key)
    if expected_ids is not None:
        require(ids == expected_ids, "Dummy case IDs differ from the core catalog")
    counts = Counter(row["status"] for row in rows)
    require(
        report["status"] == "PASSED"
        and counts.get("REACHED", 0)
        and counts.keys() <= {"REACHED", "SKIP"},
        f"Dummy status {report['status']} with counts {dict(counts)}",
    )
    require(
        all(
            type(row.get("calls")) is int and row["calls"] > 0
            for row in rows
            if row["status"] == "REACHED"
        ),
        "Dummy REACHED record has no invocation count",
    )
    skipped = {row[key] for row in rows if row["status"] == "SKIP"}
    require(
        skipped == allowed_skips,
        f"Unexpected skips: {sorted(skipped - allowed_skips)}; "
        f"expected skips not seen: {sorted(allowed_skips - skipped)}",
    )
    return {"counts": dict(counts), "ids": sorted(ids)}


def run(command, env, log_path, timeout):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def main():
    description, instructions = __doc__.split("\n\n", 1)
    parser = argparse.ArgumentParser(
        description=description,
        epilog=instructions,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--op", required=True, help="Definition name and override key")
    parser.add_argument("--marker", required=True, help="Exact pytest marker")
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--bench-file", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New report directory"
    )
    parser.add_argument(
        "--baseline", type=Path, help="Earlier run directory for coverage comparison"
    )
    parser.add_argument(
        "--allowed-skips",
        type=Path,
        help="JSON map from dummy stage to exact skipped IDs",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per command")
    args = parser.parse_args()

    root = Path.cwd()
    output = args.output.resolve()
    try:
        require(re.fullmatch(r"\w+(?:\.\w+)*", args.op), "Invalid operator name")
        require(re.fullmatch(r"\w+", args.marker), "Expected one exact pytest marker")
        require(args.timeout > 0, "Timeout must be positive")
        require(
            root not in output.parents and output != root,
            "Place reports outside the checkout",
        )
        require(
            ":" not in str(output), "Report path cannot contain ':' in override syntax"
        )
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        definition = read_json(root / "definitions" / f"{args.op}.json")
        require(definition["name"] == args.op, "Definition name does not match --op")
        for file in (args.test_file, args.bench_file):
            require(
                file.is_file() and root in file.resolve().parents,
                f"Invalid repo file: {file}",
            )
        skips = read_json(args.allowed_skips) if args.allowed_skips else {}
        require(
            isinstance(skips, dict) and skips.keys() <= set(DUMMY_PHASES),
            "Invalid allowed-skips stage",
        )
        for ids in skips.values():
            require(
                isinstance(ids, list)
                and all(isinstance(i, str) and i for i in ids)
                and len(set(ids)) == len(ids),
                "Allowed skips must be a list of unique nonempty IDs",
            )
        output.mkdir(parents=True, exist_ok=False)
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))

    env = {**os.environ, "KGS_FLAGGEMS_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), str(root), env.get("PYTHONPATH")) if part
    )
    pytest = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    test = [str(args.test_file.resolve()), "-m", args.marker]
    bench = [str(args.bench_file.resolve()), "-m", args.marker, "--level", "core"]
    stages = {}
    summary = {
        "operator": args.op,
        "marker": args.marker,
        "source_revision": rev,
        "repo": str(root),
        "python": sys.executable,
        "stages": stages,
        "not_run": ["ref-as-candidate", "timed-core"],
    }

    def check(name, command, validate, *, accepted=(0,), extra_env=None):
        stage = stages[name] = {"command": command, "log": f"{name}.log"}
        try:
            code = run(
                command,
                {**env, **(extra_env or {})},
                output / stage["log"],
                args.timeout,
            )
            stage["exit_code"] = code
            require(code in accepted, f"Command exited with {code}")
            result = validate()
            stage.update(status="PASSED", **result)
            return result
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.TimeoutExpired,
        ) as exc:
            stage.update(status="FAILED", error=str(exc))
            return None
        finally:
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    core_json = output / "core-cases.json"
    core = check(
        "list-cases",
        pytest
        + bench
        + ["--list-cases", "--record", "json", "--output", str(core_json)],
        lambda: {"cases": catalog(read_json(core_json), args.op)},
    )
    inspect_json = output / "inspect.json"

    def inspect_result():
        require(core is not None, "No valid local catalog for comparison")
        check_inspect(read_json(inspect_json), args.op, core["cases"])
        return {"case_count": len(core["cases"])}

    check(
        "inspect",
        [sys.executable, "-c", INSPECT, args.op, rev, str(inspect_json)],
        inspect_result,
    )
    candidate = output / "dummy_candidate.py"
    candidate.write_text(f"from {PROBE} import candidate as run\n", encoding="utf-8")
    dummy_runs = (
        ("dummy-default", "accuracy", test),
        ("dummy-cpu", "accuracy", test + ["--ref", "cpu"]),
        ("dummy-core", "benchmark", bench + ["--preflight-only"]),
    )
    for name, phase, selection in dummy_runs:
        report = output / f"{name}.json"
        ids = None
        if phase == "benchmark" and core is not None:
            ids = unique_ids(core["cases"], "case_id")

        def probe_result():
            require(phase != "benchmark" or ids is not None, "No valid core catalog")
            return check_dummy(read_json(report), phase, ids, set(skips.get(name, [])))

        check(
            name,
            pytest
            + ["-p", PROBE]
            + selection
            + ["--override", f"{args.op}:{candidate}:run"],
            probe_result,
            accepted=(0, 1),
            extra_env={
                "KGS_INJECTION_PHASE": phase,
                "KGS_INJECTION_NONCE": uuid4().hex,
                "KGS_INJECTION_REPORT": str(report),
            },
        )

    if args.baseline:
        baseline = stages["baseline-comparison"] = {}
        try:
            old = read_json(args.baseline / "summary.json")
            require(
                old["operator"] == args.op and old["marker"] == args.marker,
                "Baseline operator or marker differs",
            )
            require(core is not None, "No valid local catalog")
            old_cases = catalog(read_json(args.baseline / core_json.name), args.op)
            require(
                canonical(old_cases) == canonical(core["cases"]), "Core catalog changed"
            )
            for name in DUMMY_PHASES[:2]:
                old_ids = unique_ids(
                    read_json(args.baseline / f"{name}.json")["records"], "nodeid"
                )
                require(
                    old_ids == set(stages[name]["ids"]),
                    f"{name} accuracy node IDs changed",
                )
            baseline.update(status="PASSED")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            baseline.update(status="FAILED", error=str(exc))
    else:
        summary["not_run"].append("baseline-comparison")

    success = all(stage["status"] == "PASSED" for stage in stages.values())
    summary["status"] = "PASSED" if success else "FAILED"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, stage in stages.items():
        print(
            f"{name}: {stage['status']}"
            + (f" — {stage['error']}" if "error" in stage else "")
        )
    print(f"Reports: {output}")
    print("Not run: " + ", ".join(summary["not_run"]))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
