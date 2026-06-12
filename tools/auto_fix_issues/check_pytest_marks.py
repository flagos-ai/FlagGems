#!/usr/bin/env python3
"""Check that pytest marks in test/benchmark files match operators.yaml IDs.

Only checks the files passed as arguments — does not scan the whole repo.
Exit code 0 = all marks valid, 1 = issues found.

Checks performed:
1. Every operator mark must exist in conf/operators.yaml
2. For single-operator files (test_<op>.py), marks must start with <op>
3. If test function name matches an operator exactly, mark must match too
4. Completeness: test<->benchmark pairing, operator existence in operators.yaml
"""

import argparse
import ast
import glob
import os
import re
import sys

import yaml

BUILTIN_MARKS = frozenset({
    "parametrize", "skip", "skipif", "xfail", "usefixtures",
    "filterwarnings", "timeout", "tryfirst", "trylast",
})


def load_operator_ids(yaml_path: str) -> set[str]:
    """Load all operator IDs from conf/operators.yaml.

    IDs like 'baddbmm.out' are converted to 'baddbmm_out' to match pytest mark naming.
    Both the original ID and the converted form are included.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    ids = set()
    for op in data.get("ops", []):
        op_id = op["id"]
        ids.add(op_id)
        # pytest marks use underscores where operators.yaml uses dots
        if "." in op_id:
            ids.add(op_id.replace(".", "_"))
    return ids


def extract_marks(filepath: str) -> list[dict]:
    """Extract pytest.mark.<name> decorators from a Python file.

    Returns list of {"mark": str, "line": int, "func": str}.
    """
    with open(filepath) as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            mark_name = _extract_mark_name(decorator)
            if mark_name and mark_name not in BUILTIN_MARKS:
                results.append({
                    "mark": mark_name,
                    "line": decorator.lineno,
                    "func": node.name,
                })
    return results


def _extract_mark_name(node: ast.expr) -> str | None:
    """Extract 'xxx' from @pytest.mark.xxx or @pytest.mark.xxx(...)."""
    # @pytest.mark.xxx(...)
    if isinstance(node, ast.Call):
        node = node.func

    # @pytest.mark.xxx
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
        and node.value.attr == "mark"
    ):
        return node.attr

    return None


def _get_file_operator(filepath: str) -> str | None:
    """Extract operator name from file path.

    tests/test_<op>.py -> <op>
    benchmark/test_<op>.py -> <op>
    benchmark/test_<op>_perf.py -> <op>
    """
    basename = os.path.basename(filepath)
    m = re.match(r"test_(.+)\.py$", basename)
    if m:
        op = m.group(1)
        # Strip common benchmark suffixes
        for suffix in ("_perf", "_perf_parallel"):
            if op.endswith(suffix):
                op = op[: -len(suffix)]
                break
        return op
    return None


def check_files(files: list[str], operator_ids: set[str]) -> list[str]:
    """Check marks in files against operator_ids. Returns list of error strings."""
    errors = []
    for filepath in files:
        if not os.path.isfile(filepath):
            continue
        marks = extract_marks(filepath)
        if not marks:
            continue

        file_op = _get_file_operator(filepath)

        for m in marks:
            # Check 1: mark must exist in operators.yaml
            if m["mark"] not in operator_ids:
                errors.append(
                    f"{filepath}:{m['line']}: mark '{m['mark']}' on {m['func']} "
                    f"not found in operators.yaml"
                )
                continue

            # Check 2: for test_<op>.py files where <op> is a known operator,
            # mark must start with <op>
            if file_op and file_op in operator_ids and not m["mark"].startswith(file_op):
                errors.append(
                    f"{filepath}:{m['line']}: mark '{m['mark']}' on {m['func']} "
                    f"does not match file operator '{file_op}'"
                )
                continue

            # Check 3: if function name is test_<op> and <op> is a known operator,
            # the mark must match <op> (not a parent operator).
            # e.g. test_baddbmm_out must have mark baddbmm_out, not baddbmm
            func_name = m["func"]
            if func_name.startswith("test_"):
                func_op = func_name[5:]  # strip "test_"
                if func_op in operator_ids and m["mark"] != func_op:
                    errors.append(
                        f"{filepath}:{m['line']}: mark '{m['mark']}' on {m['func']} "
                        f"should be '{func_op}' (exact operator match)"
                    )
    return errors


def check_completeness(all_files: list[str], operator_ids: set[str]) -> list[str]:
    """Check that commit files have matching counterparts.

    - New test/benchmark file test_<op>.py -> operators.yaml must have <op>
    - test file without benchmark (or vice versa) -> error only if not found on disk
    """
    errors = []

    # Collect operator names from test/benchmark files in this commit
    test_ops = set()
    bench_ops = set()
    for f in all_files:
        op = _get_file_operator(f)
        if not op:
            continue
        if "/tests/" in f or f.startswith("tests/"):
            test_ops.add(op)
        elif "/benchmark/" in f or f.startswith("benchmark/"):
            bench_ops.add(op)

    # Check: test files should have operator in operators.yaml
    for op in test_ops:
        if op not in operator_ids:
            errors.append(
                f"tests/test_{op}.py: operator '{op}' not found in operators.yaml"
            )

    # Check: benchmark files should have operator in operators.yaml
    for op in bench_ops:
        if op not in operator_ids:
            errors.append(
                f"benchmark/test_{op}*.py: operator '{op}' not found in operators.yaml"
            )

    # Check: if test in commit but no benchmark on disk (and vice versa)
    for op in test_ops - bench_ops:
        if not glob.glob(f"benchmark/test_{op}*.py"):
            errors.append(
                f"tests/test_{op}.py exists but no benchmark file found on disk"
            )
    for op in bench_ops - test_ops:
        if not glob.glob(f"tests/test_{op}*.py"):
            errors.append(
                f"benchmark/test_{op}*.py exists but no test file found on disk"
            )

    return errors


def find_operators_yaml(start_dir: str) -> str | None:
    """Walk up from start_dir to find conf/operators.yaml."""
    d = os.path.abspath(start_dir)
    for _ in range(10):
        candidate = os.path.join(d, "conf", "operators.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def main():
    parser = argparse.ArgumentParser(description="Check pytest marks against operators.yaml")
    parser.add_argument("files", nargs="+", help="Test/benchmark .py files to check")
    parser.add_argument(
        "--operators-yaml", default=None,
        help="Path to operators.yaml (default: auto-detect conf/operators.yaml)",
    )
    parser.add_argument(
        "--all-files", nargs="*", default=None,
        help="All committed files (for completeness check: test<->benchmark pairing)",
    )
    args = parser.parse_args()

    # Resolve files - expand directories
    py_files = []
    for f in args.files:
        if os.path.isdir(f):
            for name in sorted(os.listdir(f)):
                if name.startswith("test") and name.endswith(".py"):
                    py_files.append(os.path.join(f, name))
        elif f.endswith(".py"):
            py_files.append(f)

    if not py_files:
        sys.exit(0)

    # Find operators.yaml
    yaml_path = args.operators_yaml or find_operators_yaml(os.getcwd())
    if not yaml_path or not os.path.isfile(yaml_path):
        print("ERROR: Cannot find conf/operators.yaml", file=sys.stderr)
        sys.exit(2)

    operator_ids = load_operator_ids(yaml_path)

    # Check 1-3: mark validation
    errors = check_files(py_files, operator_ids)

    # Check 4: completeness (test <-> benchmark pairing, operator existence)
    completeness_files = args.all_files if args.all_files is not None else py_files
    errors.extend(check_completeness(completeness_files, operator_ids))

    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
