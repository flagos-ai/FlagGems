from __future__ import annotations

import argparse
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
DOCS_MD = REPO_ROOT / "docs" / "operators.md"
DOCS_TXT = REPO_ROOT / "docs" / "operators.txt"

BUILTIN_MARKS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "timeout",
    "tryfirst",
    "trylast",
}

NON_OPERATOR_MARKS = {
    "inplace",
    "enable",
    "enable_with_exclude",
    "only_enable",
    "linear",
    "matmul",
}

EXCLUDED_MARKS = BUILTIN_MARKS | NON_OPERATOR_MARKS
VALIDATION_EXCLUDES = {"test_named_ops.py"}
POST_FILTER_EXACT = {"dgeglu", "dreglu", "conv_depthwise2d", "contiguous"}
POST_FILTER_SUFFIXES = ("_out", "_padding", "_backward", "_bwd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract operator pytest marks from tests and generate docs outputs."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate operator tests without writing docs outputs.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print extracted operators to stdout.",
    )
    return parser.parse_args()


def iter_test_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test"):
            yield node


def extract_mark_name(decorator: ast.expr) -> str | None:
    node = decorator.func if isinstance(decorator, ast.Call) else decorator
    parts: list[str] = []

    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if not isinstance(node, ast.Name):
        return None

    parts.append(node.id)
    dotted = ".".join(reversed(parts))
    prefix = "pytest.mark."
    if not dotted.startswith(prefix):
        return None
    return dotted[len(prefix) :]


def parse_test_file(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    operators: set[str] = set()
    function_marks: dict[str, list[str]] = {}

    for func in iter_test_functions(tree):
        marks = []
        for decorator in func.decorator_list:
            mark = extract_mark_name(decorator)
            if mark is None or mark in EXCLUDED_MARKS:
                continue
            marks.append(mark)
            operators.add(mark)
        function_marks[func.name] = marks

    return sorted(operators), function_marks


def post_filter_operators(operators: set[str]) -> list[str]:
    filtered = {
        operator
        for operator in operators
        if operator not in POST_FILTER_EXACT
        and not operator.endswith(POST_FILTER_SUFFIXES)
    }
    return sorted(filtered)


def collect_operators() -> list[str]:
    operators: set[str] = set()
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        file_operators, _ = parse_test_file(path)
        operators.update(file_operators)
    return post_filter_operators(operators)


def validate_operator_tests() -> list[str]:
    failures: list[str] = []
    for path in sorted(TESTS_DIR.glob("test_*_ops.py")):
        if path.name in VALIDATION_EXCLUDES:
            continue
        _, function_marks = parse_test_file(path)
        for func_name, marks in sorted(function_marks.items()):
            if not marks:
                failures.append(f"{path.relative_to(REPO_ROOT)}::{func_name}")
    return failures


def render_markdown(operators: list[str]) -> str:
    lines = ["## Operator List", ""]
    lines.extend("- " + operator.replace("_", "\\_") for operator in operators)
    lines.append("")
    return "\n".join(lines)


def render_text(operators: list[str]) -> str:
    return "\n".join(operators) + "\n"


def write_outputs(operators: list[str]) -> None:
    DOCS_MD.write_text(render_markdown(operators))
    DOCS_TXT.write_text(render_text(operators))


def main() -> int:
    args = parse_args()

    failures = validate_operator_tests()
    if failures:
        print("[ERROR] Missing operator pytest marks in:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    operators = collect_operators()

    if args.stdout:
        print("\n".join(operators))

    if not args.check_only:
        write_outputs(operators)
        print(f"Generated {DOCS_MD.relative_to(REPO_ROOT)} ({len(operators)})")
        print(f"Generated {DOCS_TXT.relative_to(REPO_ROOT)} ({len(operators)})")
    else:
        print(f"Validated operator marks ({len(operators)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
