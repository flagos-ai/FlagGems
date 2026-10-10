#!/usr/bin/env python3
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

"""Derive which operators are affected by a PR based on git diff.

Outputs:
  - changed_operators: JSON list of operator IDs
  - changed_files: JSON list of changed file paths
  - has_changes: 'true' or 'false'

Exit codes:
  0 - success
  2 - script internal error
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

# Paths relative to repo root
OPERATORS_YAML = "conf/operators.yaml"
OPS_DIR = "src/flag_gems/ops"
TESTS_DIR = "tests"
INIT_FILE = "src/flag_gems/__init__.py"

# Patterns that map file paths to operator IDs
OPS_FILE_RE = re.compile(r"^src/flag_gems/ops/(.+)\.py$")
# Backend operator implementations, e.g.
#   src/flag_gems/runtime/backend/_kunlunxin/ops/attention.py -> attention
#   src/flag_gems/runtime/backend/_nvidia/hopper/ops/mm.py     -> mm
# The operator id is the file stem, matching the generic ops/ convention and
# the pytest marker each test uses.
BACKEND_OPS_FILE_RE = re.compile(
    r"^src/flag_gems/runtime/backend/_[^/]+/(?:[^/]+/)*ops/(.+)\.py$"
)
FUSED_FILE_RE = re.compile(
    r"^src/flag_gems/(?:runtime/backend/_[^/]+/(?:[^/]+/)*)?fused/(.+)\.py$"
)
TEST_FILE_RE = re.compile(r"^tests/test_(.+)\.py$")
TEST_ALIASES_YAML = "conf/ci_test_aliases.yaml"

# Some backend modules implement several operators, or use a different name
# from their public API. Keep their complete test files together: a single
# filename-derived marker can miss every affected test.
BACKEND_TEST_FILES = {
    "src/flag_gems/runtime/backend/_kunlunxin/ops/div.py": (
        "tests/test_div.py",
        "tests/test_divide.py",
        "tests/test_true_divide.py",
        "tests/test_trunc_divide.py",
        "tests/test_floor_divide.py",
        "tests/test_remainder.py",
    ),
    "src/flag_gems/runtime/backend/_kunlunxin/ops/attention.py": (
        "tests/test_scaled_dot_product_attention.py",
        "tests/test_flash_attention_backward.py",
        "tests/test_flash_attention.py",
        "tests/test_flash_attn_varlen_func.py",
    ),
    "src/flag_gems/runtime/backend/_kunlunxin/fused/flashmla_sparse.py": (
        "tests/test_flash_mla_sparse_fwd.py",
    ),
}


def expected_marker(op_id: str, all_op_ids: set) -> str:
    """Return the pytest marker name for an operator id.

    Mirrors ``tools/run_tests.py:op_marker`` and
    ``tools/ci_checks/check_operator_markers.py:expected_marker`` so all callers
    agree on how an operator maps to its marker (the #6359 convention):

      1. Non-underscore ids are used verbatim (``abs`` -> ``abs``).
      2. Ids with a leading underscore drop it (``_pad_enum`` -> ``pad_enum``),
         because ``pytest.mark._pad_enum`` is rejected by attribute access.
      3. If the stripped name collides with a distinct operator id (``_stack``
         vs the separate ``stack`` op), prefix ``underscore_`` instead
         (``_stack`` -> ``underscore_stack``).
    """
    if not op_id.startswith("_"):
        return op_id
    stripped = op_id.lstrip("_")
    if stripped in all_op_ids:
        return f"underscore_{stripped}"
    return stripped


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Cannot verify initializer history: {result.stderr.strip()}")
    return result.stdout.strip()


def _commit(ref: str) -> str:
    return _git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")


def initializer_context(base: str | None, head: str | None) -> tuple[str, str]:
    """Bind a local explicit comparison or PR event to the tested checkout."""
    actual = _commit("HEAD")
    if base is not None or head is not None:
        if not base or not head:
            raise ValueError("Initializer proof requires both --base and --head")
        base, head = _commit(base), _commit(head)
        if head != actual:
            raise ValueError(
                "Initializer proof head does not match the tested checkout"
            )
        bases = _git("merge-base", "--all", base, head).splitlines()
        if len(bases) != 1 or bases[0] == head:
            raise ValueError("Initializer proof requires one distinct merge base")
        return bases[0], head
    expected = (
        os.environ.get("FLAGGEMS_CI_BASE_SHA", ""),
        os.environ.get("FLAGGEMS_CI_PR_HEAD_SHA", ""),
        os.environ.get("FLAGGEMS_CI_TESTED_SHA", ""),
    )
    if os.environ.get("FLAGGEMS_CI_EVENT") != "pull_request" or not all(
        re.fullmatch(r"[0-9a-f]{40}", value) for value in expected
    ):
        raise ValueError(
            "Initializer proof requires explicit refs or PR merge metadata"
        )
    base, pr_head, head = expected
    if head != actual:
        raise ValueError("PR event head does not match the tested checkout")
    parents = _git("rev-list", "--parents", "-n", "1", actual).split()[1:]
    if parents != [base, pr_head] or base == pr_head:
        raise ValueError(
            "Tested PR merge must have the event base and head as its two parents"
        )
    # Both parent objects must be present; a shallow single-commit checkout is
    # insufficient even when the commit object contains their names.
    _commit(base)
    _commit(pr_head)
    return base, head


def _git_file(ref: str, path: str) -> bytes | None:
    entry = _git("ls-tree", ref, "--", path)
    if not entry:
        return None
    metadata, name = entry.split("\t", 1)
    mode, kind, oid = metadata.split()
    if name != path or kind != "blob" or mode not in {"100644", "100755"}:
        raise ValueError(f"Not a regular source file at {ref}:{path}")
    result = subprocess.run(["git", "cat-file", "blob", oid], capture_output=True)
    if result.returncode:
        raise ValueError(f"Cannot read source blob at {ref}:{path}")
    return result.stdout


def _check_tested_file(ref: str, path: str) -> None:
    expected = _git_file(ref, path)
    if Path(path).is_symlink():
        raise ValueError(f"Tested file is a symbolic link: {path}")
    actual = Path(path).read_bytes() if Path(path).is_file() else None
    # Windows checkouts may use CRLF; compare the actual Python/test content.
    normalized = lambda data: data.replace(b"\r\n", b"\n") if data is not None else None
    if normalized(expected) != normalized(actual):
        raise ValueError(f"Tested file differs from {ref}: {path}")


def _initializer_parts(content: bytes | None, path: str):
    tree = ast.parse(content or b"", filename=path)
    imports, exports, events = {}, [], []
    saw_exports = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            for name in node.names:
                bound = name.asname or name.name
                if name.name == "*" or bound in imports:
                    raise ValueError(f"Ambiguous initializer import in {path}")
                value = (node.level, node.module, name.name, name.asname)
                imports[bound] = value
                events.append(("import", value))
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            value = ast.literal_eval(node.value)
            if (
                saw_exports
                or not isinstance(value, (list, tuple))
                or not all(isinstance(name, str) for name in value)
                or len(value) != len(set(value))
            ):
                raise ValueError(f"Initializer needs one literal __all__ in {path}")
            saw_exports = True
            exports = list(value)
            events.append(("exports",))
        else:
            if any(
                isinstance(child, ast.Name) and child.id == "__all__"
                for child in ast.walk(node)
            ):
                raise ValueError(f"Initializer has dynamic __all__ usage: {path}")
            events.append(("statement", ast.dump(node, include_attributes=False)))
    return imports, exports, events


def _import_module(
    ref: str, path: str, binding: tuple, visited: set, tested: str
) -> str:
    level, module, name, _alias = binding
    directory = Path(path).parent
    for _ in range(level - 1):
        directory = directory.parent
    target = directory.joinpath(*module.split("."))
    if not target.as_posix().startswith("src/flag_gems/"):
        raise ValueError(f"Initializer import leaves the source tree: {path}")
    source = target.with_suffix(".py").as_posix()
    package = (target / "__init__.py").as_posix()
    source_content = _git_file(ref, source)
    content = _git_file(ref, package)
    if source_content is not None and content is not None:
        raise ValueError(f"Ambiguous module/package import in {path}: {module}")
    # An untracked same-name package can otherwise shadow a committed module.
    _check_tested_file(tested, source)
    _check_tested_file(tested, package)
    if source_content is not None:
        return source
    key = (ref, package, name)
    if content is None or key in visited:
        raise ValueError(f"Cannot resolve initializer import {module}.{name} in {path}")
    imports, _exports, _events = _initializer_parts(content, package)
    # A re-export must be static; an assignment/call could replace its binding.
    for node in ast.parse(content, filename=package).body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        raise ValueError(f"Re-export package is not static: {package}")
    if name not in imports:
        raise ValueError(f"No static re-export for {name} in {package}")
    return _import_module(ref, package, imports[name], visited | {key}, tested)


def initializer_modules(path: str, base: str, head: str) -> set[str]:
    """Allow only import/export deltas, covering both old and new bindings."""
    _check_tested_file(head, path)
    before = _initializer_parts(_git_file(base, path), path)
    after = _initializer_parts(_git_file(head, path), path)
    old_imports, old_exports, old_events = before
    new_imports, new_exports, new_events = after
    retained = set(old_imports.values()) & set(new_imports.values())
    has_exports = ("exports",) in old_events and ("exports",) in new_events

    def stable_events(events):
        return [
            event
            for event in events
            if event[0] == "statement"
            or (event[0] == "import" and event[1] in retained)
            or (event[0] == "exports" and has_exports)
        ]

    if stable_events(old_events) != stable_events(new_events):
        raise ValueError(
            f"Initializer executable statements or ordering changed: {path}"
        )
    retained_exports = set(old_exports) & set(new_exports)
    if [name for name in old_exports if name in retained_exports] != [
        name for name in new_exports if name in retained_exports
    ]:
        raise ValueError(f"Initializer retained exports were reordered: {path}")
    names = set(old_exports) ^ set(new_exports)
    if (("exports",) in old_events) != (("exports",) in new_events):
        # Introducing/removing __all__ changes implicit star exports too.
        names |= old_imports.keys() | new_imports.keys()
    names |= {
        name
        for name in old_imports.keys() | new_imports.keys()
        if old_imports.get(name) != new_imports.get(name)
    }
    modules = set()
    for name in names:
        found = False
        for ref, imports in ((base, old_imports), (head, new_imports)):
            if name in imports:
                modules.add(_import_module(ref, path, imports[name], set(), head))
                found = True
        if not found:
            raise ValueError(f"No imported binding for changed export {name} in {path}")
    return modules


def derive_test_files(
    changed_files: list[str], *, base: str | None = None, head: str | None = None
) -> list[str]:
    """Resolve source changes to full correctness files, or fail closed.

    Aliases name test files, not pytest markers. Do not mistake an unrelated
    changed test or benchmark for coverage of an unmapped implementation.
    Every changed source or runtime configuration path needs its own mapping.
    A sibling operator or an explicitly changed test does not establish that
    it covers an otherwise unmapped helper, registration, or configuration.
    """
    alias_path = Path(TEST_ALIASES_YAML)
    aliases = yaml.safe_load(alias_path.read_text(encoding="utf-8")) or {}
    if not isinstance(aliases, dict):
        raise ValueError(f"Invalid test aliases: {TEST_ALIASES_YAML}")
    targets = set()
    context = None
    for filepath in changed_files:
        if (
            filepath.startswith("tests/")
            and Path(filepath).name.startswith("test")
            and filepath.endswith(".py")
            and not Path(filepath).is_file()
        ):
            raise ValueError(f"Missing changed correctness test {filepath}")
        if not filepath.startswith("src/flag_gems/"):
            continue
        if Path(filepath).suffix in {".md", ".rst"}:
            continue
        match = (
            OPS_FILE_RE.match(filepath)
            or BACKEND_OPS_FILE_RE.match(filepath)
            or FUSED_FILE_RE.match(filepath)
        )
        if match and Path(match.group(1)).name == "__init__":
            if context is None:
                context = initializer_context(base, head)
                _check_tested_file(context[1], TEST_ALIASES_YAML)
            modules = initializer_modules(filepath, *context)
            candidates = derive_test_files(sorted(modules))
            for candidate in candidates:
                _check_tested_file(context[1], candidate)
            targets.update(candidates)
            continue
        if filepath in BACKEND_TEST_FILES:
            candidates = list(BACKEND_TEST_FILES[filepath])
        else:
            if match is None or Path(match.group(1)).name == "__init__":
                raise ValueError(f"No correctness test mapping for {filepath}")
            stem = match.group(1)
            name = Path(stem).name
            alias = aliases.get(name)
            if alias is not None:
                if not isinstance(alias, str) or not re.fullmatch(
                    r"test_[A-Za-z0-9_]+", alias
                ):
                    raise ValueError(f"Invalid test alias for {name}: {alias!r}")
                candidates = [f"tests/{alias}.py"]
            else:
                # Existing leading-underscore test filenames take precedence.
                # Search nested test directories as well (for example DSA).
                candidates = sorted(
                    str(path.as_posix())
                    for path in Path(TESTS_DIR).rglob(f"test_{name}.py")
                )
                if not candidates and name.startswith("_"):
                    candidates = sorted(
                        str(path.as_posix())
                        for path in Path(TESTS_DIR).rglob(f"test_{name.lstrip('_')}.py")
                    )
        if not candidates:
            raise ValueError(f"No correctness test mapping for {filepath}")
        for candidate in candidates:
            if not Path(candidate).is_file():
                raise ValueError(f"Missing correctness test {candidate} for {filepath}")
            targets.add(candidate)
    return sorted(targets)


def validate_junit(path: str) -> None:
    """Reject absent, malformed, empty or all-skipped test results."""
    root = ET.parse(path).getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError(f"Invalid JUnit root in {path}")
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError(f"No tests executed: {path}")
    if (
        next(root.iter("failure"), None) is not None
        or next(root.iter("error"), None) is not None
    ):
        raise ValueError(f"Failed tests in {path}")
    if not any(case.find("skipped") is None for case in cases):
        raise ValueError(f"All tests skipped: {path}")


def get_diff_files(
    base_sha: str, head_sha: str, *, include_deleted: bool = False
) -> list[str]:
    """Get list of changed files introduced by the PR branch.

    Uses a three-dot diff (``base...head``), which compares ``head`` against
    the merge-base of ``base`` and ``head``. This restricts the result to
    changes the PR branch actually introduced, excluding files that diverged
    on the base branch after the PR was created. This matches the semantics of
    GitHub's "Files changed" view.

    Test selection includes deleted paths so removing an implementation or its
    coverage cannot look like a documentation-only change. The legacy rule
    checker keeps its existing ACMR scope.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMRD" if include_deleted else "--diff-filter=ACMR",
                f"{base_sha}...{head_sha}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError as e:
        print(f"::error::Failed to get git diff: {e.stderr}", file=sys.stderr)
        sys.exit(2)


def load_operators_yaml() -> dict[str, dict]:
    """Load operators.yaml and return a dict keyed by operator id."""
    yaml_path = Path(OPERATORS_YAML)
    if not yaml_path.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    ops = data.get("ops", [])
    return {op["id"]: op for op in ops if "id" in op}


def filename_to_operator_id(filename: str) -> str | None:
    """Convert a filename stem to a potential operator id.

    Handles cases like:
      - test_abs.py -> abs
      - test__reshape_alias.py -> _reshape_alias
      - abs.py (in ops/) -> abs
    """
    # Strip leading underscore that is part of the name, not a prefix artifact
    return filename if filename else None


def derive_operators(changed_files: list[str], all_operators: dict) -> list[str]:
    """Given changed files, derive which operator IDs are affected."""
    changed_ops = set()

    for filepath in changed_files:
        # Case 1: operators.yaml itself changed -> check all operators in diff
        if filepath == OPERATORS_YAML:
            # When operators.yaml changes, we flag all operators for full check
            # In practice, individual checks will determine what to validate
            changed_ops.add("__operators_yaml_changed__")
            continue

        # Case 2: ops source file changed (generic or backend-specific)
        m = OPS_FILE_RE.match(filepath) or BACKEND_OPS_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            # Handle subdirectory ops like ops/sub/file.py -> sub/file
            # But most ops are flat: ops/abs.py -> abs
            op_id = stem.replace("/", "_")
            if op_id == "__init__":
                continue
            # Try exact match first
            if op_id in all_operators:
                changed_ops.add(op_id)
            else:
                # Try without trailing underscore (inplace variants)
                # e.g., ops file might be abs_.py for abs_ operator
                changed_ops.add(op_id)
            continue

        # Case 3: test file changed
        m = TEST_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            if stem in all_operators:
                changed_ops.add(stem)
            else:
                # Still track it, checks can use it
                changed_ops.add(stem)
            continue

        # Case 4: __init__.py changed -> full init check needed
        if filepath == INIT_FILE:
            changed_ops.add("__init_changed__")
            continue

    # Remove sentinel markers from operator list for downstream
    sentinel_markers = {"__operators_yaml_changed__", "__init_changed__"}
    real_ops = sorted(changed_ops - sentinel_markers)

    return real_ops


def set_output(name: str, value: str):
    """Set a GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            # Use delimiter for multiline values
            if "\n" in value:
                f.write(f"{name}<<EOF\n{value}\nEOF\n")
            else:
                f.write(f"{name}={value}\n")
    else:
        # Running locally, just print
        print(f"  {name}={value}")


def main():
    parser = argparse.ArgumentParser(
        description="Derive changed operators from PR diff"
    )
    parser.add_argument("--base", help="Base commit SHA (three-dot diff with --head)")
    parser.add_argument("--head", help="Head commit SHA")
    parser.add_argument(
        "--changed-files",
        help="Space- or newline-separated list of changed file paths, used "
        "instead of a git diff. Handy for callers (e.g. tools/test-op.sh) that "
        "already have the file list but not the base/head SHAs.",
    )
    parser.add_argument(
        "--ops-only",
        action="store_true",
        help="Print only the derived pytest markers (one per line) to stdout, "
        "with no diagnostics. Intended for shell consumption. Markers follow "
        "the #6359 convention (e.g. _pad_enum -> pad_enum).",
    )
    parser.add_argument(
        "--test-files", action="store_true", help="Print full correctness test paths"
    )
    parser.add_argument(
        "--validate-junit", metavar="PATH", help="Require executed passing tests"
    )
    args = parser.parse_args()

    if args.validate_junit:
        try:
            validate_junit(args.validate_junit)
        except (OSError, ET.ParseError, ValueError) as exc:
            parser.exit(2, f"::error::{exc}\n")
        return
    if args.changed_files is not None:
        changed_files = [f for f in args.changed_files.split() if f.strip()]
    elif args.base and args.head:
        changed_files = get_diff_files(
            args.base, args.head, include_deleted=args.test_files
        )
    else:
        parser.error("provide either --changed-files or both --base and --head")

    if args.test_files:
        try:
            for test_file in derive_test_files(
                changed_files, base=args.base, head=args.head
            ):
                print(test_file)
        except (OSError, ValueError, SyntaxError, yaml.YAMLError) as exc:
            parser.exit(2, f"::error::{exc}\n")
        return

    all_operators = load_operators_yaml()
    all_op_ids = set(all_operators)
    changed_ops = derive_operators(changed_files, all_operators)

    if args.ops_only:
        # Machine-readable: just the markers, one per line. Apply the marker
        # convention here so callers select the right tests (a raw id like
        # _pad_enum would deselect everything).
        for op in changed_ops:
            print(expected_marker(op, all_op_ids))
        return

    if args.base and args.head:
        print(f"Comparing {args.base}..{args.head}")
    print(f"Changed files ({len(changed_files)}):")
    for f in changed_files[:20]:
        print(f"  {f}")
    if len(changed_files) > 20:
        print(f"  ... and {len(changed_files) - 20} more")

    print(f"Total operators in registry: {len(all_operators)}")
    print(f"Changed operators ({len(changed_ops)}):")
    for op in changed_ops[:20]:
        print(f"  {op}")
    if len(changed_ops) > 20:
        print(f"  ... and {len(changed_ops) - 20} more")

    # Set outputs
    ops_json = json.dumps(changed_ops)
    files_json = json.dumps(changed_files)
    has_changes = "true" if changed_ops else "false"

    set_output("changed_operators", ops_json)
    set_output("changed_files", files_json)
    set_output("has_changes", has_changes)

    print(f"\nhas_changes={has_changes}")


if __name__ == "__main__":
    main()
