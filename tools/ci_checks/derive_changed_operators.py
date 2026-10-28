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
TEST_FILE_RE = re.compile(r"^tests/test_(.+)\.py$")
# Shared utility modules (e.g. src/flag_gems/utils/foreach.py). They have no
# operator id of their own; they are attributed to the ops modules that import
# them.
UTILS_FILE_RE = re.compile(r"^src/flag_gems/utils/(.+)\.py$")


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


def declared_markers() -> set[str]:
    """Return the pytest markers declared by the test files.

    A marker that no test declares selects nothing, so emitting it makes the
    caller's ``pytest -m`` run deselect everything. Used to drop markers
    derived from backend helper modules (whose file stem is not an operator id
    and which no test marks).
    """
    declared: set[str] = set()
    for path in Path(TESTS_DIR).glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "mark"
            ):
                declared.add(node.attr)
    return declared


def get_diff_files(base_sha: str, head_sha: str) -> list[str]:
    """Get list of changed files introduced by the PR branch.

    Uses a three-dot diff (``base...head``), which compares ``head`` against
    the merge-base of ``base`` and ``head``. This restricts the result to
    changes the PR branch actually introduced, excluding files that diverged
    on the base branch after the PR was created. This matches the semantics of
    GitHub's "Files changed" view.
    """
    # Fail early with a clear message if either endpoint isn't a local commit.
    # Otherwise `git diff base...head` reports the internal "Invalid symmetric
    # difference expression", which looks like a syntax error but actually means
    # the SHA's commit object is absent (e.g. robust-checkout didn't fetch the
    # base). git's own wording misleads whoever reads the CI log.
    for label, sha in (("base", base_sha), ("head", head_sha)):
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            print(
                f"::error::{label} SHA {sha} is not present in the local repo "
                f"(its commit object was never fetched), so base...head diffs "
                f"cannot resolve. Ensure the checkout fetches this ref.",
                file=sys.stderr,
            )
            sys.exit(2)

    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
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


def resolve_multi_op_impl(filepath: str, all_operators: dict) -> list[str]:
    """Resolve the operator ids exported by a shared backend impl module.

    Some backend implementation files implement several operators at once (e.g.
    ``_kunlunxin/ops/_foreach_binary_scalar.py`` exports every
    ``_foreach_*.Scalar``/``.ScalarList`` overload), so their file stem is not
    an operator id.  The ops ``__init__.py`` re-exports each of them as
    ``from ._module import (name, ...)``, and each ``name`` is an
    ``operators.yaml`` ``for`` value with ``.`` replaced by ``_``, so names can
    be resolved back to operator ids.

    Return the resolved ids, or [] when the file is not a multi-op impl (or
    nothing can be resolved).
    """
    init_path = Path(filepath).with_name("__init__.py")
    if not init_path.exists():
        return []
    # Reverse map: "<for value with . -> _>" (the re-export name) -> op id.
    # Example: _foreach_mul_.Scalar -> _foreach_mul__Scalar -> foreach_mul_scalar_.
    rev = {}
    # Some ops declare an overload-qualified ``for`` value but are re-exported
    # under the plain name (``_foreach_norm.Scalar`` -> ``_foreach_norm``), so
    # fall back to the base name when it is unambiguous.
    base = {}
    for op in all_operators.values():
        for f in op.get("for", []) or []:
            rev[f.replace(".", "_")] = op["id"]
            ids = base.setdefault(f.split(".")[0], [])
            if op["id"] not in ids:
                ids.append(op["id"])
    try:
        src = init_path.read_text(encoding="utf-8")
    except OSError:
        return []
    stem = Path(filepath).stem
    # Both import styles occur: backend packages use a relative re-export
    # (``from ._module import (...)``), the generic ``flag_gems/ops/__init__.py``
    # uses an absolute one (``from flag_gems.ops._module import (...)``). Accept
    # either, anchored on the module's last component being exactly the stem.
    import_re = rf"from\s+[\w.]*\.{re.escape(stem)}\s+import\s+"
    m = re.search(import_re + r"\((.*?)\)", src, re.DOTALL)
    if m is None:
        m = re.search(import_re + r"([^\n()]+)", src)
    if m is None:
        return []
    ids = []
    for token in re.split(r"[,\s()]+", m.group(1)):
        name = token.split("#")[0].strip()
        if not name:
            continue
        op_id = rev.get(name)
        if op_id is None:
            candidates = base.get(name, [])
            op_id = candidates[0] if len(candidates) == 1 else None
        if op_id is not None and op_id not in ids:
            ids.append(op_id)
    return ids


def resolve_impl_operators(stem: str, filepath: str, all_operators: dict) -> list[str]:
    """Operator ids implemented by one ops module, given its path inside ``ops/``.

    ``stem`` is the file path relative to its ``ops/`` directory (``abs``,
    ``sub/file``); keep passing it rather than re-deriving from ``filepath`` so
    nested generic ops keep the ``sub_file`` id form. Falls back to the raw stem
    when nothing can be resolved, leaving behavior for unknown files unchanged.
    """
    # Handle subdirectory ops like ops/sub/file.py -> sub/file
    # But most ops are flat: ops/abs.py -> abs
    op_id = stem.replace("/", "_")
    if op_id == "__init__":
        return []
    # Try exact match first
    if op_id in all_operators:
        return [op_id]
    # The file stem is not an operator id: it may be a shared impl module
    # exporting several operators (e.g. _foreach_binary_scalar.py). Resolve the
    # real ids from the ops/__init__.py re-exports.
    resolved = resolve_multi_op_impl(filepath, all_operators)
    return resolved or [op_id]


def utils_dependents(filepath: str) -> list[str]:
    """Ops implementation files that import a changed ``utils/`` module.

    A shared module such as ``src/flag_gems/utils/foreach.py`` -- the executor
    behind every ``_foreach_*`` operator -- has no operator id of its own and
    matches none of the other patterns, so a change to it would select no tests
    at all. Attribute it to the ops modules that import it; they resolve to
    their operator ids through :func:`resolve_impl_operators`.
    """
    stem = Path(filepath).stem
    if stem == "__init__":
        return []
    dotted = f"flag_gems.utils.{stem}"
    pattern = re.compile(rf"(?:^|\s)(?:from|import)\s+{re.escape(dotted)}(?:\s|\.|$)")
    dependents = []
    for candidate in sorted(Path("src/flag_gems").rglob("*.py")):
        rel = candidate.as_posix()
        m = OPS_FILE_RE.match(rel) or BACKEND_OPS_FILE_RE.match(rel)
        if m is None:
            continue
        try:
            src = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if pattern.search(src):
            dependents.append((m.group(1), rel))
    return dependents


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
            changed_ops.update(
                resolve_impl_operators(m.group(1), filepath, all_operators)
            )
            continue

        # Case 5: shared utils module changed. ``src/flag_gems/utils/foreach.py``
        # is the executor behind every ``_foreach_*`` operator and matches no
        # other pattern, so without this the derivation selects no tests and the
        # change would be reviewed with an empty test run.
        m = UTILS_FILE_RE.match(filepath)
        if m:
            for stem, dependent in utils_dependents(filepath):
                changed_ops.update(
                    resolve_impl_operators(stem, dependent, all_operators)
                )
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
    args = parser.parse_args()

    if args.changed_files:
        changed_files = [f for f in args.changed_files.split() if f.strip()]
    elif args.base and args.head:
        changed_files = get_diff_files(args.base, args.head)
    else:
        parser.error("provide either --changed-files or both --base and --head")

    all_operators = load_operators_yaml()
    all_op_ids = set(all_operators)
    changed_ops = derive_operators(changed_files, all_operators)

    if args.ops_only:
        # Machine-readable: just the markers, one per line. Apply the marker
        # convention here so callers select the right tests (a raw id like
        # _pad_enum would deselect everything). Markers that no test declares
        # are dropped: a backend helper module (e.g.
        # _kunlunxin/ops/fused_moe_kernel.py) has no marker of its own, and
        # emitting one would make the caller's pytest run deselect everything.
        declared = declared_markers()
        for op in changed_ops:
            marker = expected_marker(op, all_op_ids)
            if op in all_op_ids or marker in declared:
                print(marker)
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
