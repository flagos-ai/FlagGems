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

"""Static check for NO_TORCH_COMPUTE_FALLBACK in production operator code.

Flags calls to torch compute APIs (torch.<op>, torch.nn.functional.*,
torch.ops.aten.*) under src/flag_gems/{ops,fused,experimental_ops} and the
vendor ops/fused/fla directories. Memory/creation/metadata helpers are allowed. A line may opt out
with a trailing '# allow-torch-compute' comment, which reviewers should question.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_TORCH_CALLS = {
    "empty",
    "empty_like",
    "empty_strided",
    "zeros",
    "zeros_like",
    "as_strided",
    "device",
    "Size",
    "finfo",
    "iinfo",
    "get_default_dtype",
    "is_tensor",
    "is_floating_point",
    "no_grad",
    "inference_mode",
    # creation / metadata only, no device compute
    "tensor",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "arange",
    "from_numpy",
    "result_type",
    "promote_types",
    "is_grad_enabled",
    "set_grad_enabled",
}
# torch sub-namespaces that are runtime/infra rather than tensor compute.
ALLOWED_TORCH_NAMESPACES = {
    "cuda",
    "version",
    "library",
    "distributed",
    "utils",
    "jit",
    "compiler",
    "_C",
    "testing",
    "amp",
    # vendor device runtimes (stream/event/graph/device management)
    "npu",
    "musa",
    "mlu",
    "xpu",
    "mps",
    "gcu",
    "ixuca",
}
PRAGMA = "allow-torch-compute"
CORE_DIRS = ("ops", "fused", "experimental_ops")
VENDOR_SUBDIRS = ("ops", "fused", "fla")

# Per-file exemptions for torch compute calls that dispatch to a vendor-native
# kernel rather than falling back to CPU/CUDA.  Key: path suffix (posix);
# value: set of fully-qualified call names that are allowed in that file.
FILE_ALLOWLIST: dict[str, set[str]] = {}


def dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def functional_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "torch.nn.functional" and item.asname:
                    aliases.add(item.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch.nn":
                for item in node.names:
                    if item.name == "functional":
                        aliases.add(item.asname or item.name)
    return aliases


def classify(name: str, aliases: set[str]) -> bool:
    parts = name.split(".")
    if parts[0] in aliases:
        return True
    if parts[0] != "torch" or len(parts) < 2:
        return False
    if name.startswith("torch.nn.functional."):
        return True
    if name.startswith("torch.ops.aten."):
        return True
    if parts[1] in ALLOWED_TORCH_NAMESPACES or parts[1] in ALLOWED_TORCH_CALLS:
        return False
    # torch.ops.<custom> and torch.nn.* modules are not compute calls we can judge.
    if parts[1] in ("ops", "nn"):
        return False
    # Only lowercase names are functions; torch.Tensor / torch.dtype are types.
    return parts[1][:1].islower()


def added_lines(path: Path, base: str, head: str | None) -> set[int]:
    """Line numbers added relative to base; head=None compares with the working tree."""
    rev = f"{base}...{head}" if head else base
    out = subprocess.run(
        ["git", "diff", "-U0", "--no-color", rev, "--", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines: set[int] = set()
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", out, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def check_file(path: Path, only_lines: set[int] | None = None) -> list[str]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: cannot parse ({exc.msg})"]
    lines = source.splitlines()
    aliases = functional_aliases(tree)
    posix = path.as_posix()
    file_allowed = next(
        (v for k, v in FILE_ALLOWLIST.items() if posix.endswith(k)), set()
    )
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        if not name or not classify(name, aliases):
            continue
        if name in file_allowed:
            continue
        if PRAGMA in lines[node.lineno - 1]:
            continue
        if only_lines is not None and node.lineno not in only_lines:
            continue
        found.append(f"{path}:{node.lineno}: forbidden torch compute call {name}()")
    return found


def default_files(root: Path) -> list[Path]:
    pkg = root / "src" / "flag_gems"
    files: set[Path] = set()
    for sub in CORE_DIRS:
        files.update((pkg / sub).rglob("*.py"))
    for vendor in (pkg / "runtime" / "backend").glob("_*"):
        for sub in VENDOR_SUBDIRS:
            files.update(vendor.rglob(f"{sub}/**/*.py"))
    return sorted(p for p in files if p.is_file())


def in_scope(path: Path) -> bool:
    posix = path.as_posix()
    if any(f"src/flag_gems/{sub}/" in posix for sub in CORE_DIRS):
        return True
    if "src/flag_gems/runtime/backend/_" in posix:
        return any(f"/{sub}/" in posix for sub in VENDOR_SUBDIRS)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files to check (default: all production operator files)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--base",
        help="only report calls on lines added between this git ref and --head",
    )
    parser.add_argument(
        "--head", help="head ref for --base (default: the working tree)"
    )
    parser.add_argument(
        "--warn-only", action="store_true", help="report violations but exit 0"
    )
    args = parser.parse_args()

    if args.paths:
        files = [
            p for p in args.paths if p.suffix == ".py" and p.is_file() and in_scope(p)
        ]
    else:
        files = default_files(args.root)

    problems: list[str] = []
    for path in files:
        only = added_lines(path, args.base, args.head) if args.base else None
        problems.extend(check_file(path, only))

    for line in problems:
        print(line)
    print(f"checked {len(files)} files, {len(problems)} violation(s)")
    if problems and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
