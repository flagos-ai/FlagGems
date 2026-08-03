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

"""Sync pyproject.toml extras from backends.yaml.

Reads backends.yaml and generates the [project.optional-dependencies] section
in pyproject.toml, ensuring the two stay in sync.

For each backend, the extras list is: deps + compiler package.
Compiler selection: flagtree if defined, otherwise triton.

Usage:
    python3 tools/sync_pyproject_extras.py --check    # CI: exit 1 if drift detected
    python3 tools/sync_pyproject_extras.py --write    # update pyproject.toml in-place
    python3 tools/sync_pyproject_extras.py --diff     # show what would change
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BACKENDS_YAML = ROOT / "src" / "flag_gems" / "backends.yaml"
PYPROJECT_TOML = ROOT / "pyproject.toml"

# Extras not generated from backends.yaml — preserved as-is
PRESERVED_EXTRAS = {"test", "example"}

# Markers in pyproject.toml that delimit the auto-generated section
BEGIN_MARKER = "# --- BEGIN auto-generated from backends.yaml ---"
END_MARKER = "# --- END auto-generated from backends.yaml ---"


def load_backends() -> dict:
    with open(BACKENDS_YAML) as f:
        return yaml.safe_load(f)


def build_extras(backends: dict) -> dict[str, list[str]]:
    """Build extras dict from backends.yaml data."""
    extras = {}
    for name, cfg in backends["backends"].items():
        deps = list(cfg.get("deps", []))

        # Add compiler: flagtree if defined, otherwise triton.
        # Both are single package specs. flagtree installs into the
        # triton namespace, so the two are mutually exclusive.
        flagtree = cfg.get("flagtree")
        triton = cfg.get("triton")

        if flagtree:
            deps.append(flagtree)
        elif triton:
            deps.append(triton)

        extras[name] = deps
    return extras


def format_extras_toml(extras: dict[str, list[str]]) -> str:
    """Format extras as TOML [project.optional-dependencies] entries."""
    lines = []
    for name in sorted(extras.keys()):
        deps = extras[name]
        lines.append(f"{name} = [")
        for dep in deps:
            lines.append(f'    "{dep}",')
        lines.append("]")
        lines.append("")
    return "\n".join(lines)


def read_pyproject() -> str:
    return PYPROJECT_TOML.read_text()


def extract_current_extras(content: str) -> dict[str, list[str]]:
    """Parse current extras from pyproject.toml."""
    extras = {}
    # Match patterns like: name = [\n  "dep1",\n  "dep2",\n]
    pattern = re.compile(
        r"^(\S+)\s*=\s*\[\s*\n((?:\s+.*\n)*?)\s*\]",
        re.MULTILINE,
    )

    # Find the optional-dependencies section
    section_match = re.search(
        r"\[project\.optional-dependencies\]\s*\n(.*?)(?=\n\[|\Z)",
        content,
        re.DOTALL,
    )
    if not section_match:
        return extras

    section = section_match.group(1)
    for m in pattern.finditer(section):
        name = m.group(1)
        deps_block = m.group(2)
        deps = []
        for line in deps_block.strip().splitlines():
            line = line.strip().rstrip(",").strip('"').strip("'")
            if line and not line.startswith("#"):
                deps.append(line)
        extras[name] = deps
    return extras


def replace_generated_section(content: str, generated: str) -> str:
    """Replace the auto-generated section between markers, or insert markers."""
    if BEGIN_MARKER in content and END_MARKER in content:
        # Replace between markers
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        replacement = f"{BEGIN_MARKER}\n{generated}{END_MARKER}"
        return pattern.sub(replacement, content)
    else:
        # Find [project.optional-dependencies] and insert markers
        section_pattern = re.compile(r"(\[project\.optional-dependencies\]\s*\n)")
        match = section_pattern.search(content)
        if not match:
            print(
                "ERROR: [project.optional-dependencies] not found in pyproject.toml",
                file=sys.stderr,
            )
            sys.exit(1)

        # Find the end of the section (next [section] or EOF)
        section_start = match.end()
        next_section = re.search(r"\n\[(?!project\.optional)", content[section_start:])
        if next_section:
            section_end = section_start + next_section.start()
        else:
            section_end = len(content)

        # Extract preserved extras from current section
        current_section = content[section_start:section_end]
        preserved_lines = []
        for extra_name in PRESERVED_EXTRAS:
            # Find and keep preserved extra blocks
            extra_pattern = re.compile(
                rf"^({re.escape(extra_name)}\s*=\s*\[.*?\])\s*$",
                re.MULTILINE | re.DOTALL,
            )
            m = extra_pattern.search(current_section)
            if m:
                preserved_lines.append(m.group(1))

        preserved_block = "\n\n".join(preserved_lines)
        if preserved_block:
            preserved_block = "\n" + preserved_block + "\n"

        new_section = f"{BEGIN_MARKER}\n{generated}{END_MARKER}\n{preserved_block}"
        return content[:section_start] + new_section + content[section_end:]


def compare_extras(
    current: dict[str, list[str]],
    generated: dict[str, list[str]],
) -> list[str]:
    """Return list of differences (order-insensitive)."""
    diffs = []
    all_names = sorted(set(current.keys()) | set(generated.keys()))
    for name in all_names:
        if name in PRESERVED_EXTRAS:
            continue
        cur = set(current.get(name, []))
        gen = set(generated.get(name, []))
        if cur != gen:
            added = sorted(gen - cur)
            removed = sorted(cur - gen)
            if added or removed:
                diffs.append(f"  {name}:")
                for dep in added:
                    diffs.append(f"    + {dep}")
                for dep in removed:
                    diffs.append(f"    - {dep}")
    return diffs


def main():
    parser = argparse.ArgumentParser(
        description="Sync pyproject.toml extras from backends.yaml"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="Check for drift (exit 1 if different)"
    )
    group.add_argument(
        "--write", action="store_true", help="Update pyproject.toml in-place"
    )
    group.add_argument("--diff", action="store_true", help="Show what would change")
    args = parser.parse_args()

    backends = load_backends()
    generated = build_extras(backends)

    content = read_pyproject()
    current = extract_current_extras(content)

    diffs = compare_extras(current, generated)

    if args.check:
        if diffs:
            print("ERROR: pyproject.toml extras drift detected!")
            print("Run 'python3 tools/sync_pyproject_extras.py --write' to fix.\n")
            print("\n".join(diffs))
            sys.exit(1)
        else:
            print("OK: pyproject.toml extras match backends.yaml")
            sys.exit(0)

    elif args.diff:
        if diffs:
            print("Differences found:")
            print("\n".join(diffs))
        else:
            print("No differences.")

    elif args.write:
        if not diffs:
            print("No changes needed.")
            return

        generated_toml = format_extras_toml(generated)
        new_content = replace_generated_section(content, generated_toml)
        PYPROJECT_TOML.write_text(new_content)
        print(f"Updated {PYPROJECT_TOML}")
        print("Changes:")
        print("\n".join(diffs))


if __name__ == "__main__":
    main()
