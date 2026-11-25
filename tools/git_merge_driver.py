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

"""
Git merge driver for FlagGems sorted registration files.

Merges conf/operators.yaml, src/flag_gems/__init__.py (_FULL_CONFIG),
and src/flag_gems/ops/__init__.py (__all__) by set-union + re-sort,
avoiding conflicts that arise purely from two branches inserting entries
in the same alphabetical neighbourhood.

Register once per clone:
    git config merge.flaggems-sort.driver \\
        "python tools/git_merge_driver.py %O %A %B %L %P"
    git config merge.flaggems-sort.name "FlagGems sorted-list merge driver"

The .gitattributes file assigns merge=flaggems-sort to the relevant
files; git then invokes this driver instead of the standard line-diff
merge whenever those files conflict.

Exit codes:
    0  Merge succeeded; %A contains the resolved result.
    1  Unresolvable conflict (e.g. same entry modified differently in
       both branches); git falls back to its standard conflict markers.
"""

import ast
import sys
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _casefold_key(s: str) -> str:
    return s.casefold()


# ---------------------------------------------------------------------------
# operators.yaml
# ---------------------------------------------------------------------------


def _load_yaml_ops(path: Path) -> tuple[dict, list[dict]]:
    """Return (raw_data, ops_list). raw_data retains header fields."""
    with open(path) as f:
        data = _yaml.safe_load(f)
    return data, data.get("ops", [])


def _yaml_header_text(path: Path) -> str:
    """Return lines before the 'ops:' key so copyright is preserved."""
    lines = path.read_text().splitlines()
    header = []
    for line in lines:
        header.append(line)
        if line.strip() == "ops:":
            break
    return "\n".join(header) + "\n"


def merge_operators_yaml(base: Path, ours: Path, theirs: Path) -> bool:
    """
    Merge operators.yaml by id.

    - Entries present only in ours or only in theirs are added.
    - Entries deleted in ours (present in base, absent in ours) are removed,
      even if theirs still has them (ours deletion wins, matching git's default
      resolve behaviour).
    - Entries that exist in both ours and theirs with different content are a
      true conflict; return False so git can show conflict markers.
    - Result is written back to `ours` and sorted by id.casefold().
    """
    if _yaml is None:
        return False  # pyyaml unavailable — fall back to git's text merge
    _, base_ops = _load_yaml_ops(base)
    ours_data, ours_ops = _load_yaml_ops(ours)
    _, theirs_ops = _load_yaml_ops(theirs)

    base_by_id = {op["id"]: op for op in base_ops}
    ours_by_id = {op["id"]: op for op in ours_ops}
    theirs_by_id = {op["id"]: op for op in theirs_ops}

    all_ids = set(base_by_id) | set(ours_by_id) | set(theirs_by_id)

    merged: dict[str, dict] = {}
    for op_id in all_ids:
        in_base = op_id in base_by_id
        in_ours = op_id in ours_by_id
        in_theirs = op_id in theirs_by_id

        if not in_ours and not in_theirs:
            continue  # deleted by both

        if in_ours and not in_theirs:
            if not in_base:
                merged[op_id] = ours_by_id[op_id]  # added by ours only
            else:
                merged[op_id] = ours_by_id[op_id]  # kept by ours, deleted by theirs → ours wins
            continue

        if in_theirs and not in_ours:
            if not in_base:
                merged[op_id] = theirs_by_id[op_id]  # added by theirs only
            # else: deleted by ours → ours deletion wins, skip
            continue

        # Present in both ours and theirs
        if ours_by_id[op_id] == theirs_by_id[op_id]:
            merged[op_id] = ours_by_id[op_id]
        elif not in_base:
            # Both added with different content — true conflict
            return False
        elif ours_by_id[op_id] == base_by_id[op_id]:
            merged[op_id] = theirs_by_id[op_id]  # only theirs changed
        elif theirs_by_id[op_id] == base_by_id[op_id]:
            merged[op_id] = ours_by_id[op_id]  # only ours changed
        else:
            return False  # both changed differently — true conflict

    sorted_ops = sorted(merged.values(), key=lambda op: op["id"].casefold())

    # Preserve copyright header; rebuild ops body via pyyaml
    header = _yaml_header_text(ours)
    ours_data["ops"] = sorted_ops
    dumped = _yaml.dump(ours_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    # Strip everything up to and including "ops:" from the dump; we keep our
    # header which already has that line.
    dump_lines = dumped.splitlines()
    ops_start = next((i + 1 for i, l in enumerate(dump_lines) if l.strip() == "ops:"), 0)
    ops_body = "\n".join(dump_lines[ops_start:])

    ours.write_text(header + ops_body + "\n")
    return True


# ---------------------------------------------------------------------------
# _FULL_CONFIG in src/flag_gems/__init__.py
# ---------------------------------------------------------------------------


def _parse_full_config(source: str) -> tuple[int, int, list[tuple[str, str]]]:
    """
    Return (start_line_0idx, end_line_0idx, entries) where each entry is
    (key_string, raw_source_text_with_trailing_comma).
    start/end are the lines of `_FULL_CONFIG = (` and `)`.
    """
    tree = ast.parse(source)
    config_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_FULL_CONFIG"
                for t in node.targets
            )
            and isinstance(node.value, ast.Tuple)
        ):
            config_node = node
            break

    if config_node is None:
        return -1, -1, []

    lines = source.splitlines()
    entries = []
    prev_end = config_node.lineno - 1  # 0-indexed

    for elt in config_node.value.elts:
        if isinstance(elt, ast.Tuple) and elt.elts:
            key_node = elt.elts[0]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
                sl = elt.lineno - 1
                el = elt.end_lineno - 1
                # Absorb preceding comment lines
                cs = sl
                while cs - 1 > prev_end and lines[cs - 1].strip().startswith("#"):
                    cs -= 1
                raw = "\n".join(lines[cs : el + 1])
                entries.append((key, raw))
                prev_end = el

    return config_node.lineno - 1, config_node.end_lineno - 1, entries


def merge_full_config(base: Path, ours: Path, theirs: Path) -> bool:
    base_src = base.read_text()
    ours_src = ours.read_text()
    theirs_src = theirs.read_text()

    _, _, base_entries = _parse_full_config(base_src)
    start_line, end_line, ours_entries = _parse_full_config(ours_src)
    _, _, theirs_entries = _parse_full_config(theirs_src)

    if start_line < 0:
        return True  # No _FULL_CONFIG — nothing to merge

    base_by_key = dict(base_entries)
    ours_by_key = dict(ours_entries)
    theirs_by_key = dict(theirs_entries)

    all_keys = list(dict.fromkeys(
        [k for k, _ in base_entries]
        + [k for k, _ in ours_entries]
        + [k for k, _ in theirs_entries]
    ))

    merged: dict[str, str] = {}
    for key in all_keys:
        in_base = key in base_by_key
        in_ours = key in ours_by_key
        in_theirs = key in theirs_by_key

        if not in_ours and not in_theirs:
            continue
        if in_ours and not in_theirs:
            merged[key] = ours_by_key[key]
            continue
        if in_theirs and not in_ours:
            if not in_base:
                merged[key] = theirs_by_key[key]
            continue
        # Both have it
        if ours_by_key[key] == theirs_by_key[key]:
            merged[key] = ours_by_key[key]
        elif not in_base:
            return False
        elif ours_by_key[key] == base_by_key.get(key, ""):
            merged[key] = theirs_by_key[key]
        elif theirs_by_key[key] == base_by_key.get(key, ""):
            merged[key] = ours_by_key[key]
        else:
            return False

    sorted_entries = sorted(merged.values(), key=lambda raw: raw.strip().lstrip("(").split('"')[1].casefold() if '"' in raw else raw.casefold())

    ours_lines = ours_src.splitlines()
    new_lines = ours_lines[:start_line]
    new_lines.append("_FULL_CONFIG = (")
    for raw in sorted_entries:
        new_lines.extend(raw.split("\n"))
    new_lines.append(")")
    new_lines.extend(ours_lines[end_line + 1:])

    ours.write_text("\n".join(new_lines) + "\n")
    return True


# ---------------------------------------------------------------------------
# __all__ in ops/__init__.py (and other __init__.py files)
# ---------------------------------------------------------------------------


def _parse_all(source: str) -> tuple[int, int, list[str]]:
    """Return (start_0idx, end_0idx, items). -1,-1,[] if no __all__ found."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return -1, -1, []

    all_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        and isinstance(node.value, ast.List)
    ]
    if len(all_nodes) != 1:
        return -1, -1, []

    node = all_nodes[0]
    items = [
        elt.value
        for elt in node.value.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]
    return node.lineno - 1, node.end_lineno - 1, items


def merge_all_list(base: Path, ours: Path, theirs: Path) -> bool:
    base_src = base.read_text()
    ours_src = ours.read_text()
    theirs_src = theirs.read_text()

    _, _, base_items = _parse_all(base_src)
    start_line, end_line, ours_items = _parse_all(ours_src)
    _, _, theirs_items = _parse_all(theirs_src)

    if start_line < 0:
        return True  # No __all__ list to merge

    base_set = set(base_items)
    ours_set = set(ours_items)
    theirs_set = set(theirs_items)

    # Items added by either side
    added = (ours_set - base_set) | (theirs_set - base_set)
    # Items deleted by either side (deletion wins, same as git default)
    deleted_by_ours = base_set - ours_set
    deleted_by_theirs = base_set - theirs_set
    deleted = deleted_by_ours | deleted_by_theirs

    merged = sorted((base_set | added) - deleted, key=str.casefold)

    # Detect quote style from first real item in ours block
    ours_lines = ours_src.splitlines()
    quote = '"'
    indent = 4
    for line in ours_lines[start_line + 1 : end_line + 1]:
        s = line.strip()
        if s and s != "]":
            if line != line.lstrip():
                indent = len(line) - len(line.lstrip())
            quote = '"' if '"' in s else "'"
            break
    indent_str = " " * indent

    new_lines = ours_lines[:start_line]
    new_lines.append("__all__ = [")
    for item in merged:
        new_lines.append(f"{indent_str}{quote}{item}{quote},")
    new_lines.append("]")
    new_lines.extend(ours_lines[end_line + 1:])

    ours.write_text("\n".join(new_lines) + "\n")
    return True


# ---------------------------------------------------------------------------
# Main — invoked as: git_merge_driver.py %O %A %B %L %P
# ---------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <base> <ours> <theirs> [<marker_size> <path>]", file=sys.stderr)
        return 1

    base = Path(sys.argv[1])
    ours = Path(sys.argv[2])
    theirs = Path(sys.argv[3])
    # sys.argv[4] is %L (conflict marker size), sys.argv[5] is %P (path hint)
    path_hint = sys.argv[5] if len(sys.argv) > 5 else ours.name

    ok = True

    if path_hint.endswith("operators.yaml"):
        ok = merge_operators_yaml(base, ours, theirs)
    elif path_hint.endswith("__init__.py"):
        # Apply both _FULL_CONFIG (main __init__) and __all__
        if not merge_full_config(base, ours, theirs):
            ok = False
        if ok and not merge_all_list(base, ours, theirs):
            ok = False
    else:
        return 1  # Unknown file type — let git do its default merge

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
