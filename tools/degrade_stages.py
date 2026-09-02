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
Degrade operator maturity stages based on the recorded daily failures.

Some operators only fail occasionally in the daily test workflow, which
makes the failures hard to reproduce and easy to forget. To keep the
declared operator maturity (the ``stages`` history in
conf/operators.yaml) honest, this script reads the failure database
maintained by tools/record_failures.py and demotes every operator that
failed in at least --min-failures daily runs during the last
--window-days days (issue #4100):

    stable -> beta
    beta   -> alpha

The demotion appends a new stage entry ``- <stage>: '<version>'`` to the
operator's ``stages`` history, keeping the file comments and formatting
intact (the YAML is edited textually, not re-serialized). ``--version``
defaults to the highest release version already present in the file,
i.e. the current release.

An operator is demoted by at most one level per release: if the last
stage entry is already a demotion made by this tool at the current
version, the operator is skipped until the next version bump. Operators
already at ``alpha`` (or ``removed``/without stages) are never touched.

The script only reports and edits conf/operators.yaml; committing and
pushing the result is up to the caller (see .github/workflows/daily.yaml).

Usage:
    python tools/degrade_stages.py \
        --db .github/daily_failures.json --yaml conf/operators.yaml
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ordered from the most to the least mature stage; demotion moves right.
STAGE_ORDER = ["removed", "stable", "beta", "alpha"]
DEMOTION = {"stable": "beta", "beta": "alpha"}

OP_ID_RE = re.compile(r"^  - id: (\S+)\s*$")
STAGE_HEADER_RE = re.compile(r"^    stages:\s*$")
STAGE_ENTRY_RE = re.compile(r"^      - ([A-Za-z]+): '([^']*)'\s*$")


def version_key(version):
    """Best-effort sort key for release strings like '5.4'."""
    try:
        return tuple(int(p) for p in str(version).split("."))
    except ValueError:
        return ()


def parse_blocks(lines):
    """Map operator id -> (block_start, block_end) line ranges."""
    starts = [
        (m.group(1), i)
        for i, m in (
            (line_no, OP_ID_RE.match(line)) for line_no, line in enumerate(lines)
        )
        if m
    ]
    blocks = {}
    for pos, (op_id, start) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else len(lines)
        blocks[op_id] = (start, end)
    return blocks


def parse_stage_entries(lines, start, end):
    """Return the (stage, version, line_no) entries of an operator block."""
    entries = []
    in_stages = False
    for i in range(start, end):
        line = lines[i]
        if STAGE_HEADER_RE.match(line):
            in_stages = True
            continue
        if in_stages:
            m = STAGE_ENTRY_RE.match(line)
            if m:
                entries.append((m.group(1), m.group(2), i))
            else:
                break
    return entries


def current_stage(entries):
    return entries[-1][0] if entries else None


def already_demoted(entries, demote_version):
    """True if the last entry is a demotion made by this tool in this release."""
    if len(entries) < 2:
        return False
    prev_stage = entries[-2][0]
    last_stage, last_version = entries[-1][0], entries[-1][1]
    demoted_now = STAGE_ORDER.index(last_stage) > STAGE_ORDER.index(prev_stage)
    return demoted_now and last_version == demote_version


def main():
    parser = argparse.ArgumentParser(
        description="Degrade operator stages from recorded failures (issue #4100)"
    )
    parser.add_argument(
        "--db",
        default=str(ROOT / ".github" / "daily_failures.json"),
        help="failure database written by tools/record_failures.py",
    )
    parser.add_argument(
        "--yaml",
        default=str(ROOT / "conf" / "operators.yaml"),
        help="operators.yaml to edit in place",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=15,
        help="count failures within this many days (default: 15)",
    )
    parser.add_argument(
        "--min-failures",
        type=int,
        default=2,
        help="demote operators failing in at least this many runs (default: 2)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="version recorded in the demotion entry "
        "(default: highest version present in operators.yaml)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[degrade_stages] failure database not found: {db_path}")
        return 0
    try:
        db = json.loads(db_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[degrade_stages] failed to parse {db_path}: {e}")
        return 0

    today = datetime.now(timezone.utc).date()
    cutoff = (today - timedelta(days=args.window_days)).isoformat()
    fail_runs = {}  # op id -> number of failing runs in the window
    for run in db.get("runs", []):
        if (run.get("date") or "") < cutoff:
            continue
        for op in run.get("failed_ops", {}):
            fail_runs[op] = fail_runs.get(op, 0) + 1
    if not fail_runs:
        print(
            f"[degrade_stages] no failing operator in the last "
            f"{args.window_days} days, nothing to do"
        )
        return 0

    yaml_path = Path(args.yaml)
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    blocks = parse_blocks(lines)

    if args.version:
        demote_version = args.version
    else:
        all_versions = []
        for start, end in blocks.values():
            all_versions.extend(v for _, v, _ in parse_stage_entries(lines, start, end))
        known = [version_key(v) for v in all_versions if version_key(v)]
        demote_version = (
            ".".join(str(p) for p in max(known)) if known else today.strftime("%Y%m%d")
        )

    demotions = []  # (insert_at, op_id, from_stage, to_stage)
    for op_id, fail_count in sorted(fail_runs.items()):
        if fail_count < args.min_failures:
            print(
                f"[degrade_stages] {op_id}: failed in {fail_count} run(s), "
                f"below the threshold of {args.min_failures}, skipped"
            )
            continue
        if op_id not in blocks:
            print(f"[degrade_stages] {op_id}: not found in operators.yaml, " "skipped")
            continue
        start, end = blocks[op_id]
        entries = parse_stage_entries(lines, start, end)
        stage = current_stage(entries)
        target = DEMOTION.get(stage)
        if target is None:
            print(
                f"[degrade_stages] {op_id}: stage '{stage}' cannot be "
                "demoted, skipped"
            )
            continue
        if already_demoted(entries, demote_version):
            print(
                f"[degrade_stages] {op_id}: already demoted to "
                f"'{entries[-1][0]}' at version {demote_version}, skipped"
            )
            continue
        demotions.append((entries[-1][2] + 1, op_id, stage, target))

    for insert_at, op_id, from_stage, to_stage in sorted(
        demotions, key=lambda d: d[0], reverse=True
    ):
        lines.insert(insert_at, f"      - {to_stage}: '{demote_version}'")
        print(
            f"[degrade_stages] {op_id}: {from_stage} -> {to_stage} "
            f"at version {demote_version}"
        )

    if demotions:
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"[degrade_stages] {len(demotions)} operator(s) demoted in " f"{yaml_path}"
        )
    else:
        print("[degrade_stages] no operator demoted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
