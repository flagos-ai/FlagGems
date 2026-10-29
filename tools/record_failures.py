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
Record failed operators of a daily run into a structured JSON database.

tests/conftest.py dumps the result of every test (together with the
operator marks attached to it) into accuracy_result.json when the pytest
session finishes. This script aggregates the failures of one daily run
into .github/daily_failures.json (committed to the repository), so the
occasional, hard-to-reproduce operator failures are persisted and can be
tracked as the code base evolves (issue #4100).

The database layout is::

    {
      "runs": [
        {
          "run": "20260831",                  # daily run tag
          "date": "2026-08-31",               # ISO date of the run
          "failed_ops": {"add": 2, ...},      # failures per operator mark
          "failed_tests": {                   # failed test -> its op marks
              "tests/test_binary.py::test_add[...](...)": ["add"]
          }
        },
        ...
      ]
    }

Re-recording the same run tag replaces its entry, so the step is safe to
retry. Records older than --window-days are pruned to keep the database
bounded (tools/degrade_stages.py looks at the same window).

Usage:
    python tools/record_failures.py --run 20260831
    python tools/record_failures.py --run 20260831-manual \
        --results accuracy_result.json --db .github/daily_failures.json
"""

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "accuracy_result.json"
DEFAULT_DB = ROOT / ".github" / "daily_failures.json"


def run_tag_to_date(run_tag):
    """Derive the ISO date of the run from its tag (e.g. 20260831-manual)."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(?:-|$)", run_tag or "")
    if m:
        try:
            return date(*(int(g) for g in m.groups())).isoformat()
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Record failed operators of a daily run (issue #4100)"
    )
    parser.add_argument(
        "--run",
        default="",
        help="run tag of the daily job, e.g. 20260831 or 20260831-manual",
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="per-test results JSON written by tests/conftest.py",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="failure database to update",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=15,
        help="prune records older than this many days (default: 15)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="ISO date of the run (default: derived from --run, else today)",
    )
    args = parser.parse_args()

    run_tag = args.run.strip() or datetime.now(timezone.utc).strftime("%Y%m%d")
    run_date = args.date or run_tag_to_date(run_tag)
    if run_date is None:
        run_date = datetime.now(timezone.utc).date().isoformat()

    results_path = Path(args.results)
    if not results_path.exists():
        print(
            f"[record_failures] results file not found: {results_path}, "
            "nothing to record"
        )
        return 0

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[record_failures] failed to parse {results_path}: {e}")
        return 0

    failed_tests = {}
    failed_ops = {}
    for nodeid, info in data.items():
        if not isinstance(info, dict) or info.get("result") != "failed":
            continue
        op_marks = sorted(set(info.get("opname") or []))
        failed_tests[nodeid] = op_marks
        for op in op_marks:
            failed_ops[op] = failed_ops.get(op, 0) + 1

    db_path = Path(args.db)
    try:
        db = json.loads(db_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        db = {}
    runs = [r for r in db.get("runs", []) if r.get("run") != run_tag]
    runs.append(
        {
            "run": run_tag,
            "date": run_date,
            "failed_ops": dict(sorted(failed_ops.items())),
            "failed_tests": dict(sorted(failed_tests.items())),
        }
    )

    cutoff = (
        date.fromisoformat(run_date) - timedelta(days=args.window_days)
    ).isoformat()
    runs = [r for r in runs if (r.get("date") or "") >= cutoff]
    runs.sort(key=lambda r: (r.get("date") or "", r.get("run") or ""))
    db["runs"] = runs

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(
        json.dumps(db, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    print(
        f"[record_failures] run {run_tag} ({run_date}): "
        f"{len(failed_tests)} failed tests, {len(failed_ops)} affected ops; "
        f"database {db_path} holds {len(runs)} runs"
    )
    for op, count in sorted(failed_ops.items()):
        print(f"[record_failures]   {op}: {count} failed test(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
