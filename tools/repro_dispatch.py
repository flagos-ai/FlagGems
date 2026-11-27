#!/usr/bin/env python3
"""Collect and run selected FlagGems tests in native and dispatch modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest


class Collector:
    def __init__(self):
        self.nodes = []

    def pytest_collection_finish(self, session):
        self.nodes = [item.nodeid for item in session.items]


class Recorder:
    def __init__(self, report: Path, mode: str):
        self.report = report
        self.mode = mode
        self.nodes = {}

    def pytest_runtest_logreport(self, report):
        entry = self.nodes.setdefault(report.nodeid, {})
        entry[report.when] = report.outcome
        if report.failed:
            entry["reason"] = str(report.longrepr)

    def pytest_sessionfinish(self, session, exitstatus):
        existing = json.loads(self.report.read_text()) if self.report.exists() else {}
        existing[self.mode] = {"exit_status": int(exitstatus), "nodes": self.nodes}
        self.report.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


class EnableGems:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        import flag_gems

        with flag_gems.use_gems():
            yield


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", help="pytest target paths")
    parser.add_argument("--output-dir", type=Path, default=Path("dispatch-repro"))
    parser.add_argument(
        "--execute", action="store_true", help="run tests on the current device"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = (
        str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    targets = [str(root / t) if not Path(t).is_absolute() else t for t in args.targets]
    collector = Collector()
    status = pytest.main(["-q", "--collect-only", *targets], plugins=[collector])
    (out / "collection.log").write_text(
        f"pytest collection exit status: {int(status)}\ncollected nodes: {len(collector.nodes)}\n",
        encoding="utf-8",
    )
    if int(status) != 0 or not collector.nodes:
        print(
            f"Collection failed or returned zero nodes; see {out / 'collection.log'}",
            file=sys.stderr,
        )
        return 2
    (out / "nodeids.txt").write_text(
        "\n".join(collector.nodes) + "\n", encoding="utf-8"
    )
    print(f"Collected {len(collector.nodes)} nodes: {out / 'nodeids.txt'}")
    if not args.execute:
        print(
            "Collection only. --execute runs each node in native and Gems modes in this process."
        )
        return 0

    codes = []
    for mode in ("native", "flag_gems_dispatch"):
        os.environ["FLAGGEMS_REPRO_MODE"] = mode
        command = ["-q", "--repro-dispatch-report", str(out / "results.json")]
        if args.quick:
            command.append("--quick")
        codes.append(
            int(
                pytest.main(
                    [*command, *collector.nodes],
                    plugins=[EnableGems()] if mode != "native" else [],
                )
            )
        )
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit(main())
