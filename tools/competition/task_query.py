#!/usr/bin/env python3

import argparse
from pathlib import Path

from tools.competition.task_config import find_task, load_tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-yaml", type=Path, required=True)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument(
        "--field",
        choices=["name", "correctness_tests", "benchmark_tests"],
        required=True,
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks_yaml)
    t = find_task(tasks, args.task_id)
    if t is None:
        return 1

    if args.field == "name":
        print(t.name)
        return 0

    items = getattr(t, args.field)
    print(" ".join(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
