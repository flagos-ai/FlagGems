from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    name: str
    aliases: List[str]
    correctness_tests: List[str]
    benchmark_tests: List[str]


def load_tasks(tasks_yaml: Path) -> List[TaskSpec]:
    with open(tasks_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tasks: List[TaskSpec] = []
    for t in data.get("tasks", []) or []:
        task_id = str(t.get("id", "")).strip()
        name = str(t.get("name", "")).strip()
        if not task_id or not name:
            continue

        aliases = [str(x).strip() for x in (t.get("aliases") or []) if str(x).strip()]
        if task_id not in aliases:
            aliases = [task_id] + aliases

        correctness_tests = [
            str(x).strip() for x in (t.get("correctness_tests") or []) if str(x).strip()
        ]
        benchmark_tests = [
            str(x).strip() for x in (t.get("benchmark_tests") or []) if str(x).strip()
        ]

        tasks.append(
            TaskSpec(
                task_id=task_id,
                name=name,
                aliases=aliases,
                correctness_tests=correctness_tests,
                benchmark_tests=benchmark_tests,
            )
        )

    return tasks


def find_task(tasks: List[TaskSpec], task_id: str) -> TaskSpec | None:
    for t in tasks:
        if t.task_id == task_id:
            return t
    return None
