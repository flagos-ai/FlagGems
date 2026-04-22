#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tools.competition.task_config import TaskSpec, load_tasks


def _normalize(s: str) -> str:
    return (s or "").lower()


def _split_csv(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in re.split(r"[,;]", s) if x.strip()]


def resolve_task_ids(
    tasks: Sequence[TaskSpec],
    *,
    explicit: Optional[str],
    pr_title: str,
    changed_files: Sequence[str],
) -> List[str]:
    by_id: Dict[str, TaskSpec] = {t.task_id: t for t in tasks}

    def _alias_to_id(token: str) -> Optional[str]:
        tok = token.strip()
        if not tok:
            return None
        if tok in by_id:
            return tok
        low = _normalize(tok)
        for t in tasks:
            for a in t.aliases + [t.name]:
                if _normalize(a) == low:
                    return t.task_id
        return None

    resolved: List[str] = []

    if explicit:
        for token in _split_csv(explicit):
            tid = _alias_to_id(token)
            if tid and tid not in resolved:
                resolved.append(tid)
        return resolved

    title_low = _normalize(pr_title)
    for t in tasks:
        hit = False
        for a in t.aliases + [t.name]:
            a_low = _normalize(a)
            if a_low and a_low in title_low:
                hit = True
                break
        if hit:
            resolved.append(t.task_id)

    if resolved:
        return resolved

    changed_joined = "\n".join(changed_files)
    changed_low = _normalize(changed_joined)
    for t in tasks:
        for a in t.aliases + [t.name]:
            a_low = _normalize(a)
            if a_low and a_low in changed_low:
                if t.task_id not in resolved:
                    resolved.append(t.task_id)
                break

    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-yaml", type=Path, required=True)
    parser.add_argument("--pr-title", type=str, default="")
    parser.add_argument("--changed-files", type=str, default="")
    parser.add_argument("--explicit", type=str, default="")
    parser.add_argument("--format", choices=["plain", "json"], default="plain")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks_yaml)

    changed_files = [x for x in args.changed_files.split() if x]
    explicit = args.explicit.strip() or None

    task_ids = resolve_task_ids(
        tasks,
        explicit=explicit,
        pr_title=args.pr_title,
        changed_files=changed_files,
    )

    if args.format == "json":
        import json

        sys_out = {"task_ids": task_ids}
        print(json.dumps(sys_out, ensure_ascii=False))
    else:
        print(" ".join(task_ids))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
