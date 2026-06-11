#!/usr/bin/env python3
"""Create GitHub PRs for successfully fixed issues."""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import load_config, load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_TARGET_REPO = "flagos-ai/FlagGems"


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def has_commits_ahead(flaggems_dir: str, branch: str, base_branch: str) -> bool:
    result = subprocess.run(
        ["git", "log", "--oneline", f"{base_branch}..{branch}"],
        cwd=flaggems_dir,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def should_skip(issue: dict) -> str | None:
    """Return skip reason, or None if PR should be created."""
    if issue["status"] != "success":
        return f"status={issue['status']}"
    cc = issue.get("cc_result") or {}
    notes = cc.get("notes", "")
    if "already_fixed" in notes:
        return "already_fixed"
    files = cc.get("files_modified", [])
    if isinstance(files, list) and len(files) == 0 and not notes:
        return "no files modified"
    return None


def build_pr_body(issue: dict) -> str:
    cc = issue.get("cc_result") or {}
    test = cc.get("test_results") or {}
    bench = cc.get("benchmark_results") or {}
    files = cc.get("files_modified") or []

    files_str = "\n".join(f"  - `{f}`" for f in files) if files else "  - (none)"

    # Extract numeric issue ID for internal tracking
    issue_id_raw = str(issue["issue_id"])
    # "441-internal" -> "441", plain "441" stays as is
    numeric_id = issue_id_raw.split("-")[0]

    # Build issue tracking section
    issue_lines = [f"- WEEKTEST-{numeric_id}"]
    github_issue = issue.get("github_issue") or cc.get("github_issue")
    if github_issue:
        issue_lines.append(f"- Fixes #{github_issue}")

    issue_section = "\n".join(issue_lines)

    return f"""## Summary
- **Operator:** {issue['operator']}
- **Error type:** {issue['error_type']}
- **Root cause:** {cc.get('root_cause', 'N/A')}
- **Fix:** {cc.get('fix_description', 'N/A')}
- **Files modified:**
{files_str}

## Verification
- Accuracy test: {test.get('passed', '?')}/{test.get('total', '?')} passed
- Benchmark: {'passed' if bench.get('all_success') else bench.get('notes', '?')}
- Format check: {'passed' if issue.get('format_check_passed') else 'N/A'}

## Test Plan
- [ ] `{test.get('test_command', 'N/A')}`
- [ ] `{bench.get('benchmark_command', 'N/A')}`

## Issue
{issue_section}
"""


def run(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)

    flaggems_dir = config.get("flaggems_dir", os.path.dirname(os.path.dirname(script_dir)))
    base_branch = config.get("base_branch", "master")
    results_dir = os.path.join(script_dir, config.get("results_dir", "results"))
    summary_path = os.path.join(results_dir, "summary.json")
    target_repo = args.target_repo

    # Determine fork owner for cross-repo PRs (--head owner:branch)
    origin_url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=flaggems_dir,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Extract owner from https://github.com/OWNER/REPO.git or git@gcom:OWNER/REPO.git
    fork_owner = origin_url.replace(".git", "").split("/")[-2].split(":")[-1]

    if not os.path.exists(summary_path):
        logger.error(f"Summary not found: {summary_path}")
        return

    if not args.dry_run:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("gh auth failed. Run: gh auth login")
            print(result.stderr)
            return

    summary = load_summary(summary_path)
    filter_ids = set(args.issues.split(",")) if args.issues else None

    created = []
    skipped = []

    for key, issue in summary["issues"].items():
        issue_id = str(issue["issue_id"])
        operator = issue["operator"]
        branch = issue.get("branch")
        label = f"{issue_id}/{operator}"

        if filter_ids and issue_id not in filter_ids:
            continue

        skip_reason = should_skip(issue)
        if skip_reason:
            skipped.append((label, skip_reason))
            logger.info(f"[SKIP] {label}: {skip_reason}")
            continue

        if not branch:
            skipped.append((label, "no branch"))
            logger.info(f"[SKIP] {label}: no branch info")
            continue

        if not has_commits_ahead(flaggems_dir, branch, base_branch):
            skipped.append((label, "no commits ahead"))
            logger.info(f"[SKIP] {label}: no commits ahead of {base_branch}")
            continue

        cc = issue.get("cc_result") or {}
        title = f"[KernelGen] Fix {operator}: {issue['error_type']}"
        body = build_pr_body(issue)

        if args.dry_run:
            draft_str = "ready" if args.ready else "draft"
            print(f"\n[DRY-RUN] Would create PR ({draft_str}):")
            print(f"  Branch: {branch}")
            print(f"  Title:  {title}")
            print(f"  Target: {target_repo}:{base_branch}")
            print(f"  Files:  {cc.get('files_modified', [])}")
            created.append(label)
            continue

        # Push branch
        push_result = subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=flaggems_dir,
            capture_output=True,
            text=True,
        )
        if push_result.returncode != 0:
            logger.error(f"[PUSH FAILED] {label}: {push_result.stderr}")
            skipped.append((label, "push failed"))
            continue

        # Create PR
        pr_cmd = [
            "gh", "pr", "create",
            "--repo", target_repo,
            "--base", base_branch,
            "--head", f"{fork_owner}:{branch}",
            "--title", title,
            "--body", body,
        ]
        if not args.ready:
            pr_cmd.append("--draft")
        pr_result = subprocess.run(
            pr_cmd,
            cwd=flaggems_dir,
            capture_output=True,
            text=True,
        )
        if pr_result.returncode != 0:
            logger.error(f"[PR FAILED] {label}: {pr_result.stderr}")
            skipped.append((label, "pr creation failed"))
            continue

        pr_url = pr_result.stdout.strip()
        logger.info(f"[CREATED] {label}: {pr_url}")
        created.append(label)

        if args.interval > 0:
            logger.debug(f"Waiting {args.interval}s before next PR...")
            time.sleep(args.interval)

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Summary: {len(created)} PRs created, {len(skipped)} skipped")
    if skipped:
        print("Skipped:")
        for label, reason in skipped:
            print(f"  - {label}: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Create PRs for auto-fixed issues")
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without pushing or creating PRs")
    parser.add_argument("--issues", help="Comma-separated issue IDs to process (e.g. 441-internal,427-internal)")
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO, help=f"Target repo for PRs (default: {DEFAULT_TARGET_REPO})")
    parser.add_argument("--ready", action="store_true", help="Create PRs as ready for review (default: draft)")
    parser.add_argument("--interval", type=int, default=0, help="Seconds to wait between PR creations (default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    run(args)


if __name__ == "__main__":
    main()
