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

"""Collect GitHub Actions rule-check workflow data and push to Feishu Bitable."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime

# --- Feishu API ---


def get_feishu_token(app_id: str, app_secret: str) -> str:
    """Get tenant_access_token from Feishu."""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    if result.get("code") != 0:
        raise RuntimeError(f"Failed to get Feishu token: {result}")
    return result["tenant_access_token"]


def feishu_request(method: str, url: str, token: str, data: dict | None = None) -> dict:
    """Make a Feishu API request."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def get_existing_run_numbers(token: str, app_token: str, table_id: str) -> set[int]:
    """Get all run_numbers already in the Feishu table."""
    run_numbers: set[int] = set()
    page_token = None
    base_url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}"
        f"/tables/{table_id}/records"
    )

    while True:
        url = f'{base_url}?page_size=500&field_names=["run_number"]'
        if page_token:
            url += f"&page_token={page_token}"
        result = feishu_request("GET", url, token)
        if result.get("code") != 0:
            raise RuntimeError(f"Failed to list records: {result}")

        items = result.get("data", {}).get("items") or []
        for item in items:
            rn = item.get("fields", {}).get("run_number")
            if rn is not None:
                run_numbers.add(int(rn))

        if not result.get("data", {}).get("has_more"):
            break
        page_token = result["data"].get("page_token")

    return run_numbers


def batch_create_records(
    token: str, app_token: str, table_id: str, records: list[dict]
) -> int:
    """Batch create records in Feishu Bitable. Returns count of created records."""
    url = (
        f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}"
        f"/tables/{table_id}/records/batch_create"
    )
    created = 0
    # Feishu batch API supports up to 500 records per call
    for i in range(0, len(records), 500):
        batch = records[i : i + 500]
        data = {"records": [{"fields": r} for r in batch]}
        result = feishu_request("POST", url, token, data)
        if result.get("code") != 0:
            raise RuntimeError(f"Batch create failed: {result}")
        created += len(batch)
        if i + 500 < len(records):
            time.sleep(0.5)  # rate limit
    return created


# --- GitHub API ---


def github_get(url: str, gh_token: str | None = None) -> dict:
    """GET request to GitHub API."""
    headers = {"Accept": "application/vnd.github+json"}
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {e.code}: {body}") from e


def get_workflow_id(repo: str, workflow_name: str, gh_token: str | None = None) -> int:
    """Find workflow ID by name."""
    url = f"https://api.github.com/repos/{repo}/actions/workflows?per_page=100"
    data = github_get(url, gh_token)
    for wf in data.get("workflows", []):
        if wf["name"] == workflow_name:
            return wf["id"]
    raise ValueError(f"Workflow {workflow_name!r} not found in {repo}")


def get_workflow_runs(
    repo: str,
    workflow_id: int,
    gh_token: str | None = None,
    max_pages: int = 20,
    min_run_number: int = 0,
) -> list[dict]:
    """Get completed workflow runs, newest first. Stop when run_number <= min_run_number."""
    runs: list[dict] = []
    page = 1
    while page <= max_pages:
        url = (
            f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}"
            f"/runs?status=completed&per_page=100&page={page}"
        )
        data = github_get(url, gh_token)
        page_runs = data.get("workflow_runs", [])
        if not page_runs:
            break

        for run in page_runs:
            if run["run_number"] <= min_run_number:
                return runs
            runs.append(run)

        page += 1
        time.sleep(0.5)  # be nice to GitHub API

    return runs


def get_run_jobs(repo: str, run_id: int, gh_token: str | None = None) -> list[dict]:
    """Get jobs for a specific workflow run."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    data = github_get(url, gh_token)
    return data.get("jobs", [])


def get_job_logs(repo: str, job_id: int, gh_token: str | None = None) -> str:
    """Get raw logs for a specific job. Returns empty string on failure."""
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    headers = {"Accept": "application/vnd.github+json"}
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    # GitHub returns 302 to blob storage. We need to:
    # 1. Make initial request without following redirect
    # 2. Get the Location header
    # 3. Fetch the blob URL directly (no auth needed)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        opener = urllib.request.build_opener(NoRedirect)
        req = urllib.request.Request(url, headers=headers)
        opener.open(req, timeout=30)
        return ""  # unexpected: no redirect
    except urllib.error.HTTPError as e:
        if e.code in (301, 302):
            blob_url = e.headers.get("Location", "")
            if blob_url:
                try:
                    req2 = urllib.request.Request(blob_url)
                    with urllib.request.urlopen(req2, timeout=60) as resp:
                        return resp.read().decode("utf-8", errors="replace")
                except (urllib.error.HTTPError, urllib.error.URLError):
                    return ""
        return ""
    except urllib.error.URLError:
        return ""


def extract_error_lines(logs: str, max_lines: int = 20) -> str:
    """Extract ##[error] lines from job logs, limited to max_lines."""
    error_lines = []
    for line in logs.splitlines():
        if "##[error]" in line:
            # Strip timestamp prefix and ##[error] tag
            content = line.split("##[error]", 1)[1].strip()
            if content and content != "Process completed with exit code 1.":
                error_lines.append(content)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for line in error_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return "\n".join(unique[:max_lines])


# --- Data transformation ---


def parse_iso_to_timestamp_ms(iso_str: str | None) -> int | None:
    """Convert ISO 8601 string to milliseconds timestamp (Feishu date format)."""
    if not iso_str:
        return None
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def compute_duration(jobs: list[dict]) -> int | None:
    """Compute total duration from jobs start/end times."""
    starts = []
    ends = []
    for job in jobs:
        if job.get("started_at"):
            starts.append(
                datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
            )
        if job.get("completed_at"):
            ends.append(
                datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
            )
    if starts and ends:
        return int((max(ends) - min(starts)).total_seconds())
    return None


def run_to_record(run: dict, jobs: list[dict], error_log: str = "") -> dict:
    """Transform a GitHub workflow run + jobs into a Feishu record."""
    failed_jobs = [j["name"] for j in jobs if j.get("conclusion") == "failure"]
    failed_steps = []
    for job in jobs:
        if job.get("conclusion") == "failure":
            for step in job.get("steps", []):
                if step.get("conclusion") == "failure":
                    failed_steps.append(f"{job['name']}/{step['name']}")

    pr_url = ""
    if run.get("pull_requests"):
        pr_number = run["pull_requests"][0].get("number")
        if pr_number:
            repo_url = run.get("repository", {}).get("html_url", "")
            pr_url = f"{repo_url}/pull/{pr_number}"

    record = {
        "run_id": str(run["id"]),
        "run_number": run["run_number"],
        "branch": run.get("head_branch", ""),
        "conclusion": run.get("conclusion", "unknown"),
        "created_at": parse_iso_to_timestamp_ms(run.get("created_at")),
        "duration_s": compute_duration(jobs),
        "failed_jobs": ", ".join(failed_jobs) if failed_jobs else "",
        "failed_steps": ", ".join(failed_steps) if failed_steps else "",
        "error_log": error_log,
        "pr_url": {"link": pr_url, "text": pr_url} if pr_url else None,
        "author": run.get("actor", {}).get("login", ""),
    }
    # Remove None values (Feishu doesn't accept null for some field types)
    return {k: v for k, v in record.items() if v is not None}


# --- Main ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-id", required=True, help="Feishu App ID")
    parser.add_argument("--app-secret", required=True, help="Feishu App Secret")
    parser.add_argument(
        "--table-token",
        default="Asc0bBvQvaffmAsf6o2cr9Ksn71",
        help="Feishu Bitable app token",
    )
    parser.add_argument(
        "--table-id",
        default="tblLu60y04AZZdwB",
        help="Feishu Bitable table ID",
    )
    parser.add_argument(
        "--repo",
        default="flagos-ai/FlagGems",
        help="GitHub repository (owner/name)",
    )
    parser.add_argument(
        "--workflow",
        default="rule-check",
        help="GitHub workflow name",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub token (optional, for higher rate limits)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill all historical data (ignore existing records)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print records without writing to Feishu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Get Feishu token
    print("Authenticating with Feishu...")
    feishu_token = get_feishu_token(args.app_id, args.app_secret)

    # 2. Check existing data for incremental sync
    min_run_number = 0
    if not args.backfill:
        print("Checking existing records in Feishu table...")
        existing = get_existing_run_numbers(
            feishu_token, args.table_token, args.table_id
        )
        if existing:
            min_run_number = max(existing)
            print(
                f"  Found {len(existing)} existing records, max run_number={min_run_number}"
            )
        else:
            print("  No existing records found, will sync all.")

    # 3. Get workflow ID
    print(f"Looking up workflow '{args.workflow}' in {args.repo}...")
    workflow_id = get_workflow_id(args.repo, args.workflow, args.github_token)
    print(f"  Workflow ID: {workflow_id}")

    # 4. Fetch runs from GitHub
    print(f"Fetching runs (run_number > {min_run_number})...")
    runs = get_workflow_runs(
        args.repo, workflow_id, args.github_token, min_run_number=min_run_number
    )
    print(f"  Found {len(runs)} new runs to sync.")

    if not runs:
        print("Nothing to sync.")
        return 0

    # 5. For each run, get jobs and build records
    records: list[dict] = []
    for i, run in enumerate(runs):
        run_id = run["id"]
        print(
            f"  [{i + 1}/{len(runs)}] Run #{run['run_number']} ({run['conclusion']})..."
        )
        jobs = get_run_jobs(args.repo, run_id, args.github_token)

        # Collect error logs for failed jobs
        error_log = ""
        if run.get("conclusion") == "failure":
            error_parts = []
            for job in jobs:
                if job.get("conclusion") == "failure":
                    logs = get_job_logs(args.repo, job["id"], args.github_token)
                    errors = extract_error_lines(logs)
                    if errors:
                        error_parts.append(f"[{job['name']}]\n{errors}")
            error_log = "\n\n".join(error_parts)

        record = run_to_record(run, jobs, error_log)
        records.append(record)
        time.sleep(0.3)  # rate limit

    # 6. Push to Feishu
    if args.dry_run:
        print("\nDry run - records to write:")
        for r in records[:5]:
            print(json.dumps(r, indent=2, ensure_ascii=False))
        if len(records) > 5:
            print(f"  ... and {len(records) - 5} more")
        return 0

    print(f"\nWriting {len(records)} records to Feishu...")
    created = batch_create_records(
        feishu_token, args.table_token, args.table_id, records
    )
    print(f"Done. {created} records created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
