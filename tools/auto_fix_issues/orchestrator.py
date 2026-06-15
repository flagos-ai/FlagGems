#!/usr/bin/env python3
"""Orchestrator for auto-fixing FlagGems issues using Claude Code."""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from device_manager import DeviceManager

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def load_dotenv(env_path: str = None):
    """Load .env file into os.environ (simple key=value parser)."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if val and val[0] in ('"', "'") and val[-1] == val[0]:
                    val = val[1:-1]
                if key:
                    os.environ[key] = val
    logger.debug(f"Loaded .env from {env_path}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    if yaml is None:
        print("Error: 'pyyaml' is required. Install with: pip install pyyaml")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_issues(path: str) -> list[dict]:
    """Load issues from a YAML file.

    Expected format:
        issues:
          - id: 418
            operator: sparse_mla_fwd_interface
            type: accuracy_fail
            severity: major
            test_cmd: "pytest -m 'sparse_mla_fwd_interface' tests/ --ref cpu -vs"
            benchmark_cmd: "pytest -m 'sparse_mla_fwd_interface' benchmark/ --level core --record log"
    """
    if yaml is None:
        print("Error: 'pyyaml' is required. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        data = yaml.safe_load(f)

    issues = data.get("issues", [])
    if not issues:
        return []

    required_fields = {"id", "type", "test_cmd"}
    validated = []
    for issue in issues:
        missing = required_fields - set(issue.keys())
        # scope: repo issues use title+operators instead of operator
        is_repo_scope = issue.get("scope") == "repo"
        if not is_repo_scope and "operator" not in issue:
            missing.add("operator")
        if missing:
            logger.warning(f"Issue entry missing fields {missing}, skipping: {issue}")
            continue
        issue.setdefault("severity", "unknown")
        issue.setdefault("benchmark_cmd", "")
        issue.setdefault("scope", "operator")
        # For repo-scope issues, derive operator name from title for branch naming
        if is_repo_scope and "operator" not in issue:
            import re
            slug = issue.get("title", f"repo-issue-{issue['id']}")
            slug = slug.lower().replace(" ", "-")
            slug = re.sub(r"[^a-z0-9\-]", "", slug)
            slug = re.sub(r"-+", "-", slug).strip("-")[:50]
            issue["operator"] = slug
        validated.append(issue)

    return validated


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render_template(template_path: str, variables: dict) -> str:
    """Render a template file with {{VAR}} substitution."""
    with open(template_path) as f:
        content = f.read()
    for key, value in variables.items():
        content = content.replace(f"{{{{{key}}}}}", str(value))
    return content


# ---------------------------------------------------------------------------
# Worktree management
# ---------------------------------------------------------------------------


def create_worktree(flaggems_dir: str, issue: dict, base_branch: str = "master") -> tuple[str, str]:
    """Create a git worktree for an issue fix. Returns (worktree_path, branch_name)."""
    issue_id = issue["id"]
    operator = issue["operator"]
    branch_name = f"fix/issue-{issue_id}-{operator}"
    worktree_path = os.path.join(flaggems_dir, ".worktrees", f"fix-{issue_id}-{operator}")

    # Clean up any existing worktree
    subprocess.run(
        ["git", "worktree", "remove", "--force", worktree_path],
        cwd=flaggems_dir,
        capture_output=True,
    )
    if os.path.exists(worktree_path):
        import shutil
        shutil.rmtree(worktree_path, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=flaggems_dir, capture_output=True)

    # Delete branch if it exists
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=flaggems_dir,
        capture_output=True,
    )

    # Create worktree based on base_branch
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)
    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, worktree_path, base_branch],
        cwd=flaggems_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create worktree for issue-{issue_id}/{operator}: {result.stderr}"
        )

    logger.info(f"Created worktree for issue-{issue_id}/{operator} at {worktree_path}")
    return worktree_path, branch_name


# ---------------------------------------------------------------------------
# CC process management
# ---------------------------------------------------------------------------


def launch_cc(
    issue: dict,
    worktree_path: str,
    gpu_id: int,
    config: dict,
    template_path: str,
    log_dir: str,
    attempt: int = 0,
) -> subprocess.Popen:
    """Launch a Claude Code process for an issue fix."""
    issue_id = issue["id"]
    operator = issue["operator"]
    task_name = f"issue-{issue_id}-{operator}"

    variables = {
        "ISSUE_ID": str(issue_id),
        "OPERATOR": operator,
        "ERROR_TYPE": issue["type"],
        "SEVERITY": issue.get("severity", "unknown"),
        "GPU_ID": str(gpu_id),
        "WORK_DIR": worktree_path,
        "PYTHON_PATH": config.get("python_path", "python"),
        "TEST_CMD": issue["test_cmd"],
        "BENCHMARK_CMD": issue.get("benchmark_cmd", ""),
        "SCOPE": issue.get("scope", "operator"),
        "TITLE": issue.get("title", ""),
        "GITHUB_ISSUE": str(issue.get("github_issue", "")),
        "GEMS_VENDOR": config.get("gems_vendor", "nvidia"),
        "OPERATORS": ", ".join(issue.get("operators", [])),
        "FILES": ", ".join(issue.get("files", [])),
    }
    prompt = render_template(template_path, variables)

    log_path = os.path.join(log_dir, f"{task_name}.attempt-{attempt + 1}.log")

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["IS_SANDBOX"] = "1"

    gems_vendor = config.get("gems_vendor")
    if gems_vendor:
        env["GEMS_VENDOR"] = gems_vendor

    _token = env.get("ANTHROPIC_AUTH_TOKEN", "")
    _base = env.get("ANTHROPIC_BASE_URL", "")
    logger.debug(
        f"CC env for {task_name}: "
        f"AUTH_TOKEN={'set(' + _token[:8] + '...)' if _token else 'MISSING'}, "
        f"BASE_URL={_base or 'MISSING'}"
    )

    claude_bin = config.get("claude_bin", "claude")
    cmd = [
        claude_bin,
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]

    budget = config.get("budget_per_op")
    if budget:
        cmd.extend(["--max-budget-usd", str(budget)])

    stdout_path = os.path.join(log_dir, f"{task_name}.attempt-{attempt + 1}.jsonl")
    stdout_file = open(stdout_path, "w")
    stderr_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=worktree_path,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
    except Exception:
        stdout_file.close()
        stderr_file.close()
        raise

    proc._stdout_path = stdout_path
    proc._stderr_path = log_path
    proc._stdout_file = stdout_file
    proc._stderr_file = stderr_file

    logger.info(f"Launched CC for {task_name} (PID={proc.pid}, GPU={gpu_id})")
    return proc


def _kill_cc_process(proc: subprocess.Popen):
    """Kill a CC process and its entire process group, then close file handles."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Escalate to SIGKILL
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Process {proc.pid} did not exit after SIGKILL, abandoning")
    if not proc._stdout_file.closed:
        proc._stdout_file.close()
    if not proc._stderr_file.closed:
        proc._stderr_file.close()


def _extract_json_object(text: str, start: int) -> str | None:
    """Extract a complete JSON object from text using brace counting."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_cc_result(proc: subprocess.Popen, issue: dict, worktree_path: str = None) -> dict:
    """Parse stream-json output from a CC process.

    Looks for the JSON result block with issue_id, test_passed, etc.
    """
    issue_id = issue["id"]
    operator = issue["operator"]
    task_name = f"issue-{issue_id}-{operator}"

    try:
        if not proc._stdout_file.closed:
            proc._stdout_file.close()
        if not proc._stderr_file.closed:
            proc._stderr_file.close()

        result_text = ""
        with open(proc._stdout_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result":
                    result_text = event.get("result", "")
                    break

        if result_text:
            # Try fenced JSON block first (brace-counting to handle nested objects)
            fence_start = result_text.find("```json")
            if fence_start != -1:
                brace_start = result_text.find("{", fence_start)
                if brace_start != -1:
                    extracted = _extract_json_object(result_text, brace_start)
                    if extracted:
                        try:
                            return json.loads(extracted)
                        except json.JSONDecodeError:
                            pass

            # Try to find any top-level JSON object with brace counting
            idx = 0
            while idx < len(result_text):
                idx = result_text.find("{", idx)
                if idx == -1:
                    break
                extracted = _extract_json_object(result_text, idx)
                if extracted:
                    try:
                        parsed = json.loads(extracted)
                        if "issue_id" in parsed or "status" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                idx += 1

        # Fallback: worktree has changes but CC output wasn't parseable — mark as needs_review
        if proc.returncode == 0 and worktree_path:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                logger.warning(
                    f"CC output not parseable for {task_name}, "
                    f"worktree has changes but cannot confirm test results"
                )
                return {
                    "issue_id": str(issue_id),
                    "operator": operator,
                    "status": "needs_review",
                    "test_passed": None,
                    "benchmark_passed": None,
                    "format_check_passed": None,
                    "error_message": "CC output not parseable; worktree has changes but results unverified",
                    "notes": "Manual review required",
                }

    except Exception as e:
        logger.warning(f"Failed to parse CC output for {task_name}: {e}")

    return {
        "issue_id": str(issue_id),
        "operator": operator,
        "status": "failed",
        "test_passed": False,
        "benchmark_passed": False,
        "format_check_passed": False,
        "error_message": "Failed to parse CC output",
    }


def post_commit_format_check(worktree_path: str, python_path: str, log_path: str = None) -> dict | None:
    """Auto-fix code style (black + isort) and verify with flake8 after agent commits.

    This is a deterministic, safe post-processing step that does not change code logic.
    Returns {"passed": bool, "auto_fixed": bool, "flake8_errors": str | None}, or None on error.
    """
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        logger.info(f"[FORMAT] {msg}")

    try:
        # Get .py files changed in the HEAD commit only (not upstream diffs)
        diff_result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only",
             "--diff-filter=ACMR", "-r", "HEAD", "--", "*.py"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        py_files = [f for f in diff_result.stdout.strip().split("\n") if f.endswith(".py")]
        if not py_files:
            log("No .py files changed — skipping")
            _write_format_log(log_path, log_lines)
            return {"passed": True, "auto_fixed": False, "flake8_errors": None}

        log(f"Files to check: {py_files}")

        # Auto-fix with black + isort (deterministic, safe)
        black_result = subprocess.run(
            [python_path, "-m", "black"] + py_files,
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        log(f"black: exit={black_result.returncode}")
        if black_result.stderr.strip():
            log(f"  {black_result.stderr.strip()}")

        isort_result = subprocess.run(
            [python_path, "-m", "isort", "--profile", "black"] + py_files,
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        log(f"isort: exit={isort_result.returncode}")

        # Check if black/isort made any changes
        status_result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        auto_fixed = bool(status_result.stdout.strip())

        if auto_fixed:
            log(f"Auto-fixed files: {status_result.stdout.strip()}")
            subprocess.run(["git", "add"] + py_files, cwd=worktree_path, capture_output=True)
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )
            log("Amended commit with style fixes")
        else:
            log("No style fixes needed")

        # Final check with flake8
        flake8_result = subprocess.run(
            [python_path, "-m", "flake8",
             "--ignore=F405,E731,W503,E203", "--max-line-length=120"] + py_files,
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        flake8_passed = flake8_result.returncode == 0
        flake8_errors = flake8_result.stdout.strip() if not flake8_passed else None
        log(f"flake8: {'PASS' if flake8_passed else 'FAIL'}")
        if flake8_errors:
            log(f"  {flake8_errors}")

        # Check pytest marks consistency for test/benchmark files
        mark_errors = None
        test_bench_files = [f for f in py_files if f.startswith(("tests/", "benchmark/"))]
        if test_bench_files:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mark_script = os.path.join(script_dir, "check_pytest_marks.py")
            mark_result = subprocess.run(
                [python_path, mark_script] + test_bench_files + ["--all-files"] + py_files,
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )
            mark_passed = mark_result.returncode == 0
            log(f"pytest-mark-check: {'PASS' if mark_passed else 'FAIL'}")
            if not mark_passed:
                mark_errors = mark_result.stdout.strip() or mark_result.stderr.strip()
                log(f"  {mark_errors}")
        else:
            log("pytest-mark-check: SKIP (no test/benchmark files)")

        all_passed = flake8_passed and (mark_errors is None)
        log(f"Result: {'PASS' if all_passed else 'FAIL'} (auto_fixed={auto_fixed})")

        # Combine all errors
        all_errors = "\n".join(filter(None, [flake8_errors, mark_errors]))

        _write_format_log(log_path, log_lines)
        return {
            "passed": all_passed,
            "auto_fixed": auto_fixed,
            "flake8_errors": all_errors or None,
        }

    except Exception as e:
        log(f"Format check failed: {e}")
        _write_format_log(log_path, log_lines)
        return None


def _write_format_log(log_path: str | None, lines: list[str]):
    """Write format check log to file."""
    if not log_path:
        return
    try:
        with open(log_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def generate_timeline(jsonl_path: str, task_name: str) -> str | None:
    """Generate a human-readable timeline from a CC stream-json log."""
    timeline_path = jsonl_path.replace(".jsonl", ".timeline.txt")
    try:
        events = []
        with open(jsonl_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        out: list[str] = []
        step = 0

        def _format_tool_use(name: str, inp: dict) -> str:
            if name == "Bash":
                return inp.get("command", "")
            elif name in ("Read", "Write"):
                return inp.get("file_path", "")
            elif name == "Edit":
                s = inp.get("file_path", "")
                old = inp.get("old_string", "")
                new = inp.get("new_string", "")
                return f"{s}\n--- old ---\n{old}\n+++ new +++\n{new}"
            elif name in ("Grep", "Glob"):
                return f"pattern={inp.get('pattern', '')}  path={inp.get('path', '')}"
            else:
                return json.dumps(inp, ensure_ascii=False)

        for event in events:
            etype = event.get("type", "")

            if etype == "system" and event.get("subtype") == "init":
                out.append(f"=== {task_name} ===")
                out.append(f"Session: {event.get('session_id', '?')}")
                out.append(f"Model: {event.get('model', '?')}")
                out.append("")
                continue

            if etype == "result":
                step += 1
                out.append(f"[{step}] Result:")
                out.append(event.get("result", ""))
                out.append("")
                continue

            if etype == "user":
                contents = event.get("message", {}).get("content", [])
                if isinstance(contents, list):
                    for c in contents:
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            content_val = c.get("content", "")
                            if content_val:
                                out.append(f"    -> Output:")
                                out.append(str(content_val))
                                out.append("")
                            break
                continue

            if etype != "assistant":
                continue

            contents = event.get("message", {}).get("content", [])
            if not isinstance(contents, list):
                continue

            for content in contents:
                if not isinstance(content, dict):
                    continue
                ctype = content.get("type", "")

                if ctype == "thinking":
                    step += 1
                    out.append(f"[{step}] Thinking:")
                    out.append(content.get("thinking", ""))
                    out.append("")

                elif ctype == "text":
                    text = content.get("text", "")
                    if text.strip():
                        step += 1
                        out.append(f"[{step}] Text:")
                        out.append(text)
                        out.append("")

                elif ctype == "tool_use":
                    step += 1
                    name = content.get("name", "?")
                    inp = content.get("input", {})
                    out.append(f"[{step}] Tool({name}):")
                    out.append(_format_tool_use(name, inp))
                    out.append("")

        with open(timeline_path, "w") as f:
            f.write("\n".join(out))

        logger.info(f"Generated timeline for {task_name}: {timeline_path}")
        return timeline_path

    except Exception as e:
        logger.warning(f"Failed to generate timeline for {task_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Summary management
# ---------------------------------------------------------------------------


class Summary:
    """Manages the summary.json file with real-time updates."""

    def __init__(self, path: str):
        self.path = path
        self.data = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "summary": {
                "total": 0,
                "success": 0,
                "failed": 0,
                "needs_review": 0,
                "in_progress": 0,
            },
            "issues": {},
        }
        self._save()

    def add_issue(self, issue: dict, gpu_id: int, attempt: int):
        """Record that an issue fix task has started."""
        key = f"issue-{issue['id']}"
        self.data["issues"][key] = {
            "issue_id": issue["id"],
            "operator": issue["operator"],
            "error_type": issue["type"],
            "severity": issue.get("severity", "unknown"),
            "scope": issue.get("scope", "operator"),
            "github_issue": issue.get("github_issue"),
            "status": "in_progress",
            "gpu_id": gpu_id,
            "attempt": attempt,
            "worktree_path": None,
            "branch": None,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "test_passed": None,
            "benchmark_passed": None,
            "format_check_passed": None,
            "error_message": None,
            "cc_result": None,
        }
        self._recount()
        self._save()

    def update_issue(self, issue_id, **kwargs):
        """Update fields for an issue."""
        key = f"issue-{issue_id}"
        if key in self.data["issues"]:
            self.data["issues"][key].update(kwargs)
            self._recount()
            self._save()

    def finalize(self):
        """Mark the run as complete."""
        self.data["end_time"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def _recount(self):
        """Recount summary statistics."""
        issues = self.data["issues"]
        self.data["summary"]["total"] = len(issues)
        self.data["summary"]["success"] = sum(
            1 for v in issues.values() if v["status"] == "success"
        )
        self.data["summary"]["failed"] = sum(
            1 for v in issues.values() if v["status"] in ("failed", "cancelled")
        )
        self.data["summary"]["needs_review"] = sum(
            1 for v in issues.values() if v["status"] == "needs_review"
        )
        self.data["summary"]["in_progress"] = sum(
            1 for v in issues.values() if v["status"] in ("in_progress", "retrying")
        )

    def _save(self):
        """Write summary to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run(args):
    """Main orchestration loop."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)

    flaggems_dir = config.get("flaggems_dir", os.path.dirname(os.path.dirname(script_dir)))
    template_path = os.path.join(script_dir, config.get("template", "templates/fix_issue.md"))
    results_dir = os.path.join(script_dir, config.get("results_dir", "results"))
    log_dir = os.path.join(results_dir, "logs")
    summary_path = os.path.join(results_dir, "summary.json")
    max_retries = config.get("max_retries", 2)
    timeout_per_op = config.get("timeout_per_op", 3600) or 0
    poll_interval = config.get("poll_interval", 10)
    base_branch = config.get("base_branch", "master")

    os.makedirs(log_dir, exist_ok=True)

    # Load issues
    issues_path = args.issues_file or os.path.join(script_dir, "issues_to_fix.yaml")
    issues = load_issues(issues_path)
    if not issues:
        logger.error("No issues to process. Check your issues_to_fix.yaml.")
        return

    issue_tags = ["#%s/%s" % (i["id"], i["operator"]) for i in issues]
    logger.info(f"Loaded {len(issues)} issues: {issue_tags}")

    # Initialize device manager
    device_cfg = config.get("device", {}) or {}
    device_mgr = DeviceManager(
        lock_dir=device_cfg.get("lock_dir", "/tmp/auto_fix_gpu_locks"),
        gpu_ids=device_cfg.get("gpu_ids"),
    )

    # Initialize summary
    summary = Summary(summary_path)

    # Task queue: (issue_dict, attempt_number)
    queue = deque((issue, 0) for issue in issues)
    # Running tasks: {issue_key: (process, gpu_id, attempt, issue, worktree_path, start_time)}
    running: dict[str, tuple] = {}

    # Graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested, exiting immediately")
            os.system("stty sane 2>/dev/null")
            os._exit(1)
        shutdown_requested = True
        logger.warning(f"Shutdown requested (signal={sig}), killing {len(running)} running tasks...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(
        f"Starting orchestrator: {len(issues)} issues, "
        f"{len(device_mgr.gpu_ids)} GPUs, max_retries={max_retries}"
    )

    while (queue or running) and not shutdown_requested:
        # Launch new tasks if GPUs are available
        while queue and not shutdown_requested:
            gpu_id = device_mgr.acquire()
            if gpu_id is None:
                break

            issue, attempt = queue.popleft()
            issue_id = issue["id"]
            issue_key = f"issue-{issue_id}"

            try:
                worktree_path, branch = create_worktree(flaggems_dir, issue, base_branch)
                proc = launch_cc(issue, worktree_path, gpu_id, config, template_path, log_dir, attempt)

                running[issue_key] = (proc, gpu_id, attempt, issue, worktree_path, time.time())

                summary.add_issue(issue, gpu_id, attempt + 1)
                summary.update_issue(issue_id, worktree_path=worktree_path, branch=branch)

            except Exception as e:
                logger.error(f"Failed to launch CC for {issue_key}: {e}")
                device_mgr.release(gpu_id)
                if attempt + 1 < max_retries:
                    queue.append((issue, attempt + 1))
                else:
                    summary.add_issue(issue, gpu_id, attempt + 1)
                    summary.update_issue(
                        issue_id,
                        status="failed",
                        error_message=str(e),
                        end_time=datetime.now(timezone.utc).isoformat(),
                    )

        # Check running tasks
        for issue_key in list(running.keys()):
            proc, gpu_id, attempt, issue, worktree_path, start_time = running[issue_key]
            issue_id = issue["id"]

            # Check for timeout
            if timeout_per_op and proc.poll() is None and time.time() - start_time > timeout_per_op:
                logger.error(
                    f"[TIMEOUT] {issue_key} exceeded {timeout_per_op}s, killing process"
                )
                _kill_cc_process(proc)
                duration = time.time() - start_time
                device_mgr.release(gpu_id)
                del running[issue_key]
                summary.update_issue(
                    issue_id,
                    status="failed",
                    test_passed=False,
                    benchmark_passed=False,
                    format_check_passed=False,
                    duration_seconds=round(duration),
                    end_time=datetime.now(timezone.utc).isoformat(),
                    error_message=f"Timed out after {timeout_per_op}s",
                )
                continue

            if proc.poll() is not None:
                duration = time.time() - start_time
                device_mgr.release(gpu_id)
                del running[issue_key]

                # Parse result and generate timeline
                result = parse_cc_result(proc, issue, worktree_path)
                task_name = f"issue-{issue_id}-{issue['operator']}"
                generate_timeline(proc._stdout_path, task_name)

                # Post-commit format check: auto-fix black/isort, verify flake8
                if proc.returncode == 0:
                    format_log = proc._stdout_path.replace(".jsonl", ".format-check.log")
                    format_result = post_commit_format_check(
                        worktree_path,
                        config.get("python_path", "python"),
                        log_path=format_log,
                    )
                    if format_result is not None:
                        result["format_check_passed"] = format_result["passed"]
                        if format_result.get("auto_fixed"):
                            result.setdefault("notes", "")
                            if result["notes"]:
                                result["notes"] += "; "
                            result["notes"] += "auto-fixed code style (black/isort)"

                success = (
                    result.get("status") == "success"
                    and result.get("test_passed", False)
                    and proc.returncode == 0
                )

                if success:
                    logger.info(f"[SUCCESS] {issue_key} (attempt {attempt+1}, {duration:.0f}s)")
                    summary.update_issue(
                        issue_id,
                        status="success",
                        test_passed=result.get("test_passed", True),
                        benchmark_passed=result.get("benchmark_passed"),
                        format_check_passed=result.get("format_check_passed"),
                        duration_seconds=round(duration),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        cc_result=result,
                    )
                elif attempt + 1 < max_retries:
                    logger.warning(
                        f"[RETRY] {issue_key} (attempt {attempt+1}/{max_retries}, "
                        f"reason: {result.get('error_message', 'unknown')})"
                    )
                    summary.update_issue(
                        issue_id,
                        status="retrying",
                        duration_seconds=round(duration),
                        error_message=result.get("error_message"),
                        cc_result=result,
                    )
                    queue.append((issue, attempt + 1))
                else:
                    logger.error(
                        f"[FAILED] {issue_key} after {attempt+1} attempts: "
                        f"{result.get('error_message', 'unknown')}"
                    )
                    summary.update_issue(
                        issue_id,
                        status="failed",
                        test_passed=result.get("test_passed", False),
                        benchmark_passed=result.get("benchmark_passed", False),
                        format_check_passed=result.get("format_check_passed", False),
                        duration_seconds=round(duration),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        error_message=result.get("error_message"),
                        cc_result=result,
                    )
                    # Clean up worktree after exhausting all retries
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", worktree_path],
                        cwd=flaggems_dir,
                        capture_output=True,
                    )

        if running:
            for _ in range(poll_interval):
                if shutdown_requested:
                    break
                time.sleep(1)

    # Handle shutdown: kill running tasks
    if shutdown_requested:
        for issue_key, (proc, gpu_id, attempt, issue, wt, st) in running.items():
            _kill_cc_process(proc)
            device_mgr.release(gpu_id)
            summary.update_issue(
                issue["id"],
                status="cancelled",
                end_time=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(time.time() - st),
            )
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                cwd=flaggems_dir,
                capture_output=True,
            )

    device_mgr.release_all()
    subprocess.run(["git", "worktree", "prune"], cwd=flaggems_dir, capture_output=True)
    summary.finalize()

    try:
        os.system("stty sane 2>/dev/null")
    except Exception:
        pass

    # Print final summary
    s = summary.data["summary"]
    logger.info(
        f"Done: {s['total']} total, {s['success']} success, {s['failed']} failed"
    )
    print(f"\nResults saved to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fix FlagGems issues using Claude Code"
    )
    parser.add_argument(
        "issues_file", nargs="?",
        help="Path to issues YAML file (default: issues_to_fix.yaml)",
    )
    parser.add_argument("-c", "--config", help="Path to config.yaml")
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
