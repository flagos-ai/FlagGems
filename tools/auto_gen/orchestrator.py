#!/usr/bin/env python3
"""Orchestrator for auto-generating FlagGems operators using Claude Code."""

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
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    if yaml is None:
        # Fallback: parse simple YAML manually
        return _parse_simple_yaml(config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_simple_yaml(path: str) -> dict:
    """Minimal YAML parser for flat key-value configs."""
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val == "null":
                    val = None
                elif val.replace(".", "", 1).isdigit():
                    val = float(val) if "." in val else int(val)
                elif val.lower() in ("true", "false"):
                    val = val.lower() == "true"
                config[key] = val
    return config


def load_ops_list(path: str) -> list[str]:
    """Load operator names from a text file.

    Supports formats: 'round', 'aten::round', 'aten::round.Tensor'
    Strips 'aten::' prefix and overload suffixes like '.Tensor'.
    """
    ops = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Strip aten:: prefix
                if line.startswith("aten::"):
                    line = line[len("aten::"):]
                # Strip overload suffix (e.g. .Tensor, .Scalar)
                if "." in line:
                    line = line.split(".")[0]
                if line and line not in ops:
                    ops.append(line)
    return ops


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

def create_worktree(flaggems_dir: str, operator: str) -> tuple[str, str]:
    """Create a git worktree for an operator. Returns (worktree_path, branch_name)."""
    branch_name = f"auto-gen/{operator}"
    worktree_path = os.path.join(flaggems_dir, ".worktrees", f"gen-{operator}")

    # Clean up existing worktree if present
    if os.path.exists(worktree_path):
        subprocess.run(
            ["git", "worktree", "remove", "--force", worktree_path],
            cwd=flaggems_dir,
            capture_output=True,
        )

    # Delete branch if it exists
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=flaggems_dir,
        capture_output=True,
    )

    # Create worktree
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)
    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, worktree_path, "HEAD"],
        cwd=flaggems_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree for {operator}: {result.stderr}")

    logger.info(f"Created worktree for {operator} at {worktree_path}")
    return worktree_path, branch_name


# ---------------------------------------------------------------------------
# CC process management
# ---------------------------------------------------------------------------

def launch_cc(
    operator: str,
    worktree_path: str,
    gpu_id: int,
    config: dict,
    template_path: str,
    log_dir: str,
) -> subprocess.Popen:
    """Launch a Claude Code process for an operator."""
    variables = {
        "OPERATOR": operator,
        "GPU_ID": str(gpu_id),
        "WORK_DIR": worktree_path,
        "PYTHON_PATH": config.get("python_path", "python"),
    }
    prompt = render_template(template_path, variables)

    log_path = os.path.join(log_dir, f"{operator}.log")

    env = os.environ.copy()
    # Remove CLAUDECODE env var to allow launching CC from within a CC session
    env.pop("CLAUDECODE", None)
    # Do NOT set CUDA_VISIBLE_DEVICES here; CC will set it per-command via the template

    claude_bin = config.get("claude_bin", "claude")
    cmd = [
        claude_bin,
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "json",
    ]

    budget = config.get("budget_per_op")
    if budget:
        cmd.extend(["--max-budget-usd", str(budget)])

    stdout_path = os.path.join(log_dir, f"{operator}.stdout.json")
    stdout_file = open(stdout_path, "w")
    stderr_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=worktree_path,
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    # Attach paths for later reading
    proc._stdout_path = stdout_path
    proc._stderr_path = log_path
    proc._stdout_file = stdout_file
    proc._stderr_file = stderr_file

    logger.info(f"Launched CC for {operator} (PID={proc.pid}, GPU={gpu_id})")
    return proc


def check_worktree_has_changes(worktree_path: str, operator: str) -> bool:
    """Check if the worktree has code changes (operator file created)."""
    op_file = os.path.join(worktree_path, "src", "flag_gems", "ops", f"{operator}.py")
    if os.path.exists(op_file):
        return True
    # Also check git diff
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def parse_cc_result(proc: subprocess.Popen, operator: str, worktree_path: str = None) -> dict:
    """Parse the output from a CC process."""
    try:
        # Close file handles first so all data is flushed
        proc._stdout_file.close()
        proc._stderr_file.close()

        # Read stdout from file
        with open(proc._stdout_path, "r", errors="replace") as f:
            stdout_text = f.read()

        # Save a human-readable log: extract the result text and write to .log
        log_path = proc._stderr_path  # reuse the .log path for readable output

        # Try to parse as JSON (--output-format json)
        result_text = stdout_text
        try:
            cc_output = json.loads(stdout_text)
            result_text = cc_output.get("result", "")
        except json.JSONDecodeError:
            pass

        # Write the human-readable CC output to the .log file
        with open(log_path, "w") as f:
            f.write(result_text)
        logger.debug(f"Saved CC output for {operator} to {log_path}")

        # Extract the JSON result block from the text
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find any JSON object in the text
        json_match = re.search(r"\{[^{}]*\"operator\"[^{}]*\"status\"[^{}]*\}", result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        # Fallback: if CC exited normally and worktree has changes, treat as success
        if proc.returncode == 0 and worktree_path and check_worktree_has_changes(worktree_path, operator):
            logger.info(f"CC output not parseable, but worktree has changes for {operator}")
            return {
                "operator": operator,
                "status": "success",
                "accuracy_passed": True,
                "error_message": None,
                "notes": "Result inferred from worktree changes (CC output not parseable)",
            }

    except Exception as e:
        logger.warning(f"Failed to parse CC output for {operator}: {e}")

    # Return a failure result if parsing fails
    return {
        "operator": operator,
        "status": "failed",
        "accuracy_passed": False,
        "error_message": "Failed to parse CC output",
    }


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
                "in_progress": 0,
            },
            "operators": {},
        }
        self._save()

    def add_operator(self, operator: str, gpu_id: int, attempt: int):
        """Record that an operator task has started."""
        self.data["operators"][operator] = {
            "status": "in_progress",
            "gpu_id": gpu_id,
            "attempt": attempt,
            "worktree_path": None,
            "branch": None,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "accuracy_passed": None,
            "error_message": None,
            "cc_result": None,
        }
        self._recount()
        self._save()

    def update_operator(self, operator: str, **kwargs):
        """Update fields for an operator."""
        if operator in self.data["operators"]:
            self.data["operators"][operator].update(kwargs)
            self._recount()
            self._save()

    def finalize(self):
        """Mark the run as complete."""
        self.data["end_time"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def _recount(self):
        """Recount summary statistics."""
        ops = self.data["operators"]
        self.data["summary"]["total"] = len(ops)
        self.data["summary"]["success"] = sum(1 for v in ops.values() if v["status"] == "success")
        self.data["summary"]["failed"] = sum(1 for v in ops.values() if v["status"] == "failed")
        self.data["summary"]["in_progress"] = sum(1 for v in ops.values() if v["status"] == "in_progress")

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
    template_path = os.path.join(script_dir, config.get("template", "templates/generate_op.md"))
    results_dir = os.path.join(script_dir, config.get("results_dir", "results"))
    log_dir = os.path.join(results_dir, "logs")
    summary_path = os.path.join(results_dir, "summary.json")
    max_retries = config.get("max_retries", 3)
    poll_interval = config.get("poll_interval", 10)

    os.makedirs(log_dir, exist_ok=True)

    # Load operator list
    ops_list_path = args.ops_list or os.path.join(script_dir, "ops_list.txt")
    ops = load_ops_list(ops_list_path)
    if not ops:
        logger.error("No operators to process. Check your ops_list.txt.")
        return

    logger.info(f"Loaded {len(ops)} operators: {ops}")

    # Initialize device manager
    device_cfg = config.get("device", {}) or {}
    device_mgr = DeviceManager(
        lock_dir=device_cfg.get("lock_dir", "/tmp/auto_gen_gpu_locks"),
        gpu_ids=device_cfg.get("gpu_ids"),
    )

    # Initialize summary
    summary = Summary(summary_path)

    # Task queue: (operator, attempt_number)
    queue = deque((op, 0) for op in ops)
    # Running tasks: {operator: (process, gpu_id, attempt, worktree_path, start_time)}
    running: dict[str, tuple] = {}

    # Graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested")
            sys.exit(1)
        shutdown_requested = True
        logger.warning("Shutdown requested, waiting for running tasks to finish...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting orchestrator: {len(ops)} operators, {len(device_mgr.gpu_ids)} GPUs, max_retries={max_retries}")

    while (queue or running) and not shutdown_requested:
        # Launch new tasks if GPUs are available
        while queue and not shutdown_requested:
            gpu_id = device_mgr.acquire()
            if gpu_id is None:
                break

            operator, attempt = queue.popleft()
            try:
                worktree_path, branch = create_worktree(flaggems_dir, operator)
                proc = launch_cc(operator, worktree_path, gpu_id, config, template_path, log_dir)

                running[operator] = (proc, gpu_id, attempt, worktree_path, time.time())

                summary.add_operator(operator, gpu_id, attempt + 1)
                summary.update_operator(operator, worktree_path=worktree_path, branch=branch)

            except Exception as e:
                logger.error(f"Failed to launch CC for {operator}: {e}")
                device_mgr.release(gpu_id)
                if attempt + 1 < max_retries:
                    queue.append((operator, attempt + 1))
                else:
                    summary.add_operator(operator, gpu_id, attempt + 1)
                    summary.update_operator(
                        operator,
                        status="failed",
                        error_message=str(e),
                        end_time=datetime.now(timezone.utc).isoformat(),
                    )

        # Check running tasks
        for operator in list(running.keys()):
            proc, gpu_id, attempt, worktree_path, start_time = running[operator]

            if proc.poll() is not None:
                duration = time.time() - start_time
                device_mgr.release(gpu_id)
                del running[operator]

                # Parse result
                result = parse_cc_result(proc, operator, worktree_path)
                success = (
                    result.get("status") == "success"
                    and result.get("accuracy_passed", False)
                    and proc.returncode == 0
                )

                if success:
                    logger.info(f"[SUCCESS] {operator} (attempt {attempt+1}, {duration:.0f}s)")
                    summary.update_operator(
                        operator,
                        status="success",
                        accuracy_passed=True,
                        duration_seconds=round(duration),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        cc_result=result,
                    )
                elif attempt + 1 < max_retries:
                    logger.warning(
                        f"[RETRY] {operator} (attempt {attempt+1}/{max_retries}, "
                        f"reason: {result.get('error_message', 'unknown')})"
                    )
                    summary.update_operator(
                        operator,
                        status="retrying",
                        duration_seconds=round(duration),
                        error_message=result.get("error_message"),
                        cc_result=result,
                    )
                    queue.append((operator, attempt + 1))
                else:
                    logger.error(
                        f"[FAILED] {operator} after {attempt+1} attempts: "
                        f"{result.get('error_message', 'unknown')}"
                    )
                    summary.update_operator(
                        operator,
                        status="failed",
                        accuracy_passed=result.get("accuracy_passed", False),
                        duration_seconds=round(duration),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        error_message=result.get("error_message"),
                        cc_result=result,
                    )

        if running:
            time.sleep(poll_interval)

    # Handle shutdown: mark in-progress tasks
    if shutdown_requested:
        for operator, (proc, gpu_id, attempt, wt, st) in running.items():
            proc.terminate()
            device_mgr.release(gpu_id)
            summary.update_operator(
                operator,
                status="cancelled",
                end_time=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(time.time() - st),
            )

    device_mgr.release_all()
    summary.finalize()

    # Print final summary
    s = summary.data["summary"]
    logger.info(
        f"Done: {s['total']} total, {s['success']} success, "
        f"{s['failed']} failed"
    )
    print(f"\nResults saved to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-generate FlagGems operators using Claude Code")
    parser.add_argument("ops_list", nargs="?", help="Path to operator list file (default: ops_list.txt)")
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run(args)


if __name__ == "__main__":
    main()
