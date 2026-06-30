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
# Vendor device environment variable mapping
# (copied from tools/run_tests.py — maps vendor name to device visibility env vars)
# ---------------------------------------------------------------------------

VENDOR_ENV_MAP = {
    "ascend": ["ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"],
    "hygon": ["HIP_VISIBLE_DEVICES"],
    "metax": ["MACA_VISIBLE_DEVICES"],
    "mthreads": ["MUSA_VISIBLE_DEVICES"],
    "tsingmicro": ["TXDA_VISIBLE_DEVICES"],
    "iluvatar": ["ILUVATAR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"],
    "thead": ["CUDA_VISIBLE_DEVICES"],
    "cambricon": ["MLU_VISIBLE_DEVICES"],
    "kunlunxin": ["CUDA_VISIBLE_DEVICES"],
    "sunrise": ["TANG_VISIBLE_DEVICES"],
}

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
        print("Error: 'pyyaml' is required but not installed. Please install it with: pip install pyyaml")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


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


def load_vendor_ops(path: str) -> list[dict]:
    """Load vendor ops from a YAML file.

    Expected format:
        ops:
          - operator: softmax
          - operator: layernorm
            type: accuracy_fail
            error_desc: "fp16 precision exceeds rtol=1e-3"
            test_cmd: "pytest -m layernorm tests/ -vs"
    """
    if yaml is None:
        print("Error: 'pyyaml' is required. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        data = yaml.safe_load(f)

    ops = data.get("ops", [])
    validated = []
    for op in ops:
        if "operator" not in op:
            logger.warning(f"Op entry missing 'operator', skipping: {op}")
            continue
        op.setdefault("type", "")
        op.setdefault("error_desc", "")
        op.setdefault("test_cmd", "")
        validated.append(op)
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

def create_worktree(flaggems_dir: str, operator: str, vendor: str = None, base_branch: str = "master") -> tuple[str, str]:
    """Create a git worktree for an operator. Returns (worktree_path, branch_name)."""
    if vendor:
        branch_name = f"auto-gen/{vendor}/{operator}"
        worktree_path = os.path.join(flaggems_dir, ".worktrees", f"gen-{vendor}-{operator}")
    else:
        branch_name = f"auto-gen/{operator}"
        worktree_path = os.path.join(flaggems_dir, ".worktrees", f"gen-{operator}")

    # Always clean up: remove worktree, delete leftover directory, prune git records
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
    vendor_op: dict = None,
) -> subprocess.Popen:
    """Launch a Claude Code process for an operator."""
    vendor = config.get("vendor")

    # Build device prefix
    if vendor:
        env_vars = VENDOR_ENV_MAP.get(vendor, ["CUDA_VISIBLE_DEVICES"])
        device_prefix = " ".join(f"{v}={gpu_id}" for v in env_vars)
    else:
        device_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id}"

    variables = {
        "OPERATOR": operator,
        "GPU_ID": str(gpu_id),
        "WORK_DIR": worktree_path,
        "PYTHON_PATH": config.get("python_path", "python"),
        "DEVICE_PREFIX": device_prefix,
        "VENDOR": vendor or "",
        "VENDOR_OPS_DIR": f"src/flag_gems/runtime/backend/_{vendor}/ops" if vendor else "",
        "ERROR_TYPE": (vendor_op or {}).get("type", ""),
        "ERROR_DESC": (vendor_op or {}).get("error_desc", ""),
        "TEST_CMD": (vendor_op or {}).get("test_cmd", ""),
    }
    prompt = render_template(template_path, variables)

    log_path = os.path.join(log_dir, f"{operator}.log")

    env = os.environ.copy()
    # Remove CLAUDECODE env var to allow launching CC from within a CC session
    env.pop("CLAUDECODE", None)
    # Allow --dangerously-skip-permissions under root
    env["IS_SANDBOX"] = "1"
    # Set vendor for FlagGems backend detection
    if vendor:
        env["GEMS_VENDOR"] = vendor
    # Do NOT set CUDA_VISIBLE_DEVICES here; CC will set it per-command via the template

    # Debug: verify API credentials are present
    _token = env.get("ANTHROPIC_AUTH_TOKEN", "")
    _base = env.get("ANTHROPIC_BASE_URL", "")
    logger.debug(f"CC env for {operator}: AUTH_TOKEN={'set(' + _token[:8] + '...)' if _token else 'MISSING'}, BASE_URL={_base or 'MISSING'}")

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

    stdout_path = os.path.join(log_dir, f"{operator}.jsonl")
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
    # Attach paths for later reading
    proc._stdout_path = stdout_path
    proc._stderr_path = log_path
    proc._stdout_file = stdout_file
    proc._stderr_file = stderr_file

    logger.info(f"Launched CC for {operator} (PID={proc.pid}, GPU={gpu_id})")
    return proc


def _kill_cc_process(proc: subprocess.Popen):
    """Kill a CC process and its entire process group, then close file handles."""
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
    proc._stdout_file.close()
    proc._stderr_file.close()


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
    """Parse stream-json output from a CC process.

    The .jsonl file contains one JSON object per line. We look for the last
    line with "type": "result" to get the final result, then extract the
    operator JSON from the result text.
    """
    try:
        # Close file handles first so all data is flushed
        proc._stdout_file.close()
        proc._stderr_file.close()

        # Parse stream-json: read lines and find the result event
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

        # Extract the operator JSON result block from the result text
        if result_text:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to find any JSON object with operator/status fields
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


def generate_timeline(jsonl_path: str, operator: str) -> str | None:
    """Generate a human-readable timeline from a CC stream-json log.

    Writes a .timeline.txt file next to the .jsonl and returns its path.
    """
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
                out.append(f"=== {operator} ===")
                out.append(f"Session: {event.get('session_id', '?')}")
                out.append(f"Model: {event.get('model', '?')}")
                out.append("")
                continue

            if etype == "result":
                step += 1
                out.append(f"[{step}] ✅ Result:")
                out.append(event.get("result", ""))
                out.append("")
                continue

            if etype == "user":
                # Extract tool result output
                tool_result = event.get("tool_use_result")
                if isinstance(tool_result, dict):
                    output = tool_result.get("stdout", "") or tool_result.get("stderr", "")
                    if output:
                        out.append(f"    ↳ Output:")
                        out.append(str(output))
                        out.append("")
                        continue
                # Fallback: check message.content for tool_result entries
                contents = event.get("message", {}).get("content", [])
                if isinstance(contents, list):
                    for c in contents:
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            content_val = c.get("content", "")
                            if content_val:
                                out.append(f"    ↳ Output:")
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
                    out.append(f"[{step}] 🤔 Thinking:")
                    out.append(content.get("thinking", ""))
                    out.append("")

                elif ctype == "text":
                    text = content.get("text", "")
                    if text.strip():
                        step += 1
                        out.append(f"[{step}] 💬 Text:")
                        out.append(text)
                        out.append("")

                elif ctype == "tool_use":
                    step += 1
                    name = content.get("name", "?")
                    inp = content.get("input", {})
                    out.append(f"[{step}] 🔧 {name}:")
                    out.append(_format_tool_use(name, inp))
                    out.append("")

        with open(timeline_path, "w") as f:
            f.write("\n".join(out))

        logger.info(f"Generated timeline for {operator}: {timeline_path}")
        return timeline_path

    except Exception as e:
        logger.warning(f"Failed to generate timeline for {operator}: {e}")
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
        self.data["summary"]["failed"] = sum(1 for v in ops.values() if v["status"] in ("failed", "cancelled"))
        self.data["summary"]["in_progress"] = sum(1 for v in ops.values() if v["status"] in ("in_progress", "retrying"))

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
    results_dir = os.path.join(script_dir, config.get("results_dir", "results"))
    log_dir = os.path.join(results_dir, "logs")
    summary_path = os.path.join(results_dir, "summary.json")
    max_retries = config.get("max_retries", 3)
    timeout_per_op = config.get("timeout_per_op", 1800) or 0
    poll_interval = config.get("poll_interval", 10)
    vendor = config.get("vendor")
    base_branch = config.get("base_branch", "master")

    # Select template based on mode
    if vendor:
        template_path = os.path.join(script_dir, config.get("vendor_template", "templates/generate_vendor_op.md"))
    else:
        template_path = os.path.join(script_dir, config.get("template", "templates/generate_op.md"))

    os.makedirs(log_dir, exist_ok=True)

    # Load operator list
    ops_list_path = args.ops_list or os.path.join(
        script_dir, "vendor_ops.yaml" if vendor else "ops_list.txt"
    )
    if vendor:
        vendor_ops = load_vendor_ops(ops_list_path)
        if not vendor_ops:
            logger.error("No operators to process. Check your vendor_ops.yaml.")
            return
        ops = [op["operator"] for op in vendor_ops]
        vendor_ops_map = {op["operator"]: op for op in vendor_ops}
        logger.info(f"Vendor mode ({vendor}): loaded {len(ops)} operators: {ops}")
    else:
        ops = load_ops_list(ops_list_path)
        vendor_ops_map = {}
        if not ops:
            logger.error("No operators to process. Check your ops_list.txt.")
            return
        logger.info(f"Loaded {len(ops)} operators: {ops}")

    # Fetch upstream to ensure base_branch is up-to-date
    auto_fetch = config.get("auto_fetch_upstream", True)
    if not args.skip_fetch and auto_fetch:
        logger.info("Fetching upstream to ensure base_branch is current...")
        result = subprocess.run(
            ["git", "fetch", "upstream"],
            cwd=flaggems_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if "does not resolve" in result.stderr or "Unknown remote" in result.stderr:
                logger.error(
                    "upstream remote not found. Add it with:\n"
                    "  git remote add upstream https://github.com/flagos-ai/FlagGems.git\n"
                    "Or run with --skip-fetch to bypass."
                )
            else:
                logger.error(
                    f"git fetch upstream failed: {result.stderr.strip()}\n"
                    f"Check network connection or run with --skip-fetch to bypass."
                )
            sys.exit(1)
        logger.info("Fetch completed successfully")
    elif args.skip_fetch or not auto_fetch:
        logger.warning(
            f"Skipping upstream fetch (--skip-fetch or auto_fetch_upstream=false). "
            f"Worktrees will be based on local {base_branch} which may be outdated."
        )

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
            logger.warning("Force shutdown requested, exiting immediately")
            os.system("stty sane 2>/dev/null")
            os._exit(1)
        shutdown_requested = True
        logger.warning(f"Shutdown requested (signal={sig}), killing {len(running)} running tasks...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting orchestrator: {len(ops)} operators, {len(device_mgr.gpu_ids)} GPUs, max_retries={max_retries}"
                + (f", vendor={vendor}" if vendor else ""))

    while (queue or running) and not shutdown_requested:
        # Launch new tasks if GPUs are available
        while queue and not shutdown_requested:
            gpu_id = device_mgr.acquire()
            if gpu_id is None:
                break

            operator, attempt = queue.popleft()
            try:
                worktree_path, branch = create_worktree(flaggems_dir, operator, vendor, base_branch)
                vendor_op = vendor_ops_map.get(operator)
                proc = launch_cc(operator, worktree_path, gpu_id, config, template_path, log_dir, vendor_op)

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

            # Check for timeout
            if timeout_per_op and proc.poll() is None and time.time() - start_time > timeout_per_op:
                logger.error(f"[TIMEOUT] {operator} exceeded {timeout_per_op}s, killing process")
                _kill_cc_process(proc)
                duration = time.time() - start_time
                device_mgr.release(gpu_id)
                del running[operator]
                summary.update_operator(
                    operator,
                    status="failed",
                    accuracy_passed=False,
                    duration_seconds=round(duration),
                    end_time=datetime.now(timezone.utc).isoformat(),
                    error_message=f"Timed out after {timeout_per_op}s",
                )
                continue

            if proc.poll() is not None:
                duration = time.time() - start_time
                device_mgr.release(gpu_id)
                del running[operator]

                # Parse result and generate timeline
                result = parse_cc_result(proc, operator, worktree_path)
                generate_timeline(proc._stdout_path, operator)

                # Vendor cross-check: files_created should reference the vendor dir
                if vendor and result.get("status") == "success":
                    files = result.get("files_created", [])
                    vendor_dir = f"_{vendor}"
                    if files and not any(vendor_dir in f for f in files):
                        logger.warning(
                            f"[WARN] {operator}: files_created does not reference "
                            f"vendor dir '{vendor_dir}': {files}"
                        )

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

    # Handle shutdown: kill running tasks immediately
    if shutdown_requested:
        for operator, (proc, gpu_id, attempt, wt, st) in running.items():
            _kill_cc_process(proc)
            device_mgr.release(gpu_id)
            summary.update_operator(
                operator,
                status="cancelled",
                end_time=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(time.time() - st),
            )

    device_mgr.release_all()
    summary.finalize()

    # Restore terminal state (claude CLI may leave it in raw/no-echo mode)
    try:
        os.system("stty sane 2>/dev/null")
    except Exception:
        pass

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
    parser.add_argument("--skip-fetch", action="store_true", help="Skip auto-fetch of upstream remote")
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
