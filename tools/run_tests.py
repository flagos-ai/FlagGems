#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tests.py — Run accuracy and performance tests for FlagGems operators.

When stdout is a TTY, shows a live display: completed results scroll upward
while a pinned footer shows one status line per GPU. When output is piped or
redirected, falls back to plain line-by-line output with no ANSI codes.
"""
import argparse
import datetime
import json
import os
import platform
import queue as queue_module
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from decimal import getcontext
from importlib import metadata
from multiprocessing import Process, Queue
from pathlib import Path

import consts
import distro
import git
import yaml

import flag_gems

getcontext().prec = 18
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[93m"
CYAN = "\033[36m"
DIM = "\033[2m"
NC = "\033[0m"

ROOT = Path(__file__).parent.parent

OPTS = argparse.Namespace()
CFG = types.SimpleNamespace()

GLOBAL_RESULTS = {}
ENV_INFO = {}

TIMEOUT = -100
WORKER_PROCESSES = []
INTERRUPTED = False

IS_TTY = sys.stdout.isatty()
USE_COLORS = IS_TTY

if not USE_COLORS:
    RED = GREEN = YELLOW = CYAN = DIM = NC = ""


def pinfo(msg, **kwargs):
    print(f"{GREEN}[INFO]{NC} {msg}", flush=True, **kwargs)


def perror(msg, **kwargs):
    print(f"{RED}[ERROR]{NC} {msg}", flush=True, **kwargs)


def pwarn(msg, **kwargs):
    print(f"{YELLOW}[WARN]{NC} {msg}", flush=True, **kwargs)


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


class LiveDisplay:
    """Manages terminal output with a pinned footer for GPU status lines."""

    def __init__(self, gpu_ids, op_count, op_width=20):
        self.gpu_ids = gpu_ids
        self.op_count = op_count
        self.op_width = op_width
        self.gpu_index = {gid: i + 1 for i, gid in enumerate(gpu_ids)}
        # Match progress line width to scrolling log lines (55 + op_width visible chars).
        # Progress line: "[Progress] [" (12) + bar + "]  " (3) + nums_str
        nums_width = len(f"{op_count}/{op_count} ops")
        self.bar_width = max(20, 55 + op_width - 12 - 3 - nums_width)
        self.nums_width = nums_width
        progress_line = self._fmt_progress(0)
        gpu_lines = [f"{DIM}[GPU {gid:2d}] idle{NC}" for gid in gpu_ids]
        self.footer = [progress_line] + gpu_lines
        self.n = len(self.footer)
        self.footer_drawn = False

    def _fmt_progress(self, tests_done):
        total_tests = self.op_count * 2
        color = GREEN if tests_done >= total_tests else CYAN
        bar = (
            _progress_bar(tests_done, total_tests, self.bar_width, color=color)
            if self.op_count
            else " " * self.bar_width
        )
        ops_done = tests_done // 2
        nums = f"{ops_done}/{self.op_count} ops"
        return f"[Progress] [{color}{bar}{NC}]  {nums:>{self.nums_width}}"

    def _draw_footer(self):
        if not IS_TTY:
            return
        for line in self.footer:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.footer_drawn = True

    def _erase_footer(self):
        if not IS_TTY or not self.footer_drawn:
            return
        for _ in range(self.n):
            sys.stdout.write("\033[A\033[2K")

    def init(self):
        if IS_TTY:
            self._draw_footer()

    def log(self, msg):
        """Print a scrolling log line above the footer."""
        if IS_TTY:
            self._erase_footer()
            sys.stdout.write(msg + "\n")
            self._draw_footer()
        else:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def update_gpu(self, gpu_id, status_line):
        """Update a GPU's footer line."""
        idx = self.gpu_index.get(gpu_id)
        if idx is None:
            return
        self.footer[idx] = status_line
        if IS_TTY:
            self._erase_footer()
            self._draw_footer()

    def update_progress(self, tests_done):
        """Update the global progress bar."""
        self.footer[0] = self._fmt_progress(tests_done)
        if IS_TTY:
            self._erase_footer()
            self._draw_footer()
        else:
            sys.stdout.write(self.footer[0] + "\n")
            sys.stdout.flush()

    def finish(self):
        """Clear the footer when done."""
        if IS_TTY:
            self._erase_footer()
            sys.stdout.flush()


def _progress_bar(done, total, width=40, color=""):
    if not total:
        return " " * width
    frac = done * width / total
    full = int(frac)
    has_half = (frac - full) >= 0.5 and full < width
    empty = width - full - (1 if has_half else 0)
    bar = "█" * full
    if has_half:
        bar += f"{DIM}█{NC}{color}"
    bar += " " * empty
    return bar


def _format_status(status, dur):
    STATUS_MAP = {
        "Passed": (GREEN, "OK"),
        "Failed": (RED, "FAILED"),
        "Timeout": (RED, "TIMEOUT"),
        "Error": (RED, "ERROR"),
        "NotFound": (YELLOW, "NOTFOUND"),
        "Skipped": (YELLOW, "SKIPPED"),
    }
    color, label = STATUS_MAP.get(status, (YELLOW, status.upper()))
    return f"{color}[{label:<8} {dur:>6.1f}s]{NC}"


def get_ops_from_inventory():
    catalog = []
    try:
        op_inventory = ROOT / "conf" / "operators.yaml"
        with open(str(op_inventory), "r") as f:
            data = yaml.safe_load(f)
            catalog = data.get("ops", [])
    except Exception as e:
        perror(f"Failed to load operator inventory: {e}")
    return catalog


def _probe_torch():
    ENV_INFO.setdefault("torch", {})
    try:
        import torch

        version = torch.__version__
        ENV_INFO["torch"]["version"] = version
        pinfo(f"PyTorch detected ... {version}")
    except Exception as e:
        perror(f"pytorch not installed, please fix it - {e}")
        sys.exit(-1)

    try:
        cuda_available = torch.cuda.is_available()
        ENV_INFO["torch"]["cuda_available"] = cuda_available
        pinfo(f"PyTorch CUDA support ... {cuda_available}")
    except Exception:
        ENV_INFO["torch"]["cuda_available"] = False

    try:
        dev_name = torch.cuda.get_device_name()
        ENV_INFO["torch"]["device_name"] = dev_name
        pinfo(f"PyTorch device name ... {dev_name}")
    except Exception:
        ENV_INFO["torch"]["device_name"] = "N/A"

    try:
        dev_count = torch.cuda.device_count()
        ENV_INFO["torch"]["device_count"] = dev_count
        pinfo(f"PyTorch device count ... {dev_count}")
    except Exception:
        ENV_INFO["torch"]["device_count"] = 0
        dev_count = 0

    if dev_count == 0:
        try:
            # Is this a TsingMicro chip?
            import torch_txda

            dev_count = torch_txda.device_count()

            ENV_INFO["torch"]["device_count"] = dev_count
            pinfo(f"TorchTXDA device count ... {dev_count}")
        except Exception:
            pass


def _probe_triton():
    try:
        version = metadata.version("flagtree")
        ENV_INFO["flagtree"] = version
        pinfo(f"FlagTree (flagtree) detected ... {version}")
        has_flagtree = True
    except Exception:
        has_flagtree = False
        ENV_INFO["flagtree"] = None
        pwarn("FlagTree (flagtree) not installed, testing Triton ...")

    try:
        import triton

        version = triton.__version__
        ENV_INFO["triton"] = {"version": version}
        pinfo(f"Triton (triton) detected ... {version}")

        if version:
            has_config = hasattr(triton, "Config")
            ENV_INFO["triton"]["has_config"] = has_config
            pinfo(f"Triton (triton) has Config ... [{has_config}]")
    except Exception:
        ENV_INFO["triton"] = None
        if not has_flagtree:
            perror("Neither FlagTree nor Triton is installed, please fix it.")
            sys.exit(-1)


def _probe_flaggems():
    try:
        version = flag_gems.__version__
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        ver_str = f"{version}+git{sha[:8]}"
        ENV_INFO["flag_gems"] = {"version": ver_str}
        pinfo(f"flag_gems detected ... {ver_str}")
    except RuntimeError as e:
        perror(f"{e}")
        sys.exit(-1)
    except Exception as e:
        perror(f"{e}")
        perror("flag_gems has not been installed, please run `uv pip install -e .`")
        sys.exit(-1)

    try:
        vendor = flag_gems.vendor_name
        ENV_INFO["flag_gems"]["vendor"] = vendor
        pinfo(f"flag_gems vendor detection ... {vendor}")
    except Exception as e:
        perror(f"{e}")
        perror("flag_gems failed to detect vendor info.`")
        sys.exit(-1)

    try:
        device = flag_gems.device
        ENV_INFO["flag_gems"]["device"] = device
        pinfo(f"flag_gems device detection ... {device}")
    except Exception as e:
        perror(f"{e}")
        perror("flag_gems failed to detect device info.`")
        sys.exit(-1)


def probe_env():
    ENV_INFO["architecture"] = platform.machine()
    ENV_INFO["os_name"] = distro.id()
    ENV_INFO["os_release"] = distro.version()
    ENV_INFO["python"] = platform.python_version()

    _probe_torch()
    _probe_triton()
    _probe_flaggems()


def get_env(gpu_ids):
    env = os.environ.copy()
    vendor = ENV_INFO.get("flag_gems", {}).get("vendor", "")

    vendor_env_map = {
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

    env_vars = vendor_env_map.get(vendor, ["CUDA_VISIBLE_DEVICES"])
    for var in env_vars:
        env[var] = gpu_ids
    return env


def run_cmd(op, cmd, cwd=None, env=None, timeout=600, flavor=None):
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if CFG.dump_output:
        op_dir = CFG.output_dir.joinpath(op)
        stdout_log = str(op_dir / f"{flavor}_stdout.log")
        stderr_log = str(op_dir / f"{flavor}_stderr.log")
        try:
            stdout = open(stdout_log, "w")
            stderr = open(stderr_log, "w")
        except Exception:
            pass

    p = subprocess.Popen(
        shlex.split(cmd),
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )

    try:
        p.wait(timeout=timeout)
        return p.returncode
    except subprocess.TimeoutExpired:
        pgid = os.getpgid(p.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return TIMEOUT
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return TIMEOUT
    except Exception as e:
        perror(f"run_cmd failed: {e}")
        return -1
    finally:
        if stdout != subprocess.DEVNULL:
            stdout.close()
        if stderr != subprocess.DEVNULL:
            stderr.close()


def parse_accuracy_data(result_file):
    raw_data = {}
    try:
        with result_file.open("r") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {
            "total": 0,
            "skipped": 0,
            "failed": 0,
            "passed": 0,
            "status": "Error",
            "details": {"error": f"Invalid JSON in {result_file}"},
        }

    passed = []
    skipped = {}
    failed = {}
    num_skipped = 0
    num_failed = 0
    num_passed = 0
    skipped_with_issue = False
    for test_case, item in raw_data.items():
        case_str = test_case[: test_case.find("[")]
        result = item.get("result", "")
        params = [case_str]
        for k, v in item.get("params", {}).items():
            params.append(str(v).replace(" ", ""))
        param_str = ":".join(params)

        if result == "passed":
            passed.append(param_str)
            num_passed += 1
        elif result == "skipped":
            reason = item.get("reason", "Unknown")
            if "Issue" in reason:
                skipped_with_issue = True
            skipped.setdefault(reason, set())
            skipped[reason].add(param_str)
            num_skipped += 1
        else:
            reason = item.get("reason", "Unknown")
            failed.setdefault(reason, set())
            failed[reason].add(param_str)
            num_failed += 1

    num_total = num_passed + num_skipped + num_failed
    result = {
        "total": num_total,
        "skipped": num_skipped,
        "failed": num_failed,
        "passed": num_passed,
        "details": {},
    }
    if len(skipped) == 0 and len(failed) == 0:
        if len(passed) == 0:
            result["status"] = "NotFound"
        else:
            result["status"] = "Passed"
        return result

    if num_failed > 0:
        result["status"] = "Failed"
        for k, v in failed.items():
            failed[k] = list(v)
        result["details"]["failed"] = failed
        return result

    if skipped_with_issue:
        result["status"] = "Failed"
    else:
        result["status"] = "Skipped"

    for k, v in skipped.items():
        skipped[k] = list(v)
    result["details"]["skipped"] = skipped
    return result


def parse_perf_data(op, result_file):
    raw_data = {}
    try:
        with result_file.open("r") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {
            "status": "Error",
            "reason": f"Invalid JSON in {result_file}",
        }

    data = raw_data.get(op, {})
    if not data:
        return {"status": "NotFound"}

    result = data.get("result", "NotFound")
    if result in ["failed", "skipped"]:
        return {
            "status": result.title(),
            "reason": data.get("reason", "Unknown"),
            "test_case": data.get("test_case", "Unknown"),
        }

    bench_res = {}
    records = data.get("details", [])

    for item in records:
        dtype = consts.DTYPE_MAP.get(item["dtype"], item["dtype"])
        details = {}
        total = 0.0
        count = 0
        for res in item.get("result", []):
            shape = str(res.get("shape_detail", "Unknown")).replace(" ", "")
            details.setdefault(shape, {})
            details[shape]["base"] = res.get("latency_base", 0.0)
            details[shape]["gems"] = res.get("latency", 0.0)
            speedup = res.get("speedup", 0.0)
            details[shape]["speedup"] = speedup
            count += 1
            total += speedup

        if details:
            bench_res[dtype] = {
                "result": "OK",
                "details": details,
                "speedup": total / count,
            }
        else:
            bench_res[dtype] = {
                "result": "Unknown",
                "details": {},
                "speedup": 0,
            }

    return {
        "status": result.title(),
        "data": bench_res,
        "test_case": data.get("test_case", "Unknown"),
    }


def run_accuracy_q(gpu_id, op):
    """Run accuracy test for one op. Returns result dict."""
    env = get_env(str(gpu_id))

    if op in CFG.skip_cpu_tests:
        cmd = f'pytest -m "{op}" --record json --output accuracy_{op}.json -vs'
    else:
        cmd = (
            f'pytest -m "{op}" --record json --output accuracy_{op}.json --ref cpu -vs'
        )

    accuracy_dir = ROOT.joinpath("tests")
    result_file = accuracy_dir / f"accuracy_{op}.json"
    if result_file.exists():
        result_file.unlink()

    op_dir = CFG.output_dir.joinpath(op)
    ensure_dir(op_dir)
    dur = time.time()
    code = run_cmd(op, cmd, cwd=accuracy_dir, env=env, flavor="accuracy")
    dur = time.time() - dur

    if code == TIMEOUT:
        return {
            "status": "Timeout",
            "exit_code": TIMEOUT,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": dur,
        }

    if not result_file.exists():
        return {
            "status": "Error",
            "exit_code": code,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 1,
            "duration": dur,
            "data_file": None,
        }

    op_dir = CFG.output_dir.joinpath(op)
    dest = op_dir / "accuracy_result.json"
    shutil.move(result_file, str(dest))
    result_file = dest

    result = parse_accuracy_data(result_file)
    result["exit_code"] = code
    result["duration"] = dur
    result["data_file"] = str(result_file.relative_to(CFG.output_dir))
    return result


def run_benchmark_q(gpu_id, op):
    """Run benchmark for one op. Returns result dict."""
    env = get_env(str(gpu_id))

    benchmark_dir = ROOT / "benchmark"
    result_file = benchmark_dir / f"benchmark_{op}.json"
    if result_file.exists():
        result_file.unlink()

    op_dir = CFG.output_dir.joinpath(op)
    ensure_dir(op_dir)

    dur = time.time()
    cmd = f'pytest -m "{op}" --level core --record json --output benchmark_{op}.json'
    if ENV_INFO["flag_gems"]["vendor"] == "kunlunxin":
        cmd += " --fg_mode operator"
    code = run_cmd(op, cmd, cwd=benchmark_dir, env=env, flavor="performance")
    dur = time.time() - dur

    if code == TIMEOUT:
        return {
            "status": "Timeout",
            "exit_code": TIMEOUT,
            "duration": dur,
            "data": {},
        }

    if not result_file.exists():
        return {
            "status": "NotFound",
            "duration": dur,
            "exit_code": code,
            "data": {},
        }

    dest = op_dir / "performance_result.json"
    shutil.move(result_file, str(dest))
    result_file = dest

    record = {
        "duration": dur,
        "exit_code": code,
        "data_file": str(result_file.relative_to(CFG.output_dir)),
        "data": {},
    }
    record.update(parse_perf_data(op, result_file))
    return record


def worker_proc(gpu_id, work_queue, display_queue):
    # Suppress direct stdout/stderr from worker processes to prevent
    # corrupting the main process's terminal cursor positioning.
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    worker_result = {}
    while True:
        try:
            op = work_queue.get_nowait()
        except queue_module.Empty:
            break
        op = op.strip()
        if not op:
            continue

        op_dir = CFG.output_dir.joinpath(op)
        ensure_dir(op_dir)

        display_queue.put(("start", gpu_id, "accuracy", op))
        acc = run_accuracy_q(gpu_id, op)
        display_queue.put(
            (
                "done",
                gpu_id,
                "accuracy",
                op,
                acc.get("status", "Error"),
                acc.get("duration", 0),
            )
        )

        display_queue.put(("start", gpu_id, "benchmark", op))
        perf = run_benchmark_q(gpu_id, op)
        display_queue.put(
            (
                "done",
                gpu_id,
                "benchmark",
                op,
                perf.get("status", "Error"),
                perf.get("duration", 0),
            )
        )

        customized_ops = [o[0] for o in flag_gems.runtime.backend.get_customized_ops()]
        result = {
            "customized": op in customized_ops,
            "accuracy": acc,
            "performance": perf,
        }
        worker_result.setdefault(op, result)

        json_path = CFG.output_dir.joinpath(f"summary{gpu_id}.json")
        tmp_path = json_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(worker_result, f, indent=2)
        os.replace(tmp_path, json_path)

    display_queue.put(("exit", gpu_id))


def display_loop(queue, display, n_workers):
    exited = 0
    tests_done = 0
    per_gpu_done = {gid: 0 for gid in display.gpu_ids}

    while exited < n_workers:
        try:
            msg = queue.get(timeout=1)
        except Exception:
            continue

        kind = msg[0]

        if kind == "exit":
            gpu_id = msg[1]
            n = per_gpu_done.get(gpu_id, 0)
            display.update_gpu(gpu_id, f"{DIM}[GPU {gpu_id:2d}] done ({n} ops){NC}")
            exited += 1

        elif kind == "start":
            _, gpu_id, phase, op = msg
            label = "accuracy " if phase == "accuracy" else "benchmark"
            op_display = (
                op
                if len(op) <= display.op_width
                else op[: display.op_width - 3] + "..."
            )
            op_col = op_display.ljust(display.op_width)
            n = per_gpu_done.get(gpu_id, 0)
            if IS_TTY:
                display.update_gpu(
                    gpu_id,
                    f"[GPU {gpu_id:2d}] ({n:>3} done)  {label} {op_col}",
                )
            else:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                display.log(f"[INFO] [{ts}][GPU {gpu_id:2d}]" f" {label} {op_col} ...")

        elif kind == "done":
            _, gpu_id, phase, op, status, dur = msg
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            label = "accuracy " if phase == "accuracy" else "benchmark"
            op_display = (
                op
                if len(op) <= display.op_width
                else op[: display.op_width - 3] + "..."
            )
            op_col = op_display.ljust(display.op_width)
            status_str = _format_status(status, dur)
            log_line = (
                f"{GREEN}[INFO]{NC} [{ts}][GPU {gpu_id:2d}]"
                f" {label} {op_col} {status_str}"
            )

            tests_done += 1
            if phase == "benchmark":
                per_gpu_done[gpu_id] = per_gpu_done.get(gpu_id, 0) + 1

            ops_done = tests_done // 2
            total_ops = display.op_count
            pct = ops_done * 100 // total_ops
            if not IS_TTY:
                total_w = len(str(total_ops))
                log_line += f"  ({pct:>3}% {ops_done:>{total_w}}/{total_ops} ops)"

            # Update progress state BEFORE log so the footer is drawn once
            # with the correct progress value.
            display.footer[0] = display._fmt_progress(tests_done)
            display.log(log_line)


def cleanup_intermediate_files():
    patterns = [
        (ROOT / "tests", "accuracy_*.json"),
        (ROOT / "benchmark", "benchmark_*.json"),
    ]
    for directory, pattern in patterns:
        for f in directory.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass

    if hasattr(CFG, "output_dir"):
        for f in CFG.output_dir.glob("summary*.tmp"):
            try:
                f.unlink()
            except OSError:
                pass


def terminate_workers():
    for p in WORKER_PROCESSES:
        if p.is_alive():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    for p in WORKER_PROCESSES:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()


def handle_interrupt(signum, frame):
    global INTERRUPTED
    if INTERRUPTED:
        return
    INTERRUPTED = True
    if IS_TTY:
        sys.stdout.write("\n")
    pwarn("Interrupted. Cleaning up ...")
    terminate_workers()
    cleanup_intermediate_files()
    pwarn("Cleanup done.")
    sys.exit(1)


def get_ops_to_test():
    op_catalog = get_ops_from_inventory()
    skip_cpu_tests = []
    for op in op_catalog:
        labels = op.get("labels", [])
        if "NoCPU" in labels:
            skip_cpu_tests.append(op["id"])
    CFG.skip_cpu_tests = skip_cpu_tests

    if OPTS.ops:
        ops = []
        for op in OPTS.ops.split(","):
            ops.append(op.strip().lstrip("_"))
        return ops

    if OPTS.op_list_file:
        lines = []
        try:
            with open(OPTS.op_list_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            perror(f"Failed reading the specified op list file: {e}")
            return []

        ops = []
        for ln in lines:
            ln = ln.strip()
            if ln.startswith("#"):
                continue
            ops.append(ln.lstrip("_"))
        return ops

    effective_stages = []
    for s in OPTS.stages.split(","):
        stage = s.strip()
        if stage not in ["alpha", "beta", "stable", "all", "removed"]:
            pwarn(f"ignoring unsupported stage name '{s}'...")
            continue
        if stage == "all":
            effective_stages = ["alpha", "beta", "stable"]
            break
        effective_stages.append(stage)

    if not effective_stages:
        effective_stages = ["stable"]

    ops = []
    for op in op_catalog:
        stages = op.get("stages", [])
        if len(stages) == 0:
            continue
        stage = next(iter(stages[-1].keys()), None)
        if stage not in effective_stages:
            continue
        if OPTS.start is not None and op["id"] < OPTS.start:
            continue
        ops.append(op["id"])

    return ops


def main():
    global OPTS

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops", required=False, help="a comma-separated list of op IDs"
    )
    parser.add_argument(
        "--op-list-file", required=False, help="path to operator list file"
    )
    parser.add_argument("--start", required=False, help="the ID of the first operator")
    parser.add_argument("--gpus", default="0", help="a comma-separated list of GPU IDs")
    parser.add_argument(
        "--output-dir", default="results", help="relative path to root for test data"
    )
    parser.add_argument(
        "--stages",
        required=False,
        default="stable",
        help="a comma-separate list of op stages",
    )
    parser.add_argument(
        "--dump-output",
        action="store_true",
        default=False,
        help="Dump stdout/stderr of each test to log files",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Control ANSI color output: auto (TTY only), always, or never",
    )
    OPTS = parser.parse_args()
    CFG.dump_output = OPTS.dump_output
    CFG.start = OPTS.start

    # Apply color mode (IS_TTY controls cursor-based footer, USE_COLORS controls ANSI colors)
    global USE_COLORS, RED, GREEN, YELLOW, CYAN, DIM, NC
    if OPTS.color == "always":
        USE_COLORS = True
        RED, GREEN, YELLOW, CYAN, DIM, NC = (
            "\033[31m",
            "\033[32m",
            "\033[93m",
            "\033[36m",
            "\033[2m",
            "\033[0m",
        )
    elif OPTS.color == "never":
        USE_COLORS = False
        RED = GREEN = YELLOW = CYAN = DIM = NC = ""

    probe_env()

    ops = get_ops_to_test()
    op_count = len(ops)
    if op_count == 0:
        pwarn("No operators to test. Please specify at lease one operator.")
        sys.exit(1)
    pinfo(f"Testing {op_count} operators ...")

    CFG.ops = ops

    output_dir = Path(OPTS.output_dir)
    ensure_dir(output_dir)
    CFG.output_dir = output_dir

    if OPTS.gpus.strip().lower() == "all":
        dev_count = ENV_INFO.get("torch", {}).get("device_count", 0)
        if dev_count == 0:
            perror("--gpus all specified but no devices detected.")
            sys.exit(1)
        gpu_ids = list(range(dev_count))
    else:
        gpu_list = OPTS.gpus.strip().split(",")
        if len(gpu_list) == 0:
            pwarn("Empty GPU list specified.")
            sys.exit(1)
        gpu_ids = [int(x) for x in gpu_list if x.strip()]
    gpu_count = len(gpu_ids)

    op_width = min(max(len(op) for op in ops), 40) if ops else 20

    work_queue = Queue()
    for op in ops:
        work_queue.put(op)

    display_queue = Queue()
    display = LiveDisplay(gpu_ids, op_count, op_width=op_width)

    for gpu in gpu_ids:
        p = Process(target=worker_proc, args=(gpu, work_queue, display_queue))
        p.start()
        WORKER_PROCESSES.append(p)

    display.init()
    display_loop(display_queue, display, gpu_count)

    for p in WORKER_PROCESSES:
        p.join()

    display.finish()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    op_data = {}
    for gpu_id in gpu_ids:
        gpu_file = CFG.output_dir.joinpath(f"summary{gpu_id}.json")
        if not gpu_file.exists():
            perror(f"GPU {gpu_id} failed to produce a summary, recovery needed.")
            continue
        with gpu_file.open("r") as f:
            try:
                result = json.load(f)
            except (json.JSONDecodeError, ValueError):
                perror(f"GPU {gpu_id} summary is invalid JSON, skipping.")
                continue
            op_data.update(result)

    final_data = {
        "timestamp": timestamp,
        "env": ENV_INFO,
        "result": op_data,
    }

    json_path = CFG.output_dir.joinpath("summary.json")
    with json_path.open("w") as f:
        json.dump(final_data, f, indent=2)

    cleanup_intermediate_files()
    pinfo("Test completed.")


if __name__ == "__main__":
    main()
