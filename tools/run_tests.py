#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tests.py

Loops through the operator inventory and run tests identified.
The result is saved as a JSON/YAML file for inspection.

Usage:
    python run_tests.py --op-list /path/to/ops.txt --gpus 0,1,2,3
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, getcontext
from importlib import metadata
from pathlib import Path

import distro
from openpyxl import Workbook

# increase decimal precision
getcontext().prec = 18

# Global lock for writing result file and the result file
SUMMARY_LOCK = threading.Lock()
GLOBAL_RESULTS = {}
ENV_INFO = {}
HAS_TRITON = False
HAS_FLAGTREE = False

# robust numeric validator
NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def pinfo(str, **args):
    print(f"Info: {str}", **args)


def perror(str, **args):
    print(f"\033[31mError\033[0m: {str}", **args)


def pwarn(str, **args):
    print(f"\033[93mWarning \033[0m: {str}", **args)


def init():
    try:
        import torch

        version = torch.__version__
        ENV_INFO["torch"] = version
        pinfo(f"PyTorch detected ... {version}")
    except Exception as e:
        perror(f"pytorch not installed, please fix it - {e}")
        sys.exit(-1)

    try:
        version = metadata.version("flagtree")
        ENV_INFO["flagtree"] = version
        pinfo(f"flagtree detected ... {version}")
        HAS_FLAGTREE = True
    except Exception:
        HAS_FLAGTREE = False
        ENV_INFO["flagtree"] = None
        pwarn("flagtree not installed, testing Triton ...")

    try:
        import triton

        version = triton.__version__
        ENV_INFO["triton"] = version
        pinfo(f"triton detected ... {version}")
        if HAS_FLAGTREE:
            perror(
                "Both FlagTree and Triton are installed, please uninstall one of them."
            )
            sys.exit(-1)
    except Exception:
        ENV_INFO["triton"] = None
        if not HAS_FLAGTREE:
            perror("Neither FlagTree nor Triton is installed, please fix it.")
            sys.exit(-1)

    ENV_INFO["architecture"] = platform.machine()
    ENV_INFO["os_name"] = distro.id()
    ENV_INFO["os_release"] = distro.version()
    ENV_INFO["python"] = platform.python_version()

    print(json.dumps(ENV_INFO))

    try:
        # This may print an error "no device detected on your machine."
        import flag_gems

        version = flag_gems.__version__
        ENV_INFO["flag_gems"] = version
        pinfo(f"flag_gems detected ... {version}")
    except RuntimeError as e:
        perror(f"{e}")
        sys.exit(-1)
    except Exception as e:
        perror(f"{e}")
        perror("flag_gems has not been installed, please run `uv pip install -e .`")
        sys.exit(-1)


# TODO:
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd_capture(cmd, cwd=None, env=None):
    print(f"[INFO] CMD: {cmd}  (cwd={cwd})")
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return out or "", err or "", p.returncode


def is_number(s):
    return bool(NUM_RE.match(s.strip()))


def to_decimal(s):
    stripped = s.strip()
    if not is_number(stripped):
        raise ValueError(f"Not numeric: {s}")
    return Decimal(stripped)


def parse_pytest_summary_from_text(text):
    """
    Return: passed, failed, skipped, errors, total
    """
    ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    clean = ANSI_RE.sub("", text)
    counters = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
    }

    for m in re.finditer(r"(\d+)\s+([A-Za-z_]+)", clean):
        num = int(m.group(1))
        key = m.group(2).lower()
        if key in counters:
            counters[key] = num

    passed = counters["passed"]
    failed = counters["failed"]
    skipped = counters["skipped"]
    errors = counters["errors"]
    total = passed + failed + skipped
    return passed, failed, skipped, errors, total


def run_accuracy(op, gpu_id, flaggems_path, op_dir):
    print(f"[INFO][GPU {gpu_id}] Starting accuracy for '{op}'")
    env = os.environ.copy()
    # TODO(Qiming): env var name has to change
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    no_cpu_list = [
        "get_scheduler_metadata",
        "grouped_topk",
        "per_token_group_quant_fp8",
        "flash_attention_forward",
    ]
    if f"{op}" in no_cpu_list:
        cmd = f'pytest -m "{op}" -vs'
    else:
        cmd = f'pytest -m "{op}" --ref cpu -vs'
    out, err, code = run_cmd_capture(
        cmd, cwd=os.path.join(flaggems_path, "tests"), env=env
    )

    combined = out + "\n" + err
    acc_log = os.path.join(op_dir, "accuracy.log")
    with open(acc_log, "w") as f:
        f.write(combined)

    passed, failed, skipped, errors, total = parse_pytest_summary_from_text(combined)

    # ✅ 新 status 规则
    if failed > 0:
        status = "FAIL"
    elif errors > 0 and total == 0:
        status = "FAIL"  # pytest 没跑起来
    elif passed == 0:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "total": total,
        "status": status,
        "log_path": acc_log,
        "exit_code": code,
    }


def run_benchmark_and_parse(op, gpu_id, flaggems_path, op_dir):
    print(f"[INFO][GPU {gpu_id}] Starting benchmark for '{op}'")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    benchmark_dir = os.path.join(flaggems_path, "benchmark")
    ensure_dir(benchmark_dir)

    pattern = f"result-m_{op}--level_core--record_log.log"
    for p in Path(benchmark_dir).glob(pattern):
        try:
            p.unlink()
        except Exception:
            pass

    cmd = f'pytest -m "{op}" --level core --record log'
    out, err, code = run_cmd_capture(cmd, cwd=benchmark_dir, env=env)

    # TODO(Qiming): handle code

    perf_log = os.path.join(op_dir, "perf.log")
    with open(perf_log, "w") as f:
        f.write(out + "\n" + err)

    perf_result_file = None
    for p in Path(benchmark_dir).glob(pattern):
        perf_result_file = str(p)
        break

    if not perf_result_file:
        return {
            "status": "NO_RESULT",
            "perf_console_log": perf_log,
            "perf_result_file": None,
            "parsed_summary": None,
            "performance_rows": [],
        }

    dest = os.path.join(op_dir, os.path.basename(perf_result_file))
    shutil.move(perf_result_file, dest)
    perf_result_file = dest

    with open(perf_result_file) as f:
        lines = f.readlines()
    seen = set()
    uniq = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)
    with open(perf_result_file, "w") as f:
        f.writelines(uniq)

    parsed_summary_path = os.path.join(op_dir, "parsed_summary.log")
    try:
        cmd = f'python3 summary_for_plot.py "{perf_result_file}"'
        out2, err2, _ = run_cmd_capture(cmd, cwd=benchmark_dir)
        combined2 = out2 + "\n" + err2

        with open(parsed_summary_path, "w") as f:
            f.write(combined2)
    except Exception:
        pass

    performance_rows = []
    try:
        with open(parsed_summary_path) as f:
            lines = f.readlines()

        start = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("XCCL"):
                continue
            if line.lower().startswith("op_name"):
                start = True
                continue
            if not start:
                continue

            cols = re.split(r"\s+", line)
            if len(cols) < 8:
                continue

            row = {
                "func_name": cols[0],
                "float16": cols[1],
                "float32": cols[2],
                "bfloat16": cols[3],
                "int16": cols[4],
                "int32": cols[5],
                "bool": cols[6],
                "cfloat": cols[7],
            }

            vals = []
            for v in row.values():
                try:
                    dv = to_decimal(v)
                    if dv > 0:
                        vals.append(dv)
                except Exception:
                    pass

            row["avg_speedup"] = (
                str(round(float(sum(vals) / len(vals)), 6)) if vals else "0"
            )
            performance_rows.append(row)

    except Exception:
        pass

    return {
        "status": "OK",
        "perf_console_log": perf_log,
        "perf_result_file": perf_result_file,
        "parsed_summary": parsed_summary_path,
        "performance_rows": performance_rows,
    }


def worker_proc(gpu_id, ops_list, flaggems_path, results_dir):
    for op in ops_list:
        op = op.strip()
        if not op:
            continue

        op_dir = os.path.join(results_dir, op)
        ensure_dir(op_dir)

        acc = run_accuracy(op, gpu_id, flaggems_path, op_dir)
        perf = run_benchmark_and_parse(op, gpu_id, flaggems_path, op_dir)

        with SUMMARY_LOCK:
            GLOBAL_RESULTS[op] = {"gpu": gpu_id, "accuracy": acc, "performance": perf}
            write_summary_json(results_dir)
            write_xlsx(results_dir)


def write_summary_json(results_dir):
    json_path = os.path.join(results_dir, "summary.json")
    data = [
        {
            "operator": op,
            "accuracy": info["accuracy"],
            "performance": info["performance"]["performance_rows"],
        }
        for op, info in GLOBAL_RESULTS.items()
    ]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def write_xlsx(path):
    xlsx_path = os.path.join(path, "summary.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    ws.append(
        [
            "operator",
            "acc_status",
            "passed",
            "failed",
            "skipped",
            "errors",
            "total",
            "acc_exit_code",
            "func_name",
            "avg_speedup",
            "float16",
            "float32",
            "bfloat16",
            "int16",
            "int32",
            "bool",
            "cfloat",
            "perf_status",
            "perf_console_log",
            "perf_result_file",
            "parsed_summary",
        ]
    )

    for op, info in GLOBAL_RESULTS.items():
        acc = info["accuracy"]
        perf = info["performance"]
        rows = perf["performance_rows"] or [{}]
        first = True

        for r in rows:
            ws.append(
                [
                    op if first else "",
                    acc["status"] if first else "",
                    acc["passed"] if first else "",
                    acc["failed"] if first else "",
                    acc["skipped"] if first else "",
                    acc["errors"] if first else "",
                    acc["total"] if first else "",
                    acc["exit_code"] if first else "",
                    r.get("func_name", ""),
                    r.get("avg_speedup", ""),
                    r.get("float16", ""),
                    r.get("float32", ""),
                    r.get("bfloat16", ""),
                    r.get("int16", ""),
                    r.get("int32", ""),
                    r.get("bool", ""),
                    r.get("cfloat", ""),
                    perf["status"],
                    perf["perf_console_log"],
                    perf["perf_result_file"],
                    perf["parsed_summary"],
                ]
            )
            first = False

    wb.save(xlsx_path)


def main():
    init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--flaggems", required=True)
    parser.add_argument("--op-list", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    # TODO(Qiming): parse backend and probe customized or not
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    with open(args.op_list) as f:
        ops = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"results_{now_ts()}"
    )
    ensure_dir(results_dir)

    tasks = {gpu_id: [] for gpu_id in gpu_ids}
    for i, op in enumerate(ops):
        tasks[gpu_ids[i % len(gpu_ids)]].append(op)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as exe:
        futures = []
        for gpu in gpu_ids:
            if tasks[gpu]:
                futures.append(
                    exe.submit(worker_proc, gpu, tasks[gpu], args.flaggems, results_dir)
                )
        for f in as_completed(futures):
            f.result()

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
