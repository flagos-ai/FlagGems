#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import openpyxl
import yaml

# increase decimal precision
getcontext().prec = 18

# Global lock for writing result file and the result file
SUMMARY_LOCK = threading.Lock()
GLOBAL_RESULTS = {}
ENV_INFO = {}
HAS_TRITON = False
HAS_FLAGTREE = False
ROOT = Path(__file__).parent.parent
OUPUT_DIR = None

NO_CPU_LIST = [
    "flash_attention_forward",
    "get_scheduler_metadata",
    "grouped_topk",
    "per_token_group_quant_fp8",
]

DTYPE_MAP = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.bool": "bool",
    "torch.complex64": "cf64",
}

# Regex for numeric validator
NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

# Regex for ANSI
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def pinfo(str, **args):
    print(f"\033[32m[INFO]\033[0m {str}", **args)


def perror(str, **args):
    print(f"\033[31m[ERROR]\033[0m {str}", **args)


def pwarn(str, **args):
    print(f"\033[93m[WARNING]\033[0m {str}", **args)


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def to_decimal(s):
    stripped = s.strip()
    is_number = bool(NUM_RE.match(stripped))
    if not is_number:
        raise ValueError(f"Not numeric: {s}")
    return Decimal(stripped)


def get_ops():
    catalog = []
    try:
        op_inventory = ROOT / "conf" / "operators.yaml"  # noqa: E226
        with open(str(op_inventory), "r") as f:
            data = yaml.safe_load(f)
            catalog = data.get("ops", [])
    except Exception as e:
        perror(f"Failed to load operator inventory: {e}")

    return catalog


def init():
    ENV_INFO["architecture"] = platform.machine()
    ENV_INFO["os_name"] = distro.id()
    ENV_INFO["os_release"] = distro.version()
    ENV_INFO["python"] = platform.python_version()

    try:
        import torch

        version = torch.__version__
        ENV_INFO["torch"] = {"version": version}
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

    try:
        version = metadata.version("flagtree")
        ENV_INFO["flagtree"] = version
        pinfo(f"FlagTree (flagtree) detected ... {version}")
        HAS_FLAGTREE = True
    except Exception:
        HAS_FLAGTREE = False
        ENV_INFO["flagtree"] = None
        pwarn("FlagTree (flagtree) not installed, testing Triton ...")

    try:
        import triton

        version = triton.__version__
        ENV_INFO["triton"] = {"version": version}
        pinfo(f"Triton (triton) detected ... {version}")

        # TODO(Qiming): Fix this. FlagTree contains a Triton, which should not be treated as conflict.
        # if HAS_FLAGTREE:
        #     perror(
        #        "Both FlagTree and Triton are installed, please uninstall one of them."
        #    )
        #    sys.exit(-1)

        if version:
            has_config = hasattr(triton, "Config")
            ENV_INFO["triton"]["has_config"] = has_config
            pinfo(f"Triton (triton) has Config ... [{has_config}]")

    except Exception:
        ENV_INFO["triton"] = None
        if not HAS_FLAGTREE:
            perror("Neither FlagTree nor Triton is installed, please fix it.")
            sys.exit(-1)

    try:
        # This may print an error "no device detected on your machine."
        import flag_gems

        version = flag_gems.__version__
        ENV_INFO["flag_gems"] = {"version": version}
        pinfo(f"flag_gems detected ... {version}")
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


def run_cmd_capture(cmd, cwd=None, env=None):
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return out or "", err or "", p.returncode


def parse_accuracy_log(text):
    counter = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "total": 0,
    }

    clean = ANSI_RE.sub("", text)
    for m in re.finditer(r"(\d+)\s+([A-Za-z_]+)", clean):
        num = int(m.group(1))
        key = m.group(2).lower()
        if key in counter:
            counter[key] = num

    total = counter["failed"] + counter["passed"] + counter["skipped"]
    counter["total"] = total

    if counter["failed"] > 0:
        counter["status"] = "FAIL"
    elif counter["errors"] > 0 and total == 0:
        counter["status"] = "FAIL"  # pytest failed to start
    elif counter["passed"] == 0:
        counter["status"] = "FAIL"
    else:
        counter["status"] = "PASS"

    return counter


def get_env(gpu_ids):
    env = os.environ.copy()
    vendor = ENV_INFO.get("flag_gems", {}).get("vendor", "")

    if vendor == "ascend":
        env["ASCEND_RT_VISIBLE_DEVICES"] = gpu_ids
        env["NPU_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "mthreads":
        env["MUSA_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "hygon":
        env["HIP_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "tsingmicro":
        env["TXDA_VISIBLE_DEVICES"] = gpu_ids
        return env

    # NOTE: Iluvatar said to support CUDA_VISIBLE_DEVICES as well
    if vendor == "iluvatar":
        env["ILUVATAR_VISIBLE_DEVICES"] = gpu_ids
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return env

    # MetaX is using CUDA_VISIBLE_DEVICES as well
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    return env


def dedup(fn):
    with open(fn) as f:
        lines = f.readlines()

    # Compress verbose output
    seen = set()
    uniq = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            uniq.append(line)

    with open(fn, "w") as f:
        f.writelines(uniq)


def run_accuracy(op, gpu_id, op_dir):
    pinfo(f"[GPU {gpu_id:2d}] Running accuracy tests for '{op}'")
    env = get_env(str(gpu_id))

    if f"{op}" in NO_CPU_LIST:
        cmd = f'pytest -m "{op}" -vs'
    else:
        cmd = f'pytest -m "{op}" --ref cpu -vs'
    out, err, code = run_cmd_capture(cmd, cwd=ROOT.joinpath("tests"), env=env)

    combined = out + "\n" + err
    log_file = op_dir.joinpath("accuracy.log")
    with open(log_file, "w") as f:
        f.write(combined)

    result = parse_accuracy_log(combined)
    result["log"] = str(log_file.relative_to(OUTPUT_DIR))
    result["exit_code"] = code

    return result


def parse_perf_log(op_dir):
    record = {}

    # Parse log output first
    perf_log_file = op_dir / "performance_output.log"
    lines = []
    with perf_log_file.open("r") as f:
        lines = f.readlines()
    line_no = 0
    while line_no < len(lines):
        line = lines[line_no]
        if "deselected / 0 selected" in line:
            record = {
                "result": "NOT TESTED",
                "error": "No test case.",
            }
            return record

        if "FAILED" in line and "Operator" in line and "dtype" in line:
            # skip stack trace
            if "print" in line:
                continue
            pos1 = line.find("dtype=")
            pos2 = line.find(" ", pos1)
            dtype = line[pos1 + 6 : pos2]
            dtype = DTYPE_MAP.get(dtype, dtype)
            pos1 = line.find("<<<") + 3
            pos2 = line.find(">>>")
            err_str = line[pos1:pos2]
            while (pos2 < 0) and line_no < len(lines):
                line_no += 1
                line = lines[line_no]
                pos2 = line.find(">>>")
                err_str += line[:pos2]
            record.setdefault(dtype, {})
            record[dtype].setdefault("result", "FAILED")
            record[dtype].setdefault("error", err_str)
        line_no += 1

    # Check if there are usable records
    perf_res_file = op_dir / "performance_result.log"
    log_lines = []
    with perf_res_file.open("r") as f:
        log_lines = [
            line for line in f.read().strip().split("\n") if line.startswith("[INFO] {")
        ]

    for line in log_lines:
        item = {}
        try:
            item = json.loads(line[7:])
        except Exception:
            # Bad (corrupted) JSON or empty string
            continue

        dtype = DTYPE_MAP.get(item["dtype"], item["dtype"])
        details = {}
        total = 0.0
        count = 0
        # Iterate through shapes
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
            record[dtype] = {
                "result": "OK",
                "details": details,
                "speedup": total / count,
            }
        else:
            record[dtype] = {
                "result": "Incomplete",
                "details": details,
                "speedup": 0,
            }

    return record


def run_benchmark(op, gpu_id, op_dir):
    """Run benchmark for a specific operator on a specific GPU/DCU.

    This returns a dict as report summary.
    """
    pinfo(f"[GPU {gpu_id:2d}] Running performance benchmark for '{op}'")

    env = get_env(str(gpu_id))

    benchmark_dir = ROOT / "benchmark"
    pattern = f"result-m_{op}--level_core--record_log.log"
    for p in benchmark_dir.glob(pattern):
        try:
            p.unlink()
        except Exception:
            pass

    cmd = f'pytest -m "{op}" --level core --record log'
    out, err, code = run_cmd_capture(cmd, cwd=benchmark_dir, env=env)

    # Write raw command output
    output_file = op_dir.joinpath("performance_output.log")
    with open(output_file, "w") as f:
        f.write(out + "\n---\n" + err)

    # Search for record logs which may and may not be there
    result_file = None
    for p in benchmark_dir.glob(pattern):
        result_file = str(p)
        break

    # Not found
    if not result_file:
        return {
            "status": "NO_RESULT",
            "log": str(output_file.relative_to(OUTPUT_DIR)),
            "result": None,
            "data": [],
        }

    # Move record log to output directory
    dest = op_dir / "performance_result.log"
    shutil.move(result_file, str(dest))
    result_file = dest

    # Remove duplicate lines in the result file
    dedup(result_file)

    perf_result = parse_perf_log(op_dir)

    return {
        "status": "OK",
        "log": str(output_file.relative_to(OUTPUT_DIR)),
        "result": str(result_file.relative_to(OUTPUT_DIR)),
        "data": perf_result,
    }


def worker_proc(gpu_id, ops_list):
    for op in ops_list:
        op = op.strip()
        if not op:
            continue

        op_dir = OUTPUT_DIR.joinpath(op)
        ensure_dir(op_dir)

        acc = run_accuracy(op, gpu_id, op_dir)
        perf = run_benchmark(op, gpu_id, op_dir)

        with SUMMARY_LOCK:
            GLOBAL_RESULTS[op] = {
                "gpu": gpu_id,
                "accuracy": acc,
                "performance": perf,
            }
            write_summary()

            # TODO(Qiming): Enable this conditionally
            # write_xlsx()


def write_summary():
    json_path = OUTPUT_DIR.joinpath("summary.json")
    data = [
        {
            "operator": op,
            "accuracy": info["accuracy"],
            "performance": info["performance"],
        }
        for op, info in GLOBAL_RESULTS.items()
    ]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def write_xlsx(path):
    xlsx_path = path / "summary.xlsx"
    wb = openpyxl.Workbook()
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
                    perf["log"],
                    perf["result"],
                    perf["summary"],
                ]
            )
            first = False

    wb.save(str(xlsx_path))


def main():
    global OUTPUT_DIR

    init()
    op_catalog = get_ops()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--flaggems", required=False)
    parser.add_argument("--op-list", required=False)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # TODO(Qiming): parse backend and probe customized or not
    ops = []
    for op in op_catalog:
        if not op.get("stages", {}).get("stable", None):
            continue
        if "exposed" in op and op["exposed"] is False:
            continue
        ops.append(op["name"])

    if args.op_list:
        lines = []
        try:
            with open(args.op_list, "r") as f:
                lines = f.readlines()
        except Exception as e:
            perror(f"Failed reading the specified op list file: {e}")
            sys.exit(1)

        ops = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]

    if len(ops) == 0:
        pwarn("No operators to test. Please specify at lease one operator.")
        sys.exit(1)
    else:
        pinfo(f"Testing {len(ops)} operators ...")

    now_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = ROOT.joinpath(f"results_{now_ts}")
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    ensure_dir(OUTPUT_DIR)

    # Split the operators among GPUs
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    gpu_count = len(gpu_ids)
    tasks = {gpu_id: [] for gpu_id in gpu_ids}
    for i, op in enumerate(ops):
        tasks[gpu_ids[i % gpu_count]].append(op)

    with ThreadPoolExecutor(max_workers=gpu_count) as exe:
        futures = []
        for gpu in gpu_ids:
            if tasks[gpu]:
                futures.append(exe.submit(worker_proc, gpu, tasks[gpu]))
        for f in as_completed(futures):
            f.result()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_path = OUTPUT_DIR.joinpath("summary.json")
    data = {}
    with open(json_path, "r") as f:
        result = json.load(f)
        data["result"] = result
        data["env"] = ENV_INFO
        data["timestamp"] = timestamp

    with open(json_path, "w") as f:
        f.write(json.dumps(data))

    pinfo("Test completed.")


if __name__ == "__main__":
    main()
