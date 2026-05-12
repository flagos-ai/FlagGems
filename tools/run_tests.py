#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
from decimal import getcontext
from importlib import metadata
from multiprocessing import Process
from pathlib import Path

import distro
import git
import yaml

import flag_gems

# increase decimal precision
getcontext().prec = 18

# Global lock for writing result file and the result file
GLOBAL_RESULTS = {}
ENV_INFO = {}
HAS_TRITON = False
HAS_FLAGTREE = False
ROOT = Path(__file__).parent.parent
OUPUT_DIR = None
OP_LIST = []
DUMP_OUTPUT = False
TIMEOUT = -100
# A list of operators that can only run on GPU/DCUs
NO_CPU_LIST = []
DTYPE_MAP = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.bool": "bool",
    "torch.complex64": "cf64",
}


def pinfo(str, **args):
    print(f"\033[32m[INFO]\033[0m {str}", flush=True, **args)


def perror(str, **args):
    print(f"\033[31m[ERROR]\033[0m {str}", flush=True, **args)


def pwarn(str, **args):
    print(f"\033[93m[WARN]\033[0m {str}", flush=True, **args)


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def get_ops_from_inventory():
    catalog = []
    try:
        op_inventory = ROOT / "conf" / "operators.yaml"  # noqa: E226
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

    return


def _probe_triton():
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

    return


def _probe_flaggems():
    try:
        # This may print an error "no device detected on your machine."
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

    return


def probe_env():
    ENV_INFO["architecture"] = platform.machine()
    ENV_INFO["os_name"] = distro.id()
    ENV_INFO["os_release"] = distro.version()
    ENV_INFO["python"] = platform.python_version()

    _probe_torch()
    _probe_triton()
    _probe_flaggems()


def run_cmd(op, cmd, cwd=None, env=None, timeout=600, flavor=None):
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if DUMP_OUTPUT:
        op_dir = OUTPUT_DIR.joinpath(op)
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
    except subprocess.TimeoutExpired:
        pgid = os.getpgid(p.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            os.killpg(pgid, signal.SIGKILL)

        return p.returncode
    except Exception as e:
        perror(f"run_cmd failed: {e}")
        return -1


def parse_accuracy_data(result_file):
    raw_data = {}
    with result_file.open("r") as f:
        raw_data = json.load(f)

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
            if reason.find("Issue"):
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

    # Something is wrong ... check failed first
    if num_failed > 0:
        result["status"] = "Failed"
        for k, v in failed.items():
            failed[k] = list(v)
        result["details"]["failed"] = failed
        return result

    # There are skipped tests
    if skipped_with_issue:
        result["status"] = "Failed"
    else:
        result["status"] = "Skipped"

    for k, v in skipped.items():
        skipped[k] = list(v)
    result["details"]["skipped"] = skipped

    return result


def get_env(gpu_ids):
    env = os.environ.copy()

    vendor = ENV_INFO.get("flag_gems", {}).get("vendor", "")

    if vendor == "ascend":
        env["ASCEND_RT_VISIBLE_DEVICES"] = gpu_ids
        env["NPU_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "hygon":
        env["HIP_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "metax":
        env["MACA_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "mthreads":
        env["MUSA_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "tsingmicro":
        env["TXDA_VISIBLE_DEVICES"] = gpu_ids
        return env

    # NOTE: Iluvatar said to support CUDA_VISIBLE_DEVICES as well
    if vendor == "iluvatar":
        env["ILUVATAR_VISIBLE_DEVICES"] = gpu_ids
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "thead":
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "cambricon":
        env["MLU_VISIBLE_DEVICES"] = gpu_ids
        return env

    if vendor == "kunlunxin":
        env["XPU_VISIBLE_DEVICES"] = gpu_ids
        return env

    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    return env


def run_accuracy(gpu_id, start, index, count):
    op = OP_LIST[start + index].strip()
    n = (index + 1) * 10 // count
    prog = "█" * n + " " * (10 - n)
    nums = f"{index + 1}/{count}"
    pinfo(f"[GPU {gpu_id:2d}][{nums:>7}][{prog}] Running accuracy tests for '{op}'")

    env = get_env(str(gpu_id))

    if op in NO_CPU_LIST:
        cmd = f'pytest -m "{op}" --record json --output accuracy_{op}.json -vs'
    else:
        cmd = (
            f'pytest -m "{op}" --record json --output accuracy_{op}.json --ref cpu -vs'
        )

    accuracy_dir = ROOT.joinpath("tests")
    result_file = accuracy_dir / f"accuracy_{op}.json"
    if result_file.exists():
        result_file.unlink()

    op_dir = OUTPUT_DIR.joinpath(op)
    ensure_dir(op_dir)
    start = time.time()
    code = run_cmd(op, cmd, cwd=accuracy_dir, env=env, flavor="accuracy")
    end = time.time()

    if code == TIMEOUT:  # Timeout
        return {
            "status": "Timeout",
            "exit_code": TIMEOUT,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": end - start,
        }
    # There are rare cases where the pytest process aborts
    # with no result file generated.
    if not result_file.exists():
        return {
            "status": "Error",
            "exit_code": code,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 1,
            "duration": end - start,
            "data_file": None,
        }

    op_dir = OUTPUT_DIR.joinpath(op)
    dest = op_dir / "accuracy_result.json"
    shutil.move(result_file, str(dest))
    result_file = dest

    result = parse_accuracy_data(result_file)
    result["exit_code"] = code
    result["duration"] = end - start
    result["data_file"] = str(result_file.relative_to(OUTPUT_DIR))

    return result


def parse_perf_data(op, result_file):
    raw_data = {}
    with result_file.open("r") as f:
        raw_data = json.load(f)

    data = raw_data.get(op, {})
    if not data:
        return {
            "status": "NotFound",
        }

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


def run_benchmark(gpu_id, start, index, count):
    """Run benchmark for a specific operator on a specific GPU/DCU.

    This returns a dict as report summary.
    """
    op = OP_LIST[start + index].strip()
    n = (index + 1) * 10 // count
    prog = "█" * n + " " * (10 - n)
    if (index + 1) == count:
        prog = f"\033[32m{prog}\033[0m"
    nums = f"{index + 1}/{count}"
    pinfo(f"[GPU {gpu_id:2d}][{nums:>7}][{prog}] Running perf benchmark for '{op}'")

    env = get_env(str(gpu_id))

    benchmark_dir = ROOT / "benchmark"
    result_file = benchmark_dir / f"benchmark_{op}.json"
    if result_file.exists():
        result_file.unlink()

    op_dir = OUTPUT_DIR.joinpath(op)
    ensure_dir(op_dir)

    start = time.time()
    cmd = f'pytest -m "{op}" --level core --record json --output benchmark_{op}.json'
    code = run_cmd(op, cmd, cwd=benchmark_dir, env=env, flavor="performance")
    end = time.time()

    # Not found
    if not result_file.exists():
        return {
            "status": "NotFound",
            "exit_code": code,
            "data": [],
        }

    # Move record log to output directory
    dest = op_dir / "performance_result.json"
    shutil.move(result_file, str(dest))
    result_file = dest

    record = {
        "duration": end - start,
        "exit_code": code,
        "data_file": str(result_file.relative_to(OUTPUT_DIR)),
        "data": {},
    }
    record.update(parse_perf_data(op, result_file))

    return record


def worker_proc(gpu_id, start, count):
    worker_result = {}
    for i in range(count):
        op = OP_LIST[start + i].strip()
        if not op:
            continue

        op_dir = OUTPUT_DIR.joinpath(op)
        ensure_dir(op_dir)

        acc = run_accuracy(gpu_id, start, i, count)
        perf = run_benchmark(gpu_id, start, i, count)

        customized_ops = [
            op[0] for op in flag_gems.runtime.backend.get_customized_ops()
        ]
        result = {
            "customized": op in customized_ops,
            "accuracy": acc,
            "performance": perf,
        }
        worker_result.setdefault(op, result)

        json_path = OUTPUT_DIR.joinpath(f"summary{gpu_id}.json")
        with open(json_path, "w") as f:
            json.dump(worker_result, f, indent=2)

    return


def get_ops_to_test(ops_file, ops_list, stages):
    # Build list of operators which do NOT support CPU mode
    op_catalog = get_ops_from_inventory()
    for op in op_catalog:
        labels = op.get("labels", [])
        if "NoCPU" in labels:
            NO_CPU_LIST.append(op["id"])

    # This is the highest priority
    if ops_list:
        ops = []
        for op in ops_list.split(","):
            # Leading underscores are not valid pytest marks
            ops.append(op.strip().lstrip("_"))

        return ops

    # Parse the op list file if specified
    if ops_file:
        lines = []
        try:
            with open(ops_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            perror(f"Failed reading the specified op list file: {e}")
            return []

        ops = []
        for ln in lines:
            ln = ln.strip()
            # comment line
            if ln.startswith("#"):
                continue
            # Remove leading underscore to make valid pytest mark
            ops.append(ln.lstrip("_"))

        return ops

    # Now fall-back to inventory
    effective_stages = []
    for s in stages.split(","):
        stage = s.strip()
        if stage not in ["alpha", "beta", "stable", "all", "removed"]:
            pwarn(f"ignoring unsupported stage name '{s}'...")
            continue
        # Stop checking if 'all' specified
        if stage == "all":
            effective_stages = ["alpha", "beta", "stable"]
            break
        effective_stages.append(stage)

    # Fall back to 'stable' if no effective filter specified
    if not effective_stages:
        effective_stages = ["stable"]

    ops = []
    for op in op_catalog:
        stages = op.get("stages", [])
        if len(stages) == 0:
            # won't happen
            continue
        stage = next(iter(stages[-1].keys()), None)
        if stage not in effective_stages:
            continue
        ops.append(op["id"])

    return ops


def main():
    global OUTPUT_DIR
    global OP_LIST
    global DUMP_OUTPUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--op-list-file", required=False)
    parser.add_argument("--ops", required=False)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stages", required=False, default="stable")
    parser.add_argument(
        "--dump-output",
        action="store_true",
        default=False,
        help="Dump stdout/stderr of each test to log files",
    )
    args = parser.parse_args()

    DUMP_OUTPUT = args.dump_output

    # Probe environment setttings
    probe_env()

    ops = get_ops_to_test(args.op_list_file, args.ops, args.stages)
    op_count = len(ops)
    if op_count == 0:
        pwarn("No operators to test. Please specify at lease one operator.")
        sys.exit(1)
    else:
        pinfo(f"Testing {op_count} operators ...")

    # Set global variable for convenience
    OP_LIST = ops

    OUTPUT_DIR = ROOT.joinpath("results")
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    ensure_dir(OUTPUT_DIR)

    # Split the operators among GPUs
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    gpu_count = len(gpu_ids)
    if gpu_count == 1:
        worker_proc(gpu_ids[0], 0, op_count)
    else:
        processes = []
        m, n = divmod(op_count, gpu_count)
        start = 0
        for i, gpu in enumerate(gpu_ids):
            if i < n:
                count = m + 1
            else:
                count = m
            p = Process(target=worker_proc, args=(gpu, start, count))
            p.start()
            processes.append(p)
            start += count

        for p in processes:
            p.join()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    op_data = {}
    for gpu_id in gpu_ids:
        gpu_file = OUTPUT_DIR.joinpath(f"summary{gpu_id}.json")
        if not gpu_file.exists():
            perror(f"GPU {gpu_id} failed to produce a summary, recovery needed.")
            continue
        with gpu_file.open("r") as f:
            result = json.load(f)
            op_data.update(result)

    final_data = {
        "timestamp": timestamp,
        "env": ENV_INFO,
        "result": op_data,
    }

    json_path = OUTPUT_DIR.joinpath("summary.json")
    with json_path.open("w") as f:
        json.dump(final_data, f, indent=2)

    pinfo("Test completed.")


if __name__ == "__main__":
    main()
