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

"""Collect CI environment diagnostics for debugging GPU test failures.

This script collects system and environment information that is useful
when diagnosing CI failures (especially GPU-related). It outputs a summary
to stdout and optionally writes a JSON report.

Intended to run as a post-failure step in GPU CI workflows.

Exit codes:
  0 - always (diagnostic tool, never blocks)
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: list[str], timeout: int = 10) -> str:
    """Run a command and return its output, or error message."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() or result.stderr.strip()
    except FileNotFoundError:
        return f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(error: {e})"


def collect_python_info() -> dict:
    """Collect Python environment info."""
    info = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }

    # Check key packages
    packages = ["torch", "triton", "flag_gems", "pytest"]
    pkg_versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "installed (no version)")
        except ImportError:
            pkg_versions[pkg] = "not installed"
    info["packages"] = pkg_versions

    return info


def collect_gpu_info() -> dict:
    """Collect GPU information."""
    info = {}

    # nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        info["nvidia_smi"] = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        )
        info["nvidia_smi_full"] = run_cmd(["nvidia-smi"])
    else:
        info["nvidia_smi"] = "(nvidia-smi not found)"

    # CUDA version
    nvcc = shutil.which("nvcc")
    if nvcc:
        info["nvcc_version"] = run_cmd(["nvcc", "--version"])
    else:
        info["nvcc_version"] = "(nvcc not found)"

    # CUDA visible devices
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")

    return info


def collect_env_vars() -> dict:
    """Collect relevant environment variables."""
    relevant_prefixes = [
        "CUDA_",
        "TRITON_",
        "TORCH_",
        "FLAGGEMS_",
        "NCCL_",
        "LD_LIBRARY_PATH",
        "PATH",
    ]
    env_vars = {}
    for key, value in sorted(os.environ.items()):
        for prefix in relevant_prefixes:
            if key.startswith(prefix) or key == prefix:
                env_vars[key] = value
                break
    return env_vars


def collect_disk_info() -> dict:
    """Collect disk space information."""
    info = {}
    try:
        usage = shutil.disk_usage("/")
        info["total_gb"] = round(usage.total / (1024**3), 1)
        info["free_gb"] = round(usage.free / (1024**3), 1)
        info["used_percent"] = round((usage.used / usage.total) * 100, 1)
    except Exception as e:
        info["error"] = str(e)
    return info


def collect_memory_info() -> dict:
    """Collect system memory information."""
    info = {}
    if platform.system() == "Linux":
        meminfo = run_cmd(["free", "-h"])
        info["free_output"] = meminfo
    elif platform.system() == "Darwin":
        info["vm_stat"] = run_cmd(["vm_stat"])
    return info


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect CI environment diagnostics")
    parser.add_argument(
        "--output",
        help="Path to write JSON report",
        default="",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact summary only",
    )
    args = parser.parse_args()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": collect_python_info(),
        "gpu": collect_gpu_info(),
        "disk": collect_disk_info(),
        "memory": collect_memory_info(),
        "env_vars": collect_env_vars(),
    }

    if args.compact:
        print("=== CI Environment Diagnostics ===")
        print(f"Python: {report['python']['version']} ({sys.executable})")
        print(f"Platform: {platform.platform()}")
        pkgs = report["python"]["packages"]
        print(f"PyTorch: {pkgs.get('torch', '?')}")
        print(f"Triton: {pkgs.get('triton', '?')}")
        print(f"FlagGems: {pkgs.get('flag_gems', '?')}")
        print(f"GPU: {report['gpu'].get('nvidia_smi', '?')}")
        print(
            f"Disk: {report['disk'].get('free_gb', '?')}GB free "
            f"({report['disk'].get('used_percent', '?')}% used)"
        )
    else:
        print("=== CI Environment Diagnostics ===\n")
        print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
