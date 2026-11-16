# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
vLLM Benchmark Core Script

功能:
    构建并执行 `vllm bench serve` 命令，解析输出结果中的各项性能指标
    (如吞吐量、TTFT、TPOT 等)，并将结构化结果保存到指定目录。

前置条件:
    1. vLLM 服务端已启动并处于就绪状态。
    2. 需通过环境变量传入配置（通常由 perf.sh 通过 `source config.env` 自动注入）。
       如：VLLM_QWEN3_PORT, SERVED_MODEL_NAME, BENCH_NUM_PROMPTS, ENABLE_PROFILER 等。

使用方法:
    # 通常作为 perf.sh 的子模块被调用。
    # 若需单独调试或发起单次请求，可手动执行：
    
    python3 all_perf.py --input-len 4096 --output-len 1024 --concurrency 16

    # 仅打印生成的 vllm bench 命令，不实际执行
    python3 all_perf.py --dry-run

参数说明:
    --input-len   : 输入序列长度 (默认读取环境变量 BENCH_INPUT_LEN，或 128)
    --output-len  : 输出序列长度 (默认读取环境变量 BENCH_OUTPUT_LEN，或 128)
    --concurrency : 并发请求数 (默认读取环境变量 BENCH_CONCURRENCY，或 1)
    --dry-run     : 仅打印命令，不实际执行压测
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# =============================================================================
# Global Configuration
# =============================================================================
SERVER_HOST = os.environ.get("BENCH_HOST", "127.0.0.1")
SERVER_PORT = os.environ.get("VLLM_QWEN3_PORT", "8000")
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen36")
TOKENIZER_PATH = os.environ.get("BENCH_TOKENIZER_PATH", "/data/subingzhen/Qwen3.6-27B")

DEFAULT_INPUT_LEN = int(os.environ.get("BENCH_INPUT_LEN", "128"))
DEFAULT_OUTPUT_LEN = int(os.environ.get("BENCH_OUTPUT_LEN", "128"))
DEFAULT_CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "1"))

BENCH_WARMUPS = int(os.environ.get("BENCH_WARMUPS", "10"))
BENCH_NUM_PROMPTS = int(os.environ.get("BENCH_NUM_PROMPTS", "500"))
ENABLE_PROFILER = os.environ.get("ENABLE_PROFILER", "false").lower() == "true"

OUTPUT_DIR = Path(os.environ.get("BENCH_OUTPUT_DIR", "./unit_test_output"))

METRIC_PATTERNS = {
    "Successful requests": r"Successful requests:\s+(\d+)",
    "Output token throughput (tok/s)": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "Total token throughput (tok/s)": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "Median TTFT (ms)": r"Median TTFT \(ms\):\s+([\d.]+)",
    "Median TPOT (ms)": r"Median TPOT \(ms\):\s+([\d.]+)",
    "Median ITL (ms)": r"Median ITL \(ms\):\s+([\d.]+)",
}

# =============================================================================
# Section: Abstract Methods (Functions)
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="vLLM Benchmark Script")
    parser.add_argument("--input-len", type=int, default=DEFAULT_INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true", help="Print command only, do not execute")
    return parser.parse_args()

def build_command(input_len: int, output_len: int, concurrency: int) -> List[str]:
    """Build the vllm bench serve command."""
    cmd = [
        "vllm", "bench", "serve",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--model", MODEL_NAME,
        "--tokenizer", TOKENIZER_PATH,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--endpoint", "/v1/completions",
        "--ignore-eos",
        "--trust-remote-code",
        "--num-prompts", str(BENCH_NUM_PROMPTS),
        "--max-concurrency", str(concurrency),
        "--num-warmups", str(BENCH_WARMUPS),
        "--seed", "0",
    ]

    if ENABLE_PROFILER:
        cmd.append("--profile")

    return cmd

def parse_output(output: str) -> Dict[str, Optional[float]]:
    """Parse vllm bench serve output text and extract metrics."""
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            val = match.group(1)
            metrics[key] = float(val) if "." in val else int(val)
        else:
            metrics[key] = None
    return metrics

def run_benchmark(input_len: int, output_len: int, concurrency: int) -> Optional[Dict[str, float]]:
    """Execute the benchmark and parse the results."""
    cmd = build_command(input_len, output_len, concurrency)
    print(f"\n[CMD] {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metrics = parse_output(result.stdout)
        print("  [INFO] Benchmark completed. Core metrics:")
        for key, val in metrics.items():
            if val is not None:
                print(f"    {key}: {val}")
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Execution failed.\n  Stderr: {e.stderr[:2000]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}", file=sys.stderr)
        return None

def save_results(metrics: Dict[str, float], args: argparse.Namespace) -> None:
    """Save the parsed metrics to a timestamped text file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"perf_results_in{args.input_len}_out{args.output_len}_c{args.concurrency}_{timestamp}.txt"
    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Config: Input={args.input_len}, Output={args.output_len}, Concurrency={args.concurrency}\n")
        f.write(f"Prompts={BENCH_NUM_PROMPTS}, Warmups={BENCH_WARMUPS}\n")
        for key, val in metrics.items():
            if val is not None:
                f.write(f"{key}: {val}\n")

    print(f"\n[INFO] Results saved to: {filepath}")

# =============================================================================
# Section: Main Flow Control
# =============================================================================
def main() -> None:
    args = parse_args()

    print(f"[INFO] Test Config: Input={args.input_len}, Output={args.output_len}, Concurrency={args.concurrency}")
    print(f"[INFO] Bench Config: Total Prompts={BENCH_NUM_PROMPTS}, Warmups={BENCH_WARMUPS}")
    print(f"[INFO] Profiler Enabled: {ENABLE_PROFILER}")

    if args.dry_run:
        cmd = build_command(args.input_len, args.output_len, args.concurrency)
        print(f"[DRY RUN] Generated Command:\n{' '.join(cmd)}")
        return

    # Execute benchmark
    metrics = run_benchmark(args.input_len, args.output_len, args.concurrency)
    if metrics is None:
        sys.exit(1)

    # Save results
    save_results(metrics, args)

if __name__ == "__main__":
    main()
