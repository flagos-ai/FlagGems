#!/usr/bin/env python3
"""
收集各个操作的性能基准测试结果
"""
import subprocess
import sys
import os
from pathlib import Path

BRANCHES = {
    "codex/cosh": {
        "test_file": "benchmark/test_unary_pointwise_perf.py",
        "mark": "cosh",
        "dtypes": ["float16", "float32"],
    },
    "codex/ctc_loss": {
        "test_file": "benchmark/test_reduction_perf.py",
        "mark": "ctc_loss",
        "dtypes": ["float32"],
    },
    "codex/max_pool3d": {
        "test_file": "benchmark/test_reduction_perf.py",
        "mark": "max_pool3d",
        "dtypes": ["float16", "float32"],
    },
    "codex/grid_sample": {
        "test_file": "benchmark/test_special_perf.py",
        "mark": "grid_sample",
        "dtypes": ["float16", "float32"],
    },
    "codex/svd": {
        "test_file": "benchmark/test_special_perf.py",
        "mark": "svd",
        "dtypes": ["float32"],
    },
    "codex/avg_pool3d": {
        "test_file": "benchmark/test_reduction_perf.py",
        "mark": "avg_pool3d",
        "dtypes": ["float16", "float32"],
    },
}

def run_benchmark(branch, config):
    """运行单个操作的基准测试"""
    print(f"\n{'='*60}")
    print(f"Testing branch: {branch}")
    print(f"{'='*60}\n")
    
    # 切换到分支
    subprocess.run(["git", "checkout", branch], cwd="/home/qinhaiyan/FlagGems", check=True)
    subprocess.run(["git", "pull", "origin", branch], cwd="/home/qinhaiyan/FlagGems", check=True)
    
    # 构建 pytest 命令
    cmd = [
        "python", "-m", "pytest",
        config["test_file"],
        "-m", config["mark"],
        "-s",
        "--level", "core",
        "--mode", "kernel",
        "--metrics", "latency_base",
        "--metrics", "latency",
        "--metrics", "speedup",
        "--warmup", "50",
        "--iter", "100",
    ]
    
    # 添加 dtype 参数
    for dtype in config["dtypes"]:
        cmd.extend(["--dtypes", dtype])
    
    # 运行测试
    try:
        result = subprocess.run(cmd, cwd="/home/qinhaiyan/FlagGems", 
                              capture_output=True, text=True, timeout=300)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"Timeout after 300 seconds for {branch}"
    except Exception as e:
        return f"Error running benchmark for {branch}: {str(e)}"

def main():
    results_file = Path("/home/qinhaiyan/FlagGems/perf_results_summary.txt")
    
    with open(results_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("FlagGems Performance Benchmark Results\n")
        f.write("="*60 + "\n")
        f.write(f"Generated on: {subprocess.check_output(['date']).decode().strip()}\n\n")
        
        for branch, config in BRANCHES.items():
            output = run_benchmark(branch, config)
            f.write(f"\n{'='*60}\n")
            f.write(f"Branch: {branch}\n")
            f.write(f"{'='*60}\n\n")
            f.write(output)
            f.write("\n\n")
            print(f"Completed {branch}")
    
    print(f"\nAll benchmarks completed. Results saved to: {results_file}")

if __name__ == "__main__":
    main()
