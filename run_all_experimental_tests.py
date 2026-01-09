#!/usr/bin/env python3
"""
批量运行所有 experimental_ops 测试，收集结果到 JSON 文件

支持三种测试模式:
1. GPU 正确性测试 (默认)
2. CPU 正确性测试 (--ref cpu)
3. 性能测试 (benchmark)

用法:
    python run_all_experimental_tests.py                    # 运行所有测试
    python run_all_experimental_tests.py --mode gpu         # 仅 GPU 正确性
    python run_all_experimental_tests.py --mode cpu         # 仅 CPU 正确性
    python run_all_experimental_tests.py --mode benchmark   # 仅性能测试
    python run_all_experimental_tests.py --mode all         # 全部测试
    python run_all_experimental_tests.py --file abs_test.py # 测试单个文件
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_test_file(test_file, mode="gpu", timeout=300):
    """
    运行单个测试文件，返回结果
    
    Args:
        test_file: 测试文件路径
        mode: 测试模式 - "gpu", "cpu", "benchmark"
        timeout: 超时时间（秒）
    """
    result = {
        "file": test_file.name,
        "mode": mode,
        "status": "unknown",
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "duration": 0,
        "error_messages": [],
    }
    
    start_time = time.time()
    
    # 构建 pytest 命令
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "-q"]
    
    if mode == "cpu":
        # CPU 正确性测试，使用 --ref cpu
        cmd.extend(["--ref", "cpu"])
        # 排除 benchmark/performance 测试
        cmd.extend(["-k", "not benchmark and not performance and not perf"])
    elif mode == "benchmark":
        # 仅运行 benchmark/performance 测试
        cmd.extend(["-k", "benchmark or performance or perf"])
    else:  # gpu
        # GPU 正确性测试，排除 benchmark 测试
        cmd.extend(["-k", "not benchmark and not performance and not perf"])
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(test_file.parent.parent.parent),  # FlagGems root
        )
        
        result["duration"] = round(time.time() - start_time, 2)
        result["returncode"] = proc.returncode
        
        # 解析输出
        output = proc.stdout + proc.stderr
        
        # 查找测试统计
        passed_match = re.search(r'(\d+) passed', output)
        if passed_match:
            result["passed"] = int(passed_match.group(1))
        
        failed_match = re.search(r'(\d+) failed', output)
        if failed_match:
            result["failed"] = int(failed_match.group(1))
        
        error_match = re.search(r'(\d+) error', output)
        if error_match:
            result["errors"] = int(error_match.group(1))
        
        skipped_match = re.search(r'(\d+) skipped', output)
        if skipped_match:
            result["skipped"] = int(skipped_match.group(1))
        
        # 检查是否没有匹配的测试
        if "no tests ran" in output.lower() or (result["passed"] == 0 and result["failed"] == 0 and result["errors"] == 0):
            result["status"] = "no_tests"
            result["skipped"] = 1
        elif proc.returncode == 0:
            result["status"] = "passed"
        elif result["failed"] > 0 or result["errors"] > 0:
            result["status"] = "failed"
            # 提取错误信息
            error_lines = []
            for line in output.split('\n'):
                if 'FAILED' in line or 'ERROR' in line or 'AssertionError' in line:
                    error_lines.append(line.strip())
            result["error_messages"] = error_lines[:10]
        else:
            result["status"] = "error"
            result["error_messages"] = [output[-2000:]]
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration"] = timeout
        result["error_messages"] = [f"Test timed out after {timeout}s"]
        
    except Exception as e:
        result["status"] = "error"
        result["error_messages"] = [str(e)]
        result["duration"] = round(time.time() - start_time, 2)
    
    return result


def get_status_icon(status):
    """获取状态对应的图标"""
    icons = {
        "passed": "✅",
        "failed": "❌",
        "error": "💥",
        "timeout": "⏱️",
        "no_tests": "⚪",
    }
    return icons.get(status, "❓")


def run_test_suite(test_files, mode, results_dict, timeout=300):
    """运行一组测试"""
    mode_names = {
        "gpu": "GPU 正确性测试",
        "cpu": "CPU 正确性测试 (--ref cpu)",
        "benchmark": "性能/加速比测试",
    }
    
    print(f"\n{'='*70}")
    print(f"🔸 {mode_names[mode]} ({len(test_files)} 个文件)")
    print(f"{'='*70}")
    
    suite_results = {
        "mode": mode,
        "total_files": len(test_files),
        "summary": {"passed": 0, "failed": 0, "error": 0, "timeout": 0, "no_tests": 0},
        "tests": [],
    }
    
    for i, test_file in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {test_file.name}...", end=" ", flush=True)
        
        result = run_test_file(test_file, mode=mode, timeout=timeout)
        suite_results["tests"].append(result)
        
        # 更新统计
        status = result["status"]
        if status in suite_results["summary"]:
            suite_results["summary"][status] += 1
        else:
            suite_results["summary"]["error"] += 1
        
        # 打印结果
        icon = get_status_icon(status)
        if status == "passed":
            print(f"{icon} PASSED ({result['passed']} tests, {result['duration']}s)")
        elif status == "failed":
            print(f"{icon} FAILED ({result['failed']} failed, {result['passed']} passed)")
        elif status == "no_tests":
            print(f"{icon} NO TESTS (跳过)")
        elif status == "timeout":
            print(f"{icon} TIMEOUT")
        else:
            print(f"{icon} ERROR ({result['duration']}s)")
    
    results_dict[mode] = suite_results
    return suite_results


def print_summary(results):
    """打印总结报告"""
    print(f"\n{'='*70}")
    print(f"📊 测试总结报告")
    print(f"{'='*70}")
    
    total_passed = 0
    total_failed = 0
    total_error = 0
    total_timeout = 0
    
    for mode, suite in results.items():
        if mode in ["start_time", "end_time"]:
            continue
        
        summary = suite["summary"]
        total_passed += summary["passed"]
        total_failed += summary["failed"]
        total_error += summary["error"]
        total_timeout += summary["timeout"]
        
        mode_names = {
            "gpu": "GPU 正确性",
            "cpu": "CPU 正确性",
            "benchmark": "性能测试",
        }
        
        print(f"\n🔹 {mode_names.get(mode, mode)}:")
        print(f"   ✅ 通过: {summary['passed']}")
        print(f"   ❌ 失败: {summary['failed']}")
        print(f"   💥 错误: {summary['error']}")
        print(f"   ⏱️ 超时: {summary['timeout']}")
        print(f"   ⚪ 无测试: {summary.get('no_tests', 0)}")
        
        # 列出失败的文件
        failed_tests = [t for t in suite["tests"] if t["status"] in ["failed", "error"]]
        if failed_tests:
            print(f"   失败文件:")
            for t in failed_tests[:10]:  # 最多显示10个
                print(f"      - {t['file']}")
    
    print(f"\n{'='*70}")
    print(f"📈 总计:")
    print(f"   ✅ 通过: {total_passed}")
    print(f"   ❌ 失败: {total_failed}")
    print(f"   💥 错误: {total_error}")
    print(f"   ⏱️ 超时: {total_timeout}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="批量运行 experimental_ops 测试")
    parser.add_argument(
        "--mode", 
        choices=["gpu", "cpu", "benchmark", "all"],
        default="all",
        help="测试模式: gpu=GPU正确性, cpu=CPU正确性(--ref cpu), benchmark=性能测试, all=全部"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="仅测试指定文件 (如 abs_test.py)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="单个测试文件超时时间（秒），默认300"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径"
    )
    
    args = parser.parse_args()
    
    # 获取测试文件
    test_dir = Path(__file__).parent / "tests" / "experimental_ops"
    
    if args.file:
        # 测试指定文件
        test_file = test_dir / args.file
        if not test_file.exists():
            print(f"❌ 文件不存在: {test_file}")
            sys.exit(1)
        test_files = [test_file]
    else:
        # 测试所有文件
        test_files = sorted([f for f in test_dir.glob("*_test.py") if f.name != "__init__.py"])
    
    print(f"{'='*70}")
    print(f"🚀 experimental_ops 批量测试")
    print(f"{'='*70}")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 测试文件数: {len(test_files)}")
    print(f"🔧 测试模式: {args.mode}")
    print(f"⏱️ 超时设置: {args.timeout}s")
    
    results = {
        "start_time": datetime.now().isoformat(),
    }
    
    # 根据模式运行测试
    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["gpu", "cpu", "benchmark"]
    else:
        modes_to_run = [args.mode]
    
    for mode in modes_to_run:
        run_test_suite(test_files, mode, results, timeout=args.timeout)
    
    results["end_time"] = datetime.now().isoformat()
    
    # 保存结果
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(__file__).parent / "experimental_ops_test_results.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print_summary(results)
    
    print(f"\n📝 结果已保存到: {output_file}")
    print(f"📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
