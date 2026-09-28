#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci_run_tests.py - CI Environment Batch Test Runner for FlagGems Operators
"""

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import atexit
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
import distro
from multiprocessing import Process

try:
    import flag_gems
except ImportError:
    print("[ERROR] flag_gems not installed. Please run: pip install -e .")
    sys.exit(1)

# ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[93m"
CYAN = "\033[36m"
DIM = "\033[2m"
NC = "\033[0m"

ROOT = Path(__file__).parent.parent.parent
TESTS_DIR = ROOT / "tests"

# ============================================================================
# SKIP LIST - Default skip lists (can be overridden by --skip-file)
# ============================================================================

DEFAULT_SKIP_OPERATORS: Set[str] = {
    # Known broken operators that cause CUDA illegal memory access
    # "scaled_dot_product_attention",
}

DEFAULT_SKIP_TEST_PATTERNS: Set[str] = {
    # Skip specific test functions that cause issues
    #"test_scaled_dot_product_attention_legacy_backward",
    #"test_scaled_dot_product_attention_square_qk_even_mn",
    #"test_scaled_dot_product_attention_nonsquare_qk",
    #"test_scaled_dot_product_attention_legacy",
    #"test_scaled_dot_product_flash_attention",
}

DEFAULT_SKIP_TEST_FILES: Set[str] = {
    #"test_scaled_dot_product_attention.py",
}

# Runtime skip lists (will be updated from file)
SKIP_OPERATORS: Set[str] = set(DEFAULT_SKIP_OPERATORS)
SKIP_TEST_PATTERNS: Set[str] = set(DEFAULT_SKIP_TEST_PATTERNS)
SKIP_TEST_FILES: Set[str] = set(DEFAULT_SKIP_TEST_FILES)

def load_skip_file(file_path: Path) -> Dict[str, Set[str]]:
    """
    Load skip lists from a YAML or text file.

    Supported formats:

    YAML format:
        skip_operators:
          - scaled_dot_product_attention
          - scaled_softmax

        skip_test_patterns:
          - test_scaled_dot_product_attention_legacy_backward
          - test_scaled_dot_product_attention_square_qk_even_mn

        skip_test_files:
          - test_scaled_dot_product_attention.py
          - test_scaled_softmax.py

    Simple text format (one item per line, # for comments):
        # Skip operators
        scaled_dot_product_attention
        scaled_softmax

        # Skip test patterns
        test_scaled_dot_product_attention_legacy_backward

        # Skip test files
        test_scaled_dot_product_attention.py

    Args:
        file_path: Path to the skip list file

    Returns:
        Dictionary with keys: 'skip_operators', 'skip_test_patterns', 'skip_test_files'
    """
    result = {
        'skip_operators': set(),
        'skip_test_patterns': set(),
        'skip_test_files': set(),
    }

    if not file_path.exists():
        print(f"[WARN] Skip file not found: {file_path}")
        return result

    try:
        content = file_path.read_text()

        # Try to parse as YAML first
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                # YAML format
                if 'skip_operators' in data:
                    result['skip_operators'] = set(data['skip_operators'] or [])
                if 'skip_test_patterns' in data:
                    result['skip_test_patterns'] = set(data['skip_test_patterns'] or [])
                if 'skip_test_files' in data:
                    result['skip_test_files'] = set(data['skip_test_files'] or [])
                return result
        except yaml.YAMLError:
            pass

        # Fallback: parse as text file
        current_section = None
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                # Check for section headers in comments
                if 'skip_operators' in line.lower():
                    current_section = 'skip_operators'
                elif 'skip_test_patterns' in line.lower():
                    current_section = 'skip_test_patterns'
                elif 'skip_test_files' in line.lower():
                    current_section = 'skip_test_files'
                continue

            if current_section:
                result[current_section].add(line)
            else:
                # Auto-detect based on content
                if line.endswith('.py'):
                    result['skip_test_files'].add(line)
                elif line.startswith('test_'):
                    result['skip_test_patterns'].add(line)
                else:
                    result['skip_operators'].add(line)

        return result

    except Exception as e:
        print(f"[ERROR] Failed to load skip file {file_path}: {e}")
        return result


def apply_skip_file(file_path: Path):
    """
    Load and apply skip lists from file.

    Args:
        file_path: Path to the skip list file
    """
    global SKIP_OPERATORS, SKIP_TEST_PATTERNS, SKIP_TEST_FILES

    if file_path is None:
        return

    data = load_skip_file(file_path)

    if data['skip_operators']:
        SKIP_OPERATORS = data['skip_operators']
        print(f"[INFO] Loaded {len(SKIP_OPERATORS)} skip_operators from {file_path}")

    if data['skip_test_patterns']:
        SKIP_TEST_PATTERNS = data['skip_test_patterns']
        print(f"[INFO] Loaded {len(SKIP_TEST_PATTERNS)} skip_test_patterns from {file_path}")

    if data['skip_test_files']:
        SKIP_TEST_FILES = data['skip_test_files']
        print(f"[INFO] Loaded {len(SKIP_TEST_FILES)} skip_test_files from {file_path}")


def info(msg, **kwargs):
    print(f"{GREEN}[INFO]{NC} {msg}", flush=True, **kwargs)


def error(msg, **kwargs):
    print(f"{RED}[ERROR]{NC} {msg}", flush=True, **kwargs)


def warn(msg, **kwargs):
    print(f"{YELLOW}[WARN]{NC} {msg}", flush=True, **kwargs)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    p.chmod(0o755)


def get_stable_ops() -> List[str]:
    catalog_file = ROOT / "conf" / "operators.yaml"
    if not catalog_file.exists():
        error(f"Operator catalog not found: {catalog_file}")
        return []

    try:
        with open(catalog_file, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        error(f"Failed to parse {catalog_file}: {e}")
        return []

    ops = data.get("ops", [])
    stable_ops = []

    for op in ops:
        stages = op.get("stages", [])
        if not stages:
            continue

        latest_stage = stages[-1]
        stage_name = next(iter(latest_stage.keys()), None)

        if stage_name == "stable":
            op_id = op.get("id")
            if op_id:
                stable_ops.append(op_id)

    # Filter out skipped operators
    filtered_ops = [op for op in stable_ops if op not in SKIP_OPERATORS]

    if len(filtered_ops) < len(stable_ops):
        skipped_count = len(stable_ops) - len(filtered_ops)
        warn(f"Skipping {skipped_count} operators from SKIP_OPERATORS list")

    info(f"Found {len(filtered_ops)} stable operators (after filtering)")
    return filtered_ops


def get_ops_to_test(args) -> List[str]:
    if args.ops:
        ops = [op.strip().lstrip("_") for op in args.ops.split(",") if op.strip()]
        filtered_ops = [op for op in ops if op not in SKIP_OPERATORS]
        if len(filtered_ops) < len(ops):
            skipped_count = len(ops) - len(filtered_ops)
            warn(f"Skipping {skipped_count} operators from SKIP_OPERATORS list")
        info(f"Using explicitly specified operators: {len(filtered_ops)}")
        return filtered_ops

    if args.op_list_file:
        try:
            with open(args.op_list_file, "r") as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            ops = [op.lstrip("_") for op in lines]
            filtered_ops = [op for op in ops if op not in SKIP_OPERATORS]
            if len(filtered_ops) < len(ops):
                skipped_count = len(ops) - len(filtered_ops)
                warn(f"Skipping {skipped_count} operators from SKIP_OPERATORS list")
            info(f"Read {len(filtered_ops)} operators from {args.op_list_file}")
            return filtered_ops
        except Exception as e:
            error(f"Failed to read {args.op_list_file}: {e}")
            return []

    return get_stable_ops()


def build_mark_expression(ops: List[str]) -> str:
    quoted_ops = [f'"{op}"' for op in ops]
    return " or ".join(quoted_ops)


def build_pytest_command(ops: List[str], args, output_name: str = "accuracy_all.json") -> str:
    """
    Build the pytest command line with skip filters
    """
    mark_expr = build_mark_expression(ops)

    # Build exclude patterns for test functions using -k
    # -k supports expressions like "not test_foo and not test_bar"
    if SKIP_TEST_PATTERNS:
        exclude_parts = []
        for pattern in SKIP_TEST_PATTERNS:
            exclude_parts.append(f'not "{pattern}"')
        exclude_expr = " and ".join(exclude_parts)
        # Combine with mark expression using -k
        # Note: -k filters test names, -m filters markers
        cmd = f'pytest -m "{mark_expr}" -k "{exclude_expr}"'
    else:
        cmd = f'pytest -m "{mark_expr}"'

    # Skip test files using --ignore
    for test_file in SKIP_TEST_FILES:
        cmd += f' --ignore={TESTS_DIR / test_file}'

    # Add other options
    cmd += f' --record json --output {output_name}'

    if args.quick:
        cmd += " --quick"

    if args.first_parameter_only:
        cmd += " --first-parameter-only"

    if args.verbose:
        cmd += " -v"

    cmd += " --continue-on-collection-errors"
    cmd += " --tb=short"
    cmd += " -ra"
    cmd += " --maxfail=1000"

    return cmd


def run_pytest_with_timeout(cmd: str, env: dict, timeout: int = 72000) -> tuple:
    info(f"Running pytest...")
    info(f"Working directory: {TESTS_DIR}")

    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(TESTS_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            start_new_session=True,
            bufsize=1,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            info(f"Test timed out after {timeout} seconds, terminating...")
            pgid = os.getpgid(process.pid)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = "", "Process killed due to timeout"

            return_code = -100

    except Exception as e:
        info(f"Error running pytest: {e}")
        return -1, "", str(e), time.time() - start_time

    duration = time.time() - start_time
    info(f"Pytest completed with return code: {return_code}, duration: {duration:.2f}s")
    return return_code, stdout, stderr, duration


def find_result_file(output_name: str = "accuracy_all.json") -> Optional[Path]:
    target = TESTS_DIR / output_name
    if target.exists():
        return target

    # Fallback for compatibility
    fallback = TESTS_DIR / "accuracy_result.json"
    if fallback.exists():
        return fallback

    return None


def parse_results(result_file: Path) -> Dict:
    if not result_file.exists():
        return {}

    try:
        with open(result_file, "r") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        error(f"Failed to parse {result_file}: {e}")
        return {}

    grouped_results = {}

    for test_case, data in raw_data.items():
        # Skip tests that match skip patterns (already filtered by pytest, but keep as safety)
        # Check if test file is in skip list
        if "::" in test_case:
            file_part = test_case.split("::")[0]
            if "/" in file_part:
                test_file = file_part.split("/")[-1]
            else:
                test_file = file_part
            if test_file in SKIP_TEST_FILES:
                continue

        op_name = None

        if "::" in test_case:
            func_part = test_case.split("::")[-1]
            if func_part.startswith("test_"):
                op_name = func_part[5:]
            else:
                op_name = func_part

            if "[" in op_name:
                op_name = op_name.split("[")[0]
        else:
            op_name = test_case
            if "[" in op_name:
                op_name = op_name.split("[")[0]

        if op_name:
            if op_name not in grouped_results:
                grouped_results[op_name] = {}
            grouped_results[op_name][test_case] = data

    return grouped_results


def generate_summary(grouped_results: Dict, ops: List[str], duration: float) -> Dict:
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "duration_seconds": duration,
        "operators": {},
        "totals": {
            "total_ops": len(ops),
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "not_found": 0,
        },
        "skip_lists": {
            "skip_operators": sorted(SKIP_OPERATORS),
            "skip_test_patterns": sorted(SKIP_TEST_PATTERNS),
            "skip_test_files": sorted(SKIP_TEST_FILES),
        }
    }

    for op in ops:
        op_data = grouped_results.get(op, {})
        op_stats = {
            "total": len(op_data),
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "passed_cases": [],
            "failed_cases": [],
            "skipped_cases": [],
            "error_cases": [],
        }

        for case_id, case_data in op_data.items():
            result = case_data.get("result", "unknown")

            test_name = case_id
            if "::" in test_name:
                test_name = test_name.split("::")[-1]

            if result == "passed":
                op_stats["passed"] += 1
                op_stats["passed_cases"].append(test_name)
            elif result == "failed":
                op_stats["failed"] += 1
                op_stats["failed_cases"].append(test_name)
            elif result == "skipped":
                op_stats["skipped"] += 1
                op_stats["skipped_cases"].append(test_name)
            else:
                op_stats["errors"] += 1
                op_stats["error_cases"].append(test_name)

            summary["totals"]["total_cases"] += 1
            if result == "passed":
                summary["totals"]["passed"] += 1
            elif result == "failed":
                summary["totals"]["failed"] += 1
            elif result == "skipped":
                summary["totals"]["skipped"] += 1
            else:
                summary["totals"]["errors"] += 1

        if op_stats["total"] == 0:
            op_stats["status"] = "NotFound"
            summary["totals"]["not_found"] += 1
        elif op_stats["failed"] > 0 or op_stats["errors"] > 0:
            op_stats["status"] = "Failed"
        else:
            op_stats["status"] = "Passed"

        # Clean up empty lists
        if not op_stats["passed_cases"]:
            del op_stats["passed_cases"]
        if not op_stats["failed_cases"]:
            del op_stats["failed_cases"]
        if not op_stats["skipped_cases"]:
            del op_stats["skipped_cases"]
        if not op_stats["error_cases"]:
            del op_stats["error_cases"]

        summary["operators"][op] = op_stats

    return summary


def print_summary(summary: Dict):
    info("=" * 70)
    info("Test Summary")
    info("=" * 70)
    info(f"Total duration: {summary['duration_seconds']:.2f}s")
    info(f"Operators tested: {summary['totals']['total_ops']}")
    info(f"Total test cases: {summary['totals']['total_cases']}")
    info(f"  {GREEN}Passed: {summary['totals']['passed']}{NC}")
    info(f"  {RED}Failed: {summary['totals']['failed']}{NC}")
    info(f"  {YELLOW}Skipped: {summary['totals']['skipped']}{NC}")
    info(f"  Errors: {summary['totals']['errors']}")
    info(f"  Not found: {summary['totals']['not_found']}")
    info("=" * 70)

    # Show skip list summary
    if summary.get('skip_lists'):
        skip_info = summary['skip_lists']
        if skip_info.get('skip_operators'):
            info(f"Skipped operators: {len(skip_info['skip_operators'])}")
        if skip_info.get('skip_test_patterns'):
            info(f"Skipped test patterns: {len(skip_info['skip_test_patterns'])}")
        if skip_info.get('skip_test_files'):
            info(f"Skipped test files: {len(skip_info['skip_test_files'])}")

    failed_ops = [op for op, stats in summary['operators'].items() 
                 if stats['status'] == 'Failed']
    if failed_ops:
        warn(f"\nFailed operators ({len(failed_ops)}):")
        for op in failed_ops[:20]:
            stats = summary['operators'][op]
            warn(f"  - {op}: {stats['failed']} failed, {stats['errors']} errors")
            if stats.get('failed_cases'):
                warn(f"    Failed cases:")
                for case in stats['failed_cases'][:5]:
                    warn(f"      - {case}")
                if len(stats['failed_cases']) > 5:
                    warn(f"      ... and {len(stats['failed_cases']) - 5} more")
        if len(failed_ops) > 20:
            warn(f"  ... and {len(failed_ops) - 20} more")

    not_found_ops = [op for op, stats in summary['operators'].items() 
                    if stats['status'] == 'NotFound']
    if not_found_ops:
        warn(f"\nOperators with no tests ({len(not_found_ops)}):")
        for op in not_found_ops[:10]:
            warn(f"  - {op}")
        if len(not_found_ops) > 10:
            warn(f"  ... and {len(not_found_ops) - 10} more")

def collect_all_marks(args, env: dict, output_dir: Path) -> Dict:
    """
    Collect all test marks using --collect-marks.
    Returns a mapping from test_case to list of marks.
    The marks file is saved in the output directory for later analysis.
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Use absolute path for marks file
    marks_file = output_dir.absolute() / "collected_marks.yaml"

    # Remove existing marks file if present
    if marks_file.exists():
        try:
            marks_file.unlink()
        except Exception:
            pass

    # Build collect command with absolute path
    collect_cmd = f'pytest --collect-marks={marks_file} --continue-on-collection-errors'

    if args.quick:
        collect_cmd += " --quick"

    if args.first_parameter_only:
        collect_cmd += " --first-parameter-only"

    # Skip test files - use absolute path
    for test_file in SKIP_TEST_FILES:
        collect_cmd += f' --ignore={TESTS_DIR / test_file}'

    info(f"Collecting test marks...")
    info(f"Command: {collect_cmd}")
    info(f"Marks file: {marks_file}")

    # Always try to collect marks, regardless of exit code
    collect_result = None
    try:
        collect_result = subprocess.run(
            collect_cmd,
            cwd=str(TESTS_DIR),
            env=env,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Log the exit code for debugging
        info(f"Mark collection exit code: {collect_result.returncode}")

        # Log the complete stderr for debugging (but truncate if too long)
        if collect_result.stderr:
            stderr_lines = collect_result.stderr.split('\n')
            # Filter out expected warnings
            expected_warnings = ["no tests ran", "TEST_RESULTS has 0 entries", "No test results collected"]
            error_lines = []
            for line in stderr_lines:
                if line.strip() and not any(w in line for w in expected_warnings):
                    error_lines.append(line)

            if error_lines:
                warn(f"Collection stderr: {' '.join(error_lines[:5])}...")

        # Exit codes 0, 3, 5 are acceptable (3 = pytest internal error, 5 = no tests)
        if collect_result.returncode not in [0, 3, 5]:
            warn(f"Mark collection returned unexpected exit code: {collect_result.returncode}")

    except subprocess.TimeoutExpired:
        warn(f"Mark collection timed out after 120 seconds")
        return {}
    except Exception as e:
        warn(f"Failed to collect marks: {e}")
        return {}

    # Load marks mapping - check if file exists even if exit code was non-zero
    test_marks_map = {}
    info(f"Checking if marks file exists: {marks_file.exists()}, size: {marks_file.stat().st_size if marks_file.exists() else 0}")

    if marks_file.exists() and marks_file.stat().st_size > 0:
        try:
            with open(marks_file, "r") as f:
                content = f.read()
                if not content.strip():
                    warn(f"Marks file is empty: {marks_file}")
                else:
                    collected_data = yaml.safe_load(content)
                    if not collected_data:
                        warn(f"No data in marks file: {marks_file}")
                    else:
                        for item in collected_data:
                            test_file = item.get("file", "")
                            test_func = item.get("function", "")
                            test_class = item.get("class", "")
                            marks = item.get("marks", [])

                            # Build key matching JSON format
                            if test_class:
                                key = f"{test_file}::{test_class}::{test_func}"
                            else:
                                key = f"{test_file}::{test_func}"

                            # Add alternative keys
                            alt_key = None
                            if test_file.startswith("tests/"):
                                alt_file = test_file[6:]
                                if test_class:
                                    alt_key = f"{alt_file}::{test_class}::{test_func}"
                                else:
                                    alt_key = f"{alt_file}::{test_func}"

                            if key not in test_marks_map or not test_marks_map[key]:
                                test_marks_map[key] = marks
                            if alt_key and (alt_key not in test_marks_map or not test_marks_map[alt_key]):
                                test_marks_map[alt_key] = marks

                            if test_func not in test_marks_map or not test_marks_map[test_func]:
                                test_marks_map[test_func] = marks

                        info(f"Loaded marks for {len(test_marks_map)} lookup entries from {marks_file}")
                        info(f"Marks file saved for analysis: {marks_file}")
        except yaml.YAMLError as e:
            error(f"Failed to parse YAML from {marks_file}: {e}")
            # Try to show first few lines for debugging
            try:
                with open(marks_file, "r") as f:
                    lines = f.readlines()[:5]
                    warn(f"First lines of marks file: {''.join(lines)}")
            except Exception:
                pass
        except Exception as e:
            error(f"Failed to parse marks file {marks_file}: {e}")
    else:
        # File doesn't exist or is empty
        if collect_result and collect_result.stderr:
            # Check if there's an error message about the file path
            stderr = collect_result.stderr
            if "No such file or directory" in stderr or "cannot open" in stderr:
                warn(f"Collection failed: {stderr[:200]}")
            else:
                warn(f"Marks file not found: {marks_file}")
        else:
            warn(f"Marks file not found: {marks_file}")

        # Additional debug: check if any marks file was created in the tests directory
        possible_files = list(TESTS_DIR.glob("*marks*.yaml"))
        if possible_files:
            info(f"Found marks files in tests directory: {[f.name for f in possible_files]}")
            # Move the file to the output directory
            import shutil
            for f in possible_files:
                try:
                    dest = output_dir / f.name
                    shutil.move(str(f), str(dest))
                    info(f"Moved {f.name} to {dest}")
                    # Try to load it
                    test_marks_map = load_marks_file(dest)
                    if test_marks_map:
                        break
                except Exception as e:
                    warn(f"Failed to move {f}: {e}")

    if not test_marks_map:
        warn("No marks loaded. Will use function name extraction as fallback.")

    return test_marks_map


def parse_results_with_marks(result_file: Path, test_marks_map: Dict) -> Dict:
    """
    Parse test results using collected marks information.
    Uses pytest marks to identify which operator each test belongs to.
    """
    if not result_file.exists():
        return {}

    try:
        with open(result_file, "r") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        error(f"Failed to parse {result_file}: {e}")
        return {}

    grouped_results = {}
    matched_count = 0
    unmatched_count = 0
    total_count = 0

    # Debug: print sample keys from marks map
    if test_marks_map:
        sample_keys = list(test_marks_map.keys())[:3]
        info(f"Sample marks keys: {sample_keys}")

    for test_case, data in raw_data.items():
        total_count += 1

        # Skip tests that match skip patterns
        if "::" in test_case:
            file_part = test_case.split("::")[0]
            if "/" in file_part:
                test_file = file_part.split("/")[-1]
            else:
                test_file = file_part
            if test_file in SKIP_TEST_FILES:
                continue

        # Build lookup key: keep "tests/" prefix, remove parameter part
        lookup_key = test_case
        # Remove parameter part [..]
        if "[" in lookup_key:
            lookup_key = lookup_key.split("[")[0]

        # Debug: print first few lookups
        if total_count <= 3:
            info(f"Lookup key: {lookup_key}")

        # Try multiple lookup strategies
        marks = []

        # Strategy 1: Exact match with lookup_key (keep tests/ prefix)
        marks = test_marks_map.get(lookup_key, [])

        # Strategy 2: Remove "tests/" prefix
        if not marks and lookup_key.startswith("tests/"):
            alt_key = lookup_key[6:]  # Remove "tests/"
            marks = test_marks_map.get(alt_key, [])

        # Strategy 3: Try to find by function name only
        if not marks and "::" in lookup_key:
            func_name = lookup_key.split("::")[-1]
            # Search for key ending with ::func_name
            for key, m in test_marks_map.items():
                if key.endswith(f"::{func_name}"):
                    marks = m
                    break

        # Strategy 4: Try with just function name
        if not marks and "::" in lookup_key:
            func_name = lookup_key.split("::")[-1]
            marks = test_marks_map.get(func_name, [])

        if marks:
            matched_count += 1
            # Record this test case for each operator mark
            for op_name in marks:
                if op_name not in grouped_results:
                    grouped_results[op_name] = {}
                grouped_results[op_name][test_case] = data
        else:
            unmatched_count += 1
            # Fallback: extract from function name
            op_name = None
            if "::" in test_case:
                func_part = test_case.split("::")[-1]
                if func_part.startswith("test_"):
                    op_name = func_part[5:]
                else:
                    op_name = func_part
                if "[" in op_name:
                    op_name = op_name.split("[")[0]

            if op_name:
                if op_name not in grouped_results:
                    grouped_results[op_name] = {}
                grouped_results[op_name][test_case] = data

    if matched_count == 0 and total_count > 0:
        warn(f"WARNING: No tests matched using marks. Using function name extraction for all {total_count} tests.")
        # Print sample keys for debugging
        if test_marks_map:
            warn(f"Marks map has {len(test_marks_map)} entries. Sample keys: {list(test_marks_map.keys())[:5]}")
    elif unmatched_count > 0:
        warn(f"{unmatched_count} test cases had no marks, used function name fallback")

    info(f"Matched {matched_count} test cases with marks, {unmatched_count} with fallback")
    return grouped_results

def load_marks_file(marks_file: Path) -> Dict:
    """Helper function to load marks from a file."""
    test_marks_map = {}
    try:
        with open(marks_file, "r") as f:
            collected_data = yaml.safe_load(f)
            if collected_data:
                for item in collected_data:
                    test_file = item.get("file", "")
                    test_func = item.get("function", "")
                    test_class = item.get("class", "")
                    marks = item.get("marks", [])

                    if test_class:
                        key = f"{test_file}::{test_class}::{test_func}"
                    else:
                        key = f"{test_file}::{test_func}"

                    alt_key = None
                    if test_file.startswith("tests/"):
                        alt_file = test_file[6:]
                        if test_class:
                            alt_key = f"{alt_file}::{test_class}::{test_func}"
                        else:
                            alt_key = f"{alt_file}::{test_func}"

                    if key not in test_marks_map or not test_marks_map[key]:
                        test_marks_map[key] = marks
                    if alt_key and (alt_key not in test_marks_map or not test_marks_map[alt_key]):
                        test_marks_map[alt_key] = marks

                    if test_func not in test_marks_map or not test_marks_map[test_func]:
                        test_marks_map[test_func] = marks
        return test_marks_map
    except Exception:
        return {}

def run_test_batch(ops: List[str], args, env: dict, output_dir: Path, batch_num: int = None, output_name: str = "accuracy_all.json") -> tuple:
    # Step 1: Collect all test marks (once per batch)
    # Pass output_dir so marks file is saved there
    test_marks_map = collect_all_marks(args, env, output_dir)

    # Step 2: Run pytest tests
    cmd = build_pytest_command(ops, args, output_name)

    start_time = time.time()
    return_code, stdout, stderr, duration = run_pytest_with_timeout(cmd, env, args.timeout)
    total_duration = time.time() - start_time

    suffix = f"_batch{batch_num}" if batch_num else ""
    stdout_file = output_dir / f"pytest_stdout{suffix}.log"
    stderr_file = output_dir / f"pytest_stderr{suffix}.log"

    with open(stdout_file, "w") as f:
        f.write(stdout)
    with open(stderr_file, "w") as f:
        f.write(stderr)

    # Step 3: Find result file
    result_file = find_result_file(output_name)

    if result_file is None:
        error(f"Result file not found in {TESTS_DIR}")
        return False, None, total_duration

    info(f"Found result file: {result_file}")

    dest_file = output_dir / output_name
    shutil.copy(result_file, dest_file)
    info(f"Copied result to: {dest_file}")

    # Step 4: Parse results using marks
    grouped_results = parse_results_with_marks(result_file, test_marks_map)
    info(f"Parsed results for {len(grouped_results)} operators")

    # Cleanup: remove the temporary result file from tests directory
    try:
        result_file.unlink()
        info(f"Cleaned up: {result_file}")
    except Exception as e:
        warn(f"Failed to cleanup {result_file}: {e}")

    # Generate summary
    filtered_results = {op: grouped_results.get(op, {}) for op in ops}
    summary = generate_summary(filtered_results, ops, total_duration)

    return True, summary, total_duration


# ============================================================================
# Multi-GPU Support Functions
# ============================================================================

def build_gpu_env(base_env: dict, gpu_id: int) -> dict:
    """Create environment dict for a specific GPU."""
    env = base_env.copy()
    try:
        vendor = flag_gems.vendor_name
        vendor_env_map = {
            "ascend": "ASCEND_RT_VISIBLE_DEVICES",
            "hygon": "HIP_VISIBLE_DEVICES",
            "metax": "MACA_VISIBLE_DEVICES",
            "mthreads": "MUSA_VISIBLE_DEVICES",
            "tsingmicro": "TXDA_VISIBLE_DEVICES",
            "iluvatar": "ILUVATAR_VISIBLE_DEVICES",
            "thead": "CUDA_VISIBLE_DEVICES",
            "cambricon": "MLU_VISIBLE_DEVICES",
            "enflame": "TOPS_VISIBLE_DEVICES",
        }
        env_var = vendor_env_map.get(vendor, "CUDA_VISIBLE_DEVICES")
        info(f"vendor: {vendor} - {env_var}")
        env[env_var] = str(gpu_id)
    except Exception:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def run_single_gpu_mode(ops: List[str], args, env: dict, output_dir: Path, output_prefix: str = "accuracy_all") -> tuple:
    """
    Run tests in single-GPU mode (sequential batches).
    Returns (all_results, total_duration, batch_success).
    """
    max_ops_per_batch = args.batch_size
    all_results = {}
    total_duration = 0
    batch_success = True

    if len(ops) > max_ops_per_batch:
        info(f"Testing {len(ops)} operators in batches of {max_ops_per_batch}")

        for i in range(0, len(ops), max_ops_per_batch):
            batch_ops = ops[i:i+max_ops_per_batch]
            batch_num = i // max_ops_per_batch + 1
            total_batches = (len(ops) + max_ops_per_batch - 1) // max_ops_per_batch

            info(f"\n{'='*70}")
            info(f"Batch {batch_num}/{total_batches}: Testing {len(batch_ops)} operators")
            info(f"{'='*70}")

            output_name = f"{output_prefix}_batch{batch_num}.json"
            success, summary, duration = run_test_batch(batch_ops, args, env, output_dir, batch_num, output_name)

            total_duration += duration

            if success and summary and "operators" in summary:
                all_results.update(summary["operators"])
            else:
                batch_success = False
    else:
        info(f"Testing {len(ops)} operators in a single batch")
        output_name = f"{output_prefix}.json"
        success, summary, duration = run_test_batch(ops, args, env, output_dir, output_name=output_name)
        total_duration = duration

        if success and summary and "operators" in summary:
            all_results = summary["operators"]
        else:
            batch_success = False

    return all_results, total_duration, batch_success


def run_gpu_worker(gpu_id: int, ops: List[str], args, base_output_dir: Path, base_env: dict) -> None:
    """
    Worker function to run tests on a specific GPU.
    Executes in a separate process. Results saved to gpu_output_dir/summary.json.
    """
    # Set up environment for this GPU
    env = build_gpu_env(base_env, gpu_id)

    # Create GPU-specific output directory
    gpu_output_dir = base_output_dir / f"gpu_{gpu_id}"
    ensure_dir(gpu_output_dir)

    info(f"\n{'='*70}")
    info(f"GPU {gpu_id}: Starting test run for {len(ops)} operators")
    info(f"{'='*70}")

    # Run tests using single GPU logic with isolated output prefix
    output_prefix = f"accuracy_all_gpu{gpu_id}"
    all_results, total_duration, batch_success = run_single_gpu_mode(ops, args, env, gpu_output_dir, output_prefix)

    if all_results:
        # Build summary matching single GPU format exactly
        gpu_summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "duration_seconds": total_duration,
            "totals": {
                "total_ops": len(ops),
                "total_cases": sum(op.get("total", 0) for op in all_results.values()),
                "passed": sum(op.get("passed", 0) for op in all_results.values()),
                "failed": sum(op.get("failed", 0) for op in all_results.values()),
                "skipped": sum(op.get("skipped", 0) for op in all_results.values()),
                "errors": sum(op.get("errors", 0) for op in all_results.values()),
                "not_found": sum(1 for op in all_results.values() if op.get("status") == "NotFound"),
            },
            "skip_lists": {
                "skip_operators": sorted(SKIP_OPERATORS),
                "skip_test_patterns": sorted(SKIP_TEST_PATTERNS),
                "skip_test_files": sorted(SKIP_TEST_FILES),
                "skip_file_source": str(args.skip_file) if args.skip_file else "default",
            },
            "operators": all_results
        }

        summary_file = gpu_output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(gpu_summary, f, indent=2)

        info(f"GPU {gpu_id}: Summary saved to {summary_file}")
    else:
        error(f"GPU {gpu_id}: No results collected")


def merge_gpu_summaries(output_dir: Path, gpu_ids: List[int], wall_duration: float) -> Optional[Dict]:
    """Merge summaries from multiple GPU workers into a single summary."""
    merged_operators = {}
    merged_totals = {
        "total_ops": 0,
        "total_cases": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "not_found": 0,
    }
    skip_lists = None

    for gpu_id in gpu_ids:
        gpu_summary_file = output_dir / f"gpu_{gpu_id}" / "summary.json"
        if not gpu_summary_file.exists():
            warn(f"GPU {gpu_id} summary file not found: {gpu_summary_file}")
            continue

        try:
            with open(gpu_summary_file, "r") as f:
                gpu_summary = json.load(f)
        except Exception as e:
            error(f"Failed to parse GPU {gpu_id} summary: {e}")
            continue

        merged_operators.update(gpu_summary.get("operators", {}))

        totals = gpu_summary.get("totals", {})
        for key in merged_totals:
            merged_totals[key] += totals.get(key, 0)

        if skip_lists is None:
            skip_lists = gpu_summary.get("skip_lists")

    if not merged_operators:
        return None

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "duration_seconds": wall_duration,
        "totals": merged_totals,
        "skip_lists": skip_lists if skip_lists else {
            "skip_operators": sorted(SKIP_OPERATORS),
            "skip_test_patterns": sorted(SKIP_TEST_PATTERNS),
            "skip_test_files": sorted(SKIP_TEST_FILES),
            "skip_file_source": str(args.skip_file) if args.skip_file else "default",
        },
        "operators": merged_operators
    }


def main():
    parser = argparse.ArgumentParser(
        description="CI environment batch test runner for FlagGems operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stable operators with quick mode on single GPU
  python ci_run_tests.py --gpus 0 --quick --output-dir logs/ci_quick

  # Run all stable operators with quick mode on 4 GPUs in parallel
  python ci_run_tests.py --gpus 0,1,2,3 --quick --output-dir logs/ci_quick

  # Run only first parameter combination for each test function on all GPUs
  python ci_run_tests.py --gpus all --first-parameter-only --output-dir logs/ci_first_param

  # Run specific operators
  python ci_run_tests.py --ops "add,softmax,mul" --gpus 0,1

  # Use custom skip file
  python ci_run_tests.py --skip-file my_skip_list.yaml --gpus 0,1,2,3

  # Show skip list
  python ci_run_tests.py --show-skips
        """
    )

    parser.add_argument(
        "--ops",
        help="Comma-separated list of operator IDs, e.g., 'add,softmax,mul'"
    )
    parser.add_argument(
        "--op-list-file",
        help="Read operator list from file, one ID per line"
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help='GPU IDs, comma-separated, or "all" to use all GPUs. Default: "0"'
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: logs_ci_YYYYMMDD_HHMM"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run tests in quick mode (fewer test cases)"
    )
    parser.add_argument(
        "--first-parameter-only",
        action="store_true",
        help="Run only the first parameter combination for each test function"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=72000,
        help="Test timeout in seconds. Default: 72000"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not clean up temporary files (for debugging)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of operators per batch. Default: 1000"
    )
    parser.add_argument(
        "--show-skips",
        action="store_true",
        help="Show the list of skipped operators and patterns"
    )
    parser.add_argument(
        "--skip-file",
        type=Path,
        default=None,
        help="Path to skip list file (YAML or text format)"
    )

    args = parser.parse_args()

    # Load skip file if provided
    if args.skip_file:
        apply_skip_file(args.skip_file)

    # Show skip list if requested
    if args.show_skips:
        print("\n" + "=" * 70)
        print("SKIP LIST")
        print("=" * 70)
        print(f"\nSkipped operators ({len(SKIP_OPERATORS)}):")
        for op in sorted(SKIP_OPERATORS):
            print(f"  - {op}")
        print(f"\nSkipped test patterns ({len(SKIP_TEST_PATTERNS)}):")
        for pattern in sorted(SKIP_TEST_PATTERNS):
            print(f"  - {pattern}")
        print(f"\nSkipped test files ({len(SKIP_TEST_FILES)}):")
        for file in sorted(SKIP_TEST_FILES):
            print(f"  - {file}")
        if args.skip_file:
            print(f"\nSkip file source: {args.skip_file}")
        print("\n" + "=" * 70)
        sys.exit(0)

    # Parse GPU list
    if args.gpus.strip().lower() == "all":
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                error("No GPU devices detected")
                sys.exit(1)
            gpu_ids = list(range(gpu_count))
            info(f"Using all {gpu_count} GPUs")
        except Exception as e:
            error(f"Failed to detect GPUs: {e}")
            sys.exit(1)
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    if not gpu_ids:
        error("No valid GPU IDs specified")
        sys.exit(1)

    # Get operators to test
    ops = get_ops_to_test(args)
    if not ops:
        error("No operators to test")
        sys.exit(1)

    info(f"Testing {len(ops)} operators on {len(gpu_ids)} GPU(s): {gpu_ids}")

    # Create output directory
    if args.output_dir is None:
        output_dir = ROOT / f"logs_ci_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    else:
        output_dir = Path(args.output_dir)

    ensure_dir(output_dir)
    info(f"Output directory: {output_dir}")

    # Save operator list for reference
    ops_file = output_dir / "operators_list.txt"
    with open(ops_file, "w") as f:
        for op in ops:
            f.write(f"{op}\n")

    # ------------------------------------------------------------------
    # Inject deterministic random patch to avoid non-deterministic test
    # parameter generation (e.g. random.choice in test_diagonal_backward)
    # causing different test case names across runs.
    # ------------------------------------------------------------------
    patch_dir = tempfile.mkdtemp(prefix="flaggems_deterministic_")
    sitecustomize_path = os.path.join(patch_dir, "sitecustomize.py")
    with open(sitecustomize_path, "w") as f:
        f.write(
            "import random\n"
            "def _fixed_seed(*args, **kwargs):\n"
            "    random._original_seed(42)\n"
            "def _fixed_choice(seq):\n"
            "    if not seq:\n"
            "        raise IndexError('Cannot choose from an empty sequence')\n"
            "    return seq[0]\n"
            "if not hasattr(random, '_original_seed'):\n"
            "    random._original_seed = random.seed\n"
            "    random.seed = _fixed_seed\n"
            "    random._original_choice = random.choice\n"
            "    random.choice = _fixed_choice\n"
        )
    atexit.register(lambda d=patch_dir: shutil.rmtree(d, ignore_errors=True))

    base_env = os.environ.copy()
    old_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = patch_dir + (os.pathsep + old_pythonpath if old_pythonpath else "")
    # ------------------------------------------------------------------

    # Determine execution mode
    multi_gpu = len(gpu_ids) > 1

    if not multi_gpu:
        # ==================== SINGLE GPU MODE ====================
        env = build_gpu_env(base_env, gpu_ids[0])

        all_results, total_duration, batch_success = run_single_gpu_mode(ops, args, env, output_dir)

        if all_results:
            final_summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "duration_seconds": total_duration,
                "totals": {
                    "total_ops": len(ops),
                    "total_cases": sum(op.get("total", 0) for op in all_results.values()),
                    "passed": sum(op.get("passed", 0) for op in all_results.values()),
                    "failed": sum(op.get("failed", 0) for op in all_results.values()),
                    "skipped": sum(op.get("skipped", 0) for op in all_results.values()),
                    "errors": sum(op.get("errors", 0) for op in all_results.values()),
                    "not_found": sum(1 for op in all_results.values() if op.get("status") == "NotFound"),
                },
                "skip_lists": {
                    "skip_operators": sorted(SKIP_OPERATORS),
                    "skip_test_patterns": sorted(SKIP_TEST_PATTERNS),
                    "skip_test_files": sorted(SKIP_TEST_FILES),
                    "skip_file_source": str(args.skip_file) if args.skip_file else "default",
                },
                "operators": all_results
            }

            summary_file = output_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(final_summary, f, indent=2)

            print_summary(final_summary)
            sys.exit(0)
        else:
            error("No results collected")
            sys.exit(1)

    else:
        # ==================== MULTI GPU MODE ====================
        # Split ops into contiguous chunks for each GPU
        ops_per_gpu = [[] for _ in gpu_ids]
        chunk_size = (len(ops) + len(gpu_ids) - 1) // len(gpu_ids)
        for i, gpu_id in enumerate(gpu_ids):
            start = i * chunk_size
            end = min(start + chunk_size, len(ops))
            ops_per_gpu[i] = ops[start:end]

        # Launch parallel workers
        processes = []
        wall_start = time.time()

        for gpu_id, gpu_ops in zip(gpu_ids, ops_per_gpu):
            if not gpu_ops:
                info(f"GPU {gpu_id}: No operators assigned, skipping")
                continue

            p = Process(target=run_gpu_worker, args=(gpu_id, gpu_ops, args, output_dir, base_env))
            p.start()
            processes.append((gpu_id, p))
            info(f"Launched worker for GPU {gpu_id} with {len(gpu_ops)} operators")

        # Wait for all workers to complete
        for gpu_id, p in processes:
            p.join()
            info(f"GPU {gpu_id}: Worker finished with exit code {p.exitcode}")

        wall_duration = time.time() - wall_start

        # Merge results from all GPUs
        active_gpu_ids = [gid for gid, _ in processes]
        final_summary = merge_gpu_summaries(output_dir, active_gpu_ids, wall_duration)

        if final_summary:
            summary_file = output_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(final_summary, f, indent=2)

            print_summary(final_summary)
            sys.exit(0)
        else:
            error("No results collected from any GPU")
            sys.exit(1)


if __name__ == "__main__":
    main()
