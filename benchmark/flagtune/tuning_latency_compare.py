#!/usr/bin/env python3
"""Compare FlagGems default-policy and expanded-space tuning end to end.

The benchmark intentionally separates two concepts that share the FlagTune
name:

* ``default`` unsets the legacy FlagGems ``USE_FLAGTUNE`` switch. The kernel
  keeps its default config list and its ``flagtune`` LibTuner policy invokes
  the FlagTree XGBoost+GA proposer.
* ``expanded`` sets ``USE_FLAGTUNE=1``. FlagGems replaces the config list with
  the expanded parameter space and the policy deliberately uses LibTuner's
  default exhaustive search instead of the FlagTree proposer.

Each method receives an isolated ``FLAGGEMS_CACHE_DIR`` and runs two phases:

1. The cold phase removes that method's config cache, distributes shapes over
   the requested visible GPUs, synchronizes the device, and wall-clock times
   the first Gems MM call. This measurement includes configuration selection,
   compilation, benchmarking, and cache population.
2. The hot phase preserves the populated cache and invokes the existing
   FlagGems pytest performance benchmark to measure cached operator latency.

The script merges cold/hot records into ``tuning_latency_compare.csv`` and
JSONL, writes a Markdown report and manifest, and pivots the two methods into
``tuning_latency_compare_by_shape.csv``. The one-row-per-shape report contains
the tuning-time ratio and cached-latency comparison used by the higher-level
benchmark workflow.

For normal use, run ``run_tuning_latency_compare.sh``. That wrapper performs
environment checks, fixes the method set to ``default,expanded``, places both
results and caches under ``<project_root>/flagtune-benchmark-output/``, captures
the invocation, and verifies that all aggregate artifacts were produced.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen3.5-397B-A17B-p1024d1024"
DEFAULT_CACHE_ROOT = Path("/tmp/flaggems_tuning_latency")
SUPPORTED_METHODS = ("default", "expanded")
METHOD_DESCRIPTIONS = {
    "default": (
        "USE_FLAGTUNE unset: keep the default config list and use the "
        "FlagTree XGBoost+GA proposer policy."
    ),
    "expanded": (
        "USE_FLAGTUNE=1: switch to the expanded config space and exhaustively "
        "benchmark it with the LibTuner default policy."
    ),
}

DATA_LINE_RE = re.compile(
    r"^(?P<status>SUCCESS|FAILED)\s+"
    r"(?P<torch>N/A|[\d.eE+-]+)\s+"
    r"(?P<gems>N/A|[\d.eE+-]+)\s+"
    r"(?P<speedup>N/A|[\d.eE+-]+)"
    r"(?:\s+(?P<tflops>N/A|[\d.eE+-]+))?\s+"
    r"(?P<shape>\[.*\])\s*$"
)
TORCH_SIZE_RE = re.compile(r"torch\.Size\(\[(?P<dims>[^\]]+)\]\)")


@dataclass(frozen=True)
class ShapeRecord:
    dims: Tuple[int, ...]
    count: int = 1

    @property
    def shape_key(self) -> str:
        return ",".join(str(dim) for dim in self.dims)

    @property
    def display(self) -> str:
        return ", ".join(str(dim) for dim in self.dims)


@dataclass(frozen=True)
class BenchRecord:
    status: str
    torch_ms: Optional[float]
    gems_ms: Optional[float]
    speedup: Optional[float]
    tflops: Optional[float]
    shape_text: str
    shape_key: str


@dataclass(frozen=True)
class ColdRecord:
    status: str
    shape_key: str
    cold_gems_ms: Optional[float]
    cold_torch_ms: Optional[float]
    error: str = ""


@dataclass(frozen=True)
class CacheSummary:
    config_cache_db_bytes: int = 0
    benchmark_cache_rows: int = 0


@dataclass(frozen=True)
class ColdRunResult:
    returncode: int
    records: List[ColdRecord]
    log_path: Path
    jsonl_path: Path
    cold_pass_wall_s: float
    cache_summary: CacheSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FlagGems benchmarks twice per method: cold cache for tuning "
            "cost approximation and hot cache for cached inference latency."
        )
    )
    parser.add_argument("--_worker-cold-first-call", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shape-yaml", default=None, help="Input FlagTune-style shape yaml.")
    parser.add_argument("--model", default=None, help="Model name under shape-config/ when --shape-yaml is omitted.")
    parser.add_argument("--op", default="mm", help="Pytest benchmark marker/operator name.")
    parser.add_argument("--methods", default="default,expanded", help="Comma-separated methods: default,expanded.")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="Root directory for per-method FlagGems caches.")
    parser.add_argument(
        "--clear-cache-policy",
        default="per-method",
        choices=["per-method"],
        help="Cache clearing policy. v1 supports one isolated cache per method.",
    )
    parser.add_argument("--cold-warmup", type=int, default=0)
    parser.add_argument("--cold-iter", type=int, default=1)
    parser.add_argument("--hot-warmup", type=int, default=1000)
    parser.add_argument("--hot-iter", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--dtypes", default="bfloat16")
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument("--start-shape", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_existing_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_shape_yaml(args: argparse.Namespace, invocation_dir: Path) -> Path:
    if args.shape_yaml:
        path = resolve_existing_path(args.shape_yaml, invocation_dir)
        if not path.exists():
            raise FileNotFoundError(f"--shape-yaml not found: {path}")
        return path

    model = args.model or DEFAULT_MODEL
    candidates = [
        invocation_dir / "shape-config" / f"{model}.yaml",
        invocation_dir / f"{model}.yaml",
        REPO_ROOT.parent / "autotune" / "FlagGems" / "FlagTune" / "shape-config" / f"{model}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot resolve shape yaml for model {model!r}. Tried:\n  {tried}")


def resolve_count_yaml(shape_yaml: Path) -> Optional[Path]:
    stem = shape_yaml.stem
    if stem.endswith("_count"):
        return shape_yaml
    count_path = shape_yaml.with_name(f"{stem}_count.yaml")
    return count_path if count_path.exists() else None


def method_list(methods_text: str) -> List[str]:
    methods = [method.strip() for method in methods_text.split(",") if method.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method")
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"Unsupported method(s): {', '.join(unsupported)}. Supported: {supported}")
    return methods


def normalize_shape_row(row: Sequence[Any]) -> ShapeRecord:
    values = [int(value) for value in row]
    if len(values) < 4:
        raise ValueError(f"Shape row must contain at least 4 integers: {row!r}")
    dims = tuple(values[:4])
    count = int(values[4]) if len(values) >= 5 else 1
    return ShapeRecord(dims=dims, count=count)


def load_shape_records(shape_yaml: Path, op: str) -> List[ShapeRecord]:
    payload = load_yaml(shape_yaml)
    op_payload = payload.get(op) or {}
    raw_shapes = op_payload.get("shapes") or []
    records = []
    for row in raw_shapes:
        if isinstance(row, (list, tuple)):
            records.append(normalize_shape_row(row))
    return records


def load_count_map(count_yaml: Optional[Path], op: str) -> Dict[str, int]:
    if count_yaml is None:
        return {}
    try:
        records = load_shape_records(count_yaml, op)
    except Exception:
        return {}
    return {record.shape_key: record.count for record in records}


def select_shape_records(
    shape_yaml: Path,
    op: str,
    start_shape: int,
    max_shapes: Optional[int],
) -> List[ShapeRecord]:
    records = load_shape_records(shape_yaml, op)
    count_map = load_count_map(resolve_count_yaml(shape_yaml), op)
    if count_map:
        records = [
            ShapeRecord(record.dims, count_map.get(record.shape_key, record.count))
            for record in records
        ]
        records.sort(key=lambda record: record.count, reverse=True)
    if start_shape:
        records = records[start_shape:]
    if max_shapes is not None:
        records = records[:max_shapes]
    if not records:
        raise ValueError(f"No shapes selected from {shape_yaml} for op={op}")
    return records


def write_filtered_shape_yaml(output_path: Path, op: str, records: Sequence[ShapeRecord]) -> None:
    payload = {
        op: {
            "shapes": [list(record.dims) for record in records],
            "shape_desc": "B, M, N, K",
        }
    }
    write_yaml(output_path, payload)


def to_float(value: str) -> Optional[float]:
    if value == "N/A":
        return None
    return float(value)


def infer_bmnk_from_shape_text(shape_text: str) -> Optional[Tuple[int, int, int, int]]:
    sizes: List[Tuple[int, ...]] = []
    for match in TORCH_SIZE_RE.finditer(shape_text):
        dims = tuple(
            int(token.strip())
            for token in match.group("dims").split(",")
            if token.strip()
        )
        sizes.append(dims)

    if len(sizes) < 2:
        return None
    a_shape, b_shape = sizes[0], sizes[1]
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None
    m, k = a_shape
    b0, b1 = b_shape
    if b0 == k:
        return (1, m, b1, k)
    if b1 == k:
        return (1, m, b0, k)
    return None


def parse_benchmark_output(text: str) -> List[BenchRecord]:
    records: List[BenchRecord] = []
    for line in text.splitlines():
        match = DATA_LINE_RE.match(line.strip())
        if match is None:
            continue
        shape_text = match.group("shape")
        dims = infer_bmnk_from_shape_text(shape_text)
        shape_key = ",".join(str(dim) for dim in dims) if dims is not None else shape_text
        records.append(
            BenchRecord(
                status=match.group("status"),
                torch_ms=to_float(match.group("torch")),
                gems_ms=to_float(match.group("gems")),
                speedup=to_float(match.group("speedup")),
                tflops=to_float(match.group("tflops") or "N/A"),
                shape_text=shape_text,
                shape_key=shape_key,
            )
        )
    return records


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def clear_config_cache(cache_dir: Path) -> None:
    shutil.rmtree(cache_dir / "config_cache", ignore_errors=True)


def method_env(method: str, cache_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["FLAGGEMS_CACHE_DIR"] = str(cache_dir)
    if method == "expanded":
        env["USE_FLAGTUNE"] = "1"
        env.pop("FLAGTUNE_INCLUDE", None)
    else:
        env.pop("USE_FLAGTUNE", None)
        env.pop("FLAGTUNE_INCLUDE", None)
    return env


def visible_devices_env() -> str:
    return "CUDA_VISIBLE_DEVICES"


def dtype_name(dtypes: str) -> str:
    dtype = dtypes.split(",")[0].strip()
    if not dtype:
        raise ValueError("--dtypes must contain at least one dtype")
    return dtype


def shape_flops(record: ShapeRecord) -> int:
    b, m, n, k = record.dims[:4]
    return max(1, b) * max(1, m) * max(1, n) * max(1, k) * 2


def split_shape_records(records: Sequence[ShapeRecord], parallel: int) -> List[List[ShapeRecord]]:
    workers = max(1, min(int(parallel) if parallel else 1, len(records)))
    buckets: List[List[ShapeRecord]] = [[] for _ in range(workers)]
    costs = [0] * workers
    for record in sorted(records, key=shape_flops, reverse=True):
        target = min(range(workers), key=lambda idx: costs[idx])
        buckets[target].append(record)
        costs[target] += shape_flops(record)
    order = {record.shape_key: idx for idx, record in enumerate(records)}
    for bucket in buckets:
        bucket.sort(key=lambda record: order[record.shape_key])
    return [bucket for bucket in buckets if bucket]


def serialize_shape_records(path: Path, records: Sequence[ShapeRecord]) -> None:
    payload = [
        {"dims": list(record.dims), "count": record.count}
        for record in records
    ]
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def load_shape_json(path: Path) -> List[ShapeRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ShapeRecord(tuple(int(dim) for dim in item["dims"]), int(item.get("count", 1)))
        for item in payload
    ]


def torch_dtype_from_name(name: str):
    import torch

    normalized = name.replace("torch.", "").strip()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported cold worker dtype: {name}")
    return mapping[normalized]


def synchronize_device(flag_gems_module) -> None:
    flag_gems_module.runtime.torch_device_fn.synchronize()


def run_cold_worker(args: argparse.Namespace) -> int:
    if args.op != "mm":
        raise ValueError("cold first-call worker currently supports --op mm only")
    if args.shape_json is None:
        raise ValueError("--shape-json is required for cold first-call worker")
    if args.cache_dir is None or args.method is None:
        raise ValueError("--cache-dir and --method are required for cold first-call worker")

    records = load_shape_json(Path(args.shape_json))

    import torch
    import flag_gems

    dtype = torch_dtype_from_name(args.dtype or args.dtypes)
    device = flag_gems.device

    for record in records:
        b, m, n, k = record.dims[:4]
        status = "SUCCESS"
        cold_gems_ms: Optional[float] = None
        error = ""
        try:
            if b != 1:
                raise ValueError(f"mm cold worker expects B=1, got shape={record.display}")
            a = torch.randn((m, k), dtype=dtype, device=device)
            b_tensor = torch.randn((k, n), dtype=dtype, device=device)
            synchronize_device(flag_gems)
            start = time.perf_counter()
            with flag_gems.use_gems(exclude=["zero_"]):
                torch.Tensor.mm(a, b_tensor)
            synchronize_device(flag_gems)
            cold_gems_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            status = "FAILED"
            error = str(exc)

        print(
            json.dumps(
                {
                    "shape": record.display,
                    "shape_key": record.shape_key,
                    "count": record.count,
                    "method": args.method,
                    "status": status,
                    "cold_gems_ms": cold_gems_ms,
                    "cold_torch_ms": None,
                    "cache_dir": args.cache_dir,
                    "gpu_id": args.gpu_id,
                    "error": error,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


def parse_cold_worker_output(text: str) -> List[ColdRecord]:
    records: List[ColdRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "shape_key" not in payload or "cold_gems_ms" not in payload:
            continue
        records.append(
            ColdRecord(
                status=str(payload.get("status", "FAILED")),
                shape_key=str(payload["shape_key"]),
                cold_gems_ms=payload.get("cold_gems_ms"),
                cold_torch_ms=payload.get("cold_torch_ms"),
                error=str(payload.get("error") or ""),
            )
        )
    return records


def summarize_config_cache(cache_dir: Path) -> CacheSummary:
    config_cache = cache_dir / "config_cache"
    db_bytes = 0
    benchmark_rows = 0
    for db_path in config_cache.glob("*.db"):
        db_bytes += db_path.stat().st_size
        try:
            con = sqlite3.connect(db_path)
            try:
                tables = [
                    row[0]
                    for row in con.execute(
                        "select name from sqlite_master where type='table'"
                    )
                ]
                for table in tables:
                    if "_benchmark" not in table:
                        continue
                    quoted = '"' + table.replace('"', '""') + '"'
                    benchmark_rows += int(
                        con.execute(f"select count(*) from {quoted}").fetchone()[0]
                    )
            finally:
                con.close()
        except sqlite3.Error:
            continue
    return CacheSummary(
        config_cache_db_bytes=db_bytes,
        benchmark_cache_rows=benchmark_rows,
    )


def run_cold_first_call_pass(
    *,
    method: str,
    shape_records: Sequence[ShapeRecord],
    cache_dir: Path,
    output_dir: Path,
    op: str,
    parallel: int,
    dtypes: str,
) -> ColdRunResult:
    start = time.perf_counter()
    chunks = split_shape_records(shape_records, parallel)
    all_records: List[ColdRecord] = []
    log_parts_by_worker: Dict[int, List[str]] = {}
    returncode = 0

    def launch_worker(worker_id: int, chunk: Sequence[ShapeRecord]):
        shape_json = output_dir / f"cold_{method}_worker_{worker_id}_shapes.json"
        serialize_shape_records(shape_json, chunk)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker-cold-first-call",
            "--shape-json",
            str(shape_json),
            "--op",
            op,
            "--dtype",
            dtype_name(dtypes),
            "--method",
            method,
            "--cache-dir",
            str(cache_dir),
            "--gpu-id",
            str(worker_id),
        ]
        env = method_env(method, cache_dir)
        env[visible_devices_env()] = str(worker_id)
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        worker_records = parse_cold_worker_output(completed.stdout)
        log_parts = [
            f"[worker] {worker_id}",
            f"[shape_json] {shape_json}",
            f"[cmd] {format_command(cmd)}",
            f"[returncode] {completed.returncode}",
            "",
            "[stdout]",
            completed.stdout,
            "",
            "[stderr]",
            completed.stderr,
            "",
        ]
        return worker_id, completed.returncode, worker_records, log_parts

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(launch_worker, worker_id, chunk)
            for worker_id, chunk in enumerate(chunks)
        ]
        for future in concurrent.futures.as_completed(futures):
            worker_id, worker_returncode, worker_records, log_parts = future.result()
            if worker_returncode != 0 and returncode == 0:
                returncode = worker_returncode
            all_records.extend(worker_records)
            log_parts_by_worker[worker_id] = log_parts

    shape_order = {record.shape_key: idx for idx, record in enumerate(shape_records)}
    all_records.sort(key=lambda record: shape_order.get(record.shape_key, len(shape_order)))
    log_parts: List[str] = []
    for worker_id in sorted(log_parts_by_worker):
        log_parts.extend(log_parts_by_worker[worker_id])

    cold_pass_wall_s = time.perf_counter() - start
    cache_summary = summarize_config_cache(cache_dir)
    log_path = output_dir / f"cold_{method}.log"
    jsonl_path = output_dir / f"cold_{method}.jsonl"
    log_path.write_text(
        "\n".join(
            [
                f"[method] {method}",
                "[pass] cold",
                "[source] first_call_wall_clock",
                f"[cache_dir] {cache_dir}",
                f"[cold_pass_wall_s] {cold_pass_wall_s:.6f}",
                f"[config_cache_db_bytes] {cache_summary.config_cache_db_bytes}",
                f"[benchmark_cache_rows] {cache_summary.benchmark_cache_rows}",
                "",
                *log_parts,
            ]
        ),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(
                json.dumps(
                    {
                        "shape_key": record.shape_key,
                        "status": record.status,
                        "cold_gems_ms": record.cold_gems_ms,
                        "cold_torch_ms": record.cold_torch_ms,
                        "error": record.error,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return ColdRunResult(
        returncode=returncode,
        records=all_records,
        log_path=log_path,
        jsonl_path=jsonl_path,
        cold_pass_wall_s=cold_pass_wall_s,
        cache_summary=cache_summary,
    )


def build_pytest_command(
    shape_yaml: Path,
    op: str,
    warmup: int,
    iteration: int,
    parallel: int,
    dtypes: str,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "benchmark/test_blas_perf_parallel.py",
        "-m",
        op,
        "-s",
        "--shape_file",
        str(shape_yaml),
        "--level",
        "core",
        "--mode",
        "kernel",
        "--parallel",
        str(parallel),
        "--warmup",
        str(warmup),
        "--iter",
        str(iteration),
        "-v",
        "--dtypes",
        dtypes,
    ]


def run_benchmark_pass(
    *,
    pass_name: str,
    method: str,
    shape_yaml: Path,
    cache_dir: Path,
    output_dir: Path,
    op: str,
    warmup: int,
    iteration: int,
    parallel: int,
    dtypes: str,
) -> Tuple[subprocess.CompletedProcess[str], List[BenchRecord], Path]:
    cmd = build_pytest_command(shape_yaml, op, warmup, iteration, parallel, dtypes)
    env = method_env(method, cache_dir)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    log_path = output_dir / f"{pass_name}_{method}.log"
    log_path.write_text(
        "\n".join(
            [
                f"[method] {method}",
                f"[pass] {pass_name}",
                f"[cache_dir] {cache_dir}",
                f"[FLAGGEMS_CACHE_DIR] {env.get('FLAGGEMS_CACHE_DIR', '')}",
                f"[USE_FLAGTUNE] {env.get('USE_FLAGTUNE', '')}",
                f"[cmd] {format_command(cmd)}",
                f"[returncode] {completed.returncode}",
                "",
                "[stdout]",
                completed.stdout,
                "",
                "[stderr]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return completed, parse_benchmark_output(completed.stdout), log_path


def record_by_key(records: Sequence[BenchRecord]) -> Dict[str, BenchRecord]:
    return {record.shape_key: record for record in records}


def pct_delta(new: Optional[float], base: Optional[float]) -> Optional[float]:
    if new is None or base is None or base == 0:
        return None
    return (new / base - 1.0) * 100.0


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def merge_method_rows(
    method: str,
    shape_records: Sequence[ShapeRecord],
    cold_records: Sequence[ColdRecord],
    hot_records: Sequence[BenchRecord],
    cache_dir: Path,
    cold_result: ColdRunResult,
    hot_completed: subprocess.CompletedProcess[str],
) -> List[Dict[str, Any]]:
    cold_by_key = record_by_key(cold_records)
    hot_by_key = record_by_key(hot_records)
    rows: List[Dict[str, Any]] = []
    for shape in shape_records:
        cold = cold_by_key.get(shape.shape_key)
        hot = hot_by_key.get(shape.shape_key)
        errors = []
        if cold_result.returncode != 0:
            errors.append(f"cold returncode={cold_result.returncode}")
        if hot_completed.returncode != 0:
            errors.append(f"hot returncode={hot_completed.returncode}")
        if cold is None:
            errors.append("missing cold benchmark row")
        elif cold.status != "SUCCESS":
            errors.append(cold.error or f"cold status={cold.status}")
        if hot is None:
            errors.append("missing hot benchmark row")
        elif hot.status != "SUCCESS":
            errors.append(f"hot status={hot.status}")
        status = "ok" if not errors else "error"
        cold_gems = cold.cold_gems_ms if cold else None
        hot_gems = hot.gems_ms if hot else None
        rows.append(
            {
                "shape": shape.display,
                "shape_key": shape.shape_key,
                "count": shape.count,
                "method": method,
                "status": status,
                "cold_gems_ms": cold_gems,
                "hot_gems_ms": hot_gems,
                "cold_torch_ms": cold.cold_torch_ms if cold else None,
                "torch_ms": hot.torch_ms if hot else None,
                "speedup": hot.speedup if hot else None,
                "cold_hot_ratio": safe_ratio(cold_gems, hot_gems),
                "cache_dir": str(cache_dir),
                "cold_wall_source": "first_call_wall_clock",
                "hot_latency_source": "pytest_do_bench",
                "cold_pass_wall_s": cold_result.cold_pass_wall_s,
                "config_cache_db_bytes": cold_result.cache_summary.config_cache_db_bytes,
                "benchmark_cache_rows": cold_result.cache_summary.benchmark_cache_rows,
                "error": "; ".join(errors),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "shape",
        "shape_key",
        "count",
        "method",
        "status",
        "cold_gems_ms",
        "hot_gems_ms",
        "cold_torch_ms",
        "torch_ms",
        "speedup",
        "cold_hot_ratio",
        "cache_dir",
        "cold_wall_source",
        "hot_latency_source",
        "cold_pass_wall_s",
        "config_cache_db_bytes",
        "benchmark_cache_rows",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_by_shape_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Pivot default/expanded rows into the one-row-per-shape report."""
    required_methods = set(SUPPORTED_METHODS)
    shape_order: List[str] = []
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        shape_key = str(row["shape_key"])
        shape = str(row["shape"])
        if shape.replace(" ", "") != shape_key:
            raise ValueError(
                f"shape/shape_key mismatch: shape={shape!r}, shape_key={shape_key!r}"
            )
        if shape_key not in grouped:
            grouped[shape_key] = {}
            shape_order.append(shape_key)
        method = str(row["method"])
        if method in grouped[shape_key]:
            raise ValueError(f"duplicate method={method!r} for shape_key={shape_key!r}")
        grouped[shape_key][method] = row

    fieldnames = [
        "shape",
        "shape_key",
        "count",
        "default_tuning_ms",
        "expanded_tuning_ms",
        "default_vs_expanded_tuning_speedup",
        "default_hot_ms",
        "expanded_hot_ms",
        "default_perf_pct_of_expanded_hot",
        "default_torch_ms",
        "expanded_torch_ms",
        "cold_wall_source",
        "hot_latency_source",
    ]

    def csv_float(value: Optional[float]) -> str:
        return "" if value is None else f"{float(value):.6f}"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for shape_key in shape_order:
            method_rows = grouped[shape_key]
            missing = required_methods - set(method_rows)
            if missing:
                raise ValueError(
                    f"shape_key={shape_key!r} is missing methods: {sorted(missing)}"
                )
            default = method_rows["default"]
            expanded = method_rows["expanded"]
            default_tuning = default.get("cold_gems_ms")
            expanded_tuning = expanded.get("cold_gems_ms")
            default_hot = default.get("hot_gems_ms")
            expanded_hot = expanded.get("hot_gems_ms")
            hot_ratio = safe_ratio(expanded_hot, default_hot)
            writer.writerow(
                {
                    "shape": default["shape"],
                    "shape_key": shape_key,
                    "count": default["count"],
                    "default_tuning_ms": csv_float(default_tuning),
                    "expanded_tuning_ms": csv_float(expanded_tuning),
                    "default_vs_expanded_tuning_speedup": csv_float(
                        safe_ratio(expanded_tuning, default_tuning)
                    ),
                    "default_hot_ms": csv_float(default_hot),
                    "expanded_hot_ms": csv_float(expanded_hot),
                    "default_perf_pct_of_expanded_hot": csv_float(
                        hot_ratio * 100.0 if hot_ratio is not None else None
                    ),
                    "default_torch_ms": csv_float(default.get("torch_ms")),
                    "expanded_torch_ms": csv_float(expanded.get("torch_ms")),
                    "cold_wall_source": default.get("cold_wall_source"),
                    "hot_latency_source": default.get("hot_latency_source"),
                }
            )


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    materialized = [value for value in values if value is not None]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]], methods: Sequence[str]) -> None:
    lines: List[str] = []
    lines.append("# FlagGems Tuning/Latency Compare")
    lines.append("")
    lines.append("Cold cache Gems time is measured by wall-clock timing the first Gems call; hot cache Gems time is measured by the pytest do_bench latency path.")
    lines.append("")
    lines.append("- `default`: " + METHOD_DESCRIPTIONS["default"])
    lines.append("- `expanded`: " + METHOD_DESCRIPTIONS["expanded"])
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append("| Method | Rows | OK | Avg cold Gems ms | Avg hot Gems ms | Avg cold/hot ratio | Cold pass wall s | Cache DB bytes | Benchmark cache rows |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        ok_rows = [row for row in method_rows if row["status"] == "ok"]
        first_row = method_rows[0] if method_rows else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(len(method_rows)),
                    str(len(ok_rows)),
                    fmt(mean(row["cold_gems_ms"] for row in ok_rows)),
                    fmt(mean(row["hot_gems_ms"] for row in ok_rows)),
                    fmt(mean(row["cold_hot_ratio"] for row in ok_rows), digits=3),
                    fmt(first_row.get("cold_pass_wall_s"), digits=3),
                    str(first_row.get("config_cache_db_bytes", "-")),
                    str(first_row.get("benchmark_cache_rows", "-")),
                ]
            )
            + " |"
        )

    if "default" in methods and "expanded" in methods:
        lines.extend(["", "## Expanded vs Default", ""])
        lines.append("| Shape | Count | Cold delta % | Hot delta % |")
        lines.append("| --- | ---: | ---: | ---: |")
        by_shape_method = {(row["shape_key"], row["method"]): row for row in rows}
        shape_keys = []
        for row in rows:
            if row["shape_key"] not in shape_keys:
                shape_keys.append(row["shape_key"])
        for shape_key in shape_keys:
            default = by_shape_method.get((shape_key, "default"))
            expanded = by_shape_method.get((shape_key, "expanded"))
            if default is None or expanded is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        default["shape"],
                        str(default["count"]),
                        fmt(pct_delta(expanded["cold_gems_ms"], default["cold_gems_ms"]), digits=2),
                        fmt(pct_delta(expanded["hot_gems_ms"], default["hot_gems_ms"]), digits=2),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Per Shape", ""])
    lines.append("| Shape | Count | Method | Status | Cold Gems ms | Hot Gems ms | Torch ms | Speedup | Cold/Hot |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["shape"],
                    str(row["count"]),
                    row["method"],
                    row["status"],
                    fmt(row["cold_gems_ms"]),
                    fmt(row["hot_gems_ms"]),
                    fmt(row["torch_ms"]),
                    fmt(row["speedup"], digits=3),
                    fmt(row["cold_hot_ratio"], digits=3),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(run_name: str) -> Path:
    return SCRIPT_DIR / "results" / run_name


def main() -> int:
    args = parse_args()
    if args._worker_cold_first_call:
        return run_cold_worker(args)
    if args.op != "mm":
        raise ValueError("tuning_latency_compare v1 cold first-call mode supports --op mm only")

    invocation_dir = Path.cwd()
    shape_yaml = resolve_shape_yaml(args, invocation_dir)
    methods = method_list(args.methods)
    run_name = args.run_name or f"{shape_yaml.stem}_{args.op}"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(run_name)
    if not output_dir.is_absolute():
        output_dir = invocation_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shape_records = select_shape_records(
        shape_yaml=shape_yaml,
        op=args.op,
        start_shape=args.start_shape,
        max_shapes=args.max_shapes,
    )
    filtered_shape_yaml = output_dir / "selected_shapes.yaml"
    write_filtered_shape_yaml(filtered_shape_yaml, args.op, shape_records)

    cache_root = Path(args.cache_root).expanduser()
    if not cache_root.is_absolute():
        cache_root = invocation_dir / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "shape_yaml": str(shape_yaml),
        "selected_shape_yaml": str(filtered_shape_yaml),
        "methods": methods,
        "method_descriptions": {
            method: METHOD_DESCRIPTIONS[method] for method in methods
        },
        "cache_root": str(cache_root),
        "cold_warmup": args.cold_warmup,
        "cold_iter": args.cold_iter,
        "hot_warmup": args.hot_warmup,
        "hot_iter": args.hot_iter,
        "parallel": args.parallel,
        "dtypes": args.dtypes,
        "cold_wall_source": "first_call_wall_clock",
        "hot_latency_source": "pytest_do_bench",
        "method_summaries": {},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for method in methods:
        cache_dir = cache_root / f"{run_name}_{method}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        clear_config_cache(cache_dir)

        print(f"[RUN] method={method} pass=cold cache={cache_dir}", flush=True)
        cold_result = run_cold_first_call_pass(
            method=method,
            shape_records=shape_records,
            cache_dir=cache_dir,
            output_dir=output_dir,
            op=args.op,
            parallel=args.parallel,
            dtypes=args.dtypes,
        )
        print(f"[LOG] {cold_result.log_path}", flush=True)
        print(f"[JSONL] {cold_result.jsonl_path}", flush=True)
        manifest["method_summaries"][method] = {
            "cold_pass_wall_s": cold_result.cold_pass_wall_s,
            "config_cache_db_bytes": cold_result.cache_summary.config_cache_db_bytes,
            "benchmark_cache_rows": cold_result.cache_summary.benchmark_cache_rows,
        }
        if args.fail_fast and cold_result.returncode != 0:
            return cold_result.returncode

        print(f"[RUN] method={method} pass=hot cache={cache_dir}", flush=True)
        hot_completed, hot_records, hot_log = run_benchmark_pass(
            pass_name="hot",
            method=method,
            shape_yaml=filtered_shape_yaml,
            cache_dir=cache_dir,
            output_dir=output_dir,
            op=args.op,
            warmup=args.hot_warmup,
            iteration=args.hot_iter,
            parallel=args.parallel,
            dtypes=args.dtypes,
        )
        print(f"[LOG] {hot_log}", flush=True)
        if args.fail_fast and hot_completed.returncode != 0:
            return hot_completed.returncode

        all_rows.extend(
            merge_method_rows(
                method,
                shape_records,
                cold_result.records,
                hot_records,
                cache_dir,
                cold_result,
                hot_completed,
            )
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "tuning_latency_compare.csv"
    jsonl_path = output_dir / "tuning_latency_compare.jsonl"
    md_path = output_dir / "tuning_latency_compare.md"
    by_shape_csv_path = output_dir / "tuning_latency_compare_by_shape.csv"
    write_csv(csv_path, all_rows)
    write_jsonl(jsonl_path, all_rows)
    write_markdown(md_path, all_rows, methods)
    if set(SUPPORTED_METHODS).issubset(methods):
        write_by_shape_csv(by_shape_csv_path, all_rows)
    print(f"[DONE] wrote {csv_path}")
    print(f"[DONE] wrote {jsonl_path}")
    print(f"[DONE] wrote {md_path}")
    if set(SUPPORTED_METHODS).issubset(methods):
        print(f"[DONE] wrote {by_shape_csv_path}")
    return 0 if all(row["status"] == "ok" for row in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
