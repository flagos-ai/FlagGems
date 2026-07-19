#!/usr/bin/env python3
"""Directly microbenchmark Hopper MM configuration-selection policies.

This low-level diagnostic benchmark unwraps the LibTuner attached to the
FlagGems Hopper MM/GEMV kernels and evaluates three policy paths for every
selected model-derived shape:

1. ``default`` exhaustively benchmarks the kernel's built-in config list.
2. ``flagtune_xgb_only`` asks the FlagTree model for its top-k configs and
   benchmarks only those valid predictions.
3. The historical ``flagtune_xgb_ga`` output label calls the configured tuner
   policy with ``USE_FLAGTUNE=1`` and the expanded configs. With the current
   FlagGems legacy-switch semantics, that flag deliberately selects exhaustive
   expanded-space search rather than the FlagTree proposer. The old label is
   retained for output compatibility; use the end-to-end workflow for the
   canonical ``default`` versus ``expanded`` comparison.

The input must be a FlagTune shape YAML whose ``mm.shapes`` entries contain
``B, M, N, K, Count``. Shapes are sorted by ``Count`` before ``--start-shape``
and ``--max-shapes`` are applied. The script writes one CSV row and one JSONL
record per shape/policy pair, including tuning time, final latency, benchmark
count, selected config, and optional comparison with an existing benchmark DB.

Run this script in an environment where the intended FlagGems and FlagTree
sources are importable (normally through ``PYTHONPATH``) and a Hopper-class
CUDA GPU is visible. This is a policy-level diagnostic, not the complete
cold-cache/hot-cache workflow. For the reproducible end-to-end comparison and
per-shape aggregation, use ``run_tuning_latency_compare.sh`` instead.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SHAPE_YAML = (
    Path(__file__).resolve().parents[3]
    / "autotune"
    / "FlagGems"
    / "FlagTune"
    / "shape-config"
    / "Qwen3.5-397B-A17B-p32768d1024_count.yaml"
)
OP_ID = "flaggems/mm_general_tma"


@dataclass(frozen=True)
class Shape:
    batch: int
    m: int
    n: int
    k: int
    count: int

    @property
    def variant(self) -> str:
        return "gemv" if self.n <= 1 else "mm_general_tma"

    @property
    def shape_dict(self) -> Dict[str, int]:
        return {
            "M": self.m,
            "N": self.n,
            "K": self.k,
            "stride_am": self.k,
            "stride_bk": self.n,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare default, FlagTune XGB-only, and FlagTune XGB+GA for Hopper mm."
    )
    parser.add_argument("--shape-yaml", default=str(DEFAULT_SHAPE_YAML))
    parser.add_argument("--max-shapes", type=int, default=10)
    parser.add_argument("--start-shape", type=int, default=0)
    parser.add_argument("--include-gemv", action="store_true", default=True)
    parser.add_argument("--exclude-gemv", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--benchmark-db", default=None)
    parser.add_argument("--output-csv", default="mm_flagtune_benchmark.csv")
    parser.add_argument("--output-jsonl", default="mm_flagtune_benchmark.jsonl")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_shapes(path: Path, include_gemv: bool) -> List[Shape]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        raw_shapes = ((payload.get("mm") or {}).get("shapes") or [])
    except ModuleNotFoundError:
        raw_shapes = load_mm_shapes_without_yaml(path)
    shapes: List[Shape] = []
    for row in raw_shapes:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        b, m, n, k, count = [int(x) for x in row[:5]]
        if n <= 1 and not include_gemv:
            continue
        shapes.append(Shape(b, m, n, k, count))
    shapes.sort(key=lambda s: s.count, reverse=True)
    return shapes


def load_mm_shapes_without_yaml(path: Path) -> List[List[int]]:
    rows: List[List[int]] = []
    in_mm = False
    in_shapes = False
    current: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if not raw_line.startswith(" ") and stripped.endswith(":"):
                section = stripped[:-1]
                in_mm = section == "mm"
                in_shapes = False
                continue
            if not in_mm:
                continue
            if stripped == "shapes:":
                in_shapes = True
                continue
            if stripped.startswith("shape_desc:"):
                break
            if not in_shapes:
                continue
            if stripped.startswith("- - "):
                if len(current) >= 5:
                    rows.append(current[:5])
                current = [int(stripped.split("- - ", 1)[1])]
                continue
            if stripped.startswith("- ") and current:
                current.append(int(stripped.split("- ", 1)[1]))
        if len(current) >= 5:
            rows.append(current[:5])
    return rows


def unwrap_tuner(kernel_entry: Any) -> Any:
    fn = getattr(kernel_entry, "fn", kernel_entry)
    while fn is not None:
        if getattr(fn, "policy", None) is not None and getattr(fn, "_bench", None) is not None:
            return fn
        fn = getattr(fn, "fn", None)
    raise RuntimeError(f"Cannot find LibTuner inside {kernel_entry!r}")


def config_record(config: Any) -> Dict[str, Any]:
    try:
        data = dict(config.all_kwargs())
    except Exception:
        data = dict(getattr(config, "kwargs", {}) or {})
        for attr in ("num_warps", "num_stages", "num_ctas"):
            if hasattr(config, attr):
                data[attr] = getattr(config, attr)
    return {k: int(v) if isinstance(v, int) and not isinstance(v, bool) else v for k, v in data.items()}


def make_tensors(shape: Shape, dtype_name: str, device: str):
    import torch

    dtype = getattr(torch, dtype_name)
    a = torch.randn((shape.m, shape.k), device=device, dtype=dtype)
    b = torch.randn((shape.k, shape.n), device=device, dtype=dtype)
    c_dtype = torch.float32 if dtype == torch.float32 else dtype
    c = torch.empty((shape.m, shape.n), device=device, dtype=c_dtype)
    return a, b, c


def make_kernel_call(shape: Shape, tensors: Tuple[Any, Any, Any]):
    import torch
    import triton

    mm_ops = importlib.import_module("flag_gems.runtime.backend._nvidia.hopper.ops.mm")

    a, b, c = tensors
    if shape.variant == "gemv":
        tuner = unwrap_tuner(mm_ops.gemv_kernel)

        def grid(meta):
            return (triton.cdiv(shape.m, meta["BLOCK_M"]),)

        kernel_args = (
            a,
            b,
            c,
            shape.m,
            shape.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
        )
        kernel_kwargs = {
            "grid": grid,
            "IS_FP64": a.dtype == torch.float64,
            "warmup": False,
        }
        default_configs = list(tuner._flagtune_default_configs)
        expand_name = "gemv"
    else:
        tuner = unwrap_tuner(mm_ops.mm_kernel_general_host_tma)
        from triton.tools.tensor_descriptor import TensorDescriptor

        dummy_block = [1, 1]
        a_row_major = a.stride(1) == 1
        b_row_major = b.stride(1) == 1
        a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
        b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
        dtype_str = str(a.dtype).split(".")[-1]

        def grid(meta):
            return (
                triton.cdiv(shape.m, meta["BLOCK_M"]) * triton.cdiv(shape.n, meta["BLOCK_N"]),
            )

        kernel_args = (
            a_desc,
            b_desc,
            c_desc,
            shape.m,
            shape.n,
            shape.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
        kernel_kwargs = {
            "grid": grid,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            "dtype": dtype_str,
            "warmup": False,
        }
        default_configs = list(mm_ops.matmul_get_configs())
        expand_name = "mm_general_tma"
    return tuner, kernel_args, kernel_kwargs, default_configs, expand_name


def make_bench(tuner: Any, kernel_args: Tuple[Any, ...], kernel_kwargs: Dict[str, Any], device: Any):
    from flag_gems.runtime import torch_device_fn

    calls = {"count": 0}

    def bench(config: Any) -> List[float]:
        calls["count"] += 1
        with torch_device_fn.device(device):
            ret = tuner._bench(*kernel_args, config=config, **kernel_kwargs)
        return list(ret)

    return bench, calls


def run_policy(
    label: str,
    tuner: Any,
    policy_fn: Any,
    configs: Iterable[Any],
    kernel_args: Tuple[Any, ...],
    kernel_kwargs: Dict[str, Any],
    device: Any,
) -> Dict[str, Any]:
    bench, calls = make_bench(tuner, kernel_args, kernel_kwargs, device)
    old_nargs = getattr(tuner, "nargs", None)
    tuner.nargs = dict(zip(tuner.arg_names, kernel_args))
    try:
        start = time.perf_counter()
        best_config, timings = policy_fn(bench, list(configs), kernel_args, kernel_kwargs)
        tuning_time_s = time.perf_counter() - start
        final_latency_ms = float(bench(best_config)[0])
    finally:
        tuner.nargs = old_nargs
    return {
        "mode": label,
        "status": "ok",
        "tuning_time_s": tuning_time_s,
        "latency_ms": final_latency_ms,
        "bench_count": calls["count"],
        "best_config": config_record(best_config),
        "timed_config_count": len(timings) if isinstance(timings, dict) else None,
    }


def run_xgb_only(
    tuner: Any,
    expanded_configs: List[Any],
    shape: Shape,
    kernel_args: Tuple[Any, ...],
    kernel_kwargs: Dict[str, Any],
    device: Any,
) -> Dict[str, Any]:
    from flag_gems.utils.libentry import _configs_to_dicts_for_proposer
    from triton.flagtune.predict import make_config_proposer
    from triton.flagtune.registry import get as get_op
    from triton.flagtune.registry import resolve_operator_id

    operator_id = resolve_operator_id(OP_ID)
    op_info = get_op(operator_id)
    proposer = make_config_proposer({"op_id": OP_ID})
    initial = _configs_to_dicts_for_proposer(
        expanded_configs, op_info.param_space.all_field_names
    )
    start = time.perf_counter()
    result_dicts = proposer(None, shape.shape_dict, initial, {"op_id": OP_ID})
    xgb_time_s = time.perf_counter() - start
    validate = getattr(op_info, "validate_shape_config", None)
    if validate is not None:
        result_dicts = [cfg for cfg in result_dicts if validate(shape.shape_dict, cfg)]
    if not result_dicts:
        raise RuntimeError(f"XGB-only proposer returned no valid configs for {shape}")

    bench, calls = make_bench(tuner, kernel_args, kernel_kwargs, device)
    best_config = None
    best_latency = float("inf")
    timings = {}
    failed_configs = 0
    old_nargs = getattr(tuner, "nargs", None)
    tuner.nargs = dict(zip(tuner.arg_names, kernel_args))
    start = time.perf_counter()
    try:
        for cfg_dict in result_dicts:
            cfg = op_info.to_config(cfg_dict)
            flagtune_pre_hook = getattr(tuner, "_flagtune_pre_hook", None)
            if cfg.pre_hook is None and flagtune_pre_hook is not None:
                # Keep XGB-only measurement faithful to LibTuner's FlagTune
                # path: TMA descriptors must update block_shape for each fresh
                # Config produced by the proposer.
                cfg.pre_hook = flagtune_pre_hook
            try:
                lat = float(bench(cfg)[0])
            except Exception:
                failed_configs += 1
                timings[cfg] = float("inf")
                continue
            timings[cfg] = lat
            if lat < best_latency:
                best_latency = lat
                best_config = cfg
        tuning_time_s = xgb_time_s + (time.perf_counter() - start)
    finally:
        tuner.nargs = old_nargs
    if best_config is None:
        raise RuntimeError(
            f"XGB-only proposer returned {len(result_dicts)} configs, "
            f"but all failed to benchmark for {shape}"
        )
    return {
        "mode": "flagtune_xgb_only",
        "status": "ok",
        "tuning_time_s": tuning_time_s,
        "xgb_time_s": xgb_time_s,
        "latency_ms": best_latency,
        "bench_count": calls["count"],
        "best_config": config_record(best_config),
        "timed_config_count": len(timings),
        "failed_config_count": failed_configs,
    }


def expanded_configs(expand_name: str):
    from flag_gems import runtime
    mm_ops = importlib.import_module("flag_gems.runtime.backend._nvidia.hopper.ops.mm")

    pre_hook = mm_ops.matmul_tma_set_block_size_hook if expand_name == "mm_general_tma" else None
    configs = runtime.ops_get_configs(
        expand_name,
        yaml_path=mm_ops.EXPAND_CONFIG_FILENAME,
        pre_hook=pre_hook,
    )
    if not configs:
        raise RuntimeError(f"No expanded configs for {expand_name}")
    return list(configs)


def sqlite_path_from_url_or_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    text = str(value)
    if text.startswith("sqlite:///"):
        text = text[len("sqlite:///") :]
    return Path(text).expanduser()


def normalize_dtype(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("torch."):
        text = text.split(".", 1)[1]
    return text


def best_from_db(db_path: Optional[Path], shape: Shape, dtype_name: str) -> Dict[str, Any]:
    if db_path is None:
        return {"db_status": "not_requested"}
    if not db_path.exists():
        return {"db_status": f"missing:{db_path}"}

    kernel_substr = "gemv_kernel" if shape.variant == "gemv" else "mm_kernel_general_host_tma"
    best: Optional[Dict[str, Any]] = None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if "benchmark" in row[0] and kernel_substr in row[0]
        ]
        for table in tables:
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info([{table}])")]
            if "p50" not in cols:
                continue
            for row in conn.execute(f"SELECT * FROM [{table}]"):
                row_dict = dict(row)
                try:
                    if shape.variant == "gemv":
                        matched = (
                            int(row_dict.get("key_0")) == shape.m
                            and int(row_dict.get("key_1")) == shape.k
                            and int(row_dict.get("key_2")) == shape.k
                            and int(row_dict.get("key_3")) == shape.n
                        )
                        row_dtype = normalize_dtype(row_dict.get("key_4", "unknown"))
                    else:
                        matched = (
                            int(row_dict.get("key_0")) == shape.m
                            and int(row_dict.get("key_1")) == shape.n
                            and int(row_dict.get("key_2")) == shape.k
                            and int(row_dict.get("key_3")) == shape.k
                            and int(row_dict.get("key_4")) == shape.n
                        )
                        row_dtype = normalize_dtype(row_dict.get("key_5", "unknown"))
                    dtype_matches = row_dtype in ("unknown", "none", "") or row_dtype == dtype_name
                    if not matched or not dtype_matches:
                        continue
                    latency = float(row_dict["p50"])
                except Exception:
                    continue
                config = {
                    k: row_dict[k]
                    for k in (
                        "BLOCK_M",
                        "BLOCK_N",
                        "BLOCK_K",
                        "GROUP_M",
                        "num_warps",
                        "num_stages",
                        "num_ctas",
                    )
                    if k in row_dict and row_dict[k] is not None
                }
                if best is None or latency < best["db_best_latency_ms"]:
                    best = {
                        "db_status": "ok",
                        "db_best_latency_ms": latency,
                        "db_best_config": config,
                        "db_table": table,
                    }
    finally:
        conn.close()
    return best or {"db_status": "no_match"}


def add_gap(result: Dict[str, Any], db_info: Dict[str, Any]) -> Dict[str, Any]:
    out = {**result, **db_info}
    db_lat = out.get("db_best_latency_ms")
    lat = out.get("latency_ms")
    if isinstance(db_lat, (int, float)) and db_lat > 0 and isinstance(lat, (int, float)):
        out["gap_vs_db_best_pct"] = (lat / db_lat - 1.0) * 100.0
    return out


def write_outputs(rows: List[Dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    fieldnames = sorted({key for row in rows for key in row})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.exclude_gemv:
        args.include_gemv = False
    if args.model_dir:
        os.environ["TRITON_FLAGTUNE_MODEL_DIR"] = args.model_dir
    os.environ["TRITON_FLAGTUNE_TOP_K"] = str(args.top_k)

    import torch

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)

    shapes = load_shapes(Path(args.shape_yaml), include_gemv=args.include_gemv)
    shapes = shapes[args.start_shape :]
    if args.max_shapes is not None:
        shapes = shapes[: args.max_shapes]

    db_info_by_shape = {}
    db_path = sqlite_path_from_url_or_path(args.benchmark_db)
    rows: List[Dict[str, Any]] = []

    try:
        import triton.flagtune as flagtune_mod

        flagtune_path = getattr(flagtune_mod, "__file__", "<unknown>")
    except Exception as exc:
        flagtune_path = f"unavailable:{exc}"

    print(
        json.dumps(
            {
                "shape_yaml": str(args.shape_yaml),
                "shape_count": len(shapes),
                "dtype": args.dtype,
                "op_id": OP_ID,
                "flagtune_path": flagtune_path,
                "model_dir": os.environ.get("TRITON_FLAGTUNE_MODEL_DIR"),
                "benchmark_db": str(db_path) if db_path else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for index, shape in enumerate(shapes):
        base = {
            "shape_index": index + args.start_shape,
            "B": shape.batch,
            "M": shape.m,
            "N": shape.n,
            "K": shape.k,
            "Count": shape.count,
            "variant": shape.variant,
            "dtype": args.dtype,
            "flagtune_op_id": OP_ID,
            "expand_op_name": shape.variant,
        }
        try:
            tensors = make_tensors(shape, args.dtype, args.device)
            tuner, kernel_args, kernel_kwargs, default_configs, expand_name = make_kernel_call(
                shape, tensors
            )
            expanded = expanded_configs(expand_name)
            db_info = best_from_db(db_path, shape, args.dtype)
            db_info_by_shape[index] = db_info

            default_result = run_policy(
                "default",
                tuner,
                lambda bench, configs, call_args, call_kwargs: __import__(
                    "flag_gems.utils.libentry", fromlist=["LibTuner"]
                ).LibTuner.get("default").policy(
                    tuner, bench, configs, call_args, call_kwargs
                ),
                default_configs,
                kernel_args,
                kernel_kwargs,
                device,
            )
            rows.append(add_gap({**base, **default_result}, db_info))

            xgb_result = run_xgb_only(
                tuner, expanded, shape, kernel_args, kernel_kwargs, device
            )
            rows.append(add_gap({**base, **xgb_result}, db_info))

            old_use_flagtune = os.environ.get("USE_FLAGTUNE")
            os.environ["USE_FLAGTUNE"] = "1"
            try:
                ga_result = run_policy(
                    "flagtune_xgb_ga",
                    tuner,
                    tuner.policy,
                    expanded,
                    kernel_args,
                    kernel_kwargs,
                    device,
                )
            finally:
                if old_use_flagtune is None:
                    os.environ.pop("USE_FLAGTUNE", None)
                else:
                    os.environ["USE_FLAGTUNE"] = old_use_flagtune
            if ga_result.get("bench_count", 0) < args.top_k:
                ga_result["status"] = "suspect_fallback_or_low_bench_count"
            rows.append(add_gap({**base, **ga_result}, db_info))

            print(
                f"[{index + 1}/{len(shapes)}] {shape.variant} "
                f"M={shape.m} N={shape.n} K={shape.k} done",
                flush=True,
            )
        except Exception as exc:
            err = {
                **base,
                "mode": "error",
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            rows.append(err)
            print(f"[ERROR] {base}: {exc!r}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break

    write_outputs(rows, Path(args.output_csv), Path(args.output_jsonl))
    print(f"Wrote {len(rows)} rows to {args.output_csv} and {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
