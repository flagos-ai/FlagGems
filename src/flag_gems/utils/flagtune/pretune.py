#!/usr/bin/env python3
"""Command-line orchestration for offline FlagGems Pretune runs.

The command validates a combined FlagTree/FlagGems operator YAML, selects
workload rows and variants, then delegates execution to the generic multi-GPU
benchmark scheduler. It is the collection entry point, not a model trainer or
policy switcher.

CLI arguments:
  * ``--shape-config``, ``--flagtune-config``, and ``--op`` select workload
    rows, the safe operator contract, and ``operator[/variant]``.
  * ``--database``, ``--parallel``, and ``--dtypes`` control latency storage,
    workers, and runtime dtypes. ``--warmup``/``--iter`` control candidate
    config search; the ``--latency-*`` options control the independent fresh
    measurement of the selected config.
  * ``--sort`` and ``--max-shapes`` select records after variant filtering;
    ``--output``, ``--fail-fast``, and ``--dry-run`` control artifacts/execution.

Environment variables:
  * ``FLAGGEMS_DB_URL`` is the inherited database URL when ``--database`` is
    absent; ``FLAGGEMS_CACHE_DIR`` chooses the default SQLite cache directory.
  * Backend visibility variables such as ``CUDA_VISIBLE_DEVICES``,
    ``ROCR_VISIBLE_DEVICES``, and ``HIP_VISIBLE_DEVICES`` limit visible device
    tokens and are captured in the run manifest.
  * ``USE_FLAGTUNE`` and ``FLAGTUNE_INCLUDE`` are FlagGems-local policy inputs.
  * ``TRITON_USE_FLAGTUNE``, ``FLAGTUNE_DISABLE_OPS``,
    ``TRITON_FLAGTUNE_TOP_K``, ``TRITON_FLAGTUNE_MODEL_DIR``,
    ``FLAGTUNE_MODEL_CACHE``, and ``FLAGTUNE_DISABLE_REMOTE`` are passed to
    FlagTree's proposer/model-resolution path unchanged and recorded for audit.

Dry runs print a JSON plan and create nothing. Real runs create timestamped
CSV, JSONL, manifest, and combined-log artifacts while LibTuner uses the chosen
database. Successful runs remove worker task/result/log files by default;
``--keep-intermediate-files`` retains them. Selection order is variant
filtering, sorting, then ``max_shapes``. The CLI cannot clear/resume databases
or compare modes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from flag_gems.utils.flagtune.benchmark import (
    BenchmarkError,
    DEFAULT_BENCHMARK_ITERATIONS_MS,
    DEFAULT_BENCHMARK_WARMUP_MS,
    DEFAULT_LATENCY_ITERATIONS_MS,
    DEFAULT_LATENCY_TRIALS,
    DEFAULT_LATENCY_WARMUP_MS,
    parse_sqlite_url,
    run_shape_config_benchmarks,
)
from flag_gems.utils.flagtune.operator_config import (
    OperatorBenchmarkSpec,
    OperatorConfigError,
    initialize_planning_context,
    load_operator_benchmark_spec,
)
from flag_gems.utils.flagtune.records import PlanningContext, ShapeRecord
from flag_gems.utils.flagtune.pretune_io import (
    PretuneIOError,
    combine_logs,
    load_shape_config,
    make_run_dir,
    remove_intermediate_artifacts,
    write_manifest,
    write_outputs,
)
from flag_gems.utils.flagtune.output_schema import SCHEMA_VERSION

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "flagtune-pretune-output"
STRATEGY_ENV_NAMES = (
    "USE_FLAGTUNE",
    "FLAGTUNE_INCLUDE",
    "TRITON_USE_FLAGTUNE",
    "FLAGTUNE_DISABLE_OPS",
    "TRITON_FLAGTUNE_TOP_K",
    "TRITON_FLAGTUNE_MODEL_DIR",
    "FLAGTUNE_MODEL_CACHE",
    "FLAGTUNE_DISABLE_REMOTE",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
)


class PretuneError(RuntimeError):
    """Report a user-facing planning, validation, or execution error."""


@dataclass(frozen=True)
class SortSpec:
    """Describe shape selection order and the resolved random seed, if any."""

    mode: str
    seed: Optional[int] = None


@dataclass(frozen=True)
class DatabasePlan:
    """Describe the resolved database URL, backend kind, path, and source."""

    url: str
    kind: str
    sqlite_path: Optional[Path]
    source: str


def build_parser() -> argparse.ArgumentParser:
    """Build the public Pretune argument parser and its documented defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Pretune FlagGems shapes with the policy selected by the current "
            "FlagGems/FlagTree environment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shape-config", required=True, help="Shape config YAML.")
    parser.add_argument(
        "--flagtune-config",
        required=True,
        help="Data-driven FlagTune operator and Pretune YAML.",
    )
    parser.add_argument("--op", required=True, help="operator or operator/variant.")
    parser.add_argument(
        "--database",
        help="SQLite database file. Overrides inherited FLAGGEMS_DB_URL.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="GPU worker count; defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--dtypes",
        default="bfloat16",
        help="Comma-separated input tensor dtypes; one value broadcasts.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_BENCHMARK_WARMUP_MS,
        help="Warmup milliseconds for each candidate config measurement.",
    )
    parser.add_argument(
        "--iter",
        dest="iterations",
        type=int,
        default=DEFAULT_BENCHMARK_ITERATIONS_MS,
        help="Measurement milliseconds for each candidate config.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=DEFAULT_LATENCY_WARMUP_MS,
        help="Fresh selected-config LibTuner warmup milliseconds per trial.",
    )
    parser.add_argument(
        "--latency-iter",
        dest="latency_iterations",
        type=int,
        default=DEFAULT_LATENCY_ITERATIONS_MS,
        help="Fresh selected-config LibTuner measurement milliseconds per trial.",
    )
    parser.add_argument(
        "--latency-trials",
        type=int,
        default=DEFAULT_LATENCY_TRIALS,
        help="Independent fresh selected-config LibTuner trials.",
    )
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument(
        "--sort",
        dest="sort_text",
        default="default",
        help="default, count_ascending, count_descending, or random[=seed].",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--keep-intermediate-files",
        action="store_true",
        help="Retain benchmark worker task, result, and log files.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid numeric CLI options before runtime initialization."""
    if args.parallel is not None and args.parallel <= 0:
        raise PretuneError("--parallel must be a positive integer")
    if args.warmup < 0:
        raise PretuneError("--warmup must be a non-negative integer")
    if args.iterations <= 0:
        raise PretuneError("--iter must be a positive integer")
    if args.latency_warmup < 0:
        raise PretuneError("--latency-warmup must be a non-negative integer")
    if args.latency_iterations <= 0:
        raise PretuneError("--latency-iter must be a positive integer")
    if args.latency_trials <= 0:
        raise PretuneError("--latency-trials must be a positive integer")
    if args.max_shapes is not None and args.max_shapes <= 0:
        raise PretuneError("--max-shapes must be a positive integer")


def parse_op(text: str) -> tuple[str, Optional[str]]:
    """Split ``operator[/variant]`` while rejecting empty or extra segments."""
    parts = [part.strip() for part in text.split("/")]
    if len(parts) not in (1, 2) or not all(parts):
        raise PretuneError("--op must be operator or operator/<variant>")
    return parts[0], parts[1] if len(parts) == 2 else None


def parse_sort(text: str) -> SortSpec:
    """Parse a selection-order expression and resolve an optional random seed.

    An unseeded ``random`` uses ``SystemRandom`` and returns the actual seed so
    it can be persisted in the manifest and reused.
    """
    if text in ("default", "count_ascending", "count_descending"):
        return SortSpec(text)
    if text == "random":
        return SortSpec("random", random.SystemRandom().getrandbits(64))
    match = re.fullmatch(r"random=(-?[0-9]+)", text)
    if match:
        return SortSpec("random", int(match.group(1)))
    raise PretuneError(
        "--sort must be default, count_ascending, count_descending, "
        "random, or random=<integer seed>"
    )


def load_shape_records(path: Path, spec: OperatorBenchmarkSpec) -> list[ShapeRecord]:
    """Load generic YAML and validate rows through the compiled shape schema.

    Args:
        path: Shape-config YAML path.
        spec: Compiled data-driven operator contract.

    Returns:
        Validated source-order records without resolved variants.

    Raises:
        PretuneError: If generic parsing or shape validation fails.
    """

    try:
        shape_config = load_shape_config(path, spec.public_operator_name)
        return spec.shape.build_records(shape_config)
    except (PretuneIOError, OperatorConfigError, RuntimeError, ValueError) as exc:
        raise PretuneError(str(exc)) from exc


def select_shape_records(
    records: Sequence[ShapeRecord],
    spec: OperatorBenchmarkSpec,
    requested_variant: Optional[str],
    sort_spec: SortSpec,
    max_shapes: Optional[int],
) -> list[ShapeRecord]:
    """Resolve variants, filter, sort, limit, and index shape records.

    Count ordering requires every retained record to define ``Count``.  Random
    order is deterministic for the seed stored in ``sort_spec``.  Runtime task
    order remains independent because workers consume the selected list in
    round-robin chunks.
    """
    operator_info = spec.operator_info
    if (
        requested_variant is not None
        and requested_variant not in operator_info.variants
    ):
        raise PretuneError(
            f"unknown {operator_info.op_id} variant {requested_variant!r}; "
            f"registered variants: {sorted(operator_info.variants)}"
        )

    selected = []
    for record in records:
        try:
            variant = spec.resolve_variant(record.values)
        except (RuntimeError, ValueError) as exc:
            raise PretuneError(str(exc)) from exc
        if requested_variant is None or variant == requested_variant:
            selected.append(replace(record, variant=variant))
    if not selected:
        suffix = f"/{requested_variant}" if requested_variant else ""
        raise PretuneError(
            f"no shapes remain after filtering for {operator_info.op_id}{suffix}"
        )

    if sort_spec.mode.startswith("count_"):
        if any(record.count is None for record in selected):
            raise PretuneError(
                f"--sort {sort_spec.mode} requires Count for every selected shape"
            )
        selected.sort(
            key=lambda record: int(record.count),
            reverse=sort_spec.mode == "count_descending",
        )
    elif sort_spec.mode == "random":
        random.Random(sort_spec.seed).shuffle(selected)
    if max_shapes is not None:
        selected = selected[:max_shapes]
    return [
        replace(record, selected_index=index) for index, record in enumerate(selected)
    ]


def _sqlite_url(path: Path) -> str:
    """Convert a resolved filesystem path to a file-backed SQLite URL."""
    return f"sqlite:///{path}"


def default_database_url(
    args: argparse.Namespace, context: PlanningContext
) -> DatabasePlan:
    """Resolve database precedence without opening or modifying the backend.

    Precedence is ``--database``, inherited ``FLAGGEMS_DB_URL``, then the
    vendor/Triton-specific FlagGems SQLite cache path.
    """
    if args.database:
        path = Path(args.database).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return DatabasePlan(_sqlite_url(path), "sqlite", path, "--database")
    inherited = os.environ.get("FLAGGEMS_DB_URL")
    if inherited:
        kind, path = parse_sqlite_url(inherited)
        return DatabasePlan(inherited, kind, path, "FLAGGEMS_DB_URL")
    cache_root = Path(os.environ.get("FLAGGEMS_CACHE_DIR", Path.home() / ".flaggems"))
    path = (
        cache_root.expanduser()
        / "config_cache"
        / (
            f"TunedConfig_{context.vendor_name}_triton_"
            f"{context.triton_major}_{context.triton_minor}.db"
        )
    ).resolve()
    return DatabasePlan(_sqlite_url(path), "sqlite", path, "FlagGems default")


def visible_device_tokens(context: PlanningContext) -> list[str]:
    """Return launcher tokens validated by the central device preflight."""
    if len(context.device_tokens) != context.visible_device_count:
        raise PretuneError(
            "device preflight returned inconsistent token and device counts"
        )
    return list(context.device_tokens)


def sanitize_db_url(url: str) -> str:
    """Redact a non-SQLite database password before logging or serialization."""
    if url.startswith("sqlite"):
        return url
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "<non-sqlite database>"
    if parsed.password is None:
        return url
    user = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{user}:***@{host}{port}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def environment_snapshot() -> dict[str, Optional[str]]:
    """Capture policy-related environment values without changing them."""
    return {name: os.environ.get(name) for name in STRATEGY_ENV_NAMES}


def dry_run_summary(
    args: argparse.Namespace,
    records: Sequence[ShapeRecord],
    context: PlanningContext,
    database: DatabasePlan,
    sort_spec: SortSpec,
    gpu_tokens: Sequence[str],
) -> dict[str, Any]:
    """Build the JSON-safe execution plan printed by ``--dry-run``."""
    workers = min(args.parallel or context.visible_device_count, len(records))
    return {
        "dry_run": True,
        "project_root": str(PROJECT_ROOT),
        "shape_config": str(Path(args.shape_config).expanduser().resolve()),
        "flagtune_config": str(Path(args.flagtune_config).expanduser().resolve()),
        "op": args.op,
        "selected_shapes": len(records),
        "variants": {
            name: sum(record.variant == name for record in records)
            for name in context.operator_variants
        },
        "sort": sort_spec.mode,
        "random_seed": sort_spec.seed,
        "dtypes": args.dtypes,
        "warmup": args.warmup,
        "iter": args.iterations,
        "tuning_run_mode": "force_policy",
        "latency_warmup": args.latency_warmup,
        "latency_iter": args.latency_iterations,
        "latency_trials": args.latency_trials,
        "parallel": workers,
        "backend": context.backend_name,
        "gpu_tokens": list(gpu_tokens[:workers]),
        "device_names": list(context.device_names),
        "device_architectures": list(context.device_architectures),
        "worker_shape_counts": [
            len(range(worker_id, len(records), workers)) for worker_id in range(workers)
        ],
        "database": sanitize_db_url(database.url),
        "database_source": database.source,
        "output_root": str(Path(args.output).expanduser().resolve()),
        "keep_intermediate_files": bool(args.keep_intermediate_files),
        "strategy_environment": environment_snapshot(),
    }


def run_main(args: argparse.Namespace) -> int:
    """Plan and optionally execute a Pretune invocation.

    Args:
        args: Parsed arguments from :func:`build_parser`.

    Returns:
        ``0`` for a successful dry or real run and ``1`` when real execution
        produced failed/missing rows, worker failures, or a shard-merge failure.

    Raises:
        PretuneError: For invalid planning inputs or a batch setup failure.

    Side effects:
        Real runs create artifacts, launch GPU worker subprocesses, and may add
        database rows.  Dry runs perform dependency and runtime/registry
        discovery but do not create output directories or write the database.
    """
    validate_args(args)
    operator_name, requested_variant = parse_op(args.op)
    try:
        config_path = Path(args.flagtune_config).expanduser().resolve()
        spec = load_operator_benchmark_spec(config_path)
    except (OperatorConfigError, OSError, ValueError) as exc:
        raise PretuneError(str(exc)) from exc
    if operator_name != spec.public_operator_name:
        raise PretuneError(
            f"--op name {operator_name!r} does not match public operator "
            f"{spec.public_operator_name!r} derived from {spec.op_id!r}"
        )
    sort_spec = parse_sort(args.sort_text)
    shape_path = Path(args.shape_config).expanduser().resolve()
    source_records = load_shape_records(shape_path, spec)
    try:
        context, operator_info = initialize_planning_context(spec)
    except (OperatorConfigError, RuntimeError, ValueError) as exc:
        raise PretuneError(str(exc)) from exc
    if spec.op_id != operator_info.op_id:
        raise PretuneError(
            f"config op_id {spec.op_id!r} does not match FlagTree op_id "
            f"{operator_info.op_id!r}"
        )
    selected = select_shape_records(
        source_records,
        spec,
        requested_variant,
        sort_spec,
        args.max_shapes,
    )
    if context.visible_device_count <= 0:
        raise PretuneError("no visible devices")
    parallel = args.parallel or context.visible_device_count
    if parallel > context.visible_device_count:
        raise PretuneError(
            f"--parallel {parallel} exceeds "
            f"{context.visible_device_count} visible devices"
        )
    workers = min(parallel, len(selected))
    tokens = visible_device_tokens(context)
    database = default_database_url(args, context)

    if args.dry_run:
        print(
            json.dumps(
                dry_run_summary(args, selected, context, database, sort_spec, tokens),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    run_dir = make_run_dir(Path(args.output).expanduser().resolve(), args.op)
    started_at = datetime.now().astimezone().isoformat()
    start = time.perf_counter()
    try:
        batch = run_shape_config_benchmarks(
            [(record.to_benchmark_shape(), None) for record in selected],
            operator_config=config_path,
            dtypes=args.dtypes,
            warmup=args.warmup,
            iterations=args.iterations,
            tuning_run_mode="force_policy",
            latency_warmup=args.latency_warmup,
            latency_iterations=args.latency_iterations,
            latency_trials=args.latency_trials,
            parallel=workers,
            gpu_tokens=tokens[:workers],
            database_url=database.url,
            work_dir=run_dir,
            fail_fast=args.fail_fast,
        )
    except BenchmarkError as exc:
        raise PretuneError(str(exc)) from exc
    rows = batch.results
    failed_rows = sum(row.get("status") != "ok" for row in rows)
    missing_rows = len(selected) - len(rows)
    write_outputs(run_dir, rows, spec.shape.identity)
    cached_count = sum(
        int(row.get("benchmark_cache_hit_count") or 0) for row in rows
    )
    measured_count = sum(
        int(row.get("benchmark_success_count") or 0) for row in rows
    )
    failed = (
        failed_rows > 0
        or missing_rows > 0
        or any(code != 0 for code in batch.worker_returncodes)
        or batch.database_merge.get("status") == "failed"
    )
    combine_logs(
        run_dir,
        batch.worker_log_paths,
        [
            f"run_dir={run_dir}",
            f"shape_config={shape_path}",
            f"op={args.op}",
            f"database={sanitize_db_url(database.url)}",
            f"workers={workers}",
            f"worker_returncodes={batch.worker_returncodes}",
            f"failed_rows={failed_rows}",
            f"missing_rows={missing_rows}",
            f"database_merge={json.dumps(batch.database_merge, sort_keys=True)}",
        ],
    )
    cleanup_error = ""
    removed_intermediate_files: list[str] = []
    if not args.keep_intermediate_files and not failed:
        try:
            removed_intermediate_files = remove_intermediate_artifacts(
                run_dir,
                [
                    run_dir / "benchmark-workers",
                    run_dir / "database-shards",
                ],
            )
        except PretuneIOError as exc:
            cleanup_error = str(exc)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "wall_time_s": time.perf_counter() - start,
            "project_root": str(PROJECT_ROOT),
            "script": str(SCRIPT_PATH),
            "run_dir": str(run_dir),
        },
        "inputs": {
            "shape_config": str(shape_path),
            "flagtune_config": str(config_path),
            "flagtune_config_sha256": spec.source_sha256,
            "dtypes": args.dtypes,
            "warmup": args.warmup,
            "iter": args.iterations,
            "tuning_run_mode": "force_policy",
            "latency_warmup": args.latency_warmup,
            "latency_iter": args.latency_iterations,
            "latency_trials": args.latency_trials,
            "database": sanitize_db_url(database.url),
            "database_source": database.source,
            "database_kind": database.kind,
            "strategy_environment": environment_snapshot(),
        },
        "selection": {
            "source_shape_count": len(source_records),
            "selected_shape_count": len(selected),
            "op": args.op,
            "op_id": spec.op_id,
            "public_op_name": operator_name,
            "requested_variant": requested_variant,
            "variant_counts": {
                name: sum(record.variant == name for record in selected)
                for name in context.operator_variants
            },
            "sort": sort_spec.mode,
            "random_seed": sort_spec.seed,
            "max_shapes": args.max_shapes,
        },
        "benchmark_summary": {
            "result_row_count": len(rows),
            "failed_row_count": failed_rows,
            "missing_row_count": missing_rows,
            "cached_count": cached_count,
            "measured_count": measured_count,
            "parallel": workers,
            "backend": context.backend_name,
            "visible_device_count": context.visible_device_count,
            "gpu_tokens": tokens[:workers],
            "gpu_names": list(context.device_names),
            "device_architectures": list(context.device_architectures),
            "database_merge": batch.database_merge,
            "database_merge_error": batch.database_merge_error,
            "database_shards": [str(path) for path in batch.database_shards],
            "worker_returncodes": batch.worker_returncodes,
            "fail_fast": args.fail_fast,
            "fail_fast_triggered": batch.fail_fast_triggered,
        },
        "artifacts": {
            "pretune_csv": str(run_dir / "pretune.csv"),
            "pretune_jsonl": str(run_dir / "pretune.jsonl"),
            "pretune_log": str(run_dir / "pretune.log"),
        },
        "retention": {
            "keep_intermediate_files": bool(args.keep_intermediate_files),
            "intermediate_files_retained": bool(
                args.keep_intermediate_files or failed or cleanup_error
            ),
            "removed_intermediate_files": removed_intermediate_files,
            "cleanup_error": cleanup_error,
            "retained_because_run_failed": bool(failed),
        },
    }
    write_manifest(run_dir / "manifest.json", manifest)
    print(f"Pretune output: {run_dir}")
    if batch.database_merge_error:
        print(batch.database_merge_error, file=sys.stderr)
    if cleanup_error:
        print(cleanup_error, file=sys.stderr)
    return 1 if failed or cleanup_error else 0


def main() -> int:
    """CLI entry point that converts :class:`PretuneError` to exit code 2."""
    args = build_parser().parse_args()
    try:
        return run_main(args)
    except PretuneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
