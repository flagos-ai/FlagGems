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

"""Command-line collection and XGBoost-ranker training for one FlagTune variant.

This offline tool benchmarks the complete or sampled runtime Expanded + Default
candidate space for one YAML variant, appends measurements to streaming JSONL,
performs one grouped XGBoost ranking fit, and exports a versioned self-contained archive
at ``<output>/<op_id>/<variant>/<dtype_key>/model.tar.gz`` for later platform
package assembly.

CLI arguments:
  * ``--shape-config``, ``--flagtune-config``, and ``--variant`` select the
    workload, combined operator contract, and exactly one model variant.
  * ``--model-version`` is the strict SemVer artifact revision; database, GPU,
    timing, batching, sampling, output, and XGBoost options control collection
    and fitting. See ``--help`` for the full parser and defaults.

Environment variables:
  * ``FLAGTREE_AABS`` is saved, forced to ``"0"`` during worker collection, and
    restored afterwards. This preserves raw configured combinations so their
    feature rows remain aligned; callers cannot override that behavior here.
  * ``FLAGTUNE_TRAIN_PROGRESS_INTERVAL`` is similarly saved, set from the
    progress option for worker status reporting, then restored. A non-negative
    value is expected by the executor.

Worker logs stream to the console. Successful runs remove worker files and the
streaming training corpus by default after exporting the model;
``--keep-intermediate-files`` preserves the former behavior for debugging. The
full corpus is not retained in parent memory, but XGBoost fitting is a single
global fit rather than incremental training. ``--max-configs-per-shape`` bounds
model training data. Workers bypass LibTuner's best-config cache but may reuse
per-config latency rows from ``--database``. BenchmarkCache v2 keys preserve
the exact raw shape, tensor dtypes, and warmup/repetition protocol, while the
separate best-config cache may still use shape-normalization strategies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from flag_gems.flagtune.cli.pretune import (  # noqa: E402
    PretuneError,
    environment_snapshot,
    load_shape_records,
    parse_max_shapes,
    parse_sort,
    planner_benchmark_shape,
    sanitize_db_url,
    select_shape_records,
    visible_device_tokens,
)
from flag_gems.flagtune.collection.scheduler import (  # noqa: E402
    DEFAULT_BENCHMARK_ITERATIONS_MS,
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_BENCHMARK_RETRIES,
    DEFAULT_BENCHMARK_WARMUP_MS,
    BenchmarkError,
    run_shape_config_benchmarks,
)
from flag_gems.flagtune.contracts.operator import (  # noqa: E402
    OperatorConfigError,
    initialize_planning_context,
    load_operator_benchmark_spec,
)
from flag_gems.flagtune.config_space import (  # noqa: E402
    runtime_configs_hash,
    runtime_configs_for_variant,
)
from flag_gems.flagtune.reporting.artifacts import (  # noqa: E402
    PretuneIOError,
    make_run_dir,
    remove_intermediate_artifacts,
    write_manifest,
)
from flag_gems.flagtune.reporting.schema import (  # noqa: E402
    SCHEMA_VERSION,
    pretune_json_row,
    rounded_ms,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "flagtune-train-output"


class TrainError(RuntimeError):
    """Report a user-facing collection or training failure."""


def _status(message: str) -> None:
    """Print one timestamped lifecycle message and flush it immediately.

    Args:
        message: Human-readable phase, path, count, or completion information.

    Output:
        A single stdout line suitable for interactive terminals and captured CI
        logs. Status messages remain enabled even when ``--no-progress`` hides
        fine-grained progress updates.
    """
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[FlagTune train {timestamp}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the public training CLI parser.

    Output:
        An ``ArgumentParser`` covering collection inputs, GPU concurrency,
        timing, progress, memory sampling, output, and XGBoost hyperparameters.

    Note:
        Semantic checks that depend on combinations of arguments are deferred to
        :func:`validate_args`; parsing alone does not initialize CUDA or FlagTune.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark a FlagTune parameter space and train XGB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shape-config", required=True, help="Shape config YAML.")
    parser.add_argument(
        "--flagtune-config",
        required=True,
        help="Multi-variant FlagTune training YAML containing --variant.",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Variant name selecting one model from --flagtune-config.",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="Strict SemVer 2.0 revision stored below the model identity path.",
    )
    parser.add_argument(
        "--database",
        help=(
            "SQLite per-config latency cache to reuse/fill; best-config entries "
            "are ignored. Defaults to a fresh <run-dir>/benchmark.db."
        ),
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
        help="Warmup milliseconds for each config measurement.",
    )
    parser.add_argument(
        "--iter",
        dest="iterations",
        type=int,
        default=DEFAULT_BENCHMARK_ITERATIONS_MS,
        help="Measurement milliseconds for each config.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("replay", "event"),
        default=DEFAULT_BENCHMARK_MODE,
        help="Architecture-neutral config timing mode.",
    )
    parser.add_argument(
        "--benchmark-retries",
        type=int,
        default=DEFAULT_BENCHMARK_RETRIES,
        help="Timed replay samples sharing the total --iter budget.",
    )
    parser.add_argument(
        "--max-shapes",
        type=parse_max_shapes,
        default=None,
        help=(
            "Maximum selected shapes as a positive integer or percentage, "
            "for example 50 or 50%%. Percentages are rounded down after "
            "variant filtering."
        ),
    )
    parser.add_argument(
        "--sort",
        dest="sort_text",
        default="default",
        help="default, count_ascending, count_descending, or random[=seed].",
    )
    parser.add_argument(
        "--shape-batch-size",
        type=int,
        default=None,
        help="Shapes per subprocess batch; default is four per active worker.",
    )
    parser.add_argument(
        "--max-configs-per-shape",
        type=int,
        default=None,
        help="Deterministically sample at most this many finite configs for XGB.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--keep-intermediate-files",
        action="store_true",
        help=(
            "Retain benchmark_data.jsonl, collection worker files, failure "
            "records, and a run-local benchmark database after model export."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Measured configs between worker progress lines; 0 disables them.",
    )

    training = parser.add_argument_group("XGBoost")
    training.add_argument("--n-estimators", type=int, default=1200)
    training.add_argument("--max-depth", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=0.03)
    training.add_argument("--subsample", type=float, default=0.95)
    training.add_argument("--colsample-bytree", type=float, default=0.95)
    training.add_argument("--reg-lambda", type=float, default=1.5)
    training.add_argument("--reg-alpha", type=float, default=0.0)
    training.add_argument("--min-child-weight", type=float, default=1.0)
    training.add_argument("--gamma", type=float, default=0.0)
    training.add_argument("--max-bin", type=int, default=512)
    training.add_argument("--n-jobs", type=int, default=4)
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--min-train-rows", type=int, default=8)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI values without creating output files or GPU workers.

    Args:
        args: Namespace returned by :func:`build_parser`.

    Raises:
        TrainError: If a numeric bound, probability, or model-ID path is unsafe
            or inconsistent.

    Limitation:
        Config-dependent validation happens later in :func:`run_main`; this
        lightweight phase does not import CUDA or parse the training YAML.
    """
    if not str(args.flagtune_config).strip():
        raise TrainError("--flagtune-config must be non-empty")
    if args.parallel is not None and args.parallel <= 0:
        raise TrainError("--parallel must be a positive integer")
    if args.warmup < 0:
        raise TrainError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise TrainError("--iter must be positive")
    if args.benchmark_retries <= 0:
        raise TrainError("--benchmark-retries must be positive")
    for name in (
        "shape_batch_size",
        "n_estimators",
        "max_depth",
        "max_bin",
        "n_jobs",
        "min_train_rows",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise TrainError(f"--{name.replace('_', '-')} must be positive")
    if args.max_configs_per_shape is not None and args.max_configs_per_shape < 2:
        raise TrainError("--max-configs-per-shape must be at least 2")
    if args.progress_interval < 0:
        raise TrainError("--progress-interval must be non-negative")
    for name in ("learning_rate", "subsample", "colsample_bytree"):
        value = float(getattr(args, name))
        if value <= 0:
            raise TrainError(f"--{name.replace('_', '-')} must be positive")
    if args.subsample > 1 or args.colsample_bytree > 1:
        raise TrainError("--subsample and --colsample-bytree must not exceed 1")
    if not args.variant.strip() or "/" in args.variant:
        raise TrainError("--variant must be a non-empty single-segment name")
    try:
        from triton.flagtune.contract.archive import validate_model_version

        validate_model_version(args.model_version)
    except (ImportError, ValueError) as exc:
        raise TrainError(str(exc)) from exc


def _progress(total: int, enabled: bool) -> Any:
    """Create a shape-level progress reporter with optional tqdm support.

    Args:
        total: Total selected shape count.
        enabled: Whether shape progress should be visible.

    Returns:
        An object exposing ``update(count)`` and ``close()``. When disabled, the
        returned null reporter performs no I/O. Without tqdm, an enabled reporter
        emits plain text after each completed shape batch.
    """
    if not enabled:

        class _NullProgress:
            """Implement the progress interface without producing output."""

            def update(self, count: int) -> None:
                """Accept a completed count and intentionally do nothing."""
                del count

            def close(self) -> None:
                """Close the null reporter without side effects."""
                return None

        return _NullProgress()

    if enabled:
        try:
            from tqdm.auto import tqdm

            return tqdm(total=total, desc="Benchmark shapes", unit="shape")
        except ImportError:
            pass

    class _FallbackProgress:
        """Report completed shape batches through flushed text lines."""

        def __init__(self) -> None:
            """Initialize the completed-shape counter at zero."""
            self.done = 0

        def update(self, count: int) -> None:
            """Add completed shapes and print the current total."""
            self.done += count
            print(f"Benchmark shapes: {self.done}/{total}", flush=True)

        def close(self) -> None:
            """Close the fallback reporter; no external resource is held."""
            return None

    return _FallbackProgress()


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield non-empty, order-preserving slices of at most ``size`` values.

    The caller validates ``size`` as positive. Slices retain references to the
    original objects, limiting parent memory while benchmark payloads are built.
    """
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _grouped_chunks(
    values: Sequence[Any], size: int, key: Any
) -> Iterable[Sequence[Any]]:
    """Yield batches that never split equal ranking groups.

    FlagTree's ranker requires every serialized ``ranking_group`` to occupy one
    contiguous run in ``benchmark_data.jsonl``.  Collection workers are
    intentionally batched for bounded memory, so ordinary positional chunks
    can split duplicate normalized shapes across batches.  Group records first
    and pack whole groups into batches; an individual group larger than
    ``size`` is emitted as one oversized batch rather than being split.
    """
    groups: dict[Any, list[Any]] = {}
    order: list[Any] = []
    for value in values:
        group_key = key(value)
        if group_key not in groups:
            groups[group_key] = []
            order.append(group_key)
        groups[group_key].append(value)

    batch: list[Any] = []
    for group_key in order:
        group = groups[group_key]
        if batch and len(batch) + len(group) > size:
            yield batch
            batch = []
        if len(group) > size:
            if batch:
                yield batch
                batch = []
            yield group
            continue
        batch.extend(group)
    if batch:
        yield batch


def _collection_group_key(
    record: Any,
    spec: Any,
    variant_info: Any,
    platform_key: Optional[str],
) -> str:
    """Build the final ranker group identity before collection batching."""
    payload = planner_benchmark_shape(record, spec, platform_key)
    values = payload.get("values") if isinstance(payload, Mapping) else None
    if not isinstance(values, Mapping):
        values = payload
    try:
        dimensions = variant_info.normalize_inputs(values)
    except Exception:
        # The executor will report the detailed schema error.  Keeping a stable
        # fallback key here still prevents byte-identical records from being
        # split across batches before that validation runs.
        dimensions = dict(values) if isinstance(values, Mapping) else repr(values)
    route = payload.get("route") if isinstance(payload, Mapping) else None
    if not isinstance(route, Mapping):
        route = {}
    physical_variant = str(record.variant or variant_info.name)
    route_variant = route.get("route_variant") or physical_variant
    tuning_variant = route.get("tuning_variant") or physical_variant
    stage = route.get("stage") or (
        "partial" if str(tuning_variant).endswith("_partial") else "public"
    )
    latency_scope = (
        "partial_kernel" if stage == "partial" else "public_kernel"
    )
    return json.dumps(
        {
            "operator_id": variant_info.op_id,
            "variant": tuning_variant,
            "route_variant": route_variant,
            "stage": stage,
            "latency_scope": latency_scope,
            "dimensions": dimensions,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )


def _database_url(args: argparse.Namespace, run_dir: Path) -> tuple[str, Path, str]:
    """Resolve the collection SQLite URL, absolute path, and provenance label.

    An explicit ``--database`` may be relative to the current working directory.
    Otherwise a fresh ``benchmark.db`` is placed inside the timestamped run,
    avoiding stale best-config cache entries from earlier training attempts.
    """
    if args.database:
        path = Path(args.database).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return f"sqlite:///{path}", path, "--database"
    path = (run_dir / "benchmark.db").resolve()
    return f"sqlite:///{path}", path, "run directory"


def _append_collection_rows(
    data_path: Path,
    failure_path: Path,
    results: Sequence[Mapping[str, Any]],
    variant_info: Any,
    shape_fields: Sequence[str],
    expected_config_count: Optional[int] = None,
) -> tuple[int, int, int]:
    """Validate and append flattened per-config records to streaming JSONL.

    Args:
        data_path: Append-only training JSONL destination.
        failure_path: Append-only shape-level failure JSONL destination.
        results: Ordered generic executor rows from one benchmark batch.
        variant_info: Registered FlagTune variant used to normalize inputs.
        shape_fields: Operator-defined workload identity fields.
        expected_config_count: Backward-compatible fallback for legacy worker
            rows that predate ``timed_config_count``.
    Returns:
        ``(written_rows, finite_latency_rows, failed_shapes)``.

    Implementation and pitfalls:
        Rows are written one at a time with strict JSON finite-number handling.
        A shape with missing or collapsed config timings is written only to the
        failure file, preventing silent feature/target misalignment.
    """
    written = 0
    finite = 0
    failed_shapes = 0
    # Worker results arrive in completion order.  The FlagTree ranker consumes
    # JSONL groups as contiguous runs.  Sort by the exact structured group key
    # (rather than source index alone) because distinct source rows can share
    # normalized model inputs, e.g. duplicate Count/layout recipes.
    def result_order_key(result: Mapping[str, Any]) -> tuple[int, str, int]:
        try:
            shape_values = result.get("model_inputs")
            if not isinstance(shape_values, Mapping):
                shape_values = {
                    name: result[name]
                    for name in variant_info.input_names
                    if name in result
                }
            inputs = variant_info.normalize_inputs(shape_values)
            expected_scope = (
                "partial_kernel"
                if result.get("stage") == "partial"
                or str(result.get("tuning_variant", "")).endswith("_partial")
                or str(result.get("variant", "")).endswith("_partial")
                else "public_kernel"
            )
            group = {
                "operator_id": variant_info.op_id,
                "variant": result.get("tuning_variant") or variant_info.name,
                "route_variant": result.get("route_variant") or variant_info.name,
                "stage": result.get("stage") or expected_scope.removesuffix("_kernel"),
                "latency_scope": expected_scope,
                "dimensions": inputs,
                "model_dtype_key": result.get("dtype_key"),
            }
            group_key = json.dumps(group, sort_keys=True, separators=(",", ":"))
        except Exception:
            group_key = ""
        value = result.get("source_index", result.get("task_index"))
        try:
            source_index = int(value) if value is not None else 0
        except (TypeError, ValueError):
            source_index = 0
        return (0 if group_key else 1, group_key, source_index)

    ordered_results = sorted(results, key=result_order_key)
    with (
        data_path.open("a", encoding="utf-8") as data_file,
        failure_path.open("a", encoding="utf-8") as failure_file,
    ):
        for result in ordered_results:
            if result.get("status") == "skipped":
                continue
            missing_shape_fields = [name for name in shape_fields if name not in result]
            workload_dimensions = {
                name: result[name] for name in shape_fields if name in result
            }
            if missing_shape_fields:
                failed = pretune_json_row(result, shape_fields)
                failed["collection_error"] = (
                    "missing workload dimensions: " + ", ".join(missing_shape_fields)
                )
                failure_file.write(json.dumps(failed, sort_keys=True, allow_nan=False))
                failure_file.write("\n")
                failed_shapes += 1
                continue
            timings = result.get("config_timings")
            if result.get("status") != "ok" or not isinstance(timings, list):
                failure_file.write(
                    json.dumps(
                        pretune_json_row(result, shape_fields),
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                failure_file.write("\n")
                failed_shapes += 1
                continue
            expected_scope = (
                "partial_kernel"
                if result.get("stage") == "partial"
                or str(result.get("tuning_variant", "")).endswith("_partial")
                or str(result.get("variant", "")).endswith("_partial")
                else "public_kernel"
            )
            if result.get("latency_scope") not in {None, expected_scope}:
                failed = pretune_json_row(result, shape_fields)
                failed["collection_error"] = "latency_scope does not match tuning stage"
                failure_file.write(json.dumps(failed, sort_keys=True, allow_nan=False) + "\n")
                failed_shapes += 1
                continue
            if any(timing.get("latency_scope") not in {None, expected_scope} for timing in timings):
                failed = pretune_json_row(result, shape_fields)
                failed["collection_error"] = "config timing latency_scope mismatch"
                failure_file.write(json.dumps(failed, sort_keys=True, allow_nan=False) + "\n")
                failed_shapes += 1
                continue
            timed_config_count = result.get("timed_config_count")
            if timed_config_count is None:
                timed_config_count = (
                    expected_config_count
                    if expected_config_count is not None
                    else len(timings)
                )
            if (
                not isinstance(timed_config_count, int)
                or timed_config_count <= 0
                or len(timings) != timed_config_count
            ):
                failed = pretune_json_row(result, shape_fields)
                failed["collection_error"] = (
                    f"timed_config_count={timed_config_count!r} does not match "
                    f"{len(timings)} config timings"
                )
                failure_file.write(json.dumps(failed, sort_keys=True, allow_nan=False))
                failure_file.write("\n")
                failed_shapes += 1
                continue
            shape_values = result.get("model_inputs")
            if not isinstance(shape_values, Mapping):
                shape_values = {
                    name: result[name]
                    for name in variant_info.input_names
                    if name in result
                }
            try:
                inputs = variant_info.normalize_inputs(shape_values)
            except Exception as exc:
                failed = pretune_json_row(result, shape_fields)
                failed["collection_error"] = f"cannot normalize training inputs: {exc}"
                failure_file.write(json.dumps(failed, sort_keys=True, allow_nan=False))
                failure_file.write("\n")
                failed_shapes += 1
                continue
            ranking_group = {
                "operator_id": variant_info.op_id,
                "variant": result.get("tuning_variant") or variant_info.name,
                "route_variant": result.get("route_variant") or variant_info.name,
                "stage": result.get("stage") or expected_scope.removesuffix("_kernel"),
                "latency_scope": expected_scope,
                "dimensions": inputs,
                "model_dtype_key": result["dtype_key"],
            }
            model_dtype_values = result.get("model_dtypes") or [
                *result["input_dtypes"],
                *result["output_dtypes"],
            ]
            model_input_count = len(result["input_dtypes"])
            for config_order, timing in enumerate(timings):
                serialized_timing = {
                    name: (rounded_ms(value) if name.endswith("_ms") else value)
                    for name, value in dict(timing).items()
                }
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "input_row_index": result.get("source_index"),
                    "operator": {
                        "id": variant_info.op_id,
                        "variant": variant_info.name,
                    },
                    "workload": {
                        "dimensions": workload_dimensions,
                        "Count": result.get("Count"),
                    },
                    "ranking_group": ranking_group,
                    "Count": result.get("Count"),
                    "dtypes": {
                        # FlagTree validates model_identity.dtype_key from this
                        # field.  For partial kernels the third role is the
                        # FP32 P workspace, so dtypes must describe A/B/P.
                        "inputs": model_dtype_values[:model_input_count],
                        "outputs": model_dtype_values[model_input_count:],
                    },
                    # Public output dtype and model identity are deliberately
                    # separate for partial kernels: C is the public BF16
                    # output, while the partial model's third role is the
                    # FP32 P workspace.
                    "model_dtypes": result.get("model_dtypes")
                    or [
                        *result["input_dtypes"],
                        *result["output_dtypes"],
                    ],
                    "public_dtypes": {
                        "inputs": result["input_dtypes"],
                        "outputs": result["output_dtypes"],
                    },
                    "model_identity": {
                        "platform_key": result.get("platform_key"),
                        "dtype_key": result["dtype_key"],
                    },
                    "device": {
                        "gpu_token": result.get("gpu"),
                        "name": result.get("gpu_name"),
                        "metadata": result.get("gpu_metadata"),
                    },
                    "benchmark_protocol": result.get("benchmark_protocol"),
                    "inputs": inputs,
                    "config_order": config_order,
                    **serialized_timing,
                }
                data_file.write(
                    json.dumps(
                        row, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                )
                data_file.write("\n")
                written += 1
                if timing.get("latency_ms") is not None:
                    finite += 1
    return written, finite, failed_shapes


def _training_options(args: argparse.Namespace) -> Any:
    """Translate CLI hyperparameters into FlagTree's typed training options.

    Importing the FlagTree training module is delayed until collection planning
    is complete, keeping ``--help`` and early argument errors lightweight.
    """
    from triton.flagtune.training.ranker import XGBoostTrainingOptions

    return XGBoostTrainingOptions(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        max_bin=args.max_bin,
        n_jobs=args.n_jobs,
        seed=args.seed,
        min_train_rows=args.min_train_rows,
        max_configs_per_shape=args.max_configs_per_shape,
        show_progress=not args.no_progress,
    )


def run_main(args: argparse.Namespace) -> int:
    """Execute registry validation, GPU collection, XGBoost fit, and export.

    Args:
        args: Validated or parser-produced CLI namespace.

    Returns:
        Zero after all selected shapes, model files, and the manifest complete.

    Raises:
        TrainError: For config mismatch, GPU/worker failure, incomplete data,
            database merge failure, or model training/export errors.

    Implementation:
        Shapes are selected before output creation. Collection is split into
        bounded batches, with AABS disabled only in inherited worker environments.
        The environment is restored before the global XGBoost ranking fit.
    """
    validate_args(args)
    _status("Loading shape config and compiling the FlagTune training config")
    try:
        config_path = Path(args.flagtune_config).expanduser().resolve()
        spec = load_operator_benchmark_spec(config_path)
        operator_info = spec.operator_info
        variant_info = operator_info.get_variant(args.variant)
        op_id = operator_info.op_id
        requested_variant = variant_info.name
        source_records = load_shape_records(
            Path(args.shape_config).expanduser().resolve(), spec
        )
        context, runtime_operator_info = initialize_planning_context(spec)
    except (
        ImportError,
        KeyError,
        OSError,
        PretuneError,
        OperatorConfigError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise TrainError(str(exc)) from exc
    if runtime_operator_info.op_id != op_id:
        raise TrainError(
            f"shape op_id {op_id!r} does not match compiled "
            f"op_id {runtime_operator_info.op_id!r}"
        )
    operation_id = f"{op_id}/{requested_variant}"
    sort_spec = parse_sort(args.sort_text)
    try:
        selected = select_shape_records(
            source_records,
            spec,
            requested_variant,
            sort_spec,
            args.max_shapes,
            {
                "platform_key": getattr(context, "vendor_name", "unknown"),
                "dtypes": args.dtypes,
                "planner_output": True,
            },
        )
    except PretuneError as exc:
        raise TrainError(str(exc)) from exc
    if context.visible_device_count <= 0:
        raise TrainError("no visible devices")
    parallel = args.parallel or context.visible_device_count
    if parallel > context.visible_device_count:
        raise TrainError(
            f"--parallel {parallel} exceeds "
            f"{context.visible_device_count} visible devices"
        )
    workers = min(parallel, len(selected))
    gpu_tokens = visible_device_tokens(context)[:workers]
    runtime_configs = runtime_configs_for_variant(
        op_id,
        requested_variant,
        platform=getattr(context, "vendor_name", None),
    )
    if runtime_configs is None:
        raise TrainError(
            f"{operation_id} has no runtime candidate resolver; "
            "runtime YAML is authoritative"
        )
    if not runtime_configs:
        raise TrainError(
            f"{operation_id} runtime Expanded + Default config space is empty"
        )
    configs = runtime_configs
    if not configs:
        raise TrainError(f"{operation_id} has an empty parameter space")
    runtime_candidate_hash = (
        runtime_configs_hash(configs) if runtime_configs is not None else None
    )
    shape_batch_size = args.shape_batch_size or max(workers, workers * 4)
    maximum_expected_rows = len(selected) * len(configs)

    plan = {
        "dry_run": bool(args.dry_run),
        "shape_config": str(Path(args.shape_config).expanduser().resolve()),
        "flagtune_config": str(Path(args.flagtune_config).expanduser().resolve()),
        "flagtune_config_sha256": spec.source_sha256,
        "op_id": op_id,
        "variant": requested_variant,
        "model_version": args.model_version,
        "selected_shape_count": len(selected),
        "max_shapes": args.max_shapes,
        "config_count_per_shape": len(configs),
        "config_source": (
            "runtime_expanded_plus_default"
            if runtime_configs is not None
            else "contract"
        ),
        "runtime_config_sha256": runtime_candidate_hash,
        "maximum_benchmark_row_count": maximum_expected_rows,
        "feature_count": len(variant_info.feature_names),
        "estimated_dense_float32_bytes": maximum_expected_rows
        * len(variant_info.feature_names)
        * 4,
        "max_configs_per_shape": args.max_configs_per_shape,
        "parallel": workers,
        "gpu_tokens": gpu_tokens,
        "backend": context.backend_name,
        "device_names": list(context.device_names),
        "device_architectures": list(context.device_architectures),
        "shape_batch_size": shape_batch_size,
        "progress_interval": args.progress_interval,
        "dtypes": args.dtypes,
        "warmup": args.warmup,
        "iter": args.iterations,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_retries": args.benchmark_retries,
        "flagtree_aabs": False,
        "tuning_run_mode": "exhaustive_collection",
        "sort": sort_spec.mode,
        "random_seed": sort_spec.seed,
        "output_root": str(Path(args.output).expanduser().resolve()),
        "keep_intermediate_files": bool(args.keep_intermediate_files),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    run_dir = make_run_dir(Path(args.output).expanduser().resolve(), operation_id)
    database_url, database_path, database_source = _database_url(args, run_dir)
    data_path = run_dir / "benchmark_data.jsonl"
    failure_path = run_dir / "benchmark_failures.jsonl"
    started_at = datetime.now().astimezone().isoformat()
    start = time.perf_counter()
    collection_rows = 0
    expected_rows = 0
    finite_rows = 0
    failed_shapes = 0
    skipped_shapes = 0
    benchmark_cache_hits = 0
    benchmark_successes = 0
    benchmark_protocols: dict[str, Mapping[str, Any]] = {}
    batch_manifests = []
    platform_keys: set[str] = set()
    dtype_keys: set[str] = set()
    ordered_dtypes: Optional[list[str]] = None
    model_dtypes: Optional[list[str]] = None
    gpu_metadata: Optional[Mapping[str, Any]] = None
    _status(
        f"Run directory: {run_dir}; shapes={len(selected)}; "
        f"configs/shape={len(configs)}; workers={workers}"
    )
    _status(f"Collection database: {sanitize_db_url(database_url)}")
    progress = _progress(len(selected), not args.no_progress)
    previous_aabs = os.environ.get("FLAGTREE_AABS")
    previous_progress_interval = os.environ.get("FLAGTUNE_TRAIN_PROGRESS_INTERVAL")
    # Training features describe the exact configured parameter combination.
    # AABS mutates block sizes per shape and can collapse several raw configs
    # into one timing-map key, so exhaustive offline collection disables it in
    # worker subprocesses.  Normal Pretune and runtime behavior are untouched.
    os.environ["FLAGTREE_AABS"] = "0"
    os.environ["FLAGTUNE_TRAIN_PROGRESS_INTERVAL"] = str(
        0 if args.no_progress else args.progress_interval
    )
    try:
        grouped_batch_key = lambda record: _collection_group_key(  # noqa: E731
            record,
            spec,
            variant_info,
            getattr(context, "vendor_name", None),
        )
        for batch_index, records in enumerate(
            _grouped_chunks(selected, shape_batch_size, grouped_batch_key)
        ):
            batch_dir = run_dir / "collection_batches" / f"batch_{batch_index:05d}"
            batch_workers = min(workers, len(records))
            _status(
                f"Starting collection batch {batch_index + 1}: "
                f"shapes={len(records)}, workers={batch_workers}"
            )
            try:
                batch = run_shape_config_benchmarks(
                    [
                        (
                            planner_benchmark_shape(
                                record,
                                spec,
                                getattr(context, "vendor_name", None),
                            ),
                            configs,
                        )
                        for record in records
                    ],
                    operator_config=config_path,
                    dtypes=args.dtypes,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    benchmark_mode=args.benchmark_mode,
                    benchmark_retries=args.benchmark_retries,
                    tuning_run_mode="exhaustive_collection",
                    parallel=batch_workers,
                    gpu_tokens=gpu_tokens[:batch_workers],
                    database_url=database_url,
                    work_dir=batch_dir,
                    fail_fast=args.fail_fast,
                    stream_worker_logs=not args.no_progress,
                )
            except BenchmarkError as exc:
                raise TrainError(str(exc)) from exc
            for result in batch.results:
                if result.get("status") != "ok":
                    continue
                protocol = result.get("benchmark_protocol")
                if isinstance(protocol, Mapping):
                    benchmark_protocols[json.dumps(protocol, sort_keys=True)] = dict(
                        protocol
                    )
                platform_keys.add(str(result.get("platform_key")))
                current_gpu_metadata = result.get("gpu_metadata")
                dtype_keys.add(str(result.get("dtype_key")))
                current_dtypes = [
                    *result.get("input_dtypes", []),
                    *result.get("output_dtypes", []),
                ]
                current_model_dtypes = list(
                    result.get("model_dtypes") or current_dtypes
                )
                if ordered_dtypes is None:
                    ordered_dtypes = current_dtypes
                    model_dtypes = current_model_dtypes
                    gpu_metadata = current_gpu_metadata
                elif current_dtypes != ordered_dtypes:
                    raise TrainError(
                        "collection produced inconsistent ordered input/output dtypes"
                    )
                elif current_model_dtypes != model_dtypes:
                    raise TrainError(
                        "collection produced inconsistent model dtype identities"
                    )
                elif current_gpu_metadata != gpu_metadata:
                    raise TrainError("collection mixed platform/architecture metadata")
                if len(platform_keys) != 1:
                    raise TrainError(
                        f"collection mixed platform identities: {sorted(platform_keys)}"
                    )
                if len(dtype_keys) != 1:
                    raise TrainError(
                        f"collection mixed dtype identities: {sorted(dtype_keys)}"
                    )
            written, finite, failures = _append_collection_rows(
                data_path,
                failure_path,
                batch.results,
                variant_info,
                spec.shape.identity,
            )
            expected_rows += written
            collection_rows += written
            finite_rows += finite
            failed_shapes += failures + max(0, len(records) - len(batch.results))
            batch_skipped = sum(
                result.get("status") == "skipped" for result in batch.results
            )
            skipped_shapes += batch_skipped
            batch_cache_hits = sum(
                int(result.get("benchmark_cache_hit_count") or 0)
                for result in batch.results
            )
            batch_successes = sum(
                int(result.get("benchmark_success_count") or 0)
                for result in batch.results
            )
            benchmark_cache_hits += batch_cache_hits
            benchmark_successes += batch_successes
            batch_manifests.append(
                {
                    "batch_index": batch_index,
                    "shape_count": len(records),
                    "result_count": len(batch.results),
                    "worker_returncodes": batch.worker_returncodes,
                    "database_merge": batch.database_merge,
                    "database_merge_error": batch.database_merge_error,
                    "data_rows": written,
                    "finite_rows": finite,
                    "failed_shapes": failures,
                    "skipped_shapes": batch_skipped,
                    "cached_count": batch_cache_hits,
                    "measured_count": batch_successes,
                }
            )
            progress.update(len(records))
            _status(
                f"Finished collection batch {batch_index + 1}: "
                f"rows={written}, finite={finite}, failed_shapes={failures}, "
                f"skipped_shapes={batch_skipped}, "
                f"latency_cache_hits={batch_cache_hits}, "
                f"new_finite_benchmarks={batch_successes}"
            )
            if args.fail_fast and (
                failures
                or any(code != 0 for code in batch.worker_returncodes)
                or batch.database_merge_error
            ):
                break
    finally:
        progress.close()
        if previous_aabs is None:
            os.environ.pop("FLAGTREE_AABS", None)
        else:
            os.environ["FLAGTREE_AABS"] = previous_aabs
        if previous_progress_interval is None:
            os.environ.pop("FLAGTUNE_TRAIN_PROGRESS_INTERVAL", None)
        else:
            os.environ["FLAGTUNE_TRAIN_PROGRESS_INTERVAL"] = previous_progress_interval

    if failed_shapes:
        raise TrainError(
            f"collection failed for {failed_shapes} shapes; inspect {failure_path}"
        )
    if collection_rows != expected_rows:
        raise TrainError(
            f"collection produced {collection_rows} rows, expected {expected_rows}"
        )

    try:
        from triton.flagtune.contract.identity import ModelIdentity
        from triton.flagtune.contract.operator_schema import model_config_sha256
        from triton.flagtune.training.ranker import (
            export_ranker_model,
            train_xgboost_ranker,
        )

        options = _training_options(args)
        _status(
            f"Starting XGBoost ranking fit: finite_rows={finite_rows}, "
            f"trees={args.n_estimators}, max_depth={args.max_depth}"
        )
        model, training_summary = train_xgboost_ranker(variant_info, data_path, options)
        training_summary = {
            **training_summary,
            "benchmark_protocols": list(benchmark_protocols.values()),
            "runtime_config_sha256": runtime_candidate_hash,
        }
        if (
            not platform_keys
            or not dtype_keys
            or ordered_dtypes is None
            or model_dtypes is None
            or gpu_metadata is None
        ):
            raise TrainError(
                "collection produced no successful platform/dtype identity"
            )
        identity = ModelIdentity(
            platform_key=next(iter(platform_keys)),
            op_id=op_id,
            variant=requested_variant,
            dtype_key=next(iter(dtype_keys)),
        )
        exported = export_ranker_model(
            model,
            variant_info,
            run_dir,
            training_summary,
            identity=identity,
            dtypes=model_dtypes,
            gpu=gpu_metadata,
            model_version=args.model_version,
        )
        _status(
            f"XGBoost fit and export finished in "
            f"{training_summary['xgboost_fit_elapsed_s']:.2f}s"
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise TrainError(f"XGBoost training failed: {exc}") from exc

    intermediate_paths = [
        run_dir / "collection_batches",
        data_path,
        failure_path,
    ]
    if database_source == "run directory":
        intermediate_paths.append(database_path)
    cleanup_error = ""
    removed_intermediate_files: list[str] = []
    if not args.keep_intermediate_files:
        try:
            removed_intermediate_files = remove_intermediate_artifacts(
                run_dir, intermediate_paths
            )
        except PretuneIOError as exc:
            cleanup_error = str(exc)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "dry_run": False,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "wall_time_s": time.perf_counter() - start,
            "run_dir": str(run_dir),
            "output_root": plan["output_root"],
        },
        "inputs": {
            "shape_config": plan["shape_config"],
            "flagtune_config": plan["flagtune_config"],
            "flagtune_config_sha256": plan["flagtune_config_sha256"],
            "dtypes": args.dtypes,
            "warmup": args.warmup,
            "iter": args.iterations,
            "benchmark_mode": args.benchmark_mode,
            "benchmark_retries": args.benchmark_retries,
            "database": sanitize_db_url(database_url),
            "database_path": str(database_path),
            "database_source": database_source,
            "strategy_environment": environment_snapshot(),
            "flagtree_aabs": False,
            "tuning_run_mode": "exhaustive_collection",
        },
        "selection": {
            "op_id": op_id,
            "variant": requested_variant,
            "selected_shape_count": len(selected),
            "config_count_per_shape": len(configs),
            "config_source": plan["config_source"],
            "contract_config_count": None,
            "runtime_config_count": (
                len(runtime_configs) if runtime_configs is not None else None
            ),
            "runtime_config_sha256": runtime_candidate_hash,
            "expected_benchmark_row_count": expected_rows,
            "max_configs_per_shape": args.max_configs_per_shape,
            "feature_count": len(variant_info.feature_names),
            "estimated_dense_float32_bytes": plan["estimated_dense_float32_bytes"],
            "sort": sort_spec.mode,
            "random_seed": sort_spec.seed,
        },
        "benchmark_summary": {
            "collection_row_count": collection_rows,
            "finite_collection_row_count": finite_rows,
            "failed_shape_count": failed_shapes,
            "skipped_shape_count": skipped_shapes,
            "cached_count": benchmark_cache_hits,
            "measured_count": benchmark_successes,
            "resolved_protocols": list(benchmark_protocols.values()),
            "parallel": workers,
            "gpu_tokens": gpu_tokens,
            "shape_batch_size": shape_batch_size,
            "progress_interval": args.progress_interval,
            "collection_batches": batch_manifests,
        },
        "model": {
            "path": str(exported.model_path),
            "version": args.model_version,
            "config": exported.model_config,
            "config_sha256": model_config_sha256(exported.model_config),
            "training_summary": training_summary,
        },
        "artifacts": {
            "benchmark_data": str(data_path) if data_path.exists() else None,
            "benchmark_failure_data": (
                str(failure_path) if failure_path.exists() else None
            ),
        },
        "retention": {
            "keep_intermediate_files": bool(args.keep_intermediate_files),
            "intermediate_files_retained": bool(
                args.keep_intermediate_files or cleanup_error
            ),
            "removed_intermediate_files": removed_intermediate_files,
            "cleanup_error": cleanup_error,
        },
    }
    write_manifest(run_dir / "manifest.json", manifest)
    if cleanup_error:
        raise TrainError(cleanup_error)
    print(f"FlagTune training output: {run_dir}")
    print(f"Model: {exported.model_path}")
    return 0


def main() -> int:
    """Parse CLI arguments, run training, and convert user errors to exit code 2."""
    args = build_parser().parse_args()
    _status("Starting FlagTune offline training")
    try:
        return run_main(args)
    except (PretuneError, TrainError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
