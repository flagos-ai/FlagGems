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

"""Re-measure two Pretune runs' selected configs in one interleaved process.

Pretune writes ``latency_p50_ms`` from a fresh measurement taken inside the
process that tuned the shape.  Comparing a baseline run against an ours run
therefore compares two processes separated by the training stage, and at small
shape sizes that between-process error term is larger than the effect under
test: on a T-Head PPU, rows whose two runs had selected the *same* config still
reported 82.6%-108.8% relative throughput.

This command removes that error term.  It reads both Pretune CSV artifacts,
takes the config each run selected, and measures both configs for every shape
back to back in one process, alternating which side is measured first on every
trial.  Only the latency columns of the two inputs are rewritten, so
``compare.py`` consumes the result unchanged.

CLI arguments:
  * ``--baseline`` and ``--ours`` are the Pretune CSV artifacts to re-measure.
    ``--shape-config`` and ``--flagtune-config`` must be the ones both runs used;
    rows are joined to shape records by the original input row index.
  * ``--trials`` sets measurements per config per sweep and ``--repeats`` sets
    how many independent sweeps run.  The reported latency is the median over
    sweeps of the median over trials.
  * ``--latency-warmup``, ``--latency-iter``, ``--benchmark-mode``, and
    ``--benchmark-retries`` set the Triton timing protocol.
  * ``--noise-floor-ms`` marks shapes too small for that protocol to resolve.
    Marked rows are reported separately rather than dropped.
  * ``--regression-threshold-pct`` selects which rows the summary lists.  A
    listed row is only reported as a confirmed regression when its shortfall
    also exceeds its own spread across sweeps, because a row that moves more
    between sweeps than it falls below the baseline has not been shown to
    regress at all.

Artifacts are ``baseline_pretune.csv``, ``ours_pretune.csv``, and
``rebench_summary.json`` under ``--output``.  Rows that either run did not
complete are passed through untouched and counted as skipped.

Unlike Pretune this command is deliberately single-process and single-device.
Interleaving both sides in one process is the entire point, so the sweep cannot
be spread over workers, and Pretune's per-worker device pinning is unavailable.
``--device`` therefore restricts this process to one device before the FlagGems
runtime is imported; importing it later caches the visible device count and the
restriction stops taking effect.  An already-set backend visibility variable is
authoritative and is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# Importing anything under ``flag_gems`` runs its package initializer, which
# probes the backend and caches the visible device count.  Every such import is
# therefore deferred until ``main`` has pinned this process to one device.

DEFAULT_DEVICE = "0"
VISIBILITY_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "MACA_VISIBLE_DEVICES",
    "MUSA_VISIBLE_DEVICES",
)
DEFAULT_TRIALS = 9
DEFAULT_REPEATS = 3
DEFAULT_NOISE_FLOOR_MS = 0.002
DEFAULT_REGRESSION_THRESHOLD_PCT = 95.0
LATENCY_SOURCE = "libtuner_interleaved_rebench"
SIDES = ("baseline", "ours")
REQUIRED_COLUMNS = (
    "input_row_index",
    "variant",
    "status",
    "best_config",
    "latency_p20_ms",
    "latency_p50_ms",
    "latency_p80_ms",
    "latency_source",
    "latency_warmup_ms",
    "latency_measurement_ms",
    "latency_trials",
)


class RebenchError(RuntimeError):
    """Report a user-facing input, join, or measurement failure."""


def _pin_single_device(token: str) -> str:
    """Restrict this process to one device and describe the active setting.

    Args:
        token: Launcher device token to expose when nothing is pinned yet.

    Returns:
        A ``NAME=value`` description of the visibility setting now in force,
        used in diagnostics.

    Notes:
        A non-empty backend visibility variable already present in the
        environment is authoritative and is left untouched, so a caller that
        pinned a specific device keeps it.  Otherwise the CUDA-family variable
        is set; on a backend with its own variable, export that variable before
        invoking this command.

    Limitations:
        This must run before the FlagGems runtime is imported.  The backend
        caches its visible device count during package initialization, and a
        later change to the environment no longer has any effect.
    """
    for name in VISIBILITY_VARIABLES:
        current = str(os.environ.get(name, "")).strip()
        if current:
            return f"{name}={current}"
    os.environ[VISIBILITY_VARIABLES[0]] = str(token)
    return f"{VISIBILITY_VARIABLES[0]}={token}"


def build_parser() -> argparse.ArgumentParser:
    """Build the public re-measurement CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-measure two Pretune runs' selected configs interleaved in one "
            "process."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", required=True, help="Baseline pretune.csv.")
    parser.add_argument("--ours", required=True, help="Ours pretune.csv.")
    parser.add_argument(
        "--shape-config", required=True, help="Shape YAML both runs used."
    )
    parser.add_argument(
        "--flagtune-config", required=True, help="Operator YAML both runs used."
    )
    parser.add_argument("--output", required=True, help="Artifact directory.")
    parser.add_argument(
        "--dtypes",
        default="bfloat16",
        help="Comma-separated input tensor dtypes; one value broadcasts.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Measurements per config per sweep.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Independent sweeps over every shape.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=(
            "Launcher device token this process is restricted to. Ignored when "
            "a backend visibility variable is already set."
        ),
    )
    parser.add_argument("--latency-warmup", type=int, default=25)
    parser.add_argument(
        "--latency-iter", dest="latency_iterations", type=int, default=100
    )
    parser.add_argument(
        "--benchmark-mode", choices=("event", "replay"), default="replay"
    )
    parser.add_argument("--benchmark-retries", type=int, default=10)
    parser.add_argument(
        "--noise-floor-ms",
        type=float,
        default=DEFAULT_NOISE_FLOOR_MS,
        help="Baseline latency at or below this is unresolvable by the protocol.",
    )
    parser.add_argument(
        "--regression-threshold-pct",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD_PCT,
        help="Relative throughput below this is listed in the summary.",
    )
    return parser


def _read_pretune_csv(path: Path, label: str) -> tuple[list[dict[str, str]], list[str]]:
    """Read one Pretune CSV and verify the columns this command rewrites."""
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise RebenchError(f"cannot read {label} CSV {path}: {exc}") from exc
    if not rows:
        raise RebenchError(f"{label} CSV {path} has no rows")
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise RebenchError(
            f"{label} CSV {path} is missing required columns: {missing}. "
            "Re-measurement needs schema v3 Pretune output."
        )
    return rows, fieldnames


def _index_rows(
    rows: Sequence[Mapping[str, str]], label: str
) -> dict[int, dict[str, str]]:
    """Index Pretune rows by their original input row index."""
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            index = int(row["input_row_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RebenchError(
                f"{label} CSV has a row with an unusable input_row_index"
            ) from exc
        if index in indexed:
            raise RebenchError(f"{label} CSV repeats input_row_index {index}")
        indexed[index] = dict(row)
    return indexed


def _selected_config(row: Mapping[str, str], label: str, index: int) -> dict[str, Any]:
    """Parse one run's selected config into a flattened config record."""
    text = (row.get("best_config") or "").strip()
    if not text:
        raise RebenchError(f"{label} row {index} has no best_config")
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RebenchError(
            f"{label} row {index} has an unparsable best_config: {exc}"
        ) from exc
    if not isinstance(config, Mapping) or not config:
        raise RebenchError(f"{label} row {index} best_config is not a config mapping")
    return dict(config)


def _resolve_dtypes(dtypes: str, tensor_count: int) -> list[str]:
    """Normalize and broadcast requested dtypes to the tensor-recipe count."""
    from triton.flagtune.contract.identity import normalize_dtype_name

    requested = [item.strip() for item in dtypes.split(",") if item.strip()]
    if not requested:
        raise RebenchError("--dtypes must contain at least one dtype")
    try:
        requested = [normalize_dtype_name(item) for item in requested]
    except ValueError as exc:
        raise RebenchError(str(exc)) from exc
    if len(requested) == 1:
        return requested * tensor_count
    if len(requested) != tensor_count:
        raise RebenchError(
            f"--dtypes has {len(requested)} values but benchmark.tensors has "
            f"{tensor_count} inputs"
        )
    return requested


def _sweep_quantiles(
    trials: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Reduce one sweep's trials to a median p20/p50/p80 triple."""
    return tuple(statistics.median(values) for values in zip(*trials))


def _reduce_sweeps(
    sweeps: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Reduce per-sweep quantiles to the reported median triple."""
    return tuple(statistics.median(values) for values in zip(*sweeps))


def _rewrite_latency(
    row: Mapping[str, str],
    quantiles: tuple[float, float, float],
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, str]:
    """Copy one Pretune row with its latency columns replaced."""
    from flag_gems.flagtune.reporting.schema import format_ms

    updated = dict(row)
    p20, p50, p80 = quantiles
    updated["latency_p20_ms"] = format_ms(p20)
    updated["latency_p50_ms"] = format_ms(p50)
    updated["latency_p80_ms"] = format_ms(p80)
    updated["latency_source"] = LATENCY_SOURCE
    updated["latency_warmup_ms"] = str(warmup)
    updated["latency_measurement_ms"] = str(iterations)
    updated["latency_trials"] = str(trials)
    return updated


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> None:
    """Write one rewritten Pretune CSV preserving the input column order."""
    try:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
    except OSError as exc:
        raise RebenchError(f"cannot write {path}: {exc}") from exc


def _count_of(row: Mapping[str, str]) -> Optional[int]:
    """Return one row's workload Count, or ``None`` when it is unavailable."""
    try:
        count = int(row["Count"])
    except (KeyError, TypeError, ValueError):
        return None
    return count if count > 0 else None


def measure_all(
    *,
    spec: Any,
    config_path: Path,
    device_runtime: Any,
    records_by_index: Mapping[int, Any],
    joined: Mapping[int, Mapping[str, Mapping[str, str]]],
    dtype_names: Sequence[str],
    trials: int,
    repeats: int,
    warmup: int,
    iterations: int,
    benchmark_mode: str,
    benchmark_retries: int,
) -> dict[int, dict[str, list[tuple[float, float, float]]]]:
    """Run every sweep and return per-row, per-side, per-sweep quantiles.

    Args:
        spec: Compiled operator contract shared by both Pretune runs.
        config_path: Operator YAML path the worker reloads.
        device_runtime: Runtime already probed and validated by ``main`` for the
            single pinned device.
        records_by_index: Shape records keyed by original input row index.
        joined: Measurable rows keyed by index, each mapping side to its row.
        dtype_names: Resolved ordered tensor-recipe dtype names.
        trials: Measurements per config per sweep.
        repeats: Independent sweeps over every shape.
        warmup: Triton warmup duration in milliseconds.
        iterations: Triton repetition duration in milliseconds.
        benchmark_mode: Architecture-neutral ``event`` or ``replay`` mode.
        benchmark_retries: Replay samples sharing each trial's budget.

    Returns:
        ``{index: {side: [ (p20, p50, p80) per sweep ]}}``.

    Raises:
        RebenchError: If the worker cannot be constructed or a shape cannot be
        prepared or measured.

    Implementation:
        One :class:`BenchmarkWorker` serves every sweep so tuner state and
        compiled kernels are reused. Sweeps are the outer loop rather than the
        inner one: a per-shape repeat would sample the same few seconds of
        machine state three times, while a repeated full sweep re-samples each
        shape minutes apart and is what actually exposes drift.
    """
    from flag_gems.flagtune.contracts.operator import OperatorConfigError
    from flag_gems.flagtune.runtime.device import DeviceProbeError
    from flag_gems.flagtune.runtime.executor import (
        BenchmarkExecutionError,
        BenchmarkWorker,
        prepare_benchmark_case,
    )

    try:
        worker = BenchmarkWorker(str(config_path), device_runtime=device_runtime)
    except (BenchmarkExecutionError, OperatorConfigError, ImportError) as exc:
        raise RebenchError(f"cannot start the benchmark worker: {exc}") from exc

    payloads: dict[int, Any] = {}
    config_records: dict[int, dict[str, dict[str, Any]]] = {}
    for index in sorted(joined):
        record = records_by_index.get(index)
        if record is None:
            raise RebenchError(
                f"input_row_index {index} is absent from the shape config; "
                "the CSVs and --shape-config are not from the same run"
            )
        try:
            payloads[index] = prepare_benchmark_case(
                spec, record.to_benchmark_shape(), None, index
            )
        except BenchmarkExecutionError as exc:
            raise RebenchError(f"row {index}: {exc}") from exc
        config_records[index] = {
            side: _selected_config(joined[index][side], side, index) for side in SIDES
        }

    collected: dict[int, dict[str, list[tuple[float, float, float]]]] = {
        index: {side: [] for side in SIDES} for index in payloads
    }
    total = len(payloads) * repeats
    done = 0
    start = time.perf_counter()
    for sweep in range(repeats):
        for index in sorted(payloads):
            try:
                samples = worker.measure_configs(
                    payloads[index],
                    dtype_names=dtype_names,
                    config_records=config_records[index],
                    warmup=warmup,
                    iterations=iterations,
                    trials=trials,
                    benchmark_mode=benchmark_mode,
                    benchmark_retries=benchmark_retries,
                )
            except (BenchmarkExecutionError, DeviceProbeError) as exc:
                raise RebenchError(f"row {index} sweep {sweep}: {exc}") from exc
            for side in SIDES:
                collected[index][side].append(_sweep_quantiles(samples[side]))
            done += 1
            if done % 25 == 0 or done == total:
                elapsed = time.perf_counter() - start
                print(
                    f"[rebench] {done}/{total} measurements  {elapsed:.0f}s",
                    flush=True,
                )
    return collected


def summarize(
    *,
    collected: Mapping[int, Mapping[str, Sequence[tuple[float, float, float]]]],
    joined: Mapping[int, Mapping[str, Mapping[str, str]]],
    skipped: Sequence[int],
    sweeps: int,
    noise_floor_ms: float,
    threshold_pct: float,
) -> dict[str, Any]:
    """Build the machine-readable re-measurement summary.

    Args:
        collected: Per-row, per-side, per-sweep quantiles from :func:`measure_all`.
        joined: Measurable rows keyed by index, each mapping side to its row.
        skipped: Indexes carried through without measurement.
        sweeps: Number of sweeps behind each row, which decides whether a row
            has an error bar at all.
        noise_floor_ms: Baseline latency at or below this is unresolvable.
        threshold_pct: Relative throughput below this is listed.

    Returns:
        A JSON-compatible mapping with one entry per measured row and the
        aggregate ratios that are the intended acceptance metric.

    Notes:
        ``sweep_spread_pct`` is the range of a row's own relative throughput
        across sweeps and serves as its error bar. ``regression_confirmed``
        requires the shortfall below the threshold to exceed that range, so a
        row that moves more between sweeps than it falls behind is listed but
        not counted. With few sweeps the range is a coarse, deliberately
        conservative interval; more sweeps tighten it. A single sweep has no
        range to compare against, so ``regression_confirmed`` is ``None`` rather
        than trivially true for every row below the threshold.
    """
    from flag_gems.flagtune.reporting.schema import rounded_derived, rounded_ms

    confirmable = sweeps >= 2
    rows: list[dict[str, Any]] = []
    for index in sorted(collected):
        per_side = {side: _reduce_sweeps(collected[index][side]) for side in SIDES}
        baseline_p50 = per_side["baseline"][1]
        ours_p50 = per_side["ours"][1]
        relative = baseline_p50 / ours_p50 * 100.0
        sweep_relatives = [
            base[1] / ours[1] * 100.0
            for base, ours in zip(
                collected[index]["baseline"], collected[index]["ours"]
            )
        ]
        source = joined[index]["baseline"]
        spread = max(sweep_relatives) - min(sweep_relatives)
        below_threshold = relative < threshold_pct
        rows.append(
            {
                "input_row_index": index,
                "variant": source.get("variant", ""),
                "Count": _count_of(source),
                "baseline_latency_p50_ms": rounded_ms(baseline_p50),
                "ours_latency_p50_ms": rounded_ms(ours_p50),
                "relative_throughput_pct": rounded_derived(relative),
                "sweep_relative_throughput_pct": [
                    rounded_derived(value) for value in sweep_relatives
                ],
                "sweep_spread_pct": rounded_derived(spread),
                "same_config": (
                    joined[index]["baseline"].get("best_config")
                    == joined[index]["ours"].get("best_config")
                ),
                "below_noise_floor": baseline_p50 <= noise_floor_ms,
                "below_threshold": below_threshold,
                "regression_confirmed": (
                    (below_threshold and (threshold_pct - relative) > spread)
                    if confirmable
                    else None
                ),
            }
        )

    equal_baseline = sum(row["baseline_latency_p50_ms"] for row in rows)
    equal_ours = sum(row["ours_latency_p50_ms"] for row in rows)
    weighted = [row for row in rows if row["Count"]]
    weighted_baseline = sum(
        row["baseline_latency_p50_ms"] * row["Count"] for row in weighted
    )
    weighted_ours = sum(row["ours_latency_p50_ms"] * row["Count"] for row in weighted)
    below = [row for row in rows if row["below_threshold"]]
    return {
        "rows": rows,
        "aggregate": {
            "measured_row_count": len(rows),
            "skipped_row_count": len(skipped),
            "skipped_input_row_indexes": sorted(skipped),
            "equal_weight_relative_throughput_pct": rounded_derived(
                equal_baseline / equal_ours * 100.0 if equal_ours else None
            ),
            "count_weighted_relative_throughput_pct": rounded_derived(
                weighted_baseline / weighted_ours * 100.0 if weighted_ours else None
            ),
            "count_weighted_row_count": len(weighted),
            "median_relative_throughput_pct": rounded_derived(
                statistics.median(row["relative_throughput_pct"] for row in rows)
            ),
            "median_sweep_spread_pct": rounded_derived(
                statistics.median(row["sweep_spread_pct"] for row in rows)
            ),
            "below_threshold_row_count": len(below),
            "below_threshold_above_noise_floor_row_count": sum(
                1 for row in below if not row["below_noise_floor"]
            ),
            "regression_confirmed_row_count": (
                sum(1 for row in rows if row["regression_confirmed"])
                if confirmable
                else None
            ),
            "sweeps": sweeps,
            "below_noise_floor_row_count": sum(
                1 for row in rows if row["below_noise_floor"]
            ),
            "regression_threshold_pct": threshold_pct,
            "noise_floor_ms": noise_floor_ms,
        },
    }


def _confirmed_text(value: Optional[bool]) -> str:
    """Render a tri-state confirmation for the summary table."""
    return "n/a" if value is None else ("yes" if value else "no")


def print_summary(summary: Mapping[str, Any]) -> None:
    """Print the acceptance metrics and every row below the threshold."""
    aggregate = summary["aggregate"]
    print("[rebench] interleaved re-measurement")
    print(f"  measured rows                 {aggregate['measured_row_count']}")
    if aggregate["skipped_row_count"]:
        print(
            f"  skipped rows                  {aggregate['skipped_row_count']} "
            f"{aggregate['skipped_input_row_indexes']}"
        )
    print(
        "  equal-weight throughput       "
        f"{aggregate['equal_weight_relative_throughput_pct']}%"
    )
    print(
        "  Count-weighted throughput     "
        f"{aggregate['count_weighted_relative_throughput_pct']}% "
        f"({aggregate['count_weighted_row_count']} rows carry a Count)"
    )
    print(
        "  median per-row throughput     "
        f"{aggregate['median_relative_throughput_pct']}%"
    )
    print(
        "  median per-row sweep spread   "
        f"{aggregate['median_sweep_spread_pct']}%  (this row's own error bar)"
    )
    print(
        f"  rows below {aggregate['regression_threshold_pct']}%             "
        f"{aggregate['below_threshold_row_count']}, of which "
        f"{aggregate['below_threshold_above_noise_floor_row_count']} are above the "
        f"{aggregate['noise_floor_ms']} ms noise floor"
    )
    if aggregate["regression_confirmed_row_count"] is None:
        print(
            "  confirmed regressions         n/a  "
            "(needs at least 2 sweeps to have an error bar)"
        )
    else:
        print(
            "  confirmed regressions         "
            f"{aggregate['regression_confirmed_row_count']}  "
            "(shortfall exceeds that row's own sweep spread)"
        )
    below = [row for row in summary["rows"] if row["below_threshold"]]
    if not below:
        return
    print(
        "  %-5s %-13s %11s %11s %8s %8s %6s %10s"
        % (
            "index",
            "variant",
            "baseline_ms",
            "ours_ms",
            "rel%",
            "spread%",
            "floor",
            "confirmed",
        )
    )
    for row in sorted(below, key=lambda item: item["relative_throughput_pct"]):
        print(
            "  %-5s %-13s %11.6f %11.6f %8.1f %8.1f %6s %10s"
            % (
                row["input_row_index"],
                row["variant"][:13],
                row["baseline_latency_p50_ms"],
                row["ours_latency_p50_ms"],
                row["relative_throughput_pct"],
                row["sweep_spread_pct"],
                "yes" if row["below_noise_floor"] else "no",
                _confirmed_text(row["regression_confirmed"]),
            )
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Join, re-measure, rewrite, and summarize two Pretune artifacts."""
    args = build_parser().parse_args(argv)
    if args.trials <= 0:
        raise RebenchError("--trials must be positive")
    if args.repeats <= 0:
        raise RebenchError("--repeats must be positive")
    if args.latency_warmup < 0:
        raise RebenchError("--latency-warmup must be non-negative")
    if args.latency_iterations <= 0:
        raise RebenchError("--latency-iter must be positive")
    if args.benchmark_retries <= 0:
        raise RebenchError("--benchmark-retries must be positive")
    if args.noise_floor_ms < 0:
        raise RebenchError("--noise-floor-ms must be non-negative")

    pinned = _pin_single_device(args.device)
    from flag_gems.flagtune.cli.pretune import PretuneError, load_shape_records
    from flag_gems.flagtune.contracts.operator import (
        OperatorConfigError,
        load_operator_benchmark_spec,
    )
    from flag_gems.flagtune.runtime.device import (
        DeviceProbeError,
        probe_flagtune_environment,
    )

    try:
        environment = probe_flagtune_environment()
    except DeviceProbeError as exc:
        raise RebenchError(str(exc)) from exc
    if environment.device_count != 1:
        raise RebenchError(
            "re-measurement interleaves both sides in one process and needs "
            f"exactly one visible device, but {environment.device_count} are "
            f"visible with {pinned}. Export the backend visibility variable "
            "with a single token, for example CUDA_VISIBLE_DEVICES=0."
        )

    config_path = Path(args.flagtune_config).expanduser().resolve()
    shape_path = Path(args.shape_config).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    try:
        spec = load_operator_benchmark_spec(config_path)
    except (OperatorConfigError, OSError, ValueError) as exc:
        raise RebenchError(str(exc)) from exc

    paths = {
        "baseline": Path(args.baseline).expanduser().resolve(),
        "ours": Path(args.ours).expanduser().resolve(),
    }
    tables = {side: _read_pretune_csv(paths[side], side) for side in SIDES}
    indexed = {side: _index_rows(tables[side][0], side) for side in SIDES}
    if set(indexed["baseline"]) != set(indexed["ours"]):
        missing_ours = sorted(set(indexed["baseline"]) - set(indexed["ours"]))
        missing_baseline = sorted(set(indexed["ours"]) - set(indexed["baseline"]))
        raise RebenchError(
            "input_row_index sets differ: "
            f"missing from ours={missing_ours}; "
            f"missing from baseline={missing_baseline}"
        )

    joined: dict[int, dict[str, dict[str, str]]] = {}
    skipped: list[int] = []
    for index in sorted(indexed["baseline"]):
        pair = {side: indexed[side][index] for side in SIDES}
        if any(pair[side].get("status") != "ok" for side in SIDES):
            skipped.append(index)
            continue
        if pair["baseline"].get("variant") != pair["ours"].get("variant"):
            raise RebenchError(
                f"row {index} has variant "
                f"{pair['baseline'].get('variant')!r} in baseline but "
                f"{pair['ours'].get('variant')!r} in ours"
            )
        joined[index] = pair
    if not joined:
        raise RebenchError("no row completed in both runs, nothing to re-measure")

    try:
        records = load_shape_records(shape_path, spec)
    except PretuneError as exc:
        raise RebenchError(f"cannot load {shape_path}: {exc}") from exc
    records_by_index = {record.source_index: record for record in records}

    dtype_names = _resolve_dtypes(args.dtypes, len(spec.benchmark.tensors))
    collected = measure_all(
        spec=spec,
        config_path=config_path,
        device_runtime=environment.runtime,
        records_by_index=records_by_index,
        joined=joined,
        dtype_names=dtype_names,
        trials=args.trials,
        repeats=args.repeats,
        warmup=args.latency_warmup,
        iterations=args.latency_iterations,
        benchmark_mode=args.benchmark_mode,
        benchmark_retries=args.benchmark_retries,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        rows, fieldnames = tables[side]
        rewritten = []
        for row in rows:
            index = int(row["input_row_index"])
            if index not in collected:
                rewritten.append(dict(row))
                continue
            rewritten.append(
                _rewrite_latency(
                    row,
                    _reduce_sweeps(collected[index][side]),
                    warmup=args.latency_warmup,
                    iterations=args.latency_iterations,
                    trials=args.trials * args.repeats,
                )
            )
        _write_csv(output_dir / f"{side}_pretune.csv", fieldnames, rewritten)

    summary = summarize(
        collected=collected,
        joined=joined,
        skipped=skipped,
        sweeps=args.repeats,
        noise_floor_ms=args.noise_floor_ms,
        threshold_pct=args.regression_threshold_pct,
    )
    summary["inputs"] = {
        "device_visibility": pinned,
        "baseline": str(paths["baseline"]),
        "ours": str(paths["ours"]),
        "shape_config": str(shape_path),
        "flagtune_config": str(config_path),
        "dtypes": dtype_names,
    }
    summary["protocol"] = {
        "latency_source": LATENCY_SOURCE,
        "trials_per_sweep": args.trials,
        "sweeps": args.repeats,
        "latency_warmup_ms": args.latency_warmup,
        "latency_measurement_ms": args.latency_iterations,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_retries": args.benchmark_retries,
    }
    try:
        (output_dir / "rebench_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise RebenchError(f"cannot write the summary: {exc}") from exc

    print_summary(summary)
    print(f"[rebench] artifacts in {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RebenchError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
