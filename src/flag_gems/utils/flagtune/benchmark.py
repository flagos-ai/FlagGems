"""Operator-independent multi-GPU benchmark scheduling and DB coordination.

Inputs:
    Public callers provide shapes with optional config collections, one safely
    compiled operator YAML, GPU tokens, timing options, a database URL, and a
    writable work directory.

Outputs:
    :func:`run_shape_config_benchmarks` returns ordered executor-produced result
    rows plus worker exit codes, fail-fast state, logs, database merge metadata,
    and any recoverable SQLite shards.

Implementation:
    The parent converts cases through the generic executor, distributes tasks
    round robin, and starts one long-lived subprocess per device. Each worker
    exposes exactly one adapter-selected device, reuses one executor, and streams JSONL
    results.  File-backed SQLite uses one snapshot shard per concurrent worker;
    shards are merged in worker order with ``INSERT OR IGNORE`` so existing
    target rows win. Callers select an explicit LibTuner config-selection mode
    that is forwarded through the private worker CLI rather than process-global
    environment state.

Limitations:
    Executor payloads and results must be JSON serializable. In-memory SQLite is
    unsupported with multiple workers.  Shared non-SQLite backends receive
    concurrent writes without scheduler-level transactions.  Fail-fast stops
    new work but cannot retract completed kernels or database writes.  The
    scheduler does not understand operator shapes, validate numerical results,
    or choose tuning policy.

Environment variables:
    Backend-owned visibility variables are read to validate/select worker
    tokens; each spawned worker receives exactly one selected token. ``FLAGGEMS_DB_URL``
    is inherited only when callers omit an explicit database URL, then each
    worker process is given the resolved value. The scheduler does not interpret
    either value as a hardware reservation, and tokens need not be physical GPU
    ordinals under a scheduler or container runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class BenchmarkError(RuntimeError):
    """A batch or worker error that is safe to show to callers."""


TUNING_RUN_MODES = (
    "normal",
    "force_policy",
    "exhaustive_collection",
)
DEFAULT_BENCHMARK_WARMUP_MS = 25
DEFAULT_BENCHMARK_ITERATIONS_MS = 100
DEFAULT_LATENCY_WARMUP_MS = 25
DEFAULT_LATENCY_ITERATIONS_MS = 100
DEFAULT_LATENCY_TRIALS = 3


@dataclass(frozen=True)
class BenchmarkCase:
    """Explicitly frame an opaque shape and its optional opaque configs.

    Both values are interpreted by the generic data-driven executor.
    """

    shape: Any
    configs: Any = None


@dataclass(frozen=True)
class BenchmarkTask:
    """Pair a scheduler-owned identifier with an executor-owned payload."""

    task_index: int
    payload: Any

    def to_json(self) -> dict[str, Any]:
        """Return the exact dictionary written to a worker task file."""
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "BenchmarkTask":
        """Reconstruct a task from a decoded worker task-file entry."""
        return cls(task_index=int(payload["task_index"]), payload=payload["payload"])


@dataclass(frozen=True)
class BenchmarkBatchResult:
    """Return batch rows together with process and database recovery metadata."""

    results: list[dict[str, Any]]
    worker_returncodes: list[int]
    fail_fast_triggered: bool
    database_url: str
    database_merge: dict[str, Any]
    database_merge_error: str
    database_shards: list[Path]
    worker_log_paths: list[Path]


def _prepare_tasks(
    shape_configs: Sequence[Any], operator_config: Path | str
) -> list[BenchmarkTask]:
    """Convert user cases to indexed, JSON-serializable executor payloads."""
    if not shape_configs:
        raise BenchmarkError("shape_configs must contain at least one case")
    from flag_gems.utils.flagtune.executor import prepare_benchmark_case
    from flag_gems.utils.flagtune.operator_config import load_operator_benchmark_spec

    try:
        spec = load_operator_benchmark_spec(operator_config)
    except Exception as exc:
        raise BenchmarkError(f"cannot compile operator config: {exc}") from exc
    tasks = []
    for task_index, item in enumerate(shape_configs):
        if isinstance(item, BenchmarkCase):
            shape, configs = item.shape, item.configs
        elif isinstance(item, tuple) and len(item) == 2:
            shape, configs = item
        else:
            shape, configs = item, None
        try:
            payload = prepare_benchmark_case(spec, shape, configs, task_index)
            json.dumps(payload)
        except Exception as exc:
            raise BenchmarkError(
                f"executor failed to prepare case {task_index}: {exc}"
            ) from exc
        tasks.append(BenchmarkTask(task_index, payload))
    return tasks


def _sqlite_url(path: Path) -> str:
    """Convert an absolute filesystem path to the expected SQLite URL form."""
    return f"sqlite:///{path}"


def parse_sqlite_url(url: str) -> tuple[str, Optional[Path]]:
    """Classify a URL as file SQLite, memory SQLite, or a shared backend."""
    if url == "sqlite:///:memory:":
        return "sqlite_memory", None
    if url.startswith("sqlite:///"):
        return "sqlite", Path(url[len("sqlite:///") :]).expanduser().resolve()
    return "shared", None


def backup_sqlite(source: Path, destination: Path) -> None:
    """Create a consistent SQLite snapshot, or an empty destination if absent."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return
    with (
        sqlite3.connect(source) as source_conn,
        sqlite3.connect(destination) as dest_conn,
    ):
        source_conn.backup(dest_conn)


def quote_identifier(value: str) -> str:
    """Quote an SQLite identifier that cannot be bound as a parameter."""
    return '"' + value.replace('"', '""') + '"'


def merge_sqlite_shards(target: Path, shards: Sequence[Path]) -> dict[str, int]:
    """Merge worker SQLite shards while preserving existing target rows.

    Args:
        target: Destination database, created if needed.
        shards: Worker snapshots in deterministic precedence order.

    Returns:
        Counts of visited shard tables and newly inserted rows.

    Implementation:
        All available shards are attached to one target connection.  Missing
        tables are created from shard DDL and copied with ``INSERT OR IGNORE``
        inside a target transaction.

    Limitations:
        Tables are assumed to have compatible schemas and conflict keys.  A
        failed merge leaves shards intact for manual recovery.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    table_count = 0
    inserted_rows = 0
    conn = sqlite3.connect(target, timeout=60.0)
    aliases = []
    try:
        for index, shard in enumerate(shards):
            if not shard.exists():
                continue
            alias = f"worker_{index}"
            conn.execute(
                f"ATTACH DATABASE ? AS {quote_identifier(alias)}", (str(shard),)
            )
            aliases.append(alias)
        with conn:
            for alias in aliases:
                rows = conn.execute(
                    f"SELECT name, sql FROM {quote_identifier(alias)}.sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
                for table_name, create_sql in rows:
                    if not create_sql:
                        continue
                    create_if_missing = re.sub(
                        r"^CREATE TABLE\s+",
                        "CREATE TABLE IF NOT EXISTS ",
                        create_sql,
                        count=1,
                    )
                    conn.execute(create_if_missing)
                    before = conn.total_changes
                    quoted_table = quote_identifier(table_name)
                    conn.execute(
                        f"INSERT OR IGNORE INTO main.{quoted_table} "
                        f"SELECT * FROM {quote_identifier(alias)}.{quoted_table}"
                    )
                    inserted_rows += conn.total_changes - before
                    table_count += 1
        for alias in aliases:
            conn.execute(f"DETACH DATABASE {quote_identifier(alias)}")
    finally:
        conn.close()
    return {"visited_tables": table_count, "inserted_rows": inserted_rows}


def _default_database_url() -> str:
    """Return inherited DB URL or the standard vendor/Triton SQLite cache."""
    inherited = os.environ.get("FLAGGEMS_DB_URL")
    if inherited:
        return inherited
    import triton
    from flag_gems.runtime.backend import _state
    from flag_gems.utils.code_cache import config_cache_dir

    version = triton.__version__.split(".")
    vendor = _state.vendor_module.vendor_info.vendor_name
    path = (
        config_cache_dir() / f"TunedConfig_{vendor}_triton_{version[0]}_{version[1]}.db"
    )
    return _sqlite_url(path.resolve())


def _split_tasks(
    tasks: Sequence[BenchmarkTask], workers: int
) -> list[list[BenchmarkTask]]:
    """Distribute indexed tasks round robin while preserving per-worker order."""
    chunks = [[] for _ in range(workers)]
    for index, task in enumerate(tasks):
        chunks[index % workers].append(task)
    return [chunk for chunk in chunks if chunk]


def _prepare_worker_databases(
    database_url: str, work_dir: Path, workers: int
) -> tuple[list[str], list[Path], Optional[Path]]:
    """Resolve per-worker DB URLs and create SQLite snapshot shards if needed."""
    kind, target = parse_sqlite_url(database_url)
    if kind == "sqlite_memory" and workers > 1:
        raise BenchmarkError(
            "sqlite:///:memory: cannot be used with multiple GPU workers"
        )
    if kind != "sqlite" or workers == 1:
        return [database_url] * workers, [], target
    assert target is not None
    shard_dir = work_dir / "database-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for worker_id in range(workers):
        shard = shard_dir / f"worker_{worker_id}.db"
        backup_sqlite(target, shard)
        shards.append(shard)
    return [_sqlite_url(shard) for shard in shards], shards, target


def _run_worker(args: argparse.Namespace) -> int:
    """Execute one worker's task file on exactly one visible supported device.

    Results are flushed one JSON object per line so partial progress survives a
    later failure. The generic executor owns benchmark semantics and results.
    """
    os.environ["FLAGGEMS_DB_URL"] = args.database_url

    from flag_gems.utils.flagtune.device_runtime import (
        probe_flagtune_environment,
    )

    environment = probe_flagtune_environment()
    if environment.runtime.backend != args.device_backend:
        raise BenchmarkError(
            f"worker expected backend {args.device_backend!r}, got "
            f"{environment.runtime.backend!r}"
        )
    if environment.device_count != 1:
        raise BenchmarkError(
            f"worker expected exactly one visible {args.device_backend!r} "
            f"device, got {environment.device_count}"
        )
    environment.runtime.set_device(0)
    from flag_gems.utils.flagtune.executor import (
        BenchmarkWorker,
        describe_benchmark_case,
    )

    worker = BenchmarkWorker(args.operator_config, environment.runtime)
    payload = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    tasks = [BenchmarkTask.from_json(item) for item in payload]
    had_failure = False
    result_path = Path(args.result_file)
    with result_path.open("w", encoding="utf-8") as output:
        for task in tasks:
            try:
                result = worker.benchmark(
                    task.payload,
                    dtype_names=args.dtypes.split(","),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    tuning_run_mode=args.tuning_run_mode,
                    latency_warmup=args.latency_warmup,
                    latency_iterations=args.latency_iterations,
                    latency_trials=args.latency_trials,
                    gpu_token=args.gpu_token,
                    worker_id=args.worker_id,
                )
                result["task_index"] = task.task_index
                print(
                    f"worker={args.worker_id} case={describe_benchmark_case(task.payload)} "
                    "status=ok",
                    flush=True,
                )
            except Exception as exc:
                had_failure = True
                result = worker.failure_result(
                    task.payload,
                    dtype_names=args.dtypes.split(","),
                    gpu_token=args.gpu_token,
                    worker_id=args.worker_id,
                    exc=exc,
                )
                result["task_index"] = task.task_index
                print(
                    f"worker={args.worker_id} case={describe_benchmark_case(task.payload)} "
                    f"status=failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            output.write(json.dumps(result, sort_keys=True))
            output.write("\n")
            output.flush()
            if had_failure and args.fail_fast:
                break
    return 1 if had_failure else 0


def _worker_parser() -> argparse.ArgumentParser:
    """Build the private subprocess-only CLI parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--operator-config", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--gpu-token", required=True)
    parser.add_argument("--device-backend", required=True)
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--dtypes", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--tuning-run-mode",
        choices=TUNING_RUN_MODES,
        required=True,
    )
    parser.add_argument("--latency-warmup", type=int, required=True)
    parser.add_argument("--latency-iterations", type=int, required=True)
    parser.add_argument("--latency-trials", type=int, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _forward_worker_log_updates(
    log_paths: Sequence[Path],
    states: list[tuple[int, str]],
    *,
    final: bool = False,
) -> None:
    """Copy newly appended worker-log lines to the parent process stdout.

    Args:
        log_paths: Stable worker-ID-ordered paths written by subprocesses.
        states: Mutable ``(consumed_character_count, partial_line)`` state for
            each log. The list is updated in place after every call.
        final: Flush a final unterminated line when worker processes have exited.

    Outputs:
        Each complete new line is printed immediately with a worker-ID prefix.
        Original log files are never modified and remain the authoritative
        unprefixed record.

    Implementation:
        Logs are intentionally small progress/status streams, so rereading each
        file avoids platform-specific file-following code. A partial trailing
        line is buffered until the next poll to prevent interleaved fragments.

    Limitations:
        This helper assumes log files are append-only while streaming. If an
        external process truncates a log, forwarding resumes from its start and
        may repeat lines.
    """
    for worker_id, path in enumerate(log_paths):
        if not path.exists():
            continue
        consumed, partial = states[worker_id]
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) < consumed:
            consumed, partial = 0, ""
        update = partial + content[consumed:]
        consumed = len(content)
        lines = update.split("\n")
        partial = lines.pop()
        for line in lines:
            print(f"[benchmark worker {worker_id}] {line}", flush=True)
        if final and partial:
            print(f"[benchmark worker {worker_id}] {partial}", flush=True)
            partial = ""
        states[worker_id] = (consumed, partial)


def _worker_environment(
    device_runtime: Any,
    gpu_token: str,
    database_url: str,
) -> dict[str, str]:
    """Build an isolated environment for one dedicated benchmark worker.

    Args:
        device_runtime: Registered backend adapter that owns visibility rules.
        gpu_token: Physical/logical token exposed as worker device zero.
        database_url: LibTuner database URL bound before importing FlagGems.

    Returns:
        A copy of the parent environment with authoritative worker overrides.
    """
    env = os.environ.copy()
    device_runtime.apply_worker_visibility(env, gpu_token)
    env["FLAGGEMS_DB_URL"] = database_url
    return env


def _launch_workers(
    tasks: Sequence[BenchmarkTask],
    *,
    operator_config: Path,
    work_dir: Path,
    gpu_tokens: Sequence[str],
    database_urls: Sequence[str],
    dtypes: Sequence[str],
    warmup: int,
    iterations: int,
    tuning_run_mode: str,
    latency_warmup: int,
    latency_iterations: int,
    latency_trials: int,
    fail_fast: bool,
    stream_worker_logs: bool,
    device_runtime: Any,
) -> tuple[list[dict[str, Any]], list[int], bool, list[Path]]:
    """Launch GPU subprocesses, monitor fail-fast, and collect ordered rows.

    Args:
        tasks: Prepared generic benchmark payloads.
        operator_config: Absolute data-driven operator YAML path.
        work_dir: Persistent batch directory for tasks, results, and logs.
        gpu_tokens: One physical/visible GPU token per worker.
        database_urls: One database URL per worker, including SQLite shards.
        dtype: Runtime tensor dtype name.
        warmup: Warmup duration/count forwarded to the executor.
        iterations: Measurement duration/count forwarded to the executor.
        tuning_run_mode: Explicit LibTuner config-selection behavior.
        latency_warmup: Fresh selected-config warmup milliseconds.
        latency_iterations: Fresh selected-config measurement milliseconds.
        latency_trials: Number of independent fresh selected-config trials.
        fail_fast: Terminate peers after the first nonzero worker exit.
        stream_worker_logs: Mirror appended worker log lines to parent stdout.
    Returns:
        Ordered result rows, worker return codes, the fail-fast trigger state,
        and persistent worker log paths.

    Implementation:
        Worker stdout and stderr are always combined into per-worker files. When
        streaming is enabled, the parent polling loop tails those same files,
        preserving logs while providing immediate console feedback.

    Limitations:
        Subprocesses receive termination rather than cooperative cancellation on
        fail-fast. A currently executing GPU kernel cannot be retracted.
    """
    chunks = _split_tasks(tasks, len(gpu_tokens))
    worker_dir = work_dir / "benchmark-workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    log_handles = []
    log_paths = []
    for worker_id, (chunk, gpu_token, database_url) in enumerate(
        zip(chunks, gpu_tokens, database_urls)
    ):
        task_path = worker_dir / f"worker_{worker_id}_tasks.json"
        result_path = worker_dir / f"worker_{worker_id}_results.jsonl"
        log_path = worker_dir / f"worker_{worker_id}.log"
        task_path.write_text(
            json.dumps([task.to_json() for task in chunk], indent=2), encoding="utf-8"
        )
        log_handle = log_path.open("w", encoding="utf-8")
        log_handles.append(log_handle)
        log_paths.append(log_path)
        command = [
            sys.executable,
            str(MODULE_PATH),
            "--worker",
            "--operator-config",
            str(operator_config),
            "--task-file",
            str(task_path),
            "--result-file",
            str(result_path),
            "--gpu-token",
            gpu_token,
            "--device-backend",
            device_runtime.backend,
            "--worker-id",
            str(worker_id),
            "--database-url",
            database_url,
            "--dtypes",
            ",".join(dtypes),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--tuning-run-mode",
            tuning_run_mode,
            "--latency-warmup",
            str(latency_warmup),
            "--latency-iterations",
            str(latency_iterations),
            "--latency-trials",
            str(latency_trials),
        ]
        if fail_fast:
            command.append("--fail-fast")
        env = _worker_environment(device_runtime, gpu_token, database_url)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_id, process, result_path))

    active = {
        worker_id: (process, result_path)
        for worker_id, process, result_path in processes
    }
    returncodes: dict[int, int] = {}
    fail_fast_triggered = False
    log_states = [(0, "") for _ in log_paths]
    try:
        while active:
            for worker_id, (process, _result_path) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                returncodes[worker_id] = returncode
                del active[worker_id]
                if fail_fast and returncode != 0:
                    fail_fast_triggered = True
                    for other_process, _ in active.values():
                        other_process.terminate()
                    for other_id, (other_process, _) in list(active.items()):
                        try:
                            other_code = other_process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            other_process.kill()
                            other_code = other_process.wait()
                        returncodes[other_id] = other_code
                        del active[other_id]
                    break
            if stream_worker_logs:
                _forward_worker_log_updates(log_paths, log_states)
            if active:
                time.sleep(0.05)
    finally:
        for log_handle in log_handles:
            log_handle.close()
        if stream_worker_logs:
            _forward_worker_log_updates(log_paths, log_states, final=True)

    results = []
    for _worker_id, _process, result_path in processes:
        if not result_path.exists():
            continue
        with result_path.open("r", encoding="utf-8") as handle:
            results.extend(json.loads(line) for line in handle if line.strip())
    results.sort(key=lambda row: row.get("task_index", sys.maxsize))
    codes = [returncodes.get(index, -1) for index in range(len(processes))]
    return results, codes, fail_fast_triggered, log_paths


def run_shape_config_benchmarks(
    shape_configs: Sequence[Any],
    *,
    operator_config: Path | str,
    dtypes: Sequence[str] | str = "bfloat16",
    warmup: int = DEFAULT_BENCHMARK_WARMUP_MS,
    iterations: int = DEFAULT_BENCHMARK_ITERATIONS_MS,
    tuning_run_mode: str = "normal",
    latency_warmup: int = DEFAULT_LATENCY_WARMUP_MS,
    latency_iterations: int = DEFAULT_LATENCY_ITERATIONS_MS,
    latency_trials: int = DEFAULT_LATENCY_TRIALS,
    parallel: Optional[int] = None,
    gpu_tokens: Optional[Sequence[str]] = None,
    database_url: Optional[str] = None,
    work_dir: Path | str,
    fail_fast: bool = False,
    stream_worker_logs: bool = False,
) -> BenchmarkBatchResult:
    """Run opaque shape/config cases with one long-lived worker per GPU.

    Args:
        shape_configs: Cases expressed as :class:`BenchmarkCase`, ``(shape,
            configs)`` tuples, or bare shapes.
        operator_config: Data-driven operator YAML used by the generic executor.
        dtypes: Ordered input tensor dtypes. One value broadcasts to all
            ``benchmark.invoke.args`` entries.
        warmup: Non-negative candidate-config warmup duration in milliseconds.
        iterations: Positive candidate-config measurement duration in
            milliseconds.
        tuning_run_mode: ``normal``, ``force_policy``, or
            ``exhaustive_collection`` LibTuner config-selection behavior.
        latency_warmup: Non-negative fresh selected-config warmup milliseconds.
        latency_iterations: Positive fresh selected-config measurement
            milliseconds per trial.
        latency_trials: Positive number of fresh selected-config trials.
        parallel: Maximum workers; defaults to all supplied/visible GPUs.
        gpu_tokens: Optional physical device tokens. Auto-detected through the
            active backend adapter when omitted.
        database_url: Optional backend URL; defaults to FlagGems configuration.
        work_dir: Persistent task, result, log, and shard directory.
        fail_fast: Stop peer workers after the first nonzero worker exit.
        stream_worker_logs: Mirror worker stdout/stderr to parent stdout while
            retaining the complete per-worker log files.
    Returns:
        A :class:`BenchmarkBatchResult` whose rows are sorted by input index.

    Raises:
        BenchmarkError: For invalid options, missing GPUs, config failures, or
            unsupported memory-SQLite parallelism.

    Notes:
        Successful shard merges delete the temporary shards.  Failed or skipped
        merges retain them and report their paths in the return value.
    """

    from triton.flagtune.identity import normalize_dtype_name

    requested = (
        [item.strip() for item in dtypes.split(",") if item.strip()]
        if isinstance(dtypes, str)
        else list(dtypes)
    )
    if not requested:
        raise BenchmarkError("dtypes must contain at least one dtype")
    try:
        requested = [normalize_dtype_name(item) for item in requested]
    except ValueError as exc:
        raise BenchmarkError(str(exc)) from exc
    if warmup < 0:
        raise BenchmarkError("warmup must be non-negative")
    if iterations <= 0:
        raise BenchmarkError("iterations must be positive")
    if tuning_run_mode not in TUNING_RUN_MODES:
        raise BenchmarkError(
            "tuning_run_mode must be one of "
            + ", ".join(repr(mode) for mode in TUNING_RUN_MODES)
        )
    if latency_warmup < 0:
        raise BenchmarkError("latency_warmup must be non-negative")
    if latency_iterations <= 0:
        raise BenchmarkError("latency_iterations must be positive")
    if latency_trials <= 0:
        raise BenchmarkError("latency_trials must be positive")
    config_path = Path(operator_config).expanduser().resolve()
    from flag_gems.utils.flagtune.operator_config import load_operator_benchmark_spec

    input_count = len(load_operator_benchmark_spec(config_path).benchmark.args)
    if len(requested) == 1:
        requested *= input_count
    elif len(requested) != input_count:
        raise BenchmarkError(
            f"dtypes has {len(requested)} values but invoke.args has {input_count} inputs"
        )
    tasks = _prepare_tasks(shape_configs, config_path)
    work_path = Path(work_dir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    from flag_gems.utils.flagtune.device_runtime import (
        probe_flagtune_environment,
    )

    try:
        environment = probe_flagtune_environment()
    except Exception as exc:
        raise BenchmarkError(f"FlagTune device preflight failed: {exc}") from exc
    if gpu_tokens is None:
        tokens = environment.runtime.visible_device_tokens(
            environment.device_count
        )
    else:
        tokens = [str(token) for token in gpu_tokens]
        if not tokens:
            raise BenchmarkError("gpu_tokens must contain at least one device")
    worker_limit = parallel if parallel is not None else len(tokens)
    if worker_limit <= 0:
        raise BenchmarkError("parallel must be positive")
    if worker_limit > len(tokens):
        raise BenchmarkError(
            f"parallel {worker_limit} exceeds {len(tokens)} supplied/visible GPUs"
        )
    worker_count = min(worker_limit, len(tasks))
    tokens = tokens[:worker_count]
    resolved_database_url = database_url or _default_database_url()
    database_urls, shards, target = _prepare_worker_databases(
        resolved_database_url, work_path, worker_count
    )
    results, returncodes, fail_fast_triggered, log_paths = _launch_workers(
        tasks,
        operator_config=config_path,
        work_dir=work_path,
        gpu_tokens=tokens,
        database_urls=database_urls,
        dtypes=requested,
        warmup=warmup,
        iterations=iterations,
        tuning_run_mode=tuning_run_mode,
        latency_warmup=latency_warmup,
        latency_iterations=latency_iterations,
        latency_trials=latency_trials,
        fail_fast=fail_fast,
        stream_worker_logs=stream_worker_logs,
        device_runtime=environment.runtime,
    )

    merge_summary: dict[str, Any] = {"status": "not_needed"}
    merge_error = ""
    remaining_shards = list(shards)
    if shards and not fail_fast_triggered:
        try:
            assert target is not None
            merge_summary = {"status": "ok", **merge_sqlite_shards(target, shards)}
            for shard in shards:
                shard.unlink(missing_ok=True)
            try:
                (work_path / "database-shards").rmdir()
            except OSError:
                pass
            remaining_shards = []
        except Exception:
            merge_summary = {"status": "failed"}
            merge_error = traceback.format_exc()
    elif fail_fast_triggered and shards:
        merge_summary = {"status": "skipped_fail_fast"}

    return BenchmarkBatchResult(
        results=results,
        worker_returncodes=returncodes,
        fail_fast_triggered=fail_fast_triggered,
        database_url=resolved_database_url,
        database_merge=merge_summary,
        database_merge_error=merge_error,
        database_shards=remaining_shards,
        worker_log_paths=log_paths,
    )


def benchmark_shape_configs(
    shape_configs: Sequence[Any], *, operator_config: Path | str, **kwargs: Any
) -> list[dict[str, Any]]:
    """Return only ordered rows, using a temporary work directory if omitted.

    This convenience API discards process and merge metadata.  Use
    :func:`run_shape_config_benchmarks` when recovery information matters.
    """

    work_dir = kwargs.pop("work_dir", None)
    if work_dir is not None:
        return run_shape_config_benchmarks(
            shape_configs,
            operator_config=operator_config,
            work_dir=work_dir,
            **kwargs,
        ).results
    temp_root = Path.home() / ".flaggems" / "benchmark-work"
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="batch_", dir=temp_root))
    try:
        return run_shape_config_benchmarks(
            shape_configs,
            operator_config=operator_config,
            work_dir=temporary,
            **kwargs,
        ).results
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> int:
    """Private worker entry point; return 2 for setup-level benchmark errors."""
    args = _worker_parser().parse_args()
    try:
        return _run_worker(args)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
