"""Prepare and execute data-driven FlagTune workloads on GPU workers.

The parent process uses :func:`prepare_benchmark_case` to normalize arbitrary
public API inputs into JSON-compatible task payloads.  A long-lived
:class:`BenchmarkWorker` then loads the same operator YAML, constructs only
allowlisted tensors, locates the matching LibTuner through trusted public
FlagGems metadata, and returns a stable operator-independent result mapping.

This module never imports a YAML-provided module or evaluates YAML expressions.
It runs inside a subprocess pinned to one device through the strict FlagTune
device-runtime adapter; isolation and task scheduling are provided by
``benchmark.py``.

Environment variables:
    ``FLAGTUNE_TRAIN_PROGRESS_INTERVAL`` is parsed as a non-negative integer
    only for exhaustive training collection. A positive value emits progress
    after that many successful config benchmarks; ``0`` (the default) suppresses
    those messages. Invalid values raise when the training executor queries it.
"""

from __future__ import annotations

import inspect
import math
import os
import statistics
import time
import traceback
from typing import Any, Mapping, Optional, Sequence

from triton.flagtune.expressions import evaluate_compiled
from triton.flagtune.identity import make_dtype_key, normalize_dtype_name

from .operator_config import (
    OperatorBenchmarkSpec,
    OperatorConfigError,
    load_operator_benchmark_spec,
    resolve_public_operator,
)
from .records import ShapeRecord


class BenchmarkExecutionError(RuntimeError):
    """Report a user-visible preparation or generic GPU execution failure.

    The scheduler treats this exception as a workload/configuration error rather
    than an internal process-management failure.  Per-case exceptions can also
    be serialized by :meth:`BenchmarkWorker.failure_result` when fail-fast mode
    is disabled.
    """


def _jsonable(value: Any) -> Any:
    """Recursively convert configuration metadata into JSON-compatible values.

    Args:
        value: Scalar, mapping, sequence, or opaque value obtained from a config.

    Returns:
        JSON-native scalars, string-keyed dictionaries, and lists.  Unsupported
        leaf objects are represented with ``repr`` so diagnostic metadata does
        not make worker-task serialization fail.

    Notes:
        This is a transport conversion, not a reversible serialization format.
        Operator shape values are validated separately and do not rely on the
        ``repr`` fallback.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def config_to_record(config: Any) -> dict[str, Any]:
    """Flatten a mapping or Triton Config-like object for JSON transport.

    Args:
        config: Either a mapping, an object exposing ``all_kwargs()``, or a
            Config-like object with ``kwargs`` and optional Triton launch fields.

    Returns:
        A newly allocated, string-keyed, JSON-compatible dictionary containing
        kernel parameters and available launch metadata.

    Raises:
        BenchmarkExecutionError: If no parameter mapping can be extracted or
        conversion unexpectedly does not yield a dictionary.

    Implementation:
        ``all_kwargs`` takes precedence because it combines constexpr and launch
        options for real Triton Config objects.  The fallback copies known launch
        attributes explicitly; hooks and executable objects are never carried
        into task files or result rows.
    """
    if isinstance(config, Mapping):
        payload = dict(config)
    elif hasattr(config, "all_kwargs"):
        payload = dict(config.all_kwargs())
    else:
        payload = dict(getattr(config, "kwargs", {}))
        if not payload:
            raise BenchmarkExecutionError(
                "each config must be a mapping or Triton Config-like object"
            )
        for name in (
            "num_warps",
            "num_stages",
            "num_ctas",
            "num_buffers_warp_spec",
            "num_consumer_groups",
            "reg_dec_producer",
            "reg_inc_consumer",
            "maxnreg",
        ):
            if hasattr(config, name):
                payload[name] = getattr(config, name)
    converted = _jsonable(payload)
    if not isinstance(converted, dict):
        raise BenchmarkExecutionError("config conversion did not produce a mapping")
    return converted


def _raw_case(shape: Any) -> dict[str, Any]:
    """Normalize supported shape containers without applying operator schema.

    Args:
        shape: A :class:`ShapeRecord`, mapping, nested ``{"shape": ...}``
            mapping, or positional sequence of identity values.

    Returns:
        An intermediate dictionary containing either ``values`` or ``sequence``
        plus optional count, variant, source index, and selected index metadata.

    Raises:
        BenchmarkExecutionError: If the outer container is unsupported.

    Notes:
        Mapping control keys are recognized case-insensitively.  A plain mapping
        is otherwise treated as field/value data; authoritative field checking,
        type conversion, dispatch, and index normalization happen in
        :func:`prepare_benchmark_case`.
    """
    if isinstance(shape, ShapeRecord):
        return shape.to_json()
    if isinstance(shape, Mapping):
        lowered = {str(key).lower(): value for key, value in shape.items()}
        if "values" in lowered:
            return {
                "values": dict(lowered["values"]),
                "count": lowered.get("count"),
                "variant": lowered.get("variant"),
                "source_index": lowered.get("source_index"),
                "selected_index": lowered.get("selected_index"),
            }
        if "shape" in lowered:
            nested = _raw_case(lowered["shape"])
            for name in ("count", "variant", "source_index", "selected_index"):
                if name in lowered:
                    nested[name] = lowered[name]
            return nested
        return {
            "values": dict(shape),
            "count": lowered.get("count"),
            "variant": lowered.get("variant"),
            "source_index": lowered.get("source_index"),
            "selected_index": lowered.get("selected_index"),
        }
    if isinstance(shape, (list, tuple)):
        return {
            "sequence": list(shape),
            "count": None,
            "variant": None,
            "source_index": None,
            "selected_index": None,
        }
    raise BenchmarkExecutionError("shape must be a mapping or sequence")


def prepare_benchmark_case(
    spec: OperatorBenchmarkSpec,
    shape: Any,
    configs: Optional[Sequence[Any]],
    task_index: int,
) -> dict[str, Any]:
    """Validate one generic case and produce a JSON-safe worker payload.

    Args:
        spec: Compiled operator, shape, dispatch, and benchmark contract.
        shape: Any shape container accepted by :func:`_raw_case`.
        configs: Optional explicit candidate collection.  ``None`` asks the
            worker to use the tuner's expanded/default config space; an empty
            collection is invalid.
        task_index: Stable scheduler index used as the fallback source and
            selected index and in validation diagnostics.

    Returns:
        A transport dictionary containing the source-config hash, canonical
        ordered values, normalized count, eligible variant, stable indices, and
        optional flattened config records.

    Raises:
        BenchmarkExecutionError: If the shape, count, explicit variant, or
        config collection violates the compiled schema.

    Implementation:
        Positional shapes are zipped to the declared identity order.  The shape
        schema normalizes values, first-match dispatch fills an omitted variant,
        and an explicit variant is checked against the same eligibility
        predicate.  The YAML hash binds the payload to the exact source later
        loaded by the worker.
    """
    raw = _raw_case(shape)
    if "sequence" in raw:
        if len(raw["sequence"]) != len(spec.shape.identity):
            raise BenchmarkExecutionError(
                f"case[{task_index}] sequence must have {len(spec.shape.identity)} values"
            )
        raw_values = dict(zip(spec.shape.identity, raw["sequence"]))
    else:
        raw_values = raw["values"]
    try:
        values, schema_count = spec.shape.normalize_values(
            raw_values, f"case[{task_index}]"
        )
    except OperatorConfigError as exc:
        raise BenchmarkExecutionError(str(exc)) from exc
    count = raw.get("count")
    if count is None:
        count = schema_count
    elif spec.shape.count_field:
        count = spec.shape.fields[spec.shape.count_field].normalize(
            count,
            values,
            f"case[{task_index}].{spec.shape.count_field}",
        )
    variant = raw.get("variant") or spec.resolve_variant(values)
    if variant not in spec.operator_info.variants:
        raise BenchmarkExecutionError(
            f"case[{task_index}] has unknown variant {variant!r}"
        )
    if not spec.operator_info.variants[variant].matches(values):
        raise BenchmarkExecutionError(
            f"case[{task_index}] shape is ineligible for variant {variant!r}"
        )
    prepared_configs = None
    if configs is not None:
        if not configs:
            raise BenchmarkExecutionError(
                f"case[{task_index}] has an empty config list"
            )
        prepared_configs = [config_to_record(config) for config in configs]
    source_index = raw.get("source_index")
    selected_index = raw.get("selected_index")
    return {
        "config_sha256": spec.source_sha256,
        "source_index": task_index if source_index is None else int(source_index),
        "selected_index": task_index if selected_index is None else int(selected_index),
        "values": values,
        "count": count,
        "variant": str(variant),
        "configs": prepared_configs,
    }


def describe_benchmark_case(payload: Mapping[str, Any]) -> str:
    """Return a compact ordered shape label for worker progress logs.

    Args:
        payload: Prepared worker payload containing an ordered ``values`` map.

    Returns:
        Comma-separated value strings in schema identity order.

    Notes:
        This helper is intended for diagnostics only and does not escape commas
        or preserve field names.
    """
    return ",".join(str(value) for value in payload["values"].values())


def _serialize_config_timings(timings: Any) -> list[dict[str, Any]]:
    """Convert LibTuner per-config quantiles into stable result records.

    Args:
        timings: Expected mapping from Config-like objects to one or more latency
            samples.  Non-mapping values represent unavailable timing metadata.

    Returns:
        One JSON-compatible row per mapping entry.  Up to three samples are
        interpreted as p50, p20, and p80 in LibTuner's stored order; missing,
        non-numeric, non-positive, or non-finite values become ``None`` and mark
        a nonfinite median as ``status='nonfinite'``.

    Notes:
        The serializer preserves mapping iteration order.  It does not benchmark
        or rank configs, and it intentionally returns an empty list when the
        tuner exposes no timing mapping.
    """
    if not isinstance(timings, Mapping):
        return []
    rows = []
    for config, raw_samples in timings.items():
        samples = (
            list(raw_samples)
            if isinstance(raw_samples, (list, tuple))
            else [raw_samples]
        )
        converted = []
        for sample in samples[:3]:
            try:
                value = float(sample)
            except (TypeError, ValueError):
                value = float("inf")
            converted.append(value if math.isfinite(value) and value > 0 else None)
        while len(converted) < 3:
            converted.append(None)
        rows.append(
            {
                "config": config_to_record(config),
                "latency_ms": converted[0],
                "latency_p50_ms": converted[0],
                "latency_p20_ms": converted[1],
                "latency_p80_ms": converted[2],
                "status": "ok" if converted[0] is not None else "nonfinite",
            }
        )
    return rows


def _progress_interval() -> int:
    """Read the optional per-worker explicit-config progress interval.

    Returns:
        A non-negative integer from ``FLAGTUNE_TRAIN_PROGRESS_INTERVAL``.
        Missing, invalid, and negative values disable interval reporting by
        returning zero.
    """
    try:
        return max(0, int(os.environ.get("FLAGTUNE_TRAIN_PROGRESS_INTERVAL", "0")))
    except ValueError:
        return 0


class BenchmarkWorker:
    """Execute any safely configured public FlagGems operator on one GPU.

    One instance is created per scheduler subprocess and reused for every task
    assigned to that GPU.  It caches each operator variant's expanded base config state
    but reconstructs tensors, resets dispatch caches, and clears transient tuner
    results for every shape.

    The worker assumes its process environment already restricts backend
    visibility to one device.  It does not choose a device or modify launcher
    visibility variables.
    """

    def __init__(self, config_path: str, device_runtime: Any = None) -> None:
        """Load runtime dependencies and bind the configured public operator.

        Args:
            config_path: Path to the same version-3 operator YAML used by the
                parent planner when task payloads were prepared.

        Raises:
            OperatorConfigError: If the configuration cannot be loaded safely.
            ImportError: If torch, Triton, or FlagGems is unavailable.
            AttributeError: If a caller bypassed planning and the configured
                public operator is absent from ``flag_gems``.

        Implementation:
            Heavy GPU dependencies are imported lazily inside the subprocess.
            ``base_states`` is initially empty and is populated on first use of
            each operator/variant pair by :meth:`benchmark`.
        """
        import triton

        import flag_gems
        from flag_gems.utils.flagtune.device_runtime import (
            probe_flagtune_environment,
        )

        self.spec = load_operator_benchmark_spec(config_path)
        if device_runtime is None:
            environment = probe_flagtune_environment()
            if environment.device_count != 1:
                raise BenchmarkExecutionError(
                    "FlagTune worker requires exactly one visible device, "
                    f"got {environment.device_count} for backend "
                    f"{environment.runtime.backend!r}"
                )
            device_runtime = environment.runtime
        self.device_runtime = device_runtime
        self._triton_module = triton
        self.operator = resolve_public_operator(flag_gems, self.spec.op_id)
        self.base_states: dict[tuple[str, str], tuple[list[Any], list[Any]]] = {}

    def _find_tuner(self, variant: str) -> tuple[Any, Any]:
        """Find the unique LibEntry/tuner pair for a configured variant.

        Args:
            variant: Declared variant already validated for the current shape.

        Returns:
            ``(LibEntry, LibTuner)`` resolved from the trusted public operator's
            defining module and canonical operator/variant identity.

        Raises:
            BenchmarkExecutionError: If zero or multiple tuners match, which
            usually indicates stale YAML/kernel metadata.

        Security:
            The YAML provides no module or attribute path.  Target discovery is
            delegated to ``find_flagtune_benchmark_target`` using the already
            imported public FlagGems callable.
        """
        from flag_gems.utils.libentry import find_flagtune_benchmark_target

        try:
            return find_flagtune_benchmark_target(
                self.operator, self.spec.op_id, variant
            )
        except RuntimeError as exc:
            raise BenchmarkExecutionError(
                f"cannot bind {self.spec.op_id}/{variant}: {exc}"
            ) from exc

    def _make_configs(
        self, records: Sequence[Mapping[str, Any]], tuner: Any
    ) -> list[Any]:
        """Reconstruct explicit Triton Config objects inside the GPU worker.

        Args:
            records: Flattened candidate dictionaries from parent-task JSON.
            tuner: Bound LibTuner whose hook supplies kernel-specific setup.

        Returns:
            Fresh ``triton.Config`` objects with supported launch metadata split
            from kernel kwargs and the tuner's FlagTune pre-hook attached when
            the local Triton constructor accepts it.

        Implementation:
            Constructor parameters are detected with ``inspect.signature`` for
            compatibility across Triton versions.  Unknown record keys remain
            kernel kwargs; only the fixed launch-field set is promoted.

        Limitations:
            Config serialization does not preserve arbitrary Config attributes
            or per-config hooks.  The single tuner-level pre-hook is the only
            executable state intentionally restored.
        """
        pre_hook = getattr(tuner, "_flagtune_pre_hook", None)
        if pre_hook is None and tuner.configs:
            pre_hook = getattr(tuner.configs[0], "pre_hook", None)
        signature = inspect.signature(self._triton_module.Config)
        launch_names = (
            "num_warps",
            "num_stages",
            "num_ctas",
            "num_buffers_warp_spec",
            "num_consumer_groups",
            "reg_dec_producer",
            "reg_inc_consumer",
            "maxnreg",
        )
        configs = []
        for record in records:
            kwargs = dict(record)
            constructor = {}
            for name in launch_names:
                if name in kwargs and name in signature.parameters:
                    constructor[name] = kwargs.pop(name)
            if "pre_hook" in signature.parameters:
                constructor["pre_hook"] = pre_hook
            configs.append(self._triton_module.Config(kwargs=kwargs, **constructor))
        return configs

    def _make_tensors(
        self, values: Mapping[str, Any], dtypes: Sequence[Any]
    ) -> dict[str, Any]:
        """Construct benchmark tensors from compiled allowlisted recipes.

        Args:
            values: Canonical ordered shape identity mapping.
            dtype: Runtime torch dtype selected by the CLI/benchmark API.

        Returns:
            Mapping from YAML tensor names to newly allocated device tensors.

        Implementation:
            Symbolic dimensions index ``values`` and literal dimensions are
            converted directly to integers.  The spec loader has already limited
            factory names and required runtime dtype semantics.

        Limitations:
            Tensors are regenerated for every shape and are not seeded.  This is
            suitable for performance tuning, not numerical-correctness testing.
        """
        if len(dtypes) != len(self.spec.benchmark.args):
            raise BenchmarkExecutionError(
                "ordered input dtypes must match benchmark.invoke.args"
            )
        dtype_by_tensor = {
            reference.name: dtype
            for reference, dtype in zip(self.spec.benchmark.args, dtypes)
        }
        tensors = {}
        for tensor in self.spec.benchmark.tensors:
            shape = tuple(
                int(evaluate_compiled(dim, values, {})) for dim in tensor.shape
            )
            if tensor.name not in dtype_by_tensor:
                raise BenchmarkExecutionError(
                    f"tensor {tensor.name!r} has no dtype because it is not an invoke arg"
                )
            tensors[tensor.name] = self.device_runtime.make_tensor(
                tensor.factory,
                shape,
                dtype=dtype_by_tensor[tensor.name],
            )
        return tensors

    def _invoke(self, tensors: Mapping[str, Any]) -> Any:
        """Invoke the configured public operator with positional tensor inputs.

        Args:
            tensors: Tensor mapping returned by :meth:`_make_tensors`.

        Returns:
            The public FlagGems callable's result without post-processing.

        Notes:
            Argument order is the validated YAML ``benchmark.invoke.args``
            order.  Keyword arguments and YAML-selected callables are unsupported
            by design.
        """
        return self.operator(
            *(tensors[reference.name] for reference in self.spec.benchmark.args)
        )

    def _output_dtypes(self, value: Any) -> list[str]:
        """Collect tensor result dtypes in stable mapping/sequence preorder."""
        if hasattr(value, "dtype"):
            return [normalize_dtype_name(value.dtype)]
        if isinstance(value, Mapping):
            result = []
            for item in value.values():
                result.extend(self._output_dtypes(item))
            return result
        if isinstance(value, (tuple, list)):
            result = []
            for item in value:
                result.extend(self._output_dtypes(item))
            return result
        return []

    @staticmethod
    def _single_config_quantiles(timings: Any) -> tuple[float, float, float]:
        """Return p20/p50/p80 from one LibTuner timing mapping.

        Args:
            timings: Mapping produced by ``LibTuner._bench`` for an exhaustive
                one-config policy call. Triton stores samples in p50, p20, p80
                order because Autotuner requests quantiles in that order.

        Returns:
            Finite positive ``(p20, p50, p80)`` values in milliseconds.

        Raises:
            BenchmarkExecutionError: If the selected-config pass did not
            produce exactly one complete finite timing tuple.
        """
        if not isinstance(timings, Mapping) or len(timings) != 1:
            raise BenchmarkExecutionError(
                "selected-config LibTuner benchmark must produce exactly one timing"
            )
        raw = next(iter(timings.values()))
        samples = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        if len(samples) < 3:
            raise BenchmarkExecutionError(
                "selected-config LibTuner benchmark returned fewer than three quantiles"
            )
        try:
            p50, p20, p80 = (float(samples[index]) for index in range(3))
        except (TypeError, ValueError) as exc:
            raise BenchmarkExecutionError(
                "selected-config LibTuner benchmark returned non-numeric quantiles"
            ) from exc
        if not all(math.isfinite(value) and value > 0 for value in (p20, p50, p80)):
            raise BenchmarkExecutionError(
                "selected-config LibTuner benchmark returned non-finite quantiles"
            )
        return p20, p50, p80

    def _benchmark_selected_config(
        self,
        *,
        tuner: Any,
        best_config: Any,
        warmup: int,
        iterations: int,
        trials: int,
    ) -> tuple[float, float, float]:
        """Freshly benchmark one selected config through LibTuner.

        The preceding public operator call has already constructed output
        descriptors and captured trusted low-level arguments in LibTuner.
        :meth:`LibTuner.benchmark_config` reuses that context, bypasses policy
        and persistent caches, and executes only the fixed-config kernel path.
        Repeated trials are summarized with a median for each quantile.

        Args:
            tuner: Bound LibTuner that owns the selected config and benchmarker.
            best_config: Config selected by the preceding policy call.
            warmup: Triton warmup duration in milliseconds for every trial.
            iterations: Triton repetition duration in milliseconds per trial.
            trials: Positive number of independent LibTuner measurements.

        Returns:
            Median ``(p20, p50, p80)`` latency in milliseconds.

        Limitations:
            This measures the fixed-config kernel path used by LibTuner, not
            allocation and Python overhead of the complete public operator.
        """
        trial_quantiles: list[tuple[float, float, float]] = []
        for _ in range(trials):
            samples = tuner.benchmark_config(
                best_config,
                warmup=warmup,
                rep=iterations,
                quantiles=(0.5, 0.2, 0.8),
            )
            trial_quantiles.append(
                self._single_config_quantiles(
                    {best_config: samples}
                )
            )

        return tuple(
            statistics.median(values)
            for values in zip(*trial_quantiles)
        )

    def benchmark(
        self,
        payload: Mapping[str, Any],
        *,
        dtype_names: Sequence[str],
        warmup: int,
        iterations: int,
        tuning_run_mode: str,
        latency_warmup: int,
        latency_iterations: int,
        latency_trials: int,
        gpu_token: str,
        worker_id: int,
    ) -> dict[str, Any]:
        """Tune one configured shape and return a stable generic result row.

        Args:
            payload: Prepared task mapping from
                :func:`prepare_benchmark_case`.
            dtype_name: Attribute name resolved from ``torch`` (for example,
                ``'bfloat16'``).
            warmup: Warmup milliseconds/iterations forwarded to Triton's
                ``do_bench`` according to the installed Triton API.
            iterations: Measurement repetition setting forwarded as ``rep``.
            tuning_run_mode: Explicit LibTuner config-selection mode forwarded
                by the worker scheduler.
            latency_warmup: Fresh selected-config warmup duration in milliseconds.
            latency_iterations: Fresh selected-config measurement duration in
                milliseconds per trial.
            latency_trials: Number of fresh selected-config LibTuner trials.
            gpu_token: Parent-visible GPU token recorded in output metadata.
            worker_id: Scheduler worker index used in results and progress logs.

        Returns:
            A JSON-compatible success row containing shape/variant identity,
            device metadata, first-call and steady-state latency, cache counters,
            the selected config, and optional explicit-candidate timings.

        Raises:
            BenchmarkExecutionError: If planner/worker YAML hashes differ, tuner
            binding fails, the expected variant is not executed, or explicit
            candidate mode yields no timing records.  Dependency, allocation,
            compilation, and kernel errors otherwise propagate to the scheduler.

        Implementation:
            The first task for a model ID expands and saves its baseline config
            space.  Each task installs either that space or reconstructed
            explicit configs, clears LibEntry dispatch and transient tuner state,
            constructs tensors, scopes the requested LibTuner run mode, and
            times the synchronized first public call.
            Explicit-config mode temporarily wraps ``tuner.do_bench`` to enforce
            requested warmup/repetition values and emit progress. Its selected
            timing is already a LibTuner result. Normal Pretune performs a
            second, one-config LibTuner pass with cache reads/writes disabled and
            reports the median p20/p50/p80 across the requested fresh trials.

        Notes:
            ``benchmark_success_count`` means finite cache-miss benchmark calls,
            while ``benchmark_cache_hit_count`` means reused per-config latency
            entries.  They are not candidate counts.  Exact explicit-config
            collection requires a fresh benchmark database when cached latency
            settings or source state are not compatible with the requested run.
        """
        if payload.get("config_sha256") != self.spec.source_sha256:
            raise BenchmarkExecutionError(
                "operator config changed after task preparation"
            )
        values = dict(payload["values"])
        variant = str(payload["variant"])
        input_dtypes = [normalize_dtype_name(name) for name in dtype_names]
        torch_dtypes = [self.device_runtime.dtype(name) for name in input_dtypes]
        kernel, tuner = self._find_tuner(variant)
        identity = (self.spec.op_id, variant)
        if identity not in self.base_states:
            tuner.apply_flagtune()
            self.base_states[identity] = (list(tuner.configs), list(tuner.strategy))
        base_configs, base_strategy = self.base_states[identity]
        config_records = payload.get("configs")
        active_configs = (
            base_configs
            if config_records is None
            else self._make_configs(config_records, tuner)
        )
        tuner._set_configs_and_strategy(active_configs, base_strategy)
        from flag_gems.utils.libentry import (
            LibTunerRunMode,
            clear_libentry_dispatch_cache,
        )

        clear_libentry_dispatch_cache(kernel)
        try:
            selected_run_mode = LibTunerRunMode(tuning_run_mode)
        except ValueError as exc:
            raise BenchmarkExecutionError(
                f"unsupported LibTuner run mode {tuning_run_mode!r}"
            ) from exc
        for attr in ("bench_time", "configs_timings", "best_config"):
            tuner.__dict__.pop(attr, None)

        tensors = self._make_tensors(values, torch_dtypes)
        self.device_runtime.synchronize()
        first_start = time.perf_counter()
        original_do_bench = tuner.do_bench
        progress_interval = _progress_interval()
        measured_configs = 0
        progress_start = time.perf_counter()
        if config_records is not None and progress_interval > 0:
            print(
                f"worker={worker_id} case={describe_benchmark_case(payload)} "
                f"config_benchmark_start candidates={len(active_configs)} "
                f"warmup={warmup} iter={iterations}",
                flush=True,
            )

        def configured_do_bench(kernel_call: Any, quantiles: Any) -> Any:
            """Benchmark one candidate with requested settings and report progress."""
            nonlocal measured_configs
            try:
                return self._triton_module.testing.do_bench(
                    kernel_call,
                    warmup=warmup,
                    rep=iterations,
                    quantiles=quantiles,
                )
            finally:
                measured_configs += 1
                should_report = (
                    config_records is not None
                    and progress_interval > 0
                    and (
                        measured_configs == 1
                        or measured_configs % progress_interval == 0
                        or measured_configs == len(active_configs)
                    )
                )
                if should_report:
                    elapsed = max(time.perf_counter() - progress_start, 1e-9)
                    rate = measured_configs / elapsed
                    remaining = max(len(active_configs) - measured_configs, 0)
                    eta = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"worker={worker_id} case={describe_benchmark_case(payload)} "
                        f"config_progress={measured_configs}/{len(active_configs)} "
                        f"rate={rate:.2f}_config/s eta={eta:.1f}s",
                        flush=True,
                    )

        # Both exhaustive Train and policy-forced Pretune use the public CLI
        # settings for any candidate cache miss. Pretune then performs its
        # separate fresh selected-config measurement with --latency-*.
        tuner.do_bench = configured_do_bench
        try:
            with tuner.use_benchmark_protocol(warmup, iterations):
                with tuner.use_run_mode(selected_run_mode):
                    output = self._invoke(tensors)
        finally:
            tuner.do_bench = original_do_bench
        self.device_runtime.synchronize()
        first_call_ms = (time.perf_counter() - first_start) * 1000.0
        best_config = getattr(tuner, "best_config", None)
        if best_config is None:
            raise BenchmarkExecutionError(
                f"public {self.spec.public_operator_name} call did not execute expected variant {variant!r}"
            )
        benchmark_success_count = int(getattr(tuner, "benchmark_success_count", 0))
        benchmark_cache_hit_count = int(getattr(tuner, "benchmark_cache_hit_count", 0))
        timings = getattr(tuner, "configs_timings", None)
        config_timings = _serialize_config_timings(timings)
        if config_records is not None and not config_timings:
            raise BenchmarkExecutionError(
                "explicit-config benchmark produced no per-config timings; use a fresh "
                "training database or benchmark-cache-only mode"
            )
        bench_time = getattr(tuner, "bench_time", None)
        if bench_time is not None:
            cache_hit: Optional[bool] = False
            tuning_time_ms: Optional[float] = float(bench_time) * 1000.0
        elif len(active_configs) > 1:
            cache_hit = True
            tuning_time_ms = None
        else:
            cache_hit = None
            tuning_time_ms = None
        if config_records is not None:
            if not isinstance(timings, Mapping) or best_config not in timings:
                raise BenchmarkExecutionError(
                    "LibTuner config search did not retain the selected config timing"
                )
            latency_p20, latency_p50, latency_p80 = self._single_config_quantiles(
                {best_config: timings[best_config]}
            )
            latency_source = "libtuner_config_search"
            reported_latency_trials = 1
        else:
            latency_p20, latency_p50, latency_p80 = self._benchmark_selected_config(
                tuner=tuner,
                best_config=best_config,
                warmup=latency_warmup,
                iterations=latency_iterations,
                trials=latency_trials,
            )
            latency_source = "libtuner_selected_config_fresh"
            reported_latency_trials = latency_trials
        timed_config_count = len(timings) if isinstance(timings, Mapping) else None
        if config_records is not None:
            print(
                f"worker={worker_id} case={describe_benchmark_case(payload)} "
                f"config_benchmark_complete candidates={len(active_configs)} "
                f"latency_cache_hits={benchmark_cache_hit_count} "
                f"new_finite_benchmarks={benchmark_success_count}",
                flush=True,
            )
        shape = [values[name] for name in self.spec.shape.identity]
        output_dtypes = self._output_dtypes(output)
        if not output_dtypes:
            raise BenchmarkExecutionError(
                "public operator returned no tensor output for dtype identity"
            )
        ordered_dtypes = [*input_dtypes, *output_dtypes]
        device_name = self.device_runtime.descriptor.device_name
        gpu = self.device_runtime.metadata(0)
        return {
            "source_index": payload["source_index"],
            "selected_index": payload["selected_index"],
            "op_id": self.spec.op_id,
            "op_name": self.spec.public_operator_name,
            "variant": variant,
            "shape": shape,
            "shape_key": ",".join(str(value) for value in shape),
            **values,
            "Count": payload.get("count"),
            "input_dtypes": input_dtypes,
            "output_dtypes": output_dtypes,
            "dtype_key": make_dtype_key(ordered_dtypes),
            "gpu": gpu_token,
            "gpu_name": device_name,
            "gpu_key": gpu["gpu_key"],
            "gpu_metadata": gpu,
            "worker_id": worker_id,
            "status": "ok",
            "cache_hit": cache_hit,
            "first_call_ms": first_call_ms,
            "tuning_time_ms": tuning_time_ms,
            "latency_p20_ms": latency_p20,
            "latency_p50_ms": latency_p50,
            "latency_p80_ms": latency_p80,
            "latency_source": latency_source,
            "latency_warmup_ms": (
                warmup if config_records is not None else latency_warmup
            ),
            "latency_iterations_ms": (
                iterations if config_records is not None else latency_iterations
            ),
            "latency_trial_count": reported_latency_trials,
            "candidate_config_count": len(active_configs),
            "timed_config_count": timed_config_count,
            "benchmark_cache_hit_count": benchmark_cache_hit_count,
            "benchmark_success_count": benchmark_success_count,
            "best_config": config_to_record(best_config),
            "config_timings": config_timings if config_records is not None else None,
            "error": "",
        }

    def failure_result(
        self,
        payload: Mapping[str, Any],
        *,
        dtype_names: Sequence[str],
        gpu_token: str,
        worker_id: int,
        exc: BaseException,
    ) -> dict[str, Any]:
        """Serialize one per-case exception into the stable result schema.

        Args:
            payload: Prepared task that failed during worker execution.
            dtype_names: Requested ordered input tensor dtype names.
            gpu_token: Parent-visible GPU token assigned to this worker.
            worker_id: Scheduler worker index.
            exc: Original exception, including its worker traceback.

        Returns:
            A JSON-compatible row with ``status='failed'``, preserved workload
            identity and candidate count, unavailable metrics set to ``None``,
            and the formatted exception traceback in ``error``.

        Notes:
            This method does not log, retry, or suppress the exception by itself.
            The subprocess loop calls it only when batch fail-fast is disabled.
        """
        values = dict(payload["values"])
        shape = [values[name] for name in self.spec.shape.identity]
        configs = payload.get("configs")
        return {
            "source_index": payload["source_index"],
            "selected_index": payload["selected_index"],
            "op_id": self.spec.op_id,
            "op_name": self.spec.public_operator_name,
            "variant": payload["variant"],
            "shape": shape,
            "shape_key": ",".join(str(value) for value in shape),
            **values,
            "Count": payload.get("count"),
            "input_dtypes": [normalize_dtype_name(name) for name in dtype_names],
            "output_dtypes": [],
            "dtype_key": None,
            "gpu": gpu_token,
            "gpu_name": None,
            "gpu_key": None,
            "gpu_metadata": None,
            "worker_id": worker_id,
            "status": "failed",
            "cache_hit": None,
            "first_call_ms": None,
            "tuning_time_ms": None,
            "latency_p20_ms": None,
            "latency_p50_ms": None,
            "latency_p80_ms": None,
            "latency_source": None,
            "latency_warmup_ms": None,
            "latency_iterations_ms": None,
            "latency_trial_count": None,
            "candidate_config_count": len(configs) if configs is not None else None,
            "timed_config_count": None,
            "benchmark_cache_hit_count": None,
            "benchmark_success_count": None,
            "best_config": None,
            "config_timings": None,
            "error": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
