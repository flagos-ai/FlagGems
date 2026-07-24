"""Public APIs for offline FlagTune benchmark workflows.

Purpose:
    This package exposes the operator-independent batch benchmark API used by
    :mod:`flag_gems.utils.flagtune.pretune`. Callers submit shapes and optional
    config collections together with a safe data-driven operator YAML.

Inputs and outputs:
    The main input is a sequence of shape/config cases plus worker, GPU,
    database, and timing options.  The API returns either a
    :class:`BenchmarkBatchResult` with execution metadata or its ordered result
    rows.

Implementation:
    Scheduling and process management live in ``benchmark``; schema compilation
    and execution live in ``operator_config`` and ``executor``.

Limitations:
    GPU execution requires a CUDA-capable FlagGems runtime. YAML may use only
    whitelisted tensor factories and the public FlagGems operator namespace.
"""

from .benchmark import (
    BenchmarkBatchResult,
    BenchmarkCase,
    BenchmarkError,
    BenchmarkTask,
    benchmark_shape_configs,
    run_shape_config_benchmarks,
)

__all__ = [
    "BenchmarkBatchResult",
    "BenchmarkCase",
    "BenchmarkError",
    "BenchmarkTask",
    "benchmark_shape_configs",
    "run_shape_config_benchmarks",
]
