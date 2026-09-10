"""Public, operator-independent benchmark collection API.

The scheduler owns multiprocessing, worker logs, and per-worker SQLite shards;
it returns structured rows and metadata without knowing Pretune CLI details.
"""

from .scheduler import (
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
