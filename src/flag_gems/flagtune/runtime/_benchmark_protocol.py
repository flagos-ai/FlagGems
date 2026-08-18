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
"""Adapt official Triton benchmarkers to FlagGems' benchmark protocol."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

from triton.runtime.driver import driver


class BenchmarkMode(str, Enum):
    """Select ordinary event timing or a backend replay mechanism."""

    EVENT = "event"
    REPLAY = "replay"


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Describe the exact resolved benchmark semantics used by one tuner."""

    requested_mode: BenchmarkMode
    resolved_mode: BenchmarkMode
    implementation: str
    cache_policy: str
    warmup_ms: int
    measurement_ms: int
    n_retries: int
    per_replay_ms: float | None
    fallback_reason: str | None = None

    def cache_key(self) -> tuple[Any, ...]:
        """Return a stable identity suitable for persistent benchmark caches."""
        if self.resolved_mode is BenchmarkMode.EVENT:
            return ("triton_do_bench", self.warmup_ms, self.measurement_ms)
        return (
            self.implementation,
            self.warmup_ms,
            self.measurement_ms,
            self.n_retries,
            self.per_replay_ms,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe audit metadata."""
        return {
            "requested_mode": self.requested_mode.value,
            "resolved_mode": self.resolved_mode.value,
            "implementation": self.implementation,
            "cache_policy": self.cache_policy,
            "warmup_ms": self.warmup_ms,
            "measurement_ms": self.measurement_ms,
            "n_retries": self.n_retries,
            "per_replay_ms": self.per_replay_ms,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ResolvedBenchmarker:
    """Pair a two-argument Autotuner benchmark callable with its protocol."""

    protocol: BenchmarkProtocol
    benchmark: Callable[[Callable[..., Any], Sequence[float]], Sequence[float]]


_REPLAY_IMPLEMENTATIONS = {
    "triton.backends.nvidia.driver": "triton_cuda_graph_replay_v1",
    "triton.backends.amd.driver": "triton_hip_graph_replay_v1",
}
_OFFICIAL_TRITON_REPLAY_COUNT = 10


def _replay_implementation(active: Any) -> str | None:
    """Return the stable replay identity for a supported Triton driver."""
    return _REPLAY_IMPLEMENTATIONS.get(type(active).__module__)


def _validate_request(
    mode: BenchmarkMode | str,
    warmup_ms: int,
    measurement_ms: int,
    n_retries: int,
) -> BenchmarkMode:
    try:
        selected = BenchmarkMode(mode)
    except ValueError as exc:
        raise ValueError("benchmark mode must be 'event' or 'replay'") from exc
    if not isinstance(warmup_ms, int) or isinstance(warmup_ms, bool) or warmup_ms < 0:
        raise ValueError("benchmark warmup_ms must be a non-negative integer")
    if (
        not isinstance(measurement_ms, int)
        or isinstance(measurement_ms, bool)
        or measurement_ms <= 0
    ):
        raise ValueError("benchmark measurement_ms must be a positive integer")
    if not isinstance(n_retries, int) or isinstance(n_retries, bool) or n_retries <= 0:
        raise ValueError("benchmark n_retries must be a positive integer")
    return selected


def _supports_configurable_retries(benchmarker: Callable[..., Any]) -> bool:
    """Return whether a Triton graph helper accepts ``n_retries``."""
    try:
        return "n_retries" in inspect.signature(benchmarker).parameters
    except (TypeError, ValueError):
        return False


def resolve_benchmarker(
    mode: BenchmarkMode | str,
    *,
    warmup_ms: int,
    measurement_ms: int,
    n_retries: int = 10,
    allow_fallback: bool = True,
) -> ResolvedBenchmarker:
    """Resolve event/replay timing against an official Triton installation."""
    selected = _validate_request(mode, warmup_ms, measurement_ms, n_retries)
    active = driver.active
    if selected is BenchmarkMode.REPLAY:
        implementation = _replay_implementation(active)
        if implementation is not None:
            from triton.testing import do_bench_cudagraph

            configurable_retries = _supports_configurable_retries(do_bench_cudagraph)
            effective_retries = (
                n_retries if configurable_retries else _OFFICIAL_TRITON_REPLAY_COUNT
            )
            if not configurable_retries and n_retries != effective_retries:
                warnings.warn(
                    "official Triton's graph benchmark uses a fixed replay "
                    f"count of {effective_retries}; ignoring n_retries={n_retries}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            per_replay_ms = float(measurement_ms) / effective_retries

            def replay_benchmark(kernel_call, quantiles):
                kwargs = {
                    "rep": per_replay_ms,
                    "quantiles": quantiles,
                }
                if configurable_retries:
                    kwargs["n_retries"] = effective_retries
                return do_bench_cudagraph(kernel_call, **kwargs)

            return ResolvedBenchmarker(
                protocol=BenchmarkProtocol(
                    requested_mode=selected,
                    resolved_mode=BenchmarkMode.REPLAY,
                    implementation=implementation,
                    cache_policy="warm_l2",
                    warmup_ms=warmup_ms,
                    measurement_ms=measurement_ms,
                    n_retries=effective_retries,
                    per_replay_ms=per_replay_ms,
                ),
                benchmark=replay_benchmark,
            )
        if not allow_fallback:
            raise RuntimeError(
                "active Triton backend does not provide a replay benchmarker"
            )
        fallback_reason = "active Triton backend does not provide a replay benchmarker"
        warnings.warn(
            f"{fallback_reason}; falling back to event timing",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        fallback_reason = None

    event_benchmarker = active.get_benchmarker()

    def event_benchmark(kernel_call, quantiles):
        return event_benchmarker(
            kernel_call,
            warmup=warmup_ms,
            rep=measurement_ms,
            quantiles=quantiles,
        )

    return ResolvedBenchmarker(
        protocol=BenchmarkProtocol(
            requested_mode=selected,
            resolved_mode=BenchmarkMode.EVENT,
            implementation="triton_do_bench",
            cache_policy="cold_l2",
            warmup_ms=warmup_ms,
            measurement_ms=measurement_ms,
            n_retries=1,
            per_replay_ms=None,
            fallback_reason=fallback_reason,
        ),
        benchmark=event_benchmark,
    )
