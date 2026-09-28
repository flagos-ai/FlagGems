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

"""Cost Model loading, candidate selection and process-local AUTO fallback state.

The tuner supplies dtype inference and legacy policy callbacks so this module
never imports libentry or takes ownership of ordinary tuner caches.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import triton

from flag_gems import runtime
from flag_gems.flagtune.inference.status import exception_reason, print_status

_FLAGTUNE_PROPOSER_POOL: Dict[Any, Any] = {}
_FLAGTUNE_VARIANT_INFO_POOL: Dict[Any, Any] = {}
_FLAGTUNE_AVAILABILITY: Optional[Tuple[bool, Optional[BaseException]]] = None
# One AUTO failure disables the affected operator variant in this process.
_COST_MODEL_DISABLED_OPS: set[tuple[Optional[str], Optional[str]]] = set()
_COST_MODEL_IDENTITIES: Dict[Any, Any] = {}
_MODEL_LOAD_STARTED: set[Any] = set()
_MODEL_SOURCE_CONFIGURED = False
_DEFAULT_FLAGTUNE_LOCAL_MANIFEST = Path(__file__).with_name("manifest.json")


def _configure_flagtune_model_source() -> None:
    """Set Flaggems' model source defaults once before FlagTree loads models.

    Explicit local manifests and remote URLs remain authoritative, including
    invalid/empty settings (FlagTree reports those errors). Otherwise use the
    manifest shipped with FlagGems. This does not change FlagTree's standalone
    defaults. FlagTree selects the highest SemVer package listed in it unless
    the user explicitly pins a version or disables latest-version selection.
    """
    global _MODEL_SOURCE_CONFIGURED
    if _MODEL_SOURCE_CONFIGURED:
        return
    if not any(
        name in os.environ
        for name in ("FLAGTUNE_LOCAL_MANIFEST", "FLAGTUNE_MANIFEST_URL")
    ):
        os.environ.setdefault(
            "FLAGTUNE_LOCAL_MANIFEST", str(_DEFAULT_FLAGTUNE_LOCAL_MANIFEST)
        )
    os.environ.setdefault("FLAGTUNE_MODEL_DOWNLOAD_LATEST", "1")
    _MODEL_SOURCE_CONFIGURED = True


def is_auto_disabled(op_id: Optional[str], variant: Optional[str] = None) -> bool:
    """Return whether AUTO is fused for this variant until process restart."""
    return (op_id, variant) in _COST_MODEL_DISABLED_OPS


@contextmanager
def cost_model_boundary(phase: str) -> Iterator[None]:
    """Normalize FlagTree failures while leaving legacy work untouched."""
    try:
        from triton.flagtune.runtime.errors import (
            BenchmarkError,
            ContractExecutionError,
            ModelValidationError,
            flagtune_error_boundary,
        )
    except ImportError:
        yield
        return
    error_type = {
        "preload": ModelValidationError,
        "postload": ContractExecutionError,
        "benchmark": BenchmarkError,
    }[phase]
    with flagtune_error_boundary(error_type):
        yield


class NoModelBundleMissingError(Exception):
    """Placeholder for FlagTree versions without ModelBundleMissingError."""


def model_bundle_missing_error() -> Type[BaseException]:
    try:
        from triton.flagtune.runtime.model_loader import ModelBundleMissingError
    except ImportError:
        return NoModelBundleMissingError
    return ModelBundleMissingError


def flagtune_error_types() -> Tuple[Type[BaseException], ...]:
    """Return unified FlagTree errors with compatibility for older releases."""
    error_types: list[Type[BaseException]]
    try:
        from triton.flagtune.runtime.errors import FlagTuneError
    except ImportError:
        error_types = [FileNotFoundError, model_bundle_missing_error()]
    else:
        error_types = [FlagTuneError]

    # Older FlagTree releases expose device probing and package validation
    # failures separately and do not provide ``runtime.errors.FlagTuneError``.
    # Keep these narrow compatibility exceptions in the AUTO fallback set
    # without swallowing arbitrary device, compiler, or benchmark errors.
    try:
        from triton.flagtune.runtime.device import UnsupportedFlagTuneDeviceError
    except ImportError:
        pass
    else:
        if UnsupportedFlagTuneDeviceError not in error_types:
            error_types.append(UnsupportedFlagTuneDeviceError)
    try:
        from triton.flagtune.runtime.model_loader import IncompatibleModelError
    except ImportError:
        pass
    else:
        if IncompatibleModelError not in error_types:
            error_types.append(IncompatibleModelError)
    return tuple(error_types)


def configs_to_dicts(
    configs: Iterator[triton.Config], param_fields: List[str]
) -> List[Dict[str, Any]]:
    """Project Triton configs onto the FlagTree proposer dictionary schema."""
    result = []
    for cfg in configs:
        values: Dict[str, Any] = {}
        kwargs = getattr(cfg, "kwargs", {})
        for field in param_fields:
            if field in kwargs:
                values[field] = kwargs[field]
            elif hasattr(cfg, field):
                values[field] = getattr(cfg, field)
        for attr in ("num_warps", "num_stages", "num_ctas"):
            if hasattr(cfg, attr):
                values[attr] = int(getattr(cfg, attr))
        if values:
            result.append(values)
    return result


def benchmark_config(
    bench_fn: Callable[[triton.Config], List[float]], config: triton.Config
) -> List[float]:
    """Benchmark one Cost Model candidate and reject invalid timings."""
    with cost_model_boundary("benchmark"):
        samples = bench_fn(config)
        if not samples or not all(math.isfinite(float(value)) for value in samples):
            raise RuntimeError("Cost Model candidate has no finite benchmark latency")
        return samples


def ensure_proposer(identity: Any):
    """Cache the proposer and variant by complete identity and model version."""
    _configure_flagtune_model_source()
    from triton.flagtune.runtime.proposer import load_model_bundle, make_config_proposer

    if identity not in _MODEL_LOAD_STARTED:
        _MODEL_LOAD_STARTED.add(identity)
        print_status("Model loading", identity=identity)
    loaded = load_model_bundle(
        identity.op_id,
        identity.variant,
        platform_key=identity.platform_key,
        dtype_key=identity.dtype_key,
    )
    cache_key = (identity, loaded.model_version)
    if cache_key not in _FLAGTUNE_PROPOSER_POOL:
        _FLAGTUNE_PROPOSER_POOL[cache_key] = make_config_proposer(
            identity.op_id,
            identity.variant,
            platform_key=identity.platform_key,
            dtype_key=identity.dtype_key,
        )
        _FLAGTUNE_VARIANT_INFO_POOL[cache_key] = loaded.variant
        print_status("Model loaded", identity=identity, version=loaded.model_version)
    return _FLAGTUNE_PROPOSER_POOL[cache_key], _FLAGTUNE_VARIANT_INFO_POOL[cache_key]


def identity_cache_key(
    tuner: Any, arguments: Dict[str, Any], vendor_name: str
) -> Tuple[str, str, str, str]:
    """Build the pre-load identity cache key without model-source access."""
    op_id = str(getattr(tuner, "_flagtune_op_id", "<unknown>"))
    variant = str(getattr(tuner, "_flagtune_variant", "<unknown>"))
    tensors = []
    for name in tuner.arg_names:
        value = arguments.get(name)
        if not hasattr(value, "device"):
            value = getattr(value, "base", value)
        if hasattr(value, "dtype") and hasattr(value, "device"):
            tensors.append(value)
    platform = f"{vendor_name}:" + ",".join(str(tensor.device) for tensor in tensors)
    dtype_key = ",".join(str(tensor.dtype) for tensor in tensors)
    return platform, op_id, variant, dtype_key


def runtime_candidates(tuner: Any, op_name: str, kwargs: Dict[str, Any]):
    """Resolve Cost Model's Expanded + Default runtime candidate domain."""
    runtime_configs, _ = tuner._flagtune_configs_for_mode(
        op_name, runtime.TuningMode.EXPANDED
    )
    if not runtime_configs:
        raise RuntimeError(
            f"FlagTune runtime Expanded + Default config space is empty for {op_name}"
        )
    candidate_configs = list(runtime_configs)
    early_config_prune = getattr(tuner, "early_config_prune", None)
    if early_config_prune is not None:
        candidate_configs = list(
            early_config_prune(
                candidate_configs, {**(tuner.nargs or {}), **kwargs}, **kwargs
            )
        )
    if not candidate_configs:
        raise RuntimeError(
            f"FlagTune early_config_prune returned no legal configs for {op_name}"
        )
    return candidate_configs


def run_proposer(
    tuner: Any,
    bench_fn: Callable[[triton.Config], List[float]],
    candidate_configs: List[triton.Config],
    arguments: Dict[str, Any],
    loaded: Tuple[Any, Any, Any],
):
    """Run a loaded proposer over the legal runtime candidate domain."""
    model_identity, proposer, variant_info = loaded
    identity = model_identity.artifact_key
    fields = tuple(variant_info.param_names)
    to_config = variant_info.to_config
    initial = configs_to_dicts(candidate_configs, list(fields))
    legal_keys = {tuple(config[name] for name in fields) for config in initial}

    def checked_config(config_dict):
        key = tuple(config_dict[name] for name in fields)
        if key not in legal_keys:
            raise ValueError("proposer candidate is outside the pruned runtime domain")
        config = to_config(config_dict)
        if config.pre_hook is None:
            config.pre_hook = getattr(tuner, "_flagtune_pre_hook", None)
        return config

    adapter = make_bench_adapter(bench_fn, checked_config)
    meta = {
        "op_id": model_identity.op_id,
        "variant": model_identity.variant,
        "platform_key": model_identity.platform_key,
        "dtype_key": model_identity.dtype_key,
    }
    result_dicts = proposer(adapter, arguments, initial, meta)
    if not result_dicts:
        raise RuntimeError(f"FlagTune proposer returned no configs for {identity}")

    timings: Dict[triton.Config, float] = {}
    best_config = None
    best_latency = float("inf")
    for result in result_dicts:
        config = checked_config(result)
        latency = float(benchmark_config(bench_fn, config)[0])
        timings[config] = latency
        if latency < best_latency:
            best_latency = latency
            best_config = config
    if best_config is None:
        raise RuntimeError(
            f"FlagTune proposer produced no benchmarkable configs for {identity}"
        )
    return best_config, timings


def available() -> Tuple[bool, Optional[BaseException]]:
    global _FLAGTUNE_AVAILABILITY
    if _FLAGTUNE_AVAILABILITY is None:
        try:
            from triton.flagtune.runtime.proposer import (  # noqa: F401
                load_model_bundle,
                make_config_proposer,
            )

            _FLAGTUNE_AVAILABILITY = (True, None)
        except Exception as exc:
            _FLAGTUNE_AVAILABILITY = (False, exc)
    return _FLAGTUNE_AVAILABILITY


def make_bench_adapter(
    bench_fn: Callable[[triton.Config], List[float]],
    to_config: Callable[[Dict[str, Any]], triton.Config],
):
    """Adapt a LibTuner benchmark callable to FlagTune's dictionary contract.

    Args:
        bench_fn: Callable accepting ``triton.Config`` and returning latency
            samples, normally backed by LibTuner's ``BenchmarkCache``.
        to_config: Variant converter from a complete parameter dictionary to a
            fresh ``triton.Config``.

    Returns:
        A ``BenchmarkFn(config_dict, n_runs=None)`` closure.  It converts the
        dictionary and forwards the Config to ``bench_fn``.

    Notes:
        ``n_runs`` is accepted for proposer compatibility but ignored.  The
        adapter performs no database access itself; caching remains entirely in
        ``bench_fn``. Conversion and benchmark failures propagate to the
        policy; a non-finite latency is also a failure, not a usable candidate.
    """

    def adapted(config_dict: Dict[str, Any], n_runs=None) -> List[float]:
        """Convert and benchmark one proposer candidate; ``n_runs`` is ignored."""
        return benchmark_config(bench_fn, to_config(config_dict))

    return adapted


def load_model(self, identity_key, arguments, infer_tensor_dtypes):
    """Resolve identity once per device/dtype and load its versioned proposer."""
    op_id = getattr(self, "_flagtune_op_id", None)
    variant = getattr(self, "_flagtune_variant", None)
    is_available, exc = available()
    if not is_available:
        raise RuntimeError(
            "FlagTune is enabled but the FlagTree runtime is unavailable"
        ) from exc

    from triton.flagtune.contract.identity import (
        ModelIdentity,
        discover_gpu_metadata,
        make_dtype_key,
    )

    model_identity = _COST_MODEL_IDENTITIES.get(identity_key)
    if model_identity is None:
        dtype_resolver = getattr(self, "_flagtune_dtype_resolver", None)
        if dtype_resolver is not None:
            dtypes = tuple(dtype_resolver(arguments))
        else:
            dtypes = infer_tensor_dtypes(
                arguments[name] for name in self.arg_names if name in arguments
            )
        if not dtypes:
            raise ValueError("no tensor dtypes available for FlagTune identity")
        gpu = discover_gpu_metadata()
        model_identity = ModelIdentity(
            platform_key=str(gpu["platform_key"]),
            op_id=op_id,
            variant=variant,
            dtype_key=make_dtype_key(dtypes),
        )
        _COST_MODEL_IDENTITIES[identity_key] = model_identity
    proposer, variant_info = ensure_proposer(model_identity)
    return model_identity, proposer, variant_info


def run_policy(
    self,
    bench_fn: Callable[[triton.Config], List[float]],
    configs: Iterator[triton.Config],
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    *,
    supports_cost_model: bool,
    infer_tensor_dtypes: Callable,
    default_policy: Callable,
    legacy_fallback: Callable,
) -> Tuple[triton.Config, Dict[str, float]]:
    """Run Cost Model; an AUTO failure fuses this op_id across the process."""
    op_name = (
        getattr(self, "_flagtune_op_name", None)
        or getattr(self, "_flagtune_expand_op_name", None)
        or getattr(self, "__name__", "unknown")
    )
    op_id = getattr(self, "_flagtune_op_id", None)
    intent = runtime.resolve_cost_model_intent(supports_cost_model=supports_cost_model)
    variant = getattr(self, "_flagtune_variant", None)
    disabled_key = (op_id, variant)
    if (
        intent is runtime.CostModelIntent.AUTO
        and disabled_key in _COST_MODEL_DISABLED_OPS
    ):
        # Check before tensor inspection, candidate resolution or model loading.
        return legacy_fallback(self, bench_fn, args, kwargs, op_name)
    mode = runtime.resolve_tuning_mode(op_name, supports_cost_model=supports_cost_model)
    if mode is not runtime.TuningMode.COST_MODEL:
        return default_policy(self, bench_fn, configs, args, kwargs)
    intent_required = intent is runtime.CostModelIntent.REQUIRED
    arguments = {**(self.nargs or {}), **kwargs}
    identity_key = identity_cache_key(self, arguments, runtime.device.vendor_name)

    phase = "preload"
    try:
        with cost_model_boundary(phase):
            candidates = runtime_candidates(self, op_name, kwargs)
            # A single legal runtime candidate needs neither identity discovery
            # nor a model. Its ordinary benchmark failure is not a CM failure.
            loaded = (
                load_model(self, identity_key, arguments, infer_tensor_dtypes)
                if len(candidates) > 1
                else None
            )
        if loaded is not None:
            phase = "postload"
            previous_strict = getattr(self, "_flagtune_strict_benchmark", False)
            self._flagtune_strict_benchmark = True
            try:
                with cost_model_boundary(phase):
                    return run_proposer(self, bench_fn, candidates, arguments, loaded)
            finally:
                self._flagtune_strict_benchmark = previous_strict
    except flagtune_error_types() as exc:
        if intent_required:
            print_status(
                "REQUIRED Cost Model failed",
                identity=_COST_MODEL_IDENTITIES.get(identity_key, identity_key),
                phase=phase,
                reason=exception_reason(exc),
                action="raising; no fallback",
            )
            raise
        if disabled_key not in _COST_MODEL_DISABLED_OPS:
            _COST_MODEL_DISABLED_OPS.add(disabled_key)
            fallback_mode = runtime.resolve_tuning_mode(
                op_name, supports_cost_model=False
            )
            print_status(
                "AUTO fallback",
                identity=_COST_MODEL_IDENTITIES.get(identity_key, identity_key),
                phase=phase,
                reason=exception_reason(exc),
                fallback_mode=fallback_mode.value,
                action="AUTO disabled for this variant in this process until restart",
            )
    else:
        return default_policy(self, bench_fn, candidates, args, kwargs)
    return legacy_fallback(self, bench_fn, args, kwargs, op_name)
