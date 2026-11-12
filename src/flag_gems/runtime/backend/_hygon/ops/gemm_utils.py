# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import cache
from math import inf

import torch
from triton.compiler.errors import CompilationError
from triton.runtime import errors

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
# Leave room for a complete padded tile in signed i32 coordinate arithmetic.
I32_GEMM_COORD_LIMIT = 1 << 30
_CANDIDATE_FAILURES = (CompilationError, errors.OutOfResources) + tuple(
    getattr(errors, name)
    for name in ("BackendCompilerError", "HSACOError")
    if hasattr(errors, name)
)


def gemm_has_wide_storage(*tensors):
    # Match FlagTree HCU's signed 32-bit byte-range specialization. Using
    # storage size also accounts for strided views and nonzero storage offsets.
    return any(tensor.untyped_storage().nbytes() >= 2**31 for tensor in tensors)


def supports_hcu_options(options):
    try:
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return False
    return all(name in HIPOptions.__dataclass_fields__ for name in options)


def try_gemm_candidate(call, *args, **kwargs):
    """Skip known compile/resource failures while propagating device errors."""
    try:
        return True, call(*args, **kwargs)
    except _CANDIDATE_FAILURES as error:
        logger.debug("Hygon GEMM candidate unavailable: %s", error)
    except RuntimeError as error:
        if str(error) != "PassManager::run failed":
            raise
        logger.debug("Hygon GEMM candidate compilation failed: %s", error)
    return False, None


def gemm_replay_benchmarker(warmup_ms, measurement_ms, n_retries):
    from triton.flagtune.runtime.benchmark_protocol import resolve_benchmarker

    return resolve_benchmarker(
        "replay", warmup_ms=warmup_ms, measurement_ms=measurement_ms, n_retries=n_retries
    ).benchmark


def benchmark_gemm_candidate(bench, call, *, optional):
    if optional:
        succeeded, elapsed = try_gemm_candidate(bench, call, quantiles=(0.5,))
        return float(elapsed) if succeeded else inf
    return float(bench(call, quantiles=(0.5,)))


@cache
def _capture_entry(kernel):
    return libentry()(kernel)


def capture_gemm_fallback(entry, args, kwargs):
    """Capture a cold key using a fixed Default config outside the tune caches."""
    if not torch.cuda.is_current_stream_capturing():
        return None

    tuner = entry.fn
    named = dict(zip(tuner.arg_names, args))
    named.update((key, value) for key, value in kwargs.items() if key in tuner.arg_names)
    if tuner._run_mode == "normal" and tuner.get_key(named) in tuner.cache:
        return None

    configs = [c for c in tuner._flagtune_default_configs if not c.kwargs.get("ASYNC_LOAD")]
    if tuner.early_config_prune is not None:
        configs = tuner.early_config_prune(configs, named)
    # The Hygon GEMM Default fallbacks do not require a per-config pre-hook.
    config = configs[0]
    options = config.all_kwargs()
    kernel, meta = _capture_entry(entry.jit_function).run(*args, **kwargs, **options)
    return kernel, {**meta, **options}
