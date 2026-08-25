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

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

USE_FLAGTUNE_ENV = "USE_FLAGTUNE"
USE_FLAGTUNE_COST_MODEL_ENV = "USE_FLAGTUNE_COST_MODEL"
FLAGTUNE_INCLUDE_ENV = "FLAGTUNE_INCLUDE"

_flagtune_op_registry = {}
_include_ops = None


def _platform_cost_model_available():
    """Lazily resolve the active platform package without creating import cycles."""
    from flag_gems.flagtune.runtime.model_package import (
        platform_model_package_available,
    )

    return platform_model_package_available()


class TuningMode(str, Enum):
    """Runtime configuration-selection path for one LibTuner operator."""

    DEFAULT = "default"
    EXPANDED = "expanded"
    COST_MODEL = "cost_model"


@dataclass(frozen=True)
class FlagTuneOpSpec:
    name: str
    default_enabled: bool = False
    description: str = ""


def _normalize_op_name(op_name):
    if not isinstance(op_name, str):
        raise TypeError("op_name must be a string")
    op_name = op_name.strip()
    if not op_name:
        raise ValueError("op_name must not be empty")
    return op_name


def register_flagtune_op(
    op_name,
    *,
    default=False,
    description="",
    replace=False,
):
    """Register an operator name that can be selected by flag_gems.flagtune."""
    name = _normalize_op_name(op_name)
    spec = FlagTuneOpSpec(
        name=name,
        default_enabled=bool(default),
        description=str(description or ""),
    )

    existing = _flagtune_op_registry.get(name)
    if existing is not None and not replace:
        if existing == spec:
            return existing
        raise ValueError(f"FlagTune op {name!r} is already registered")

    _flagtune_op_registry[name] = spec
    return spec


def get_flagtune_registry():
    return MappingProxyType(dict(_flagtune_op_registry))


def get_supported_flagtune_ops():
    return frozenset(_flagtune_op_registry)


def get_default_flagtune_include():
    return frozenset(
        name for name, spec in _flagtune_op_registry.items() if spec.default_enabled
    )


def _split_include(include):
    if include is None:
        return get_default_flagtune_include()
    if isinstance(include, str):
        include = include.replace(";", ",").split(",")

    try:
        ops = [str(op).strip() for op in include]
    except TypeError as err:
        raise TypeError(
            "include must be a comma-separated string or an iterable"
        ) from err

    return frozenset(op for op in ops if op)


def _normalize_include(include):
    ops = _split_include(include)
    supported_ops = get_supported_flagtune_ops()
    unsupported = sorted(ops - supported_ops)
    if unsupported:
        supported = ", ".join(sorted(supported_ops)) or "<none>"
        raise ValueError(
            f"Unsupported flagtune op(s): {', '.join(unsupported)}. "
            f"Supported ops: {supported}"
        )
    return ops


def _optional_binary_environment(name):
    """Return an optional 0/1 environment setting with strict validation."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1 when set")
    return value == "1"


def _use_flagtune_setting_from_env():
    """Return whether FlagTune is explicitly disabled or enabled."""
    return _optional_binary_environment(USE_FLAGTUNE_ENV)


def _expanded_from_env():
    """Return whether the legacy switch explicitly enables FlagTune."""
    return _use_flagtune_setting_from_env() is True


def flagtune_expanded_enabled():
    """Return whether global expanded search is enabled.

    This intentionally excludes per-operator include-list selection. It keeps
    decorators that recognize only ``USE_FLAGTUNE=1`` on their exact global
    enable semantics.
    """
    return _expanded_from_env()


def flagtune(include=None):
    """Enable runtime FlagTune for selected operators.

    Passing include=None enables the registry's default operators. Passing a
    string or iterable selects registered operators for capability-appropriate
    tuning: Expanded for an unadapted operator and Cost Model for an adapted
    operator. This API only updates the explicit include list; setting
    ``USE_FLAGTUNE=1`` enables every registered FlagTune operator.
    """
    global _include_ops
    _include_ops = _normalize_include(include)
    os.environ[FLAGTUNE_INCLUDE_ENV] = ",".join(sorted(_include_ops))


def _include_from_env():
    include = os.environ.get(FLAGTUNE_INCLUDE_ENV)
    if include is None:
        return frozenset()
    try:
        return _normalize_include(include)
    except (TypeError, ValueError) as err:
        warnings.warn(f"Invalid FlagGems FlagTune include list: {err}")
        return frozenset()


def get_flagtune_include():
    if _include_ops is not None:
        return _include_ops
    return _include_from_env()


def resolve_tuning_mode(op_name, *, supports_cost_model=False):
    """Resolve Default, Expanded, or Cost Model tuning for one operator.

    ``USE_FLAGTUNE=0`` selects the default config space. ``USE_FLAGTUNE=1``
    enables Expanded tuning for an unadapted operator and Cost Model tuning for
    an adapted operator. With neither switch set, unadapted operators default to
    Default and adapted operators default to Cost Model. ``FLAGTUNE_INCLUDE``
    applies the same capability-based selection to individual operators. An
    adapted operator uses Expanded only when ``USE_FLAGTUNE_COST_MODEL=0``. If
    the active platform has no model package, it follows the unadapted routes.
    """
    try:
        name = _normalize_op_name(op_name)
    except (TypeError, ValueError):
        return TuningMode.DEFAULT

    use_flagtune_setting = _use_flagtune_setting_from_env()
    if use_flagtune_setting is False:
        return TuningMode.DEFAULT

    # An explicit Expanded request does not need a model package. Short-circuit
    # before probing so this path remains usable in offline environments.
    if supports_cost_model:
        cost_model_value = os.environ.get(USE_FLAGTUNE_COST_MODEL_ENV)
        explicitly_expanded = use_flagtune_setting is True or (
            name in get_flagtune_include()
        )
        if (
            explicitly_expanded
            and cost_model_value is not None
            and cost_model_value.strip() == "0"
        ):
            return TuningMode.EXPANDED

    if supports_cost_model and not _platform_cost_model_available():
        supports_cost_model = False

    cost_model_setting = None
    if supports_cost_model:
        cost_model_setting = _optional_binary_environment(USE_FLAGTUNE_COST_MODEL_ENV)
        if cost_model_setting is False:
            return TuningMode.EXPANDED
        if cost_model_setting is True or use_flagtune_setting is True:
            return TuningMode.COST_MODEL
        return TuningMode.COST_MODEL
    if use_flagtune_setting is True or name in get_flagtune_include():
        return TuningMode.EXPANDED
    return TuningMode.DEFAULT


def flagtune_enabled(op_name):
    if op_name not in get_supported_flagtune_ops():
        return False
    return (
        resolve_tuning_mode(op_name, supports_cost_model=False) is TuningMode.EXPANDED
    )


def __getattr__(name):
    if name == "SUPPORTED_FLAGTUNE_OPS":
        return get_supported_flagtune_ops()
    if name == "DEFAULT_FLAGTUNE_INCLUDE":
        return get_default_flagtune_include()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


register_flagtune_op("mm", default=False, description="matrix multiplication")
register_flagtune_op("bmm", default=False, description="batched matrix multiplication")
register_flagtune_op(
    "addmm",
    default=False,
    description="matrix multiplication with bias",
)
register_flagtune_op(
    "baddbmm",
    default=False,
    description="batched matrix multiplication with bias",
)
register_flagtune_op(
    "mv",
    default=False,
    description="matrix-vector multiplication",
)
register_flagtune_op(
    "fused_marlin_moe_w4a16_int4",
    default=False,
    description="W4A16 INT4 fused Marlin MoE GEMM",
)
register_flagtune_op(
    "fused_marlin_moe_w4a16_int4_gemm_silu",
    default=False,
    description="W4A16 INT4 fused Marlin MoE GEMM with SiLU",
)
register_flagtune_op(
    "fused_marlin_moe_w4a16_mxfp4",
    default=False,
    description="MXFP4 fused Marlin MoE GEMM",
)
register_flagtune_op(
    "fused_marlin_moe_w4a16_mxfp4_gemm_silu",
    default=False,
    description="MXFP4 fused Marlin MoE GEMM with SiLU",
)
register_flagtune_op(
    "mul",
    default=False,
    description="elementwise multiplication",
)
register_flagtune_op(
    "compute_global_topk_indices_and_lens",
    default=False,
    description="DeepSeekV4 global top-k index conversion and length computation",
)
register_flagtune_op(
    "w8a8_block_fp8_matmul",
    default=False,
    description="W8A8 block FP8 matrix multiplication",
)
register_flagtune_op(
    "w8a8_block_fp8_bmm",
    default=False,
    description="W8A8 block FP8 batched matrix multiplication",
)

# DEFAULT_FLAGTUNE_INCLUDE and SUPPORTED_FLAGTUNE_OPS are provided by __getattr__.
__all__ = [  # noqa: F822
    "DEFAULT_FLAGTUNE_INCLUDE",
    "FLAGTUNE_INCLUDE_ENV",
    "FlagTuneOpSpec",
    "SUPPORTED_FLAGTUNE_OPS",
    "TuningMode",
    "USE_FLAGTUNE_COST_MODEL_ENV",
    "USE_FLAGTUNE_ENV",
    "flagtune",
    "flagtune_expanded_enabled",
    "flagtune_enabled",
    "get_default_flagtune_include",
    "get_flagtune_include",
    "get_flagtune_registry",
    "get_supported_flagtune_ops",
    "register_flagtune_op",
    "resolve_tuning_mode",
]
