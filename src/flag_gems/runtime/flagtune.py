import os
import warnings
from dataclasses import dataclass
from types import MappingProxyType

USE_FLAGTUNE_ENV = "USE_FLAGTUNE"
FLAGTUNE_INCLUDE_ENV = "FLAGTUNE_INCLUDE"

_flagtune_op_registry = {}
_include_ops = None


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


def flagtune(include=None):
    """Enable runtime FlagTune for selected operators.

    Passing include=None enables the registry's default operators. Passing a
    string or iterable selects the registered operators that should use
    expanded tuning spaces when their LibTuner runs. This API only updates the
    explicit include list; setting USE_FLAGTUNE=1 remains the legacy opt-in for
    enabling every registered FlagTune operator.
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
        warnings.warn(f"Invalid {FLAGTUNE_INCLUDE_ENV}: {err}")
        return frozenset()


def get_flagtune_include():
    if _include_ops is not None:
        return _include_ops
    return _include_from_env()


def flagtune_enabled(op_name):
    try:
        op_name = _normalize_op_name(op_name)
    except (TypeError, ValueError):
        return False
    if op_name not in get_supported_flagtune_ops():
        return False
    return os.environ.get(USE_FLAGTUNE_ENV) == "1" or op_name in get_flagtune_include()


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
    "fused_marlin_moe_mxfp4",
    default=False,
    description="MXFP4 fused Marlin MoE GEMM",
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
    "USE_FLAGTUNE_ENV",
    "flagtune",
    "flagtune_enabled",
    "get_default_flagtune_include",
    "get_flagtune_include",
    "get_flagtune_registry",
    "get_supported_flagtune_ops",
    "register_flagtune_op",
]
