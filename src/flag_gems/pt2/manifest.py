# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Declarative compiler contracts for existing FlagGems implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class CompileKind(str, Enum):
    ATEN_NATIVE = "aten_native"
    TRITON_TRACEABLE = "triton_traceable"
    OPAQUE_CUSTOM = "opaque_custom"
    RESOLVE_ONLY = "resolve_only"
    EAGER_ONLY = "eager_only"


@dataclass(frozen=True)
class CompileOpSpec:
    """Compiler-facing facts; never an alternative kernel implementation."""

    op_name: str
    kind: CompileKind
    source_kernel: str
    mutates_args: tuple[str, ...] = ()
    dynamic_dims: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["kind"] = self.kind.value
        return result


_MANIFEST: dict[str, CompileOpSpec] = {}


def register_compile_spec(spec: CompileOpSpec) -> CompileOpSpec:
    previous = _MANIFEST.get(spec.op_name)
    if previous is not None and previous != spec:
        raise RuntimeError(f"Conflicting compiler spec for {spec.op_name!r}")
    _MANIFEST[spec.op_name] = spec
    return spec


def get_compile_spec(op_name: str) -> CompileOpSpec:
    return _MANIFEST[op_name]


def get_compile_manifest() -> tuple[CompileOpSpec, ...]:
    return tuple(_MANIFEST[name] for name in sorted(_MANIFEST))


def iter_compile_manifest() -> Iterable[CompileOpSpec]:
    return iter(get_compile_manifest())


__all__ = [
    "CompileKind",
    "CompileOpSpec",
    "get_compile_manifest",
    "get_compile_spec",
    "iter_compile_manifest",
    "register_compile_spec",
]
