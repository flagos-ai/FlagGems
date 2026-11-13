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

"""Operator-agnostic route resolver dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .mm import resolve_mm_route

RouteResolver = Callable[[Any, Any, Mapping[str, Any] | None], dict[str, Any]]
ROUTE_RESOLVERS: dict[str, RouteResolver] = {
    "mm": resolve_mm_route,
    "flaggems/mm": resolve_mm_route,
}


def has_route_resolver(op_id: str) -> bool:
    """Whether real tensors, rather than contract predicates, select a route."""
    key = str(op_id)
    return key in ROUTE_RESOLVERS or key.rsplit("/", 1)[-1] in ROUTE_RESOLVERS


def tuning_variant(spec: Any, variant: str) -> str:
    """Use the contract's stage binding without interpreting kernel names."""
    info = spec.operator_info.variants.get(variant)
    return str(getattr(info, "route_binding", None) or variant)


def register_route_resolver(op_id: str, resolver: RouteResolver) -> None:
    """Register an operator resolver without changing shared dispatch code."""
    ROUTE_RESOLVERS[str(op_id)] = resolver


def resolve_route(
    op_id: str,
    a: Any,
    b: Any = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a route through the resolver registered for ``op_id``.

    The compatibility ``a``/``b`` form remains supported for existing MM
    callers.  New executors may pass a tensor mapping; conventional ``a``/``b``
    and ``lhs``/``rhs`` names are accepted without making operator resolvers
    depend on a particular benchmark YAML.
    """
    key = str(op_id)
    try:
        resolver = ROUTE_RESOLVERS.get(key) or ROUTE_RESOLVERS[key.rsplit("/", 1)[-1]]
    except KeyError as exc:
        raise ValueError(
            f"no route resolver registered for operator {op_id!r}"
        ) from exc
    if isinstance(a, Mapping) and b is None:
        tensors = a
        a = tensors.get("a", tensors.get("lhs"))
        b = tensors.get("b", tensors.get("rhs"))
    return resolver(a, b, runtime_context)


__all__ = [
    "ROUTE_RESOLVERS",
    "has_route_resolver",
    "register_route_resolver",
    "resolve_route",
    "tuning_variant",
]
