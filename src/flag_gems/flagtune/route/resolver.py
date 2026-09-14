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


def register_route_resolver(op_id: str, resolver: RouteResolver) -> None:
    """Register an operator resolver without changing shared dispatch code."""
    ROUTE_RESOLVERS[str(op_id)] = resolver


def resolve_route(
    op_id: str,
    a: Any,
    b: Any = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a route through the resolver registered for ``op_id``."""
    key = str(op_id)
    try:
        resolver = ROUTE_RESOLVERS.get(key) or ROUTE_RESOLVERS[key.rsplit("/", 1)[-1]]
    except KeyError as exc:
        raise ValueError(f"no route resolver registered for operator {op_id!r}") from exc
    return resolver(a, b, runtime_context)


__all__ = ["ROUTE_RESOLVERS", "register_route_resolver", "resolve_route"]
