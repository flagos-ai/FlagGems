"""MM route implementations grouped by operator and backend."""

from collections.abc import Mapping
from typing import Any

from ..common import make_recipe_id as _make_recipe_id
from ..common import recipe_layout_metadata
from .resolver import (
    ADAPTED_VARIANTS,
    ROUTE_TO_STAGE,
    _select_mm_route,
    route_metadata_for_variant,
    resolve_mm_route,
)


def make_recipe_id(
    op_id: str,
    platform: str,
    values: Mapping[str, Any],
    input_dtypes: Any,
    output_dtypes: Any,
    variant: str | None,
    dynamic_inputs: Mapping[str, Any] | None = None,
) -> str:
    """Build the MM identity while retaining its public/stage route mapping."""
    route_variant = next(
        (
            route
            for route, (stage_variant, _stage) in ROUTE_TO_STAGE.items()
            if stage_variant == variant
        ),
        variant,
    )
    return _make_recipe_id(
        op_id,
        platform,
        values,
        input_dtypes,
        output_dtypes,
        variant,
        dynamic_inputs,
        route_variant=route_variant,
    )

__all__ = [
    "ADAPTED_VARIANTS",
    "ROUTE_TO_STAGE",
    "_select_mm_route",
    "route_metadata_for_variant",
    "make_recipe_id",
    "recipe_layout_metadata",
    "resolve_mm_route",
]
