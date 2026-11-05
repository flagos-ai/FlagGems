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

"""MM route implementations grouped by operator and backend."""

from collections.abc import Mapping
from typing import Any

from ..common import make_recipe_id as _make_recipe_id
from ..common import recipe_layout_metadata
from .resolver import (
    ADAPTED_VARIANTS,
    ROUTE_TO_STAGE,
    _select_mm_route,
    resolve_mm_route,
    route_metadata_for_variant,
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
