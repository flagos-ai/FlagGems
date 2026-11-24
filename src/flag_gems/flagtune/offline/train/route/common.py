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

"""Shared route utilities independent of a specific operator."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from typing import Any

BACKEND_MODULES = {
    "mm": {
        "metax": "flag_gems.runtime.backend._metax.ops.mm",
        "nvidia": "flag_gems.runtime.backend._nvidia.hopper.ops.mm",
    },
}


def _normalized_platform(text: str) -> str | None:
    """Map a vendor/device label to the catalog's normalized platform name."""
    text = text.lower()
    if "metax" in text or "maca" in text:
        return "metax"
    if "nvidia" in text or "hopper" in text or "cuda" in text:
        return "nvidia"
    if "hygon" in text:
        return "hygon"
    if "mthreads" in text:
        return "mthreads"
    if "thead" in text:
        return "thead"
    return None


def platform(runtime_context: Mapping[str, Any] | None, value: Any = None) -> str:
    """Resolve a normalized backend platform from context, device, or FlagGems."""
    context = runtime_context or {}
    for key in ("platform", "platform_key", "vendor", "vendor_name"):
        candidate = context.get(key)
        if candidate:
            normalized = _normalized_platform(str(candidate))
            if normalized is not None:
                return normalized
    device = getattr(value, "device", None)
    device_type = str(getattr(device, "type", device or "")).lower()
    normalized = _normalized_platform(device_type)
    if normalized is not None:
        return normalized
    try:
        import flag_gems

        normalized = _normalized_platform(str(getattr(flag_gems, "vendor_name", "")))
        if normalized is not None:
            return normalized
    except Exception:
        pass
    return "unknown"


def backend_module(op_id: str, platform_name: str):
    """Load the implementation module registered for an operator/platform."""
    module_name = BACKEND_MODULES.get(op_id, {}).get(platform_name)
    if not module_name:
        return None
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


def recipe_layout_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return layout and stride metadata for a matrix recipe."""
    m = int(values.get("M", 0))
    n = int(values.get("N", 0))
    k = int(values.get("K", 0))
    a_layout = str(values.get("A_layout", "contiguous"))
    b_layout = str(values.get("B_layout", "contiguous"))
    if a_layout not in {"contiguous", "transposed_2d"}:
        a_layout = "contiguous"
    if b_layout not in {"contiguous", "transposed_2d"}:
        b_layout = "contiguous"
    return {
        "layouts": {"A": a_layout, "B": b_layout, "C": "contiguous"},
        "stride_rules": {
            "A": [k, 1] if a_layout == "contiguous" else [1, m],
            "B": [n, 1] if b_layout == "contiguous" else [1, k],
            "C": [n, 1],
        },
    }


def make_recipe_id(
    op_id: str,
    platform_name: str,
    values: Mapping[str, Any],
    input_dtypes: Any,
    output_dtypes: Any,
    variant: str | None,
    dynamic_inputs: Mapping[str, Any] | None = None,
    route_variant: str | None = None,
) -> str:
    """Build a stable globally unique identity for one complete recipe."""
    payload = {
        "op_id": op_id,
        "platform": platform_name,
        "values": dict(values),
        "input_dtypes": input_dtypes,
        "output_dtypes": output_dtypes,
        "variant": variant,
        "tuning_variant": variant,
        "route_variant": route_variant if route_variant is not None else variant,
        "dynamic_inputs": dict(dynamic_inputs or {}),
        "layout_metadata": recipe_layout_metadata(values),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "BACKEND_MODULES",
    "backend_module",
    "make_recipe_id",
    "platform",
    "recipe_layout_metadata",
]
