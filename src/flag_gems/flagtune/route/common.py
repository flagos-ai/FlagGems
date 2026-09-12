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


def platform(runtime_context: Mapping[str, Any] | None, value: Any = None) -> str:
    """Resolve a normalized backend platform from context, device, or FlagGems."""
    context = runtime_context or {}
    for key in ("platform", "platform_key", "vendor", "vendor_name"):
        candidate = context.get(key)
        if candidate:
            text = str(candidate).lower()
            if "metax" in text or "maca" in text:
                return "metax"
            if "nvidia" in text or "hopper" in text or "cuda" in text:
                return "nvidia"
    device = getattr(value, "device", None)
    device_type = str(getattr(device, "type", device or "")).lower()
    if "maca" in device_type or "metax" in device_type:
        return "metax"
    if "cuda" in device_type:
        return "nvidia"
    try:
        import flag_gems

        vendor = str(getattr(flag_gems, "vendor_name", "")).lower()
        if "metax" in vendor or "maca" in vendor:
            return "metax"
        if "nvidia" in vendor or "cuda" in vendor:
            return "nvidia"
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
