"""MM-specific route resolution and runtime metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..common import (
    backend_module,
    platform,
)
from .hopper import select_mm_route as _select_hopper_mm_route
from .metax import select_mm_route as _select_metax_mm_route


ADAPTED_VARIANTS = {
    "nvidia": frozenset(
        {"gemv", "splitk_two_step", "splitk", "general_tma", "tma_transposed_direct"}
    ),
    "metax": frozenset(
        {
            "metax_gemv_k_parallel",
            "metax_gemv",
            "metax_splitk_two_step",
            "metax_splitk",
            "metax_nn",
            "metax_nt",
            "metax_general",
        }
    ),
}

ROUTE_TO_STAGE = {
    "metax_gemv_k_parallel": ("metax_gemv_k_parallel_partial", "partial"),
    "metax_splitk_two_step": ("metax_splitk_two_step_partial", "partial"),
    "splitk_two_step": ("splitk_two_step_partial", "partial"),
}


def _select_mm_route(a: Any, b: Any, module: Any | None = None) -> str:
    """Dispatch tensor route selection to the platform-specific route module."""
    if module is None:
        platform_name = platform({}, a)
        module = backend_module("mm", platform_name)
    else:
        module_name = str(getattr(module, "__name__", ""))
        platform_name = "metax" if "_metax" in module_name else "nvidia"
    if module is None:
        raise RuntimeError(
            f"MM route selector is unavailable for platform {platform_name!r}; "
            "route prediction requires real tensors and the backend selector"
        )
    if platform_name == "metax":
        return _select_metax_mm_route(a, b, module)
    return _select_hopper_mm_route(a, b, module)


def _recipe_values(a: Any, b: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(a, Mapping):
        values = dict(a)
    else:
        values = dict(context.get("recipe", {}))
        if hasattr(a, "shape") and hasattr(b, "shape"):
            values.setdefault("M", int(a.shape[0]))
            values.setdefault("K", int(a.shape[1]))
            values.setdefault("N", int(b.shape[1]))
            values.setdefault("A_layout", "transposed_2d" if a.stride(1) != 1 else "contiguous")
            values.setdefault("B_layout", "transposed_2d" if b.stride(0) == 1 else "contiguous")
            values.setdefault("stride_am", int(a.stride(0)))
            values.setdefault("stride_ak", int(a.stride(1)))
            values.setdefault("stride_bk", int(b.stride(0)))
            values.setdefault("stride_bn", int(b.stride(1)))
    return values


def route_metadata_for_variant(
    variant: str,
    platform_name: str,
    dynamic_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build route metadata from an already-selected planner variant.

    This helper deliberately does not inspect shape values.  It is used when
    Expanded/planner output already carries the route decision and therefore
    must not re-run a tensor-only backend selector on a recipe mapping.
    """
    physical = str(variant)
    for route, (stage_variant, _stage) in ROUTE_TO_STAGE.items():
        if physical == stage_variant:
            physical = route
            break
    adapted = physical in ADAPTED_VARIANTS.get(platform_name, frozenset())
    tuning_variant, stage = ROUTE_TO_STAGE.get(physical, (physical, "public"))
    return {
        "route_variant": physical,
        "tuning_variant": tuning_variant if adapted else None,
        "stage": stage if adapted else None,
        "latency_scope": "partial_kernel" if stage == "partial" and adapted else "public_kernel",
        "physical_route": physical,
        "cost_model_variant": tuning_variant if adapted else None,
        "dynamic_inputs": dict(dynamic_inputs or {}),
        "adapted": adapted,
        "reason": (
            "explicit planner route"
            if adapted
            else f"explicit planner route is not adapted: {physical!r}"
        ),
        "platform": platform_name,
    }


def _dynamic_inputs(module: Any, physical: str, a: Any, b: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not hasattr(a, "shape") or not hasattr(b, "shape"):
        return result
    m, k = (int(a.shape[0]), int(a.shape[1]))
    n = int(b.shape[1])
    try:
        if physical == "metax_gemv_k_parallel":
            result["SPLIT_K"] = int(module._gemv_k_parallel_split_k(m, k))
        elif physical == "metax_splitk_two_step":
            result["SPLIT_K"] = int(module._select_two_step_split_k(m, n, k) or 1)
        elif physical == "splitk_two_step":
            result["SPLIT_K"] = int(module._splitk_two_step_split_k(a, m, n, k))
        elif physical == "general_tma":
            result["USE_TMA"] = True
        elif physical == "warp_specialized":
            result["WS_PLAN"] = int(module._select_warp_specialized_dispatch_plan(a, b, module.torch.empty((m, n), device=a.device, dtype=module.get_higher_dtype(a.dtype, b.dtype)), m, n, k) or 0)
            result["B_ROW_MAJOR"] = bool(b.stride(1) == 1)
        elif physical in {"tma_transposed_direct", "tma_transposed_splitk"}:
            result["B_ROW_MAJOR"] = bool(b.stride(1) == 1)
    except Exception:
        # Dynamic values are audit metadata. Route resolution itself remains
        # authoritative even when a device-specific helper is unavailable.
        pass
    return result


def resolve_mm_route(a: Any, b: Any, runtime_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve one MM recipe to a physical route and model variant.

    ``a``/``b`` must be real tensors for prediction.  Non-tensor callers may
    only provide an already-computed route in ``runtime_context`` (for example
    ``route``/``physical_route`` from planner output); shape/layout heuristics
    are intentionally not supported.
    """
    context = dict(runtime_context or {})
    platform_name = platform(context, a)
    tensor_pair = hasattr(a, "shape") and hasattr(b, "shape")
    module = backend_module("mm", platform_name) if tensor_pair else None
    values = _recipe_values(a, b, context)
    if tensor_pair:
        physical = _select_mm_route(a, b, module)
        source = "flagtune_selector"
        dynamic = _dynamic_inputs(module, physical, a, b)
    else:
        explicit = context.get("route")
        if isinstance(explicit, Mapping):
            physical = explicit.get("physical_route") or explicit.get("route_variant")
            dynamic = explicit.get("dynamic_inputs", context.get("dynamic_inputs", {}))
            if explicit.get("platform"):
                platform_name = platform({"platform_key": explicit["platform"]})
        else:
            physical = context.get("physical_route") or context.get("route_variant")
            dynamic = context.get("dynamic_inputs", {})
        if not physical:
            raise ValueError(
                "MM route prediction requires real tensors or an explicit "
                "route decision; recipe mappings cannot be classified heuristically"
            )
        result = route_metadata_for_variant(str(physical), platform_name, dynamic)
        result["values"] = values
        return result
    result = route_metadata_for_variant(physical, platform_name, dynamic)
    result["values"] = values
    result["reason"] = f"{source}: adapted variant" if result["adapted"] else (
        f"{source}: planned_skip for physical route {physical!r}"
    )
    return result
