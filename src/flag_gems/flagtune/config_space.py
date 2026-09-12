"""Shared resolution of runtime-owned FlagTune candidate spaces."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


_MM_RUNTIME_OPS = {
    "nvidia": {
        "general_tma": "mm_general_tma",
        "gemv": "gemv",
        "splitk": "mm_splitk",
        "splitk_two_step": "mm_splitk_two_step",
        "splitk_two_step_partial": "mm_splitk_two_step",
        "tma_transposed_direct": "mm_tma_transposed_direct",
    },
    "metax": {
        "metax_general": "mm",
        "metax_nn": "mm_nn",
        "metax_nt": "mm_nt",
        "metax_gemv": "gemv",
        "metax_gemv_k_parallel": "gemv_k_parallel",
        "metax_gemv_k_parallel_partial": "gemv_k_parallel",
        "metax_splitk": "mm_splitk",
        "metax_splitk_two_step": "mm_splitk_two_step",
        "metax_splitk_two_step_partial": "mm_splitk_two_step",
    },
}

_RUNTIME_OPS = {
    "flaggems/mm": _MM_RUNTIME_OPS,
    "flaggems/mul": {
        "scalar": "mul",
        "broadcast_2d": "mul_broadcast_2d",
    },
}


def runtime_op_name_for_variant(
    op_id: str, variant: str, platform: Optional[str] = None
) -> Optional[str]:
    """Return the runtime expand-op name for a known operator variant."""
    if platform is None:
        from flag_gems import runtime

        platform = str(runtime.device.vendor_name)
    platform_ops = _RUNTIME_OPS.get(op_id, {})
    if op_id == "flaggems/mul":
        return platform_ops.get(variant)
    return platform_ops.get(str(platform), {}).get(variant)


def runtime_configs_for_variant(
    op_id: str,
    variant: str,
    *,
    platform: Optional[str] = None,
    yaml_path: Optional[str] = None,
    pre_hook: Any = None,
) -> Optional[list[Any]]:
    """Resolve a runtime Expanded + Default config space when one is known.

    ``None`` means the operator has no runtime mapping and must be rejected by
    runtime-authoritative callers. An empty list means a mapped runtime space is
    unavailable and should also be treated as an error.
    """
    from flag_gems import runtime

    # The runtime loader's generic registry does not always discover the
    # backend-specific expand YAML (notably Hopper's nested directory). Use
    # the same module-owned path that libentry decorators pass during Expanded
    # collection, while keeping the runtime YAML as the sole source of truth.
    platform_text = str(platform or runtime.device.vendor_name).lower()
    if "metax" in platform_text or "maca" in platform_text:
        platform_name = "metax"
    elif "nvidia" in platform_text or "hopper" in platform_text or "cuda" in platform_text:
        platform_name = "nvidia"
    else:
        platform_name = platform_text
    expand_yaml_path = yaml_path
    if expand_yaml_path is None and op_id == "flaggems/mm":
        from flag_gems.flagtune.route.common import backend_module

        backend = backend_module("mm", platform_name)
        expand_yaml_path = getattr(backend, "EXPAND_CONFIG_FILENAME", None)

    op_name = runtime_op_name_for_variant(op_id, variant, platform_name)
    if op_name is None:
        return None
    configs = runtime.ops_get_configs(
        op_name,
        yaml_path=expand_yaml_path,
        pre_hook=pre_hook,
    )
    return list(configs)


def runtime_configs_hash(configs: list[Any]) -> str:
    """Return a stable digest for a materialized runtime candidate domain."""
    normalized = []
    for config in configs:
        normalized.append(
            {
                "kwargs": dict(sorted(getattr(config, "kwargs", {}).items())),
                "launch": {
                    name: getattr(config, name, None)
                    for name in (
                        "num_warps",
                        "num_stages",
                        "num_ctas",
                        "num_buffers_warp_spec",
                        "num_consumer_groups",
                        "reg_dec_producer",
                        "reg_inc_consumer",
                        "maxnreg",
                    )
                },
                # The hook itself is executable and not serializable. Its
                # presence is nevertheless part of the candidate semantics;
                # the source YAML/config hash binds the concrete hook code.
                "has_pre_hook": getattr(config, "pre_hook", None) is not None,
            }
        )
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), default=repr
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
