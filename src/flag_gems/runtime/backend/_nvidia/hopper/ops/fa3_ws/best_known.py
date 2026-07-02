"""Best-known FA3 routing policy.

The policy is intentionally conservative: keep FA3 on workloads where the
current implementation is consistently ahead, and transparently route weak
decode/paged cases to the existing FA2 launcher unless the user explicitly
forces a FA3 family for experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


_BEST_ROUTE_MODES = ("auto", "fa3_only", "fa2_only")
ROUTE_CURRENT_FA3 = "current_fa3"
ROUTE_RESTORED_FA3 = "restored_fa3"
ROUTE_FA2_FALLBACK = "fa2_fallback"


@dataclass(frozen=True)
class FA3BestRoute:
    route: str
    workload: str
    reason: str


def fa3_tle_best_route_mode() -> str:
    value = os.getenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "fa3_only").strip().lower()
    if value not in _BEST_ROUTE_MODES:
        raise RuntimeError(
            "invalid FLAG_GEMS_FA3_TLE_BEST_ROUTE="
            f"{value!r}; expected one of {', '.join(_BEST_ROUTE_MODES)}"
        )
    return value


def classify_fa3_workload(
    *,
    total_q: int,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_paged: bool,
) -> str:
    prefix = "paged" if is_paged else "dense"
    avg_q = total_q / max(batch_size, 1)

    if max_seqlen_q <= 8:
        return f"{prefix}_decode"
    if max_seqlen_q <= 32 and max_seqlen_k >= 1024:
        return f"{prefix}_decodeish_long_k"
    if max_seqlen_q <= 128 and max_seqlen_k <= 128:
        return f"{prefix}_short"
    if max_seqlen_q <= 128:
        return f"{prefix}_mixed_short"
    if is_paged and max_seqlen_q > 512 and avg_q <= 128:
        return "paged_serve_mixed"
    if max_seqlen_q <= 1024 and max_seqlen_k <= 1024:
        return f"{prefix}_medium_or_prefill"
    if is_paged:
        return "paged_uniform_prefill"
    return "dense_prefill_or_long"


def select_fa3_best_route(
    *,
    total_q: int,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_paged: bool,
    force_family_id: int,
) -> FA3BestRoute:
    mode = fa3_tle_best_route_mode()
    workload = classify_fa3_workload(
        total_q=total_q,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_paged=is_paged,
    )

    if mode == "fa2_only":
        return FA3BestRoute(ROUTE_FA2_FALLBACK, workload, "env fa2_only")
    if mode == "fa3_only":
        return FA3BestRoute(ROUTE_CURRENT_FA3, workload, "env fa3_only")
    if force_family_id != -1:
        return FA3BestRoute(ROUTE_CURRENT_FA3, workload, "forced FA3 family")

    if workload in ("dense_prefill_or_long", "paged_serve_mixed"):
        return FA3BestRoute(ROUTE_CURRENT_FA3, workload, "best-known FA3 win")

    return FA3BestRoute(
        ROUTE_FA2_FALLBACK,
        workload,
        "best-known FA2 fallback until restored FA3 kernel is verified",
    )


__all__ = [
    "FA3BestRoute",
    "ROUTE_CURRENT_FA3",
    "ROUTE_RESTORED_FA3",
    "ROUTE_FA2_FALLBACK",
    "classify_fa3_workload",
    "fa3_tle_best_route_mode",
    "select_fa3_best_route",
]
