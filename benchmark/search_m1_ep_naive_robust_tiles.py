#!/usr/bin/env python3
"""Tail-first raw-direct tile search for fused-MoE M=1/I=2048.

This benchmark deliberately differs from the older endpoint-oriented search:

* screening covers no-local, singleton p0..p7, and representative masks for
  every local-route count before a candidate can be promoted;
* candidates are ranked by their worst measured mask first, never by an
  average that can hide a route-position regression;
* the best coordinate candidates are crossed into joint G1/G2 plans and the
  promoted plans can be audited over all 256 top-8 local/remote masks;
* the formal aggregate is a ratio of absolute mean latencies under the exact
  EP16 hypergeometric prior, not an average of percentages;
* every comparison is an in-process alternating ABBA/BAAB CUDA Graph A/B and
  checks eager/Graph/cross-plan bitwise equality with output aliasing cache2 by
  default.

Only benchmark-time selectors and routing gates are monkeypatched.  This file
does not modify the production fused-MoE implementation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

TOPK = 8
GLOBAL_EXPERTS = 288
LOCAL_EXPERTS = 18
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
GOAL_REDUCTION_PCT = 10.0
FLOAT_COMPARISON_EPS = 1e-9


@dataclass(frozen=True)
class Tile:
    """Actual output tile dimensions seen by one fused-MoE GEMM stage."""

    bn: int
    bk: int
    warps: int
    stages: int
    maxnreg: int | None = None

    def selector_config(self, *, paired_gemm1: bool) -> dict[str, int]:
        config = {
            "BLOCK_SIZE_M": 16,
            # fused_experts_impl halves selector BN when it enables the paired
            # gate/up dot, so expose twice the actual G1 tile here.
            "BLOCK_SIZE_N": 2 * self.bn if paired_gemm1 else self.bn,
            "BLOCK_SIZE_K": self.bk,
            "GROUP_SIZE_M": 1,
            "num_warps": self.warps,
            "num_stages": self.stages,
        }
        if self.maxnreg is not None:
            config["maxnreg"] = self.maxnreg
        return config


@dataclass(frozen=True)
class Plan:
    name: str
    stage: str
    g1: Tile
    g2: Tile

    def selector_configs(self) -> tuple[dict[str, int], dict[str, int]]:
        return (
            self.g1.selector_config(paired_gemm1=True),
            self.g2.selector_config(paired_gemm1=False),
        )


BASE_G1 = Tile(bn=32, bk=128, warps=4, stages=3)
BASE_G2 = Tile(bn=128, bk=64, warps=4, stages=4)
BASELINE = Plan("baseline", "baseline", BASE_G1, BASE_G2)

# These were already measured and rejected.  The robust suites must not spend
# another GPU sweep on them.  GROUP_SIZE_M is fixed at one throughout this
# script; prior static group 2/4/8, cluster, and warp-specialization failures
# are intentionally outside the candidate space.
KNOWN_REJECTED_STAGE_TILES = {
    ("g1", Tile(bn=16, bk=128, warps=2, stages=3)),
    ("g1", Tile(bn=16, bk=128, warps=2, stages=5)),
    ("g2", Tile(bn=64, bk=64, warps=4, stages=4)),
}


def tile_label(tile: Tile) -> str:
    label = f"bn{tile.bn}_bk{tile.bk}_w{tile.warps}_s{tile.stages}"
    if tile.maxnreg is not None:
        label += f"_r{tile.maxnreg}"
    return label


def add_coordinate_plan(plans: dict[str, Plan], *, stage: str, tile: Tile) -> None:
    if (stage, tile) in KNOWN_REJECTED_STAGE_TILES:
        return
    if stage == "g1":
        plan = Plan(f"g1_{tile_label(tile)}", stage, tile, BASE_G2)
    elif stage == "g2":
        plan = Plan(f"g2_{tile_label(tile)}", stage, BASE_G1, tile)
    else:
        raise ValueError(f"coordinate stage must be g1 or g2, got {stage}")
    if plan.g1 == BASE_G1 and plan.g2 == BASE_G2:
        return
    plans.setdefault(plan.name, plan)


def make_coordinate_plans(suite: str) -> list[Plan]:
    """Build new tail-oriented candidates without known failed policies."""
    if suite == "boundary":
        core_names = {plan.name for plan in make_coordinate_plans("core")}
        return [
            BASELINE,
            *[
                plan
                for plan in make_coordinate_plans("extended")
                if plan.name not in core_names
            ],
        ]
    plans: dict[str, Plan] = {BASELINE.name: BASELINE}

    # Keep the baseline N tile geometry, which keeps the number and ordering of
    # route CTAs unchanged.  Vary K depth and launch occupancy controls first;
    # this is the lowest-risk way to improve p6/p7 without moving useful CTAs.
    g1_core: dict[int, tuple[tuple[int, int], ...]] = {
        32: ((2, 2), (2, 3), (4, 2), (4, 3), (4, 4)),
        64: ((2, 2), (2, 3), (4, 2), (4, 3), (4, 4)),
        128: ((2, 2), (2, 4), (4, 2), (4, 4), (8, 2), (8, 3)),
        256: ((4, 2), (4, 3), (8, 2), (8, 3)),
        512: ((8, 1), (8, 2)),
    }
    for bk, launch_pairs in g1_core.items():
        for warps, stages in launch_pairs:
            add_coordinate_plan(
                plans,
                stage="g1",
                tile=Tile(32, bk, warps, stages),
            )

    g2_core: dict[int, tuple[tuple[int, int], ...]] = {
        32: ((2, 2), (2, 3), (4, 2), (4, 3), (4, 4)),
        64: ((2, 2), (2, 3), (4, 2), (4, 3), (4, 4)),
        128: ((2, 2), (2, 3), (4, 2), (4, 3), (4, 4)),
        256: ((4, 2), (4, 3), (8, 2), (8, 3)),
        512: ((8, 1), (8, 2)),
    }
    for bk, launch_pairs in g2_core.items():
        for warps, stages in launch_pairs:
            add_coordinate_plan(
                plans,
                stage="g2",
                tile=Tile(128, bk, warps, stages),
            )

    # NCU measured roughly 64/67 registers per thread for the two production
    # stages.  maxnreg is an unexplored launch control that can trade spills for
    # occupancy without changing route CTA placement.
    for maxnreg in (40, 48, 56, 64, 72, 80, 96, 128):
        add_coordinate_plan(
            plans,
            stage="g1",
            tile=Tile(
                BASE_G1.bn,
                BASE_G1.bk,
                BASE_G1.warps,
                BASE_G1.stages,
                maxnreg,
            ),
        )
        add_coordinate_plan(
            plans,
            stage="g2",
            tile=Tile(
                BASE_G2.bn,
                BASE_G2.bk,
                BASE_G2.warps,
                BASE_G2.stages,
                maxnreg,
            ),
        )

    if suite == "extended":
        # Boundary N tiles change the CTA wave geometry.  They are kept in a
        # separate suite because they are higher-risk for route-position tails.
        for bn, bks, launch_pairs in (
            (16, (64, 128, 256), ((4, 2), (4, 3), (4, 4), (8, 2))),
            (64, (64, 128, 256), ((4, 2), (4, 3), (8, 2), (8, 3))),
            (128, (64, 128), ((8, 1), (8, 2))),
        ):
            for bk in bks:
                for warps, stages in launch_pairs:
                    add_coordinate_plan(
                        plans,
                        stage="g1",
                        tile=Tile(bn, bk, warps, stages),
                    )
        for bn, bks, launch_pairs in (
            (64, (32, 64, 128, 256), ((4, 2), (4, 3), (8, 2), (8, 3))),
            (256, (32, 64, 128, 256), ((4, 2), (4, 3), (4, 4), (8, 2))),
            (512, (64, 128), ((8, 1), (8, 2))),
        ):
            for bk in bks:
                for warps, stages in launch_pairs:
                    add_coordinate_plan(
                        plans,
                        stage="g2",
                        tile=Tile(bn, bk, warps, stages),
                    )
    elif suite != "core":
        raise ValueError(f"unknown suite: {suite}")

    return list(plans.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("core", "boundary", "extended"), default="core"
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated coordinate plan names; baseline is implicit",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug prefix limit")
    parser.add_argument("--list-plans", action="store_true")
    parser.add_argument(
        "--screen-masks",
        choices=("endpoints", "stress", "all"),
        default="stress",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=300)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument(
        "--joint-top-per-stage",
        type=int,
        default=3,
        help="cross the N most tail-robust coordinate plans from each stage",
    )
    parser.add_argument(
        "--audit-all-masks-top",
        type=int,
        default=3,
        help="formal 256-mask audit for the N most tail-robust plans; zero disables",
    )
    parser.add_argument("--audit-rounds", type=int, default=5)
    parser.add_argument("--audit-replays", type=int, default=500)
    parser.add_argument(
        "--disjoint-output",
        action="store_true",
        help="disable the default modular output/cache2 alias contract",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--allow-any-visible-device",
        action="store_true",
        help="debug only; formal runs require CUDA_VISIBLE_DEVICES=7",
    )
    return parser.parse_args()


def select_plans(args: argparse.Namespace) -> list[Plan]:
    all_plans = make_coordinate_plans(args.suite)
    if args.only:
        by_name = {plan.name: plan for plan in all_plans}
        names = [name.strip() for name in args.only.split(",") if name.strip()]
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"unknown --only plans: {missing}")
        plans = [BASELINE, *(by_name[name] for name in names)]
    else:
        plans = all_plans
    if args.limit:
        if args.limit < 2:
            raise ValueError("--limit must leave baseline and at least one candidate")
        plans = plans[: args.limit]
    return list(dict.fromkeys(plans))


def route_mask_positions(route_mask: int) -> list[int]:
    if not 0 <= route_mask < (1 << TOPK):
        raise ValueError(f"route mask must be an 8-bit integer, got {route_mask}")
    return [position for position in range(TOPK) if route_mask & (1 << position)]


def make_route_for_mask(route_mask: int, *, device: torch.device) -> torch.Tensor:
    # Remote routes and local routes are both unique, matching top-k semantics.
    route = LOCAL_EXPERTS + torch.arange(TOPK, device=device, dtype=torch.int32)
    for local_expert, position in enumerate(route_mask_positions(route_mask)):
        route[position] = local_expert
    return route.view(1, TOPK)


def stress_route_masks() -> list[int]:
    """Tail probes spanning every route count and singleton position."""
    masks = [0, *(1 << position for position in range(TOPK))]
    representatives = {
        2: (0x03, 0xC0, 0x81, 0x24),
        3: (0x07, 0xE0, 0x49, 0x92),
        4: (0x0F, 0xF0, 0x55, 0xAA),
        5: (0x1F, 0xF8, 0xB6, 0x6D),
        6: (0x3F, 0xFC, 0x7E, 0xDB),
        7: (0x7F, 0xFE, 0xFD, 0xBF),
    }
    for local_count, count_masks in representatives.items():
        if any(mask.bit_count() != local_count for mask in count_masks):
            raise AssertionError(f"invalid stress masks for route count {local_count}")
        masks.extend(count_masks)
    masks.append((1 << TOPK) - 1)
    return list(dict.fromkeys(masks))


def masks_for_screen(kind: str) -> list[int]:
    if kind == "endpoints":
        return [0, *(1 << position for position in range(TOPK)), 0xFF]
    if kind == "stress":
        return stress_route_masks()
    if kind == "all":
        return list(range(1 << TOPK))
    raise ValueError(f"unknown screen mask set: {kind}")


def selector_for(plan: Plan) -> Callable[..., dict[str, int]]:
    g1, g2 = plan.selector_configs()

    def selector(*positional, gemm_stage=None, **_keyword):
        stage = gemm_stage if gemm_stage is not None else positional[-1]
        return (g1 if stage == "gemm1" else g2).copy()

    return selector


def capture(fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / replays)


def paired_graph_ab(
    baseline: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    *,
    rounds: int,
    replays: int,
) -> dict[str, object]:
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    paired_reductions: list[float] = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            bracket = (
                ("baseline", baseline),
                ("candidate", candidate),
                ("candidate", candidate),
                ("baseline", baseline),
            )
        else:
            bracket = (
                ("candidate", candidate),
                ("baseline", baseline),
                ("baseline", baseline),
                ("candidate", candidate),
            )
        values = {"baseline": [], "candidate": []}
        for name, graph in bracket:
            values[name].append(elapsed_ms(graph, replays))
        baseline_value = statistics.mean(values["baseline"])
        candidate_value = statistics.mean(values["candidate"])
        baseline_samples.append(baseline_value)
        candidate_samples.append(candidate_value)
        paired_reductions.append(100.0 * (1.0 - candidate_value / baseline_value))

    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    return {
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "reduction_of_medians_pct": 100.0 * (1.0 - candidate_median / baseline_median),
        "median_paired_reduction_pct": statistics.median(paired_reductions),
        "positive_rounds": sum(value > 0 for value in paired_reductions),
        "total_rounds": rounds,
        "baseline_samples_ms": baseline_samples,
        "candidate_samples_ms": candidate_samples,
        "paired_reductions_pct": paired_reductions,
    }


def expected_dispatch(plan: Plan, stage: str) -> dict[str, int]:
    tile = plan.g1 if stage == "g1" else plan.g2
    expected = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": tile.bn,
        "BLOCK_SIZE_K": tile.bk,
        "GROUP_SIZE_M": 1,
        "num_warps": tile.warps,
        "num_stages": tile.stages,
    }
    if tile.maxnreg is not None:
        expected["maxnreg"] = tile.maxnreg
    return expected


def validate_dispatch(plan: Plan, entries: list[dict[str, object]]) -> None:
    if not entries:
        raise AssertionError(f"no dispatch was observed for {plan.name}")
    observed_stages = {str(entry["stage"]) for entry in entries}
    if observed_stages != {"g1", "g2"}:
        raise AssertionError(
            f"{plan.name} observed dispatch stages {sorted(observed_stages)}"
        )
    for entry in entries:
        stage = str(entry["stage"])
        config = entry["config"]
        expected = expected_dispatch(plan, stage)
        actual = {key: config.get(key) for key in expected}
        if actual != expected:
            raise AssertionError(
                f"{plan.name}/{stage} dispatch mismatch: {actual} != {expected}"
            )
        if entry["sorted_token_ids_is_none"] is not True:
            raise AssertionError(f"{plan.name}/{stage} did not use raw direct routing")
        if entry["skip_invalid_experts"] is not True:
            raise AssertionError(f"{plan.name}/{stage} did not skip remote experts")


def benchmark_masks(
    *,
    fm,
    op: Callable[[], torch.Tensor],
    output: torch.Tensor,
    ids: torch.Tensor,
    plans: list[Plan],
    graphs: dict[str, torch.cuda.CUDAGraph],
    route_masks: list[int],
    rounds: int,
    replays: int,
    warmup_replays: int,
    summary_only: bool,
    phase: str,
) -> dict[str, dict[str, dict[str, object]]]:
    baseline_graph = graphs[BASELINE.name]
    candidates = [plan for plan in plans if plan.name != BASELINE.name]
    results: dict[str, dict[str, dict[str, object]]] = {
        plan.name: {} for plan in candidates
    }
    for mask_index, route_mask in enumerate(route_masks):
        ids.copy_(make_route_for_mask(route_mask, device=ids.device))
        fm._get_ep_decode_config = selector_for(BASELINE)
        baseline_eager = op().clone()
        torch.cuda.synchronize()
        baseline_graph.replay()
        torch.cuda.synchronize()
        # Avoid retaining the shared output tensor: both graphs intentionally
        # alias it, so correctness needs an immediate value copy after replay.
        baseline_graph_output = output.clone()
        torch.cuda.synchronize()

        for plan in candidates:
            candidate_graph = graphs[plan.name]
            fm._get_ep_decode_config = selector_for(plan)
            candidate_eager = op().clone()
            torch.cuda.synchronize()
            candidate_graph.replay()
            torch.cuda.synchronize()
            candidate_graph_output = output.clone()
            torch.cuda.synchronize()

            for _ in range(warmup_replays):
                baseline_graph.replay()
                candidate_graph.replay()
            torch.cuda.synchronize()
            timing = paired_graph_ab(
                baseline_graph,
                candidate_graph,
                rounds=rounds,
                replays=replays,
            )
            result = {
                "route_mask": route_mask,
                "local_positions": route_mask_positions(route_mask),
                "local_count": route_mask.bit_count(),
                **timing,
                "candidate_bitwise_vs_baseline": bool(
                    torch.equal(candidate_eager, baseline_eager)
                ),
                "baseline_graph_bitwise": bool(
                    torch.equal(baseline_graph_output, baseline_eager)
                ),
                "candidate_graph_bitwise": bool(
                    torch.equal(candidate_graph_output, candidate_eager)
                ),
            }
            if summary_only:
                result.pop("baseline_samples_ms")
                result.pop("candidate_samples_ms")
                result.pop("paired_reductions_pct")
            results[plan.name][str(route_mask)] = result

        print(
            json.dumps(
                {
                    "progress": phase,
                    "mask": route_mask,
                    "completed": mask_index + 1,
                    "total": len(route_masks),
                    "candidates": len(candidates),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    return results


def hypergeometric_count_probability(local_count: int) -> float:
    return (
        math.comb(LOCAL_EXPERTS, local_count)
        * math.comb(GLOBAL_EXPERTS - LOCAL_EXPERTS, TOPK - local_count)
        / math.comb(GLOBAL_EXPERTS, TOPK)
    )


def summarize_masks(
    plan: Plan,
    mask_results: dict[str, dict[str, object]],
    *,
    complete_mask_space: bool,
) -> dict[str, object]:
    values = list(mask_results.values())
    reductions = [float(value["reduction_of_medians_pct"]) for value in values]
    singleton = {
        str(position): mask_results[str(1 << position)]
        for position in range(TOPK)
        if str(1 << position) in mask_results
    }
    bitwise_all = all(
        value["candidate_bitwise_vs_baseline"]
        and value["baseline_graph_bitwise"]
        and value["candidate_graph_bitwise"]
        for value in values
    )

    by_count = {}
    count_ratio_reductions = []
    for local_count in range(TOPK + 1):
        count_values = [
            value for value in values if int(value["local_count"]) == local_count
        ]
        if not count_values:
            continue
        mean_baseline = statistics.mean(
            float(value["baseline_median_ms"]) for value in count_values
        )
        mean_candidate = statistics.mean(
            float(value["candidate_median_ms"]) for value in count_values
        )
        ratio_reduction = 100.0 * (1.0 - mean_candidate / mean_baseline)
        count_ratio_reductions.append(ratio_reduction)
        by_count[str(local_count)] = {
            "masks": len(count_values),
            "ratio_of_means_reduction_pct": ratio_reduction,
            "worst_mask_reduction_pct": min(
                float(value["reduction_of_medians_pct"]) for value in count_values
            ),
            "positive_masks": sum(
                float(value["reduction_of_medians_pct"]) > 0 for value in count_values
            ),
        }

    worst = min(values, key=lambda value: float(value["reduction_of_medians_pct"]))
    summary: dict[str, object] = {
        "name": plan.name,
        "stage": plan.stage,
        "g1": asdict(plan.g1),
        "g2": asdict(plan.g2),
        "masks": len(values),
        "bitwise_all": bitwise_all,
        "positive_masks": sum(value > 0 for value in reductions),
        "worst_mask": int(worst["route_mask"]),
        "worst_local_positions": worst["local_positions"],
        "worst_reduction_pct": float(worst["reduction_of_medians_pct"]),
        "route_count_min_ratio_of_means_reduction_pct": min(count_ratio_reductions),
        "by_local_count": by_count,
    }
    if singleton:
        singleton_reductions = [
            float(singleton[str(position)]["reduction_of_medians_pct"])
            for position in range(TOPK)
        ]
        singleton_baseline = statistics.mean(
            float(value["baseline_median_ms"]) for value in singleton.values()
        )
        singleton_candidate = statistics.mean(
            float(value["candidate_median_ms"]) for value in singleton.values()
        )
        summary["singleton_reductions_pct_p0_to_p7"] = singleton_reductions
        summary["singleton_min_reduction_pct"] = min(singleton_reductions)
        summary["singleton_ratio_of_means_reduction_pct"] = 100.0 * (
            1.0 - singleton_candidate / singleton_baseline
        )

    # This is formal only for the complete 256-mask space.  For screening,
    # equal weighting within a deliberately tail-heavy sample is not a router
    # prior and must not be presented as an expected production speedup.
    if complete_mask_space:
        expected_baseline = 0.0
        expected_candidate = 0.0
        for value in values:
            local_count = int(value["local_count"])
            probability = hypergeometric_count_probability(local_count) / math.comb(
                TOPK, local_count
            )
            expected_baseline += probability * float(value["baseline_median_ms"])
            expected_candidate += probability * float(value["candidate_median_ms"])
        expected_reduction = 100.0 * (1.0 - expected_candidate / expected_baseline)
        summary.update(
            {
                "uniform_hypergeometric_expected_baseline_ms": expected_baseline,
                "uniform_hypergeometric_expected_candidate_ms": expected_candidate,
                "uniform_hypergeometric_ratio_of_means_reduction_pct": (
                    expected_reduction
                ),
                "all_masks_nonregressing": all(
                    value >= -FLOAT_COMPARISON_EPS for value in reductions
                ),
                "all_route_count_means_nonregressing": all(
                    value >= -FLOAT_COMPARISON_EPS for value in count_ratio_reductions
                ),
                "meets_expected_10pct": (
                    expected_reduction >= GOAL_REDUCTION_PCT - FLOAT_COMPARISON_EPS
                ),
                "meets_worst_mask_10pct": (
                    min(reductions) >= GOAL_REDUCTION_PCT - FLOAT_COMPARISON_EPS
                ),
                "meets_expected_10pct_with_nonregressing_tail": (
                    expected_reduction >= GOAL_REDUCTION_PCT - FLOAT_COMPARISON_EPS
                    and min(reductions) >= -FLOAT_COMPARISON_EPS
                ),
            }
        )
    return summary


def robust_rank_key(summary: dict[str, object]) -> tuple[float, float, float]:
    # Correctness failures sort last.  Tail is the primary objective; route
    # count and singleton aggregates only break ties.
    if not summary["bitwise_all"]:
        return (-math.inf, -math.inf, -math.inf)
    return (
        float(summary["worst_reduction_pct"]),
        float(summary["route_count_min_ratio_of_means_reduction_pct"]),
        float(summary.get("singleton_ratio_of_means_reduction_pct", -math.inf)),
    )


def make_joint_plans(
    plans_by_name: dict[str, Plan],
    screen_summaries: list[dict[str, object]],
    top_per_stage: int,
) -> list[Plan]:
    if top_per_stage <= 0:
        return []
    ranked = sorted(screen_summaries, key=robust_rank_key, reverse=True)
    g1_plans = [
        plans_by_name[str(summary["name"])]
        for summary in ranked
        if summary["stage"] == "g1" and summary["bitwise_all"]
    ][:top_per_stage]
    g2_plans = [
        plans_by_name[str(summary["name"])]
        for summary in ranked
        if summary["stage"] == "g2" and summary["bitwise_all"]
    ][:top_per_stage]
    joints = []
    for g1_plan in g1_plans:
        for g2_plan in g2_plans:
            joints.append(
                Plan(
                    name=(
                        f"joint_g1_{tile_label(g1_plan.g1)}"
                        f"__g2_{tile_label(g2_plan.g2)}"
                    ),
                    stage="joint",
                    g1=g1_plan.g1,
                    g2=g2_plan.g2,
                )
            )
    return joints


def plan_hardware_metadata(plan: Plan, sm_count: int) -> dict[str, object]:
    g1_programs = TOPK * math.ceil(INTERMEDIATE_SIZE / plan.g1.bn)
    g2_programs = TOPK * math.ceil(HIDDEN_SIZE / plan.g2.bn)
    return {
        "g1_programs_all_routes": g1_programs,
        "g2_programs_all_routes": g2_programs,
        "g1_grid_waves_over_sms": g1_programs / sm_count,
        "g2_grid_waves_over_sms": g2_programs / sm_count,
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            args.rounds,
            args.replays,
            args.audit_rounds,
            args.audit_replays,
        )
        <= 0
    ):
        raise ValueError("round and replay counts must be positive")
    if args.warmup_replays < 0:
        raise ValueError("warmup-replays must be nonnegative")
    if args.joint_top_per_stage < 0 or args.audit_all_masks_top < 0:
        raise ValueError("joint/audit top counts must be nonnegative")

    coordinate_plans = select_plans(args)
    screen_masks = masks_for_screen(args.screen_masks)
    if args.list_plans:
        print(
            json.dumps(
                {
                    "suite": args.suite,
                    "coordinate_plans": len(coordinate_plans),
                    "screen_masks": screen_masks,
                    "known_rejected_stage_tiles_excluded": [
                        {"stage": stage, "tile": asdict(tile)}
                        for stage, tile in sorted(
                            KNOWN_REJECTED_STAGE_TILES,
                            key=lambda item: (item[0], tile_label(item[1])),
                        )
                    ],
                    "plans": [
                        {
                            "name": plan.name,
                            "stage": plan.stage,
                            "g1": asdict(plan.g1),
                            "g2": asdict(plan.g2),
                        }
                        for plan in coordinate_plans
                    ],
                },
                indent=2,
            )
        )
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not args.allow_any_visible_device and visible_devices != "7":
        raise RuntimeError(
            "formal runs are restricted to CUDA_VISIBLE_DEVICES=7; "
            "pass --allow-any-visible-device only for debugging"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no performance result was produced")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)

    hidden = torch.randn((1, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    w1 = torch.empty(
        (LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ).normal_(std=HIDDEN_SIZE**-0.5)
    w2 = torch.empty(
        (LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ).normal_(std=INTERMEDIATE_SIZE**-0.5)
    weights = torch.rand((1, TOPK), device=device, dtype=torch.float32)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    ids = make_route_for_mask(1, device=device)
    expert_map = torch.full((GLOBAL_EXPERTS,), -1, device=device, dtype=torch.int32)
    expert_map[:LOCAL_EXPERTS] = torch.arange(
        LOCAL_EXPERTS, device=device, dtype=torch.int32
    )
    cache13 = torch.empty(
        TOPK * max(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    cache2 = torch.empty(TOPK * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    alias_output_cache2 = not args.disjoint_output
    output = (
        cache2[:HIDDEN_SIZE].view(1, HIDDEN_SIZE)
        if alias_output_cache2
        else torch.empty_like(hidden)
    )

    def op() -> torch.Tensor:
        return fm.fused_experts_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            global_num_experts=GLOBAL_EXPERTS,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=output,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
        )

    original_selector = fm._get_ep_decode_config
    original_naive_gate = fm._should_use_ep_naive_route
    original_route_block_gate = fm._should_use_ep_route_block
    original_local_rank_gate = getattr(fm, "_should_use_ep_m1_i2048_local_rank", None)
    original_dispatch = fm.dispatch_fused_moe_kernel
    active_plan = ""
    dispatch_audit: dict[str, list[dict[str, object]]] = {}

    def dispatch_spy(*dispatch_args, **dispatch_kwargs):
        config = dispatch_args[12]
        stage = "g1" if dispatch_kwargs.get("FUSE_SILU", False) else "g2"
        if active_plan:
            dispatch_audit.setdefault(active_plan, []).append(
                {
                    "stage": stage,
                    "config": config.copy(),
                    "sorted_token_ids_is_none": dispatch_args[7] is None,
                    "skip_invalid_experts": dispatch_kwargs.get(
                        "skip_invalid_experts", False
                    ),
                }
            )
        return original_dispatch(*dispatch_args, **dispatch_kwargs)

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    capture_errors: dict[str, str] = {}

    def capture_plans(plans: list[Plan]) -> list[Plan]:
        nonlocal active_plan
        captured = []
        for index, plan in enumerate(plans):
            active_plan = plan.name
            fm._get_ep_decode_config = selector_for(plan)
            try:
                graphs[plan.name] = capture(op)
                validate_dispatch(plan, dispatch_audit.get(plan.name, []))
                captured.append(plan)
            except Exception as error:
                capture_errors[plan.name] = f"{type(error).__name__}: {error}"[:1000]
                if len(capture_errors) <= 3:
                    print(
                        json.dumps(
                            {
                                "progress": "capture_error",
                                "plan": plan.name,
                                "error": capture_errors[plan.name],
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                torch.cuda.synchronize()
            if (index + 1) % 16 == 0 or index + 1 == len(plans):
                print(
                    json.dumps(
                        {
                            "progress": "capture",
                            "completed": index + 1,
                            "total": len(plans),
                            "valid": len(captured),
                            "errors": len(capture_errors),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        active_plan = ""
        return captured

    try:
        fm._should_use_ep_naive_route = lambda *_args, **_kwargs: True
        fm._should_use_ep_route_block = lambda *_args, **_kwargs: False
        if original_local_rank_gate is not None:
            # Keep this benchmark's reference and every candidate on raw
            # direct routing even when another worktree experiment enables a
            # launch-free local-rank kernel concurrently.
            fm._should_use_ep_m1_i2048_local_rank = lambda *_args, **_kwargs: False
        fm.dispatch_fused_moe_kernel = dispatch_spy

        valid_coordinate_plans = capture_plans(coordinate_plans)
        if BASELINE.name not in graphs:
            raise RuntimeError(
                f"baseline capture failed: {capture_errors.get(BASELINE.name)}"
            )
        if len(valid_coordinate_plans) < 2:
            raise RuntimeError("no coordinate candidate compiled successfully")

        coordinate_results = benchmark_masks(
            fm=fm,
            op=op,
            output=output,
            ids=ids,
            plans=valid_coordinate_plans,
            graphs=graphs,
            route_masks=screen_masks,
            rounds=args.rounds,
            replays=args.replays,
            warmup_replays=args.warmup_replays,
            summary_only=args.summary_only,
            phase="coordinate_screen",
        )
        coordinate_by_name = {plan.name: plan for plan in valid_coordinate_plans}
        coordinate_summaries = [
            summarize_masks(
                coordinate_by_name[name],
                values,
                complete_mask_space=len(screen_masks) == (1 << TOPK),
            )
            for name, values in coordinate_results.items()
        ]
        coordinate_summaries.sort(key=robust_rank_key, reverse=True)

        joint_plans = make_joint_plans(
            coordinate_by_name,
            coordinate_summaries,
            args.joint_top_per_stage,
        )
        valid_joint_plans = capture_plans(joint_plans)
        joint_results = {}
        joint_summaries = []
        if valid_joint_plans:
            joint_results = benchmark_masks(
                fm=fm,
                op=op,
                output=output,
                ids=ids,
                plans=[BASELINE, *valid_joint_plans],
                graphs=graphs,
                route_masks=screen_masks,
                rounds=args.rounds,
                replays=args.replays,
                warmup_replays=args.warmup_replays,
                summary_only=args.summary_only,
                phase="joint_screen",
            )
            joint_by_name = {plan.name: plan for plan in valid_joint_plans}
            joint_summaries = [
                summarize_masks(
                    joint_by_name[name],
                    values,
                    complete_mask_space=len(screen_masks) == (1 << TOPK),
                )
                for name, values in joint_results.items()
            ]
            joint_summaries.sort(key=robust_rank_key, reverse=True)

        all_valid_plans = [
            *[plan for plan in valid_coordinate_plans if plan.name != BASELINE.name],
            *valid_joint_plans,
        ]
        all_plan_by_name = {plan.name: plan for plan in all_valid_plans}
        all_screen_summaries = sorted(
            [*coordinate_summaries, *joint_summaries],
            key=robust_rank_key,
            reverse=True,
        )

        promoted_names = [
            str(summary["name"])
            for summary in all_screen_summaries
            if summary["bitwise_all"]
        ][: args.audit_all_masks_top]
        formal_results = {}
        formal_summaries = []
        if promoted_names:
            promoted_plans = [all_plan_by_name[name] for name in promoted_names]
            formal_results = benchmark_masks(
                fm=fm,
                op=op,
                output=output,
                ids=ids,
                plans=[BASELINE, *promoted_plans],
                graphs=graphs,
                route_masks=list(range(1 << TOPK)),
                rounds=args.audit_rounds,
                replays=args.audit_replays,
                warmup_replays=args.warmup_replays,
                summary_only=args.summary_only,
                phase="formal_256_mask_audit",
            )
            formal_summaries = [
                summarize_masks(
                    all_plan_by_name[name],
                    formal_results[name],
                    complete_mask_space=True,
                )
                for name in promoted_names
            ]
            formal_summaries.sort(key=robust_rank_key, reverse=True)
    finally:
        fm._get_ep_decode_config = original_selector
        fm._should_use_ep_naive_route = original_naive_gate
        fm._should_use_ep_route_block = original_route_block_gate
        if original_local_rank_gate is not None:
            fm._should_use_ep_m1_i2048_local_rank = original_local_rank_gate
        fm.dispatch_fused_moe_kernel = original_dispatch

    device_properties = torch.cuda.get_device_properties(0)
    plan_metadata = {
        plan.name: plan_hardware_metadata(plan, device_properties.multi_processor_count)
        for plan in [
            *valid_coordinate_plans,
            *valid_joint_plans,
        ]
    }
    result = {
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_devices": visible_devices,
        "sm_count": device_properties.multi_processor_count,
        "shape": {
            "M": 1,
            "global_E": GLOBAL_EXPERTS,
            "local_E": LOCAL_EXPERTS,
            "H": HIDDEN_SIZE,
            "I": INTERMEDIATE_SIZE,
            "topk": TOPK,
            "dtype": "torch.bfloat16",
        },
        "contract": {
            "routing": "forced raw direct; route-block disabled",
            "timing": "same-process alternating ABBA/BAAB CUDA Graph",
            "alias_output_cache2": alias_output_cache2,
            "candidate_ranking": "worst mask, then worst route-count ratio-of-means",
            "formal_prior": "Hypergeometric(N=288,K=18,n=8), uniform masks within count",
        },
        "search": {
            "suite": args.suite,
            "screen_mask_kind": args.screen_masks,
            "screen_masks": screen_masks,
            "screen_rounds": args.rounds,
            "screen_replays": args.replays,
            "warmup_replays": args.warmup_replays,
            "joint_top_per_stage": args.joint_top_per_stage,
            "formal_top": args.audit_all_masks_top,
            "formal_rounds": args.audit_rounds,
            "formal_replays": args.audit_replays,
            "coordinate_requested": len(coordinate_plans) - 1,
            "coordinate_captured": len(valid_coordinate_plans) - 1,
            "joint_requested": len(joint_plans),
            "joint_captured": len(valid_joint_plans),
            "capture_errors": capture_errors,
            "known_rejected_stage_tiles_excluded": [
                {"stage": stage, "tile": asdict(tile)}
                for stage, tile in sorted(
                    KNOWN_REJECTED_STAGE_TILES,
                    key=lambda item: (item[0], tile_label(item[1])),
                )
            ],
        },
        "coordinate_screen_summaries": coordinate_summaries,
        "joint_screen_summaries": joint_summaries,
        "formal_256_mask_summaries": formal_summaries,
        "formal_any_meets_expected_10pct": any(
            summary["meets_expected_10pct"] for summary in formal_summaries
        ),
        "formal_any_meets_expected_10pct_with_nonregressing_tail": any(
            summary["meets_expected_10pct_with_nonregressing_tail"]
            for summary in formal_summaries
        ),
        "plan_hardware_metadata": plan_metadata,
        "dispatch_audit": dispatch_audit,
    }
    if not args.summary_only:
        result["coordinate_screen_results"] = coordinate_results
        result["joint_screen_results"] = joint_results
        result["formal_256_mask_results"] = formal_results

    rendered = json.dumps(result, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
