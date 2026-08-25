#!/usr/bin/env python3
"""Full-operator CUDA Graph search for fused-MoE M=1 EP-naive tiles.

This experiment exhaustively searches one stage at a time while holding the
other stage at the current production plan:

* paired GEMM1: actual BN {16,32,64} x BK {64,128,256} x W {2,4,8}
  x stages {2,3,4,5};
* GEMM2: BN {64,128,256} x BK {32,64,128} x W {2,4,8}
  x stages {2,3,4,5}, GROUP_SIZE_M=1.

Every point captures and times the complete ``fused_experts_impl`` graph.  A
single captured graph is replayed after changing the route tensor between one
local route and eight local routes, proving graph input replay semantics.  The
production module is monkeypatched only while captures/eager references are
built; no production source is modified.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Plan:
    name: str
    stage: str
    g1: tuple[int, int, int, int]
    g2: tuple[int, int, int, int]

    def selector_configs(self) -> tuple[dict[str, int], dict[str, int]]:
        g1_bn, g1_bk, g1_warps, g1_stages = self.g1
        g2_bn, g2_bk, g2_warps, g2_stages = self.g2
        return (
            {
                "BLOCK_SIZE_M": 16,
                # Production halves selector BN for the paired gate/up dot.
                "BLOCK_SIZE_N": 2 * g1_bn,
                "BLOCK_SIZE_K": g1_bk,
                "GROUP_SIZE_M": 1,
                "num_warps": g1_warps,
                "num_stages": g1_stages,
            },
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": g2_bn,
                "BLOCK_SIZE_K": g2_bk,
                "GROUP_SIZE_M": 1,
                "num_warps": g2_warps,
                "num_stages": g2_stages,
            },
        )


BASE_G1 = (32, 128, 4, 3)
BASE_G2 = (128, 64, 4, 4)


def make_plans() -> list[Plan]:
    plans = [Plan("baseline", "baseline", BASE_G1, BASE_G2)]
    for values in itertools.product(
        (16, 32, 64),
        (64, 128, 256),
        (2, 4, 8),
        (2, 3, 4, 5),
    ):
        if values == BASE_G1:
            continue
        bn, bk, warps, stages = values
        plans.append(
            Plan(
                f"g1_bn{bn}_bk{bk}_w{warps}_s{stages}",
                "g1",
                values,
                BASE_G2,
            )
        )
    for values in itertools.product(
        (64, 128, 256),
        (32, 64, 128),
        (2, 4, 8),
        (2, 3, 4, 5),
    ):
        if values == BASE_G2:
            continue
        bn, bk, warps, stages = values
        plans.append(
            Plan(
                f"g2_bn{bn}_bk{bk}_w{warps}_s{stages}",
                "g2",
                BASE_G1,
                values,
            )
        )
    # Joint validation point selected only after the two exhaustive coordinate
    # sweeps.  Keeping it named makes the long A/B command reproducible.
    plans.append(
        Plan(
            "combined_g1_bn16_bk128_w2_s5_g2_bn64_bk64_w4_s4",
            "combined",
            (16, 128, 2, 5),
            (64, 64, 4, 4),
        )
    )
    plans.append(
        Plan(
            "combined_g1_bn16_bk128_w2_s3_g2_bn64_bk64_w4_s4",
            "combined",
            (16, 128, 2, 3),
            (64, 64, 4, 4),
        )
    )
    return plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=400)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        choices=(1280, 2048),
        default=2048,
        help="fused-MoE routed expert intermediate size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="debug-only prefix limit; zero searches the complete space",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated named candidates to retest with baseline",
    )
    parser.add_argument(
        "--audit-all-route-counts",
        action="store_true",
        help="with --only, A/B the candidate at every local route count 0..8",
    )
    parser.add_argument(
        "--audit-singleton-positions",
        action="store_true",
        help="with --only, A/B one local route at each top-k position",
    )
    parser.add_argument("--audit-rounds", type=int, default=7)
    parser.add_argument("--audit-replays", type=int, default=1000)
    parser.add_argument("--audit-summary-only", action="store_true")
    parser.add_argument(
        "--alias-output-cache2",
        action="store_true",
        help="Exercise the modular caller layout where output aliases cache2.",
    )
    return parser.parse_args()


def selector_for(plan: Plan):
    g1, g2 = plan.selector_configs()

    def selector(*positional, gemm_stage=None, **_keyword):
        stage = gemm_stage if gemm_stage is not None else positional[-1]
        return (g1 if stage == "gemm1" else g2).copy()

    return selector


def capture(fn):
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


def elapsed(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / replays)


def make_route(local_routes: int, *, device: torch.device) -> torch.Tensor:
    if not 0 <= local_routes <= 8:
        raise ValueError(f"local_routes must be in [0, 8], got {local_routes}")
    route = 18 + torch.arange(8, device=device, dtype=torch.int32)
    route[:local_routes] = torch.arange(local_routes, device=device, dtype=torch.int32)
    return route.view(1, 8)


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.replays <= 0 or args.warmup_replays < 0:
        raise ValueError(
            "rounds/replays must be positive and warmup-replays nonnegative"
        )

    fm = importlib.import_module("flag_gems.fused.fused_moe")
    plans = make_plans()
    if args.only:
        requested = args.only.split(",")
        matches = [plan for name in requested for plan in plans if plan.name == name]
        if len(matches) != len(requested):
            found = {plan.name for plan in matches}
            missing = [name for name in requested if name not in found]
            raise ValueError(f"unknown --only candidates: {missing}")
        plans = [plans[0], *matches]
    if args.limit:
        plans = plans[: args.limit]
    plans_by_name = {plan.name: plan for plan in plans}
    torch.manual_seed(20260824)

    m, global_e, local_e, h, intermediate, topk = (
        1,
        288,
        18,
        4096,
        args.intermediate_size,
        8,
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden = torch.randn((m, h), device=device, dtype=dtype)
    w1 = torch.empty(
        (local_e, 2 * intermediate, h), device=device, dtype=dtype
    ).normal_(std=h**-0.5)
    w2 = torch.empty((local_e, h, intermediate), device=device, dtype=dtype).normal_(
        std=intermediate**-0.5
    )
    weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype)
    ids = make_route(1, device=device)
    expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device=device, dtype=torch.int32)
    cache13 = torch.empty(
        m * topk * max(2 * intermediate, h), device=device, dtype=dtype
    )
    cache2 = torch.empty(m * topk * intermediate, device=device, dtype=dtype)
    output = (
        cache2[: m * h].view(m, h)
        if args.alias_output_cache2
        else torch.empty_like(hidden)
    )

    def op():
        original_local_rank_gate = fm._should_use_ep_m1_i2048_local_rank
        fm._should_use_ep_m1_i2048_local_rank = lambda *_args, **_kwargs: False
        try:
            return fm.fused_experts_impl(
                hidden,
                w1,
                w2,
                weights,
                ids,
                global_num_experts=global_e,
                expert_map=expert_map,
                gemm1_clamp_limit=10.0,
                output=output,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
            )
        finally:
            fm._should_use_ep_m1_i2048_local_rank = original_local_rank_gate

    original_selector = fm._get_ep_decode_config
    original_naive_gate = fm._should_use_ep_naive_route
    original_route_block_gate = fm._should_use_ep_route_block
    # This file measures raw-route GEMM configs. Keep both the baseline and
    # candidates on direct routing even when integrated fast paths exist for
    # the same M=1/I2048 shape.  ``op`` suppresses local-rank dispatch for each
    # eager/capture call and restores it immediately afterwards.
    fm._should_use_ep_naive_route = lambda *_args, **_kwargs: True
    fm._should_use_ep_route_block = lambda *_args, **_kwargs: False
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    capture_errors: dict[str, str] = {}
    try:
        for index, plan in enumerate(plans):
            fm._get_ep_decode_config = selector_for(plan)
            try:
                graphs[plan.name] = capture(op)
            except Exception as error:
                capture_errors[plan.name] = (f"{type(error).__name__}: {error}")[:500]
                torch.cuda.synchronize()
            if index and index % 24 == 0:
                print(
                    json.dumps(
                        {
                            "progress": "capture",
                            "completed": index,
                            "total": len(plans),
                            "valid": len(graphs),
                            "errors": len(capture_errors),
                        }
                    ),
                    flush=True,
                )
    finally:
        fm._get_ep_decode_config = original_selector

    if "baseline" not in graphs:
        raise RuntimeError(f"baseline capture failed: {capture_errors.get('baseline')}")

    valid_plans = [plan for plan in plans if plan.name in graphs]
    correctness: dict[str, dict[str, dict[str, object]]] = {}
    try:
        for local_routes in (1, 8):
            route_name = "one_local" if local_routes == 1 else "all_local8"
            ids.copy_(make_route(local_routes, device=device))
            route_correctness: dict[str, dict[str, object]] = {}

            fm._get_ep_decode_config = selector_for(plans_by_name["baseline"])
            baseline_eager = op().clone()
            torch.cuda.synchronize()

            for plan in valid_plans:
                fm._get_ep_decode_config = selector_for(plan)
                eager = op().clone()
                torch.cuda.synchronize()
                graphs[plan.name].replay()
                torch.cuda.synchronize()
                graph_actual = output.clone()
                torch.cuda.synchronize()
                route_correctness[plan.name] = {
                    "bitwise_vs_baseline": bool(torch.equal(eager, baseline_eager)),
                    "graph_bitwise_vs_eager": bool(torch.equal(graph_actual, eager)),
                    "max_abs_vs_baseline": float(
                        (eager.float() - baseline_eager.float()).abs().max().item()
                    ),
                }
            correctness[route_name] = route_correctness
    finally:
        fm._get_ep_decode_config = original_selector

    timings: dict[str, dict[str, dict[str, object]]] = {}
    baseline_graph = graphs["baseline"]
    for local_routes in (1, 8):
        route_name = "one_local" if local_routes == 1 else "all_local8"
        ids.copy_(make_route(local_routes, device=device))
        for _ in range(args.warmup_replays):
            baseline_graph.replay()
        torch.cuda.synchronize()

        route_timings: dict[str, dict[str, object]] = {}
        for index, plan in enumerate(valid_plans):
            graph = graphs[plan.name]
            for _ in range(args.warmup_replays):
                graph.replay()
            torch.cuda.synchronize()

            baseline_samples = []
            candidate_samples = []
            paired_reductions = []
            for round_index in range(args.rounds):
                if round_index % 2 == 0:
                    a1 = elapsed(baseline_graph, args.replays)
                    b1 = elapsed(graph, args.replays)
                    b2 = elapsed(graph, args.replays)
                    a2 = elapsed(baseline_graph, args.replays)
                else:
                    b1 = elapsed(graph, args.replays)
                    a1 = elapsed(baseline_graph, args.replays)
                    a2 = elapsed(baseline_graph, args.replays)
                    b2 = elapsed(graph, args.replays)
                baseline_value = 0.5 * (a1 + a2)
                candidate_value = 0.5 * (b1 + b2)
                baseline_samples.append(baseline_value)
                candidate_samples.append(candidate_value)
                paired_reductions.append(
                    100.0 * (1.0 - candidate_value / baseline_value)
                )
            route_timings[plan.name] = {
                "baseline_median_ms": statistics.median(baseline_samples),
                "candidate_median_ms": statistics.median(candidate_samples),
                "median_paired_reduction_pct": statistics.median(paired_reductions),
                "baseline_samples_ms": baseline_samples,
                "candidate_samples_ms": candidate_samples,
            }
            if index and index % 24 == 0:
                print(
                    json.dumps(
                        {
                            "progress": "timing",
                            "routing": route_name,
                            "completed": index,
                            "total": len(valid_plans),
                        }
                    ),
                    flush=True,
                )
        timings[route_name] = route_timings

    rows = []
    for plan in valid_plans:
        one = timings["one_local"][plan.name]
        dense = timings["all_local8"][plan.name]
        one_correct = correctness["one_local"][plan.name]
        dense_correct = correctness["all_local8"][plan.name]
        row = {
            "name": plan.name,
            "stage": plan.stage,
            "g1": plan.g1,
            "g2": plan.g2,
            "one_local_ms": one["candidate_median_ms"],
            "one_local_reduction_pct": one["median_paired_reduction_pct"],
            "all_local8_ms": dense["candidate_median_ms"],
            "all_local8_reduction_pct": dense["median_paired_reduction_pct"],
            "worst_reduction_pct": min(
                one["median_paired_reduction_pct"],
                dense["median_paired_reduction_pct"],
            ),
            "bitwise_both": bool(
                one_correct["bitwise_vs_baseline"]
                and dense_correct["bitwise_vs_baseline"]
            ),
            "graph_bitwise_both": bool(
                one_correct["graph_bitwise_vs_eager"]
                and dense_correct["graph_bitwise_vs_eager"]
            ),
            "max_abs": max(
                one_correct["max_abs_vs_baseline"],
                dense_correct["max_abs_vs_baseline"],
            ),
        }
        rows.append(row)

    ranked = sorted(rows, key=lambda row: row["worst_reduction_pct"], reverse=True)
    qualifying = [
        row
        for row in ranked
        if row["name"] != "baseline"
        and row["worst_reduction_pct"] >= 1.0
        and row["bitwise_both"]
        and row["graph_bitwise_both"]
    ]
    correctness_failures = [
        row["name"]
        for row in rows
        if not row["bitwise_both"] or not row["graph_bitwise_both"]
    ]
    top_one = sorted(
        rows, key=lambda row: row["one_local_reduction_pct"], reverse=True
    )[: args.top_k]
    top_dense = sorted(
        rows, key=lambda row: row["all_local8_reduction_pct"], reverse=True
    )[: args.top_k]
    route_count_audit = {}
    route_count_audit_summary = {}
    if args.audit_all_route_counts:
        if not args.only or len(valid_plans) < 2:
            raise ValueError("--audit-all-route-counts requires valid --only plans")
        candidate_plans = [plan for plan in valid_plans if plan.name != "baseline"]
        route_count_audit = {plan.name: {} for plan in candidate_plans}
        try:
            for local_routes in range(9):
                ids.copy_(make_route(local_routes, device=device))
                fm._get_ep_decode_config = selector_for(plans_by_name["baseline"])
                baseline_eager = op().clone()
                torch.cuda.synchronize()
                baseline_graph.replay()
                torch.cuda.synchronize()
                baseline_graph_actual = output.clone()
                torch.cuda.synchronize()
                for candidate_plan in candidate_plans:
                    candidate_graph = graphs[candidate_plan.name]
                    fm._get_ep_decode_config = selector_for(candidate_plan)
                    candidate_eager = op().clone()
                    torch.cuda.synchronize()
                    candidate_graph.replay()
                    torch.cuda.synchronize()
                    candidate_graph_actual = output.clone()
                    torch.cuda.synchronize()

                    for _ in range(args.warmup_replays):
                        baseline_graph.replay()
                        candidate_graph.replay()
                    torch.cuda.synchronize()
                    baseline_samples = []
                    candidate_samples = []
                    reductions = []
                    for round_index in range(args.audit_rounds):
                        if round_index % 2 == 0:
                            a1 = elapsed(baseline_graph, args.audit_replays)
                            b1 = elapsed(candidate_graph, args.audit_replays)
                            b2 = elapsed(candidate_graph, args.audit_replays)
                            a2 = elapsed(baseline_graph, args.audit_replays)
                        else:
                            b1 = elapsed(candidate_graph, args.audit_replays)
                            a1 = elapsed(baseline_graph, args.audit_replays)
                            a2 = elapsed(baseline_graph, args.audit_replays)
                            b2 = elapsed(candidate_graph, args.audit_replays)
                        a_value = 0.5 * (a1 + a2)
                        b_value = 0.5 * (b1 + b2)
                        baseline_samples.append(a_value)
                        candidate_samples.append(b_value)
                        reductions.append(100.0 * (1.0 - b_value / a_value))
                    route_count_audit[candidate_plan.name][str(local_routes)] = {
                        "baseline_ms": statistics.median(baseline_samples),
                        "candidate_ms": statistics.median(candidate_samples),
                        "median_paired_reduction_pct": statistics.median(reductions),
                        "candidate_bitwise_vs_baseline": bool(
                            torch.equal(candidate_eager, baseline_eager)
                        ),
                        "baseline_graph_bitwise": bool(
                            torch.equal(baseline_graph_actual, baseline_eager)
                        ),
                        "candidate_graph_bitwise": bool(
                            torch.equal(candidate_graph_actual, candidate_eager)
                        ),
                    }
        finally:
            fm._get_ep_decode_config = original_selector
        for name, count_results in route_count_audit.items():
            reductions = {
                count: values["median_paired_reduction_pct"]
                for count, values in count_results.items()
            }
            # Top-k selects distinct experts. Under a uniform global router,
            # the number of routes owned by one EP16 rank is hypergeometric:
            # population 288, 18 local successes, 8 draws without replacement.
            probabilities = [
                math.comb(local_e, count)
                * math.comb(global_e - local_e, topk - count)
                / math.comb(global_e, topk)
                for count in range(topk + 1)
            ]
            expected_baseline_ms = sum(
                probabilities[count] * count_results[str(count)]["baseline_ms"]
                for count in range(topk + 1)
            )
            expected_candidate_ms = sum(
                probabilities[count] * count_results[str(count)]["candidate_ms"]
                for count in range(topk + 1)
            )
            worst_count = min(reductions, key=reductions.get)
            route_count_audit_summary[name] = {
                "worst_route_count": int(worst_count),
                "worst_reduction_pct": reductions[worst_count],
                "max_absolute_regression_us": 1000.0
                * max(
                    values["candidate_ms"] - values["baseline_ms"]
                    for values in count_results.values()
                ),
                "all_route_counts_non_regressing": all(
                    value >= 0.0 for value in reductions.values()
                ),
                "all_route_counts_at_least_1pct": all(
                    value >= 1.0 for value in reductions.values()
                ),
                "reductions_pct_0_to_8": [reductions[str(count)] for count in range(9)],
                "uniform_hypergeometric_probabilities_0_to_8": probabilities,
                "uniform_expected_baseline_ms": expected_baseline_ms,
                "uniform_expected_candidate_ms": expected_candidate_ms,
                "uniform_expected_reduction_pct": 100.0
                * (1.0 - expected_candidate_ms / expected_baseline_ms),
                "bitwise_all": all(
                    values["candidate_bitwise_vs_baseline"]
                    and values["baseline_graph_bitwise"]
                    and values["candidate_graph_bitwise"]
                    for values in count_results.values()
                ),
            }

    singleton_position_audit = {}
    singleton_position_audit_summary = {}
    if args.audit_singleton_positions:
        if not args.only or len(valid_plans) < 2:
            raise ValueError("--audit-singleton-positions requires valid --only plans")
        candidate_plans = [plan for plan in valid_plans if plan.name != "baseline"]
        singleton_position_audit = {plan.name: {} for plan in candidate_plans}
        try:
            for position in range(topk):
                singleton_route = make_route(0, device=device)
                # Change only route position; keep the expert weight address
                # fixed so cache behavior is not mixed into this audit.
                singleton_route[0, position] = 0
                ids.copy_(singleton_route)
                fm._get_ep_decode_config = selector_for(plans_by_name["baseline"])
                baseline_eager = op().clone()
                torch.cuda.synchronize()
                baseline_graph.replay()
                torch.cuda.synchronize()
                baseline_graph_actual = output.clone()
                torch.cuda.synchronize()
                for candidate_plan in candidate_plans:
                    candidate_graph = graphs[candidate_plan.name]
                    fm._get_ep_decode_config = selector_for(candidate_plan)
                    candidate_eager = op().clone()
                    torch.cuda.synchronize()
                    candidate_graph.replay()
                    torch.cuda.synchronize()
                    candidate_graph_actual = output.clone()
                    torch.cuda.synchronize()

                    for _ in range(args.warmup_replays):
                        baseline_graph.replay()
                        candidate_graph.replay()
                    torch.cuda.synchronize()
                    baseline_samples = []
                    candidate_samples = []
                    reductions = []
                    for round_index in range(args.audit_rounds):
                        if round_index % 2 == 0:
                            a1 = elapsed(baseline_graph, args.audit_replays)
                            b1 = elapsed(candidate_graph, args.audit_replays)
                            b2 = elapsed(candidate_graph, args.audit_replays)
                            a2 = elapsed(baseline_graph, args.audit_replays)
                        else:
                            b1 = elapsed(candidate_graph, args.audit_replays)
                            a1 = elapsed(baseline_graph, args.audit_replays)
                            a2 = elapsed(baseline_graph, args.audit_replays)
                            b2 = elapsed(candidate_graph, args.audit_replays)
                        a_value = 0.5 * (a1 + a2)
                        b_value = 0.5 * (b1 + b2)
                        baseline_samples.append(a_value)
                        candidate_samples.append(b_value)
                        reductions.append(100.0 * (1.0 - b_value / a_value))
                    singleton_position_audit[candidate_plan.name][str(position)] = {
                        "baseline_ms": statistics.median(baseline_samples),
                        "candidate_ms": statistics.median(candidate_samples),
                        "median_paired_reduction_pct": statistics.median(reductions),
                        "candidate_bitwise_vs_baseline": bool(
                            torch.equal(candidate_eager, baseline_eager)
                        ),
                        "baseline_graph_bitwise": bool(
                            torch.equal(baseline_graph_actual, baseline_eager)
                        ),
                        "candidate_graph_bitwise": bool(
                            torch.equal(candidate_graph_actual, candidate_eager)
                        ),
                    }
        finally:
            fm._get_ep_decode_config = original_selector
        for name, position_results in singleton_position_audit.items():
            position_reductions = [
                position_results[str(position)]["median_paired_reduction_pct"]
                for position in range(topk)
            ]
            position_expected_baseline_ms = statistics.mean(
                position_results[str(position)]["baseline_ms"]
                for position in range(topk)
            )
            position_expected_candidate_ms = statistics.mean(
                position_results[str(position)]["candidate_ms"]
                for position in range(topk)
            )
            singleton_position_audit_summary[name] = {
                "reductions_pct_position_0_to_7": position_reductions,
                "median_reduction_pct": statistics.median(position_reductions),
                "worst_reduction_pct": min(position_reductions),
                "position_uniform_expected_baseline_ms": position_expected_baseline_ms,
                "position_uniform_expected_candidate_ms": position_expected_candidate_ms,
                "position_uniform_expected_reduction_pct": 100.0
                * (
                    1.0 - position_expected_candidate_ms / position_expected_baseline_ms
                ),
                "max_absolute_regression_us": 1000.0
                * max(
                    values["candidate_ms"] - values["baseline_ms"]
                    for values in position_results.values()
                ),
                "all_positions_non_regressing": all(
                    value >= 0.0 for value in position_reductions
                ),
                "bitwise_all": all(
                    values["candidate_bitwise_vs_baseline"]
                    and values["baseline_graph_bitwise"]
                    and values["candidate_graph_bitwise"]
                    for values in position_results.values()
                ),
            }
    result = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "M": m,
            "global_E": global_e,
            "local_E": local_e,
            "H": h,
            "I": intermediate,
            "topk": topk,
            "dtype": str(dtype),
        },
        "alias_output_cache2": args.alias_output_cache2,
        "search": {
            "g1_space": "3 BN x 3 BK x 3 warps x 4 stages = 108",
            "g2_space": "3 BN x 3 BK x 3 warps x 4 stages = 108",
            "captured_plans": len(valid_plans),
            "capture_errors": len(capture_errors),
            "rounds": args.rounds,
            "replays_per_bracket": args.replays,
        },
        "baseline": next(row for row in rows if row["name"] == "baseline"),
        "qualifying_count": len(qualifying),
        "qualifying": qualifying[: args.top_k],
        "top_joint": ranked[: args.top_k],
        "top_one_local": top_one,
        "top_all_local8": top_dense,
        "correctness_failures": correctness_failures,
        "capture_error_examples": dict(list(capture_errors.items())[:20]),
        "route_count_audit_summary": route_count_audit_summary,
        "route_count_audit": {} if args.audit_summary_only else route_count_audit,
        "singleton_position_audit_summary": singleton_position_audit_summary,
        "singleton_position_audit": (
            {} if args.audit_summary_only else singleton_position_audit
        ),
    }
    fm._should_use_ep_naive_route = original_naive_gate
    fm._should_use_ep_route_block = original_route_block_gate
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
