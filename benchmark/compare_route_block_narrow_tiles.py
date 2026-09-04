#!/usr/bin/env python3
"""Compare legacy direct, shared route-block, and production narrow route-block.

All policies call ``fused_experts_impl`` with generic fused GEMM kernels,
caller-owned workspaces, and the plugin-style output/cache2 alias:

* legacy baseline: raw-route GEMMs with shared tiles;
* shared baseline: G1 BN64/BK128/W4/S3 (effective fused BN32),
  G2 BN128/BK64/W4/S4 through route-block alignment;
* default candidate/production M1 plan: G1 BN32/BK128/W2/S3
  (effective fused BN16), G2 BN64/BK64/W4/S4 through route-block alignment.

No production source is changed. Timing uses alternating symmetric brackets.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from dataclasses import dataclass

import torch

POLICIES = ("legacy_direct", "shared_route_block", "candidate_route_block")
SCENARIOS = (
    *(f"singleton-p{position}" for position in range(8)),
    "no-local",
    "all-local",
    "repeated-local",
    "local2-prefix",
    "local2-spread",
    "local4-prefix",
    "local4-spread",
)


@dataclass(frozen=True)
class CandidatePlan:
    g1_block_n: int
    g1_warps: int
    g1_stages: int
    g2_block_n: int
    g2_warps: int
    g2_stages: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--ep-rank", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument("--g1-block-n", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--g1-warps", type=int, choices=(2, 4), default=2)
    parser.add_argument("--g1-stages", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--g2-block-n", type=int, choices=(64, 128), default=64)
    parser.add_argument("--g2-warps", type=int, choices=(2, 4), default=4)
    parser.add_argument("--g2-stages", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def make_routes(
    scenario: str,
    *,
    global_experts: int,
    local_experts: int,
    shard_begin: int,
) -> torch.Tensor:
    topk = 8
    flat = torch.arange(topk, device="cuda", dtype=torch.int32)
    remote_index = flat.remainder(global_experts - local_experts)
    ids = remote_index + (remote_index >= shard_begin).to(torch.int32) * local_experts

    if scenario.startswith("singleton-p"):
        position = int(scenario.removeprefix("singleton-p"))
        ids[position] = shard_begin
    elif scenario == "no-local":
        pass
    elif scenario == "all-local":
        ids = shard_begin + flat.remainder(local_experts)
    elif scenario == "repeated-local":
        ids.fill_(shard_begin)
    elif scenario in {
        "local2-prefix",
        "local2-spread",
        "local4-prefix",
        "local4-spread",
    }:
        positions = {
            "local2-prefix": (0, 1),
            "local2-spread": (0, 7),
            "local4-prefix": (0, 1, 2, 3),
            "local4-spread": (0, 2, 5, 7),
        }[scenario]
        for local_id, position in enumerate(positions):
            ids[position] = shard_begin + local_id
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    return ids.view(1, topk)


def elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / replays)


def capture_under_policy(
    fm,
    op,
    *,
    m1_plan: dict[str, dict],
    direct_route: bool,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, int, list[dict]]:
    original_m1_plan = fm._HOPPER_EP_M1_I2048_PLAN
    original_route_gate = fm._should_use_ep_naive_route
    original_route_block_gate = fm._should_use_ep_route_block
    original_local_rank_gate = fm._should_use_ep_m1_i2048_local_rank
    original_align = fm.moe_align_block_size_ep_route_block
    original_dispatch = fm.dispatch_fused_moe_kernel
    alignment_calls = 0
    dispatch_audit: list[dict] = []

    def select_naive_route(*_args, **_kwargs):
        return direct_route

    def select_route_block(*_args, **_kwargs):
        return not direct_route

    def align_spy(*args, **kwargs):
        nonlocal alignment_calls
        alignment_calls += 1
        return original_align(*args, **kwargs)

    def dispatch_spy(*args, **kwargs):
        config = args[12]
        dispatch_audit.append(
            {
                "stage": "g1" if args[1] is w1 else "g2" if args[1] is w2 else "?",
                "sorted_token_ids_is_none": args[7] is None,
                "expert_map_is_input": kwargs.get("expert_map") is not None,
                "skip_invalid_experts": bool(kwargs.get("skip_invalid_experts", False)),
                "block_m": config["BLOCK_SIZE_M"],
                "block_n": config["BLOCK_SIZE_N"],
                "block_k": config["BLOCK_SIZE_K"],
                "num_warps": config["num_warps"],
                "num_stages": config["num_stages"],
                "fuse_silu": bool(kwargs.get("FUSE_SILU", False)),
            }
        )
        return original_dispatch(*args, **kwargs)

    # Production selects this plan only after the strict route-block gate. Swap
    # the plan itself—not the earlier shared-config selector—so each side
    # exercises the exact production override point instead of self-comparing.
    fm._HOPPER_EP_M1_I2048_PLAN = {
        stage: config.copy() for stage, config in m1_plan.items()
    }
    fm._should_use_ep_naive_route = select_naive_route
    fm._should_use_ep_route_block = select_route_block
    fm._should_use_ep_m1_i2048_local_rank = lambda *_args, **_kwargs: False
    fm.moe_align_block_size_ep_route_block = align_spy
    fm.dispatch_fused_moe_kernel = dispatch_spy
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                op()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        eager = op().clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = op()
    finally:
        fm._HOPPER_EP_M1_I2048_PLAN = original_m1_plan
        fm._should_use_ep_naive_route = original_route_gate
        fm._should_use_ep_route_block = original_route_block_gate
        fm._should_use_ep_m1_i2048_local_rank = original_local_rank_gate
        fm.moe_align_block_size_ep_route_block = original_align
        fm.dispatch_fused_moe_kernel = original_dispatch
    return graph, eager, graph_output, alignment_calls, dispatch_audit[-2:]


def benchmark_scenario(
    *,
    fm,
    scenario: str,
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    expert_map: torch.Tensor,
    global_experts: int,
    local_experts: int,
    shard_begin: int,
    candidate_plan: CandidatePlan,
    rounds: int,
    replays: int,
    summary_only: bool,
) -> dict:
    m, hidden_size = hidden.shape
    topk = weights.shape[1]
    intermediate_size = w2.shape[2]
    ids = make_routes(
        scenario,
        global_experts=global_experts,
        local_experts=local_experts,
        shard_begin=shard_begin,
    )
    shared_plan = {
        stage: config.copy() for stage, config in fm._HOPPER_EP_DECODE_PLAN.items()
    }
    policy_plans = {
        "legacy_direct": shared_plan,
        "shared_route_block": {
            stage: config.copy() for stage, config in fm._HOPPER_EP_DECODE_PLAN.items()
        },
        "candidate_route_block": {
            "gemm1": {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": candidate_plan.g1_block_n,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": candidate_plan.g1_warps,
                "num_stages": candidate_plan.g1_stages,
            },
            "gemm2": {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": candidate_plan.g2_block_n,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": candidate_plan.g2_warps,
                "num_stages": candidate_plan.g2_stages,
            },
        },
    }
    graphs = {}
    eager_outputs = {}
    graph_outputs = {}
    alignment_calls = {}
    dispatch_audits = {}
    keepalive = []
    for policy, m1_plan in policy_plans.items():
        cache13 = torch.empty(
            m * topk * max(2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=hidden.dtype,
        )
        cache2 = torch.empty(
            m * topk * intermediate_size, device="cuda", dtype=hidden.dtype
        )
        output = cache2[: m * hidden_size].view(m, hidden_size)
        keepalive.extend((cache13, cache2, output))

        def op(_cache13=cache13, _cache2=cache2, _output=output):
            return fm.fused_experts_impl(
                hidden,
                w1,
                w2,
                weights,
                ids,
                global_num_experts=global_experts,
                expert_map=expert_map,
                gemm1_clamp_limit=10.0,
                output=_output,
                intermediate_cache13=_cache13,
                intermediate_cache2=_cache2,
            )

        graph, eager, graph_output, align_count, dispatch_audit = capture_under_policy(
            fm,
            op,
            m1_plan=m1_plan,
            direct_route=policy == "legacy_direct",
            w1=w1,
            w2=w2,
        )
        graphs[policy] = graph
        eager_outputs[policy] = eager
        graph_outputs[policy] = graph_output
        alignment_calls[policy] = align_count
        dispatch_audits[policy] = dispatch_audit

    expected_dispatch_tiles = {
        "legacy_direct": (
            {
                "stage": "g1",
                "block_n": fm._HOPPER_EP_DECODE_PLAN["gemm1"]["BLOCK_SIZE_N"] // 2,
                "num_warps": fm._HOPPER_EP_DECODE_PLAN["gemm1"]["num_warps"],
                "sorted_token_ids_is_none": True,
                "expert_map_is_input": True,
                "skip_invalid_experts": True,
            },
            {
                "stage": "g2",
                "block_n": fm._HOPPER_EP_DECODE_PLAN["gemm2"]["BLOCK_SIZE_N"],
                "num_warps": fm._HOPPER_EP_DECODE_PLAN["gemm2"]["num_warps"],
                "sorted_token_ids_is_none": True,
                "expert_map_is_input": True,
                "skip_invalid_experts": True,
            },
        ),
        "shared_route_block": (
            {
                "stage": "g1",
                "block_n": fm._HOPPER_EP_DECODE_PLAN["gemm1"]["BLOCK_SIZE_N"] // 2,
                "num_warps": fm._HOPPER_EP_DECODE_PLAN["gemm1"]["num_warps"],
                "sorted_token_ids_is_none": False,
                "expert_map_is_input": False,
                "skip_invalid_experts": False,
            },
            {
                "stage": "g2",
                "block_n": fm._HOPPER_EP_DECODE_PLAN["gemm2"]["BLOCK_SIZE_N"],
                "num_warps": fm._HOPPER_EP_DECODE_PLAN["gemm2"]["num_warps"],
                "sorted_token_ids_is_none": False,
                "expert_map_is_input": False,
                "skip_invalid_experts": False,
            },
        ),
        "candidate_route_block": (
            {
                "stage": "g1",
                "block_n": candidate_plan.g1_block_n // 2,
                "num_warps": candidate_plan.g1_warps,
                "sorted_token_ids_is_none": False,
                "expert_map_is_input": False,
                "skip_invalid_experts": False,
            },
            {
                "stage": "g2",
                "block_n": candidate_plan.g2_block_n,
                "num_warps": candidate_plan.g2_warps,
                "sorted_token_ids_is_none": False,
                "expert_map_is_input": False,
                "skip_invalid_experts": False,
            },
        ),
    }
    for policy, expected_stages in expected_dispatch_tiles.items():
        expected_alignment = policy != "legacy_direct"
        if (alignment_calls[policy] > 0) != expected_alignment:
            raise AssertionError(
                f"{policy} route-block alignment count was {alignment_calls[policy]}"
            )
        observed_stages = dispatch_audits[policy]
        if len(observed_stages) != len(expected_stages):
            raise AssertionError(
                f"{policy} expected two GEMM dispatches, got {observed_stages}"
            )
        for observed, expected in zip(observed_stages, expected_stages):
            for key, value in expected.items():
                if observed[key] != value:
                    raise AssertionError(
                        f"{policy} dispatch mismatch for {key}: "
                        f"expected {value}, got {observed[key]}"
                    )

    for _ in range(50):
        for policy in POLICIES:
            graphs[policy].replay()
    torch.cuda.synchronize()

    raw_samples = {policy: [] for policy in POLICIES}
    paired_samples = {policy: [] for policy in POLICIES}
    for round_idx in range(rounds):
        order = POLICIES if round_idx % 2 == 0 else tuple(reversed(POLICIES))
        bracket = (*order, *reversed(order))
        round_samples = {policy: [] for policy in POLICIES}
        for policy in bracket:
            sample = elapsed_ms(graphs[policy], replays)
            raw_samples[policy].append(sample)
            round_samples[policy].append(sample)
        for policy in POLICIES:
            paired_samples[policy].append(statistics.mean(round_samples[policy]))

    medians = {
        policy: statistics.median(samples) for policy, samples in paired_samples.items()
    }
    candidate_reductions = [
        100.0 * (1.0 - candidate / current)
        for current, candidate in zip(
            paired_samples["shared_route_block"],
            paired_samples["candidate_route_block"],
        )
    ]
    shared_vs_legacy_reductions = [
        100.0 * (1.0 - shared / legacy)
        for legacy, shared in zip(
            paired_samples["legacy_direct"],
            paired_samples["shared_route_block"],
        )
    ]
    candidate_vs_legacy_reductions = [
        100.0 * (1.0 - candidate / legacy)
        for legacy, candidate in zip(
            paired_samples["legacy_direct"],
            paired_samples["candidate_route_block"],
        )
    ]
    reference = eager_outputs["legacy_direct"]
    candidate = eager_outputs["candidate_route_block"]
    error = (candidate.float() - reference.float()).abs()
    mapped_ids = expert_map[ids]
    result = {
        "scenario": scenario,
        "local_routes": int((mapped_ids >= 0).sum().item()),
        "local_routes_per_expert": sorted(
            [
                int((mapped_ids == local_id).sum().item())
                for local_id in range(local_experts)
                if bool((mapped_ids == local_id).any().item())
            ],
            reverse=True,
        ),
        "median_ms": medians,
        "candidate_vs_shared_route_block_reduction_pct": 100.0
        * (1.0 - medians["candidate_route_block"] / medians["shared_route_block"]),
        "shared_route_block_vs_legacy_direct_reduction_pct": 100.0
        * (1.0 - medians["shared_route_block"] / medians["legacy_direct"]),
        "candidate_vs_legacy_direct_reduction_pct": 100.0
        * (1.0 - medians["candidate_route_block"] / medians["legacy_direct"]),
        "candidate_vs_shared_paired_reduction_median_pct": statistics.median(
            candidate_reductions
        ),
        "shared_vs_legacy_paired_reduction_median_pct": statistics.median(
            shared_vs_legacy_reductions
        ),
        "candidate_vs_legacy_paired_reduction_median_pct": statistics.median(
            candidate_vs_legacy_reductions
        ),
        "absolute_delta_us": 1000.0
        * (medians["shared_route_block"] - medians["candidate_route_block"]),
        "candidate_vs_shared_positive_rounds": sum(
            value > 0 for value in candidate_reductions
        ),
        "candidate_vs_legacy_positive_rounds": sum(
            value > 0 for value in candidate_vs_legacy_reductions
        ),
        "total_rounds": len(candidate_reductions),
        "bitwise": bool(torch.equal(reference, candidate))
        and all(torch.equal(reference, eager_outputs[policy]) for policy in POLICIES),
        "max_abs_error": float(error.max().item()),
        "graph_bitwise": {
            policy: bool(torch.equal(eager_outputs[policy], graph_outputs[policy]))
            for policy in POLICIES
        },
        "alignment_calls_during_capture": alignment_calls,
        "output_aliases_cache2": True,
        "timing_order": "ABCCBA/CBAABC alternating",
    }
    if not summary_only:
        result["dispatch_audit"] = dispatch_audits
        result["raw_samples_ms"] = raw_samples
        result["paired_samples_ms"] = paired_samples
        result["candidate_vs_shared_paired_reductions_pct"] = candidate_reductions
        result["shared_vs_legacy_paired_reductions_pct"] = shared_vs_legacy_reductions
        result["candidate_vs_legacy_paired_reductions_pct"] = (
            candidate_vs_legacy_reductions
        )
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.ep_rank < 16:
        raise ValueError("ep-rank must be in [0, 16)")
    if args.rounds <= 0 or args.replays <= 0:
        raise ValueError("rounds and replays must be positive")

    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)
    m, global_e, local_e, hidden_size, intermediate_size, topk = (
        1,
        288,
        18,
        4096,
        2048,
        8,
    )
    shard_begin = args.ep_rank * local_e
    dtype = torch.bfloat16
    hidden = torch.randn((m, hidden_size), device="cuda", dtype=dtype)
    w1 = torch.empty(
        (local_e, 2 * intermediate_size, hidden_size), device="cuda", dtype=dtype
    )
    w1.normal_(std=hidden_size**-0.5)
    w2 = torch.empty(
        (local_e, hidden_size, intermediate_size), device="cuda", dtype=dtype
    )
    w2.normal_(std=intermediate_size**-0.5)
    weights = torch.rand((m, topk), device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(dtype)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device="cuda", dtype=torch.int32
    )
    candidate_plan = CandidatePlan(
        args.g1_block_n,
        args.g1_warps,
        args.g1_stages,
        args.g2_block_n,
        args.g2_warps,
        args.g2_stages,
    )
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    results = [
        benchmark_scenario(
            fm=fm,
            scenario=scenario,
            hidden=hidden,
            w1=w1,
            w2=w2,
            weights=weights,
            expert_map=expert_map,
            global_experts=global_e,
            local_experts=local_e,
            shard_begin=shard_begin,
            candidate_plan=candidate_plan,
            rounds=args.rounds,
            replays=args.replays,
            summary_only=args.summary_only,
        )
        for scenario in scenarios
    ]
    experimental_plan = fm._HOPPER_EP_M1_I2048_PLAN
    experimental_candidate = CandidatePlan(
        experimental_plan["gemm1"]["BLOCK_SIZE_N"],
        experimental_plan["gemm1"]["num_warps"],
        experimental_plan["gemm1"]["num_stages"],
        experimental_plan["gemm2"]["BLOCK_SIZE_N"],
        experimental_plan["gemm2"]["num_warps"],
        experimental_plan["gemm2"]["num_stages"],
    )
    singleton_results = [
        result for result in results if result["scenario"].startswith("singleton-p")
    ]
    valid_results = [
        result for result in results if result["scenario"] != "repeated-local"
    ]
    aggregate = {
        "measured_valid_scenarios": len(valid_results),
        "all_policy_bitwise": all(result["bitwise"] for result in results),
    }
    if valid_results:
        aggregate.update(
            {
                "candidate_vs_legacy_positive_medians": sum(
                    result["candidate_vs_legacy_direct_reduction_pct"] > 0
                    for result in valid_results
                ),
                "candidate_vs_legacy_min_reduction_pct": min(
                    result["candidate_vs_legacy_direct_reduction_pct"]
                    for result in valid_results
                ),
            }
        )
    if len(singleton_results) == topk:
        aggregate["singleton_position_uniform_reduction_pct"] = {
            "shared_route_block_vs_legacy_direct": 100.0
            * (
                1.0
                - statistics.mean(
                    result["median_ms"]["shared_route_block"]
                    for result in singleton_results
                )
                / statistics.mean(
                    result["median_ms"]["legacy_direct"] for result in singleton_results
                )
            ),
            "candidate_vs_shared_route_block": 100.0
            * (
                1.0
                - statistics.mean(
                    result["median_ms"]["candidate_route_block"]
                    for result in singleton_results
                )
                / statistics.mean(
                    result["median_ms"]["shared_route_block"]
                    for result in singleton_results
                )
            ),
            "candidate_vs_legacy_direct": 100.0
            * (
                1.0
                - statistics.mean(
                    result["median_ms"]["candidate_route_block"]
                    for result in singleton_results
                )
                / statistics.mean(
                    result["median_ms"]["legacy_direct"] for result in singleton_results
                )
            ),
        }
    output = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "M": m,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "topk": topk,
            "global_experts": global_e,
            "local_experts": local_e,
        },
        "candidate_plan_before_fused_g1_halving": candidate_plan.__dict__,
        "candidate_matches_experimental_plan": candidate_plan == experimental_candidate,
        "legacy_reference": "shared M<=128 fused-MoE plan through raw direct routing",
        "tile_reference": "shared M<=128 fused-MoE plan through route-block alignment",
        "candidate": "experimental M1 narrow plan through route-block alignment",
        "aggregate": aggregate,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
