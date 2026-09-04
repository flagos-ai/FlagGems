#!/usr/bin/env python3
"""Compare compact EP alignment with a direct-route fused-MoE prototype.

The compact policy forces the production alignment path.  The candidate policy
forces the kernel-side global-to-local mapping and remote-expert early return,
including for M > 1.  Production dispatch is not changed by this benchmark.

Timing alternates ABBA and BAAB brackets.  Each reported round sample is the
mean of the two measurements for that policy, reducing sensitivity to gradual
GPU clock drift.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics

import torch

ROUTE_KINDS = (
    "uniform",
    "no-local",
    "local1",
    "local2",
    "local4",
    "all-local",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument(
        "--intermediate-size", type=int, choices=(1280, 2048), default=2048
    )
    parser.add_argument("--route-kind", choices=ROUTE_KINDS, default="uniform")
    parser.add_argument(
        "--local-routes",
        type=int,
        default=-1,
        help="Override route-kind with an exact number of local routes in the batch.",
    )
    parser.add_argument(
        "--local-position",
        type=int,
        default=0,
        help="First local top-k position for local1/local2/local4 patterns.",
    )
    parser.add_argument(
        "--flat-local-offset",
        type=int,
        default=0,
        help="First flattened route used by --local-routes.",
    )
    parser.add_argument("--ep-rank", type=int, default=7)
    parser.add_argument(
        "--direct-group-size-m",
        type=int,
        default=1,
        help="Override GROUP_SIZE_M only for the forced direct-route policy.",
    )
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument(
        "--alias-output-cache2",
        action="store_true",
        help="Exercise the modular caller layout where output aliases cache2.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--audit-singleton-positions",
        action="store_true",
        help="Replay one captured graph for every flattened singleton position.",
    )
    parser.add_argument(
        "--audit-route-masks",
        action="store_true",
        help="A/B all 2^(M*topk) local-route masks; currently requires M=1.",
    )
    return parser.parse_args()


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    eager = fn().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = fn()
    return graph, eager, graph_result


def make_routes(
    *,
    m: int,
    topk: int,
    global_e: int,
    local_e: int,
    shard_begin: int,
    route_kind: str,
    local_position: int,
    local_routes: int,
    flat_local_offset: int,
) -> torch.Tensor:
    if not 0 <= local_position < topk:
        raise ValueError(f"local-position must be in [0, {topk}), got {local_position}")

    if route_kind == "uniform" and local_routes < 0:
        logits = torch.randn((m, global_e), device="cuda")
        return torch.topk(torch.sigmoid(logits), topk, dim=-1).indices.to(torch.int32)

    flat = torch.arange(m * topk, device="cuda", dtype=torch.int32)
    remote_index = flat.remainder(global_e - local_e)
    ids = (remote_index + (remote_index >= shard_begin).to(torch.int32) * local_e).view(
        m, topk
    )
    if local_routes >= 0:
        if local_routes > m * topk:
            raise ValueError("local-routes cannot exceed M * topk")
        route_indices = (
            flat_local_offset + torch.arange(local_routes, device="cuda")
        ).remainder(m * topk)
        ids.view(-1)[route_indices] = shard_begin + torch.arange(
            local_routes, device="cuda", dtype=torch.int32
        ).remainder(local_e)
        return ids

    local_per_token = {
        "no-local": 0,
        "local1": 1,
        "local2": 2,
        "local4": 4,
        "all-local": topk,
    }[route_kind]
    for token_idx in range(m):
        for local_idx in range(local_per_token):
            route_idx = (local_position + local_idx) % topk
            ids[token_idx, route_idx] = (
                shard_begin + (token_idx * local_per_token + local_idx) % local_e
            )
    return ids


def elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / replays)


def main():
    args = parse_args()
    if args.m <= 0:
        raise ValueError("m must be positive")
    if args.rounds <= 0 or args.replays <= 0:
        raise ValueError("rounds and replays must be positive")
    if not 0 <= args.ep_rank < 16:
        raise ValueError("ep-rank must be in [0, 16)")
    if args.direct_group_size_m <= 0:
        raise ValueError("direct-group-size-m must be positive")

    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)

    m, global_e, local_e, h, intermediate, topk = (
        args.m,
        288,
        18,
        4096,
        args.intermediate_size,
        8,
    )
    dtype = torch.bfloat16
    shard_begin = args.ep_rank * local_e
    hidden = torch.randn((m, h), device="cuda", dtype=dtype)
    w1 = torch.empty((local_e, 2 * intermediate, h), device="cuda", dtype=dtype)
    w1.normal_(std=h**-0.5)
    w2 = torch.empty((local_e, h, intermediate), device="cuda", dtype=dtype)
    w2.normal_(std=intermediate**-0.5)
    ids = make_routes(
        m=m,
        topk=topk,
        global_e=global_e,
        local_e=local_e,
        shard_begin=shard_begin,
        route_kind=args.route_kind,
        local_position=args.local_position,
        local_routes=args.local_routes,
        flat_local_offset=args.flat_local_offset,
    )
    weights = torch.rand((m, topk), device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype)

    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device="cuda", dtype=torch.int32
    )
    mapped_ids = expert_map[ids].view(-1)

    original_naive_gate = fm._should_use_ep_naive_route
    original_route_block_gate = fm._should_use_ep_route_block
    original_local_rank_gate = fm._should_use_ep_m1_i2048_local_rank
    original_ep_config = fm._get_ep_decode_config
    original_align = fm.moe_align_block_size
    original_dispatch = fm.dispatch_fused_moe_kernel
    active_policy = ""
    alignment_calls = {"compact": 0, "direct_route_candidate": 0}
    dispatch_audit = {"compact": [], "direct_route_candidate": []}

    def force_compact(*_args, **_kwargs):
        return False

    def force_direct_route(*_args, **_kwargs):
        return True

    def ep_config_spy(*config_args, **config_kwargs):
        config = original_ep_config(*config_args, **config_kwargs)
        if config is not None and active_policy == "direct_route_candidate":
            config["GROUP_SIZE_M"] = args.direct_group_size_m
        return config

    def align_spy(*align_args, **align_kwargs):
        alignment_calls[active_policy] += 1
        return original_align(*align_args, **align_kwargs)

    def dispatch_spy(*dispatch_args, **dispatch_kwargs):
        dispatch_audit[active_policy].append(
            {
                "sorted_token_ids_is_none": dispatch_args[7] is None,
                "expert_map_is_input": dispatch_kwargs.get("expert_map") is expert_map,
                "skip_invalid_experts": bool(
                    dispatch_kwargs.get("skip_invalid_experts", False)
                ),
            }
        )
        return original_dispatch(*dispatch_args, **dispatch_kwargs)

    policies = {
        "compact": force_compact,
        "direct_route_candidate": force_direct_route,
    }
    graphs = {}
    eager_outputs = {}
    graph_outputs = {}
    keepalive = []
    fm._get_ep_decode_config = ep_config_spy
    fm.moe_align_block_size = align_spy
    fm.dispatch_fused_moe_kernel = dispatch_spy
    # This benchmark isolates raw-route GEMM indexing from compact alignment.
    # Disable newer production fast paths so neither can pre-empt the raw
    # direct-versus-compact A/B at M=1/I=2048.
    fm._should_use_ep_route_block = force_compact
    fm._should_use_ep_m1_i2048_local_rank = force_compact
    try:
        for name, route_gate in policies.items():
            active_policy = name
            cache13 = torch.empty(
                m * topk * max(2 * intermediate, h), device="cuda", dtype=dtype
            )
            cache2 = torch.empty(m * topk * intermediate, device="cuda", dtype=dtype)
            if args.alias_output_cache2:
                output = cache2[: m * h].view(m, h)
            else:
                output = torch.empty_like(hidden)
            keepalive.extend((cache13, cache2, output))

            def op(_cache13=cache13, _cache2=cache2, _output=output):
                return fm.fused_experts_impl(
                    hidden,
                    w1,
                    w2,
                    weights,
                    ids,
                    global_num_experts=global_e,
                    expert_map=expert_map,
                    gemm1_clamp_limit=10.0,
                    output=_output,
                    intermediate_cache13=_cache13,
                    intermediate_cache2=_cache2,
                )

            fm._should_use_ep_naive_route = route_gate
            graph, eager, graph_output = capture(op)
            graphs[name] = graph
            eager_outputs[name] = eager
            graph_outputs[name] = graph_output
    finally:
        fm._should_use_ep_naive_route = original_naive_gate
        fm._should_use_ep_route_block = original_route_block_gate
        fm._should_use_ep_m1_i2048_local_rank = original_local_rank_gate
        fm._get_ep_decode_config = original_ep_config
        fm.moe_align_block_size = original_align
        fm.dispatch_fused_moe_kernel = original_dispatch

    for _ in range(50):
        graphs["compact"].replay()
        graphs["direct_route_candidate"].replay()
    torch.cuda.synchronize()

    if args.audit_route_masks:
        if m != 1:
            raise ValueError("--audit-route-masks currently requires --m 1")
        remote_ids = make_routes(
            m=m,
            topk=topk,
            global_e=global_e,
            local_e=local_e,
            shard_begin=shard_begin,
            route_kind="no-local",
            local_position=0,
            local_routes=-1,
            flat_local_offset=0,
        )
        mask_results = []
        policies_order = tuple(policies)
        for route_mask in range(1 << topk):
            ids.copy_(remote_ids)
            local_positions = [
                position for position in range(topk) if route_mask & (1 << position)
            ]
            for local_expert, position in enumerate(local_positions):
                ids[0, position] = shard_begin + local_expert
            for name in policies_order:
                graphs[name].replay()
            torch.cuda.synchronize()
            bitwise = torch.equal(
                graph_outputs["compact"], graph_outputs["direct_route_candidate"]
            )

            samples = {name: [] for name in policies_order}
            for round_idx in range(args.rounds):
                order = (
                    policies_order
                    if round_idx % 2 == 0
                    else tuple(reversed(policies_order))
                )
                bracket = (*order, *reversed(order))
                round_values = {name: [] for name in policies_order}
                for name in bracket:
                    round_values[name].append(elapsed_ms(graphs[name], args.replays))
                for name in policies_order:
                    samples[name].append(statistics.mean(round_values[name]))
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            local_count = len(local_positions)
            count_probability = (
                math.comb(local_e, local_count)
                * math.comb(global_e - local_e, topk - local_count)
                / math.comb(global_e, topk)
            )
            mask_probability = count_probability / math.comb(topk, local_count)
            mask_results.append(
                {
                    "route_mask": route_mask,
                    "local_positions": local_positions,
                    "local_count": local_count,
                    "probability": mask_probability,
                    "median_ms": medians,
                    "direct_reduction_pct": 100.0
                    * (1.0 - medians["direct_route_candidate"] / medians["compact"]),
                    "bitwise": bool(bitwise),
                }
            )

        expected_compact_ms = sum(
            result["probability"] * result["median_ms"]["compact"]
            for result in mask_results
        )
        expected_direct_ms = sum(
            result["probability"] * result["median_ms"]["direct_route_candidate"]
            for result in mask_results
        )
        by_count = {}
        for local_count in range(topk + 1):
            count_results = [
                result
                for result in mask_results
                if result["local_count"] == local_count
            ]
            mean_compact_ms = statistics.mean(
                result["median_ms"]["compact"] for result in count_results
            )
            mean_direct_ms = statistics.mean(
                result["median_ms"]["direct_route_candidate"]
                for result in count_results
            )
            by_count[str(local_count)] = {
                "masks": len(count_results),
                "probability": sum(result["probability"] for result in count_results),
                "mean_compact_ms": mean_compact_ms,
                "mean_direct_ms": mean_direct_ms,
                "ratio_of_means_direct_reduction_pct": 100.0
                * (1.0 - mean_direct_ms / mean_compact_ms),
                "worst_mask_reduction_pct": min(
                    result["direct_reduction_pct"] for result in count_results
                ),
                "positive_masks": sum(
                    result["direct_reduction_pct"] > 0 for result in count_results
                ),
            }
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "M": m,
                    "intermediate_size": intermediate,
                    "ep_rank": args.ep_rank,
                    "audit": "all_route_masks",
                    "timing_order": "ABBA/BAAB alternating",
                    "rounds": args.rounds,
                    "replays": args.replays,
                    "uniform_unique_expert_prior": {
                        "expected_compact_ms": expected_compact_ms,
                        "expected_direct_ms": expected_direct_ms,
                        "expected_direct_reduction_pct": 100.0
                        * (1.0 - expected_direct_ms / expected_compact_ms),
                    },
                    "positive_masks": sum(
                        result["direct_reduction_pct"] > 0 for result in mask_results
                    ),
                    "total_masks": len(mask_results),
                    "worst_mask": min(
                        mask_results, key=lambda result: result["direct_reduction_pct"]
                    ),
                    "by_local_count": by_count,
                    "all_bitwise": all(result["bitwise"] for result in mask_results),
                    "results": [] if args.summary_only else mask_results,
                },
                indent=2,
            )
        )
        return

    if args.audit_singleton_positions:
        remote_ids = make_routes(
            m=m,
            topk=topk,
            global_e=global_e,
            local_e=local_e,
            shard_begin=shard_begin,
            route_kind="no-local",
            local_position=0,
            local_routes=-1,
            flat_local_offset=0,
        )
        position_results = []
        for position in range(m * topk):
            ids.copy_(remote_ids)
            # Isolate route position only: keep the selected local expert and
            # therefore its weight address constant across every replay.
            ids.view(-1)[position] = shard_begin
            for name in policies:
                graphs[name].replay()
            torch.cuda.synchronize()
            bitwise = torch.equal(
                graph_outputs["compact"], graph_outputs["direct_route_candidate"]
            )

            position_samples = {name: [] for name in policies}
            for round_idx in range(args.rounds):
                order = (
                    ("compact", "direct_route_candidate")
                    if round_idx % 2 == 0
                    else ("direct_route_candidate", "compact")
                )
                bracket = (*order, *reversed(order))
                round_samples = {name: [] for name in policies}
                for name in bracket:
                    round_samples[name].append(elapsed_ms(graphs[name], args.replays))
                for name in policies:
                    position_samples[name].append(statistics.mean(round_samples[name]))
            position_medians = {
                name: statistics.median(values)
                for name, values in position_samples.items()
            }
            reductions = [
                100 * (1 - candidate / compact)
                for compact, candidate in zip(
                    position_samples["compact"],
                    position_samples["direct_route_candidate"],
                )
            ]
            position_results.append(
                {
                    "flat_position": position,
                    "token": position // topk,
                    "topk_position": position % topk,
                    "median_ms": position_medians,
                    "reduction_pct": 100
                    * (
                        1
                        - position_medians["direct_route_candidate"]
                        / position_medians["compact"]
                    ),
                    "paired_reduction_median_pct": statistics.median(reductions),
                    "bitwise": bool(bitwise),
                }
            )
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "M": m,
                    "intermediate_size": intermediate,
                    "ep_rank": args.ep_rank,
                    "direct_group_size_m": args.direct_group_size_m,
                    "audit": "singleton_flat_positions",
                    "singleton_global_expert": shard_begin,
                    "results": position_results,
                    "all_bitwise": all(item["bitwise"] for item in position_results),
                    "position_uniform_mean_reduction_pct": statistics.mean(
                        item["reduction_pct"] for item in position_results
                    ),
                },
                indent=2,
            )
        )
        return

    raw_samples = {name: [] for name in policies}
    paired_samples = {name: [] for name in policies}
    for round_idx in range(args.rounds):
        if round_idx % 2 == 0:
            order = ("compact", "direct_route_candidate")
        else:
            order = ("direct_route_candidate", "compact")
        bracket = (*order, *reversed(order))
        round_samples = {name: [] for name in policies}
        for name in bracket:
            sample = elapsed_ms(graphs[name], args.replays)
            raw_samples[name].append(sample)
            round_samples[name].append(sample)
        for name in policies:
            paired_samples[name].append(statistics.mean(round_samples[name]))

    reference = eager_outputs["compact"]
    medians = {
        name: statistics.median(values) for name, values in paired_samples.items()
    }
    paired_reductions = [
        100 * (1 - candidate / compact)
        for compact, candidate in zip(
            paired_samples["compact"], paired_samples["direct_route_candidate"]
        )
    ]
    unique_dispatch = {
        name: [
            dict(values) for values in {tuple(sorted(item.items())) for item in audits}
        ]
        for name, audits in dispatch_audit.items()
    }
    result = {
        "device": torch.cuda.get_device_name(),
        "M": m,
        "intermediate_size": intermediate,
        "route_kind": args.route_kind,
        "requested_local_routes": args.local_routes,
        "local_position": args.local_position,
        "flat_local_offset": args.flat_local_offset,
        "ep_rank": args.ep_rank,
        "direct_group_size_m": args.direct_group_size_m,
        "local_routes": int((mapped_ids >= 0).sum().item()),
        "local_routes_per_token": [
            int(value) for value in (expert_map[ids] >= 0).sum(dim=1).cpu().tolist()
        ],
        "alias_output_cache2": args.alias_output_cache2,
        "timing_order": "ABBA/BAAB alternating",
        "median_ms": medians,
        "candidate_reduction_pct": 100
        * (1 - medians["direct_route_candidate"] / medians["compact"]),
        "paired_reduction_median_pct": statistics.median(paired_reductions),
        "absolute_delta_us": 1000
        * (medians["compact"] - medians["direct_route_candidate"]),
        "positive_rounds": sum(value > 0 for value in paired_reductions),
        "total_rounds": len(paired_reductions),
        "bitwise": bool(
            torch.equal(reference, eager_outputs["direct_route_candidate"])
        ),
        "graph_bitwise": {
            name: bool(torch.equal(eager_outputs[name], graph_outputs[name]))
            for name in policies
        },
        "alignment_calls_during_capture": alignment_calls,
        "unique_dispatch_modes": unique_dispatch,
    }
    if not args.summary_only:
        result["raw_samples_ms"] = raw_samples
        result["paired_samples_ms"] = paired_samples
        result["paired_reductions_pct"] = paired_reductions
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
