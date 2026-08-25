#!/usr/bin/env python3
"""Prototype one-kernel alignment variants for fused-MoE EP decode.

Unlike the production two-kernel compact alignment, this prototype emits one
BM16 block per local route in a single CTA.  It preserves the original flattened
route index, so the regular fused GEMM kernels and deterministic EP combine can
be reused unchanged.  A grouped variant retains production's expert packing
while fusing count/prefix/scatter into one CTA.  This file does not alter
production dispatch.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics

import torch
import triton
import triton.language as tl

from compare_ep_naive import ROUTE_KINDS, capture, elapsed_ms, make_routes


@triton.jit
def _ep_route_block_align_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_mask, other=-1
    )
    valid_global = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global, mask=valid_global, other=-1
    )
    local_mask = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )

    ranks = tl.cumsum(local_mask.to(tl.int32), axis=0) - 1
    num_local_routes = tl.sum(local_mask.to(tl.int32), axis=0)
    tl.store(num_tokens_post_pad_ptr, num_local_routes * BLOCK_SIZE_M)
    tl.store(
        expert_ids_ptr + ranks,
        local_experts_raw.to(tl.int32),
        mask=local_mask,
    )

    lanes = tl.arange(0, BLOCK_SIZE_M)
    output_offsets = ranks[:, None] * BLOCK_SIZE_M + lanes[None, :]
    output_values = tl.where(
        lanes[None, :] == 0,
        route_offsets[:, None],
        NUM_ROUTES,
    )
    tl.store(
        sorted_token_ids_ptr + output_offsets,
        output_values,
        mask=local_mask[:, None],
    )


def ep_route_block_align(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    local_num_experts: int | None = None,
    num_warps: int = 4,
):
    del pad_sorted_ids, ignore_invalid_experts
    if expert_map is None or local_num_experts is None:
        raise ValueError("route-block prototype requires expert_map and local count")
    if block_size != 16:
        raise ValueError("route-block prototype is specialized for BM16")
    num_routes = topk_ids.numel()
    if not 0 < num_routes <= 64:
        raise ValueError("route-block prototype supports 1..64 routes")

    sorted_token_ids = torch.empty(
        (num_routes * block_size,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty((num_routes,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    _ep_route_block_align_kernel[(1,)](
        topk_ids,
        expert_map,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=num_experts,
        NUM_LOCAL_EXPERTS=local_num_experts,
        BLOCK_SIZE_M=block_size,
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        num_warps=num_warps,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


@triton.jit
def _ep_single_kernel_grouped_align_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    INIT_BLOCK: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
):
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_mask, other=-1
    )
    valid_global = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global, mask=valid_global, other=-1
    )
    local_mask = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local = tl.where(local_mask, local_experts_raw, 0).to(tl.int32)

    counts = tl.histogram(safe_local, BLOCK_EXPERT, mask=local_mask).to(tl.int32)
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < NUM_LOCAL_EXPERTS
    counts = tl.where(expert_mask, counts, 0)
    aligned_counts = tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
    starts = tl.cumsum(aligned_counts, axis=0) - aligned_counts
    total_tokens = tl.sum(aligned_counts, axis=0)
    tl.store(num_tokens_post_pad_ptr, total_tokens)

    init_offsets = tl.arange(0, INIT_BLOCK)
    tl.store(
        sorted_token_ids_ptr + init_offsets,
        NUM_ROUTES,
        mask=init_offsets < total_tokens,
    )
    for block_idx in tl.static_range(0, MAX_BLOCKS_PER_EXPERT):
        block_offset = block_idx * BLOCK_SIZE_M
        valid_block = expert_mask & (block_offset < aligned_counts)
        tl.store(
            expert_ids_ptr + starts // BLOCK_SIZE_M + block_idx,
            expert_offsets,
            mask=valid_block,
        )

    # With at most 18 local experts and 64 routes, keeping the scatter inside
    # this CTA is cheaper than launching the production atomic scatter kernel.
    for local_expert in tl.static_range(0, NUM_LOCAL_EXPERTS):
        is_expert = local_mask & (safe_local == local_expert)
        ranks = tl.cumsum(is_expert.to(tl.int32), axis=0) - 1
        start = tl.sum(tl.where(expert_offsets == local_expert, starts, 0), axis=0)
        tl.store(
            sorted_token_ids_ptr + start + ranks,
            route_offsets,
            mask=is_expert,
        )


def ep_single_kernel_grouped_align(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    local_num_experts: int | None = None,
    num_warps: int = 4,
):
    del pad_sorted_ids, ignore_invalid_experts
    if expert_map is None or local_num_experts is None:
        raise ValueError("grouped prototype requires expert_map and local count")
    if block_size != 16:
        raise ValueError("grouped prototype is specialized for BM16")
    num_routes = topk_ids.numel()
    if not 0 < num_routes <= 64:
        raise ValueError("grouped prototype supports 1..64 routes")

    max_num_tokens_padded = num_routes + local_num_experts * (block_size - 1)
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    _ep_single_kernel_grouped_align_kernel[(1,)](
        topk_ids,
        expert_map,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=num_experts,
        NUM_LOCAL_EXPERTS=local_num_experts,
        BLOCK_SIZE_M=block_size,
        BLOCK_EXPERT=triton.next_power_of_2(local_num_experts),
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        INIT_BLOCK=triton.next_power_of_2(max_num_tokens_padded),
        MAX_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, block_size),
        num_warps=num_warps,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument(
        "--intermediate-size", type=int, choices=(1280, 2048), default=2048
    )
    parser.add_argument("--route-kind", choices=ROUTE_KINDS, default="uniform")
    parser.add_argument("--local-routes", type=int, default=-1)
    parser.add_argument("--local-position", type=int, default=0)
    parser.add_argument("--flat-local-offset", type=int, default=0)
    parser.add_argument("--ep-rank", type=int, default=7)
    parser.add_argument(
        "--reference-policy",
        choices=("compact", "production-direct"),
        default="compact",
    )
    parser.add_argument(
        "--candidate-layout", choices=("route-block", "grouped"), default="grouped"
    )
    parser.add_argument("--align-warps", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument("--alias-output-cache2", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--audit-singleton-positions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)

    m, global_e, local_e, hidden_size, intermediate, topk = (
        args.m,
        288,
        18,
        4096,
        args.intermediate_size,
        8,
    )
    shard_begin = args.ep_rank * local_e
    dtype = torch.bfloat16
    hidden = torch.randn((m, hidden_size), device="cuda", dtype=dtype)
    w1 = torch.empty(
        (local_e, 2 * intermediate, hidden_size), device="cuda", dtype=dtype
    )
    w1.normal_(std=hidden_size**-0.5)
    w2 = torch.empty((local_e, hidden_size, intermediate), device="cuda", dtype=dtype)
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
    local_mask = expert_map[ids] >= 0

    original_gate = fm._should_use_ep_naive_route
    original_route_block_gate = fm._should_use_ep_route_block
    original_local_rank_gate = fm._should_use_ep_m1_i2048_local_rank
    original_align = fm.moe_align_block_size
    active_policy = ""
    align_calls = {"compact": 0, "alignment_candidate": 0}

    def force_compact(*_args, **_kwargs):
        return False

    def force_direct(*_args, **_kwargs):
        return True

    def align_dispatch(*align_args, **align_kwargs):
        align_calls[active_policy] += 1
        if active_policy == "alignment_candidate":
            candidate = (
                ep_route_block_align
                if args.candidate_layout == "route-block"
                else ep_single_kernel_grouped_align
            )
            return candidate(
                *align_args,
                **align_kwargs,
                num_warps=args.align_warps,
            )
        return original_align(*align_args, **align_kwargs)

    policies = ("compact", "alignment_candidate")
    graphs = {}
    eager_outputs = {}
    graph_outputs = {}
    keepalive = []
    fm.moe_align_block_size = align_dispatch
    # The candidate is injected through the generic alignment hook below.
    # Keep the integrated route-block path disabled so the A/B remains an
    # independent prototype comparison after production integration.
    fm._should_use_ep_route_block = force_compact
    fm._should_use_ep_m1_i2048_local_rank = force_compact
    try:
        for name in policies:
            active_policy = name
            fm._should_use_ep_naive_route = (
                force_direct
                if name == "compact" and args.reference_policy == "production-direct"
                else force_compact
            )
            cache13 = torch.empty(
                m * topk * max(2 * intermediate, hidden_size),
                device="cuda",
                dtype=dtype,
            )
            cache2 = torch.empty(m * topk * intermediate, device="cuda", dtype=dtype)
            if args.alias_output_cache2:
                output = cache2[: m * hidden_size].view(m, hidden_size)
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

            graph, eager, graph_output = capture(op)
            graphs[name] = graph
            eager_outputs[name] = eager
            graph_outputs[name] = graph_output
    finally:
        fm._should_use_ep_naive_route = original_gate
        fm._should_use_ep_route_block = original_route_block_gate
        fm._should_use_ep_m1_i2048_local_rank = original_local_rank_gate
        fm.moe_align_block_size = original_align

    for _ in range(50):
        for name in policies:
            graphs[name].replay()
    torch.cuda.synchronize()

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
            position_samples = {name: [] for name in policies}
            for round_idx in range(args.rounds):
                order = policies if round_idx % 2 == 0 else tuple(reversed(policies))
                bracket = (*order, *reversed(order))
                round_values = {name: [] for name in policies}
                for name in bracket:
                    round_values[name].append(elapsed_ms(graphs[name], args.replays))
                for name in policies:
                    position_samples[name].append(statistics.mean(round_values[name]))
            medians = {
                name: statistics.median(values)
                for name, values in position_samples.items()
            }
            position_results.append(
                {
                    "flat_position": position,
                    "token": position // topk,
                    "topk_position": position % topk,
                    "reference_ms": medians["compact"],
                    "candidate_ms": medians["alignment_candidate"],
                    "candidate_reduction_pct": 100
                    * (1 - medians["alignment_candidate"] / medians["compact"]),
                    "bitwise": bool(
                        torch.equal(
                            graph_outputs["compact"],
                            graph_outputs["alignment_candidate"],
                        )
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "M": m,
                    "intermediate_size": intermediate,
                    "ep_rank": args.ep_rank,
                    "reference_policy": args.reference_policy,
                    "candidate_layout": args.candidate_layout,
                    "audit": "singleton_flat_positions",
                    "singleton_global_expert": shard_begin,
                    "results": position_results,
                    "all_bitwise": all(item["bitwise"] for item in position_results),
                    "mean_reference_ms": statistics.mean(
                        item["reference_ms"] for item in position_results
                    ),
                    "mean_candidate_ms": statistics.mean(
                        item["candidate_ms"] for item in position_results
                    ),
                    "position_uniform_reduction_pct": 100
                    * (
                        1
                        - statistics.mean(
                            item["candidate_ms"] for item in position_results
                        )
                        / statistics.mean(
                            item["reference_ms"] for item in position_results
                        )
                    ),
                },
                indent=2,
            )
        )
        return

    raw_samples = {name: [] for name in policies}
    paired_samples = {name: [] for name in policies}
    for round_idx in range(args.rounds):
        order = policies if round_idx % 2 == 0 else tuple(reversed(policies))
        bracket = (*order, *reversed(order))
        round_samples = {name: [] for name in policies}
        for name in bracket:
            sample = elapsed_ms(graphs[name], args.replays)
            raw_samples[name].append(sample)
            round_samples[name].append(sample)
        for name in policies:
            paired_samples[name].append(statistics.mean(round_samples[name]))

    medians = {
        name: statistics.median(values) for name, values in paired_samples.items()
    }
    paired_reductions = [
        100 * (1 - candidate / compact)
        for compact, candidate in zip(
            paired_samples["compact"], paired_samples["alignment_candidate"]
        )
    ]
    result = {
        "device": torch.cuda.get_device_name(),
        "M": m,
        "intermediate_size": intermediate,
        "route_kind": args.route_kind,
        "requested_local_routes": args.local_routes,
        "flat_local_offset": args.flat_local_offset,
        "ep_rank": args.ep_rank,
        "reference_policy": args.reference_policy,
        "candidate_layout": args.candidate_layout,
        "align_warps": args.align_warps,
        "local_routes": int(local_mask.sum().item()),
        "local_routes_per_token": [
            int(value) for value in local_mask.sum(dim=1).cpu().tolist()
        ],
        "alias_output_cache2": args.alias_output_cache2,
        "median_ms": medians,
        "candidate_reduction_pct": 100
        * (1 - medians["alignment_candidate"] / medians["compact"]),
        "paired_reduction_median_pct": statistics.median(paired_reductions),
        "absolute_delta_us": 1000
        * (medians["compact"] - medians["alignment_candidate"]),
        "positive_rounds": sum(value > 0 for value in paired_reductions),
        "total_rounds": len(paired_reductions),
        "bitwise": bool(
            torch.equal(eager_outputs["compact"], eager_outputs["alignment_candidate"])
        ),
        "graph_bitwise": {
            name: bool(torch.equal(eager_outputs[name], graph_outputs[name]))
            for name in policies
        },
        "alignment_calls_during_capture": align_calls,
    }
    if not args.summary_only:
        result["raw_samples_ms"] = raw_samples
        result["paired_samples_ms"] = paired_samples
        result["paired_reductions_pct"] = paired_reductions
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
