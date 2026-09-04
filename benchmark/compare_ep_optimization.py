#!/usr/bin/env python3
"""Deterministic source-level A/B for fused-MoE EP route pruning and tiles."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics

import torch
import triton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--replays", type=int, default=500)
    ab_group = parser.add_mutually_exclusive_group()
    ab_group.add_argument(
        "--sum-ab-only",
        action="store_true",
        help="Compare optimized BM16 with legacy versus EP-aware combine only",
    )
    ab_group.add_argument(
        "--activation-ab-only",
        action="store_true",
        help=(
            "Compare separate versus fused clamped SwiGLU with all other "
            "fused-MoE optimizations held fixed"
        ),
    )
    ab_group.add_argument(
        "--headline-only",
        action="store_true",
        help="Alternate only legacy and final production graphs",
    )
    parser.add_argument(
        "--routing",
        choices=(
            "uniform",
            "all_local",
            "no_local",
            "skewed",
            "local2",
            "local4",
        ),
        default="uniform",
    )
    return parser.parse_args()


def capture_under_policy(
    fm,
    fn,
    align_fn,
    config_fn,
    ep_sum_fn,
    fused_activation_fn,
    ep_naive_route_fn,
):
    old_align = fm.moe_align_block_size
    old_config = fm._get_ep_decode_config
    old_ep_sum = fm._should_use_ep_sum
    old_fused_activation = fm._should_use_fused_clamped_swiglu
    old_ep_naive_route = fm._should_use_ep_naive_route
    old_dispatch = fm.dispatch_fused_moe_kernel
    dispatch_audit = []

    def audited_dispatch(*args, **kwargs):
        dispatch_audit.append(
            {
                "config": dict(args[12]),
                "fuse_silu": kwargs.get("FUSE_SILU", False),
                "sorted_token_ids_is_none": args[7] is None,
                "maps_experts_in_gemm": kwargs.get("expert_map") is not None,
            }
        )
        return old_dispatch(*args, **kwargs)

    fm.moe_align_block_size = align_fn
    fm._get_ep_decode_config = config_fn
    fm._should_use_ep_sum = ep_sum_fn
    fm._should_use_fused_clamped_swiglu = fused_activation_fn
    fm._should_use_ep_naive_route = ep_naive_route_fn
    fm.dispatch_fused_moe_kernel = audited_dispatch
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                output = fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        eager = fn().clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fn()
    finally:
        fm.moe_align_block_size = old_align
        fm._get_ep_decode_config = old_config
        fm._should_use_ep_sum = old_ep_sum
        fm._should_use_fused_clamped_swiglu = old_fused_activation
        fm._should_use_ep_naive_route = old_ep_naive_route
        fm.dispatch_fused_moe_kernel = old_dispatch
    return graph, eager, graph_output, dispatch_audit


def main():
    args = parse_args()
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    align = importlib.import_module("flag_gems.fused.moe_align_block_size")
    torch.manual_seed(20260824)
    m, global_e, local_e, h, intermediate, topk = (
        args.m,
        288,
        18,
        args.hidden_size,
        args.intermediate_size,
        8,
    )
    dtype = torch.bfloat16
    hidden = torch.randn((m, h), device="cuda", dtype=dtype)
    w1 = torch.empty((local_e, 2 * intermediate, h), device="cuda", dtype=dtype)
    w1.normal_(std=h**-0.5)
    w2 = torch.empty((local_e, h, intermediate), device="cuda", dtype=dtype)
    w2.normal_(std=intermediate**-0.5)
    logits = torch.randn((m, global_e), device="cuda")
    weights, ids = torch.topk(torch.sigmoid(logits), topk, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype)
    ids = ids.to(torch.int32)
    route_offsets = torch.arange(topk, device="cuda", dtype=torch.int32)
    token_offsets = torch.arange(m, device="cuda", dtype=torch.int32)[:, None]
    if args.routing == "all_local":
        # Keep the top-k IDs distinct per token while covering all 18 experts.
        ids = (token_offsets + route_offsets).remainder(local_e)
    elif args.routing == "no_local":
        ids = local_e + (token_offsets + route_offsets).remainder(global_e - local_e)
    elif args.routing == "skewed":
        # One hot local route and seven distinct remote routes per token.
        ids = local_e + (token_offsets + route_offsets).remainder(global_e - local_e)
        ids[:, 0] = 3
    elif args.routing in ("local2", "local4"):
        local_routes_per_token = int(args.routing[-1])
        ids = local_e + (token_offsets + route_offsets).remainder(global_e - local_e)
        ids[:, :local_routes_per_token] = (
            token_offsets + route_offsets[None, :local_routes_per_token]
        ).remainder(local_e)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device="cuda", dtype=torch.int32)

    def legacy_align(
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
        local_num_experts=None,
    ):
        del ignore_invalid_experts, local_num_experts
        max_padded = topk_ids.numel() + num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_padded = align.round_up(max_padded, block_size)
        sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=topk_ids.device)
        expert_ids = torch.empty(
            triton.cdiv(max_padded, block_size),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        total = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
        align.moe_align_block_size_triton(
            topk_ids, num_experts, block_size, sorted_ids, expert_ids, total
        )
        if expert_map is not None:
            expert_ids = expert_map[expert_ids]
        return sorted_ids, expert_ids, total

    current_align = fm.moe_align_block_size
    current_config = fm._get_ep_decode_config
    current_ep_sum = fm._should_use_ep_sum
    current_fused_activation = fm._should_use_fused_clamped_swiglu
    current_ep_naive_route = fm._should_use_ep_naive_route

    def forced_local_align(
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
        local_num_experts=None,
    ):
        del ignore_invalid_experts
        return current_align(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts=True,
            local_num_experts=local_num_experts,
        )

    def no_ep_config(*_args, **_kwargs):
        return None

    def legacy_sum(*_args, **_kwargs):
        return False

    def separate_activation(*_args, **_kwargs):
        return False

    def compact_route(*_args, **_kwargs):
        return False

    optimized_legacy_sum_policy = (
        forced_local_align,
        current_config,
        legacy_sum,
        current_fused_activation,
        compact_route,
    )
    policies = {
        "legacy_global_align_bm64": (
            legacy_align,
            no_ep_config,
            legacy_sum,
            separate_activation,
            compact_route,
        ),
        "local_align_only_bm64": (
            forced_local_align,
            no_ep_config,
            legacy_sum,
            separate_activation,
            compact_route,
        ),
        "ep_bm16_only_global_align": (
            legacy_align,
            current_config,
            legacy_sum,
            separate_activation,
            compact_route,
        ),
        "optimized_local_align_bm16": (
            forced_local_align,
            current_config,
            current_ep_sum,
            current_fused_activation,
            current_ep_naive_route,
        ),
    }
    if args.sum_ab_only:
        policies = {
            "optimized_legacy_sum": optimized_legacy_sum_policy,
            "optimized_local_align_bm16": policies["optimized_local_align_bm16"],
        }
    elif args.activation_ab_only:
        policies = {
            "optimized_separate_activation": (
                forced_local_align,
                current_config,
                current_ep_sum,
                separate_activation,
                current_ep_naive_route,
            ),
            "optimized_fused_activation": policies["optimized_local_align_bm16"],
        }
    elif args.headline_only:
        policies = {
            "legacy_global_align_bm64": policies["legacy_global_align_bm64"],
            "optimized_local_align_bm16": policies["optimized_local_align_bm16"],
        }

    graphs = {}
    outputs = {}
    graph_outputs = {}
    dispatch_audits = {}
    keepalive = []
    for name, (
        align_fn,
        config_fn,
        ep_sum_fn,
        fused_activation_fn,
        ep_naive_route_fn,
    ) in policies.items():
        cache13 = torch.empty(
            m * topk * max(2 * intermediate, h), device="cuda", dtype=dtype
        )
        cache2 = torch.empty(m * topk * intermediate, device="cuda", dtype=dtype)
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

        graph, eager, graph_output, dispatch_audit = capture_under_policy(
            fm,
            op,
            align_fn,
            config_fn,
            ep_sum_fn,
            fused_activation_fn,
            ep_naive_route_fn,
        )
        graphs[name] = graph
        outputs[name] = eager
        graph_outputs[name] = graph_output
        # Warmup/eager/capture repeat the same two-stage dispatch. Retain only
        # the final GEMM1/GEMM2 pair as an executable-policy audit.
        dispatch_audits[name] = dispatch_audit[-2:]

    for graph in graphs.values():
        for _ in range(20):
            graph.replay()
    torch.cuda.synchronize()

    names = list(policies)
    samples = {name: [] for name in names}
    for round_idx in range(args.rounds):
        order = names if round_idx % 2 == 0 else list(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end) / args.replays))

    # Recover the actual device-side padded counts for the two alignment/BM
    # choices without placing a synchronization in the measured graphs.
    _, _, global_total = legacy_align(ids, 64, global_e, expert_map)
    _, _, local_total = current_align(
        ids,
        16,
        global_e,
        expert_map,
        ignore_invalid_experts=True,
        local_num_experts=local_e,
    )
    global_padded_rows = int(global_total.item())
    local_padded_rows = int(local_total.item())

    reference_name = (
        "legacy_global_align_bm64"
        if "legacy_global_align_bm64" in outputs
        else next(iter(outputs))
    )
    reference = outputs[reference_name]
    baseline = statistics.median(samples[reference_name])
    results = {}
    for name in names:
        median = statistics.median(samples[name])
        difference = outputs[name].float() - reference.float()
        results[name] = {
            "samples_ms": samples[name],
            "median_ms": median,
            "speedup": baseline / median,
            "reduction_pct": 100 * (1 - median / baseline),
            "bitwise_equal_to_legacy": bool(torch.equal(outputs[name], reference)),
            "max_abs_to_legacy": float(difference.abs().max().item()),
            "relative_l2_to_legacy": float(
                torch.linalg.vector_norm(difference).item()
                / max(torch.linalg.vector_norm(reference.float()).item(), 1e-30)
            ),
            "eager_graph_bitwise_equal": bool(
                torch.equal(outputs[name], graph_outputs[name])
            ),
        }

    mapped = expert_map[ids]
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
        "routing": {
            "kind": args.routing,
            "total_routes": int(ids.numel()),
            "local_routes": int((mapped >= 0).sum().item()),
            "global_bm64_padded_rows": global_padded_rows,
            "local_bm16_padded_rows": local_padded_rows,
        },
        "results": results,
        "dispatch_audit": dispatch_audits,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
