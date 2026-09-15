#!/usr/bin/env python3
"""EP fused-MoE microbenchmark.

The production trace uses BF16 activations and expert weights, FP32 top-k router
weights, top-8 routing, 288 global experts and EP16 (18 local experts per rank).
Hidden/expert-intermediate dimensions default to the checkpoint-size-consistent
4096/2048 pair and remain command-line options until the private checkpoint
config is available locally.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics

import torch
import triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--global-experts", type=int, default=288)
    parser.add_argument("--local-experts", type=int, default=18)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--graph-replays", type=int, default=500)
    parser.add_argument(
        "--router-weight-dtype",
        "--topk-weights-dtype",
        dest="router_weight_dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help="Dtype of normalized top-k router weights (default: fp32).",
    )
    return parser.parse_args()


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            output = fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def time_graph(graph, *, rounds: int, replays: int) -> list[float]:
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) / replays))
    return samples


def main() -> None:
    args = parse_args()
    if args.global_experts != args.local_experts * 16:
        raise ValueError(
            "benchmark models EP16, so global_experts must equal 16*local_experts"
        )
    if not 0 <= args.ep_rank < 16:
        raise ValueError("ep_rank must be in [0, 16)")

    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    router_weight_dtype = (
        torch.bfloat16 if args.router_weight_dtype == "bf16" else torch.float32
    )
    device = torch.device("cuda")

    hidden = torch.randn((args.m, args.hidden_size), device=device, dtype=dtype)
    w1 = torch.empty(
        (args.local_experts, 2 * args.intermediate_size, args.hidden_size),
        device=device,
        dtype=dtype,
    ).normal_(std=args.hidden_size**-0.5)
    w2 = torch.empty(
        (args.local_experts, args.hidden_size, args.intermediate_size),
        device=device,
        dtype=dtype,
    ).normal_(std=args.intermediate_size**-0.5)

    # Top-k from iid logits gives unique experts per token and an unbiased EP
    # rank assignment while preserving a realistic, slightly skewed histogram.
    logits = torch.randn((args.m, args.global_experts), device=device)
    topk_weights, topk_ids = torch.topk(torch.sigmoid(logits), args.topk, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(
        router_weight_dtype
    )
    topk_ids = topk_ids.to(torch.int32)

    expert_map = torch.full(
        (args.global_experts,), -1, device=device, dtype=torch.int32
    )
    local_begin = args.ep_rank * args.local_experts
    expert_map[local_begin : local_begin + args.local_experts] = torch.arange(
        args.local_experts, device=device, dtype=torch.int32
    )

    cache13 = torch.empty(
        args.m * args.topk * max(2 * args.intermediate_size, args.hidden_size),
        device=device,
        dtype=dtype,
    )
    cache2 = torch.empty(
        args.m * args.topk * args.intermediate_size,
        device=device,
        dtype=dtype,
    )
    output = torch.empty_like(hidden)

    def op():
        return fm.fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation="silu",
            global_num_experts=args.global_experts,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=output,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
        )

    eager_output = op().clone()
    torch.cuda.synchronize()
    eager_samples = [
        float(
            triton.testing.do_bench(
                op, warmup=args.warmup, rep=args.rep, return_mode="median"
            )
        )
        for _ in range(args.rounds)
    ]
    graph, graph_output = capture(op)
    graph.replay()
    torch.cuda.synchronize()
    graph_samples = time_graph(graph, rounds=args.rounds, replays=args.graph_replays)

    config_dtype = fm._get_config_dtype_str(dtype=dtype)
    config_args = (
        tuple(w1.shape),
        tuple(w2.shape),
        args.topk,
        config_dtype,
    )
    gemm1 = fm.try_get_optimal_moe_config(
        *config_args,
        M=args.m,
        E=args.local_experts,
        gemm_stage="gemm1",
        enable_gemm_fast_path=True,
    )
    gemm2 = fm.try_get_optimal_moe_config(
        *config_args,
        M=args.m,
        E=args.local_experts,
        gemm_stage="gemm2",
        enable_gemm_fast_path=True,
    )
    ep_gemm1 = fm._get_ep_decode_config(
        args.m,
        args.local_experts,
        args.global_experts,
        args.topk,
        args.hidden_size,
        args.intermediate_size,
        10.0,
        config_dtype,
        expert_map,
        "gemm1",
    )
    if ep_gemm1 is not None:
        gemm1 = ep_gemm1
        gemm2 = fm._get_ep_decode_config(
            args.m,
            args.local_experts,
            args.global_experts,
            args.topk,
            args.hidden_size,
            args.intermediate_size,
            10.0,
            config_dtype,
            expert_map,
            "gemm2",
        )
    use_local_rank = topk_weights.dtype in (torch.bfloat16, torch.float32) and (
        fm._should_use_ep_m1_i2048_local_rank(
            ep_gemm1,
            expert_map,
            args.m,
            args.intermediate_size,
            ep_gemm1 is not None,
        )
    )
    use_direct_route = not use_local_rank and fm._should_use_ep_naive_route(
        ep_gemm1,
        expert_map,
        args.m,
        args.intermediate_size,
        ep_gemm1 is not None,
    )
    use_route_block = not use_local_rank and fm._should_use_ep_route_block(
        ep_gemm1,
        expert_map,
        args.m,
        args.intermediate_size,
        ep_gemm1 is not None,
    )
    if use_route_block:
        gemm1 = fm._HOPPER_EP_M1_I2048_PLAN["gemm1"].copy()
        gemm2 = fm._HOPPER_EP_M1_I2048_PLAN["gemm2"].copy()
    mapped_routes = expert_map[topk_ids]
    local_histogram = torch.bincount(
        mapped_routes[mapped_routes >= 0].long(), minlength=args.local_experts
    )
    result = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "M": args.m,
            "global_E": args.global_experts,
            "local_E": args.local_experts,
            "H": args.hidden_size,
            "I": args.intermediate_size,
            "topk": args.topk,
            "dtype": str(dtype),
            "router_weight_dtype": str(topk_weights.dtype),
        },
        "routing": {
            "total_routes": int(topk_ids.numel()),
            "local_routes": int((mapped_routes >= 0).sum().item()),
            "active_local_experts": int((local_histogram > 0).sum().item()),
            "local_histogram": local_histogram.cpu().tolist(),
        },
        "configs": {"gemm1": gemm1, "gemm2": gemm2},
        "routing_policy": {
            "local_route_rank": use_local_rank,
            "direct_global_to_local": use_direct_route,
            "route_block": use_route_block,
        },
        "eager_ms": {
            "samples": eager_samples,
            "median": statistics.median(eager_samples),
        },
        "cuda_graph_ms": {
            "samples": graph_samples,
            "median": statistics.median(graph_samples),
        },
        "correctness_signature": {
            "eager_graph_bitwise_equal": bool(torch.equal(eager_output, graph_output)),
            "max_abs": float(eager_output.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(eager_output.float()).item()),
            "finite": bool(torch.isfinite(eager_output).all().item()),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
