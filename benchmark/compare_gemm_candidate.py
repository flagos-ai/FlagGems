#!/usr/bin/env python3
"""Paired full-operator A/B for one fused-MoE GEMM plan.

Each round runs A-B-B-A (or B-A-A-B) and averages the bracketing timings, so
slow clock drift cannot turn capture/order position into an apparent win.
"""

from __future__ import annotations

import argparse
import importlib
import json
import linecache
import statistics

import torch
import triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=96)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument(
        "--routing",
        choices=("uniform", "local4", "all_local"),
        default="uniform",
    )
    parser.add_argument("--g1-bn", type=int, default=32, help="Actual paired BN")
    parser.add_argument("--g1-bk", type=int, default=128)
    parser.add_argument("--g1-group", type=int, default=1)
    parser.add_argument("--g1-warps", type=int, default=4)
    parser.add_argument("--g1-stages", type=int, default=3)
    parser.add_argument("--g1-num-ctas", type=int, default=1)
    parser.add_argument("--g2-bn", type=int, default=128)
    parser.add_argument("--g2-bk", type=int, default=64)
    parser.add_argument("--g2-group", type=int, default=1)
    parser.add_argument("--g2-warps", type=int, default=4)
    parser.add_argument("--g2-stages", type=int, default=2)
    parser.add_argument("--g2-num-ctas", type=int, default=1)
    parser.add_argument(
        "--warp-specialize",
        choices=("none", "g1", "g2", "both"),
        default="none",
        help="Clone the JIT kernel and warp-specialize selected K loops",
    )
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=1000)
    return parser.parse_args()


def config(*, bn: int, bk: int, group: int, warps: int, stages: int, num_ctas: int = 1):
    result = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": group,
        "num_warps": warps,
        "num_stages": stages,
    }
    if num_ctas != 1:
        result["num_ctas"] = num_ctas
    return result


def make_warp_specialized_kernel(original, selection: str):
    """Clone selected source loops without editing the production module."""
    source_lines = original.src.splitlines(keepends=True)
    loop_indexes = [
        index
        for index, line in enumerate(source_lines)
        if "for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):" in line
    ]
    if len(loop_indexes) != 4:
        raise RuntimeError(
            f"expected four fused-MoE K loops, found {len(loop_indexes)}"
        )
    selected_indexes = set()
    if selection in ("g1", "both"):
        selected_indexes.add(loop_indexes[0])  # paired gate/up branch
    if selection in ("g2", "both"):
        selected_indexes.add(loop_indexes[3])  # plain GEMM branch
    for index in selected_indexes:
        source_lines[index] = source_lines[index].replace(
            "range(0, tl.cdiv(K, BLOCK_SIZE_K))",
            "tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=True)",
        )
    clone_name = f"fused_moe_kernel_ws_{selection}"
    source_lines[0] = source_lines[0].replace("fused_moe_kernel", clone_name, 1)
    source = "".join(source_lines)
    filename = f"<{clone_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {}
    exec(compile(source, filename, "exec"), original.fn.__globals__, namespace)
    return triton.jit(namespace[clone_name])


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
        graph_output = fn()
    return graph, eager, graph_output


def elapsed(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / replays)


def main() -> None:
    args = parse_args()
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    baseline = (
        config(bn=64, bk=128, group=1, warps=4, stages=3),
        config(bn=128, bk=64, group=1, warps=4, stages=2),
    )
    candidate = (
        config(
            bn=2 * args.g1_bn,
            bk=args.g1_bk,
            group=args.g1_group,
            warps=args.g1_warps,
            stages=args.g1_stages,
            num_ctas=args.g1_num_ctas,
        ),
        config(
            bn=args.g2_bn,
            bk=args.g2_bk,
            group=args.g2_group,
            warps=args.g2_warps,
            stages=args.g2_stages,
            num_ctas=args.g2_num_ctas,
        ),
    )

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
        ids = (token_offsets + route_offsets).remainder(local_e)
    elif args.routing == "local4":
        ids = local_e + (token_offsets + route_offsets).remainder(global_e - local_e)
        ids[:, :4] = (token_offsets + route_offsets[None, :4]).remainder(local_e)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device="cuda", dtype=torch.int32)
    cache13 = torch.empty(
        m * topk * max(2 * intermediate, h), device="cuda", dtype=dtype
    )
    cache2 = torch.empty(m * topk * intermediate, device="cuda", dtype=dtype)
    output = torch.empty_like(hidden)

    def op():
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

    original_selector = fm._get_ep_decode_config
    original_kernel = fm.fused_moe_kernel
    candidate_kernel = (
        make_warp_specialized_kernel(original_kernel, args.warp_specialize)
        if args.warp_specialize != "none"
        else original_kernel
    )
    graphs = {}
    eager_outputs = {}
    graph_outputs = {}
    try:
        for name, plan in (("baseline", baseline), ("candidate", candidate)):
            g1, g2 = plan
            fm.fused_moe_kernel = (
                original_kernel if name == "baseline" else candidate_kernel
            )

            def selector(*positional, gemm_stage=None, **_keyword):
                stage = gemm_stage if gemm_stage is not None else positional[-1]
                return (g1 if stage == "gemm1" else g2).copy()

            fm._get_ep_decode_config = selector
            graph, eager, graph_output = capture(op)
            graphs[name] = graph
            eager_outputs[name] = eager
            graph_outputs[name] = graph_output
    finally:
        fm._get_ep_decode_config = original_selector
        fm.fused_moe_kernel = original_kernel

    graph_bitwise = {}
    for name in ("baseline", "candidate"):
        graphs[name].replay()
        torch.cuda.synchronize()
        graph_bitwise[name] = bool(
            torch.equal(graph_outputs[name].clone(), eager_outputs[name])
        )
    for graph in graphs.values():
        for _ in range(30):
            graph.replay()
    torch.cuda.synchronize()

    a_samples = []
    b_samples = []
    paired_reductions = []
    for round_index in range(args.rounds):
        if round_index % 2 == 0:
            a1 = elapsed(graphs["baseline"], args.replays)
            b1 = elapsed(graphs["candidate"], args.replays)
            b2 = elapsed(graphs["candidate"], args.replays)
            a2 = elapsed(graphs["baseline"], args.replays)
        else:
            b1 = elapsed(graphs["candidate"], args.replays)
            a1 = elapsed(graphs["baseline"], args.replays)
            a2 = elapsed(graphs["baseline"], args.replays)
            b2 = elapsed(graphs["candidate"], args.replays)
        a_value = 0.5 * (a1 + a2)
        b_value = 0.5 * (b1 + b2)
        a_samples.append(a_value)
        b_samples.append(b_value)
        paired_reductions.append(100.0 * (1.0 - b_value / a_value))

    a_median = statistics.median(a_samples)
    b_median = statistics.median(b_samples)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": {"M": m, "H": h, "I": intermediate},
                "routing": args.routing,
                "local_routes": int((expert_map[ids] >= 0).sum().item()),
                "baseline": baseline,
                "candidate": candidate,
                "baseline_samples_ms": a_samples,
                "candidate_samples_ms": b_samples,
                "baseline_median_ms": a_median,
                "candidate_median_ms": b_median,
                "median_of_paired_reduction_pct": statistics.median(paired_reductions),
                "reduction_of_medians_pct": 100.0 * (1.0 - b_median / a_median),
                "candidate_bitwise_vs_baseline": bool(
                    torch.equal(eager_outputs["candidate"], eager_outputs["baseline"])
                ),
                "graph_bitwise_vs_eager": graph_bitwise,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
