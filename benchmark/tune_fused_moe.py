#!/usr/bin/env python3
"""Same-process CUDA Graph tile sweep for the fused-MoE EP16 decode point."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics

import torch


def cfg(bm, bn, bk, warps=4, stages=3):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 1,
        "num_warps": warps,
        "num_stages": stages,
    }


def persistent(config, grid_size):
    return {**config, "PERSISTENT_GRID_SIZE": grid_size}


CANDIDATES = {
    "heuristic_bm64": (cfg(64, 128, 64), cfg(64, 128, 64)),
    "bm32_bn128_bk64_s3": (cfg(32, 128, 64), cfg(32, 128, 64)),
    "bm32_g1_bn64_bk128_s3_g2_bn128_bk64_s2": (
        cfg(32, 64, 128, stages=3),
        cfg(32, 128, 64, stages=2),
    ),
    "bm32_g1_bn128_bk128_s2_g2_bn128_bk64_s2": (
        cfg(32, 128, 128, stages=2),
        cfg(32, 128, 64, stages=2),
    ),
    "bm32_g1_bn64_bk64_s3_g2_bn256_bk64_s2": (
        cfg(32, 64, 64, stages=3),
        cfg(32, 256, 64, stages=2),
    ),
    "bm16_bn64_bk64_s2": (cfg(16, 64, 64, stages=2), cfg(16, 64, 64, stages=2)),
    "bm16_bn64_bk64_s3": (cfg(16, 64, 64), cfg(16, 64, 64)),
    "bm16_bn128_bk64_s2": (
        cfg(16, 128, 64, stages=2),
        cfg(16, 128, 64, stages=2),
    ),
    "bm16_bn128_bk64_s3": (cfg(16, 128, 64), cfg(16, 128, 64)),
    "bm16_bn256_bk64_s2": (
        cfg(16, 256, 64, stages=2),
        cfg(16, 256, 64, stages=2),
    ),
    "g1_bn64_bk128_s2_g2_bn128_bk64_s2": (
        cfg(16, 64, 128, stages=2),
        cfg(16, 128, 64, stages=2),
    ),
    "g1_bn64_bk128_s3_g2_bn128_bk64_s2": (
        cfg(16, 64, 128, stages=3),
        cfg(16, 128, 64, stages=2),
    ),
    "g1_bn64_bk128_s3_g2_persistent_grid156": (
        cfg(16, 64, 128, stages=3),
        persistent(cfg(16, 128, 64, stages=2), 156),
    ),
    "g1_bn64_bk128_s3_g2_persistent_grid312": (
        cfg(16, 64, 128, stages=3),
        persistent(cfg(16, 128, 64, stages=2), 312),
    ),
    "g1_bn64_bk128_s3_g2_persistent_grid468": (
        cfg(16, 64, 128, stages=3),
        persistent(cfg(16, 128, 64, stages=2), 468),
    ),
    "g1_bn64_bk128_s3_g2_persistent_grid546": (
        cfg(16, 64, 128, stages=3),
        persistent(cfg(16, 128, 64, stages=2), 546),
    ),
    "g1_bn128_bk128_s2_g2_bn128_bk64_s2": (
        cfg(16, 128, 128, stages=2),
        cfg(16, 128, 64, stages=2),
    ),
    "g1_bn128_bk64_s2_g2_bn256_bk64_s2": (
        cfg(16, 128, 64, stages=2),
        cfg(16, 256, 64, stages=2),
    ),
    "g1_bn64_bk64_s3_g2_bn256_bk64_w8_s2": (
        cfg(16, 64, 64, stages=3),
        cfg(16, 256, 64, warps=8, stages=2),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing",
        choices=("uniform", "all_local", "no_local", "skewed"),
        default="uniform",
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


def main():
    args = parse_args()
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)
    m, global_e, local_e, h, intermediate, topk = 96, 288, 18, 4096, 2048, 8
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
    elif args.routing == "no_local":
        ids = local_e + (token_offsets + route_offsets).remainder(global_e - local_e)
    elif args.routing == "skewed":
        ids = local_e + (token_offsets + route_offsets).remainder(global_e - local_e)
        ids[:, 0] = 3
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

    original_ep_config = fm._get_ep_decode_config
    graphs = {}
    graph_outputs = {}
    eager_outputs = {}
    try:
        for name, (g1, g2) in CANDIDATES.items():

            def make_config(stage1, stage2):
                def config(*_args, gemm_stage=None, **_kwargs):
                    if gemm_stage is None:
                        gemm_stage = _args[-1]
                    return (stage1 if gemm_stage == "gemm1" else stage2).copy()

                return config

            fm._get_ep_decode_config = make_config(g1, g2)
            eager_outputs[name] = op().clone()
            torch.cuda.synchronize()
            graph, graph_output = capture(op)
            graphs[name] = graph
            graph_outputs[name] = graph_output
    finally:
        fm._get_ep_decode_config = original_ep_config

    for graph in graphs.values():
        for _ in range(20):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in CANDIDATES}
    names = list(CANDIDATES)
    for round_idx in range(9):
        order = names if round_idx % 2 == 0 else list(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(300):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end) / 300))

    reference = eager_outputs["heuristic_bm64"]
    results = {}
    baseline = statistics.median(samples["heuristic_bm64"])
    for name in names:
        actual = eager_outputs[name]
        graph_actual = graph_outputs[name]
        median = statistics.median(samples[name])
        diff = (actual.float() - reference.float()).abs()
        results[name] = {
            "gemm1": CANDIDATES[name][0],
            "gemm2": CANDIDATES[name][1],
            "samples_ms": samples[name],
            "median_ms": median,
            "speedup": baseline / median,
            "reduction_pct": 100 * (1 - median / baseline),
            "max_abs_vs_bm64": float(diff.max().item()),
            "relative_l2_vs_bm64": float(
                torch.linalg.vector_norm(actual.float() - reference.float()).item()
                / torch.linalg.vector_norm(reference.float()).item()
            ),
            "eager_graph_bitwise_equal": bool(torch.equal(actual, graph_actual)),
        }
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "routing": args.routing,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
