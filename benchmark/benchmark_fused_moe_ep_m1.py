# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare local-rank MMA and adaptive GEMV through the full fused-MoE API.

Run on one idle H20 with PYTHONPATH=src. Both policies use identical BF16
weights, FP32 router weights, caller-owned workspaces and output/cache2 alias.
The baseline retains the earlier M1 local-rank optimization; it is not the
unoptimized global-alignment implementation. All 256 local-route masks are
tested by default. Hypergeometric weighting describes uniform EP16 routing,
not a measured model routing distribution or an end-to-end throughput result.
"""

import argparse
import importlib
import json
import math
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument(
        "--masks", default="all", help="all or comma-separated integers"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.replays, args.unroll) <= 0:
        parser.error("rounds, replays and unroll must be positive")
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    if not fm._is_h20():
        parser.error("the adaptive policy is enabled and measured on H20 only")
    masks = (
        list(range(256))
        if args.masks == "all"
        else [int(v) for v in args.masks.split(",")]
    )
    if (
        not masks
        or len(set(masks)) != len(masks)
        or any(v < 0 or v > 255 for v in masks)
    ):
        parser.error("masks must be unique integers in [0, 255]")
    torch.manual_seed(args.seed)
    options = {"device": "cuda", "dtype": torch.bfloat16}
    hidden = torch.randn(1, 4096, **options)
    w1 = torch.randn(18, 4096, 4096, **options) * 4096**-0.5
    w2 = torch.randn(18, 4096, 2048, **options) * 2048**-0.5
    weights = torch.rand(1, 8, device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    ids = torch.arange(8, device="cuda", dtype=torch.int64).view(1, 8)
    expert_map = torch.full((288,), -1, device="cuda", dtype=torch.int64)
    expert_map[:18] = torch.arange(18, device="cuda", dtype=torch.int64)
    graphs, workspaces, outputs = {}, {}, {}
    original_launcher = fm.fused_moe_ep_m1_i2048_local_rank

    for name, enabled in (("local_rank_mma", False), ("adaptive", True)):
        cache13 = torch.empty(8 * 4096, **options)
        cache2 = torch.empty(8 * 2048, **options)
        output = cache2[:4096].view(1, 4096)
        # Graphs store addresses, not Python ownership of external tensors.
        workspaces[name] = (cache13, cache2, output)
        outputs[name] = output

        def launcher(*positional, **keyword):
            keyword["use_singleton_gemv"] = enabled
            return original_launcher(*positional, **keyword)

        def run():
            return fm.fused_experts_impl(
                hidden,
                w1,
                w2,
                weights,
                ids,
                global_num_experts=288,
                expert_map=expert_map,
                gemm1_clamp_limit=10.0,
                output=output,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
            )

        fm.fused_moe_ep_m1_i2048_local_rank = launcher
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.unroll):
                    run()
            graphs[name] = graph
        finally:
            fm.fused_moe_ep_m1_i2048_local_rank = original_launcher

    rows = []
    for mask in masks:
        local_rank = 0
        route_ids = []
        for route in range(8):
            if mask & (1 << route):
                route_ids.append(local_rank)
                local_rank += 1
            else:
                route_ids.append(18 + route)
        ids.copy_(torch.tensor([route_ids], device="cuda", dtype=ids.dtype))
        values = {}
        for name, graph in graphs.items():
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            values[name] = outputs[name].clone()
        reference = values["local_rank_mma"].float()
        error = values["adaptive"].float() - reference
        relative_l2 = (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).item()
        assert relative_l2 < 0.005, (mask, relative_l2)
        if local_rank != 1:
            torch.testing.assert_close(
                values["adaptive"], values["local_rank_mma"], rtol=0, atol=0
            )
        samples = {name: [] for name in graphs}
        for round_index in range(args.rounds):
            order = list(graphs)
            if round_index % 2:
                order.reverse()
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(
                    start.elapsed_time(end) * 1000 / args.replays / args.unroll
                )
        medians = {
            name: statistics.median(timings) for name, timings in samples.items()
        }
        rows.append(
            {
                "mask": mask,
                "local_count": local_rank,
                "route_ids": route_ids,
                "median_us": medians,
                "samples_us": samples,
                "speedup": medians["local_rank_mma"] / medians["adaptive"],
                "max_abs": error.abs().max().item(),
                "relative_l2": relative_l2,
            }
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "seed": args.seed,
        "rounds": args.rounds,
        "replays": args.replays,
        "unroll": args.unroll,
        "shape": {
            "M": 1,
            "H": 4096,
            "I": 2048,
            "global_E": 288,
            "local_E": 18,
            "topk": 8,
        },
        "results": rows,
    }
    kernels = importlib.import_module("flag_gems.fused.fused_moe_ep_m1")
    resources = {}
    for name in (
        "_ep_m1_i2048_local_rank_g1",
        "_ep_m1_i2048_local_rank_g2",
        "_ep_m1_i2048_adaptive_g1",
        "_ep_m1_i2048_adaptive_g2",
    ):
        variants = []
        for cache in getattr(kernels, name).device_caches.values():
            for compiled in cache[0].values():
                variants.append(
                    {
                        "registers": compiled.n_regs,
                        "spills": compiled.n_spills,
                        "shared_bytes": compiled.metadata.shared,
                    }
                )
        resources[name] = variants
    result["compiled_resources"] = resources
    if len(masks) == 256:
        weighted = {name: 0.0 for name in graphs}
        for row in rows:
            k = row["local_count"]
            probability = (
                math.comb(18, k)
                * math.comb(270, 8 - k)
                / math.comb(288, 8)
                / math.comb(8, k)
            )
            for name in graphs:
                weighted[name] += probability * row["median_us"][name]
        result["uniform_ep16_weighted_us"] = weighted
        result["uniform_ep16_weighted_speedup"] = (
            weighted["local_rank_mma"] / weighted["adaptive"]
        )
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
