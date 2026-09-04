#!/usr/bin/env python3
"""Full-operator CUDA Graph search for EP16 fused-MoE tiles.

This is deliberately an out-of-line experiment: it monkeypatches only the
strict fused-MoE config selector while graphs are captured, leaving production
dispatch unchanged.  ``gemm1`` config BN is twice the reported paired BN,
because production halves it before launching the paired gate/up dot.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tile:
    bm: int = 16
    bn: int = 64
    bk: int = 128
    group: int = 1
    warps: int = 4
    stages: int = 3

    def config(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.bm,
            "BLOCK_SIZE_N": self.bn,
            "BLOCK_SIZE_K": self.bk,
            "GROUP_SIZE_M": self.group,
            "num_warps": self.warps,
            "num_stages": self.stages,
        }


BASE_G1 = Tile(bn=64, bk=128, group=1, warps=4, stages=3)
BASE_G2 = Tile(bn=128, bk=64, group=1, warps=4, stages=2)


def paired_g1(actual_bn: int, **kwargs: int) -> Tile:
    """Build selector config; production converts selector BN to actual_bn."""
    return Tile(bn=actual_bn * 2, **kwargs)


def add_candidate(
    candidates: dict[str, tuple[Tile, Tile]], name: str, g1: Tile, g2: Tile
) -> None:
    candidates.setdefault(name, (g1, g2))


def coarse_candidates() -> dict[str, tuple[Tile, Tile]]:
    candidates = {"baseline": (BASE_G1, BASE_G2)}

    # Paired GEMM1: actual accumulator widths are 2 * {16, 32, 64}.
    for bn in (16, 32, 64):
        for bk in (32, 64, 128, 256):
            add_candidate(
                candidates,
                f"g1_bn{bn}_bk{bk}",
                paired_g1(bn, bk=bk),
                BASE_G2,
            )
    for warps in (2, 4, 8):
        for stages in (2, 3, 4, 5):
            add_candidate(
                candidates,
                f"g1_w{warps}_s{stages}",
                paired_g1(32, bk=128, warps=warps, stages=stages),
                BASE_G2,
            )
    for group in (2, 4, 8):
        add_candidate(
            candidates,
            f"g1_group{group}",
            paired_g1(32, bk=128, group=group),
            BASE_G2,
        )

    for bn in (64, 128, 256):
        for bk in (32, 64, 128, 256):
            add_candidate(
                candidates,
                f"g2_bn{bn}_bk{bk}",
                BASE_G1,
                Tile(bn=bn, bk=bk, stages=2),
            )
    for warps in (2, 4, 8):
        for stages in (2, 3, 4, 5):
            add_candidate(
                candidates,
                f"g2_w{warps}_s{stages}",
                BASE_G1,
                Tile(bn=128, bk=64, warps=warps, stages=stages),
            )
    for group in (2, 4, 8):
        add_candidate(
            candidates,
            f"g2_group{group}",
            BASE_G1,
            Tile(bn=128, bk=64, group=group, stages=2),
        )
    return candidates


def dense_candidates() -> dict[str, tuple[Tile, Tile]]:
    candidates = {"baseline": (BASE_G1, BASE_G2)}
    for bn in (16, 32, 64):
        for group in (1, 2, 4, 8):
            add_candidate(
                candidates,
                f"g1_bn{bn}_group{group}",
                paired_g1(bn, bk=128, group=group),
                BASE_G2,
            )
    for bn in (64, 128, 256):
        for group in (1, 2, 4, 8):
            add_candidate(
                candidates,
                f"g2_bn{bn}_group{group}",
                BASE_G1,
                Tile(bn=bn, bk=64, group=group, stages=2),
            )
    for g1_group in (1, 2, 4, 8):
        for g2_group in (1, 2, 4, 8):
            add_candidate(
                candidates,
                f"groups_g1_{g1_group}_g2_{g2_group}",
                paired_g1(32, bk=128, group=g1_group),
                Tile(bn=128, bk=64, group=g2_group, stages=2),
            )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=96)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--suite", choices=("coarse", "dense"), default="coarse")
    parser.add_argument(
        "--routing",
        choices=("uniform", "local4", "all_local"),
        default="uniform",
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--replays", type=int, default=300)
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
        graph_output = fn()
    return graph, eager, graph_output


def main() -> None:
    args = parse_args()
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    candidates = coarse_candidates() if args.suite == "coarse" else dense_candidates()

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

    original_config = fm._get_ep_decode_config
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    eager_outputs: dict[str, torch.Tensor] = {}
    graph_outputs: dict[str, torch.Tensor] = {}
    errors: dict[str, str] = {}
    try:
        for name, (g1, g2) in candidates.items():

            def selector(*positional, gemm_stage=None, **_keyword):
                stage = gemm_stage if gemm_stage is not None else positional[-1]
                return (g1 if stage == "gemm1" else g2).config()

            fm._get_ep_decode_config = selector
            try:
                graph, eager, graph_output = capture(op)
                graphs[name] = graph
                eager_outputs[name] = eager
                graph_outputs[name] = graph_output
            except Exception as error:  # retain invalid-resource candidates as evidence
                errors[name] = f"{type(error).__name__}: {error}"
                torch.cuda.synchronize()
    finally:
        fm._get_ep_decode_config = original_config

    if "baseline" not in graphs:
        raise RuntimeError(f"baseline failed: {errors.get('baseline')}")

    reference = eager_outputs["baseline"]
    correctness: dict[str, dict[str, object]] = {}
    valid_names = list(graphs)
    for name in valid_names:
        graph = graphs[name]
        graph.replay()
        torch.cuda.synchronize()
        graph_actual = graph_outputs[name].clone()
        eager_actual = eager_outputs[name]
        correctness[name] = {
            "bitwise_vs_baseline": bool(torch.equal(eager_actual, reference)),
            "graph_bitwise_vs_eager": bool(torch.equal(graph_actual, eager_actual)),
            "max_abs_vs_baseline": float(
                (eager_actual.float() - reference.float()).abs().max().item()
            ),
        }

    for graph in graphs.values():
        for _ in range(20):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in valid_names}
    for round_index in range(args.rounds):
        offset = round_index % len(valid_names)
        order = valid_names[offset:] + valid_names[:offset]
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
            samples[name].append(float(start.elapsed_time(end) / args.replays))

    baseline = statistics.median(samples["baseline"])
    results = []
    for name in valid_names:
        median = statistics.median(samples[name])
        g1, g2 = candidates[name]
        results.append(
            {
                "name": name,
                "median_ms": median,
                "reduction_vs_baseline_pct": 100.0 * (1.0 - median / baseline),
                "g1_paired_actual_bn": g1.bn // 2,
                "g1": g1.config(),
                "g2": g2.config(),
                **correctness[name],
            }
        )
    results.sort(key=lambda item: item["median_ms"])
    local_routes = int((expert_map[ids] >= 0).sum().item())
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": {"M": m, "H": h, "I": intermediate},
                "routing": args.routing,
                "local_routes": local_routes,
                "baseline_ms": baseline,
                "results": results,
                "errors": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
