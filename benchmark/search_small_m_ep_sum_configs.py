#!/usr/bin/env python3
"""Sweep deterministic raw ``moe_sum_ep`` configs for fused-MoE small M.

This is an isolated benchmark: it imports the shared Triton JIT body but does
not modify production dispatch.  The default shape set models fused-MoE EP16:
BF16, H=4096, top-k=8, 288 global experts and 18 local experts.

The production autotuner key does not contain M or routing density.  By
default, this script therefore preserves its cache across all cases and primes
it with M=1/uniform, matching a process whose first decode combine has that
shape.  ``--retune-per-case`` clears only the in-process autotuner choice before
each case and reports the optimistic, independently tuned baseline instead.

Every timed launch is a CUDA Graph replay.  Candidate and baseline timings are
collected in alternating ABBA/BAAB brackets.  Correctness gates include an
explicit route-order FP32 reference, eager/current-autotune bitwise equality,
and eager/Graph replay bitwise equality.  The output defaults to a view of a
separate cache2-sized backing allocation, matching the alias-safe modular
fused-MoE caller layout; the combine input remains the non-overlapping cache3.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

import torch
import triton

from flag_gems.fused.moe_sum import (
    _moe_sum_ep_kernel,
    moe_sum_ep,
    moe_sum_ep_kernel,
)

DEFAULT_CONFIGS = (
    (64, 1),
    (64, 2),
    (64, 4),
    (128, 1),
    (128, 2),
    (128, 4),
    (256, 1),
    (256, 2),
    (256, 4),
    (256, 8),
    (512, 1),
    (512, 2),
    (512, 4),
    (512, 8),
    (1024, 1),
    (1024, 2),
    (1024, 4),
    (1024, 8),
)
ROUTINGS = ("uniform", "no-local", "all-local")


@dataclass(frozen=True)
class FixedConfig:
    block_size: int
    num_warps: int
    num_stages: int = 3

    @property
    def name(self) -> str:
        return f"B{self.block_size}_W{self.num_warps}_S{self.num_stages}"

    def as_dict(self) -> dict[str, int]:
        return {
            "block_size": self.block_size,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


def parse_config(text: str) -> FixedConfig:
    values = tuple(int(value) for value in text.split(","))
    if len(values) == 2:
        return FixedConfig(values[0], values[1])
    if len(values) == 3:
        return FixedConfig(*values)
    raise argparse.ArgumentTypeError("config must be BLOCK_SIZE,NUM_WARPS[,NUM_STAGES]")


def make_ids(
    routing: str,
    m: int,
    topk: int,
    global_experts: int,
    local_experts: int,
    device: torch.device,
) -> torch.Tensor:
    # CPU construction makes the route stream independent of the selected GPU.
    generator = torch.Generator(device="cpu")
    # Restart from the same stream so larger M values extend the smaller-M
    # cases. The first uniform row intentionally contains one local route.
    generator.manual_seed(20260824)
    rows = []
    for _ in range(m):
        if routing == "uniform":
            row = torch.randperm(global_experts, generator=generator)[:topk]
        elif routing == "no-local":
            row = (
                torch.randperm(global_experts - local_experts, generator=generator)[
                    :topk
                ]
                + local_experts
            )
        elif routing == "all-local":
            row = torch.randperm(local_experts, generator=generator)[:topk]
        else:
            raise ValueError(f"unsupported routing: {routing}")
        rows.append(row)
    return torch.stack(rows).to(device=device, dtype=torch.int32).contiguous()


def local_route_mask(
    ids: torch.Tensor, expert_map: torch.Tensor, local_experts: int
) -> torch.Tensor:
    valid = (ids >= 0) & (ids < expert_map.numel())
    safe_ids = torch.where(valid, ids, 0).to(torch.int64)
    mapped = expert_map[safe_ids]
    return valid & (mapped >= 0) & (mapped < local_experts)


def route_order_reference(inp: torch.Tensor, local_mask: torch.Tensor) -> torch.Tensor:
    # Match the kernel's route order and BF16 output boundary exactly.
    m, topk, hidden = inp.shape
    acc = torch.zeros((m, hidden), dtype=torch.float32, device=inp.device)
    zero = torch.zeros((), dtype=torch.float32, device=inp.device)
    for route_idx in range(topk):
        acc += torch.where(
            local_mask[:, route_idx, None],
            inp[:, route_idx].float(),
            zero,
        )
    return acc.to(inp.dtype)


def allocate_output(
    m: int,
    hidden: int,
    topk: int,
    intermediate: int,
    device: torch.device,
    dtype: torch.dtype,
    alias_output_cache2: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not alias_output_cache2:
        return torch.empty((m, hidden), device=device, dtype=dtype), None
    cache2_numel = m * topk * intermediate
    if cache2_numel < m * hidden:
        raise ValueError("cache2 backing is too small for the output alias")
    backing = torch.empty(cache2_numel, device=device, dtype=dtype)
    return backing[: m * hidden].view(m, hidden), backing


def launch_fixed(
    inp: torch.Tensor,
    output: torch.Tensor,
    ids: torch.Tensor,
    expert_map: torch.Tensor,
    local_experts: int,
    config: FixedConfig,
) -> None:
    m, topk, hidden = inp.shape
    grid = (m, triton.cdiv(hidden, config.block_size))
    _moe_sum_ep_kernel[grid](
        inp,
        output,
        ids,
        expert_map,
        m,
        topk,
        hidden,
        expert_map.numel(),
        local_experts,
        inp.stride(0),
        inp.stride(1),
        inp.stride(2),
        output.stride(0),
        output.stride(1),
        ids.stride(0),
        ids.stride(1),
        BLOCK_SIZE=config.block_size,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )


def capture(fn, warmups: int) -> torch.cuda.CUDAGraph:
    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        for _ in range(warmups):
            fn()
    current.wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / replays)


def paired_graph_bench(
    baseline_graph: torch.cuda.CUDAGraph,
    candidate_graph: torch.cuda.CUDAGraph,
    rounds: int,
    replays: int,
    graph_warmups: int,
) -> tuple[list[float], list[float], list[float]]:
    for replay_idx in range(graph_warmups):
        graph = baseline_graph if replay_idx % 2 == 0 else candidate_graph
        graph.replay()
    torch.cuda.synchronize()

    baseline_samples = []
    candidate_samples = []
    reductions = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            bracket = (
                ("baseline", baseline_graph),
                ("candidate", candidate_graph),
                ("candidate", candidate_graph),
                ("baseline", baseline_graph),
            )
        else:
            bracket = (
                ("candidate", candidate_graph),
                ("baseline", baseline_graph),
                ("baseline", baseline_graph),
                ("candidate", candidate_graph),
            )
        round_values: dict[str, list[float]] = {
            "baseline": [],
            "candidate": [],
        }
        for name, graph in bracket:
            round_values[name].append(elapsed_ms(graph, replays))
        baseline = statistics.mean(round_values["baseline"])
        candidate = statistics.mean(round_values["candidate"])
        baseline_samples.append(baseline)
        candidate_samples.append(candidate)
        reductions.append(100.0 * (1.0 - candidate / baseline))
    return baseline_samples, candidate_samples, reductions


def autotune_config_dict() -> dict[str, int]:
    config = moe_sum_ep_kernel.best_config
    return {
        "block_size": int(config.kwargs["BLOCK_SIZE"]),
        "num_warps": int(config.num_warps),
        "num_stages": int(config.num_stages),
    }


def config_key(config: dict[str, int]) -> str:
    return f"B{config['block_size']}_W{config['num_warps']}_" f"S{config['num_stages']}"


def summarize(cases: list[dict], configs: tuple[FixedConfig, ...]) -> dict[str, object]:
    by_m: dict[int, list[dict]] = defaultdict(list)
    for case in cases:
        by_m[case["m"]].append(case)

    def aggregate(selected_cases: list[dict]) -> list[dict]:
        rows = []
        for config in configs:
            observations = []
            for case in selected_cases:
                match = next(
                    (
                        candidate
                        for candidate in case["candidates"]
                        if candidate["name"] == config.name
                    ),
                    None,
                )
                if match is None or match.get("error") is not None:
                    observations = []
                    break
                observations.append(match)
            if not observations:
                continue
            correctness = all(
                item["reference_bitwise"]
                and item["autotune_bitwise"]
                and item["candidate_graph_bitwise"]
                and item["cross_graph_bitwise"]
                for item in observations
            )
            reductions = [item["paired_reduction_median_pct"] for item in observations]
            # Use the paired statistic for aggregation. Independent medians can
            # disagree on a bimodal clock trace even when every ABBA pair is
            # locally well controlled.
            ratios = [
                1.0 - item["paired_reduction_median_pct"] / 100.0
                for item in observations
            ]
            rows.append(
                {
                    "name": config.name,
                    **config.as_dict(),
                    "cases": len(observations),
                    "all_correct": correctness,
                    "positive_cases": sum(value > 0.0 for value in reductions),
                    "median_reduction_pct": statistics.median(reductions),
                    "worst_reduction_pct": min(reductions),
                    "best_reduction_pct": max(reductions),
                    "geomean_reduction_pct": 100.0
                    * (1.0 - math.prod(ratios) ** (1.0 / len(ratios))),
                }
            )
        return sorted(
            rows,
            key=lambda item: (
                bool(item["all_correct"]),
                float(item["worst_reduction_pct"]),
                float(item["median_reduction_pct"]),
            ),
            reverse=True,
        )

    per_m = {}
    for m, selected_cases in sorted(by_m.items()):
        ranking = aggregate(selected_cases)
        per_m[str(m)] = {
            "best_stable": ranking[0] if ranking else None,
            "ranking": ranking,
        }
    global_ranking = aggregate(cases)
    return {
        "per_m": per_m,
        "global": {
            "best_stable": global_ranking[0] if global_ranking else None,
            "ranking": global_ranking,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--global-experts", type=int, default=288)
    parser.add_argument("--local-experts", type=int, default=18)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--routing", choices=("all", *ROUTINGS), default="all")
    parser.add_argument(
        "--config",
        dest="configs",
        type=parse_config,
        action="append",
        help=(
            "candidate BLOCK_SIZE,NUM_WARPS[,NUM_STAGES]; repeat the flag "
            "to replace the default sweep"
        ),
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--capture-warmups", type=int, default=3)
    parser.add_argument("--graph-warmups", type=int, default=20)
    parser.add_argument(
        "--retune-per-case",
        action="store_true",
        help="clear the in-process production autotune choice before every case",
    )
    parser.add_argument(
        "--disjoint-output",
        action="store_true",
        help="do not place output in a separate cache2-sized backing allocation",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="include per-round timings in the JSON output",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit the full per-case candidate tables from JSON output",
    )
    args = parser.parse_args()

    configs = (
        tuple(args.configs)
        if args.configs
        else tuple(FixedConfig(*values) for values in DEFAULT_CONFIGS)
    )
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("candidate configs must be unique")
    for config in configs:
        if config.block_size <= 0 or config.block_size & (config.block_size - 1):
            raise ValueError("BLOCK_SIZE must be a positive power of two")
        if config.num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError("NUM_WARPS must be a supported positive power of two")

    torch.manual_seed(20260824)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    alias_output_cache2 = not args.disjoint_output
    expert_map = torch.full(
        (args.global_experts,), -1, dtype=torch.int32, device=device
    )
    expert_map[: args.local_experts] = torch.arange(
        args.local_experts, dtype=torch.int32, device=device
    )
    routings = ROUTINGS if args.routing == "all" else (args.routing,)

    # Avoid inheriting a choice made by unrelated code in an interactive run.
    moe_sum_ep_kernel.cache.clear()
    cases = []
    shared_autotune_config = None
    for m in args.m_values:
        for routing in routings:
            ids = make_ids(
                routing,
                m,
                args.topk,
                args.global_experts,
                args.local_experts,
                device,
            )
            mask = local_route_mask(ids, expert_map, args.local_experts)
            inp = torch.randn((m, args.topk, args.hidden), device=device, dtype=dtype)
            # Poison remote routes. A correct EP combine must never consume them.
            inp.masked_fill_(~mask[:, :, None], float("nan"))
            reference = route_order_reference(inp, mask)

            baseline_output, baseline_backing = allocate_output(
                m,
                args.hidden,
                args.topk,
                args.intermediate,
                device,
                dtype,
                alias_output_cache2,
            )

            def baseline_fn() -> None:
                moe_sum_ep(
                    inp,
                    baseline_output,
                    ids,
                    expert_map,
                    args.local_experts,
                )

            if args.retune_per_case:
                moe_sum_ep_kernel.cache.clear()
            baseline_fn()
            torch.cuda.synchronize()
            selected_autotune_config = autotune_config_dict()
            if shared_autotune_config is None:
                shared_autotune_config = selected_autotune_config
            baseline_eager = baseline_output.clone()
            baseline_reference_bitwise = bool(torch.equal(baseline_eager, reference))
            baseline_graph = capture(baseline_fn, args.capture_warmups)
            baseline_graph.replay()
            torch.cuda.synchronize()
            baseline_graph_output = baseline_output.clone()
            baseline_graph_bitwise = bool(
                torch.equal(baseline_eager, baseline_graph_output)
            )

            candidates = []
            for config in configs:
                candidate_output, candidate_backing = allocate_output(
                    m,
                    args.hidden,
                    args.topk,
                    args.intermediate,
                    device,
                    dtype,
                    alias_output_cache2,
                )

                def candidate_fn(selected: FixedConfig = config) -> None:
                    launch_fixed(
                        inp,
                        candidate_output,
                        ids,
                        expert_map,
                        args.local_experts,
                        selected,
                    )

                result = {"name": config.name, **config.as_dict()}
                try:
                    candidate_fn()
                    torch.cuda.synchronize()
                    candidate_eager = candidate_output.clone()
                    result.update(
                        {
                            "reference_bitwise": bool(
                                torch.equal(candidate_eager, reference)
                            ),
                            "autotune_bitwise": bool(
                                torch.equal(candidate_eager, baseline_eager)
                            ),
                        }
                    )
                    candidate_graph = capture(candidate_fn, args.capture_warmups)
                    candidate_graph.replay()
                    torch.cuda.synchronize()
                    candidate_graph_output = candidate_output.clone()
                    result.update(
                        {
                            "candidate_graph_bitwise": bool(
                                torch.equal(candidate_eager, candidate_graph_output)
                            ),
                            "cross_graph_bitwise": bool(
                                torch.equal(
                                    candidate_graph_output, baseline_graph_output
                                )
                            ),
                        }
                    )
                    baseline_samples, candidate_samples, reductions = (
                        paired_graph_bench(
                            baseline_graph,
                            candidate_graph,
                            args.rounds,
                            args.replays,
                            args.graph_warmups,
                        )
                    )
                    result.update(
                        {
                            "baseline_median_us": 1000.0
                            * statistics.median(baseline_samples),
                            "candidate_median_us": 1000.0
                            * statistics.median(candidate_samples),
                            "paired_reduction_median_pct": statistics.median(
                                reductions
                            ),
                            "positive_rounds": sum(value > 0.0 for value in reductions),
                            "total_rounds": len(reductions),
                            "error": None,
                        }
                    )
                    if args.include_samples:
                        result.update(
                            {
                                "baseline_samples_us": [
                                    1000.0 * value for value in baseline_samples
                                ],
                                "candidate_samples_us": [
                                    1000.0 * value for value in candidate_samples
                                ],
                                "paired_reductions_pct": reductions,
                            }
                        )
                    del candidate_graph
                except Exception as error:  # Keep the rest of the sweep useful.
                    result["error"] = f"{type(error).__name__}: {error}"
                    torch.cuda.synchronize()
                candidates.append(result)
                # Keep alias backings alive through graph timing.
                del candidate_output, candidate_backing

            valid_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("error") is None
                and candidate["reference_bitwise"]
                and candidate["autotune_bitwise"]
                and candidate["candidate_graph_bitwise"]
                and candidate["cross_graph_bitwise"]
            ]
            best = (
                max(
                    valid_candidates,
                    key=lambda candidate: candidate["paired_reduction_median_pct"],
                )
                if valid_candidates
                else None
            )
            cases.append(
                {
                    "m": m,
                    "routing": routing,
                    "local_routes": int(mask.sum().item()),
                    "local_routes_per_token": [
                        int(value) for value in mask.sum(dim=1).cpu().tolist()
                    ],
                    "autotune_config": selected_autotune_config,
                    "autotune_config_name": config_key(selected_autotune_config),
                    "baseline_reference_bitwise": baseline_reference_bitwise,
                    "baseline_graph_bitwise": baseline_graph_bitwise,
                    "best": best,
                    "candidates": candidates,
                }
            )
            del baseline_graph, baseline_output, baseline_backing

    aggregate = summarize(cases, configs)
    output_cases = cases
    if args.summary_only:
        output_cases = [
            {key: value for key, value in case.items() if key != "candidates"}
            for case in cases
        ]
    report = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "shape": {
            "m_values": args.m_values,
            "hidden": args.hidden,
            "topk": args.topk,
            "global_experts": args.global_experts,
            "local_experts": args.local_experts,
            "dtype": str(dtype),
        },
        "routing": list(routings),
        "alias_output_cache2": alias_output_cache2,
        "combine_input_overlaps_output": False,
        "timing": {
            "method": "CUDA Graph ABBA/BAAB paired",
            "rounds": args.rounds,
            "replays_per_sample": args.replays,
        },
        "autotune_mode": (
            "fresh_per_case"
            if args.retune_per_case
            else "shared_cache_M1_uniform_prime"
        ),
        "shared_autotune_config": shared_autotune_config,
        "configs": [config.as_dict() for config in configs],
        "cases": output_cases,
        "aggregate": aggregate,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
