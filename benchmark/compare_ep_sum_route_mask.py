#!/usr/bin/env python3
"""Isolated deterministic fused-MoE EP-combine experiment.

The production EP combine remaps every top-k expert id once per hidden tile.
This prototype consumes a precomputed per-token bit mask instead.  In a real
implementation the mask can be emitted by the existing compact-alignment
count/prefix kernel, which already maps every route.  Mask construction is
therefore deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from flag_gems.fused.moe_sum import moe_sum_ep


@triton.jit
def _moe_sum_ep_route_mask_kernel(
    input_ptr,
    output_ptr,
    local_route_masks_ptr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_topk: tl.constexpr,
    input_stride_hidden: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return

    route_mask = tl.load(local_route_masks_ptr + token_idx)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token
    for route_idx in tl.static_range(topk):
        local_route = (route_mask & (1 << route_idx)) != 0
        route_ptr = input_base + route_idx * input_stride_topk
        route_data = tl.load(
            route_ptr + hidden_offsets,
            mask=hidden_mask & local_route,
            other=0.0,
        )
        acc += route_data

    output_ptrs = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )
    tl.store(output_ptrs, acc, mask=hidden_mask)


@triton.jit
def _moe_sum_ep_route_mask_branch_kernel(
    input_ptr,
    output_ptr,
    local_route_masks_ptr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_topk: tl.constexpr,
    input_stride_hidden: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Specialize the overwhelmingly common zero/one-local-route cases."""
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return

    route_mask = tl.load(local_route_masks_ptr + token_idx)
    input_base = input_ptr + token_idx * input_stride_token
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if route_mask != 0:
        if (route_mask & (route_mask - 1)) == 0:
            route_idx = 0
            for candidate in tl.static_range(topk):
                route_idx = tl.where(
                    (route_mask & (1 << candidate)) != 0,
                    candidate,
                    route_idx,
                )
            route_ptr = input_base + route_idx * input_stride_topk
            acc += tl.load(
                route_ptr + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            )
        else:
            for route_idx in tl.static_range(topk):
                local_route = (route_mask & (1 << route_idx)) != 0
                route_ptr = input_base + route_idx * input_stride_topk
                route_data = tl.load(
                    route_ptr + hidden_offsets,
                    mask=hidden_mask & local_route,
                    other=0.0,
                )
                acc += route_data

    output_ptrs = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )
    tl.store(output_ptrs, acc, mask=hidden_mask)


@triton.jit
def _moe_sum_ep_route_mask_collision_kernel(
    input_ptr,
    output_ptr,
    local_route_masks_ptr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_topk: tl.constexpr,
    input_stride_hidden: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce collisions; GEMM2 is assumed to have stored unique routes."""
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    route_mask = tl.load(local_route_masks_ptr + token_idx)
    if route_mask != 0 and (route_mask & (route_mask - 1)) == 0:
        return

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token
    for route_idx in tl.static_range(topk):
        local_route = (route_mask & (1 << route_idx)) != 0
        route_ptr = input_base + route_idx * input_stride_topk
        route_data = tl.load(
            route_ptr + hidden_offsets,
            mask=hidden_mask & local_route,
            other=0.0,
        )
        acc += route_data
    output_ptrs = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )
    tl.store(output_ptrs, acc, mask=hidden_mask)


def route_masks_from_mapping(topk_ids, expert_map, local_experts):
    mapped = expert_map[topk_ids.long()]
    local = (mapped >= 0) & (mapped < local_experts)
    bits = 1 << torch.arange(
        topk_ids.shape[1], device=topk_ids.device, dtype=torch.int32
    )
    return (local.to(torch.int32) * bits).sum(dim=1, dtype=torch.int32)


def launch_mask_sum(
    inp, out, masks, block_size, num_warps, *, branch=False, collision=False
):
    m, topk, hidden = inp.shape
    grid = (m, triton.cdiv(hidden, block_size))
    if collision:
        kernel = _moe_sum_ep_route_mask_collision_kernel
    elif branch:
        kernel = _moe_sum_ep_route_mask_branch_kernel
    else:
        kernel = _moe_sum_ep_route_mask_kernel
    kernel[grid](
        inp,
        out,
        masks,
        m,
        topk,
        hidden,
        inp.stride(0),
        inp.stride(1),
        inp.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def bench(fn, rounds, warmup, rep):
    return [
        float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))
        for _ in range(rounds)
    ]


def make_routing(kind, m, topk, global_experts, local_experts, device):
    if kind == "uniform":
        return torch.stack(
            [torch.randperm(global_experts, device=device)[:topk] for _ in range(m)]
        ).to(torch.int32)
    if kind == "skewed":
        ids = torch.stack(
            [torch.randperm(global_experts, device=device)[:topk] for _ in range(m)]
        ).to(torch.int32)
        ids[: min(12, m), 0] = torch.arange(min(12, m), device=device) % local_experts
        return ids
    if kind == "all_local":
        return torch.stack(
            [torch.randperm(local_experts, device=device)[:topk] for _ in range(m)]
        ).to(torch.int32)
    if kind == "no_local":
        return (
            torch.stack(
                [
                    torch.randperm(global_experts - local_experts, device=device)[:topk]
                    for _ in range(m)
                ]
            )
            + local_experts
        ).to(torch.int32)
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--global-experts", type=int, default=288)
    parser.add_argument("--local-experts", type=int, default=18)
    parser.add_argument(
        "--routing",
        choices=("all", "uniform", "skewed", "all_local", "no_local"),
        default="all",
    )
    parser.add_argument(
        "--variant",
        choices=("all", "mask", "branch", "collision"),
        default="all",
    )
    parser.add_argument(
        "--focused",
        action="store_true",
        help="benchmark only the measured 256-thread candidate pairs",
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(20260824)
    device = torch.device("cuda")
    expert_map = torch.full(
        (args.global_experts,), -1, device=device, dtype=torch.int32
    )
    expert_map[: args.local_experts] = torch.arange(
        args.local_experts, device=device, dtype=torch.int32
    )
    inp = torch.randn(
        (args.m, args.topk, args.hidden), device=device, dtype=torch.bfloat16
    )
    ref_out = torch.empty((args.m, args.hidden), device=device, dtype=torch.bfloat16)
    mask_out = torch.empty_like(ref_out)

    configs = [(128, 1), (128, 2), (256, 1), (256, 2), (512, 2), (1024, 4)]
    if args.focused:
        configs = [(256, 1), (256, 2)]
    routings = (
        (args.routing,)
        if args.routing != "all"
        else ("uniform", "skewed", "all_local", "no_local")
    )
    variants = (
        (args.variant,) if args.variant != "all" else ("mask", "branch", "collision")
    )
    results = []
    for routing in routings:
        ids = make_routing(
            routing,
            args.m,
            args.topk,
            args.global_experts,
            args.local_experts,
            device,
        )
        masks = route_masks_from_mapping(ids, expert_map, args.local_experts)

        def ref_fn():
            moe_sum_ep(
                inp,
                ref_out,
                ids,
                expert_map,
                args.local_experts,
                fixed_block_size=512,
                fixed_num_warps=2,
            )

        ref_fn()
        torch.cuda.synchronize()
        ref_samples = bench(ref_fn, args.rounds, args.warmup, args.rep)
        candidates = []
        for block_size, num_warps in configs:
            for variant in variants:

                def mask_fn(bs=block_size, nw=num_warps, selected=variant):
                    launch_mask_sum(
                        inp,
                        mask_out,
                        masks,
                        bs,
                        nw,
                        branch=selected == "branch",
                        collision=selected == "collision",
                    )

                # Unique-route rows model the direct GEMM2 store and are left
                # untouched by the collision-only reduction.
                mask_out.copy_(ref_out)
                mask_fn()
                torch.cuda.synchronize()
                bitwise = bool(torch.equal(ref_out, mask_out))
                samples = bench(mask_fn, args.rounds, args.warmup, args.rep)
                candidates.append(
                    {
                        "block_size": block_size,
                        "num_warps": num_warps,
                        "variant": variant,
                        "median_us": statistics.median(samples) * 1000,
                        "samples_us": [x * 1000 for x in samples],
                        "bitwise": bitwise,
                    }
                )
        best = min(candidates, key=lambda x: x["median_us"])
        results.append(
            {
                "routing": routing,
                "local_routes": int(
                    sum(int(x).bit_count() for x in masks.cpu().tolist())
                ),
                "reference_us": statistics.median(ref_samples) * 1000,
                "reference_samples_us": [x * 1000 for x in ref_samples],
                "candidates": candidates,
                "best": best,
                "best_speedup_percent": 100
                * (statistics.median(ref_samples) * 1000 - best["median_us"])
                / (statistics.median(ref_samples) * 1000),
            }
        )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": [args.m, args.topk, args.hidden],
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
