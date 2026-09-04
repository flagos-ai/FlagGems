#!/usr/bin/env python3
"""Test deterministic multi-token tiles for fused-MoE EP combine."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from flag_gems.fused.moe_sum import moe_sum_ep


@triton.jit
def _moe_sum_ep_2d_kernel(
    input_ptr,
    output_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    num_global_experts: tl.constexpr,
    local_num_experts: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_topk: tl.constexpr,
    input_stride_hidden: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_hidden: tl.constexpr,
    topk_ids_stride_token: tl.constexpr,
    topk_ids_stride_topk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < num_tokens
    hidden_mask = hidden_offsets < hidden_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    input_base = input_ptr + token_offsets * input_stride_token
    ids_base = topk_ids_ptr + token_offsets * topk_ids_stride_token
    for route_idx in tl.static_range(topk):
        global_expert_raw = tl.load(
            ids_base + route_idx * topk_ids_stride_topk,
            mask=token_mask,
            other=-1,
        )
        valid_global_expert = (global_expert_raw >= 0) & (
            global_expert_raw < num_global_experts
        )
        safe_global_expert = tl.where(valid_global_expert, global_expert_raw, 0).to(
            tl.int64
        )
        local_expert_raw = tl.load(
            expert_map_ptr + safe_global_expert,
            mask=token_mask & valid_global_expert,
            other=-1,
        )
        local_route = (
            valid_global_expert
            & (local_expert_raw >= 0)
            & (local_expert_raw < local_num_experts)
        )
        route_ptrs = (
            input_base[:, None]
            + route_idx * input_stride_topk
            + hidden_offsets[None, :] * input_stride_hidden
        )
        route_data = tl.load(
            route_ptrs,
            mask=token_mask[:, None] & hidden_mask[None, :] & local_route[:, None],
            other=0.0,
        )
        acc += route_data

    output_ptrs = (
        output_ptr
        + token_offsets[:, None] * output_stride_token
        + hidden_offsets[None, :] * output_stride_hidden
    )
    tl.store(
        output_ptrs,
        acc,
        mask=token_mask[:, None] & hidden_mask[None, :],
    )


def launch(inp, out, ids, expert_map, local_experts, block_m, block_n, num_warps):
    m, topk, hidden = inp.shape
    _moe_sum_ep_2d_kernel[(triton.cdiv(m, block_m), triton.cdiv(hidden, block_n))](
        inp,
        out,
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
        out.stride(0),
        out.stride(1),
        ids.stride(0),
        ids.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def bench(fn, rounds, warmup, rep):
    return [
        float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))
        for _ in range(rounds)
    ]


def make_ids(kind, m, topk, global_e, local_e, device):
    if kind == "all_local":
        limit, offset = local_e, 0
    elif kind == "no_local":
        limit, offset = global_e - local_e, local_e
    else:
        limit, offset = global_e, 0
    ids = torch.stack(
        [torch.randperm(limit, device=device)[:topk] + offset for _ in range(m)]
    ).to(torch.int32)
    if kind == "skewed":
        ids[:12, 0] = torch.arange(12, device=device) % local_e
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(20260824)
    m, topk, hidden, global_e, local_e = 96, 8, 4096, 288, 18
    inp = torch.randn((m, topk, hidden), device="cuda", dtype=torch.bfloat16)
    ref = torch.empty((m, hidden), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(ref)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device="cuda", dtype=torch.int32)
    configs = [
        (1, 256, 1),
        (1, 256, 2),
        (1, 512, 2),
        (2, 256, 2),
        (2, 512, 4),
        (4, 128, 2),
        (4, 256, 4),
        (8, 128, 4),
    ]
    results = []
    for routing in ("uniform", "skewed", "all_local", "no_local"):
        ids = make_ids(routing, m, topk, global_e, local_e, inp.device)

        def ref_fn():
            moe_sum_ep(
                inp,
                ref,
                ids,
                expert_map,
                local_e,
                fixed_block_size=512,
                fixed_num_warps=2,
            )

        ref_fn()
        torch.cuda.synchronize()
        ref_samples = bench(ref_fn, args.rounds, args.warmup, args.rep)
        candidates = []
        for block_m, block_n, num_warps in configs:

            def candidate_fn(bm=block_m, bn=block_n, nw=num_warps):
                launch(inp, out, ids, expert_map, local_e, bm, bn, nw)

            candidate_fn()
            torch.cuda.synchronize()
            bitwise = bool(torch.equal(ref, out))
            samples = bench(candidate_fn, args.rounds, args.warmup, args.rep)
            candidates.append(
                {
                    "config": [block_m, block_n, num_warps],
                    "median_us": statistics.median(samples) * 1000,
                    "bitwise": bitwise,
                }
            )
        best = min(candidates, key=lambda item: item["median_us"])
        reference_us = statistics.median(ref_samples) * 1000
        results.append(
            {
                "routing": routing,
                "local_routes": int((expert_map[ids] >= 0).sum().item()),
                "reference_us": reference_us,
                "candidates": candidates,
                "best": best,
                "best_speedup_percent": 100
                * (reference_us - best["median_us"])
                / reference_us,
            }
        )
    print(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "results": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
