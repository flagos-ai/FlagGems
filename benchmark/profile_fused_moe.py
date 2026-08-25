#!/usr/bin/env python3
"""Single NVTX-ranged EP fused-MoE call for Nsight Compute."""

from __future__ import annotations

import argparse
import importlib
import json

import torch
import triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("legacy", "optimized", "optimized-no-fused-activation"),
        required=True,
    )
    parser.add_argument(
        "--g2-stages",
        type=int,
        choices=(2, 4),
        help="Override the strict fused-MoE GEMM2 pipeline depth for an NCU A/B.",
    )
    args = parser.parse_args()
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    align = importlib.import_module("flag_gems.fused.moe_align_block_size")

    torch.manual_seed(20260824)
    m, global_e, local_e, h, intermediate, topk = (96, 288, 18, 4096, 2048, 8)
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
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device="cuda", dtype=torch.int32)
    cache13 = torch.empty(
        m * topk * max(2 * intermediate, h), device="cuda", dtype=dtype
    )
    cache2 = torch.empty(m * topk * intermediate, device="cuda", dtype=dtype)
    output = torch.empty_like(hidden)

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

    def no_ep_config(*_args, **_kwargs):
        return None

    if args.policy == "legacy":
        fm.moe_align_block_size = legacy_align
        fm._get_ep_decode_config = no_ep_config
    elif args.policy == "optimized-no-fused-activation":
        fm._should_use_fused_clamped_swiglu = lambda *args, **kwargs: False

    if args.g2_stages is not None:
        original_selector = fm._get_ep_decode_config

        def selector(*positional, gemm_stage=None, **kwargs):
            stage = gemm_stage if gemm_stage is not None else positional[-1]
            config = (
                original_selector(*positional, gemm_stage=gemm_stage, **kwargs)
                if gemm_stage is not None
                else original_selector(*positional, **kwargs)
            )
            if config is not None and stage == "gemm2":
                config["num_stages"] = args.g2_stages
            return config

        fm._get_ep_decode_config = selector

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

    for _ in range(3):
        result = op()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("fused_moe")
    result = op()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    mapped = expert_map[ids]
    print(
        json.dumps(
            {
                "policy": args.policy,
                "device": torch.cuda.get_device_name(),
                "shape": [m, global_e, local_e, h, intermediate, topk],
                "local_routes": int((mapped >= 0).sum().item()),
                "finite": bool(torch.isfinite(result).all().item()),
                "l2": float(torch.linalg.vector_norm(result.float()).item()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
