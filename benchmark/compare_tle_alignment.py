#!/usr/bin/env python3
"""Compare an EP-aware TLE cluster alignment prototype with compact Triton.

Run this experiment with the isolated FlagTree environment, for example:

    FLAGTREE_BACKEND=nvidia CUDA_VISIBLE_DEVICES=0 \
      /workspace/.venv-flagtree-moe/bin/python \
      benchmark/compare_tle_alignment.py
"""

from __future__ import annotations

import importlib
import json
import statistics

import torch
import triton
import triton.language as tl

align = importlib.import_module("flag_gems.fused.moe_align_block_size")
if not align.HAS_TLE:
    raise RuntimeError("This experiment requires Triton/FlagTree TLE")
tle = align.tle


@triton.jit(do_not_specialize=["num_routes"])
def ep_align_tle_cluster_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_pad_ptr,
    num_routes,
    mesh: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    MAX_PADDED: tl.constexpr,
    MAX_EXPERT_BLOCKS: tl.constexpr,
    EXPERTS_PER_SHARD: tl.constexpr,
):
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    is_rank0 = cluster_rank == 0
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < NUM_LOCAL_EXPERTS
    token_offsets = tl.arange(0, BLOCK_TOKENS)

    for base in range(
        cluster_rank * BLOCK_TOKENS, MAX_PADDED, CLUSTER_SIZE * BLOCK_TOKENS
    ):
        offsets = base + token_offsets
        tl.store(
            sorted_token_ids_ptr + offsets,
            num_routes,
            mask=offsets < MAX_PADDED,
        )
    for base in range(
        cluster_rank * BLOCK_TOKENS,
        MAX_EXPERT_BLOCKS,
        CLUSTER_SIZE * BLOCK_TOKENS,
    ):
        offsets = base + token_offsets
        tl.store(expert_ids_ptr + offsets, -1, mask=offsets < MAX_EXPERT_BLOCKS)

    local_counts = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_starts = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    count_ptrs = tle.gpu.local_ptr(local_counts, (expert_offsets,))
    rank0_start_ptrs = tle.gpu.local_ptr(local_starts, (expert_offsets,))
    tl.store(count_ptrs, 0, mask=expert_mask)
    if is_rank0:
        tl.store(rank0_start_ptrs, 0, mask=expert_mask)
    tle.distributed_barrier(mesh)

    for base in range(
        cluster_rank * BLOCK_TOKENS, num_routes, CLUSTER_SIZE * BLOCK_TOKENS
    ):
        offsets = base + token_offsets
        route_mask = offsets < num_routes
        global_experts = tl.load(topk_ids_ptr + offsets, mask=route_mask, other=0).to(
            tl.int32
        )
        valid_global = (
            route_mask & (global_experts >= 0) & (global_experts < NUM_GLOBAL_EXPERTS)
        )
        safe_global = tl.where(valid_global, global_experts, 0)
        local_experts = tl.load(
            expert_map_ptr + safe_global, mask=valid_global, other=-1
        ).to(tl.int32)
        local_mask = (
            valid_global & (local_experts >= 0) & (local_experts < NUM_LOCAL_EXPERTS)
        )
        safe_local_experts = tl.where(local_mask, local_experts, 0)
        tl.atomic_add(
            tle.gpu.local_ptr(local_counts, (safe_local_experts,)),
            1,
            mask=local_mask,
            sem="relaxed",
            scope="cta",
        )

    counts = tl.load(count_ptrs, mask=expert_mask, other=0)
    rank0_starts_remote = tle.remote(local_starts, 0, scope=mesh)
    rank0_starts_remote_ptrs = tle.gpu.local_ptr(rank0_starts_remote, (expert_offsets,))
    rank_in_expert = tl.atomic_add(
        rank0_starts_remote_ptrs,
        counts,
        mask=expert_mask,
        sem="relaxed",
        scope="cta",
    )
    # Reuse the per-CTA counters as this CTA's scatter rank base.
    tl.store(count_ptrs, rank_in_expert, mask=expert_mask)
    tle.distributed_barrier(mesh)

    if is_rank0:
        total_counts = tl.load(rank0_start_ptrs, mask=expert_mask, other=0)
        aligned_counts = tl.cdiv(total_counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
        inclusive = tl.cumsum(aligned_counts, axis=0)
        starts = inclusive - aligned_counts
        tl.store(rank0_start_ptrs, starts, mask=expert_mask)
        tl.store(total_tokens_post_pad_ptr, tl.sum(aligned_counts, axis=0))
    tle.distributed_barrier(mesh)

    rank0_starts_remote = tle.remote(local_starts, 0, scope=mesh)
    starts = tl.load(
        tle.gpu.local_ptr(rank0_starts_remote, (expert_offsets,)),
        mask=expert_mask,
        other=0,
    )
    tl.store(
        tle.gpu.local_ptr(local_starts, (expert_offsets,)),
        starts,
        mask=expert_mask,
    )
    total_tokens = tl.load(total_tokens_post_pad_ptr)

    for local_expert_idx in range(EXPERTS_PER_SHARD):
        local_expert = cluster_rank * EXPERTS_PER_SHARD + local_expert_idx
        valid_expert = local_expert < NUM_LOCAL_EXPERTS
        start = tl.load(
            tle.gpu.local_ptr(local_starts, (local_expert,)),
            mask=valid_expert,
            other=0,
        )
        next_expert = local_expert + 1
        has_next = valid_expert & (next_expert < NUM_LOCAL_EXPERTS)
        end = tl.load(
            tle.gpu.local_ptr(local_starts, (next_expert,)),
            mask=has_next,
            other=0,
        )
        end = tl.where(has_next, end, total_tokens)
        start = tl.where(valid_expert, start, 0)
        end = tl.where(valid_expert, end, 0)
        for output_offset in range(start, end, BLOCK_SIZE_M):
            tl.store(expert_ids_ptr + output_offset // BLOCK_SIZE_M, local_expert)

    tle.distributed_barrier(mesh)
    for base in range(
        cluster_rank * BLOCK_TOKENS, num_routes, CLUSTER_SIZE * BLOCK_TOKENS
    ):
        offsets = base + token_offsets
        route_mask = offsets < num_routes
        global_experts = tl.load(topk_ids_ptr + offsets, mask=route_mask, other=0).to(
            tl.int32
        )
        valid_global = (
            route_mask & (global_experts >= 0) & (global_experts < NUM_GLOBAL_EXPERTS)
        )
        safe_global = tl.where(valid_global, global_experts, 0)
        local_experts = tl.load(
            expert_map_ptr + safe_global, mask=valid_global, other=-1
        ).to(tl.int32)
        local_mask = (
            valid_global & (local_experts >= 0) & (local_experts < NUM_LOCAL_EXPERTS)
        )
        safe_local_experts = tl.where(local_mask, local_experts, 0)
        rank = tl.atomic_add(
            tle.gpu.local_ptr(local_counts, (safe_local_experts,)),
            1,
            mask=local_mask,
            sem="relaxed",
            scope="cta",
        )
        start = tl.load(
            tle.gpu.local_ptr(local_starts, (safe_local_experts,)),
            mask=local_mask,
            other=0,
        )
        tl.store(sorted_token_ids_ptr + start + rank, offsets, mask=local_mask)


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


def time_graph(graph, rounds=16, replays=1000):
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
        samples.append(float(start.elapsed_time(end) * 1000 / replays))
    return samples


def validate(name, output, topk_ids, expert_map, block_size):
    sorted_ids, expert_ids, total = output
    total = int(total.item())
    mapped = expert_map[topk_ids.view(-1)]
    for local_expert in range(18):
        expected = torch.where(mapped == local_expert)[0].sort().values
        blocks = torch.where(expert_ids[: total // block_size] == local_expert)[0]
        actual = sorted_ids[
            blocks[:, None] * block_size
            + torch.arange(block_size, device="cuda")[None, :]
        ].flatten()
        actual = actual[actual < topk_ids.numel()].sort().values
        torch.testing.assert_close(
            actual,
            expected,
            check_dtype=False,
            msg=lambda message: f"{name}, local expert {local_expert}: {message}",
        )


def main():
    torch.manual_seed(20260824)
    m, topk, global_e, local_e, block_size = (96, 8, 288, 18, 16)
    logits = torch.randn((m, global_e), device="cuda")
    topk_ids = torch.topk(logits, topk, dim=-1).indices.to(torch.int32)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device="cuda", dtype=torch.int32)
    num_routes = topk_ids.numel()
    max_padded = num_routes + local_e * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)
    block_expert = triton.next_power_of_2(local_e)
    mesh = tle.device_mesh({"block_cluster": [("cluster_x", 8)]})

    variants = {"compact_triton": align.moe_align_block_size_ep_compact}
    keepalive = []
    for block_tokens in (128, 256, 512, 1024):
        for num_warps in (4, 8):
            sorted_ids = torch.empty(max_padded, device="cuda", dtype=torch.int32)
            expert_ids = torch.empty(max_blocks, device="cuda", dtype=torch.int32)
            total = torch.empty(1, device="cuda", dtype=torch.int32)
            keepalive.extend((sorted_ids, expert_ids, total))

            def run_tle(
                _sorted=sorted_ids,
                _experts=expert_ids,
                _total=total,
                _block_tokens=block_tokens,
                _num_warps=num_warps,
            ):
                ep_align_tle_cluster_kernel[(1,)](
                    topk_ids,
                    expert_map,
                    _sorted,
                    _experts,
                    _total,
                    num_routes,
                    mesh=mesh,
                    CLUSTER_SIZE=8,
                    NUM_GLOBAL_EXPERTS=global_e,
                    NUM_LOCAL_EXPERTS=local_e,
                    BLOCK_SIZE_M=block_size,
                    BLOCK_TOKENS=_block_tokens,
                    BLOCK_EXPERT=block_expert,
                    MAX_PADDED=max_padded,
                    MAX_EXPERT_BLOCKS=max_blocks,
                    EXPERTS_PER_SHARD=triton.cdiv(local_e, 8),
                    num_warps=_num_warps,
                )
                return _sorted, _experts, _total

            variants[f"tle_bt{block_tokens}_w{num_warps}"] = run_tle

    graphs = {}
    outputs = {}
    for name, fn in variants.items():
        if name == "compact_triton":
            fn = lambda: align.moe_align_block_size_ep_compact(
                topk_ids, expert_map, block_size, local_e
            )
        graph, output = capture(fn)
        graphs[name] = graph
        outputs[name] = output
        graph.replay()
        torch.cuda.synchronize()
        validate(name, output, topk_ids, expert_map, block_size)

    samples = {name: time_graph(graph) for name, graph in graphs.items()}
    baseline = statistics.median(samples["compact_triton"])
    result = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "M": m,
            "global_E": global_e,
            "local_E": local_e,
            "topk": topk,
            "block_size": block_size,
            "local_routes": int((expert_map[topk_ids] >= 0).sum().item()),
        },
        "results": {
            name: {
                "samples_us": values,
                "median_us": statistics.median(values),
                "speedup_vs_compact": baseline / statistics.median(values),
            }
            for name, values in samples.items()
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
