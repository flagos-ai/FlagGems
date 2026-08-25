#!/usr/bin/env python3
"""CUDA Graph-safe route-density prototype for EP16 fused MoE.

The prototype deliberately lives outside the production dispatch.  It builds
both BM16 and BM64 compact alignments, gates their device-side padded counts,
launches both GEMM specializations into shared caches, and performs one EP
combine.  The inactive GEMM sees a zero padded count and returns before any
output write.  No device-to-host synchronization or host conditional is used,
so one captured graph can replay sparse and dense routing inputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics

import torch
import triton
import triton.language as tl

SPARSE_GEMM1 = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3,
    "PAIR_GATE_UP_DOT": True,
    "CLAMPED_BF16_BOUNDARY": True,
    "CLAMP_LIMIT": 10.0,
}
SPARSE_GEMM2 = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}
DENSE_GEMM1 = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3,
    "PAIR_GATE_UP_DOT": True,
    "CLAMPED_BF16_BOUNDARY": True,
    "CLAMP_LIMIT": 10.0,
}
DENSE_GEMM2 = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3,
}


@triton.jit
def _gate_padded_counts_kernel(
    sparse_total_ptr,
    dense_total_ptr,
    gated_sparse_total_ptr,
    gated_dense_total_ptr,
    dense_flag_ptr,
    SPARSE_PADDED_THRESHOLD: tl.constexpr,
):
    sparse_total = tl.load(sparse_total_ptr)
    dense_total = tl.load(dense_total_ptr)
    use_dense = sparse_total > SPARSE_PADDED_THRESHOLD
    tl.store(gated_sparse_total_ptr, tl.where(use_dense, 0, sparse_total))
    tl.store(gated_dense_total_ptr, tl.where(use_dense, dense_total, 0))
    tl.store(dense_flag_ptr, use_dense.to(tl.int32))


@triton.jit
def _dual_ep_align_count_prefix_init_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    expert_starts_sparse_ptr,
    expert_starts_dense_ptr,
    expert_ranks_ptr,
    sorted_sparse_ptr,
    sorted_dense_ptr,
    expert_ids_sparse_ptr,
    expert_ids_dense_ptr,
    total_sparse_ptr,
    total_dense_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    MAX_SPARSE_PADDED: tl.constexpr,
    MAX_DENSE_PADDED: tl.constexpr,
    INIT_BLOCK: tl.constexpr,
    MAX_SPARSE_BLOCKS_PER_EXPERT: tl.constexpr,
    MAX_DENSE_BLOCKS_PER_EXPERT: tl.constexpr,
):
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_mask, other=-1
    )
    valid_global = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global, mask=valid_global, other=-1
    )
    local_mask = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local = tl.where(local_mask, local_experts_raw, 0).to(tl.int32)
    counts = tl.histogram(safe_local, BLOCK_EXPERT, mask=local_mask).to(tl.int32)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < NUM_LOCAL_EXPERTS
    counts = tl.where(expert_mask, counts, 0)
    sparse_counts = tl.cdiv(counts, 16) * 16
    dense_counts = tl.cdiv(counts, 64) * 64
    sparse_starts = tl.cumsum(sparse_counts, axis=0) - sparse_counts
    dense_starts = tl.cumsum(dense_counts, axis=0) - dense_counts
    sparse_total = tl.sum(sparse_counts, axis=0)
    dense_total = tl.sum(dense_counts, axis=0)

    tl.store(
        expert_starts_sparse_ptr + expert_offsets,
        sparse_starts,
        mask=expert_mask,
    )
    tl.store(
        expert_starts_dense_ptr + expert_offsets,
        dense_starts,
        mask=expert_mask,
    )
    tl.store(expert_ranks_ptr + expert_offsets, 0, mask=expert_mask)
    tl.store(total_sparse_ptr, sparse_total)
    tl.store(total_dense_ptr, dense_total)

    init_offsets = tl.arange(0, INIT_BLOCK)
    for base in tl.static_range(0, MAX_SPARSE_PADDED, INIT_BLOCK):
        offsets = base + init_offsets
        tl.store(
            sorted_sparse_ptr + offsets,
            NUM_ROUTES,
            mask=offsets < sparse_total,
        )
    for base in tl.static_range(0, MAX_DENSE_PADDED, INIT_BLOCK):
        offsets = base + init_offsets
        tl.store(
            sorted_dense_ptr + offsets,
            NUM_ROUTES,
            mask=offsets < dense_total,
        )

    for block_idx in tl.static_range(0, MAX_SPARSE_BLOCKS_PER_EXPERT):
        valid_block = expert_mask & (block_idx * 16 < sparse_counts)
        output_block = sparse_starts // 16 + block_idx
        tl.store(
            expert_ids_sparse_ptr + output_block,
            expert_offsets,
            mask=valid_block,
        )
    for block_idx in tl.static_range(0, MAX_DENSE_BLOCKS_PER_EXPERT):
        valid_block = expert_mask & (block_idx * 64 < dense_counts)
        output_block = dense_starts // 64 + block_idx
        tl.store(
            expert_ids_dense_ptr + output_block,
            expert_offsets,
            mask=valid_block,
        )


@triton.jit
def _dual_ep_align_scatter_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    expert_starts_sparse_ptr,
    expert_starts_dense_ptr,
    expert_ranks_ptr,
    sorted_sparse_ptr,
    sorted_dense_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route_offsets = tl.program_id(0) * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
    route_mask = route_offsets < NUM_ROUTES
    global_experts_raw = tl.load(
        topk_ids_ptr + route_offsets, mask=route_mask, other=-1
    )
    valid_global = (
        route_mask
        & (global_experts_raw >= 0)
        & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    )
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global, mask=valid_global, other=-1
    )
    local_mask = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    safe_local = tl.where(local_mask, local_experts_raw, 0).to(tl.int32)
    ranks = tl.atomic_add(expert_ranks_ptr + safe_local, 1, mask=local_mask)
    sparse_starts = tl.load(
        expert_starts_sparse_ptr + safe_local, mask=local_mask, other=0
    )
    dense_starts = tl.load(
        expert_starts_dense_ptr + safe_local, mask=local_mask, other=0
    )
    tl.store(
        sorted_sparse_ptr + sparse_starts + ranks,
        route_offsets,
        mask=local_mask,
    )
    tl.store(
        sorted_dense_ptr + dense_starts + ranks,
        route_offsets,
        mask=local_mask,
    )


def _dual_ep_align(topk_ids, expert_map, global_e, local_e):
    num_routes = topk_ids.numel()
    max_sparse = num_routes + local_e * 15
    max_dense = num_routes + local_e * 63
    sparse_sorted = torch.empty(max_sparse, device=topk_ids.device, dtype=torch.int32)
    dense_sorted = torch.empty(max_dense, device=topk_ids.device, dtype=torch.int32)
    sparse_experts = torch.empty(
        triton.cdiv(max_sparse, 16), device=topk_ids.device, dtype=torch.int32
    )
    dense_experts = torch.empty(
        triton.cdiv(max_dense, 64), device=topk_ids.device, dtype=torch.int32
    )
    sparse_total = torch.empty(1, device=topk_ids.device, dtype=torch.int32)
    dense_total = torch.empty(1, device=topk_ids.device, dtype=torch.int32)
    sparse_starts = torch.empty(local_e, device=topk_ids.device, dtype=torch.int32)
    dense_starts = torch.empty(local_e, device=topk_ids.device, dtype=torch.int32)
    expert_ranks = torch.empty(local_e, device=topk_ids.device, dtype=torch.int32)
    _dual_ep_align_count_prefix_init_kernel[(1,)](
        topk_ids,
        expert_map,
        sparse_starts,
        dense_starts,
        expert_ranks,
        sparse_sorted,
        dense_sorted,
        sparse_experts,
        dense_experts,
        sparse_total,
        dense_total,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=global_e,
        NUM_LOCAL_EXPERTS=local_e,
        BLOCK_EXPERT=triton.next_power_of_2(local_e),
        BLOCK_ROUTES=triton.next_power_of_2(num_routes),
        MAX_SPARSE_PADDED=max_sparse,
        MAX_DENSE_PADDED=max_dense,
        INIT_BLOCK=256,
        MAX_SPARSE_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, 16),
        MAX_DENSE_BLOCKS_PER_EXPERT=triton.cdiv(num_routes, 64),
        num_warps=4,
    )
    scatter_block = 128
    _dual_ep_align_scatter_kernel[(triton.cdiv(num_routes, scatter_block),)](
        topk_ids,
        expert_map,
        sparse_starts,
        dense_starts,
        expert_ranks,
        sparse_sorted,
        dense_sorted,
        NUM_ROUTES=num_routes,
        NUM_GLOBAL_EXPERTS=global_e,
        NUM_LOCAL_EXPERTS=local_e,
        BLOCK_ROUTES=scatter_block,
        num_warps=4,
    )
    return (
        sparse_sorted,
        sparse_experts,
        sparse_total,
        dense_sorted,
        dense_experts,
        dense_total,
    )


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _legacy_align(align, topk_ids, block_size, num_experts, expert_map=None, **_):
    max_padded = topk_ids.numel() + num_experts * (block_size - 1)
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


def _capture_fused_policy(fm, fn, *, legacy, align):
    if not legacy:
        return _capture(fn)
    old_align = fm.moe_align_block_size
    old_config = fm._get_ep_decode_config
    old_ep_sum = fm._should_use_ep_sum
    old_activation = fm._should_use_fused_clamped_swiglu

    def legacy_align(*args, **kwargs):
        return _legacy_align(align, *args, **kwargs)

    try:
        fm.moe_align_block_size = legacy_align
        fm._get_ep_decode_config = lambda *_args, **_kwargs: None
        fm._should_use_ep_sum = lambda *_args, **_kwargs: False
        fm._should_use_fused_clamped_swiglu = lambda *_args, **_kwargs: False
        return _capture(fn)
    finally:
        fm.moe_align_block_size = old_align
        fm._get_ep_decode_config = old_config
        fm._should_use_ep_sum = old_ep_sum
        fm._should_use_fused_clamped_swiglu = old_activation


def _make_routes(kind, m, topk, global_e, local_e, device):
    route_offsets = torch.arange(topk, device=device, dtype=torch.int32)
    token_offsets = torch.arange(m, device=device, dtype=torch.int32)[:, None]
    if kind == "all_local":
        return (token_offsets + route_offsets).remainder(local_e).contiguous()
    if kind == "uniform":
        # 131 is coprime with 288, so the flattened permutation covers all
        # global experts before repeating while retaining distinct top-k IDs.
        flat_routes = torch.arange(m * topk, device=device, dtype=torch.int64)
        return ((flat_routes * 131) % global_e).to(torch.int32).view(m, topk)
    ids = local_e + (token_offsets + route_offsets).remainder(global_e - local_e)
    if kind.startswith("local"):
        local_routes = int(kind.removeprefix("local"))
        if not 0 < local_routes < topk:
            raise ValueError(f"unsupported route kind: {kind}")
        ids[:, :local_routes] = (
            token_offsets + route_offsets[None, :local_routes]
        ).remainder(local_e)
    elif kind != "no_local":
        raise ValueError(f"unsupported route kind: {kind}")
    return ids.contiguous()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=640)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument(
        "--routes",
        default="uniform,local4,all_local,no_local",
        help=(
            "comma-separated subset of uniform, local1..local7, " "all_local, no_local"
        ),
    )
    parser.add_argument("--dense-g1-bn", type=int, default=DENSE_GEMM1["BLOCK_SIZE_N"])
    parser.add_argument("--dense-g1-bk", type=int, default=DENSE_GEMM1["BLOCK_SIZE_K"])
    parser.add_argument("--dense-g1-warps", type=int, default=DENSE_GEMM1["num_warps"])
    parser.add_argument(
        "--dense-g1-stages", type=int, default=DENSE_GEMM1["num_stages"]
    )
    parser.add_argument("--dense-g2-bn", type=int, default=DENSE_GEMM2["BLOCK_SIZE_N"])
    parser.add_argument("--dense-g2-bk", type=int, default=DENSE_GEMM2["BLOCK_SIZE_K"])
    parser.add_argument("--dense-g2-warps", type=int, default=DENSE_GEMM2["num_warps"])
    parser.add_argument(
        "--dense-g2-stages", type=int, default=DENSE_GEMM2["num_stages"]
    )
    parser.add_argument("--dense-g2-persistent-grid", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    route_kinds = tuple(filter(None, args.routes.split(",")))
    valid_route_kinds = {
        "uniform",
        *(f"local{i}" for i in range(1, 8)),
        "all_local",
        "no_local",
    }
    if not route_kinds or not set(route_kinds) <= valid_route_kinds:
        raise ValueError(f"invalid --routes value: {args.routes}")
    dense_gemm1 = {
        **DENSE_GEMM1,
        "BLOCK_SIZE_N": args.dense_g1_bn,
        "BLOCK_SIZE_K": args.dense_g1_bk,
        "num_warps": args.dense_g1_warps,
        "num_stages": args.dense_g1_stages,
    }
    dense_gemm2 = {
        **DENSE_GEMM2,
        "BLOCK_SIZE_N": args.dense_g2_bn,
        "BLOCK_SIZE_K": args.dense_g2_bk,
        "num_warps": args.dense_g2_warps,
        "num_stages": args.dense_g2_stages,
    }
    if args.dense_g2_persistent_grid:
        dense_gemm2["PERSISTENT_GRID_SIZE"] = args.dense_g2_persistent_grid
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    align = importlib.import_module("flag_gems.fused.moe_align_block_size")
    moe_sum_mod = importlib.import_module("flag_gems.fused.moe_sum")

    torch.manual_seed(20260824)
    m, global_e, local_e, h, intermediate, topk = 96, 288, 18, 4096, 2048, 8
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden = torch.randn((m, h), device=device, dtype=dtype)
    w1 = torch.empty(
        (local_e, 2 * intermediate, h), device=device, dtype=dtype
    ).normal_(std=h**-0.5)
    w2 = torch.empty((local_e, h, intermediate), device=device, dtype=dtype).normal_(
        std=intermediate**-0.5
    )
    weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype)
    ids = _make_routes("uniform", m, topk, global_e, local_e, device)
    expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device=device, dtype=torch.int32)

    def allocate_workspace():
        cache13 = torch.empty(
            m * topk * max(2 * intermediate, h), device=device, dtype=dtype
        )
        cache2 = torch.empty(m * topk * intermediate, device=device, dtype=dtype)
        output = torch.empty_like(hidden)
        return cache13, cache2, output

    legacy_cache13, legacy_cache2, legacy_output = allocate_workspace()
    current_cache13, current_cache2, current_output = allocate_workspace()
    adaptive_cache13, adaptive_cache2, adaptive_output = allocate_workspace()

    def legacy_op():
        return fm.fused_experts_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=legacy_output,
            intermediate_cache13=legacy_cache13,
            intermediate_cache2=legacy_cache2,
        )

    def current_op():
        return fm.fused_experts_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=current_output,
            intermediate_cache13=current_cache13,
            intermediate_cache2=current_cache2,
        )

    gated_sparse_total = torch.empty(1, device=device, dtype=torch.int32)
    gated_dense_total = torch.empty(1, device=device, dtype=torch.int32)
    dense_flag = torch.empty(1, device=device, dtype=torch.int32)
    adaptive_cache2_view = adaptive_cache2.view(m * topk, intermediate)
    adaptive_cache3 = adaptive_cache13[: m * topk * h].view(m, topk, h)

    def adaptive_op():
        (
            sparse_sorted,
            sparse_experts,
            sparse_total,
            dense_sorted,
            dense_experts,
            dense_total,
        ) = _dual_ep_align(
            ids,
            expert_map,
            global_e,
            local_e,
        )
        _gate_padded_counts_kernel[(1,)](
            sparse_total,
            dense_total,
            gated_sparse_total,
            gated_dense_total,
            dense_flag,
            SPARSE_PADDED_THRESHOLD=args.threshold,
            num_warps=1,
        )
        for sorted_ids, expert_ids, total, config in (
            (sparse_sorted, sparse_experts, gated_sparse_total, SPARSE_GEMM1),
            (dense_sorted, dense_experts, gated_dense_total, dense_gemm1),
        ):
            fm.dispatch_fused_moe_kernel(
                hidden,
                w1,
                adaptive_cache2_view.view(m, topk, intermediate),
                None,
                None,
                None,
                None,
                sorted_ids,
                expert_ids,
                total,
                False,
                topk,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                FUSE_SILU=True,
            )
        for sorted_ids, expert_ids, total, config in (
            (sparse_sorted, sparse_experts, gated_sparse_total, SPARSE_GEMM2),
            (dense_sorted, dense_experts, gated_dense_total, dense_gemm2),
        ):
            fm.dispatch_fused_moe_kernel(
                adaptive_cache2_view,
                w2,
                adaptive_cache3,
                None,
                None,
                None,
                weights,
                sorted_ids,
                expert_ids,
                total,
                True,
                1,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
            )
        moe_sum_mod.moe_sum_ep(
            adaptive_cache3,
            adaptive_output,
            ids,
            expert_map,
            local_e,
            fixed_block_size=512,
            fixed_num_warps=2,
        )
        return adaptive_output

    legacy_graph, legacy_graph_output = _capture_fused_policy(
        fm, legacy_op, legacy=True, align=align
    )
    current_graph, current_graph_output = _capture_fused_policy(
        fm, current_op, legacy=False, align=align
    )
    adaptive_graph, adaptive_graph_output = _capture(adaptive_op)
    graphs = {
        "legacy": legacy_graph,
        "current_bm16": current_graph,
        "adaptive_bm16_bm64": adaptive_graph,
    }
    graph_outputs = {
        "legacy": legacy_graph_output,
        "current_bm16": current_graph_output,
        "adaptive_bm16_bm64": adaptive_graph_output,
    }

    results = {}
    for route_kind in route_kinds:
        ids.copy_(_make_routes(route_kind, m, topk, global_e, local_e, device))
        for graph in graphs.values():
            for _ in range(20):
                graph.replay()
        torch.cuda.synchronize()

        samples = {name: [] for name in graphs}
        names = list(graphs)
        for round_idx in range(args.rounds):
            # Six-order Latin cycle: every policy occupies every timing
            # position in both directions.  Dense routes change H20 clocks
            # enough that simple forward/reverse ordering biases the middle
            # policy by tens of microseconds.
            rotation = round_idx % len(names)
            order = names[rotation:] + names[:rotation]
            if (round_idx // len(names)) % 2:
                order = list(reversed(order))
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(float(start.elapsed_time(end) / args.replays))

        # Re-establish one output from each graph after interleaved timing.
        actual = {}
        for name, graph in graphs.items():
            graph.replay()
            torch.cuda.synchronize()
            actual[name] = graph_outputs[name].clone()
        adaptive_eager = adaptive_op().clone()
        torch.cuda.synchronize()
        adaptive_graph.replay()
        torch.cuda.synchronize()
        adaptive_eager_graph_bitwise_equal = torch.equal(
            adaptive_eager, adaptive_graph_output
        )
        reference = actual["legacy"]
        baseline = statistics.median(samples["legacy"])
        mapped = expert_map[ids]
        route_result = {
            "local_routes": int((mapped >= 0).sum().item()),
            "dense_selected": bool(dense_flag.item()),
            "adaptive_eager_graph_bitwise_equal": bool(
                adaptive_eager_graph_bitwise_equal
            ),
            "policies": {},
        }
        for name in names:
            median = statistics.median(samples[name])
            diff = actual[name].float() - reference.float()
            route_result["policies"][name] = {
                "samples_ms": samples[name],
                "median_ms": median,
                "reduction_vs_legacy_pct": 100.0 * (1.0 - median / baseline),
                "speedup_vs_legacy": baseline / median,
                "bitwise_equal_to_legacy": bool(torch.equal(actual[name], reference)),
                "max_abs_to_legacy": float(diff.abs().max().item()),
            }
        results[route_kind] = route_result

    # Explicitly prove that the same captured adaptive graph changes policy
    # when only the route tensor contents change.
    replay_transitions = []
    for route_kind in ("uniform", "all_local", "local4", "no_local", "all_local"):
        ids.copy_(_make_routes(route_kind, m, topk, global_e, local_e, device))
        adaptive_graph.replay()
        legacy_graph.replay()
        torch.cuda.synchronize()
        replay_transitions.append(
            {
                "routing": route_kind,
                "dense_selected": bool(dense_flag.item()),
                "bitwise_equal_to_legacy": bool(
                    torch.equal(adaptive_graph_output, legacy_graph_output)
                ),
            }
        )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": {
                    "M": m,
                    "global_E": global_e,
                    "local_E": local_e,
                    "H": h,
                    "I": intermediate,
                    "topk": topk,
                    "dtype": str(dtype),
                },
                "sparse_padded_threshold": args.threshold,
                "dense_gemm1": dense_gemm1,
                "dense_gemm2": dense_gemm2,
                "results": results,
                "same_graph_route_transitions": replay_transitions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
