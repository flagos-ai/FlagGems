#!/usr/bin/env python3
"""A/B a launch-free local-rank schedule for fused-MoE M=1 EP16.

The production raw-direct path launches one route-major CTA group for every
top-k position.  A singleton local route at p7 therefore starts much later
than one at p0.  This benchmark-only prototype keeps the same three launches
(GEMM1, GEMM2, EP sum), but assigns programs by *local rank*:

* every program loads the eight route IDs and computes their local prefix;
* lane 0 owns the first local route, lane 1 the second, and so on;
* ``LANES`` controls parallelism; a lane loops over local ranks separated by
  ``LANES`` when there are more local routes than lanes;
* the grid is ``(N tiles, LANES)``, so useful lane groups are contiguous and
  independent of their original p0--p7 positions.

No alignment kernel, host-visible route count, or device-to-host sync is used.
The implementation is deliberately strict and is not imported by production.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _select_local_route(
    topk_ids_ptr,
    expert_map_ptr,
    lane,
    local_slot: tl.constexpr,
    LANES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
):
    route_offsets = tl.arange(0, 8)
    global_experts = tl.load(topk_ids_ptr + route_offsets)
    valid_global = (global_experts >= 0) & (global_experts < NUM_GLOBAL_EXPERTS)
    safe_global = tl.where(valid_global, global_experts, 0)
    local_experts = tl.load(expert_map_ptr + safe_global, mask=valid_global, other=-1)
    is_local = valid_global & (local_experts >= 0) & (local_experts < NUM_LOCAL_EXPERTS)
    local_prefix = tl.cumsum(is_local.to(tl.int32), axis=0) - 1
    target_rank = lane + local_slot * LANES
    selected = is_local & (local_prefix == target_rank)
    found = target_rank < tl.sum(is_local.to(tl.int32), axis=0)
    route = tl.sum(tl.where(selected, route_offsets, 0), axis=0)
    expert = tl.sum(tl.where(selected, local_experts, 0), axis=0)
    return found, route, expert


@triton.jit
def _lookup_original_route(
    topk_ids_ptr,
    expert_map_ptr,
    route,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
):
    global_expert = tl.load(topk_ids_ptr + route)
    valid_global = (global_expert >= 0) & (global_expert < NUM_GLOBAL_EXPERTS)
    safe_global = tl.where(valid_global, global_expert, 0)
    local_expert = tl.load(expert_map_ptr + safe_global, mask=valid_global, other=-1)
    found = valid_global & (local_expert >= 0) & (local_expert < NUM_LOCAL_EXPERTS)
    return found, local_expert


@triton.jit
def _fused_moe_local_rank_g1(
    hidden_ptr,
    w1_ptr,
    cache2_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    stride_hidden_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_cache_route,
    stride_cache_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LANES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
):
    tile_n = tl.program_id(0)
    lane = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
    tile_start = tile_n * BLOCK_SIZE_N
    offs_pair_n = tl.where(
        offs_pair < BLOCK_SIZE_N,
        tile_start + offs_pair,
        INTERMEDIATE_SIZE + tile_start + offs_pair - BLOCK_SIZE_N,
    )
    output_columns = tile_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    valid_row = offs_m == 0

    for local_slot in range(tl.cdiv(8, LANES)):
        found, route, local_expert = _select_local_route(
            topk_ids_ptr,
            expert_map_ptr,
            lane,
            local_slot,
            LANES,
            NUM_GLOBAL_EXPERTS,
            NUM_LOCAL_EXPERTS,
        )
        if found:
            a_ptrs = (
                hidden_ptr + offs_m[:, None] * 0 + offs_k[None, :] * stride_hidden_k
            )
            b_ptrs = (
                w1_ptr
                + local_expert.to(tl.int64) * stride_w1_e
                + offs_k[:, None] * stride_w1_k
                + offs_pair_n[None, :] * stride_w1_n
            )
            pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
            for _ in range(0, HIDDEN_SIZE, BLOCK_SIZE_K):
                a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
                b = tl.load(b_ptrs)
                pair_acc += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_hidden_k
                b_ptrs += BLOCK_SIZE_K * stride_w1_k

            gate_up = tl.trans(
                tl.reshape(pair_acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N)),
                (0, 2, 1),
            )
            gate_acc, up_acc = tl.split(gate_up)
            gate = gate_acc.to(tl.bfloat16).to(tl.float32)
            up = up_acc.to(tl.bfloat16).to(tl.float32)
            gate = tl.minimum(gate, CLAMP_LIMIT)
            up = tl.minimum(tl.maximum(up, -CLAMP_LIMIT), CLAMP_LIMIT)
            activated = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up
            cache_ptrs = (
                cache2_ptr
                + (route.to(tl.int64) + offs_m[:, None]) * stride_cache_route
                + output_columns[None, :] * stride_cache_n
            )
            tl.store(
                cache_ptrs,
                activated.to(tl.bfloat16),
                mask=valid_row[:, None] & (output_columns[None, :] < INTERMEDIATE_SIZE),
            )


@triton.jit
def _fused_moe_local_rank_g2(
    cache2_ptr,
    w2_ptr,
    cache3_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    stride_cache2_route,
    stride_cache2_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_cache3_route,
    stride_cache3_n,
    stride_weights_route,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LANES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
):
    tile_n = tl.program_id(0)
    lane = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    valid_row = offs_m == 0

    for local_slot in range(tl.cdiv(8, LANES)):
        found, route, local_expert = _select_local_route(
            topk_ids_ptr,
            expert_map_ptr,
            lane,
            local_slot,
            LANES,
            NUM_GLOBAL_EXPERTS,
            NUM_LOCAL_EXPERTS,
        )
        if found:
            a_ptrs = (
                cache2_ptr
                + (route.to(tl.int64) + offs_m[:, None]) * stride_cache2_route
                + offs_k[None, :] * stride_cache2_k
            )
            b_ptrs = (
                w2_ptr
                + local_expert.to(tl.int64) * stride_w2_e
                + offs_n[None, :] * stride_w2_n
                + offs_k[:, None] * stride_w2_k
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for _ in range(0, INTERMEDIATE_SIZE, BLOCK_SIZE_K):
                a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_cache2_k
                b_ptrs += BLOCK_SIZE_K * stride_w2_k

            route_weight = tl.load(topk_weights_ptr + route * stride_weights_route)
            weighted = (accumulator * route_weight.to(tl.float32)).to(tl.bfloat16)
            cache3_ptrs = (
                cache3_ptr
                + (route.to(tl.int64) + offs_m[:, None]) * stride_cache3_route
                + offs_n[None, :] * stride_cache3_n
            )
            tl.store(
                cache3_ptrs,
                weighted,
                mask=valid_row[:, None] & (offs_n[None, :] < HIDDEN_SIZE),
            )


@triton.jit
def _fused_moe_chunked_g1(
    hidden_ptr,
    w1_ptr,
    cache2_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    stride_hidden_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_cache_route,
    stride_cache_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CHUNK_TILES: tl.constexpr,
    ROTATE_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
):
    tile_in_chunk = tl.program_id(0)
    chunk_route = tl.program_id(1)
    chunk_index = chunk_route // 8
    route_slot = chunk_route - chunk_index * 8
    if ROTATE_ROUTES:
        route = (route_slot + chunk_index) % 8
    else:
        route = route_slot
    tile_n = chunk_index * CHUNK_TILES + tile_in_chunk
    found, local_expert = _lookup_original_route(
        topk_ids_ptr,
        expert_map_ptr,
        route,
        NUM_GLOBAL_EXPERTS,
        NUM_LOCAL_EXPERTS,
    )
    if found:
        offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
        tile_start = tile_n * BLOCK_SIZE_N
        offs_pair_n = tl.where(
            offs_pair < BLOCK_SIZE_N,
            tile_start + offs_pair,
            INTERMEDIATE_SIZE + tile_start + offs_pair - BLOCK_SIZE_N,
        )
        output_columns = tile_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        valid_row = offs_m == 0
        a_ptrs = hidden_ptr + offs_m[:, None] * 0 + offs_k[None, :] * stride_hidden_k
        b_ptrs = (
            w1_ptr
            + local_expert.to(tl.int64) * stride_w1_e
            + offs_k[:, None] * stride_w1_k
            + offs_pair_n[None, :] * stride_w1_n
        )
        pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
        for _ in range(0, HIDDEN_SIZE, BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
            b = tl.load(b_ptrs)
            pair_acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_hidden_k
            b_ptrs += BLOCK_SIZE_K * stride_w1_k

        gate_up = tl.trans(
            tl.reshape(pair_acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N)), (0, 2, 1)
        )
        gate_acc, up_acc = tl.split(gate_up)
        gate = gate_acc.to(tl.bfloat16).to(tl.float32)
        up = up_acc.to(tl.bfloat16).to(tl.float32)
        gate = tl.minimum(gate, CLAMP_LIMIT)
        up = tl.minimum(tl.maximum(up, -CLAMP_LIMIT), CLAMP_LIMIT)
        activated = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up
        cache_ptrs = (
            cache2_ptr
            + (route.to(tl.int64) + offs_m[:, None]) * stride_cache_route
            + output_columns[None, :] * stride_cache_n
        )
        tl.store(
            cache_ptrs,
            activated.to(tl.bfloat16),
            mask=valid_row[:, None] & (output_columns[None, :] < INTERMEDIATE_SIZE),
        )


@triton.jit
def _fused_moe_chunked_g2(
    cache2_ptr,
    w2_ptr,
    cache3_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    stride_cache2_route,
    stride_cache2_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_cache3_route,
    stride_cache3_n,
    stride_weights_route,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CHUNK_TILES: tl.constexpr,
    ROTATE_ROUTES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
):
    tile_in_chunk = tl.program_id(0)
    chunk_route = tl.program_id(1)
    chunk_index = chunk_route // 8
    route_slot = chunk_route - chunk_index * 8
    if ROTATE_ROUTES:
        route = (route_slot + chunk_index) % 8
    else:
        route = route_slot
    tile_n = chunk_index * CHUNK_TILES + tile_in_chunk
    found, local_expert = _lookup_original_route(
        topk_ids_ptr,
        expert_map_ptr,
        route,
        NUM_GLOBAL_EXPERTS,
        NUM_LOCAL_EXPERTS,
    )
    if found:
        offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        valid_row = offs_m == 0
        a_ptrs = (
            cache2_ptr
            + (route.to(tl.int64) + offs_m[:, None]) * stride_cache2_route
            + offs_k[None, :] * stride_cache2_k
        )
        b_ptrs = (
            w2_ptr
            + local_expert.to(tl.int64) * stride_w2_e
            + offs_n[None, :] * stride_w2_n
            + offs_k[:, None] * stride_w2_k
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in range(0, INTERMEDIATE_SIZE, BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_cache2_k
            b_ptrs += BLOCK_SIZE_K * stride_w2_k

        route_weight = tl.load(topk_weights_ptr + route * stride_weights_route)
        weighted = (accumulator * route_weight.to(tl.float32)).to(tl.bfloat16)
        cache3_ptrs = (
            cache3_ptr
            + (route.to(tl.int64) + offs_m[:, None]) * stride_cache3_route
            + offs_n[None, :] * stride_cache3_n
        )
        tl.store(
            cache3_ptrs,
            weighted,
            mask=valid_row[:, None] & (offs_n[None, :] < HIDDEN_SIZE),
        )


@dataclass(frozen=True)
class Plan:
    g1_block_n: int
    g1_warps: int
    g1_stages: int
    g2_block_n: int
    g2_warps: int
    g2_stages: int


PLANS = {
    "narrow": Plan(16, 2, 3, 64, 4, 4),
    "shared": Plan(32, 4, 3, 128, 4, 4),
    "narrow-g1": Plan(16, 2, 3, 128, 4, 4),
    "narrow-g2": Plan(32, 4, 3, 64, 4, 4),
}


def launch_local_rank(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    expert_map: torch.Tensor,
    cache2: torch.Tensor,
    cache3: torch.Tensor,
    output: torch.Tensor,
    *,
    lanes: int,
    plan: Plan,
) -> torch.Tensor:
    """Launch the benchmark-only M=1 local-rank GEMM pair and EP sum."""
    hidden_size = hidden.shape[1]
    intermediate_size = w2.shape[2]
    if not (
        hidden.shape == (1, 4096)
        and ids.shape == (1, 8)
        and weights.shape == (1, 8)
        and w1.shape == (18, 2 * intermediate_size, 4096)
        and w2.shape == (18, 4096, intermediate_size)
        and cache2.shape == (8, intermediate_size)
        and cache3.shape == (8, 4096)
        and output.shape == hidden.shape
        and intermediate_size == 2048
        and hidden.dtype
        == w1.dtype
        == w2.dtype
        == cache2.dtype
        == cache3.dtype
        == output.dtype
        == torch.bfloat16
        and weights.dtype in (torch.bfloat16, torch.float32)
        and ids.dtype in (torch.int32, torch.int64)
        and expert_map.dtype in (torch.int32, torch.int64)
        and expert_map.numel() == 288
        and lanes in range(1, 9)
    ):
        raise ValueError("local-rank prototype only accepts strict fused-MoE M1/I2048")

    _fused_moe_local_rank_g1[(triton.cdiv(intermediate_size, plan.g1_block_n), lanes)](
        hidden,
        w1,
        cache2,
        ids,
        expert_map,
        hidden.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        cache2.stride(0),
        cache2.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=plan.g1_block_n,
        BLOCK_SIZE_K=128,
        LANES=lanes,
        NUM_GLOBAL_EXPERTS=288,
        NUM_LOCAL_EXPERTS=18,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        CLAMP_LIMIT=10.0,
        num_warps=plan.g1_warps,
        num_stages=plan.g1_stages,
    )
    _fused_moe_local_rank_g2[(triton.cdiv(hidden_size, plan.g2_block_n), lanes)](
        cache2,
        w2,
        cache3,
        weights,
        ids,
        expert_map,
        cache2.stride(0),
        cache2.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        cache3.stride(0),
        cache3.stride(1),
        weights.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=plan.g2_block_n,
        BLOCK_SIZE_K=64,
        LANES=lanes,
        NUM_GLOBAL_EXPERTS=288,
        NUM_LOCAL_EXPERTS=18,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        num_warps=plan.g2_warps,
        num_stages=plan.g2_stages,
    )
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    fm.moe_sum_ep(cache3.view(1, 8, hidden_size), output, ids, expert_map, 18)
    return output


def launch_chunked(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    expert_map: torch.Tensor,
    cache2: torch.Tensor,
    cache3: torch.Tensor,
    output: torch.Tensor,
    *,
    g1_chunk: int,
    g2_chunk: int,
    rotate_routes: bool,
    plan: Plan,
) -> torch.Tensor:
    """Launch route/tile chunk rotation without compacting the route mask."""
    hidden_size = hidden.shape[1]
    intermediate_size = w2.shape[2]
    g1_tiles = triton.cdiv(intermediate_size, plan.g1_block_n)
    g2_tiles = triton.cdiv(hidden_size, plan.g2_block_n)
    if g1_tiles % g1_chunk or g2_tiles % g2_chunk:
        raise ValueError("chunk sizes must divide the selected plan's tile counts")

    _fused_moe_chunked_g1[(g1_chunk, g1_tiles // g1_chunk * 8)](
        hidden,
        w1,
        cache2,
        ids,
        expert_map,
        hidden.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        cache2.stride(0),
        cache2.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=plan.g1_block_n,
        BLOCK_SIZE_K=128,
        CHUNK_TILES=g1_chunk,
        ROTATE_ROUTES=rotate_routes,
        NUM_GLOBAL_EXPERTS=288,
        NUM_LOCAL_EXPERTS=18,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        CLAMP_LIMIT=10.0,
        num_warps=plan.g1_warps,
        num_stages=plan.g1_stages,
    )
    _fused_moe_chunked_g2[(g2_chunk, g2_tiles // g2_chunk * 8)](
        cache2,
        w2,
        cache3,
        weights,
        ids,
        expert_map,
        cache2.stride(0),
        cache2.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        cache3.stride(0),
        cache3.stride(1),
        weights.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=plan.g2_block_n,
        BLOCK_SIZE_K=64,
        CHUNK_TILES=g2_chunk,
        ROTATE_ROUTES=rotate_routes,
        NUM_GLOBAL_EXPERTS=288,
        NUM_LOCAL_EXPERTS=18,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        num_warps=plan.g2_warps,
        num_stages=plan.g2_stages,
    )
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    fm.moe_sum_ep(cache3.view(1, 8, hidden_size), output, ids, expert_map, 18)
    return output


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


def elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / replays)


def make_remote_routes(shard_begin: int, dtype: torch.dtype) -> torch.Tensor:
    flat = torch.arange(8, device="cuda", dtype=dtype)
    remote = flat + (flat >= shard_begin).to(dtype) * 18
    return remote.view(1, 8)


def set_route_mask(
    ids: torch.Tensor, remote_ids: torch.Tensor, shard_begin: int, route_mask: int
) -> None:
    ids.copy_(remote_ids)
    local_expert = 0
    for position in range(8):
        if route_mask & (1 << position):
            ids[0, position] = shard_begin + local_expert
            local_expert += 1


def scenario_masks() -> dict[str, int]:
    scenarios = {"no-local": 0}
    scenarios.update({f"singleton-p{position}": 1 << position for position in range(8)})
    scenarios.update(
        {
            "local2-prefix": 0b00000011,
            "local2-spread": 0b00010001,
            "local4-prefix": 0b00001111,
            "local4-spread": 0b01010101,
            "all-local": 0b11111111,
        }
    )
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule", choices=("local-rank", "chunked"), default="local-rank"
    )
    parser.add_argument(
        "--candidate-source",
        choices=("prototype", "production"),
        default="prototype",
        help="Use the benchmark-only kernel or the integrated production path.",
    )
    parser.add_argument("--lanes", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--plan", choices=tuple(PLANS), default="narrow")
    parser.add_argument("--g1-chunk", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--g2-chunk", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--no-rotate-routes", action="store_true")
    parser.add_argument("--ep-rank", type=int, choices=range(16), default=7)
    parser.add_argument("--ids-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--map-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument(
        "--router-weight-dtype",
        "--topk-weights-dtype",
        dest="router_weight_dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help="Dtype of normalized top-k router weights (default: fp32).",
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--replays", type=int, default=1500)
    parser.add_argument("--audit-route-masks", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.replays <= 0:
        raise ValueError("rounds and replays must be positive")
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)
    shard_begin = args.ep_rank * 18
    ids_dtype = torch.int32 if args.ids_dtype == "int32" else torch.int64
    map_dtype = torch.int32 if args.map_dtype == "int32" else torch.int64
    router_weight_dtype = (
        torch.bfloat16 if args.router_weight_dtype == "bf16" else torch.float32
    )
    hidden = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
    w1 = torch.empty((18, 4096, 4096), device="cuda", dtype=torch.bfloat16)
    w1.normal_(std=4096**-0.5)
    w2 = torch.empty((18, 4096, 2048), device="cuda", dtype=torch.bfloat16)
    w2.normal_(std=2048**-0.5)
    remote_ids = make_remote_routes(shard_begin, ids_dtype)
    ids = remote_ids.clone()
    ids[0, 0] = shard_begin
    weights = torch.rand((1, 8), device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum(-1, keepdim=True)).to(router_weight_dtype)
    expert_map = torch.full((288,), -1, device="cuda", dtype=map_dtype)
    expert_map[shard_begin : shard_begin + 18] = torch.arange(
        18, device="cuda", dtype=map_dtype
    )

    reference_cache13 = torch.empty(8 * 4096, device="cuda", dtype=torch.bfloat16)
    reference_cache2 = torch.empty(8 * 2048, device="cuda", dtype=torch.bfloat16)
    reference_output = reference_cache2[:4096].view(1, 4096)
    candidate_cache2 = torch.empty((8, 2048), device="cuda", dtype=torch.bfloat16)
    candidate_cache3 = torch.empty((8, 4096), device="cuda", dtype=torch.bfloat16)
    candidate_output = candidate_cache2.view(-1)[:4096].view(1, 4096)

    alignment_calls = 0
    dispatches: list[dict[str, object]] = []
    original_naive = fm._should_use_ep_naive_route
    original_local_rank = fm._should_use_ep_m1_i2048_local_rank
    original_route_block = fm._should_use_ep_route_block
    original_align = fm.moe_align_block_size
    original_dispatch = fm.dispatch_fused_moe_kernel

    def force_true(*_args, **_kwargs):
        return True

    def force_false(*_args, **_kwargs):
        return False

    def align_spy(*align_args, **align_kwargs):
        nonlocal alignment_calls
        alignment_calls += 1
        return original_align(*align_args, **align_kwargs)

    def dispatch_spy(*dispatch_args, **dispatch_kwargs):
        dispatches.append(
            {
                "config": dict(dispatch_args[12]),
                "sorted_token_ids_is_none": dispatch_args[7] is None,
                "expert_map": dispatch_kwargs.get("expert_map") is expert_map,
                "skip_invalid": dispatch_kwargs.get("skip_invalid_experts", False),
            }
        )
        return original_dispatch(*dispatch_args, **dispatch_kwargs)

    def reference_op():
        return fm.fused_experts_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            global_num_experts=288,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=reference_output,
            intermediate_cache13=reference_cache13,
            intermediate_cache2=reference_cache2,
        )

    plan = PLANS[args.plan]

    def candidate_op():
        if args.candidate_source == "production":
            return fm.fused_experts_impl(
                hidden,
                w1,
                w2,
                weights,
                ids,
                global_num_experts=288,
                expert_map=expert_map,
                gemm1_clamp_limit=10.0,
                output=candidate_output,
                intermediate_cache13=candidate_cache3,
                intermediate_cache2=candidate_cache2,
            )
        common = (
            hidden,
            w1,
            w2,
            weights,
            ids,
            expert_map,
            candidate_cache2,
            candidate_cache3,
            candidate_output,
        )
        if args.schedule == "local-rank":
            return launch_local_rank(*common, lanes=args.lanes, plan=plan)
        return launch_chunked(
            *common,
            g1_chunk=args.g1_chunk,
            g2_chunk=args.g2_chunk,
            rotate_routes=not args.no_rotate_routes,
            plan=plan,
        )

    fm._should_use_ep_naive_route = force_true
    fm._should_use_ep_m1_i2048_local_rank = force_false
    fm._should_use_ep_route_block = force_false
    fm.moe_align_block_size = align_spy
    fm.dispatch_fused_moe_kernel = dispatch_spy
    try:
        reference_graph, reference_eager, reference_graph_output = capture(reference_op)
    finally:
        fm._should_use_ep_naive_route = original_naive
        fm._should_use_ep_m1_i2048_local_rank = original_local_rank
        fm._should_use_ep_route_block = original_route_block
        fm.moe_align_block_size = original_align
        fm.dispatch_fused_moe_kernel = original_dispatch
    candidate_graph, candidate_eager, candidate_graph_output = capture(candidate_op)

    graph_capture_bitwise = {
        "reference": bool(torch.equal(reference_eager, reference_graph_output)),
        "candidate": bool(torch.equal(candidate_eager, candidate_graph_output)),
        "candidate_vs_reference": bool(torch.equal(reference_eager, candidate_eager)),
    }
    policies = ("reference", "candidate")
    graphs = {"reference": reference_graph, "candidate": candidate_graph}
    masks = (
        {f"mask-{route_mask:03d}": route_mask for route_mask in range(256)}
        if args.audit_route_masks
        else scenario_masks()
    )
    results = []
    for scenario, route_mask in masks.items():
        set_route_mask(ids, remote_ids, shard_begin, route_mask)
        for _ in range(30):
            reference_graph.replay()
            candidate_graph.replay()
        torch.cuda.synchronize()
        bitwise = bool(torch.equal(reference_output, candidate_output))
        samples = {name: [] for name in policies}
        for round_idx in range(args.rounds):
            order = policies if round_idx % 2 == 0 else tuple(reversed(policies))
            bracket = (*order, *reversed(order))
            values = {name: [] for name in policies}
            for name in bracket:
                values[name].append(elapsed_ms(graphs[name], args.replays))
            for name in policies:
                samples[name].append(statistics.mean(values[name]))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        paired_reductions = [
            100.0 * (1.0 - candidate / reference)
            for reference, candidate in zip(samples["reference"], samples["candidate"])
        ]
        local_count = route_mask.bit_count()
        count_probability = (
            math.comb(18, local_count)
            * math.comb(270, 8 - local_count)
            / math.comb(288, 8)
        )
        mask_probability = count_probability / math.comb(8, local_count)
        results.append(
            {
                "scenario": scenario,
                "route_mask": route_mask,
                "local_positions": [
                    position for position in range(8) if route_mask & (1 << position)
                ],
                "local_count": local_count,
                "probability": mask_probability,
                "median_ms": medians,
                "reduction_pct": 100.0
                * (1.0 - medians["candidate"] / medians["reference"]),
                "paired_reduction_median_pct": statistics.median(paired_reductions),
                "positive_rounds": sum(value > 0.0 for value in paired_reductions),
                "bitwise": bitwise,
            }
        )

    aggregate: dict[str, object] = {
        "scenario_ratio_of_means_reduction_pct": 100.0
        * (
            1.0
            - statistics.mean(item["median_ms"]["candidate"] for item in results)
            / statistics.mean(item["median_ms"]["reference"] for item in results)
        ),
        "positive_scenarios": sum(item["reduction_pct"] > 0.0 for item in results),
        "total_scenarios": len(results),
        "worst_scenario": min(results, key=lambda item: item["reduction_pct"]),
        "all_bitwise": all(item["bitwise"] for item in results),
    }
    if args.audit_route_masks:
        expected_reference = sum(
            item["probability"] * item["median_ms"]["reference"] for item in results
        )
        expected_candidate = sum(
            item["probability"] * item["median_ms"]["candidate"] for item in results
        )
        aggregate.update(
            {
                "expected_reference_ms": expected_reference,
                "expected_candidate_ms": expected_candidate,
                "hypergeometric_expected_reduction_pct": 100.0
                * (1.0 - expected_candidate / expected_reference),
            }
        )
        by_local_count = {}
        for local_count in range(9):
            count_results = [
                item for item in results if item["local_count"] == local_count
            ]
            mean_reference = statistics.mean(
                item["median_ms"]["reference"] for item in count_results
            )
            mean_candidate = statistics.mean(
                item["median_ms"]["candidate"] for item in count_results
            )
            by_local_count[str(local_count)] = {
                "masks": len(count_results),
                "probability": sum(item["probability"] for item in count_results),
                "mean_reference_ms": mean_reference,
                "mean_candidate_ms": mean_candidate,
                "ratio_of_means_reduction_pct": 100.0
                * (1.0 - mean_candidate / mean_reference),
                "positive_masks": sum(
                    item["reduction_pct"] > 0.0 for item in count_results
                ),
                "worst_mask_reduction_pct": min(
                    item["reduction_pct"] for item in count_results
                ),
            }
        aggregate["by_local_count"] = by_local_count

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": "M1_E288_LE18_H4096_I2048_top8_BF16_clamp10",
                "ep_rank": args.ep_rank,
                "ids_dtype": args.ids_dtype,
                "map_dtype": args.map_dtype,
                "router_weight_dtype": str(weights.dtype),
                "candidate": {
                    "source": args.candidate_source,
                    "schedule": args.schedule,
                    "lanes": args.lanes if args.schedule == "local-rank" else None,
                    "g1_chunk": args.g1_chunk if args.schedule == "chunked" else None,
                    "g2_chunk": args.g2_chunk if args.schedule == "chunked" else None,
                    "rotate_routes": (
                        not args.no_rotate_routes
                        if args.schedule == "chunked"
                        else None
                    ),
                    "plan": args.plan,
                    "config": plan.__dict__,
                    "launches": "GEMM1 + GEMM2 + moe_sum_ep (no alignment)",
                },
                "reference": "forced production raw direct shared plan",
                "reference_alignment_calls_during_capture": alignment_calls,
                "reference_dispatches": dispatches,
                "graph_capture_bitwise": graph_capture_bitwise,
                "rounds": args.rounds,
                "replays": args.replays,
                "timing_order": "ABBA/BAAB alternating",
                "aggregate": aggregate,
                "results": [] if args.summary_only else results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
