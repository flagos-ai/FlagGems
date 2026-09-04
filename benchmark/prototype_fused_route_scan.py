#!/usr/bin/env python3
"""Prototype a route-prefix scan fused into fused-MoE M=1 GEMM1.

This benchmark is intentionally disconnected from production dispatch.  It
compares the current direct-route ``fused_experts_impl`` policy with a strict
BF16/H4096/I2048/EP16/top-8 candidate that removes the standalone route-block
launch without serializing useful routes:

* the GEMM1 grid has a configurable 1/2/4/8 local-rank shards; every CTA safely
  maps top-8 global IDs, performs one in-register prefix scan, and processes
  interleaved compact local ranks owned by its shard;
* GEMM1's first N tile also publishes the selected route/expert metadata;
* GEMM2 consumes that metadata, retaining one independent CTA row per local
  route and writing weighted BF16 route rows to cache3;
* ``moe_sum_ep`` runs after GEMM2, so output may alias cache2 exactly as in the
  modular vLLM workspace layout.

All grids and branches depend only on device data, making a single captured
CUDA Graph valid across route-density and route-position changes.  Bounds are
checked before every expert-map lookup; negative, out-of-range, and very large
int64 expert IDs are therefore safe.
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


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


@triton.jit
def _route_rank_gemm1(
    hidden_ptr,
    w1_ptr,
    cache2_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    selected_routes_ptr,
    selected_experts_ptr,
    local_count_ptr,
    stride_hidden_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_cache_route,
    stride_cache_n,
    stride_ids_route,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
):
    """Select a local route rank and compute paired gate/up plus SwiGLU."""
    route_rank = tl.program_id(0)
    tile_n = tl.program_id(1)

    route_offsets = tl.arange(0, TOPK)
    global_experts_raw = tl.load(topk_ids_ptr + route_offsets * stride_ids_route)
    valid_global = (global_experts_raw >= 0) & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    # Preserve int64 until after the bounds check.  This prevents a large
    # invalid ID from wrapping into expert_map's valid range.
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global, mask=valid_global, other=-1
    )
    local_mask = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    local_ranks = tl.cumsum(local_mask.to(tl.int32), axis=0) - 1
    selected_mask = local_mask & (local_ranks == route_rank)
    local_count = tl.sum(local_mask.to(tl.int32), axis=0)
    has_route = route_rank < local_count
    selected_route = tl.sum(
        tl.where(selected_mask, route_offsets, 0).to(tl.int32), axis=0
    )
    selected_expert = tl.sum(
        tl.where(selected_mask, local_experts_raw, 0).to(tl.int32), axis=0
    )

    # Kernel ordering makes this tiny metadata table visible to GEMM2 without
    # a host sync.  One writer per address avoids atomics and races.
    if tile_n == 0:
        tl.store(selected_routes_ptr + route_rank, selected_route, mask=has_route)
        tl.store(selected_experts_ptr + route_rank, selected_expert, mask=has_route)
        if route_rank == 0:
            tl.store(local_count_ptr, local_count)

    if has_route:
        offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        valid_row = offs_m == 0
        tile_start = tile_n * BLOCK_SIZE_N
        offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
        offs_pair_n = tl.where(
            offs_pair < BLOCK_SIZE_N,
            tile_start + offs_pair,
            INTERMEDIATE_SIZE + tile_start + offs_pair - BLOCK_SIZE_N,
        )
        output_columns = tile_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        a_ptrs = hidden_ptr + offs_m[:, None] * 0 + offs_k[None, :] * stride_hidden_k
        b_ptrs = (
            w1_ptr
            + selected_expert.to(tl.int64) * stride_w1_e
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
        # Match production's explicit GEMM1 BF16 materialization boundary.
        gate = gate_acc.to(tl.bfloat16).to(tl.float32)
        up = up_acc.to(tl.bfloat16).to(tl.float32)
        gate = tl.minimum(gate, CLAMP_LIMIT)
        up = tl.minimum(tl.maximum(up, -CLAMP_LIMIT), CLAMP_LIMIT)
        activated = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up

        cache_ptrs = (
            cache2_ptr
            + (selected_route.to(tl.int64) + offs_m[:, None]) * stride_cache_route
            + output_columns[None, :] * stride_cache_n
        )
        tl.store(
            cache_ptrs,
            activated.to(tl.bfloat16),
            mask=valid_row[:, None] & (output_columns[None, :] < INTERMEDIATE_SIZE),
        )


@triton.jit
def _metadata_gemm2(
    cache2_ptr,
    w2_ptr,
    cache3_ptr,
    topk_weights_ptr,
    selected_routes_ptr,
    selected_experts_ptr,
    local_count_ptr,
    stride_cache2_route,
    stride_cache2_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_cache3_route,
    stride_cache3_n,
    stride_weights_route,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Consume metadata produced by GEMM1 and write weighted route rows."""
    route_rank = tl.program_id(0)
    tile_n = tl.program_id(1)
    local_count = tl.load(local_count_ptr)
    if route_rank < local_count:
        selected_route = tl.load(selected_routes_ptr + route_rank)
        selected_expert = tl.load(selected_experts_ptr + route_rank)
        valid_metadata = (
            (selected_route >= 0)
            & (selected_route < TOPK)
            & (selected_expert >= 0)
            & (selected_expert < NUM_LOCAL_EXPERTS)
        )
        if valid_metadata:
            offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            valid_row = offs_m == 0
            a_ptrs = (
                cache2_ptr
                + (selected_route.to(tl.int64) + offs_m[:, None]) * stride_cache2_route
                + offs_k[None, :] * stride_cache2_k
            )
            b_ptrs = (
                w2_ptr
                + selected_expert.to(tl.int64) * stride_w2_e
                + offs_n[None, :] * stride_w2_n
                + offs_k[:, None] * stride_w2_k
            )
            route_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for _ in range(0, INTERMEDIATE_SIZE, BLOCK_SIZE_K):
                a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
                b = tl.load(b_ptrs)
                route_acc += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_cache2_k
                b_ptrs += BLOCK_SIZE_K * stride_w2_k

            route_weight = tl.load(
                topk_weights_ptr + selected_route.to(tl.int64) * stride_weights_route
            )
            weighted_bf16 = (route_acc * route_weight.to(tl.float32)).to(tl.bfloat16)
            cache3_ptrs = (
                cache3_ptr
                + (selected_route.to(tl.int64) + offs_m[:, None]) * stride_cache3_route
                + offs_n[None, :] * stride_cache3_n
            )
            tl.store(
                cache3_ptrs,
                weighted_bf16,
                mask=valid_row[:, None] & (offs_n[None, :] < HIDDEN_SIZE),
            )


@triton.jit
def _local_rank_sharded_gemm1(
    hidden_ptr,
    w1_ptr,
    cache2_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    selected_routes_ptr,
    selected_experts_ptr,
    local_count_ptr,
    stride_hidden_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_cache_route,
    stride_cache_n,
    stride_ids_route,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    ROUTE_SHARDS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
):
    """Scan once, then process interleaved *local ranks* in each CTA shard.

    Unlike raw route-sharding, work distribution depends only on local-route
    density. Moving the same singleton from p0 to p7 therefore cannot create a
    shard collision or move the useful CTA row.
    """
    # Keep every N tile for one local-rank shard contiguous in launch order.
    # Reversing these axes interleaves empty and useful shards and recreates the
    # same wave-position sensitivity that this schedule is meant to remove.
    tile_n = tl.program_id(0)
    shard_idx = tl.program_id(1)
    route_offsets = tl.arange(0, TOPK)
    global_experts_raw = tl.load(topk_ids_ptr + route_offsets * stride_ids_route)
    valid_global = (global_experts_raw >= 0) & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global, mask=valid_global, other=-1
    )
    local_mask = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    local_ranks = tl.cumsum(local_mask.to(tl.int32), axis=0) - 1
    local_count = tl.sum(local_mask.to(tl.int32), axis=0)
    if tile_n == 0:
        if shard_idx == 0:
            tl.store(local_count_ptr, local_count)

    for route_slot in range(0, TOPK // ROUTE_SHARDS):
        selected_rank = shard_idx + route_slot * ROUTE_SHARDS
        has_route = selected_rank < local_count
        selected_mask = local_mask & (local_ranks == selected_rank)
        selected_route = tl.sum(
            tl.where(selected_mask, route_offsets, 0).to(tl.int32), axis=0
        )
        selected_expert = tl.sum(
            tl.where(selected_mask, local_experts_raw, 0).to(tl.int32), axis=0
        )
        if tile_n == 0:
            tl.store(
                selected_routes_ptr + selected_rank,
                selected_route,
                mask=has_route,
            )
            tl.store(
                selected_experts_ptr + selected_rank,
                selected_expert,
                mask=has_route,
            )

        if has_route:
            offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            valid_row = offs_m == 0
            tile_start = tile_n * BLOCK_SIZE_N
            offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
            offs_pair_n = tl.where(
                offs_pair < BLOCK_SIZE_N,
                tile_start + offs_pair,
                INTERMEDIATE_SIZE + tile_start + offs_pair - BLOCK_SIZE_N,
            )
            output_columns = tile_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            a_ptrs = (
                hidden_ptr + offs_m[:, None] * 0 + offs_k[None, :] * stride_hidden_k
            )
            b_ptrs = (
                w1_ptr
                + selected_expert.to(tl.int64) * stride_w1_e
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
                + (selected_route.to(tl.int64) + offs_m[:, None]) * stride_cache_route
                + output_columns[None, :] * stride_cache_n
            )
            tl.store(
                cache_ptrs,
                activated.to(tl.bfloat16),
                mask=valid_row[:, None] & (output_columns[None, :] < INTERMEDIATE_SIZE),
            )


@triton.jit
def _local_rank_sharded_gemm2(
    cache2_ptr,
    w2_ptr,
    cache3_ptr,
    topk_weights_ptr,
    selected_routes_ptr,
    selected_experts_ptr,
    local_count_ptr,
    stride_cache2_route,
    stride_cache2_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_cache3_route,
    stride_cache3_n,
    stride_weights_route,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    ROUTE_SHARDS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Consume local-rank shards while preserving independent route rows."""
    tile_n = tl.program_id(0)
    shard_idx = tl.program_id(1)
    local_count = tl.load(local_count_ptr)
    for route_slot in range(0, TOPK // ROUTE_SHARDS):
        selected_rank = shard_idx + route_slot * ROUTE_SHARDS
        if selected_rank < local_count:
            selected_route = tl.load(selected_routes_ptr + selected_rank)
            selected_expert = tl.load(selected_experts_ptr + selected_rank)
            valid_metadata = (
                (selected_route >= 0)
                & (selected_route < TOPK)
                & (selected_expert >= 0)
                & (selected_expert < NUM_LOCAL_EXPERTS)
            )
            if valid_metadata:
                offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
                offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                valid_row = offs_m == 0
                a_ptrs = (
                    cache2_ptr
                    + (selected_route.to(tl.int64) + offs_m[:, None])
                    * stride_cache2_route
                    + offs_k[None, :] * stride_cache2_k
                )
                b_ptrs = (
                    w2_ptr
                    + selected_expert.to(tl.int64) * stride_w2_e
                    + offs_n[None, :] * stride_w2_n
                    + offs_k[:, None] * stride_w2_k
                )
                route_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for _ in range(0, INTERMEDIATE_SIZE, BLOCK_SIZE_K):
                    a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
                    b = tl.load(b_ptrs)
                    route_acc += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K * stride_cache2_k
                    b_ptrs += BLOCK_SIZE_K * stride_w2_k

                route_weight = tl.load(
                    topk_weights_ptr
                    + selected_route.to(tl.int64) * stride_weights_route
                )
                weighted_bf16 = (route_acc * route_weight.to(tl.float32)).to(
                    tl.bfloat16
                )
                cache3_ptrs = (
                    cache3_ptr
                    + (selected_route.to(tl.int64) + offs_m[:, None])
                    * stride_cache3_route
                    + offs_n[None, :] * stride_cache3_n
                )
                tl.store(
                    cache3_ptrs,
                    weighted_bf16,
                    mask=valid_row[:, None] & (offs_n[None, :] < HIDDEN_SIZE),
                )


def launch_fused_route_scan(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    expert_map: torch.Tensor,
    cache2: torch.Tensor,
    cache3: torch.Tensor,
    output: torch.Tensor,
    selected_routes: torch.Tensor,
    selected_experts: torch.Tensor,
    local_count: torch.Tensor,
    g1: KernelConfig,
    g2: KernelConfig,
    route_shards: int,
) -> torch.Tensor:
    """Launch the strict M=1 prototype; all routing decisions stay on device."""
    global_e, local_e, topk, h, intermediate = 288, 18, 8, 4096, 2048
    valid = (
        hidden.dtype
        == w1.dtype
        == w2.dtype
        == weights.dtype
        == cache2.dtype
        == cache3.dtype
        == output.dtype
        == torch.bfloat16
        and ids.dtype in (torch.int32, torch.int64)
        and expert_map.dtype in (torch.int32, torch.int64)
        and hidden.shape == (1, h)
        and w1.shape == (local_e, 2 * intermediate, h)
        and w2.shape == (local_e, h, intermediate)
        and weights.shape == (1, topk)
        and ids.shape == (1, topk)
        and expert_map.shape == (global_e,)
        and cache2.shape == (topk, intermediate)
        and cache3.shape == (topk, h)
        and output.shape == hidden.shape
        and selected_routes.shape == (topk,)
        and selected_experts.shape == (topk,)
        and local_count.shape == (1,)
        and selected_routes.dtype
        == selected_experts.dtype
        == local_count.dtype
        == torch.int32
        and all(
            tensor.is_cuda and tensor.device == hidden.device and tensor.is_contiguous()
            for tensor in (
                w1,
                w2,
                weights,
                ids,
                expert_map,
                cache2,
                cache3,
                output,
                selected_routes,
                selected_experts,
                local_count,
            )
        )
        and g1.block_m == g2.block_m == 16
        and h % g1.block_k == 0
        and intermediate % g2.block_k == 0
        and intermediate % g1.block_n == 0
        and h % g2.block_n == 0
        and route_shards in (1, 2, 4, 8)
    )
    if not valid:
        raise ValueError(
            "fused-route-scan prototype only accepts strict fused-MoE M=1/I2048"
        )
    if output.data_ptr() != cache2.data_ptr():
        raise ValueError("prototype requires the plugin-style output/cache2 alias")

    _local_rank_sharded_gemm1[(triton.cdiv(intermediate, g1.block_n), route_shards)](
        hidden,
        w1,
        cache2,
        ids,
        expert_map,
        selected_routes,
        selected_experts,
        local_count,
        hidden.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        cache2.stride(0),
        cache2.stride(1),
        ids.stride(1),
        HIDDEN_SIZE=h,
        INTERMEDIATE_SIZE=intermediate,
        TOPK=topk,
        NUM_GLOBAL_EXPERTS=global_e,
        NUM_LOCAL_EXPERTS=local_e,
        ROUTE_SHARDS=route_shards,
        BLOCK_SIZE_M=g1.block_m,
        BLOCK_SIZE_N=g1.block_n,
        BLOCK_SIZE_K=g1.block_k,
        CLAMP_LIMIT=10.0,
        num_warps=g1.num_warps,
        num_stages=g1.num_stages,
    )
    _local_rank_sharded_gemm2[(triton.cdiv(h, g2.block_n), route_shards)](
        cache2,
        w2,
        cache3,
        weights,
        selected_routes,
        selected_experts,
        local_count,
        cache2.stride(0),
        cache2.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        cache3.stride(0),
        cache3.stride(1),
        weights.stride(1),
        HIDDEN_SIZE=h,
        INTERMEDIATE_SIZE=intermediate,
        TOPK=topk,
        NUM_LOCAL_EXPERTS=local_e,
        ROUTE_SHARDS=route_shards,
        BLOCK_SIZE_M=g2.block_m,
        BLOCK_SIZE_N=g2.block_n,
        BLOCK_SIZE_K=g2.block_k,
        num_warps=g2.num_warps,
        num_stages=g2.num_stages,
    )
    fm = importlib.import_module("flag_gems.fused.fused_moe")
    fm.moe_sum_ep(cache3.view(1, topk, h), output, ids, expert_map, local_e)
    return output


def _capture(fn):
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


def _elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / replays)


def _remote_ids(global_e, local_e, shard_begin, dtype):
    offsets = torch.arange(8, device="cuda", dtype=torch.int64)
    remote = offsets.remainder(global_e - local_e)
    remote += (remote >= shard_begin).to(torch.int64) * local_e
    return remote.to(dtype).view(1, 8)


def _route_mask_ids(mask, global_e, local_e, shard_begin, dtype):
    ids = _remote_ids(global_e, local_e, shard_begin, dtype)
    local_positions = [position for position in range(8) if mask & (1 << position)]
    for local_expert, position in enumerate(local_positions):
        ids[0, position] = shard_begin + local_expert
    return ids


def _invalid_ids(global_e, local_e, shard_begin, dtype):
    values = [
        -1,
        global_e,
        (1 << 40) if dtype == torch.int64 else torch.iinfo(torch.int32).max,
        shard_begin,
        0,
        shard_begin + local_e - 1,
        global_e + 17,
        shard_begin + local_e,
    ]
    return torch.tensor(values, device="cuda", dtype=dtype).view(1, 8)


def _repeated_local_ids(global_e, local_e, shard_begin, dtype):
    ids = _remote_ids(global_e, local_e, shard_begin, dtype)
    ids[0, 1] = shard_begin + 3
    ids[0, 6] = shard_begin + 3
    return ids


def _safe_local_count(ids, expert_map, global_e, local_e):
    valid_global = (ids >= 0) & (ids < global_e)
    safe = torch.where(valid_global, ids, torch.zeros_like(ids)).to(torch.int64)
    mapped = expert_map[safe]
    return int((valid_global & (mapped >= 0) & (mapped < local_e)).sum().item())


def _measure_pair(graphs, rounds, replays):
    names = tuple(graphs)
    samples = {name: [] for name in names}
    for round_idx in range(rounds):
        order = names if round_idx % 2 == 0 else tuple(reversed(names))
        bracket = (*order, *reversed(order))
        round_values = {name: [] for name in names}
        for name in bracket:
            round_values[name].append(_elapsed_ms(graphs[name], replays))
        for name in names:
            samples[name].append(statistics.mean(round_values[name]))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    reductions = [
        100.0 * (1.0 - candidate / reference)
        for reference, candidate in zip(samples["direct"], samples["fused_scan"])
    ]
    return {
        "median_ms": medians,
        "reduction_pct": 100.0 * (1.0 - medians["fused_scan"] / medians["direct"]),
        "positive_rounds": sum(value > 0 for value in reductions),
        "total_rounds": len(reductions),
        "paired_reduction_median_pct": statistics.median(reductions),
        "samples_ms": samples,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-rank", type=int, default=7)
    parser.add_argument("--ids-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--g1-block-n", type=int, choices=(16, 32), default=16)
    parser.add_argument("--g1-warps", type=int, choices=(2, 4), default=2)
    parser.add_argument("--g1-stages", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--g2-block-n", type=int, choices=(64, 128), default=64)
    parser.add_argument("--g2-warps", type=int, choices=(2, 4), default=4)
    parser.add_argument("--g2-stages", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument(
        "--route-shards",
        type=int,
        choices=(1, 2, 4, 8),
        default=4,
        help=(
            "CTA rows over compact local-route ranks; each row handles "
            "interleaved ranks dynamically."
        ),
    )
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--replays", type=int, default=1500)
    parser.add_argument(
        "--audit-route-masks",
        action="store_true",
        help="Measure all 256 M=1 local/remote masks with EP16 probabilities.",
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.ep_rank < 16:
        raise ValueError("ep-rank must be in [0, 16)")
    if args.rounds <= 0 or args.replays <= 0:
        raise ValueError("rounds and replays must be positive")

    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)
    global_e, local_e, topk, h, intermediate = 288, 18, 8, 4096, 2048
    shard_begin = args.ep_rank * local_e
    ids_dtype = torch.int32 if args.ids_dtype == "int32" else torch.int64
    g1 = KernelConfig(16, args.g1_block_n, 128, args.g1_warps, args.g1_stages)
    g2 = KernelConfig(16, args.g2_block_n, 64, args.g2_warps, args.g2_stages)

    hidden = torch.randn((1, h), device="cuda", dtype=torch.bfloat16)
    w1 = torch.empty(
        (local_e, 2 * intermediate, h), device="cuda", dtype=torch.bfloat16
    ).normal_(std=h**-0.5)
    w2 = torch.empty(
        (local_e, h, intermediate), device="cuda", dtype=torch.bfloat16
    ).normal_(std=intermediate**-0.5)
    weights = torch.rand((1, topk), device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    ids = _route_mask_ids(1, global_e, local_e, shard_begin, ids_dtype)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device="cuda", dtype=torch.int32
    )

    direct_cache13 = torch.empty(
        topk * max(2 * intermediate, h), device="cuda", dtype=torch.bfloat16
    )
    direct_cache2 = torch.empty(
        topk * intermediate, device="cuda", dtype=torch.bfloat16
    )
    direct_output = direct_cache2[:h].view(1, h)
    scan_cache13 = torch.empty_like(direct_cache13)
    scan_cache2_flat = torch.empty_like(direct_cache2)
    scan_cache2 = scan_cache2_flat.view(topk, intermediate)
    scan_cache3 = scan_cache13[: topk * h].view(topk, h)
    scan_output = scan_cache2_flat[:h].view(1, h)
    selected_routes = torch.empty(topk, device="cuda", dtype=torch.int32)
    selected_experts = torch.empty_like(selected_routes)
    local_count = torch.empty(1, device="cuda", dtype=torch.int32)

    def direct_op():
        return fm.fused_experts_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=direct_output,
            intermediate_cache13=direct_cache13,
            intermediate_cache2=direct_cache2,
        )

    def fused_scan_op():
        return launch_fused_route_scan(
            hidden,
            w1,
            w2,
            weights,
            ids,
            expert_map,
            scan_cache2,
            scan_cache3,
            scan_output,
            selected_routes,
            selected_experts,
            local_count,
            g1,
            g2,
            args.route_shards,
        )

    original_naive_gate = fm._should_use_ep_naive_route
    original_route_block_gate = fm._should_use_ep_route_block
    original_local_rank_gate = getattr(fm, "_should_use_ep_m1_i2048_local_rank", None)
    try:
        # Pin the reference to raw direct routing even if production policy
        # changes later.  The benchmark never mutates production source/defaults.
        if original_local_rank_gate is not None:
            fm._should_use_ep_m1_i2048_local_rank = lambda *_args, **_kwargs: False
        fm._should_use_ep_naive_route = lambda *_args, **_kwargs: True
        fm._should_use_ep_route_block = lambda *_args, **_kwargs: False
        direct_graph, direct_eager, direct_graph_output = _capture(direct_op)
    finally:
        fm._should_use_ep_naive_route = original_naive_gate
        fm._should_use_ep_route_block = original_route_block_gate
        if original_local_rank_gate is not None:
            fm._should_use_ep_m1_i2048_local_rank = original_local_rank_gate
    scan_graph, scan_eager, scan_graph_output = _capture(fused_scan_op)
    graphs = {"direct": direct_graph, "fused_scan": scan_graph}
    initial_capture_bitwise = {
        "direct_eager_graph": bool(torch.equal(direct_eager, direct_graph_output)),
        "scan_eager_graph": bool(torch.equal(scan_eager, scan_graph_output)),
        "direct_scan": bool(torch.equal(direct_eager, scan_eager)),
    }

    for _ in range(50):
        direct_graph.replay()
        scan_graph.replay()
    torch.cuda.synchronize()

    transitions = []
    transition_inputs = (
        ("p0", _route_mask_ids(1 << 0, global_e, local_e, shard_begin, ids_dtype)),
        ("p7", _route_mask_ids(1 << 7, global_e, local_e, shard_begin, ids_dtype)),
        ("none", _route_mask_ids(0, global_e, local_e, shard_begin, ids_dtype)),
        (
            "local4",
            _route_mask_ids(0b10101010, global_e, local_e, shard_begin, ids_dtype),
        ),
        (
            "repeated_local",
            _repeated_local_ids(global_e, local_e, shard_begin, ids_dtype),
        ),
        ("invalid", _invalid_ids(global_e, local_e, shard_begin, ids_dtype)),
        ("all", _route_mask_ids(0xFF, global_e, local_e, shard_begin, ids_dtype)),
        (
            "p0_again",
            _route_mask_ids(1 << 0, global_e, local_e, shard_begin, ids_dtype),
        ),
    )
    for name, route_ids in transition_inputs:
        ids.copy_(route_ids)
        direct_graph.replay()
        scan_graph.replay()
        torch.cuda.synchronize()
        expected_count = _safe_local_count(ids, expert_map, global_e, local_e)
        transitions.append(
            {
                "routing": name,
                "expected_local_count": expected_count,
                "metadata_local_count": int(local_count.item()),
                "bitwise": bool(torch.equal(direct_graph_output, scan_graph_output)),
                "max_abs": float(
                    (direct_graph_output.float() - scan_graph_output.float())
                    .abs()
                    .max()
                    .item()
                ),
            }
        )

    masks = (
        range(1 << topk)
        if args.audit_route_masks
        else (
            0,
            *(1 << position for position in range(topk)),
            0b00000011,
            0b10000001,
            0b00001111,
            0b10101010,
            0xFF,
        )
    )
    mask_results = []
    for mask in masks:
        ids.copy_(_route_mask_ids(mask, global_e, local_e, shard_begin, ids_dtype))
        direct_graph.replay()
        scan_graph.replay()
        torch.cuda.synchronize()
        timing = _measure_pair(graphs, args.rounds, args.replays)
        local_count_value = mask.bit_count()
        count_probability = (
            math.comb(local_e, local_count_value)
            * math.comb(global_e - local_e, topk - local_count_value)
            / math.comb(global_e, topk)
        )
        probability = count_probability / math.comb(topk, local_count_value)
        mask_results.append(
            {
                "mask": mask,
                "positions": [p for p in range(topk) if mask & (1 << p)],
                "local_count": local_count_value,
                "probability": probability,
                "bitwise": bool(torch.equal(direct_graph_output, scan_graph_output)),
                "metadata_local_count": int(local_count.item()),
                **timing,
            }
        )

    by_count = {}
    for count in range(topk + 1):
        rows = [row for row in mask_results if row["local_count"] == count]
        if not rows:
            continue
        mean_direct = statistics.mean(row["median_ms"]["direct"] for row in rows)
        mean_scan = statistics.mean(row["median_ms"]["fused_scan"] for row in rows)
        by_count[str(count)] = {
            "masks": len(rows),
            "probability_mass": sum(row["probability"] for row in rows),
            "mean_direct_ms": mean_direct,
            "mean_fused_scan_ms": mean_scan,
            "ratio_of_means_reduction_pct": 100.0 * (1.0 - mean_scan / mean_direct),
            "worst_reduction_pct": min(row["reduction_pct"] for row in rows),
        }

    expected = None
    if args.audit_route_masks:
        expected_direct = sum(
            row["probability"] * row["median_ms"]["direct"] for row in mask_results
        )
        expected_scan = sum(
            row["probability"] * row["median_ms"]["fused_scan"] for row in mask_results
        )
        expected = {
            "probability_sum": sum(row["probability"] for row in mask_results),
            "direct_ms": expected_direct,
            "fused_scan_ms": expected_scan,
            "reduction_pct": 100.0 * (1.0 - expected_scan / expected_direct),
            "positive_masks": sum(row["reduction_pct"] > 0 for row in mask_results),
            "total_masks": len(mask_results),
            "bitwise_masks": sum(row["bitwise"] for row in mask_results),
            "metadata_count_correct_masks": sum(
                row["metadata_local_count"] == row["local_count"]
                for row in mask_results
            ),
            "worst": min(mask_results, key=lambda row: row["reduction_pct"])[
                "reduction_pct"
            ],
        }

    result = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "M": 1,
            "global_E": global_e,
            "local_E": local_e,
            "H": h,
            "I": intermediate,
            "topk": topk,
            "dtype": "bf16",
            "ids_dtype": args.ids_dtype,
        },
        "candidate": (
            "GEMM1 in-register local-rank prefix scan + metadata-fed "
            "local-rank-sharded GEMM2"
        ),
        "production_default_modified": False,
        "output_cache2_alias": bool(
            direct_output.data_ptr() == direct_cache2.data_ptr()
            and scan_output.data_ptr() == scan_cache2.data_ptr()
        ),
        "route_shards": args.route_shards,
        "kernel_configs": {"g1": g1.__dict__, "g2": g2.__dict__},
        "initial_capture_bitwise": initial_capture_bitwise,
        "same_graph_route_transitions": transitions,
        "by_local_count": by_count,
        "expected_ep16": expected,
    }
    if not args.summary_only:
        result["mask_results"] = mask_results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
