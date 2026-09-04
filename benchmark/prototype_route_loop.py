#!/usr/bin/env python3
"""Prototype tile-owned route-loop kernels for fused-MoE EP decode.

This benchmark deliberately lives outside the production dispatch.  It compares
the current compact-alignment ``fused_experts_impl`` path with two strict-shape
Triton kernels:

* GEMM1 launches one program per token-group/intermediate tile and loops over
  one or more of M<=8 tokens and each token's eight routes.  Remote routes never
  enter the matrix multiply.
* GEMM2 launches one program per token-group/hidden tile and loops over the same
  tokens/routes,
  applies the router weight after the BF16 GEMM boundary, and combines local
  routes in top-k order.  This removes route/token-sized empty launches and the
  standalone EP combine from the candidate path.

The candidate is intentionally narrow: BF16, H=4096, top-k=8, global/local
experts=288/18, intermediate size 1280 or 2048, clamp=10, and M<=8.  It is an
experiment, not a fallback implementation.

Direct G2 output is valid only when output does not overlap cache2.  With
``--alias-output-cache2`` the benchmark instead writes weighted route rows to
an independent cache3 and calls ``moe_sum_ep`` after GEMM2, matching the safe
ordering required by the plugin's common workspace alias.

The token-group loop serializes multiple local routes and therefore requires a
sparse-route gate; M<=8 alone is not sufficient.  ``--route-shards`` explores
an alias-safe alternative that interleaves flattened routes over disjoint CTA
shards.  It restores parallelism when local routes land in different shards,
but deliberately remains a benchmark option because shard collisions are
route-position sensitive.
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
def _fused_moe_route_loop_g1(
    hidden_ptr,
    w1_ptr,
    cache2_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_tokens,
    topk,
    tokens_per_cta,
    intermediate_size,
    hidden_size,
    num_global_experts,
    num_local_experts,
    stride_hidden_m,
    stride_hidden_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_cache_route,
    stride_cache_n,
    stride_ids_m,
    stride_ids_route,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    EARLY_NO_LOCAL: tl.constexpr,
):
    """Fused paired gate/up GEMM and clamped SwiGLU for a token group."""
    token_group = tl.program_id(0)
    tile_n = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
    tile_start = tile_n * BLOCK_SIZE_N
    offs_pair_n = tl.where(
        offs_pair < BLOCK_SIZE_N,
        tile_start + offs_pair,
        intermediate_size + tile_start + offs_pair - BLOCK_SIZE_N,
    )
    output_columns = tile_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    valid_row = offs_m == 0
    if EARLY_NO_LOCAL:
        scan_offsets = tl.arange(0, 8)
        scan_ids_base = topk_ids_ptr + token_group * tokens_per_cta * stride_ids_m
        scan_global_raw = tl.load(scan_ids_base + scan_offsets * stride_ids_route)
        scan_valid_global = (scan_global_raw >= 0) & (
            scan_global_raw < num_global_experts
        )
        scan_safe_global = tl.where(scan_valid_global, scan_global_raw, 0).to(tl.int64)
        scan_local_raw = tl.load(
            expert_map_ptr + scan_safe_global,
            mask=scan_valid_global,
            other=-1,
        )
        scan_local = (
            scan_valid_global
            & (scan_local_raw >= 0)
            & (scan_local_raw < num_local_experts)
        )
        if tl.sum(scan_local.to(tl.int32), axis=0) == 0:
            return
    for token_offset in range(tokens_per_cta):
        token_idx = token_group * tokens_per_cta + token_offset
        ids_base = topk_ids_ptr + token_idx * stride_ids_m
        for route_idx in range(topk):
            global_expert_raw = tl.load(ids_base + route_idx * stride_ids_route)
            valid_global_expert = (global_expert_raw >= 0) & (
                global_expert_raw < num_global_experts
            )
            safe_global_expert = tl.where(valid_global_expert, global_expert_raw, 0).to(
                tl.int64
            )
            local_expert_raw = tl.load(
                expert_map_ptr + safe_global_expert,
                mask=valid_global_expert,
                other=-1,
            )
            local_route = (
                valid_global_expert
                & (local_expert_raw >= 0)
                & (local_expert_raw < num_local_experts)
            )

            # This is uniform for the program, so remote routes bypass GEMM.
            if local_route:
                local_expert = local_expert_raw.to(tl.int64)
                a_ptrs = (
                    hidden_ptr
                    + token_idx * stride_hidden_m
                    + offs_m[:, None] * 0
                    + offs_k[None, :] * stride_hidden_k
                )
                b_ptrs = (
                    w1_ptr
                    + local_expert * stride_w1_e
                    + offs_k[:, None] * stride_w1_k
                    + offs_pair_n[None, :] * stride_w1_n
                )
                pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
                for k_start in range(0, hidden_size, BLOCK_SIZE_K):
                    a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
                    b = tl.load(b_ptrs)
                    pair_acc += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K * stride_hidden_k
                    b_ptrs += BLOCK_SIZE_K * stride_w1_k

                # Preserve GEMM1's explicit BF16 boundary before activation.
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

                flat_route = token_idx * topk + route_idx
                cache_ptrs = (
                    cache2_ptr
                    + (flat_route + offs_m[:, None]) * stride_cache_route
                    + output_columns[None, :] * stride_cache_n
                )
                tl.store(
                    cache_ptrs,
                    activated.to(tl.bfloat16),
                    mask=valid_row[:, None]
                    & (output_columns[None, :] < intermediate_size),
                )


@triton.jit
def _fused_moe_route_loop_g2(
    cache2_ptr,
    w2_ptr,
    output_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_tokens,
    topk,
    tokens_per_cta,
    hidden_size,
    intermediate_size,
    num_global_experts,
    num_local_experts,
    stride_cache_route,
    stride_cache_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_output_m,
    stride_output_n,
    stride_weights_m,
    stride_weights_route,
    stride_ids_m,
    stride_ids_route,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DIRECT_SUM: tl.constexpr,
    EARLY_NO_LOCAL: tl.constexpr,
):
    """GEMM2 with optional deterministic in-kernel EP combine."""
    token_group = tl.program_id(0)
    tile_n = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    valid_row = offs_m == 0
    if EARLY_NO_LOCAL:
        scan_offsets = tl.arange(0, 8)
        scan_ids_base = topk_ids_ptr + token_group * tokens_per_cta * stride_ids_m
        scan_global_raw = tl.load(scan_ids_base + scan_offsets * stride_ids_route)
        scan_valid_global = (scan_global_raw >= 0) & (
            scan_global_raw < num_global_experts
        )
        scan_safe_global = tl.where(scan_valid_global, scan_global_raw, 0).to(tl.int64)
        scan_local_raw = tl.load(
            expert_map_ptr + scan_safe_global,
            mask=scan_valid_global,
            other=-1,
        )
        scan_local = (
            scan_valid_global
            & (scan_local_raw >= 0)
            & (scan_local_raw < num_local_experts)
        )
        if tl.sum(scan_local.to(tl.int32), axis=0) == 0:
            return
    for token_offset in range(tokens_per_cta):
        token_idx = token_group * tokens_per_cta + token_offset
        ids_base = topk_ids_ptr + token_idx * stride_ids_m
        weights_base = topk_weights_ptr + token_idx * stride_weights_m
        combined = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Match moe_sum_ep's route order and BF16 route-output boundary.
        for route_idx in range(topk):
            global_expert_raw = tl.load(ids_base + route_idx * stride_ids_route)
            valid_global_expert = (global_expert_raw >= 0) & (
                global_expert_raw < num_global_experts
            )
            safe_global_expert = tl.where(valid_global_expert, global_expert_raw, 0).to(
                tl.int64
            )
            local_expert_raw = tl.load(
                expert_map_ptr + safe_global_expert,
                mask=valid_global_expert,
                other=-1,
            )
            local_route = (
                valid_global_expert
                & (local_expert_raw >= 0)
                & (local_expert_raw < num_local_experts)
            )
            if local_route:
                local_expert = local_expert_raw.to(tl.int64)
                flat_route = token_idx * topk + route_idx
                a_ptrs = (
                    cache2_ptr
                    + (flat_route + offs_m[:, None]) * stride_cache_route
                    + offs_k[None, :] * stride_cache_k
                )
                b_ptrs = (
                    w2_ptr
                    + local_expert * stride_w2_e
                    + offs_n[None, :] * stride_w2_n
                    + offs_k[:, None] * stride_w2_k
                )
                route_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for k_start in range(0, intermediate_size, BLOCK_SIZE_K):
                    a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
                    b = tl.load(b_ptrs)
                    route_acc += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K * stride_cache_k
                    b_ptrs += BLOCK_SIZE_K * stride_w2_k

                route_weight = tl.load(weights_base + route_idx * stride_weights_route)
                weighted_bf16 = (route_acc * route_weight.to(tl.float32)).to(
                    tl.bfloat16
                )
                if DIRECT_SUM:
                    combined += weighted_bf16.to(tl.float32)
                else:
                    flat_route = token_idx * topk + route_idx
                    route_output_ptrs = (
                        output_ptr
                        + (flat_route + offs_m[:, None]) * stride_output_m
                        + offs_n[None, :] * stride_output_n
                    )
                    tl.store(
                        route_output_ptrs,
                        weighted_bf16,
                        mask=valid_row[:, None] & (offs_n[None, :] < hidden_size),
                    )

        if DIRECT_SUM:
            output_ptrs = (
                output_ptr
                + (token_idx + offs_m[:, None]) * stride_output_m
                + offs_n[None, :] * stride_output_n
            )
            tl.store(
                output_ptrs,
                combined.to(tl.bfloat16),
                mask=valid_row[:, None] & (offs_n[None, :] < hidden_size),
            )


@triton.jit
def _fused_moe_route_shard_g1(
    hidden_ptr,
    w1_ptr,
    cache2_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_tokens,
    topk,
    route_shards,
    intermediate_size,
    hidden_size,
    num_global_experts,
    num_local_experts,
    stride_hidden_m,
    stride_hidden_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_cache_route,
    stride_cache_n,
    stride_ids_m,
    stride_ids_route,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
):
    """GEMM1 for interleaved flattened-route shards."""
    shard_idx = tl.program_id(0)
    tile_n = tl.program_id(1)
    num_routes = num_tokens * topk
    routes_in_shard = tl.cdiv(num_routes, route_shards)
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
    tile_start = tile_n * BLOCK_SIZE_N
    offs_pair_n = tl.where(
        offs_pair < BLOCK_SIZE_N,
        tile_start + offs_pair,
        intermediate_size + tile_start + offs_pair - BLOCK_SIZE_N,
    )
    output_columns = tile_start + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    valid_row = offs_m == 0

    for route_slot in range(routes_in_shard):
        flat_route = shard_idx + route_slot * route_shards
        valid_route = flat_route < num_routes
        safe_route = tl.where(valid_route, flat_route, 0)
        token_idx = safe_route // topk
        route_idx = safe_route - token_idx * topk
        global_expert_raw = tl.load(
            topk_ids_ptr + token_idx * stride_ids_m + route_idx * stride_ids_route,
            mask=valid_route,
            other=-1,
        )
        valid_global_expert = (
            valid_route
            & (global_expert_raw >= 0)
            & (global_expert_raw < num_global_experts)
        )
        safe_global_expert = tl.where(valid_global_expert, global_expert_raw, 0).to(
            tl.int64
        )
        local_expert_raw = tl.load(
            expert_map_ptr + safe_global_expert,
            mask=valid_global_expert,
            other=-1,
        )
        local_route = (
            valid_global_expert
            & (local_expert_raw >= 0)
            & (local_expert_raw < num_local_experts)
        )
        if local_route:
            local_expert = local_expert_raw.to(tl.int64)
            a_ptrs = (
                hidden_ptr
                + token_idx * stride_hidden_m
                + offs_m[:, None] * 0
                + offs_k[None, :] * stride_hidden_k
            )
            b_ptrs = (
                w1_ptr
                + local_expert * stride_w1_e
                + offs_k[:, None] * stride_w1_k
                + offs_pair_n[None, :] * stride_w1_n
            )
            pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
            for k_start in range(0, hidden_size, BLOCK_SIZE_K):
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
                + (flat_route + offs_m[:, None]) * stride_cache_route
                + output_columns[None, :] * stride_cache_n
            )
            tl.store(
                cache_ptrs,
                activated.to(tl.bfloat16),
                mask=valid_row[:, None] & (output_columns[None, :] < intermediate_size),
            )


@triton.jit
def _fused_moe_route_shard_g2(
    cache2_ptr,
    w2_ptr,
    cache3_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_tokens,
    topk,
    route_shards,
    hidden_size,
    intermediate_size,
    num_global_experts,
    num_local_experts,
    stride_cache2_route,
    stride_cache2_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_cache3_route,
    stride_cache3_n,
    stride_weights_m,
    stride_weights_route,
    stride_ids_m,
    stride_ids_route,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """GEMM2 route shards writing disjoint weighted BF16 cache3 rows."""
    shard_idx = tl.program_id(0)
    tile_n = tl.program_id(1)
    num_routes = num_tokens * topk
    routes_in_shard = tl.cdiv(num_routes, route_shards)
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    valid_row = offs_m == 0

    for route_slot in range(routes_in_shard):
        flat_route = shard_idx + route_slot * route_shards
        valid_route = flat_route < num_routes
        safe_route = tl.where(valid_route, flat_route, 0)
        token_idx = safe_route // topk
        route_idx = safe_route - token_idx * topk
        global_expert_raw = tl.load(
            topk_ids_ptr + token_idx * stride_ids_m + route_idx * stride_ids_route,
            mask=valid_route,
            other=-1,
        )
        valid_global_expert = (
            valid_route
            & (global_expert_raw >= 0)
            & (global_expert_raw < num_global_experts)
        )
        safe_global_expert = tl.where(valid_global_expert, global_expert_raw, 0).to(
            tl.int64
        )
        local_expert_raw = tl.load(
            expert_map_ptr + safe_global_expert,
            mask=valid_global_expert,
            other=-1,
        )
        local_route = (
            valid_global_expert
            & (local_expert_raw >= 0)
            & (local_expert_raw < num_local_experts)
        )
        if local_route:
            local_expert = local_expert_raw.to(tl.int64)
            a_ptrs = (
                cache2_ptr
                + (flat_route + offs_m[:, None]) * stride_cache2_route
                + offs_k[None, :] * stride_cache2_k
            )
            b_ptrs = (
                w2_ptr
                + local_expert * stride_w2_e
                + offs_n[None, :] * stride_w2_n
                + offs_k[:, None] * stride_w2_k
            )
            route_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k_start in range(0, intermediate_size, BLOCK_SIZE_K):
                a = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
                b = tl.load(b_ptrs)
                route_acc += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_cache2_k
                b_ptrs += BLOCK_SIZE_K * stride_w2_k

            route_weight = tl.load(
                topk_weights_ptr
                + token_idx * stride_weights_m
                + route_idx * stride_weights_route
            )
            weighted_bf16 = (route_acc * route_weight.to(tl.float32)).to(tl.bfloat16)
            cache3_ptrs = (
                cache3_ptr
                + (flat_route + offs_m[:, None]) * stride_cache3_route
                + offs_n[None, :] * stride_cache3_n
            )
            tl.store(
                cache3_ptrs,
                weighted_bf16,
                mask=valid_row[:, None] & (offs_n[None, :] < hidden_size),
            )


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


G1_CONFIG = KernelConfig(16, 16, 128, 2, 3)
G2_CONFIG = KernelConfig(16, 64, 64, 4, 4)


def launch_route_loop(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    expert_map: torch.Tensor,
    cache2: torch.Tensor,
    cache3: torch.Tensor | None,
    output: torch.Tensor,
    *,
    direct_sum: bool,
    tokens_per_cta: int,
    route_shards: int,
    early_no_local: bool,
) -> torch.Tensor:
    """Launch the strict-shape prototype without any host-visible routing work."""
    m, hidden_size = hidden.shape
    topk = ids.shape[1]
    intermediate_size = w2.shape[2]
    local_experts = w1.shape[0]
    global_experts = expert_map.numel()
    if not (
        hidden.dtype
        == w1.dtype
        == w2.dtype
        == cache2.dtype
        == output.dtype
        == torch.bfloat16
        and weights.dtype == torch.bfloat16
        and ids.dtype in (torch.int32, torch.int64)
        and expert_map.dtype in (torch.int32, torch.int64)
        and 0 < m <= 8
        and hidden_size == 4096
        and intermediate_size in (1280, 2048)
        and topk == 8
        and global_experts == 288
        and local_experts == 18
        and w1.shape == (18, 2 * intermediate_size, hidden_size)
        and w2.shape == (18, hidden_size, intermediate_size)
        and cache2.shape == (m * topk, intermediate_size)
        and (
            cache3 is None
            or (
                cache3.dtype == torch.bfloat16
                and cache3.shape == (m, topk, hidden_size)
            )
        )
        and output.shape == hidden.shape
    ):
        raise ValueError("route-loop prototype only accepts the strict fused-MoE shape")
    if direct_sum and cache3 is not None:
        raise ValueError("cache3 must be None when direct_sum=True")
    if not direct_sum and cache3 is None:
        raise ValueError("alias-safe route-loop requires cache3")
    if tokens_per_cta not in (1, 2, 4, 8) or m % tokens_per_cta != 0:
        raise ValueError("tokens_per_cta must be 1/2/4/8 and divide M")
    if route_shards not in (0, 1, 2, 4, 8):
        raise ValueError("route_shards must be 0/1/2/4/8")
    if route_shards and (direct_sum or cache3 is None):
        raise ValueError("route shards are supported only by alias-safe cache3 mode")
    if early_no_local and (m != 1 or tokens_per_cta != 1 or route_shards):
        raise ValueError(
            "early-no-local is currently limited to the M=1 token-loop path"
        )
    tensors = (hidden, w1, w2, weights, ids, expert_map, cache2, output)
    if cache3 is not None:
        tensors += (cache3,)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all route-loop inputs and outputs must be contiguous")
    cache2_begin = cache2.data_ptr()
    cache2_end = cache2_begin + cache2.numel() * cache2.element_size()
    output_begin = output.data_ptr()
    output_end = output_begin + output.numel() * output.element_size()
    output_aliases_cache2 = output_begin < cache2_end and cache2_begin < output_end
    if direct_sum and output_aliases_cache2:
        raise ValueError("direct-sum G2 cannot write output that aliases cache2")

    if route_shards:
        _fused_moe_route_shard_g1[
            (route_shards, triton.cdiv(intermediate_size, G1_CONFIG.block_n))
        ](
            hidden,
            w1,
            cache2,
            ids,
            expert_map,
            m,
            topk,
            route_shards,
            intermediate_size,
            hidden_size,
            global_experts,
            local_experts,
            hidden.stride(0),
            hidden.stride(1),
            w1.stride(0),
            w1.stride(1),
            w1.stride(2),
            cache2.stride(0),
            cache2.stride(1),
            ids.stride(0),
            ids.stride(1),
            BLOCK_SIZE_M=G1_CONFIG.block_m,
            BLOCK_SIZE_N=G1_CONFIG.block_n,
            BLOCK_SIZE_K=G1_CONFIG.block_k,
            CLAMP_LIMIT=10.0,
            num_warps=G1_CONFIG.num_warps,
            num_stages=G1_CONFIG.num_stages,
        )
        flat_cache3 = cache3.view(m * topk, hidden_size)
        _fused_moe_route_shard_g2[
            (route_shards, triton.cdiv(hidden_size, G2_CONFIG.block_n))
        ](
            cache2,
            w2,
            flat_cache3,
            weights,
            ids,
            expert_map,
            m,
            topk,
            route_shards,
            hidden_size,
            intermediate_size,
            global_experts,
            local_experts,
            cache2.stride(0),
            cache2.stride(1),
            w2.stride(0),
            w2.stride(1),
            w2.stride(2),
            flat_cache3.stride(0),
            flat_cache3.stride(1),
            weights.stride(0),
            weights.stride(1),
            ids.stride(0),
            ids.stride(1),
            BLOCK_SIZE_M=G2_CONFIG.block_m,
            BLOCK_SIZE_N=G2_CONFIG.block_n,
            BLOCK_SIZE_K=G2_CONFIG.block_k,
            num_warps=G2_CONFIG.num_warps,
            num_stages=G2_CONFIG.num_stages,
        )
        fm = importlib.import_module("flag_gems.fused.fused_moe")
        fm.moe_sum_ep(cache3, output, ids, expert_map, local_experts)
        return output

    _fused_moe_route_loop_g1[
        (m // tokens_per_cta, triton.cdiv(intermediate_size, G1_CONFIG.block_n))
    ](
        hidden,
        w1,
        cache2,
        ids,
        expert_map,
        m,
        topk,
        tokens_per_cta,
        intermediate_size,
        hidden_size,
        global_experts,
        local_experts,
        hidden.stride(0),
        hidden.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        cache2.stride(0),
        cache2.stride(1),
        ids.stride(0),
        ids.stride(1),
        BLOCK_SIZE_M=G1_CONFIG.block_m,
        BLOCK_SIZE_N=G1_CONFIG.block_n,
        BLOCK_SIZE_K=G1_CONFIG.block_k,
        CLAMP_LIMIT=10.0,
        EARLY_NO_LOCAL=early_no_local,
        num_warps=G1_CONFIG.num_warps,
        num_stages=G1_CONFIG.num_stages,
    )
    g2_output = output if direct_sum else cache3.view(m * topk, hidden_size)
    _fused_moe_route_loop_g2[
        (m // tokens_per_cta, triton.cdiv(hidden_size, G2_CONFIG.block_n))
    ](
        cache2,
        w2,
        g2_output,
        weights,
        ids,
        expert_map,
        m,
        topk,
        tokens_per_cta,
        hidden_size,
        intermediate_size,
        global_experts,
        local_experts,
        cache2.stride(0),
        cache2.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        g2_output.stride(0),
        g2_output.stride(1),
        weights.stride(0),
        weights.stride(1),
        ids.stride(0),
        ids.stride(1),
        BLOCK_SIZE_M=G2_CONFIG.block_m,
        BLOCK_SIZE_N=G2_CONFIG.block_n,
        BLOCK_SIZE_K=G2_CONFIG.block_k,
        DIRECT_SUM=direct_sum,
        EARLY_NO_LOCAL=early_no_local,
        num_warps=G2_CONFIG.num_warps,
        num_stages=G2_CONFIG.num_stages,
    )
    if not direct_sum:
        fm = importlib.import_module("flag_gems.fused.fused_moe")
        fm.moe_sum_ep(cache3, output, ids, expert_map, local_experts)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument(
        "--intermediate-size", type=int, choices=(1280, 2048), default=2048
    )
    parser.add_argument(
        "--route-kind",
        choices=("exact", "uniform", "expected", "no-local", "all-local"),
        default="exact",
    )
    parser.add_argument(
        "--local-routes",
        type=int,
        default=1,
        help="Number of local routes used only by --route-kind=exact.",
    )
    parser.add_argument(
        "--flat-local-offset",
        type=int,
        default=0,
        help="First flattened route replaced by a local expert.",
    )
    parser.add_argument(
        "--flat-local-stride",
        type=int,
        default=1,
        help="Stride between exact local flattened routes; useful for shard collisions.",
    )
    parser.add_argument("--ep-rank", type=int, default=7)
    parser.add_argument(
        "--reference",
        choices=("compact", "production"),
        default="compact",
        help="Reference route policy; production uses the current dispatch gate.",
    )
    parser.add_argument(
        "--tokens-per-cta",
        type=int,
        choices=(0, 1, 2, 4, 8),
        default=0,
        help="Token group size; 0 chooses M for M<=2 and 1 otherwise.",
    )
    parser.add_argument("--g1-block-n", type=int, choices=(16, 32), default=16)
    parser.add_argument("--g1-warps", type=int, choices=(2, 4), default=2)
    parser.add_argument("--g1-stages", type=int, choices=(2, 3), default=3)
    parser.add_argument("--g2-block-n", type=int, choices=(64, 128), default=64)
    parser.add_argument("--g2-stages", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument(
        "--route-shards",
        type=int,
        choices=(0, 1, 2, 4, 8),
        default=0,
        help="Alias-safe interleaved flattened-route CTA shards; 0 uses token groups.",
    )
    parser.add_argument(
        "--early-no-local",
        action="store_true",
        help="Prototype an M=1 vector scan that returns before the scalar route loop.",
    )
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument(
        "--audit-route-masks",
        action="store_true",
        help=(
            "For M=1, replay both captured graphs for every local/remote top-8 "
            "mask and report the exact EP16 hypergeometric expectation."
        ),
    )
    parser.add_argument(
        "--alias-output-cache2",
        action="store_true",
        help=(
            "Use the plugin-style output/cache2 alias. The candidate then writes "
            "GEMM2 routes to cache3 and calls moe_sum_ep; direct-sum is disabled."
        ),
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def make_routes(
    m: int,
    topk: int,
    global_experts: int,
    local_experts: int,
    shard_begin: int,
    route_kind: str,
    local_routes: int,
    flat_local_offset: int,
    flat_local_stride: int,
) -> torch.Tensor:
    if route_kind == "uniform":
        logits = torch.randn((m, global_experts), device="cuda")
        return torch.topk(torch.sigmoid(logits), topk, dim=-1).indices.to(torch.int32)

    flat = torch.arange(m * topk, device="cuda", dtype=torch.int32)
    remote_index = flat.remainder(global_experts - local_experts)
    ids = remote_index + (remote_index >= shard_begin).to(torch.int32) * local_experts
    if route_kind == "no-local":
        return ids.view(m, topk)
    if route_kind == "all-local":
        return (shard_begin + flat.remainder(local_experts)).view(m, topk)
    if route_kind == "expected":
        # EP16 expectation is one local route per 16 flattened routes.  Spread
        # them across tokens instead of clustering at the beginning.
        local_routes = (m * topk + 8) // 16
        route_indices = torch.arange(local_routes, device="cuda") * 16
    elif route_kind == "exact":
        if not 0 <= local_routes <= m * topk:
            raise ValueError("local-routes must be in [0, M*8]")
        route_indices = (
            flat_local_offset
            + torch.arange(local_routes, device="cuda") * flat_local_stride
        ).remainder(m * topk)
    else:
        raise ValueError(f"unsupported route kind: {route_kind}")
    ids[route_indices] = shard_begin + torch.arange(
        local_routes, device="cuda", dtype=torch.int32
    ).remainder(local_experts)
    return ids.view(m, topk)


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
        graph_result = fn()
    return graph, eager, graph_result


def elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / replays)


def main() -> None:
    global G1_CONFIG, G2_CONFIG
    args = parse_args()
    if not 0 < args.m <= 8:
        raise ValueError("m must be in [1, 8]")
    if not 0 <= args.ep_rank < 16:
        raise ValueError("ep-rank must be in [0, 16)")
    if args.rounds <= 0 or args.replays <= 0:
        raise ValueError("rounds and replays must be positive")
    if args.flat_local_stride <= 0:
        raise ValueError("flat-local-stride must be positive")
    tokens_per_cta = (
        (args.m if args.m <= 2 else 1)
        if args.tokens_per_cta == 0
        else args.tokens_per_cta
    )
    if args.m % tokens_per_cta != 0:
        raise ValueError("tokens-per-cta must divide M")
    if args.route_shards and not args.alias_output_cache2:
        raise ValueError("route-shards requires --alias-output-cache2")
    if args.audit_route_masks and args.m != 1:
        raise ValueError("--audit-route-masks currently requires --m 1")
    G1_CONFIG = KernelConfig(16, args.g1_block_n, 128, args.g1_warps, args.g1_stages)
    G2_CONFIG = KernelConfig(16, args.g2_block_n, 64, 4, args.g2_stages)

    fm = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)
    m, global_e, local_e, hidden_size, topk = args.m, 288, 18, 4096, 8
    intermediate_size = args.intermediate_size
    shard_begin = args.ep_rank * local_e
    dtype = torch.bfloat16

    hidden = torch.randn((m, hidden_size), device="cuda", dtype=dtype)
    w1 = torch.empty(
        (local_e, 2 * intermediate_size, hidden_size), device="cuda", dtype=dtype
    )
    w1.normal_(std=hidden_size**-0.5)
    w2 = torch.empty(
        (local_e, hidden_size, intermediate_size), device="cuda", dtype=dtype
    )
    w2.normal_(std=intermediate_size**-0.5)
    ids = make_routes(
        m,
        topk,
        global_e,
        local_e,
        shard_begin,
        args.route_kind,
        args.local_routes,
        args.flat_local_offset,
        args.flat_local_stride,
    )
    weights = torch.rand((m, topk), device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(dtype)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device="cuda", dtype=torch.int32
    )

    compact_cache13 = torch.empty(
        m * topk * max(2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=dtype,
    )
    compact_cache2 = torch.empty(
        m * topk * intermediate_size, device="cuda", dtype=dtype
    )
    route_loop_cache2 = torch.empty(
        (m * topk, intermediate_size), device="cuda", dtype=dtype
    )
    if args.alias_output_cache2:
        compact_output = compact_cache2[: m * hidden_size].view(m, hidden_size)
        route_loop_output = route_loop_cache2.view(-1)[: m * hidden_size].view(
            m, hidden_size
        )
        route_loop_cache3 = torch.empty(
            (m, topk, hidden_size), device="cuda", dtype=dtype
        )
    else:
        compact_output = torch.empty_like(hidden)
        route_loop_output = torch.empty_like(hidden)
        route_loop_cache3 = None

    def compact_op():
        return fm.fused_experts_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=compact_output,
            intermediate_cache13=compact_cache13,
            intermediate_cache2=compact_cache2,
        )

    def route_loop_op():
        return launch_route_loop(
            hidden,
            w1,
            w2,
            weights,
            ids,
            expert_map,
            route_loop_cache2,
            route_loop_cache3,
            route_loop_output,
            direct_sum=not args.alias_output_cache2,
            tokens_per_cta=tokens_per_cta,
            route_shards=args.route_shards,
            early_no_local=args.early_no_local,
        )

    original_naive_gate = fm._should_use_ep_naive_route
    original_route_block_gate = fm._should_use_ep_route_block
    original_local_rank_gate = fm._should_use_ep_m1_i2048_local_rank
    if args.reference == "compact":
        # Isolate compact alignment even when production selects an M=1 fast path.
        fm._should_use_ep_naive_route = lambda *_args, **_kwargs: False
        fm._should_use_ep_route_block = lambda *_args, **_kwargs: False
        fm._should_use_ep_m1_i2048_local_rank = lambda *_args, **_kwargs: False
    try:
        compact_graph, compact_eager, compact_graph_output = capture(compact_op)
    finally:
        fm._should_use_ep_naive_route = original_naive_gate
        fm._should_use_ep_route_block = original_route_block_gate
        fm._should_use_ep_m1_i2048_local_rank = original_local_rank_gate
    route_graph, route_eager, route_graph_output = capture(route_loop_op)

    for _ in range(50):
        compact_graph.replay()
        route_graph.replay()
    torch.cuda.synchronize()

    graphs = {"reference": compact_graph, "route_loop": route_graph}
    if args.audit_route_masks:
        remote_ids = make_routes(
            m,
            topk,
            global_e,
            local_e,
            shard_begin,
            "no-local",
            0,
            0,
            1,
        )
        mask_results = []
        policy_names = tuple(graphs)
        for route_mask in range(1 << topk):
            ids.copy_(remote_ids)
            local_positions = [
                position for position in range(topk) if route_mask & (1 << position)
            ]
            # Keep local experts unique, matching top-k over unique global experts.
            for local_expert, position in enumerate(local_positions):
                ids[0, position] = shard_begin + local_expert

            for name in policy_names:
                graphs[name].replay()
            torch.cuda.synchronize()
            bitwise = torch.equal(compact_graph_output, route_graph_output)

            samples = {name: [] for name in policy_names}
            for round_idx in range(args.rounds):
                order = (
                    policy_names
                    if round_idx % 2 == 0
                    else tuple(reversed(policy_names))
                )
                bracket = (*order, *reversed(order))
                round_values = {name: [] for name in policy_names}
                for name in bracket:
                    round_values[name].append(elapsed_ms(graphs[name], args.replays))
                for name in policy_names:
                    samples[name].append(statistics.mean(round_values[name]))

            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            local_count = len(local_positions)
            count_probability = (
                math.comb(local_e, local_count)
                * math.comb(global_e - local_e, topk - local_count)
                / math.comb(global_e, topk)
            )
            mask_probability = count_probability / math.comb(topk, local_count)
            mask_results.append(
                {
                    "route_mask": route_mask,
                    "local_positions": local_positions,
                    "local_count": local_count,
                    "probability": mask_probability,
                    "median_ms": medians,
                    "route_loop_reduction_pct": 100.0
                    * (1.0 - medians["route_loop"] / medians["reference"]),
                    "bitwise": bool(bitwise),
                }
            )

        expected_reference_ms = sum(
            result["probability"] * result["median_ms"]["reference"]
            for result in mask_results
        )
        expected_route_loop_ms = sum(
            result["probability"] * result["median_ms"]["route_loop"]
            for result in mask_results
        )
        by_count = {}
        for local_count in range(topk + 1):
            count_results = [
                result
                for result in mask_results
                if result["local_count"] == local_count
            ]
            mean_reference_ms = statistics.mean(
                result["median_ms"]["reference"] for result in count_results
            )
            mean_route_loop_ms = statistics.mean(
                result["median_ms"]["route_loop"] for result in count_results
            )
            by_count[str(local_count)] = {
                "masks": len(count_results),
                "probability": sum(result["probability"] for result in count_results),
                "mean_reference_ms": mean_reference_ms,
                "mean_route_loop_ms": mean_route_loop_ms,
                "ratio_of_means_route_loop_reduction_pct": 100.0
                * (1.0 - mean_route_loop_ms / mean_reference_ms),
                "worst_mask_reduction_pct": min(
                    result["route_loop_reduction_pct"] for result in count_results
                ),
                "positive_masks": sum(
                    result["route_loop_reduction_pct"] > 0 for result in count_results
                ),
            }
        result = {
            "device": torch.cuda.get_device_name(),
            "M": m,
            "intermediate_size": intermediate_size,
            "ep_rank": args.ep_rank,
            "audit": "all_route_masks",
            "reference_policy": args.reference,
            "alias_output_cache2": args.alias_output_cache2,
            "route_shards": args.route_shards,
            "early_no_local": args.early_no_local,
            "timing_order": "ABBA/BAAB alternating",
            "rounds": args.rounds,
            "replays": args.replays,
            "kernel_configs": {
                "g1": G1_CONFIG.__dict__,
                "g2": G2_CONFIG.__dict__,
            },
            "uniform_unique_expert_prior": {
                "expected_reference_ms": expected_reference_ms,
                "expected_route_loop_ms": expected_route_loop_ms,
                "expected_route_loop_reduction_pct": 100.0
                * (1.0 - expected_route_loop_ms / expected_reference_ms),
            },
            "positive_masks": sum(
                result["route_loop_reduction_pct"] > 0 for result in mask_results
            ),
            "total_masks": len(mask_results),
            "worst_mask": min(
                mask_results, key=lambda result: result["route_loop_reduction_pct"]
            ),
            "by_local_count": by_count,
            "all_bitwise": all(result["bitwise"] for result in mask_results),
            "results": [] if args.summary_only else mask_results,
        }
        print(json.dumps(result, indent=2))
        return

    raw_samples = {name: [] for name in graphs}
    paired_samples = {name: [] for name in graphs}
    for round_idx in range(args.rounds):
        order = (
            ("reference", "route_loop")
            if round_idx % 2 == 0
            else (
                "route_loop",
                "reference",
            )
        )
        bracket = (*order, *reversed(order))
        round_samples = {name: [] for name in graphs}
        for name in bracket:
            sample = elapsed_ms(graphs[name], args.replays)
            raw_samples[name].append(sample)
            round_samples[name].append(sample)
        for name in graphs:
            paired_samples[name].append(statistics.mean(round_samples[name]))

    medians = {
        name: statistics.median(samples) for name, samples in paired_samples.items()
    }
    paired_reductions = [
        100.0 * (1.0 - candidate / compact)
        for compact, candidate in zip(
            paired_samples["reference"], paired_samples["route_loop"]
        )
    ]
    abs_error = (compact_eager.float() - route_eager.float()).abs()
    mapped_ids = expert_map[ids]
    local_mask = mapped_ids.reshape(-1) >= 0
    local_cache2_bitwise = None
    if not args.alias_output_cache2:
        compact_local_cache = compact_cache2.view(m * topk, intermediate_size)[
            local_mask
        ]
        route_local_cache = route_loop_cache2[local_mask]
        local_cache2_bitwise = bool(torch.equal(compact_local_cache, route_local_cache))
    result = {
        "device": torch.cuda.get_device_name(),
        "M": m,
        "intermediate_size": intermediate_size,
        "route_kind": args.route_kind,
        "local_routes": int(local_mask.sum().item()),
        "flat_local_offset": args.flat_local_offset,
        "flat_local_stride": args.flat_local_stride,
        "alias_output_cache2": args.alias_output_cache2,
        "reference_policy": args.reference,
        "tokens_per_cta": tokens_per_cta,
        "route_shards": args.route_shards,
        "early_no_local": args.early_no_local,
        "candidate_combine": (
            "route_shards_cache3_then_moe_sum_ep"
            if args.route_shards
            else (
                "cache3_then_moe_sum_ep"
                if args.alias_output_cache2
                else "g2_direct_output"
            )
        ),
        "local_routes_per_token": [
            int(value) for value in (mapped_ids >= 0).sum(dim=1).cpu().tolist()
        ],
        "timing_order": "ABBA/BAAB alternating",
        "kernel_configs": {
            "g1": G1_CONFIG.__dict__,
            "g2": G2_CONFIG.__dict__,
        },
        "median_ms": medians,
        "route_loop_reduction_pct": 100.0
        * (1.0 - medians["route_loop"] / medians["reference"]),
        "absolute_delta_us": 1000.0 * (medians["reference"] - medians["route_loop"]),
        "paired_reduction_median_pct": statistics.median(paired_reductions),
        "positive_rounds": sum(value > 0 for value in paired_reductions),
        "total_rounds": len(paired_reductions),
        "bitwise": bool(torch.equal(compact_eager, route_eager)),
        "max_abs_error": float(abs_error.max().item()),
        "graph_bitwise": {
            "reference": bool(torch.equal(compact_eager, compact_graph_output)),
            "route_loop": bool(torch.equal(route_eager, route_graph_output)),
        },
        "local_cache2_bitwise_after_timing": local_cache2_bitwise,
    }
    if not args.summary_only:
        result["raw_samples_ms"] = raw_samples
        result["paired_samples_ms"] = paired_samples
        result["paired_reductions_pct"] = paired_reductions
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
