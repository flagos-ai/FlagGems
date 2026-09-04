# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict single-token expert-parallel fused-MoE kernels.

The regular direct route assigns one CTA row to each original top-k position.
On Hopper, a useful route near the end of top-k can consequently start several
waves later than the same route at position zero.  The kernels below instead
assign four CTA lanes to *local-route rank*.  Lane ``r`` handles local ranks
``r`` and ``r + 4``, so moving a route between top-k positions does not move
its useful CTA row and up to four local routes remain fully parallel.

This module intentionally implements only the measured E288-to-18, H4096,
I2048, top-8 shape.  The caller owns all workspaces, GEMM2 writes weighted
BF16 route rows to cache3, and the existing deterministic EP combine runs only
after GEMM2.  Therefore the final output may safely alias cache2, as required
by the modular vLLM caller, without an alignment launch or host-visible route
decision.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.fused.moe_sum import moe_sum_ep

_GLOBAL_EXPERTS = 288
_LOCAL_EXPERTS = 18
_TOPK = 8
_HIDDEN_SIZE = 4096
_INTERMEDIATE_SIZE = 2048
_LOCAL_RANK_LANES = 4


@triton.jit
def _select_ep_local_route(
    topk_ids_ptr,
    expert_map_ptr,
    lane,
    local_slot: tl.constexpr,
    LANES: tl.constexpr,
    NUM_GLOBAL_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
):
    """Return the original route and local expert for one local-route rank."""
    route_offsets = tl.arange(0, 8)
    global_experts_raw = tl.load(topk_ids_ptr + route_offsets)
    valid_global = (global_experts_raw >= 0) & (global_experts_raw < NUM_GLOBAL_EXPERTS)
    # Preserve the input width through the bounds check.  In particular, an
    # invalid int64 ID such as +/-2**40 must not wrap into expert_map.
    safe_global = tl.where(valid_global, global_experts_raw, 0).to(tl.int64)
    local_experts_raw = tl.load(
        expert_map_ptr + safe_global,
        mask=valid_global,
        other=-1,
    )
    is_local = (
        valid_global
        & (local_experts_raw >= 0)
        & (local_experts_raw < NUM_LOCAL_EXPERTS)
    )
    local_prefix = tl.cumsum(is_local.to(tl.int32), axis=0) - 1
    target_rank = lane + local_slot * LANES
    selected = is_local & (local_prefix == target_rank)
    found = target_rank < tl.sum(is_local.to(tl.int32), axis=0)
    route = tl.sum(
        tl.where(selected, route_offsets, 0).to(tl.int32),
        axis=0,
    )
    # A valid local expert is in [0, 18), so narrowing only after validation is
    # safe for either int32 or int64 expert maps.
    local_experts = local_experts_raw.to(tl.int32)
    expert = tl.sum(tl.where(selected, local_experts, 0), axis=0)
    return found, route, expert


@triton.jit
def _ep_m1_i2048_local_rank_g1(
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
    """Paired gate/up GEMM plus clamp10 SwiGLU for local-route ranks."""
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
        found, route, local_expert = _select_ep_local_route(
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
            # Preserve the existing two-kernel contract: GEMM1 first rounds to
            # BF16, then clamp/SwiGLU promotes those rounded values to FP32.
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
def _ep_m1_i2048_local_rank_g2(
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
    """Second expert GEMM, preserving weighted-BF16 route boundaries."""
    tile_n = tl.program_id(0)
    lane = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    valid_row = offs_m == 0

    for local_slot in range(tl.cdiv(8, LANES)):
        found, route, local_expert = _select_ep_local_route(
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


def fused_moe_ep_m1_i2048_local_rank(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    cache2: torch.Tensor,
    cache3: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Launch the measured no-alignment four-lane EP M1/I2048 path."""
    if not (
        hidden.shape == (1, _HIDDEN_SIZE)
        and w1.shape == (_LOCAL_EXPERTS, 2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)
        and w2.shape == (_LOCAL_EXPERTS, _HIDDEN_SIZE, _INTERMEDIATE_SIZE)
        and topk_weights.shape == (1, _TOPK)
        and topk_ids.shape == (1, _TOPK)
        and expert_map.shape == (_GLOBAL_EXPERTS,)
        and cache2.shape == (_TOPK, _INTERMEDIATE_SIZE)
        and cache3.shape == (_TOPK, _HIDDEN_SIZE)
        and output.shape == hidden.shape
        and hidden.dtype
        == w1.dtype
        == w2.dtype
        == cache2.dtype
        == cache3.dtype
        == output.dtype
        == torch.bfloat16
        and topk_weights.dtype in (torch.bfloat16, torch.float32)
        and topk_ids.dtype in (torch.int32, torch.int64)
        and expert_map.dtype in (torch.int32, torch.int64)
        and all(
            tensor.is_contiguous()
            for tensor in (
                hidden,
                w1,
                w2,
                topk_weights,
                topk_ids,
                expert_map,
                cache2,
                cache3,
                output,
            )
        )
    ):
        raise ValueError(
            "local-rank kernel requires strict "
            "M1/E288->18/H4096/I2048/top8 BF16 tensors and "
            "BF16/FP32 router weights"
        )

    g1_block_n = 16
    _ep_m1_i2048_local_rank_g1[
        (triton.cdiv(_INTERMEDIATE_SIZE, g1_block_n), _LOCAL_RANK_LANES)
    ](
        hidden,
        w1,
        cache2,
        topk_ids,
        expert_map,
        hidden.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        cache2.stride(0),
        cache2.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=g1_block_n,
        BLOCK_SIZE_K=128,
        LANES=_LOCAL_RANK_LANES,
        NUM_GLOBAL_EXPERTS=_GLOBAL_EXPERTS,
        NUM_LOCAL_EXPERTS=_LOCAL_EXPERTS,
        HIDDEN_SIZE=_HIDDEN_SIZE,
        INTERMEDIATE_SIZE=_INTERMEDIATE_SIZE,
        CLAMP_LIMIT=10.0,
        num_warps=2,
        num_stages=3,
    )

    g2_block_n = 64
    _ep_m1_i2048_local_rank_g2[
        (triton.cdiv(_HIDDEN_SIZE, g2_block_n), _LOCAL_RANK_LANES)
    ](
        cache2,
        w2,
        cache3,
        topk_weights,
        topk_ids,
        expert_map,
        cache2.stride(0),
        cache2.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        cache3.stride(0),
        cache3.stride(1),
        topk_weights.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=g2_block_n,
        BLOCK_SIZE_K=64,
        LANES=_LOCAL_RANK_LANES,
        NUM_GLOBAL_EXPERTS=_GLOBAL_EXPERTS,
        NUM_LOCAL_EXPERTS=_LOCAL_EXPERTS,
        HIDDEN_SIZE=_HIDDEN_SIZE,
        INTERMEDIATE_SIZE=_INTERMEDIATE_SIZE,
        num_warps=4,
        num_stages=4,
    )
    moe_sum_ep(
        cache3.view(1, _TOPK, _HIDDEN_SIZE),
        output,
        topk_ids,
        expert_map,
        _LOCAL_EXPERTS,
    )
    return output


__all__ = ["fused_moe_ep_m1_i2048_local_rank"]
