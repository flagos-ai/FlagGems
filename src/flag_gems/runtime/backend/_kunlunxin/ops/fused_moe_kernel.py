# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(
    do_not_specialize=[
        "fg_num_tokens",
        "fg_top_k",
        "fg_num_experts",
        "fg_fuse_silu",
        "fg_mul_routed_weight",
        "fg_direct_routing",
        "fg_has_bias",
    ]
)
def _fused_moe_routed_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias_e,
    stride_bias_n,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ROUTED_WEIGHT_ON_INPUT: tl.constexpr,
    top_k: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    DIRECT_ROUTING: tl.constexpr,
    ALIGN_BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    fg_num_tokens,
    fg_top_k,
    fg_num_experts,
    fg_fuse_silu,
    fg_mul_routed_weight,
    fg_direct_routing,
    fg_has_bias,
):
    route = tl.program_id(0)
    n_start = tl.program_id(1) * BLOCK_SIZE_N
    if DIRECT_ROUTING:
        token = route
        expert = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        valid_route = True
    else:
        token = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert = tl.load(expert_ids_ptr + route // ALIGN_BLOCK_SIZE_M).to(tl.int64)
        valid_route = token < num_valid_tokens
    n_out = N // 2 if FUSE_SILU else N
    a_row = tl.where(valid_route, token // top_k, 0)
    if MUL_ROUTED_WEIGHT or ROUTED_WEIGHT_ON_INPUT:
        routed_weight = tl.load(
            topk_weights_ptr + token, mask=valid_route, other=0.0
        ).to(tl.float32)

    for n_offset in tl.static_range(0, BLOCK_SIZE_N):
        offs_n = n_start + n_offset
        n_mask = valid_route & (expert >= 0) & (offs_n < n_out)
        accumulator = 0.0
        if FUSE_SILU:
            up_accumulator = 0.0

        for k_block in tl.static_range(0, (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
            offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = n_mask & (offs_k < K)
            a = tl.load(
                a_ptr + a_row * stride_am + offs_k * stride_ak,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if ROUTED_WEIGHT_ON_INPUT:
                a *= routed_weight
            b = tl.load(
                b_ptr + expert * stride_be + offs_k * stride_bk + offs_n * stride_bn,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(a * b, axis=0)

            if FUSE_SILU:
                up = tl.load(
                    b_ptr
                    + expert * stride_be
                    + offs_k * stride_bk
                    + (offs_n + n_out) * stride_bn,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                up_accumulator += tl.sum(a * up, axis=0)

        if HAS_BIAS:
            accumulator += tl.load(
                b_bias_ptr + expert * stride_bias_e + offs_n * stride_bias_n,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            if FUSE_SILU:
                up_accumulator += tl.load(
                    b_bias_ptr
                    + expert * stride_bias_e
                    + (offs_n + n_out) * stride_bias_n,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)

        if FUSE_SILU:
            accumulator = accumulator * tl.sigmoid(accumulator) * up_accumulator
        if MUL_ROUTED_WEIGHT:
            accumulator *= routed_weight

        # Keep the fg_* metadata params live.
        meta_ok = (
            (fg_num_tokens >= 0)
            & (fg_top_k >= 0)
            & (fg_num_experts >= 0)
            & (fg_fuse_silu >= 0)
            & (fg_mul_routed_weight >= 0)
            & (fg_direct_routing >= 0)
            & (fg_has_bias >= 0)
        )
        tl.store(
            c_ptr + token * stride_cm + offs_n * stride_cn,
            accumulator,
            mask=valid_route & (offs_n < n_out) & meta_ok,
        )


@triton.jit(
    do_not_specialize=[
        "fg_num_tokens",
        "fg_top_k",
        "fg_num_experts",
        "fg_fuse_silu",
        "fg_mul_routed_weight",
        "fg_direct_routing",
        "fg_has_bias",
    ]
)
def _experts_route_dot_reference(
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias_e,
    stride_bias_n,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ROUTED_WEIGHT_ON_INPUT: tl.constexpr,
    top_k: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    DIRECT_ROUTING: tl.constexpr,
    ALIGN_BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    fg_num_tokens,
    fg_top_k,
    fg_num_experts,
    fg_fuse_silu,
    fg_mul_routed_weight,
    fg_direct_routing,
    fg_has_bias,
):
    route = tl.program_id(0)
    n_start = tl.program_id(1) * BLOCK_SIZE_N
    if DIRECT_ROUTING:
        token = route
        expert = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        valid_route = True
    else:
        token = tl.load(sorted_token_ids_ptr + route).to(tl.int64)
        expert = tl.load(expert_ids_ptr + route // ALIGN_BLOCK_SIZE_M).to(tl.int64)
        valid_route = token < num_valid_tokens
    n_out = N // 2 if FUSE_SILU else N
    a_row = tl.where(valid_route, token // top_k, 0)
    if MUL_ROUTED_WEIGHT or ROUTED_WEIGHT_ON_INPUT:
        routed_weight = tl.load(
            topk_weights_ptr + token, mask=valid_route, other=0.0
        ).to(tl.float32)

    for n_offset in tl.static_range(0, BLOCK_SIZE_N):
        offs_n = n_start + n_offset
        n_mask = valid_route & (expert >= 0) & (offs_n < n_out)
        accumulator = 0.0
        if FUSE_SILU:
            up_accumulator = 0.0

        for k_block in tl.static_range(0, (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
            offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = n_mask & (offs_k < K)
            a = tl.load(
                a_ptr + a_row * stride_am + offs_k * stride_ak,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if ROUTED_WEIGHT_ON_INPUT:
                a *= routed_weight
            b = tl.load(
                b_ptr + expert * stride_be + offs_k * stride_bk + offs_n * stride_bn,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(a * b, axis=0)

            if FUSE_SILU:
                up = tl.load(
                    b_ptr
                    + expert * stride_be
                    + offs_k * stride_bk
                    + (offs_n + n_out) * stride_bn,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                up_accumulator += tl.sum(a * up, axis=0)

        if HAS_BIAS:
            accumulator += tl.load(
                b_bias_ptr + expert * stride_bias_e + offs_n * stride_bias_n,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            if FUSE_SILU:
                up_accumulator += tl.load(
                    b_bias_ptr
                    + expert * stride_bias_e
                    + (offs_n + n_out) * stride_bias_n,
                    mask=n_mask,
                    other=0.0,
                ).to(tl.float32)

        if FUSE_SILU:
            accumulator = accumulator * tl.sigmoid(accumulator) * up_accumulator
        if MUL_ROUTED_WEIGHT:
            accumulator *= routed_weight

        # Keep the fg_* metadata params live.
        meta_ok = (
            (fg_num_tokens >= 0)
            & (fg_top_k >= 0)
            & (fg_num_experts >= 0)
            & (fg_fuse_silu >= 0)
            & (fg_mul_routed_weight >= 0)
            & (fg_direct_routing >= 0)
            & (fg_has_bias >= 0)
        )
        tl.store(
            c_ptr + token * stride_cm + offs_n * stride_cn,
            accumulator,
            mask=valid_route & (offs_n < n_out) & meta_ok,
        )


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    hidden_size,
    stride_im,
    stride_it,
    stride_ik,
    stride_om,
    stride_ok,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    offs_k = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_k < hidden_size
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for route in tl.static_range(0, TOP_K):
        values = tl.load(
            input_ptr + token * stride_im + route * stride_it + offs_k * stride_ik,
            mask=mask,
            other=0.0,
        )
        accumulator += values.to(tl.float32)
    tl.store(
        output_ptr + token * stride_om + offs_k * stride_ok, accumulator, mask=mask
    )


def invoke_kunlunxin_moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    logger.debug("GEMS_KUNLUNXIN MOE_SUM")
    block_size = 128
    grid = (input.size(0), triton.cdiv(input.size(2), block_size))
    _moe_sum_kernel[grid](
        input,
        output,
        input.size(2),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        TOP_K=input.size(1),
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
        isCloseVectorization=True,
        buffer_size_limit=2048,
    )


# The int64->int32 casts below create temporaries that a reader may still hold
# after the call returns; retain the last few calls' arrays to keep them alive.
_M156_KEEPALIVE = []


def invoke_kunlunxin_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_bias: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    FUSE_SILU: bool,
    direct_routing: bool = False,
    routed_weight_on_input: bool = False,
    generic_route: bool = False,
) -> None:
    logger.debug("GEMS_KUNLUNXIN INVOKE_FUSED_MOE_KERNEL")
    # Cast the routing arrays to int32. Lossless for expert ids.
    if sorted_token_ids.dtype != torch.int32:
        sorted_token_ids = sorted_token_ids.to(torch.int32)
    if expert_ids.dtype != torch.int32:
        expert_ids = expert_ids.to(torch.int32)
    if num_tokens_post_padded.dtype != torch.int32:
        num_tokens_post_padded = num_tokens_post_padded.to(torch.int32)
    _M156_KEEPALIVE.append((sorted_token_ids, expert_ids, num_tokens_post_padded))
    if len(_M156_KEEPALIVE) > 8:
        del _M156_KEEPALIVE[0 : len(_M156_KEEPALIVE) - 8]
    n_out = B.size(1) // 2 if FUSE_SILU else B.size(1)
    num_routes = C.size(0) * C.size(1) if direct_routing else sorted_token_ids.numel()
    block_size_n = 4
    # Cap the N tile for the wide DeepSeek projection.
    max_block_size_n = 8 if B.size(2) >= 7168 else 64
    program_count = num_routes * triton.cdiv(n_out, block_size_n)
    while block_size_n < max_block_size_n and (
        program_count > 65536 or (FUSE_SILU and program_count == 65536)
    ):
        block_size_n *= 2
        program_count = num_routes * triton.cdiv(n_out, block_size_n)
    if direct_routing and FUSE_SILU:
        max_block_size_k = 128
    else:
        max_block_size_k = 128
    block_size_k = min(max_block_size_k, triton.next_power_of_2(B.size(2)))
    while B.size(2) % block_size_k != 0:
        block_size_k //= 2
    # Keep the unrolled (n_offset x k_block) body count bounded. Widen the K
    # tile (power-of-two divisors of K only, up to 512), then shrink the N tile
    # until the product fits. Small-K shapes keep the legacy geometry untouched.
    while (
        (B.size(2) // block_size_k) * block_size_n > 32
        and block_size_k < 512
        and B.size(2) % (block_size_k * 2) == 0
    ):
        block_size_k *= 2
    while (
        (B.size(2) // block_size_k) * block_size_n > 32
        and block_size_n > 1
        and triton.cdiv(n_out, block_size_n // 2) <= 65535
    ):
        block_size_n //= 2
    n_blocks = triton.cdiv(n_out, block_size_n)
    align_block_size_m = config["BLOCK_SIZE_M"]
    grid = (num_routes, n_blocks)
    route_kernel = (
        _experts_route_dot_reference if generic_route else _fused_moe_routed_gemm_kernel
    )
    route_kernel[grid](
        A,
        B,
        C,
        B_bias,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        topk_weights.numel() if topk_weights is not None else C.size(0) * C.size(1),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ROUTED_WEIGHT_ON_INPUT=routed_weight_on_input,
        top_k=top_k,
        HAS_BIAS=B_bias is not None,
        FUSE_SILU=FUSE_SILU,
        DIRECT_ROUTING=direct_routing,
        ALIGN_BLOCK_SIZE_M=align_block_size_m,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_N=block_size_n,
        fg_num_tokens=A.size(0),
        fg_top_k=top_k,
        fg_num_experts=B.size(0),
        fg_fuse_silu=1 if FUSE_SILU else 0,
        fg_mul_routed_weight=1 if mul_routed_weight else 0,
        fg_direct_routing=1 if direct_routing else 0,
        fg_has_bias=1 if B_bias is not None else 0,
        num_warps=1,
        num_stages=1,
        isCloseVectorization=True,
        isCloseUnrollControl=True,
        buffer_size_limit=2048,
    )


# ---------------------------------------------------------------------------
# Sorted-token grouped GEMM via tl.dot (tensor-core path).
#
# The naive routed GEMM above reduces each output element with a scalar
# `tl.sum(a * b)` and never reaches the tensor cores, making it ~100x slower
# than a real grouped GEMM. This path sorts tokens by expert with
# moe_align_block_size, runs two tiled tl.dot GEMMs (gate/up then down) over
# the sorted stream, and scatters the result back. It is taken only for plain
# bf16/fp16 shapes whose K and N divide the tile sizes; every other
# configuration keeps the naive kernel above.
# ---------------------------------------------------------------------------
_MOE_DOT_BLOCK_M = 16
_MOE_DOT_BLOCK_N = 64
_MOE_DOT_BLOCK_K = 64


@triton.jit
def _fused_moe_dot_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sids_ptr,
    eids_ptr,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    top_k: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if GATHER_ROWS:
        # GEMM1: the operand rows come from hidden_states[token], where
        # token = sorted_token_ids // top_k. The padding sentinel maps to the
        # extra zero row the caller appends to A.
        rows = tl.load(sids_ptr + offs_m) // top_k
    else:
        # GEMM2: the operand is already in sorted order, indexed by the tile's
        # own row range.
        rows = offs_m
    off_expert = tl.load(eids_ptr + pid_m)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _fused_moe_scatter_kernel(
    src_ptr,
    dst_ptr,
    sids_ptr,
    width,
    num_valid,
    stride_sm,
    stride_dm,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0)
    cb = tl.program_id(1)
    sid = tl.load(sids_ptr + p)
    # Valid routes map to their own row; padding routes collapse onto the extra
    # sink row the caller allocates (their source values are all zero).
    dst_row = tl.where(sid < num_valid, sid, num_valid)
    offs_c = cb * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(src_ptr + p * stride_sm + offs_c)
    tl.store(dst_ptr + dst_row * stride_dm + offs_c, vals)


def invoke_kunlunxin_fused_moe_dot_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: int,
    top_k: int,
    gather_rows: bool,
) -> None:
    logger.debug("GEMS_KUNLUNXIN INVOKE_FUSED_MOE_DOT_GEMM")
    n_out = C.size(1)
    k_size = B.size(2)
    grid = (
        num_tokens_post_padded // _MOE_DOT_BLOCK_M,
        triton.cdiv(n_out, _MOE_DOT_BLOCK_N),
    )
    _fused_moe_dot_gemm_kernel[grid](
        A,
        B,
        C,
        sorted_token_ids,
        expert_ids,
        n_out,
        k_size,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        top_k=top_k,
        GATHER_ROWS=gather_rows,
        BLOCK_M=_MOE_DOT_BLOCK_M,
        BLOCK_N=_MOE_DOT_BLOCK_N,
        BLOCK_K=_MOE_DOT_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )


def invoke_kunlunxin_fused_moe_scatter(
    src: torch.Tensor,
    dst: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_valid: int,
) -> None:
    logger.debug("GEMS_KUNLUNXIN INVOKE_FUSED_MOE_SCATTER")
    width = src.size(1)
    block = 256
    while width % block != 0:
        block //= 2
    grid = (src.size(0), width // block)
    _fused_moe_scatter_kernel[grid](
        src,
        dst,
        sorted_token_ids,
        width,
        num_valid,
        src.stride(0),
        dst.stride(0),
        BLOCK=block,
        num_warps=1,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# On-device padded->slot routing table.
#
# For every position p with 0 <= sids[p] < numel, slot sids[p] maps to expert
# eids[p // align]. Unwritten slots keep the -1 sentinel.
# FG_MOE_DISPATCH_HOSTDECODE=1 restores the host decoder (A/B, rollback).
# ---------------------------------------------------------------------------
_PADDED_TABLE_BLOCK = 1024
_DEVICE_DECODE = os.environ.get("FG_MOE_DISPATCH_HOSTDECODE", "0") != "1"
_BLK_TABLE_CACHE = {}


def _blk_table(p, align, device):
    key = (p, align, str(device))
    t = _BLK_TABLE_CACHE.get(key)
    if t is None:
        t = torch.arange(p, device=device, dtype=torch.int32) // align
        if len(_BLK_TABLE_CACHE) >= 64:
            _BLK_TABLE_CACHE.clear()
        _BLK_TABLE_CACHE[key] = t
    return t


@triton.jit(
    do_not_specialize=["numel", "e_numel", "P"],
)
def _padded_slot_experts_kernel(
    sids_ptr,
    eids_ptr,
    blk_ptr,
    out_ptr,
    numel,
    e_numel,
    P,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < P
    s = tl.load(sids_ptr + offs, mask=m, other=-1)
    blk = tl.load(blk_ptr + offs, mask=m, other=0)
    ok = m & (s >= 0) & (s < numel) & (blk < e_numel)
    e = tl.load(eids_ptr + blk, mask=ok, other=-1)
    tl.store(out_ptr + s, e, mask=ok)


def _slot_experts_device(sorted_token_ids, expert_ids, align, numel, device):
    """Device-side decoder (see note above). None when it cannot apply, so the
    caller falls back to the host decoder."""
    sids = sorted_token_ids.detach().reshape(-1)
    eids = expert_ids.detach().reshape(-1)
    if not (sids.is_cuda and eids.is_cuda):
        return None
    if sids.dtype != torch.int32:
        sids = sids.to(torch.int32)
    if eids.dtype != torch.int32:
        eids = eids.to(torch.int32)
    sids = sids.contiguous()
    eids = eids.contiguous()
    p = sids.numel()
    if p == 0 or numel <= 0 or align <= 0:
        return None
    blk = _blk_table(p, align, device)
    out = torch.full((numel,), -1, device=device, dtype=torch.int32)
    grid = (triton.cdiv(p, _PADDED_TABLE_BLOCK),)
    _padded_slot_experts_kernel[grid](
        sids,
        eids,
        blk,
        out,
        numel,
        eids.numel(),
        p,
        BLOCK=_PADDED_TABLE_BLOCK,
        num_warps=1,
        num_stages=1,
    )
    return out


def _padded_to_slot_experts(sorted_token_ids, expert_ids, config, numel, device):
    """Decode moe_align_block_size padded arrays into the direct-routing
    slot->expert table (int32, [numel]). Returns None when the arrays cannot be
    trusted (caller falls back to the raw padded launch)."""
    align = int(config["BLOCK_SIZE_M"])
    if align <= 0:
        return None
    sids = sorted_token_ids.detach().reshape(-1).to(torch.int32).cpu().numpy()
    eids = expert_ids.detach().reshape(-1).to(torch.int32).cpu().numpy()
    valid = sids < numel
    pos = np.nonzero(valid)[0]
    if pos.size != numel:
        return None  # duplicated / missing slots -> untrusted
    blk = pos // align
    if blk.max() >= eids.size:
        return None  # expert_ids shorter than the used region -> untrusted
    out = np.full((numel,), -1, dtype=np.int32)
    out[sids[valid].astype(np.int64)] = eids[blk]
    if (out < 0).any():
        return None
    return torch.from_numpy(out).to(device=device, dtype=torch.int32)


def dispatch_kunlunxin_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale,
    B_scale,
    B_zp,
    topk_weights,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape=None,
    B_bias=None,
    FUSE_SILU: bool = False,
    direct_sum: bool = False,
    out_top_k: int = 1,
) -> None:
    logger.debug("GEMS_KUNLUNXIN DISPATCH_FUSED_MOE_KERNEL")
    del compute_type
    unsupported = (
        A_scale is not None
        or B_scale is not None
        or B_zp is not None
        or use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or per_channel_quant
        or block_shape is not None
        or direct_sum
        or out_top_k != 1
    )
    if unsupported:
        raise NotImplementedError(
            "Kunlunxin dispatch_fused_moe_kernel supports only unquantized routed GEMM"
        )

    # Padded -> direct-equivalent restructure (see _padded_to_slot_experts).
    numel = A.size(0) * top_k
    slot_experts = None
    if A.dtype in (torch.bfloat16, torch.float16) and B_bias is None and not FUSE_SILU:
        # Restructure only for the validated acceleration profile (bf16, no
        # bias, no fused silu); everything else keeps the legacy padded launch.
        # FG_MOE_DISPATCH_HOSTDECODE=1 restores the host decoder.
        align = int(config["BLOCK_SIZE_M"])
        if _DEVICE_DECODE and align > 0:
            try:
                slot_experts = _slot_experts_device(
                    sorted_token_ids, expert_ids, align, numel, A.device
                )
            except Exception:
                slot_experts = None
        if slot_experts is None:
            try:
                slot_experts = _padded_to_slot_experts(
                    sorted_token_ids, expert_ids, config, numel, A.device
                )
            except Exception:
                slot_experts = None

    if slot_experts is not None:
        invoke_kunlunxin_fused_moe_kernel(
            A,
            B,
            C,
            B_bias,
            None,
            slot_experts,
            slot_experts,
            slot_experts,
            False,
            top_k,
            config,
            FUSE_SILU=FUSE_SILU,
            direct_routing=True,
        )
        if mul_routed_weight and topk_weights is not None:
            C.mul_(topk_weights.to(C.dtype).unsqueeze(-1))
        return

    # Fallback: original padded launch (correct, unaccelerated).
    invoke_kunlunxin_fused_moe_kernel(
        A,
        B,
        C,
        B_bias,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        FUSE_SILU,
    )
