# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from typing import Any, Optional

import torch
import triton
import triton.language as tl


@triton.jit
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

        for k_block in tl.static_range(
            0, (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        ):
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

        tl.store(
            c_ptr + token * stride_cm + offs_n * stride_cn,
            accumulator,
            mask=offs_n < n_out,
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
    tl.store(output_ptr + token * stride_om + offs_k * stride_ok, accumulator, mask=mask)


def invoke_kunlunxin_moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
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
) -> None:
    n_out = B.size(1) // 2 if FUSE_SILU else B.size(1)
    num_routes = (
        C.size(0) * C.size(1) if direct_routing else sorted_token_ids.numel()
    )
    block_size_n = 4
    # Wide N tiles stall XPU lowering for the 7168-wide DeepSeek projection.
    max_block_size_n = 8 if B.size(2) >= 7168 else 64
    program_count = num_routes * triton.cdiv(n_out, block_size_n)
    while block_size_n < max_block_size_n and (
        program_count > 65536 or (FUSE_SILU and program_count == 65536)
    ):
        block_size_n *= 2
        program_count = num_routes * triton.cdiv(n_out, block_size_n)
    if direct_routing and FUSE_SILU:
        max_block_size_k = 256
    else:
        max_block_size_k = 1024
    block_size_k = min(max_block_size_k, triton.next_power_of_2(B.size(2)))
    while B.size(2) % block_size_k != 0:
        block_size_k //= 2
    n_blocks = triton.cdiv(n_out, block_size_n)
    align_block_size_m = config["BLOCK_SIZE_M"]
    grid = (num_routes, n_blocks)
    _fused_moe_routed_gemm_kernel[grid](
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
        topk_weights.numel()
        if topk_weights is not None
        else C.size(0) * C.size(1),
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
        num_warps=1,
        num_stages=1,
        isCloseVectorization=True,
        isCloseUnrollControl=True,
        buffer_size_limit=2048,
    )


def dispatch_kunlunxin_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
    B_bias: Optional[torch.Tensor] = None,
    FUSE_SILU: bool = False,
    direct_sum: bool = False,
    out_top_k: int = 1,
) -> None:
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
