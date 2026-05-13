# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from the vLLM project (https://github.com/vllm-project/vllm).
# Source: vllm/model_executor/layers/fused_moe/topk_softplus_sqrt_kernels.cu
#
# This Triton implementation is based on the CUDA kernel from vLLM 0.20.0.
# The kernel fuses softplus, sqrt, top-k selection, and optional renormalization
# for MoE gating in models like DeepSeek-V3/V4.

"""TopK Softplus-Sqrt gating kernel in Triton.

Optimized v27: num_warps=1 + all v19-v26 wins.
Key insight: For 256 experts, CUDA uses exactly 1 warp (32 threads) per row,
with each thread holding 8 elements. Using num_warps=4 adds warp scheduling
overhead without helping the 256-element reduction. Combining num_warps=1
(matching CUDA's single-warp-per-row) with tensor caching (v19), score-arithmetic
weight extraction (v20), and the max+compare index recovery (v26) should
minimize overhead.

Eliminates the store-load-store pattern for renormalization by storing weights
during the loop and re-reading with scale at the end.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_topk_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    raw = tl.sqrt(x)

    # Scores for top-k selection (with optional bias)
    if HAS_BIAS:
        bias = tl.load(
            e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0
        ).to(tl.float32)
        scores = raw + bias
    else:
        scores = raw
    scores = tl.where(emask, scores, -float("inf"))

    out_base = pid * topk
    weight_sum = 0.0

    for k_idx in tl.static_range(topk):
        max_score = tl.max(scores, axis=0)
        is_max = scores == max_score
        match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=0)
        eidx = best_slot.to(tl.int32)

        if HAS_BIAS:
            bias_at_eidx = tl.load(e_score_correction_bias_ptr + eidx)
            w = max_score - bias_at_eidx
        else:
            w = max_score

        weight_sum += w
        tl.store(topk_weights_ptr + out_base + k_idx, w)
        tl.store(topk_indices_ptr + out_base + k_idx, eidx)
        tl.store(
            token_expert_indices_ptr + out_base + k_idx,
            (pid * topk + k_idx).to(tl.int32),
        )

        # Zero out winner
        scores = tl.where(expert_offsets == eidx, -float("inf"), scores)

    # Renormalize: re-read weights and apply scale
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor

    for k_idx in tl.static_range(topk):
        w = tl.load(topk_weights_ptr + out_base + k_idx)
        tl.store(topk_weights_ptr + out_base + k_idx, w * scale)


@triton.jit
def _hash_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    input_tokens_ptr,
    hash_indices_table_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Hash mode: expert indices come from lookup table."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    x = tl.sqrt(x)

    # Get expert indices from lookup table
    token_id = tl.load(input_tokens_ptr + pid)
    k_offsets = tl.arange(0, BLOCK_K)
    kmask = k_offsets < topk
    expert_ids = tl.load(
        hash_indices_table_ptr + token_id * topk + k_offsets, mask=kmask, other=0
    )

    # Gather weights for each selected expert
    weight_sum = 0.0
    weights = tl.zeros([BLOCK_K], dtype=tl.float32)

    for k_idx in tl.static_range(topk):
        eidx = tl.sum(tl.where(k_offsets == k_idx, expert_ids, 0))
        w = tl.sum(tl.where(expert_offsets == eidx, x, 0.0))
        weight_sum += w
        weights = tl.where(k_offsets == k_idx, w, weights)

    # Apply renormalization + scaling
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor
    weights = weights * scale

    # Single burst store
    out_base = pid * topk
    tl.store(topk_weights_ptr + out_base + k_offsets, weights, mask=kmask)
    tl.store(topk_indices_ptr + out_base + k_offsets, expert_ids, mask=kmask)
    tei = (pid * topk + k_offsets).to(tl.int32)
    tl.store(token_expert_indices_ptr + out_base + k_offsets, tei, mask=kmask)


def _topk_softplus_sqrt_kernel(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    e_score_correction_bias=None,
    input_tokens=None,
    hash_indices_table=None,
):
    """Low-level kernel launcher matching vLLM CUDA operator interface.

    Writes results in-place into pre-allocated output tensors.

    Args:
        topk_weights: Output tensor [num_tokens, topk], dtype float32
        topk_indices: Output tensor [num_tokens, topk], dtype int32
        token_expert_indices: Output tensor [num_tokens, topk], dtype int32
        gating_output: Gating logits [num_tokens, num_experts]
        renormalize: Whether to renormalize weights
        routed_scaling_factor: Scaling factor for final weights
        e_score_correction_bias: Optional bias for expert scores [num_experts]
        input_tokens: Token IDs for hash mode [num_tokens]
        hash_indices_table: Hash table mapping tokens to expert indices
    """
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    if num_tokens == 0:
        return

    BLOCK_E = triton.next_power_of_2(num_experts)

    if input_tokens is not None and hash_indices_table is not None:
        BLOCK_K = triton.next_power_of_2(topk)
        grid = (num_tokens,)
        _hash_kernel[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            e_score_correction_bias
            if e_score_correction_bias is not None
            else gating_output,
            input_tokens,
            hash_indices_table,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=e_score_correction_bias is not None,
            BLOCK_E=BLOCK_E,
            BLOCK_K=BLOCK_K,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (num_tokens,)
    _fused_topk_kernel[grid](
        gating_output,
        topk_weights,
        topk_indices,
        token_expert_indices,
        e_score_correction_bias
        if e_score_correction_bias is not None
        else gating_output,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        HAS_BIAS=e_score_correction_bias is not None,
        BLOCK_E=BLOCK_E,
        num_warps=1,
        num_stages=1,
    )


def topk_softplus_sqrt(
    hidden_states,
    gating_output,
    topk,
    renormalize,
    routed_scaling_factor,
    e_score_correction_bias=None,
    input_tokens=None,
    hash_indices_table=None,
):
    """Fused topk + softplus + sqrt kernel for MoE gating.

    API aligned with vLLM's fused_topk_bias (scoring_func="sqrtsoftplus").

    Args:
        hidden_states: Input hidden states (not used in computation, kept for
            API compatibility with vLLM's fused_topk_bias interface)
        gating_output: Gating logits, shape (num_tokens, num_experts)
        topk: Number of experts to select per token
        renormalize: Whether to renormalize weights to sum to routed_scaling_factor
        routed_scaling_factor: Scaling factor applied to final weights
        e_score_correction_bias: Optional bias for expert selection
        input_tokens: Token IDs for hash mode
        hash_indices_table: Token-to-expert lookup table for hash mode

    Returns:
        topk_weights: Selected expert weights, shape (num_tokens, topk)
        topk_indices: Selected expert indices, shape (num_tokens, topk)
        token_expert_indices: Flattened indices, shape (num_tokens, topk)
    """
    num_tokens = gating_output.shape[0]
    device = gating_output.device

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=device
    )

    _topk_softplus_sqrt_kernel(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        input_tokens=input_tokens,
        hash_indices_table=hash_indices_table,
    )

    return topk_weights, topk_indices, token_expert_indices
